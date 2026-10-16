use add_result::decode_positions_commit;
use ahash::{AHashMap, AHashSet};
use futures::future;
use indexmap::IndexMap;
use memmap2::{Mmap, MmapMut, MmapOptions};
use num_derive::FromPrimitive;

use search::{QueryType, Search};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use smallvec::SmallVec;
use std::{
    cmp,
    collections::HashMap,
    fmt::{self},
    fs::{self, File},
    io::{BufRead, BufReader, Read, Seek, Write},
    path::Path,
    sync::{Arc, LazyLock},
    thread::available_parallelism,
    time::Instant,
};
use tokio::sync::{RwLock, Semaphore};
use utils::{read_u32, write_u16};

#[cfg(feature = "zh")]
use jieba_rs::Jieba;
use crate::{
    INDEX_RUNTIME,
    add_result::{self, B, K, SIGMA},
    clustering::{ClusterHeader, ParentMedoid},
    commit::Commit,
    search::{
        self, FacetFilter, ResultObject, ResultSort, ResultType, SearchLexicalShard, SearchMode,
    },
    tokenizer::tokenizer,
    utils::{
        self, read_u16, read_u16_ref, read_u32_ref, read_u64, read_u64_ref, write_f32, write_f64,
        write_i8, write_i16, write_i32, write_i64, write_u32, write_u64,
    },
    vector::{Inference, Precision, Quantization, VectorHeader, read_min_max},
    vector_similarity::{TurboQuant, VectorSimilarity},
};

use ahash::RandomState;

pub(crate) const INDEX_FILENAME: &str = "index.bin";
pub(crate) const DOCSTORE_FILENAME: &str = "docstore.bin";
pub(crate) const DELETE_FILENAME: &str = "delete.bin";
pub(crate) const SCHEMA_FILENAME: &str = "schema.json";
pub(crate) const META_FILENAME: &str = "index.json";
pub(crate) const FACET_FILENAME: &str = "facet.bin";
pub(crate) const FACET_VALUES_FILENAME: &str = "facet.json";

pub(crate) const VERSION: &str = env!("CARGO_PKG_VERSION");

pub(crate) const VECTOR_FILENAME: &str = "vector.bin";

const INDEX_HEADER_SIZE: u64 = 4;
/// Current on-disk format major version; opening requires an exact version match.
pub const INDEX_FORMAT_VERSION_MAJOR: u16 = 7;
/// Current on-disk format minor version; opening requires an exact version match.
pub const INDEX_FORMAT_VERSION_MINOR: u16 = 0;

/// Maximum processed positions per term per document: default=65_536. E.g. 65,536 * 'the' per document, exceeding positions are ignored for search.
pub const MAX_POSITIONS_PER_TERM: usize = 65_536;
pub(crate) const STOP_BIT: u8 = 0b10000000;
pub(crate) const FIELD_STOP_BIT_1: u8 = 0b0010_0000;
pub(crate) const FIELD_STOP_BIT_2: u8 = 0b0100_0000;
/// maximum number of documents per block
pub const ROARING_BLOCK_SIZE: usize = 65_536;

pub(crate) const SPEEDUP_FLAG: bool = true;
pub(crate) const SORT_FLAG: bool = true;

pub(crate) const MAX_QUERY_TERM_NUMBER: usize = 100;
pub(crate) const SEGMENT_KEY_CAPACITY: usize = 1000;

/// A document is a flattened, single level of key-value pairs, where key is an arbitrary string, and value represents any valid JSON value.
pub type Document = IndexMap<String, Value>;

/// Compression type for document store
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub enum DocumentCompression {
    /// No compression (fastest, largest size)
    None,
    /// Snappy document compression (default).
    Snappy,
}

impl fmt::Display for DocumentCompression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DocumentCompression::None => write!(f, "None"),
            DocumentCompression::Snappy => write!(f, "Snappy"),
        }
    }
}

/// Committed index data is accessed through memory-mapped files.
/// Uncommitted writes remain in memory until commit.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub enum AccessType {
    /// Memory-map committed index data; the OS manages resident pages.
    Mmap = 1,
}

/// Similarity type defines the scoring and ranking of the search results:
/// - Bm25f: considers documents composed from several fields, with different field lengths and importance
/// - Bm25fProximity: considers term proximity, e.g. for implicit phrase search with improved relevancy
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq, Default)]
pub enum LexicalSimilarity {
    /// Bm25f considers documents composed from several fields, with different field lengths and importance
    Bm25f = 0,
    /// Bm25fProximity considers term proximity, e.g. for implicit phrase search with improved relevancy
    #[default]
    Bm25fProximity = 1,
}

impl fmt::Display for LexicalSimilarity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LexicalSimilarity::Bm25f => write!(f, "Bm25f"),
            LexicalSimilarity::Bm25fProximity => write!(f, "Bm25fProximity"),
        }
    }
}

/// Defines tokenizer behavior:
/// AsciiAlphabetic
/// - Only ASCII alphabetic chars are recognized as token.
///
/// UnicodeAlphanumeric
/// - All Unicode alphanumeric chars are recognized as token.
/// - Allows '+' '-' '#' in middle or end of a token: c++, c#, block-max.
///
/// UnicodeAlphanumericFolded
/// - All Unicode alphanumeric chars are recognized as token.
/// - Allows '+' '-' '#' in middle or end of a token: c++, c#, block-max.
/// - Diacritics, accents, zalgo text, umlaut, bold, italic, full-width UTF-8 characters are converted into its basic representation.
/// - Apostroph handling prevents that short term parts preceding or following the apostroph get indexed (e.g. "s" in "someone's").
/// - Tokenizing might be slower due to folding and apostroph processing.
///
/// UnicodeAlphanumericZH
/// - Implements Chinese word segmentation to segment continuous Chinese text into tokens for indexing and search.
/// - Supports mixed Latin and Chinese texts
/// - Requires feature #[cfg(feature = "zh")]
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq, Copy, Default)]
pub enum TokenizerType {
    /// Only ASCII alphabetic chars are recognized as token.
    #[default]
    AsciiAlphabetic = 0,
    /// All Unicode alphanumeric chars are recognized as token.
    /// Allow '+' '-' '#' in middle or end of a token: c++, c#, block-max.
    UnicodeAlphanumeric = 1,
    /// All Unicode alphanumeric chars are recognized as token.
    /// Allows '+' '-' '#' in middle or end of a token: c++, c#, block-max.
    /// Diacritics, accents, zalgo text, umlaut, bold, italic, full-width UTF-8 characters are converted into its basic representation.
    /// Apostroph handling prevents that short term parts preceding or following the apostroph get indexed (e.g. "s" in "someone's").
    /// Tokenizing might be slower due to folding and apostroph processing.
    UnicodeAlphanumericFolded = 2,
    /// Tokens are separated by whitespace.
    Whitespace = 3,
    /// Tokens are separated by whitespace. Token are converted to lowercase.
    WhitespaceLowercase = 4,
    /// Implements Chinese word segmentation to segment continuous Chinese text into tokens for indexing and search.
    /// Supports mixed Latin and Chinese texts
    /// Requires feature #[cfg(feature = "zh")]
    #[cfg(feature = "zh")]
    UnicodeAlphanumericZH = 5,
}

impl fmt::Display for TokenizerType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TokenizerType::AsciiAlphabetic => write!(f, "AsciiAlphabetic"),
            TokenizerType::UnicodeAlphanumeric => write!(f, "UnicodeAlphanumeric"),
            TokenizerType::UnicodeAlphanumericFolded => write!(f, "UnicodeAlphanumericFolded"),
            TokenizerType::Whitespace => write!(f, "Whitespace"),
            TokenizerType::WhitespaceLowercase => write!(f, "WhitespaceLowercase"),
            #[cfg(feature = "zh")]
            TokenizerType::UnicodeAlphanumericZH => write!(f, "UnicodeAlphanumericZH"),
        }
    }
}

pub(crate) struct LevelIndex {
    pub docstore_pointer_docs_pointer: usize,
    pub document_length_compressed_array_pointer: usize,
}

/// Posting lists are divided into blocks of a doc id range of 65.536 (16 bit).
/// Each block can be compressed with a different method.
#[derive(Default, Debug, Deserialize, Serialize, Clone)]
pub(crate) struct BlockObjectIndex {
    pub max_block_score: f32,
    pub block_id: u32,
    pub compression_type_pointer: u32,
    pub posting_count: u16,
    pub max_docid: u16,
    pub max_p_docid: u16,
    pub pointer_pivot_p_docid: u16,
}

/// PostingListObjectIndex owns all blocks of a postinglist of a term
#[derive(Default)]
pub(crate) struct PostingListObjectIndex {
    pub posting_count: u32,
    pub posting_count_ngram_1: u32,
    pub posting_count_ngram_2: u32,
    pub posting_count_ngram_3: u32,
    pub max_list_score: f32,
    pub blocks: Vec<BlockObjectIndex>,
}

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
pub(crate) struct PostingListObject0 {
    pub pointer_first: usize,
    pub pointer_last: usize,
    pub posting_count: usize,

    pub max_block_score: f32,
    pub max_docid: u16,
    pub max_p_docid: u16,

    pub ngram_type: NgramType,
    pub term_ngram1: String,
    pub term_ngram2: String,
    pub term_ngram3: String,
    pub posting_count_ngram_1: f32,
    pub posting_count_ngram_2: f32,
    pub posting_count_ngram_3: f32,
    pub posting_count_ngram_1_compressed: u8,
    pub posting_count_ngram_2_compressed: u8,
    pub posting_count_ngram_3_compressed: u8,

    pub position_count: usize,
    pub pointer_pivot_p_docid: u16,
    pub size_compressed_positions_key: usize,
    pub docid_delta_max: u16,
    pub docid_old: u16,
    pub compression_type_pointer: u32,
}

/// Type of posting list compression.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, FromPrimitive)]
pub(crate) enum CompressionType {
    Delta = 0,
    Array = 1,
    Bitmap = 2,
    Rle = 3,
}

pub(crate) struct QueueObject<'a> {
    pub query_list: Vec<PostingListObjectQuery<'a>>,
    pub query_index: usize,
    pub max_score: f32,
}

/// PostingListObjectQuery manages thes posting list for each unique query term during intersection.
#[derive(Clone)]
pub(crate) struct PostingListObjectQuery<'a> {
    pub posting_count: u32,
    pub max_list_score: f32,
    pub blocks: &'a Vec<BlockObjectIndex>,
    pub blocks_index: usize,

    pub term: String,
    pub key0: u32,

    pub compression_type: CompressionType,
    pub rank_position_pointer_range: u32,
    pub compressed_doc_id_range: usize,
    pub pointer_pivot_p_docid: u16,

    pub posting_pointer: usize,
    pub posting_pointer_previous: usize,

    pub byte_array: &'a [u8],

    pub p_block: i32,
    pub p_block_max: i32,
    pub p_docid: usize,
    pub p_docid_count: usize,

    pub rangebits: i32,
    pub docid: i32,
    pub bitposition: u32,

    pub intersect: u64,
    pub ulong_pos: usize,

    pub run_end: i32,
    pub p_run: i32,
    pub p_run_count: i32,
    pub p_run_sum: i32,

    pub term_index_unique: usize,
    pub positions_count: u32,
    pub positions_pointer: u32,

    pub idf: f32,
    pub idf_ngram1: f32,
    pub idf_ngram2: f32,
    pub idf_ngram3: f32,
    pub tf_ngram1: u32,
    pub tf_ngram2: u32,
    pub tf_ngram3: u32,
    pub ngram_type: NgramType,

    pub end_flag: bool,
    pub end_flag_block: bool,
    pub is_embedded: bool,
    pub embedded_positions: [u32; 4],
    pub field_vec: SmallVec<[(u16, usize); 2]>,
    pub field_vec_ngram1: SmallVec<[(u16, usize); 2]>,
    pub field_vec_ngram2: SmallVec<[(u16, usize); 2]>,
    pub field_vec_ngram3: SmallVec<[(u16, usize); 2]>,
    pub bm25_flag: bool,
}

pub(crate) static DUMMY_VEC: Vec<BlockObjectIndex> = Vec::new();
pub(crate) static DUMMY_VEC_8: Vec<u8> = Vec::new();

impl Default for PostingListObjectQuery<'_> {
    fn default() -> Self {
        Self {
            posting_count: 0,
            max_list_score: 0.0,
            blocks: &DUMMY_VEC,
            blocks_index: 0,
            term: "".to_string(),
            key0: 0,
            compression_type: CompressionType::Delta,
            rank_position_pointer_range: 0,
            compressed_doc_id_range: 0,
            pointer_pivot_p_docid: 0,
            posting_pointer: 0,
            posting_pointer_previous: 0,
            byte_array: &DUMMY_VEC_8,
            p_block: 0,
            p_block_max: 0,
            p_docid: 0,
            p_docid_count: 0,
            rangebits: 0,
            docid: 0,
            bitposition: 0,
            run_end: 0,
            p_run: 0,
            p_run_count: 0,
            p_run_sum: 0,
            term_index_unique: 0,
            positions_count: 0,
            positions_pointer: 0,
            idf: 0.0,
            idf_ngram1: 0.0,
            idf_ngram2: 0.0,
            idf_ngram3: 0.0,
            ngram_type: NgramType::SingleTerm,
            is_embedded: false,
            embedded_positions: [0; 4],
            field_vec: SmallVec::new(),
            tf_ngram1: 0,
            tf_ngram2: 0,
            tf_ngram3: 0,
            field_vec_ngram1: SmallVec::new(),
            field_vec_ngram2: SmallVec::new(),
            field_vec_ngram3: SmallVec::new(),

            end_flag: false,
            end_flag_block: false,
            bm25_flag: true,
            intersect: 0,
            ulong_pos: 0,
        }
    }
}

/// NonUniquePostingListObjectQuery manages these posting list for each non-unique query term during intersection.
/// It references to the unique query terms.
#[derive(Clone)]
pub(crate) struct NonUniquePostingListObjectQuery<'a> {
    pub term_index_unique: usize,
    pub term_index_nonunique: usize,
    pub pos: u32,
    pub p_pos: i32,
    pub positions_pointer: usize,
    pub positions_count: u32,
    pub byte_array: &'a [u8],
    pub key0: u32,
    pub is_embedded: bool,
    pub embedded_positions: [u32; 4],
    pub p_field: usize,
    pub field_vec: SmallVec<[(u16, usize); 2]>,
}

/// Terms are converted to hashs. The index is divided into key hash range partitioned segments.
/// for each strip (key hash range) a separate dictionary (key hash - posting list) is maintained.
/// The index hash multiple segments, each segments contains multiple terms, each term has a postinglist, each postinglist has multiple blocks.
pub(crate) struct SegmentIndex {
    pub byte_array_blocks_pointer: Vec<(usize, usize, u32)>,
}

/// StripObject0 defines a strip (key hash range) within level 0. Level 0 is the mutable level where all writes are taking place.
/// After each 65.536 docs the level 0 is flushed as an immutable block to the next level
#[derive(Default, Debug, Clone)]
pub(crate) struct SegmentLevel0 {
    pub segment: AHashMap<u64, PostingListObject0>,
    pub positions_compressed: Vec<u8>,
}

/// FieldType defines the type of a field in the document: u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, point, string, stringset, text.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq, Default)]
pub enum FieldType {
    /// Unsigned 8-bit integer
    U8,
    /// Unsigned 16-bit integer
    U16,
    /// Unsigned 32-bit integer
    U32,
    /// Unsigned 64-bit integer
    U64,
    /// Signed 8-bit integer
    I8,
    /// Signed 16-bit integer
    I16,
    /// Signed 32-bit integer
    I32,
    /// Signed 64-bit integer
    I64,
    /// Timestamp is identical to I64, but to be used for Unix timestamps <https://en.wikipedia.org/wiki/Unix_time>.
    /// The reason for a separate FieldType is to enable the UI to interpret I64 as timestamp without using the field name as indicator.
    /// For date facets and filtering.
    Timestamp,
    /// Floating point 32-bit
    F32,
    /// Floating point 64-bit
    F64,
    /// Boolean
    Bool,
    /// String16
    /// allows a maximum cardinality of 65_535 (16 bit) distinct values, is space-saving.
    /// For filtering and sorting with low cardinality fields, e.g. boolean fields, enum fields, country, language, status, category, tags, etc.
    String16,
    /// String32
    /// allows a maximum cardinality of 4_294_967_295 (32 bit) distinct values
    /// For filtering and sorting with low cardinality fields, e.g. boolean fields, enum fields, country, language, status, category, tags, etc.
    String32,
    /// StringSet16 is a set of strings, e.g. tags, categories, keywords, authors, genres, etc.
    /// allows a maximum cardinality of 65_535 (16 bit) distinct values, is space-saving.
    /// For filtering and sorting with low cardinality fields, e.g. boolean fields, enum fields, country, language, status, category, tags, etc.
    StringSet16,
    /// StringSet32 is a set of strings, e.g. tags, categories, keywords, authors, genres, etc.
    /// allows a maximum cardinality of 4_294_967_295 (32 bit) distinct values
    /// For filtering and sorting with low cardinality fields, e.g. boolean fields, enum fields, country, language, status, category, tags, etc.
    StringSet32,
    /// Text is a text field, that will be tokenized by the selected Tokenizer into string tokens.
    #[default]
    Text,
    /// Hierarchical JSON Object, that will be tokenized by the selected Tokenizer into string tokens.
    /// The text is extracted from all levels of the JSON object and combined into a single text string (values only, not keys).
    /// In that respect, it is similar to combined fields in other search engines.
    /// Despite being indexed as a single text string, the hierarchical structure of an JSON object is preserved in the stored document, so that it can be retrieved in the search results.
    Json,
    /// Binary encodes and stores binary data in base64 format. This field type will not be tokenized and indexed.
    /// For embedding binary data, e.g. images, audio, video, pdf, … in JSON or CSV documents. A self-contained alternative to storing URLs to external resources.
    /// Using the [Data URI scheme](https://en.wikipedia.org/wiki/Data_URI_scheme) in JavaScript you can create an Image object and put the base64 as its src, including the data:image... part like this:
    ///```javascript
    /// var image = new Image();
    /// image.src = 'data:image/png;base64,iVBORw0K...';
    /// document.body.appendChild(image);
    ///```
    Binary,
}

/// Defines a field in index schema: field, stored, indexed , field_type, facet, boost.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct SchemaField {
    /// unique name of a field
    pub field: String,
    /// only stored fields are returned in the search results
    pub store: bool,
    /// only indexed fields can be searched
    /// Even if index_lexical is true in the schema, the field in the actual document is optional. You can index documents without that field, but you cannot search for that field in those documents.
    pub index_lexical: bool,
    /// only indexed fields can be searched
    /// Vector fields are required and validated before the document is queued for indexing.
    #[serde(skip_serializing_if = "is_default_bool")]
    #[serde(default = "default_false")]
    pub index_vector: bool,
    /// type of a field
    pub field_type: FieldType,
    /// Enable metadata filtering and sorting for string and numerical fields.
    /// both numerical and string fields can be indexed (indexed=true) and stored (stored=true) in the json document,
    /// but with field_facet=true they are additionally stored in a binary format, for fast filtering and sorting without docstore access (decompression, deserialization)
    #[serde(skip_serializing_if = "is_default_bool")]
    #[serde(default = "default_false")]
    pub facet: bool,

    /// Indicate the longest field in schema.
    /// Otherwise the longest field will be automatically detected in first index_document.
    /// Setting/detecting the longest field ensures efficient index encoding.
    #[serde(skip_serializing_if = "is_default_bool")]
    #[serde(default = "default_false")]
    pub longest: bool,

    /// optional custom weight factor for Bm25 ranking
    #[serde(skip_serializing_if = "is_default_f32")]
    #[serde(default = "default_1")]
    pub boost: f32,

    #[serde(skip)]
    pub(crate) indexed_field_id: usize,
    #[serde(default)]
    pub(crate) field_id: usize,
}

/// Defines a field in index schema: field, stored, indexed , field_type, facet, boost.
/// # Parameters
/// - field: unique name of a field
/// - store: only stored fields are returned in the search results
/// - index_lexical: index field into lexical index, only indexed fields can be searched.
/// - index_vector: index field into vector index, only indexed fields can be searched.
/// - field_type: type of a field: u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, text, strings, JSON
/// - facet: enable metadata filtering and sorting by field values
/// - `longest`: This allows to annotate (manually set) the longest field in schema.
///   Otherwise the longest field will be automatically detected in first index_document.
///   Setting/detecting the longest field ensures efficient index encoding.
/// - boost: optional custom weight factor for Bm25 ranking
/// # Returns
/// - SchemaField
/// # Example
/// ```rust
/// use seekpower::index::{SchemaField, FieldType};
/// let schema_field = SchemaField::new("title".to_string(), true, true, false, FieldType::String16, false, false, 1.0);
/// ```
impl SchemaField {
    /// Creates a new SchemaField.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        field: String,
        store: bool,
        index_lexical: bool,
        index_vector: bool,
        field_type: FieldType,
        facet: bool,
        longest: bool,
        boost: f32,
    ) -> Self {
        SchemaField {
            field,
            store,
            index_lexical,
            index_vector,
            field_type,
            facet,
            longest,
            boost,

            indexed_field_id: 0,
            field_id: 0,
        }
    }
}

fn default_false() -> bool {
    false
}

fn is_default_bool(num: &bool) -> bool {
    !(*num)
}

fn default_1() -> f32 {
    1.0
}

fn is_default_f32(num: &f32) -> bool {
    *num == 1.0
}

pub(crate) struct IndexedField {
    pub schema_field_name: String,
    pub field_length_sum: usize,
    pub indexed_field_id: usize,

    pub is_longest_field: bool,
}

/// StopwordType defines the stopword behavior: None, English, German, French, Spanish, Custom.
/// Stopwords are removed, both from index and query: for compact index size and faster queries.
/// Stopword removal has drawbacks: “The Who”, “Take That”, “Let it be”, “To be or not to be”, "The The", "End of days", "What might have been" are all valid queries for bands, songs, movies, literature,
/// but become impossible when stopwords are removed.
/// The lists of stop_words and frequent_words should not overlap.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub enum StopwordType {
    /// No stopwords
    #[default]
    None,
    /// English stopwords
    English,
    /// German stopwords
    German,
    /// French stopwords
    French,
    /// Spanish stopwords
    Spanish,
    /// Custom stopwords
    Custom {
        ///List of stopwords.
        terms: Vec<String>,
    },
}

/// FrequentwordType defines the frequentword behavior: None, English, German, French, Spanish, Custom.
/// Adjacent frequent terms are combined to bi-grams, both in index and query: for shorter posting lists and faster phrase queries (only for bi-grams of frequent terms).
/// The lists of stop_words and frequent_words should not overlap.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub enum FrequentwordType {
    /// No frequent words
    None,
    /// English frequent words
    #[default]
    English,
    /// German frequent words
    German,
    /// French frequent words
    French,
    /// Spanish frequent words
    Spanish,
    /// Custom frequent words
    Custom {
        ///List of frequent terms, max. 256 terms.
        terms: Vec<String>,
    },
}

/// Clustering defines the clustering behavior for approximate nearest neighbor (ANN) search: None, Auto, Fixed(usize).
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Default)]
pub enum Clustering {
    /// Exhaustive vector search, no clustering/ANN.
    None,
    /// The number of clusters is automatically determined depending on the number of vectors per level and shard.
    #[default]
    Auto,
    /// Set the number of clusters to a fixed value per level and shard.
    Fixed(usize),
}

impl fmt::Display for Clustering {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Clustering::None => write!(f, "None"),
            Clustering::Auto => write!(f, "Auto"),
            Clustering::Fixed(value) => write!(f, "Fixed({})", value),
        }
    }
}

/// Specifies LexicalSimilarity, TokenizerType and AccessType when creating an new index
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IndexMetaObject {
    /// unique index ID
    /// Only used in Seekpower server, not by the Seekpower library itself.
    /// In the Seekpower server with REST API, the index ID is used to specify the index (within the API key) where you want to index and search.
    pub id: u64,
    /// index name: used informational purposes
    pub name: String,
    /// LexicalSimilarity defines the scoring and ranking of the search results: Bm25f or Bm25fProximity
    pub lexical_similarity: LexicalSimilarity,
    /// TokenizerType defines the tokenizer behavior: AsciiAlphabetic, UnicodeAlphanumeric, UnicodeAlphanumericFolded, UnicodeAlphanumericZH
    pub tokenizer: TokenizerType,

    /// StopwordType defines the stopword behavior: None, English, German, French, Spanish, Custom.
    /// Stopwords are removed, both from index and query: for compact index size and faster queries.
    /// Stopword removal has drawbacks: “The Who”, “Take That”, “Let it be”, “To be or not to be”, "The The", "End of days", "What might have been" are all valid queries for bands, songs, movies, literature,
    /// but become impossible when stopwords are removed.
    /// The lists of stop_words and frequent_words should not overlap.
    #[serde(default)]
    pub stop_words: StopwordType,
    /// FrequentwordType defines the frequentword behavior: None, English, German, French, Spanish, Custom.
    /// Adjacent frequent terms are combined to bi-grams, both in index and query: for shorter posting lists and faster phrase queries (only for bi-grams of frequent terms).
    /// The lists of stop_words and frequent_words should not overlap.
    #[serde(default)]
    pub frequent_words: FrequentwordType,
    /// N-gram indexing: n-grams are indexed in addition to single terms, for fast phrase search, at the cost of higher index size
    /// Preference valid both for index time and query time. Any change requires reindexing.
    /// bitwise OR flags:
    /// SingleTerm = 0b00000000, always enabled in addition to the other optional NgramSet below
    /// NgramFF    = 0b00000001, frequent frequent
    /// NgramFR    = 0b00000010, frequent rare
    /// NgramRF    = 0b00000100, rare frequent
    /// NgramFFF   = 0b00001000, frequent frequent frequent
    /// NgramRFF   = 0b00010000, rare frequent frequent
    /// NgramFFR   = 0b00100000, frequent frequent rare
    /// NgramFRF   = 0b01000000, frequent rare frequent
    /// When **minimum index size** is more important than phrase query latency, we recommend **Single Terms**:
    /// `NgramSet::SingleTerm as u8`
    /// For a **good balance of latency and index size** cost, we recommend **Single Terms + Frequent Bigrams + Frequent Trigrams** (default):
    /// `NgramSet::SingleTerm as u8 | NgramSet::NgramFF as u8 | NgramSet::NgramFFF`
    /// When **minimal phrase query latency** is more important than low index size, we recommend **Single Terms + Mixed Bigrams + Frequent Trigrams**:
    /// `NgramSet::SingleTerm as u8 | NgramSet::NgramFF as u8 | NgramSet::NgramFR as u8 | NgramSet::NgramRF | NgramSet::NgramFFF`
    #[serde(default = "ngram_indexing_default")]
    pub ngram_indexing: u8,

    /// Compression algorithm for document store: None, Snappy
    #[serde(default = "doc_store_compression_default")]
    pub document_compression: DocumentCompression,

    /// AccessType defines where the index resides during search: Mmap
    pub access_type: AccessType,

    /// clustering defines the clustering behavior for approximate nearest neighbor (ANN) search: None, Auto, Fixed(usize).
    #[serde(default)]
    pub clustering: Clustering,

    /// External vector source: a Json array field or a Binary field with Base64 network-order bytes.
    #[serde(default)]
    pub inference: Inference,
}

fn ngram_indexing_default() -> u8 {
    NgramSet::NgramFF as u8 | NgramSet::NgramFFF as u8
}

fn doc_store_compression_default() -> DocumentCompression {
    DocumentCompression::Snappy
}

/// Metadata dictionary used by filtering and sorting. Value IDs retain insertion order.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct FacetField {
    /// Field name.
    pub name: String,
    /// String values or string sets, without occurrence counts.
    pub values: IndexMap<String, Vec<String>>,
    #[serde(skip)]
    pub(crate) offset: usize,
    #[serde(skip)]
    pub(crate) field_type: FieldType,
}

/// Shard wrapped in Arc and RwLock for concurrent read and write access.
pub type ShardArc = Arc<RwLock<Shard>>;

/// Index wrapped in Arc and RwLock for concurrent read and write access.
pub type IndexArc = Arc<RwLock<Index>>;

/// The shard object of the index. It contains all levels and all segments of the index.
/// It also contains all properties that control indexing and intersection.
pub struct Shard {
    /// Current on-disk format major version; opening requires an exact version match.
    pub index_format_version_major: u16,
    /// Current on-disk format minor version; opening requires an exact version match.
    pub index_format_version_minor: u16,

    /// Number of indexed documents
    pub indexed_doc_count: usize,
    /// Number of indexed vectors
    pub(crate) indexed_vector_count: usize,
    /// Number of indexed clusters
    pub(crate) indexed_cluster_count: usize,

    /// Number of committed documents
    pub committed_doc_count: usize,
    /// The index countains indexed, but uncommitted documents. Documents will either committed automatically once the number exceeds 64K documents **per shard**, or once commit is invoked manually.
    pub(crate) uncommitted: bool,

    /// Indicates whether the index has been modified since the start.
    pub(crate) modified: bool,

    /// Defines a field in index schema: field, stored, indexed , field_type, facet, boost.
    pub schema_map: HashMap<String, SchemaField>,
    /// List of stored fields in the index: get_document works only with stored fields
    pub stored_field_names: Vec<String>,
    /// Specifies LexicalSimilarity, TokenizerType and AccessType when creating an new index
    pub meta: IndexMetaObject,

    pub(crate) is_last_level_incomplete: bool,
    pub(crate) last_level_index_file_start_pos: u64,
    pub(crate) last_level_docstore_file_start_pos: u64,
    pub(crate) last_level_vector_file_start_pos: u64,

    /// Number of allowed parallel indexed documents (default=available_parallelism). Can be used to detect wehen all indexing processes are finished.
    pub(crate) semaphore: Arc<Semaphore>,
    pub(crate) docstore_file: File,
    pub(crate) docstore_file_mmap: Mmap,

    pub(crate) delete_file: File,
    pub(crate) delete_hashset: AHashSet<usize>,

    pub(crate) index_file: File,
    pub(crate) index_path_string: String,
    pub(crate) index_file_mmap: Mmap,

    pub(crate) compressed_index_segment_block_buffer: Vec<u8>,
    pub(crate) compressed_docstore_segment_block_buffer: Vec<u8>,

    pub(crate) segment_number1: usize,

    pub(crate) document_length_normalized_average: f32,
    pub(crate) positions_sum_normalized: u64,

    pub(crate) level_index: Vec<LevelIndex>,
    pub(crate) segments_index: Vec<SegmentIndex>,
    pub(crate) segments_level0: Vec<SegmentLevel0>,

    pub(crate) segment_number_mask1: u32,

    pub(crate) indexed_field_vec: Vec<IndexedField>,
    pub(crate) indexed_field_id_bits: usize,
    pub(crate) indexed_field_id_mask: usize,
    pub(crate) longest_field_id: usize,
    pub(crate) longest_field_auto: bool,
    pub(crate) indexed_schema_vec: Vec<SchemaField>,

    pub(crate) document_length_compressed_array: Vec<[u8; ROARING_BLOCK_SIZE]>,
    pub(crate) key_count_sum: u64,

    pub(crate) block_id: usize,
    pub(crate) strip_compressed_sum: u64,
    pub(crate) postings_buffer: Vec<u8>,
    pub(crate) postings_buffer_pointer: usize,

    pub(crate) size_compressed_positions_index: u64,
    pub(crate) size_compressed_docid_index: u64,

    pub(crate) postinglist_count: usize,
    pub(crate) docid_count: usize,
    pub(crate) position_count: usize,

    pub(crate) mute: bool,
    pub(crate) frequentword_results: AHashMap<String, ResultObject>,

    pub(crate) facets: Vec<FacetField>,
    pub(crate) facets_map: AHashMap<String, usize>,
    pub(crate) facets_size_sum: usize,
    pub(crate) facets_file: File,
    pub(crate) facets_file_mmap: MmapMut,
    pub(crate) bm25_component_cache: [f32; 256],

    pub(crate) string_set_to_single_term_id_vec: Vec<AHashMap<String, AHashSet<u32>>>,

    #[cfg(feature = "zh")]
    pub(crate) word_segmentation_option: Option<Jieba>,

    pub(crate) shard_number: usize,
    pub(crate) index_option: Option<Arc<RwLock<Index>>>,

    pub(crate) stop_words: AHashSet<String>,
    pub(crate) frequent_words: Vec<String>,
    pub(crate) frequent_hashset: AHashSet<u64>,
    pub(crate) key_head_size: usize,

    /// AVX2 (x86_64) support enabled.
    pub(crate) is_avx2: bool,
    /// NEON (aarch64) support enabled.
    pub(crate) is_neon: bool,
    /// Any SIMD backend (`is_avx2 || is_neon`). Call sites dispatch on this
    /// to pick between the `*_simd` SIMD path and the scalar fallback.
    pub(crate) is_simd: bool,
    pub(crate) is_vector_indexing: bool,
    pub(crate) is_lexical_indexing: bool,
    pub(crate) vector_file: File,
    pub(crate) vector_file_mmap: Mmap,
    pub(crate) block_vector_buffer: Vec<ParentMedoid>,
    pub(crate) vector_dimensions: usize,
    pub(crate) vector_dimensions_original: usize,
    pub(crate) vector_precision: Precision,
    pub(crate) quantization: Quantization,
    pub(crate) vector_similarity: VectorSimilarity,

    pub(crate) min_vector_value: f32,
    pub(crate) max_vector_value: f32,
    pub(crate) turbo_quant: TurboQuant,
}

/// The root object of the index. It contains all levels and all segments of the index.
/// It also contains all properties that control indexing and intersection.
pub struct Index {
    pub(crate) docid_global: Arc<RwLock<usize>>,

    /// Current on-disk format major version; opening requires an exact version match.
    pub index_format_version_major: u16,
    /// Current on-disk format minor version; opening requires an exact version match.
    pub index_format_version_minor: u16,

    /// Number of indexed documents
    pub(crate) indexed_doc_count: usize,
    /// Number of indexed vectors
    pub indexed_vector_count: usize,
    /// Number of indexed clusters
    pub indexed_cluster_count: usize,

    /// Number of deleted documents
    pub(crate) deleted_doc_count: usize,
    /// Defines a field in index schema: field, stored, indexed , field_type, facet, boost.
    pub schema_map: HashMap<String, SchemaField>,
    /// List of stored fields in the index: get_document works only with stored fields
    pub stored_field_names: Vec<String>,
    /// Specifies LexicalSimilarity, TokenizerType and AccessType when creating an new index
    pub meta: IndexMetaObject,

    pub(crate) index_file: File,
    pub(crate) index_path_string: String,

    pub(crate) compressed_index_segment_block_buffer: Vec<u8>,

    pub(crate) segment_number1: usize,
    pub(crate) segment_number_mask1: u32,

    pub(crate) indexed_field_vec: Vec<IndexedField>,

    pub(crate) mute: bool,

    pub(crate) facets: Vec<FacetField>,

    pub(crate) shard_number: usize,
    pub(crate) shard_vec: Vec<Arc<RwLock<Shard>>>,

    /// The precision of the vectors: Float32 or Int8
    pub vector_precision: Precision,
    pub(crate) quantization: Quantization,
    /// The dimensions of the vectors: e.g. 64, 128, 256, 512, 1024, 768, 1536, 2048, 4096
    /// When using TurboQuant, padded to the next power of two for the Fast Walsh-Hadamard Transform.
    pub vector_dimensions: usize,
    /// The original dimensions of the vectors.
    pub vector_dimensions_original: usize,
    /// Vector similarity function for approximate nearest neighbor (ANN) search: Cosine, Euclidean, DotProduct.
    pub vector_similarity: VectorSimilarity,
    /// AVX2 support enabled
    pub(crate) is_avx2: bool,
    /// NEON support enabled (aarch64 only).
    pub(crate) is_neon: bool,
    /// Any SIMD backend enabled (`is_avx2 || is_neon`).
    pub(crate) is_simd: bool,
    /// Indicates whether the index contains vector embeddings for vector search.
    pub(crate) is_vector_indexing: bool,
    /// Indicates whether the index contains lexical term indexing for lexical search.
    pub(crate) is_lexical_indexing: bool,
    pub(crate) turbo_quant: TurboQuant,
}

/// Get the version of the Seekpower search library
pub fn version() -> &'static str {
    VERSION
}

/// N-gram indexing: n-grams are indexed in addition to single terms, for faster phrase search, at the cost of higher index size
/// Setting valid both for index time and query time. Any change requires reindexing.
/// bitwise OR flags:
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq, FromPrimitive)]
pub enum NgramSet {
    /// no n-grams, only single terms are indexed
    SingleTerm = 0b00000000,
    /// Ngram frequent frequent
    NgramFF = 0b00000001,
    /// Ngram frequent rare
    NgramFR = 0b00000010,
    /// Ngram rare frequent
    NgramRF = 0b00000100,
    /// Ngram frequent frequent frequent
    NgramFFF = 0b00001000,
    /// Ngram rare frequent frequent
    NgramRFF = 0b00010000,
    /// Ngram frequent frequent rare
    NgramFFR = 0b00100000,
    /// Ngram frequent rare frequent
    NgramFRF = 0b01000000,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq, FromPrimitive, Default)]
pub(crate) enum NgramType {
    /// no n-grams, only single terms are indexed
    #[default]
    SingleTerm = 0,
    /// Ngram frequent frequent
    NgramFF = 1,
    /// Ngram frequent rare
    NgramFR = 2,
    /// Ngram rare frequent
    NgramRF = 3,
    /// Ngram frequent frequent frequent
    NgramFFF = 4,
    /// Ngram rare frequent frequent
    NgramRFF = 5,
    /// Ngram frequent frequent rare
    NgramFFR = 6,
    /// Ngram frequent rare frequent
    NgramFRF = 7,
}

/// Create an index with memory-mapped committed data.
/// Inner data structures for create index and open_index
/// * `index_path` - index path.
/// * `meta` - index meta object.
/// * `schema` - schema.
/// * `segment_number_bits1` - number of index segments: e.g. 11 bits for 2048 segments.
/// * `mute` - prevent emitting status messages (e.g. when using pipes for data interprocess communication).
/// * `force_shard_number` - set number of shards manually or automatically.
///   - none: number of shards is set automatically = number of physical processor cores (default)
///   - small: slower indexing, higher latency, slightly higher throughput, faster realtime search, lower RAM consumption
///   - large: faster indexing, lower latency, slightly lower throughput, slower realtime search, higher RAM consumption
pub async fn create_index(
    index_path: &Path,
    meta: IndexMetaObject,
    schema: &Vec<SchemaField>,
    segment_number_bits1: usize,
    mute: bool,
    force_shard_number: Option<usize>,
) -> Result<IndexArc, String> {
    create_index_root(
        index_path,
        meta,
        schema,
        true,
        segment_number_bits1,
        mute,
        force_shard_number,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn create_index_root(
    index_path: &Path,
    meta: IndexMetaObject,
    #[allow(clippy::ptr_arg)] schema: &Vec<SchemaField>,
    serialize_schema: bool,
    segment_number_bits1: usize,
    mute: bool,
    force_shard_number: Option<usize>,
) -> Result<IndexArc, String> {
    crate::vector::validate_vector_schema(&meta.inference, schema)?;
    let segment_number1 = 1usize << segment_number_bits1;
    let segment_number_mask1 = (1u32 << segment_number_bits1) - 1;

    let index_path_buf = index_path.to_path_buf();
    let index_path_string = index_path_buf.to_str().unwrap();

    if !index_path.exists() {
        if !mute {
            log::info!("index path created: {} ", index_path_string);
        }
        fs::create_dir_all(index_path).unwrap();
    }

    match File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(Path::new(index_path).join(INDEX_FILENAME))
    {
        Ok(index_file) => {
            let mut is_vector_indexing = false;
            let mut is_lexical_indexing = false;
            let mut schema = schema.clone();
            for schema_field in schema.iter_mut() {
                if schema_field.field_type == FieldType::Binary && schema_field.index_lexical {
                    schema_field.index_lexical = false;
                }
                if schema_field.index_vector {
                    is_vector_indexing = true;
                }
                if schema_field.index_lexical {
                    is_lexical_indexing = true;
                }
            }

            let mut document_length_compressed_array: Vec<[u8; ROARING_BLOCK_SIZE]> = Vec::new();
            let mut indexed_field_vec: Vec<IndexedField> = Vec::new();
            let mut facets_vec: Vec<FacetField> = Vec::new();
            let mut facets_map: AHashMap<String, usize> = AHashMap::new();

            let mut schema_map: HashMap<String, SchemaField> = HashMap::new();
            let mut indexed_schema_vec: Vec<SchemaField> = Vec::new();
            let mut stored_field_names = Vec::new();
            let mut facets_size_sum = 0;
            let mut longest_field_id_option: Option<usize> = None;
            for (i, schema_field) in schema.iter().enumerate() {
                let mut schema_field_clone = schema_field.clone();

                schema_field_clone.indexed_field_id = indexed_field_vec.len();
                if schema_field.longest && schema_field.index_lexical {
                    longest_field_id_option = Some(schema_field_clone.indexed_field_id);
                }

                schema_field_clone.field_id = i;
                schema_map.insert(schema_field.field.clone(), schema_field_clone.clone());

                if schema_field.facet {
                    let facet_size = match schema_field.field_type {
                        FieldType::U8 => 1,
                        FieldType::U16 => 2,
                        FieldType::U32 => 4,
                        FieldType::U64 => 8,
                        FieldType::I8 => 1,
                        FieldType::I16 => 2,
                        FieldType::I32 => 4,
                        FieldType::I64 => 8,
                        FieldType::Timestamp => 8,
                        FieldType::F32 => 4,
                        FieldType::F64 => 8,
                        FieldType::String16 => 2,
                        FieldType::String32 => 4,
                        FieldType::StringSet16 => 2,
                        FieldType::StringSet32 => 4,
                        _ => 1,
                    };

                    facets_map.insert(schema_field.field.clone(), facets_vec.len());
                    facets_vec.push(FacetField {
                        name: schema_field.field.clone(),
                        values: IndexMap::new(),

                        offset: facets_size_sum,
                        field_type: schema_field.field_type.clone(),
                    });
                    facets_size_sum += facet_size;
                }

                if schema_field.index_lexical || schema_field.index_vector {
                    indexed_field_vec.push(IndexedField {
                        schema_field_name: schema_field.field.clone(),
                        is_longest_field: false,
                        field_length_sum: 0,
                        indexed_field_id: indexed_field_vec.len(),
                    });
                    indexed_schema_vec.push(schema_field_clone);
                    document_length_compressed_array.push([0; ROARING_BLOCK_SIZE]);
                }

                if schema_field.store {
                    stored_field_names.push(schema_field.field.clone());
                }
            }

            if !facets_vec.is_empty()
                && let Ok(file) = File::open(Path::new(index_path).join(FACET_VALUES_FILENAME))
            {
                let mut facets: Vec<FacetField> = serde_json::from_reader(BufReader::new(file))
                    .map_err(|e| format!("invalid metadata dictionary: {e}"))?;
                if facets_vec.len() == facets.len() {
                    for i in 0..facets.len() {
                        facets[i].offset = facets_vec[i].offset;
                        facets[i].field_type = facets_vec[i].field_type.clone();
                    }
                }
                facets_vec = facets;
            }

            let shard_number = if let Some(shard_number) = force_shard_number {
                shard_number
            } else {
                cmp::min(
                    available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or_else(|_| num_cpus::get_physical()),
                    num_cpus::get_physical(),
                )
            };

            let (vector_dimensions, vector_precision, quantization, vector_similarity) =
                match &meta.inference {
                    Inference::External {
                        dimensions,
                        precision,
                        quantization,
                        similarity,
                    } => (*dimensions, *precision, *quantization, *similarity),
                    Inference::None => (
                        0,
                        Precision::None,
                        Quantization::None,
                        VectorSimilarity::Cosine,
                    ),
                };

            let turbo_quant = if quantization == Quantization::TurboQuantI8 {
                TurboQuant::new(vector_dimensions, 1234)
            } else {
                TurboQuant::new(0, 1234)
            };

            let vector_dimensions_original = vector_dimensions;
            let vector_dimensions = if quantization == Quantization::TurboQuantI8
                && vector_precision == Precision::F32
            {
                TurboQuant::next_power_of_two(vector_dimensions)
            } else {
                vector_dimensions
            };

            // The native vector kernels consume complete SIMD lanes. Keep arbitrary
            // external dimensions on the existing scalar path, including small TQ vectors.
            let quantized = quantization != Quantization::None || vector_precision == Precision::I8;
            let lane_width = if *IS_AVX2 {
                if quantized { 32 } else { 8 }
            } else if *IS_NEON {
                if quantized { 16 } else { 4 }
            } else {
                1
            };
            let is_simd =
                *IS_SIMD && vector_dimensions > 0 && vector_dimensions.is_multiple_of(lane_width);

            let mut shard_vec: Vec<Arc<RwLock<Shard>>> = Vec::new();
            if serialize_schema {
                let mut result_object_list = Vec::new();
                let index_path_clone = Arc::new(index_path.to_path_buf());
                for i in 0..shard_number {
                    let index_path_clone2 = index_path_clone.clone();
                    let meta_clone = meta.clone();
                    let schema_clone = schema.clone();
                    let turbo_quant_clone = turbo_quant.clone();
                    result_object_list.push(tokio::spawn(async move {
                        let shard_path = index_path_clone2.join("shards").join(i.to_string());
                        let mut shard_meta = meta_clone.clone();
                        shard_meta.id = i as u64;

                        let mut shard = create_shard(
                            &shard_path,
                            &shard_meta,
                            &schema_clone,
                            serialize_schema,
                            segment_number_bits1,
                            mute,
                            longest_field_id_option,
                        )
                        .unwrap();
                        shard.shard_number = shard_number;
                        shard.vector_dimensions = vector_dimensions;
                        shard.vector_dimensions_original = vector_dimensions_original;
                        shard.vector_precision = vector_precision;
                        shard.quantization = quantization;
                        shard.vector_similarity = vector_similarity;
                        shard.is_avx2 = *IS_AVX2;
                        shard.is_neon = *IS_NEON;
                        shard.is_simd = is_simd;
                        shard.turbo_quant = turbo_quant_clone;

                        let shard_arc = Arc::new(RwLock::new(shard));
                        (shard_arc, i)
                    }));
                }
                for result_object_shard in result_object_list {
                    let ro_shard = result_object_shard.await.unwrap();
                    shard_vec.push(ro_shard.0);
                }
            }

            let mut index = Index {
                docid_global: Arc::new(RwLock::new(0)),
                index_format_version_major: INDEX_FORMAT_VERSION_MAJOR,
                index_format_version_minor: INDEX_FORMAT_VERSION_MINOR,

                index_file,
                index_path_string: index_path_string.to_owned(),
                stored_field_names,

                compressed_index_segment_block_buffer: vec![0; 10_000_000],
                indexed_doc_count: 0,
                indexed_vector_count: 0,
                indexed_cluster_count: 0,
                deleted_doc_count: 0,
                segment_number1: 0,
                segment_number_mask1: 0,
                schema_map,
                indexed_field_vec,
                meta: meta.clone(),
                mute,
                facets: facets_vec,

                shard_number,
                shard_vec,

                vector_dimensions,
                vector_dimensions_original,
                vector_precision,
                quantization,
                vector_similarity,
                is_avx2: *IS_AVX2,
                is_neon: *IS_NEON,
                is_simd,
                is_vector_indexing,
                is_lexical_indexing,
                turbo_quant,
            };

            let file_len = index.index_file.metadata().unwrap().len();
            if file_len == 0 {
                write_u16(
                    INDEX_FORMAT_VERSION_MAJOR,
                    &mut index.compressed_index_segment_block_buffer,
                    0,
                );
                write_u16(
                    INDEX_FORMAT_VERSION_MINOR,
                    &mut index.compressed_index_segment_block_buffer,
                    2,
                );
                let _ = index.index_file.write(
                    &index.compressed_index_segment_block_buffer[0..INDEX_HEADER_SIZE as usize],
                );
            } else {
                let _ = index.index_file.read(
                    &mut index.compressed_index_segment_block_buffer[0..INDEX_HEADER_SIZE as usize],
                );
                index.index_format_version_major =
                    read_u16(&index.compressed_index_segment_block_buffer, 0);
                index.index_format_version_minor =
                    read_u16(&index.compressed_index_segment_block_buffer, 2);

                if (
                    index.index_format_version_major,
                    index.index_format_version_minor,
                ) != (INDEX_FORMAT_VERSION_MAJOR, INDEX_FORMAT_VERSION_MINOR)
                {
                    return Err(format!(
                        "unsupported index format {}.{}; expected {}.{}",
                        index.index_format_version_major,
                        index.index_format_version_minor,
                        INDEX_FORMAT_VERSION_MAJOR,
                        INDEX_FORMAT_VERSION_MINOR,
                    ));
                }
            }

            index.segment_number1 = segment_number1;
            index.segment_number_mask1 = segment_number_mask1;

            if serialize_schema {
                serde_json::to_writer(
                    &File::create(Path::new(index_path).join(SCHEMA_FILENAME)).unwrap(),
                    &schema,
                )
                .unwrap();

                serde_json::to_writer(
                    &File::create(Path::new(index_path).join(META_FILENAME)).unwrap(),
                    &index.meta,
                )
                .unwrap();
            }

            let index_arc = Arc::new(RwLock::new(index));

            if serialize_schema {
                for shard in index_arc.write().await.shard_vec.iter() {
                    shard.write().await.index_option = Some(index_arc.clone());
                }
            }

            Ok(index_arc)
        }
        Err(e) => {
            log::warn!("file opening error");
            Err(e.to_string())
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn create_shard(
    index_path: &Path,
    meta: &IndexMetaObject,
    schema: &Vec<SchemaField>,
    serialize_schema: bool,
    segment_number_bits1: usize,
    mute: bool,
    longest_field_id_option: Option<usize>,
) -> Result<Shard, String> {
    let segment_number1 = 1usize << segment_number_bits1;
    let segment_number_mask1 = (1u32 << segment_number_bits1) - 1;

    let index_path_buf = index_path.to_path_buf();
    let index_path_string = index_path_buf.to_str().unwrap();

    if !index_path.exists() {
        fs::create_dir_all(index_path).unwrap();
    }

    match File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(Path::new(index_path).join(INDEX_FILENAME))
    {
        Ok(index_file) => {
            let docstore_file = File::options()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(Path::new(index_path).join(DOCSTORE_FILENAME))
                .unwrap();

            let delete_file = File::options()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(Path::new(index_path).join(DELETE_FILENAME))
                .unwrap();

            let facets_file = File::options()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(Path::new(index_path).join(FACET_FILENAME))
                .unwrap();

            let vector_file = File::options()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(Path::new(index_path).join(VECTOR_FILENAME))
                .unwrap();

            let mut document_length_compressed_array: Vec<[u8; ROARING_BLOCK_SIZE]> = Vec::new();
            let mut indexed_field_vec: Vec<IndexedField> = Vec::new();
            let mut facets_vec: Vec<FacetField> = Vec::new();
            let mut facets_map: AHashMap<String, usize> = AHashMap::new();

            let mut schema_map: HashMap<String, SchemaField> = HashMap::new();
            let mut indexed_schema_vec: Vec<SchemaField> = Vec::new();
            let mut stored_fields_flag = false;
            let mut is_vector_indexing = false;
            let mut is_lexical_indexing = false;
            let mut stored_field_names = Vec::new();
            let mut facets_size_sum = 0;
            for (i, schema_field) in schema.iter().enumerate() {
                if schema_field.index_vector {
                    is_vector_indexing = true;
                }
                if schema_field.index_lexical {
                    is_lexical_indexing = true;
                }
                let mut schema_field_clone = schema_field.clone();

                schema_field_clone.indexed_field_id = indexed_field_vec.len();

                schema_field_clone.field_id = i;
                schema_map.insert(schema_field.field.clone(), schema_field_clone.clone());

                if schema_field.facet {
                    let facet_size = match schema_field.field_type {
                        FieldType::U8 => 1,
                        FieldType::U16 => 2,
                        FieldType::U32 => 4,
                        FieldType::U64 => 8,
                        FieldType::I8 => 1,
                        FieldType::I16 => 2,
                        FieldType::I32 => 4,
                        FieldType::I64 => 8,
                        FieldType::Timestamp => 8,
                        FieldType::F32 => 4,
                        FieldType::F64 => 8,
                        FieldType::String16 => 2,
                        FieldType::String32 => 4,
                        FieldType::StringSet16 => 2,
                        FieldType::StringSet32 => 4,
                        _ => 1,
                    };

                    facets_map.insert(schema_field.field.clone(), facets_vec.len());
                    facets_vec.push(FacetField {
                        name: schema_field.field.clone(),
                        values: IndexMap::new(),

                        offset: facets_size_sum,
                        field_type: schema_field.field_type.clone(),
                    });
                    facets_size_sum += facet_size;
                }

                if schema_field.index_lexical || schema_field.index_vector {
                    indexed_field_vec.push(IndexedField {
                        schema_field_name: schema_field.field.clone(),
                        is_longest_field: false,
                        field_length_sum: 0,
                        indexed_field_id: indexed_field_vec.len(),
                    });
                    indexed_schema_vec.push(schema_field_clone);
                    document_length_compressed_array.push([0; ROARING_BLOCK_SIZE]);
                }

                if schema_field.store {
                    stored_fields_flag = true;
                    stored_field_names.push(schema_field.field.clone());
                }
            }

            let indexed_field_id_bits =
                (usize::BITS - (indexed_field_vec.len() - 1).leading_zeros()) as usize;

            let index_file_mmap;
            let docstore_file_mmap = {
                index_file_mmap = unsafe { Mmap::map(&index_file).expect("Unable to create Mmap") };
                unsafe { Mmap::map(&docstore_file).expect("Unable to create Mmap") }
            };

            if !facets_vec.is_empty()
                && let Ok(file) = File::open(Path::new(index_path).join(FACET_VALUES_FILENAME))
            {
                let mut facets: Vec<FacetField> = serde_json::from_reader(BufReader::new(file))
                    .map_err(|e| format!("invalid metadata dictionary: {e}"))?;
                if facets_vec.len() == facets.len() {
                    for i in 0..facets.len() {
                        facets[i].offset = facets_vec[i].offset;
                        facets[i].field_type = facets_vec[i].field_type.clone();
                    }
                }
                facets_vec = facets;
            }

            let facets_file_mmap = if !facets_vec.is_empty() {
                if facets_file.metadata().unwrap().len() == 0 {
                    facets_file
                        .set_len((facets_size_sum * ROARING_BLOCK_SIZE) as u64)
                        .expect("Unable to set len");
                }

                unsafe { MmapMut::map_mut(&facets_file).expect("Unable to create Mmap") }
            } else {
                unsafe { MmapMut::map_mut(&facets_file).expect("Unable to create Mmap") }
            };

            let vector_file_mmap =
                unsafe { Mmap::map(&vector_file).expect("Unable to create Mmap") };

            let facets_len = facets_vec.len();

            #[cfg(feature = "zh")]
            let word_segmentation_option = if meta.tokenizer == TokenizerType::UnicodeAlphanumericZH
            {
                Some(Jieba::new())
            } else {
                None
            };

            let shard_number = 1;

            let stop_words: AHashSet<String> = match &meta.stop_words {
                StopwordType::None => AHashSet::new(),
                StopwordType::English => FREQUENT_EN.lines().map(|x| x.to_string()).collect(),
                StopwordType::German => FREQUENT_DE.lines().map(|x| x.to_string()).collect(),
                StopwordType::French => FREQUENT_FR.lines().map(|x| x.to_string()).collect(),
                StopwordType::Spanish => FREQUENT_ES.lines().map(|x| x.to_string()).collect(),
                StopwordType::Custom { terms } => terms.iter().map(|x| x.to_string()).collect(),
            };

            let frequent_words: Vec<String> = match &meta.frequent_words {
                FrequentwordType::None => Vec::new(),
                FrequentwordType::English => {
                    let mut words: Vec<String> =
                        FREQUENT_EN.lines().map(|x| x.to_string()).collect();
                    words.sort_unstable();
                    words
                }
                FrequentwordType::German => {
                    let mut words: Vec<String> =
                        FREQUENT_DE.lines().map(|x| x.to_string()).collect();
                    words.sort_unstable();
                    words
                }
                FrequentwordType::French => {
                    let mut words: Vec<String> =
                        FREQUENT_FR.lines().map(|x| x.to_string()).collect();
                    words.sort_unstable();
                    words
                }
                FrequentwordType::Spanish => {
                    let mut words: Vec<String> =
                        FREQUENT_ES.lines().map(|x| x.to_string()).collect();
                    words.sort_unstable();
                    words
                }
                FrequentwordType::Custom { terms } => {
                    let mut words: Vec<String> = terms.iter().map(|x| x.to_string()).collect();
                    words.sort_unstable();
                    words
                }
            };

            let frequent_hashset: AHashSet<u64> = frequent_words
                .iter()
                .map(|x| hash64(x.as_bytes()))
                .collect();

            let mut index = Shard {
                semaphore: Arc::new(Semaphore::new(1)),

                index_format_version_major: INDEX_FORMAT_VERSION_MAJOR,
                index_format_version_minor: INDEX_FORMAT_VERSION_MINOR,
                docstore_file,
                delete_file,
                delete_hashset: AHashSet::new(),
                index_file,
                index_path_string: index_path_string.to_owned(),
                index_file_mmap,
                docstore_file_mmap,
                stored_field_names,
                compressed_index_segment_block_buffer: vec![0; 10_000_000],
                compressed_docstore_segment_block_buffer: if stored_fields_flag {
                    vec![0; ROARING_BLOCK_SIZE * 4]
                } else {
                    Vec::new()
                },
                document_length_normalized_average: 0.0,
                indexed_doc_count: 0,
                committed_doc_count: 0,
                is_last_level_incomplete: false,
                last_level_index_file_start_pos: 0,
                last_level_docstore_file_start_pos: 0,
                last_level_vector_file_start_pos: 0,
                positions_sum_normalized: 0,
                segment_number1: 0,
                segment_number_mask1: 0,
                level_index: Vec::new(),
                segments_index: Vec::new(),
                segments_level0: Vec::new(),

                uncommitted: false,
                modified: false,
                schema_map,
                indexed_field_id_bits,
                indexed_field_id_mask: (1usize << indexed_field_id_bits) - 1,
                longest_field_id: longest_field_id_option.unwrap_or_default(),
                longest_field_auto: longest_field_id_option.is_none(),
                indexed_field_vec,
                indexed_schema_vec,
                meta: meta.clone(),
                document_length_compressed_array,
                key_count_sum: 0,

                block_id: 0,
                strip_compressed_sum: 0,
                postings_buffer: Vec::new(),
                postings_buffer_pointer: 0,

                docid_count: 0,
                size_compressed_docid_index: 0,
                size_compressed_positions_index: 0,
                position_count: 0,
                postinglist_count: 0,
                mute,
                frequentword_results: AHashMap::new(),
                facets: facets_vec,
                facets_map,
                facets_size_sum,
                facets_file,
                facets_file_mmap,
                string_set_to_single_term_id_vec: vec![AHashMap::new(); facets_len],
                bm25_component_cache: [0.0; 256],
                #[cfg(feature = "zh")]
                word_segmentation_option,

                shard_number,
                index_option: None,
                stop_words,
                frequent_words,
                frequent_hashset,
                key_head_size: if meta.ngram_indexing == 0 {
                    20
                } else if meta.ngram_indexing < 8 {
                    22
                } else {
                    23
                },

                vector_file,
                vector_file_mmap,
                indexed_vector_count: 0,
                indexed_cluster_count: 0,
                is_vector_indexing,
                is_lexical_indexing,
                block_vector_buffer: Vec::new(),
                vector_dimensions: 0,
                vector_dimensions_original: 0,
                vector_precision: Precision::None,
                quantization: Quantization::None,
                vector_similarity: VectorSimilarity::Dot,
                is_avx2: false,
                is_neon: false,
                is_simd: false,
                min_vector_value: f32::MAX,
                max_vector_value: f32::MIN,
                turbo_quant: TurboQuant::new(0, 1234),
            };

            let file_len = index.index_file.metadata().unwrap().len();
            if file_len == 0 {
                write_u16(
                    INDEX_FORMAT_VERSION_MAJOR,
                    &mut index.compressed_index_segment_block_buffer,
                    0,
                );
                write_u16(
                    INDEX_FORMAT_VERSION_MINOR,
                    &mut index.compressed_index_segment_block_buffer,
                    2,
                );
                let _ = index.index_file.write(
                    &index.compressed_index_segment_block_buffer[0..INDEX_HEADER_SIZE as usize],
                );
            } else {
                let _ = index.index_file.read(
                    &mut index.compressed_index_segment_block_buffer[0..INDEX_HEADER_SIZE as usize],
                );
                index.index_format_version_major =
                    read_u16(&index.compressed_index_segment_block_buffer, 0);
                index.index_format_version_minor =
                    read_u16(&index.compressed_index_segment_block_buffer, 2);

                if (
                    index.index_format_version_major,
                    index.index_format_version_minor,
                ) != (INDEX_FORMAT_VERSION_MAJOR, INDEX_FORMAT_VERSION_MINOR)
                {
                    return Err(format!(
                        "unsupported index format {}.{}; expected {}.{}",
                        index.index_format_version_major,
                        index.index_format_version_minor,
                        INDEX_FORMAT_VERSION_MAJOR,
                        INDEX_FORMAT_VERSION_MINOR,
                    ));
                }
            }

            index.segment_number1 = segment_number1;
            index.segment_number_mask1 = segment_number_mask1;
            index.segments_level0 = vec![
                SegmentLevel0 {
                    segment: AHashMap::with_capacity(SEGMENT_KEY_CAPACITY),
                    ..Default::default()
                };
                index.segment_number1
            ];

            index.segments_index = Vec::new();
            for _i in 0..index.segment_number1 {
                index.segments_index.push(SegmentIndex {
                    byte_array_blocks_pointer: Vec::new(),
                });
            }

            if serialize_schema {
                serde_json::to_writer(
                    &File::create(Path::new(index_path).join(SCHEMA_FILENAME)).unwrap(),
                    &schema,
                )
                .unwrap();

                serde_json::to_writer(
                    &File::create(Path::new(index_path).join(META_FILENAME)).unwrap(),
                    &index.meta,
                )
                .unwrap();
            }

            Ok(index)
        }
        Err(e) => {
            log::warn!("file opening error");
            Err(e.to_string())
        }
    }
}

#[inline(always)]
pub(crate) fn get_document_length_compressed_mmap(
    index: &Shard,
    field_id: usize,
    block_id: usize,
    doc_id_block: usize,
) -> u8 {
    index.index_file_mmap[index.level_index[block_id].document_length_compressed_array_pointer
        + (field_id << 16)
        + doc_id_block]
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn get_max_score(
    index: &Shard,
    segment: &SegmentIndex,
    posting_count_ngram_1: u32,
    posting_count_ngram_2: u32,
    posting_count_ngram_3: u32,
    posting_count: u32,
    block_id: usize,
    max_docid: usize,
    max_p_docid: usize,
    pointer_pivot_p_docid: usize,
    compression_type_pointer: u32,
    ngram_type: &NgramType,
) -> f32 {
    let byte_array = {
        &index.index_file_mmap[segment.byte_array_blocks_pointer[block_id].0
            ..segment.byte_array_blocks_pointer[block_id].0
                + segment.byte_array_blocks_pointer[block_id].1]
    };

    let mut bm25f = 0.0;

    let rank_position_pointer_range: u32 =
        compression_type_pointer & 0b0011_1111_1111_1111_1111_1111_1111_1111;

    let posting_pointer_size_sum;
    let rank_position_pointer;

    let (posting_pointer_size, embed_flag) = if max_p_docid < pointer_pivot_p_docid {
        posting_pointer_size_sum = max_p_docid as u32 * 2;
        rank_position_pointer = read_u16(
            byte_array,
            rank_position_pointer_range as usize + posting_pointer_size_sum as usize,
        ) as u32;
        (2, (rank_position_pointer & 0b10000000_00000000) != 0)
    } else {
        posting_pointer_size_sum = (max_p_docid as u32) * 3 - pointer_pivot_p_docid as u32;
        rank_position_pointer = read_u32(
            byte_array,
            rank_position_pointer_range as usize + posting_pointer_size_sum as usize,
        );
        (
            3,
            (rank_position_pointer & 0b10000000_00000000_00000000) != 0,
        )
    };

    let positions_pointer = if embed_flag {
        rank_position_pointer_range as usize + posting_pointer_size_sum as usize
    } else {
        let pointer_value = if posting_pointer_size == 2 {
            rank_position_pointer & 0b01111111_11111111
        } else {
            rank_position_pointer & 0b01111111_11111111_11111111
        } as usize;

        rank_position_pointer_range as usize - pointer_value
    };

    let mut field_vec: SmallVec<[(u16, usize); 2]> = SmallVec::new();
    let mut field_vec_ngram1 = SmallVec::new();
    let mut field_vec_ngram2 = SmallVec::new();
    let mut field_vec_ngram3 = SmallVec::new();

    decode_positions_commit(
        posting_pointer_size,
        embed_flag,
        byte_array,
        positions_pointer,
        ngram_type,
        index.indexed_field_vec.len(),
        index.indexed_field_id_bits,
        index.indexed_field_id_mask,
        index.longest_field_id as u16,
        &mut field_vec,
        &mut field_vec_ngram1,
        &mut field_vec_ngram2,
        &mut field_vec_ngram3,
    );

    if ngram_type == &NgramType::SingleTerm
        || index.meta.lexical_similarity == LexicalSimilarity::Bm25fProximity
    {
        let idf = (((index.indexed_doc_count as f32 - posting_count as f32 + 0.5)
            / (posting_count as f32 + 0.5))
            + 1.0)
            .ln();

        for field in field_vec.iter() {
            let document_length_normalized = DOCUMENT_LENGTH_COMPRESSION[{
                get_document_length_compressed_mmap(index, field.0 as usize, block_id, max_docid)
            } as usize] as f32;

            let document_length_quotient =
                document_length_normalized / index.document_length_normalized_average;

            let tf = field.1 as f32;

            let weight = index.indexed_schema_vec[field.0 as usize].boost;

            bm25f += weight
                * idf
                * ((tf * (K + 1.0) / (tf + (K * (1.0 - B + (B * document_length_quotient)))))
                    + SIGMA);
        }
    } else if ngram_type == &NgramType::NgramFF
        || ngram_type == &NgramType::NgramFR
        || ngram_type == &NgramType::NgramRF
    {
        let idf_ngram1 = (((index.indexed_doc_count as f32 - posting_count_ngram_1 as f32 + 0.5)
            / (posting_count_ngram_1 as f32 + 0.5))
            + 1.0)
            .ln();

        let idf_ngram2 = (((index.indexed_doc_count as f32 - posting_count_ngram_2 as f32 + 0.5)
            / (posting_count_ngram_2 as f32 + 0.5))
            + 1.0)
            .ln();

        for field in field_vec_ngram1.iter() {
            let document_length_normalized = DOCUMENT_LENGTH_COMPRESSION[{
                get_document_length_compressed_mmap(index, field.0 as usize, block_id, max_docid)
            } as usize] as f32;

            let document_length_quotient =
                document_length_normalized / index.document_length_normalized_average;

            let tf_ngram1 = field.1 as f32;

            let weight = index.indexed_schema_vec[field.0 as usize].boost;

            bm25f += weight
                * idf_ngram1
                * ((tf_ngram1 * (K + 1.0)
                    / (tf_ngram1 + (K * (1.0 - B + (B * document_length_quotient)))))
                    + SIGMA);
        }

        for field in field_vec_ngram2.iter() {
            let document_length_normalized = DOCUMENT_LENGTH_COMPRESSION[{
                get_document_length_compressed_mmap(index, field.0 as usize, block_id, max_docid)
            } as usize] as f32;

            let document_length_quotient =
                document_length_normalized / index.document_length_normalized_average;

            let tf_ngram2 = field.1 as f32;

            let weight = index.indexed_schema_vec[field.0 as usize].boost;

            bm25f += weight
                * idf_ngram2
                * ((tf_ngram2 * (K + 1.0)
                    / (tf_ngram2 + (K * (1.0 - B + (B * document_length_quotient)))))
                    + SIGMA);
        }
    } else {
        let idf_ngram1 = (((index.indexed_doc_count as f32 - posting_count_ngram_1 as f32 + 0.5)
            / (posting_count_ngram_1 as f32 + 0.5))
            + 1.0)
            .ln();

        let idf_ngram2 = (((index.indexed_doc_count as f32 - posting_count_ngram_2 as f32 + 0.5)
            / (posting_count_ngram_2 as f32 + 0.5))
            + 1.0)
            .ln();

        let idf_ngram3 = (((index.indexed_doc_count as f32 - posting_count_ngram_3 as f32 + 0.5)
            / (posting_count_ngram_3 as f32 + 0.5))
            + 1.0)
            .ln();

        for field in field_vec_ngram1.iter() {
            let document_length_normalized = DOCUMENT_LENGTH_COMPRESSION[{
                get_document_length_compressed_mmap(index, field.0 as usize, block_id, max_docid)
            } as usize] as f32;

            let document_length_quotient =
                document_length_normalized / index.document_length_normalized_average;

            let tf_ngram1 = field.1 as f32;

            let weight = index.indexed_schema_vec[field.0 as usize].boost;

            bm25f += weight
                * idf_ngram1
                * ((tf_ngram1 * (K + 1.0)
                    / (tf_ngram1 + (K * (1.0 - B + (B * document_length_quotient)))))
                    + SIGMA);
        }

        for field in field_vec_ngram2.iter() {
            let document_length_normalized = DOCUMENT_LENGTH_COMPRESSION[{
                get_document_length_compressed_mmap(index, field.0 as usize, block_id, max_docid)
            } as usize] as f32;

            let document_length_quotient =
                document_length_normalized / index.document_length_normalized_average;

            let tf_ngram2 = field.1 as f32;

            let weight = index.indexed_schema_vec[field.0 as usize].boost;

            bm25f += weight
                * idf_ngram2
                * ((tf_ngram2 * (K + 1.0)
                    / (tf_ngram2 + (K * (1.0 - B + (B * document_length_quotient)))))
                    + SIGMA);
        }

        for field in field_vec_ngram3.iter() {
            let document_length_normalized = DOCUMENT_LENGTH_COMPRESSION[{
                get_document_length_compressed_mmap(index, field.0 as usize, block_id, max_docid)
            } as usize] as f32;

            let document_length_quotient =
                document_length_normalized / index.document_length_normalized_average;

            let tf_ngram3 = field.1 as f32;

            let weight = index.indexed_schema_vec[field.0 as usize].boost;

            bm25f += weight
                * idf_ngram3
                * ((tf_ngram3 * (K + 1.0)
                    / (tf_ngram3 + (K * (1.0 - B + (B * document_length_quotient)))))
                    + SIGMA);
        }
    }
    bm25f
}

/// Opens the index using memory-mapped files.
/// * `index_path` - index path.
/// * `mute` - prevent emitting status messages (e.g. when using pipes for data interprocess communication).
pub(crate) async fn open_shard(
    index_path: &Path,
    mute: bool,
    vector_type: Precision,
    vector_dimensions: usize,
) -> Result<ShardArc, String> {
    if !mute {
        log::info!("opening index ...");
    }

    let mut index_mmap_position = INDEX_HEADER_SIZE as usize;
    let mut docstore_mmap_position = 0;

    let vector_size = size_of::<VectorHeader>()
        + (vector_dimensions
            * match vector_type {
                Precision::F32 => 4,
                Precision::I8 => 1,
                Precision::None => 0,
            });

    match File::open(Path::new(index_path).join(META_FILENAME)) {
        Ok(meta_file) => {
            let meta: IndexMetaObject = serde_json::from_reader(BufReader::new(meta_file))
                .map_err(|e| format!("unsupported or invalid index metadata: {e}"))?;

            match File::open(Path::new(index_path).join(SCHEMA_FILENAME)) {
                Ok(schema_file) => {
                    let schema: Vec<SchemaField> =
                        serde_json::from_reader(BufReader::new(schema_file))
                            .map_err(|e| format!("invalid index schema: {e}"))?;

                    match create_shard(index_path, &meta, &schema, false, 11, mute, None)
                    {
                        Ok(mut shard) => {
                            let file_len = { shard.index_file_mmap.len() as u64 };

                            while (index_mmap_position as u64) < file_len {
                                let mut segment_head_vec: Vec<(u32, u32)> = Vec::new();
                                for key0 in 0..shard.segment_number1 {
                                    if key0 == 0 {
                                        shard.last_level_index_file_start_pos =
                                            index_mmap_position as u64;

                                        shard.last_level_docstore_file_start_pos =
                                            docstore_mmap_position as u64;

                                        if shard.level_index.is_empty() {
                                            let longest_field_id = {
                                                read_u16_ref(
                                                    &shard.index_file_mmap,
                                                    &mut index_mmap_position,
                                                )
                                                    as usize
                                            };

                                            for indexed_field in shard.indexed_field_vec.iter_mut()
                                            {
                                                indexed_field.is_longest_field = indexed_field
                                                    .indexed_field_id
                                                    == longest_field_id;

                                                if indexed_field.is_longest_field {
                                                    shard.longest_field_id = longest_field_id
                                                }
                                            }
                                        }

                                        let document_length_compressed_array_pointer =
                                            { index_mmap_position };

                                        for _i in 0..shard.indexed_field_vec.len() {
                                            {
                                                index_mmap_position += ROARING_BLOCK_SIZE;
                                            }
                                        }

                                        let mut docstore_pointer_docs_pointer = 0;
                                        if !shard.stored_field_names.is_empty() {
                                            {
                                                let docstore_pointer_docs_size = read_u32_ref(
                                                    &shard.docstore_file_mmap,
                                                    &mut docstore_mmap_position,
                                                )
                                                    as usize;
                                                docstore_pointer_docs_pointer =
                                                    docstore_mmap_position;
                                                docstore_mmap_position +=
                                                    docstore_pointer_docs_size;
                                            }
                                        }

                                        {
                                            let _previous_indexed_doc_count =
                                                shard.indexed_doc_count;
                                            shard.indexed_doc_count = read_u64_ref(
                                                &shard.index_file_mmap,
                                                &mut index_mmap_position,
                                            )
                                                as usize;
                                            shard.positions_sum_normalized = read_u64_ref(
                                                &shard.index_file_mmap,
                                                &mut index_mmap_position,
                                            );

                                            for _key0 in 0..shard.segment_number1 {
                                                let block_length = read_u32_ref(
                                                    &shard.index_file_mmap,
                                                    &mut index_mmap_position,
                                                );
                                                let key_count = read_u32_ref(
                                                    &shard.index_file_mmap,
                                                    &mut index_mmap_position,
                                                );

                                                segment_head_vec.push((block_length, key_count));
                                            }
                                        }

                                        shard.document_length_normalized_average =
                                            shard.positions_sum_normalized as f32
                                                / shard.indexed_doc_count as f32;

                                        shard.level_index.push(LevelIndex {
                                            docstore_pointer_docs_pointer,
                                            document_length_compressed_array_pointer,
                                        });
                                    }

                                    let block_length = segment_head_vec[key0].0;
                                    let key_count = segment_head_vec[key0].1;

                                    let key_body_pointer_write_start: u32 =
                                        key_count * shard.key_head_size as u32;

                                    {
                                        index_mmap_position +=
                                            key_count as usize * shard.key_head_size;
                                        shard.segments_index[key0].byte_array_blocks_pointer.push(
                                            (
                                                index_mmap_position,
                                                (block_length - key_body_pointer_write_start)
                                                    as usize,
                                                key_count,
                                            ),
                                        );

                                        index_mmap_position +=
                                            (block_length - key_body_pointer_write_start) as usize;
                                    }
                                }
                            }

                            shard.committed_doc_count = shard.indexed_doc_count;
                            shard.is_last_level_incomplete =
                                !shard.committed_doc_count.is_multiple_of(ROARING_BLOCK_SIZE);

                            if shard.is_vector_indexing && !shard.vector_file_mmap.is_empty() {
                                shard.indexed_vector_count = 0;

                                let mut offset = 0;
                                for _level_id in 0..shard.level_index.len() {
                                    shard.last_level_vector_file_start_pos = offset as u64;

                                    let cluster_number_bytes =
                                        &shard.vector_file_mmap[offset..offset + 4];
                                    let cluster_number = u32::from_le_bytes(
                                        cluster_number_bytes.try_into().unwrap(),
                                    )
                                        as usize;
                                    offset += 4;

                                    let mut level_vectors_count = 0;
                                    let mut start_index = 0;
                                    for _i in 0..cluster_number {
                                        let cluster_header_bytes =
                                            &shard.vector_file_mmap[offset..offset + 4];
                                        let cluster_header = ClusterHeader {
                                            start_index,
                                            child_count: u32::from_le_bytes(
                                                cluster_header_bytes.try_into().unwrap(),
                                            ),
                                        };
                                        offset += 4;
                                        start_index += cluster_header.child_count;
                                        level_vectors_count += cluster_header.child_count;
                                    }

                                    shard.indexed_vector_count += level_vectors_count as usize;
                                    shard.indexed_cluster_count += cluster_number;

                                    offset += level_vectors_count as usize * vector_size;
                                }
                            }

                            for (i, component) in shard.bm25_component_cache.iter_mut().enumerate()
                            {
                                let document_length_quotient = DOCUMENT_LENGTH_COMPRESSION[i]
                                    as f32
                                    / shard.document_length_normalized_average;
                                *component = K * (1.0 - B + B * document_length_quotient);
                            }

                            shard.string_set_to_single_term_id();

                            let mut reader = BufReader::with_capacity(8192, &shard.delete_file);
                            while let Ok(buffer) = reader.fill_buf() {
                                let length = buffer.len();

                                if length == 0 {
                                    break;
                                }

                                for i in (0..length).step_by(8) {
                                    let docid = read_u64(buffer, i);
                                    shard.delete_hashset.insert(docid as usize);
                                }

                                reader.consume(length);
                            }

                            let shard_arc = Arc::new(RwLock::new(shard));

                            warmup(&shard_arc).await;
                            Ok(shard_arc.clone())
                        }
                        Err(err) => Err(err.to_string()),
                    }
                }
                Err(err) => Err(err.to_string()),
            }
        }
        Err(err) => Err(err.to_string()),
    }
}

/// Opens the index using memory-mapped files.
/// * `index_path` - index path.
/// * `mute` - prevent emitting status messages (e.g. when using pipes for data interprocess communication).
pub async fn open_index(index_path: &Path) -> Result<IndexArc, String> {
    let start_time = Instant::now();

    match File::open(Path::new(index_path).join(META_FILENAME)) {
        Ok(meta_file) => {
            let meta: IndexMetaObject = serde_json::from_reader(BufReader::new(meta_file))
                .map_err(|e| format!("unsupported or invalid index metadata: {e}"))?;

            match File::open(Path::new(index_path).join(SCHEMA_FILENAME)) {
                Ok(schema_file) => {
                    let schema: Vec<SchemaField> =
                        serde_json::from_reader(BufReader::new(schema_file))
                            .map_err(|e| format!("invalid index schema: {e}"))?;

                    let shard_number = index_path
                        .join("shards")
                        .read_dir()
                        .unwrap()
                        .filter_map(Result::ok)
                        .filter(|entry| entry.path().is_dir())
                        .filter_map(|entry| entry.file_name().into_string().ok())
                        .filter(|name| name.parse::<usize>().is_ok())
                        .count();

                    match create_index_root(
                        index_path,
                        meta,
                        &schema,
                        false,
                        11,
                        false,
                        Some(shard_number),
                    )
                    .await
                    {
                        Ok(index_arc) => {
                            let lock = Arc::into_inner(index_arc).unwrap();
                            let index = RwLock::into_inner(lock);

                            let index_arc = Arc::new(RwLock::new(index));

                            let mut shard_vec: Vec<Arc<RwLock<Shard>>> = Vec::new();

                            let vector_type = match index_arc.read().await.quantization {
                                Quantization::ScalarQuantizationI8 => Precision::I8,
                                Quantization::TurboQuantI8 => Precision::I8,
                                _ => index_arc.read().await.vector_precision,
                            };

                            let dimensions = index_arc.read().await.vector_dimensions;

                            let paths: Vec<_> = fs::read_dir(index_path.join("shards"))
                                .unwrap()
                                .filter_map(Result::ok)
                                .collect();
                            let mut shard_handle_vec = Vec::new();
                            let index_path_clone = Arc::new(index_path.to_path_buf());
                            for i in 0..paths.len() {
                                let index_path_clone2 = index_path_clone.clone();
                                let vector_type_clone = vector_type;
                                let dimensions_clone = dimensions;
                                shard_handle_vec.push(tokio::spawn(async move {
                                    let path = index_path_clone2.join("shards").join(i.to_string());

                                    open_shard(&path, true, vector_type_clone, dimensions_clone)
                                        .await
                                }));
                            }

                            for shard_handle in shard_handle_vec {
                                let shard_arc = shard_handle
                                    .await
                                    .map_err(|e| format!("failed to open shard: {e}"))??;
                                shard_arc.write().await.index_option = Some(index_arc.clone());

                                shard_arc.write().await.quantization =
                                    index_arc.read().await.quantization;
                                shard_arc.write().await.shard_number =
                                    index_arc.read().await.shard_number;
                                shard_arc.write().await.vector_dimensions =
                                    index_arc.read().await.vector_dimensions;
                                shard_arc.write().await.vector_dimensions_original =
                                    index_arc.read().await.vector_dimensions_original;
                                shard_arc.write().await.vector_precision =
                                    index_arc.read().await.vector_precision;
                                shard_arc.write().await.vector_similarity =
                                    index_arc.read().await.vector_similarity;
                                shard_arc.write().await.is_avx2 = index_arc.read().await.is_avx2;
                                shard_arc.write().await.is_neon = index_arc.read().await.is_neon;
                                shard_arc.write().await.is_simd = index_arc.read().await.is_simd;

                                shard_arc.write().await.turbo_quant =
                                    index_arc.read().await.turbo_quant.clone();

                                if shard_arc.read().await.is_vector_indexing
                                    && !shard_arc.read().await.vector_file_mmap.is_empty()
                                    && shard_arc.read().await.quantization
                                        == Quantization::ScalarQuantizationI8
                                    && shard_arc.read().await.vector_similarity
                                        == VectorSimilarity::Euclidean
                                {
                                    let (min_vector_value, max_vector_value) = read_min_max(
                                        &shard_arc.read().await.vector_file_mmap,
                                        shard_arc.read().await.vector_dimensions,
                                    );
                                    shard_arc.write().await.min_vector_value = min_vector_value;
                                    shard_arc.write().await.max_vector_value = max_vector_value;
                                }

                                index_arc.write().await.indexed_doc_count +=
                                    shard_arc.read().await.indexed_doc_count;
                                index_arc.write().await.indexed_vector_count +=
                                    shard_arc.read().await.indexed_vector_count;
                                index_arc.write().await.indexed_cluster_count +=
                                    shard_arc.read().await.indexed_cluster_count;
                                index_arc.write().await.deleted_doc_count +=
                                    shard_arc.read().await.delete_hashset.len();
                                shard_vec.push(shard_arc);
                            }

                            let indexed_doc_count = index_arc.read().await.indexed_doc_count;
                            *index_arc.write().await.docid_global.write().await = indexed_doc_count;

                            index_arc.write().await.shard_number = shard_vec.len();

                            index_arc.write().await.shard_vec = shard_vec;

                            let _elapsed_time = start_time.elapsed().as_nanos();

                            Ok(index_arc.clone())
                        }
                        Err(err) => Err(err.to_string()),
                    }
                }
                Err(err) => Err(err.to_string()),
            }
        }
        Err(err) => Err(err.to_string()),
    }
}

pub(crate) async fn warmup(shard_object_arc: &ShardArc) {
    shard_object_arc.write().await.frequentword_results.clear();

    let frequent_words = shard_object_arc.read().await.frequent_words.clone();
    for frequentword in frequent_words.iter() {
        let results_list = shard_object_arc
            .search_lexical_shard(
                frequentword.to_owned(),
                QueryType::Union,
                false,
                0,
                1000,
                ResultType::TopkCount,
                false,
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
            .await;

        let mut index_mut = shard_object_arc.write().await;
        index_mut
            .frequentword_results
            .insert(frequentword.to_string(), results_list);
    }
}

#[derive(Default, Debug, Deserialize, Serialize, Clone)]
pub(crate) struct TermObject {
    pub key_hash: u64,
    pub key0: u32,
    pub term: String,

    pub ngram_type: NgramType,

    pub term_ngram_2: String,
    pub term_ngram_1: String,
    pub term_ngram_0: String,
    pub field_vec_ngram1: Vec<(usize, u32)>,
    pub field_vec_ngram2: Vec<(usize, u32)>,
    pub field_vec_ngram3: Vec<(usize, u32)>,

    pub field_positions_vec: Vec<Vec<u16>>,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub(crate) struct NonUniqueTermObject {
    pub term: String,
    pub ngram_type: NgramType,

    pub term_ngram_2: String,
    pub term_ngram_1: String,
    pub term_ngram_0: String,
    pub op: QueryType,
}

/// system endianess: true: the system is little endian, false: the system is big endian (network byte order).
pub static IS_SYSTEM_LE: LazyLock<bool> = LazyLock::new(|| u16::from_ne_bytes([1, 0]) == 1);

/// AVX2 support enabled
pub static IS_AVX2: LazyLock<bool> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    let is_avx2 = is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let is_avx2 = false;
    is_avx2
});

/// NEON support enabled (always `true` on aarch64 — Advanced SIMD is a
/// mandatory part of ARMv8). The flag exists for parity with `IS_AVX2`
/// so the rest of the crate can branch on it the same way.
pub static IS_NEON: LazyLock<bool> = LazyLock::new(|| {
    #[cfg(target_arch = "aarch64")]
    let is_neon = std::arch::is_aarch64_feature_detected!("neon");
    #[cfg(not(target_arch = "aarch64"))]
    let is_neon = false;
    is_neon
});

/// Any SIMD backend (AVX2 / NEON) is available. Used by call sites to gate
/// the SIMD path; on archs without a backend this is always `false`.
pub static IS_SIMD: LazyLock<bool> = LazyLock::new(|| *IS_AVX2 || *IS_NEON);

pub(crate) static HASHER_32: LazyLock<RandomState> =
    LazyLock::new(|| RandomState::with_seeds(805272099, 242851902, 646123436, 591410655));

pub(crate) static HASHER_64: LazyLock<RandomState> =
    LazyLock::new(|| RandomState::with_seeds(808259318, 750368348, 84901999, 789810389));

#[inline]
pub(crate) fn hash32(term_bytes: &[u8]) -> u32 {
    HASHER_32.hash_one(term_bytes) as u32
}

#[inline]
pub(crate) fn hash64(term_bytes: &[u8]) -> u64 {
    HASHER_64.hash_one(term_bytes)
        & 0b1111111111111111111111111111111111111111111111111111111111111000
}

static FREQUENT_EN: &str = include_str!("../assets/dictionaries/frequent_en.txt");
static FREQUENT_DE: &str = include_str!("../assets/dictionaries/frequent_de.txt");
static FREQUENT_FR: &str = include_str!("../assets/dictionaries/frequent_fr.txt");
static FREQUENT_ES: &str = include_str!("../assets/dictionaries/frequent_es.txt");

pub(crate) const NUM_FREE_VALUES: u32 = 24;

/// Compress an u32 to a byte, preserving 4 significant bits.
/// used for compressing n-gram frequent_term positions_count and doc/field length
/// Ported from Lucene SmallFloat.java https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/util/SmallFloat.java
pub(crate) fn int_to_byte4(i: u32) -> u8 {
    if i < NUM_FREE_VALUES {
        i as u8
    } else {
        let ii = i - NUM_FREE_VALUES;
        let num_bits = 32 - ii.leading_zeros();
        if num_bits < 4 {
            (NUM_FREE_VALUES + ii) as u8
        } else {
            let shift = num_bits - 4;
            (NUM_FREE_VALUES + (((ii >> shift) & 0x07) | (shift + 1) << 3)) as u8
        }
    }
}

/// Decompress a byte that has been compressed with intToByte4(int), to an u32
/// used for pre-calculating DOCUMENT_LENGTH_COMPRESSION table. Decompressing n-gram frequent_term positions_count and doc/field length via table lookup.
/// Ported from Lucene SmallFloat.java https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/util/SmallFloat.java
pub(crate) const fn byte4_to_int(b: u8) -> u32 {
    if (b as u32) < NUM_FREE_VALUES {
        b as u32
    } else {
        let i = b as u32 - NUM_FREE_VALUES;
        let bits = i & 0x07;
        let shift = i >> 3;
        if shift == 0 {
            NUM_FREE_VALUES + bits
        } else {
            NUM_FREE_VALUES + ((bits | 0x08) << (shift - 1))
        }
    }
}

/// Pre-calculated DOCUMENT_LENGTH_COMPRESSION table for fast lookup.
pub(crate) const DOCUMENT_LENGTH_COMPRESSION: [u32; 256] = {
    let mut k2 = [0; 256];
    let mut i = 0usize;
    while i < 256 {
        k2[i] = byte4_to_int(i as u8);
        i += 1;
    }
    k2
};

impl Shard {
    pub(crate) fn string_set_to_single_term_id(&mut self) {
        for (i, facet) in self.facets.iter().enumerate() {
            if facet.field_type == FieldType::StringSet16
                || facet.field_type == FieldType::StringSet32
            {
                for (idx, value) in facet.values.iter().enumerate() {
                    for term in value.1.iter() {
                        self.string_set_to_single_term_id_vec[i]
                            .entry(term.to_string())
                            .or_insert(AHashSet::from_iter(vec![idx as u32]))
                            .insert(idx as u32);
                    }
                }
            }
        }
    }

    /// Reset shard to empty, while maintaining schema
    async fn clear_shard(&mut self) {
        let semaphore = self.semaphore.clone();
        let permit = semaphore.acquire_owned().await.unwrap();

        let mut mmap_options = MmapOptions::new();
        let mmap: MmapMut = mmap_options.len(4).map_anon().unwrap();
        self.index_file_mmap = mmap
            .make_read_only()
            .expect("Unable to make Mmap read-only");

        let _ = self.index_file.rewind();
        if let Err(e) = self.index_file.set_len(0) {
            log::error!(
                "Unable to index_file.set_len in clear_index {} {} {:?}",
                self.index_path_string,
                self.indexed_doc_count,
                e
            )
        };

        if !self.compressed_docstore_segment_block_buffer.is_empty() {
            self.compressed_docstore_segment_block_buffer = vec![0; ROARING_BLOCK_SIZE * 4];
        };

        write_u16(
            INDEX_FORMAT_VERSION_MAJOR,
            &mut self.compressed_index_segment_block_buffer,
            0,
        );
        write_u16(
            INDEX_FORMAT_VERSION_MINOR,
            &mut self.compressed_index_segment_block_buffer,
            2,
        );

        let _ = self
            .index_file
            .write(&self.compressed_index_segment_block_buffer[0..INDEX_HEADER_SIZE as usize]);
        let _ = self.index_file.flush();

        self.index_file_mmap =
            unsafe { Mmap::map(&self.index_file).expect("Unable to create Mmap") };

        self.docstore_file_mmap = unsafe {
            MmapOptions::new()
                .len(0)
                .map(&self.docstore_file)
                .expect("Unable to create Mmap")
        };

        let _ = self.docstore_file.rewind();
        if let Err(e) = self.docstore_file.set_len(0) {
            log::error!("Unable to docstore_file.set_len in clear_index {:?}", e)
        };
        let _ = self.docstore_file.flush();

        let _ = self.delete_file.rewind();
        if let Err(e) = self.delete_file.set_len(0) {
            log::error!("Unable to delete_file.set_len in clear_index {:?}", e)
        };
        let _ = self.delete_file.flush();
        self.delete_hashset.clear();

        self.facets_file_mmap = unsafe {
            MmapOptions::new()
                .len(0)
                .map_mut(&self.facets_file)
                .expect("Unable to create Mmap")
        };
        let _ = self.facets_file.rewind();
        if let Err(e) = self
            .facets_file
            .set_len((self.facets_size_sum * ROARING_BLOCK_SIZE) as u64)
        {
            log::error!("Unable to facets_file.set_len in clear_index {:?}", e)
        };
        let _ = self.facets_file.flush();

        self.facets_file_mmap =
            unsafe { MmapMut::map_mut(&self.facets_file).expect("Unable to create Mmap") };
        let index_path = Path::new(&self.index_path_string);
        let _ = fs::remove_file(index_path.join(FACET_VALUES_FILENAME));
        for facet in self.facets.iter_mut() {
            facet.values.clear();
        }
        for members in self.string_set_to_single_term_id_vec.iter_mut() {
            members.clear();
        }

        if !self.stored_field_names.is_empty() {
            self.docstore_file_mmap =
                unsafe { Mmap::map(&self.docstore_file).expect("Unable to create Mmap") };
        }

        self.vector_file_mmap = unsafe {
            MmapOptions::new()
                .len(0)
                .map(&self.docstore_file)
                .expect("Unable to create Mmap")
        };
        let _ = self.vector_file.rewind();
        if let Err(e) = self.vector_file.set_len(0) {
            log::error!("Unable to vector_file.set_len in clear_index {:?}", e)
        };
        let _ = self.vector_file.flush();
        self.vector_file_mmap =
            unsafe { Mmap::map(&self.vector_file).expect("Unable to create Mmap") };
        self.indexed_vector_count = 0;
        self.indexed_cluster_count = 0;

        self.document_length_normalized_average = 0.0;
        self.indexed_doc_count = 0;
        self.committed_doc_count = 0;
        self.positions_sum_normalized = 0;

        self.level_index = Vec::new();

        for segment in self.segments_index.iter_mut() {
            segment.byte_array_blocks_pointer.clear();
        }

        for segment in self.segments_level0.iter_mut() {
            segment.segment.clear();
        }

        self.key_count_sum = 0;
        self.block_id = 0;
        self.strip_compressed_sum = 0;
        self.postings_buffer = Vec::new();
        self.postings_buffer_pointer = 0;
        self.docid_count = 0;
        self.size_compressed_docid_index = 0;
        self.size_compressed_positions_index = 0;
        self.position_count = 0;
        self.postinglist_count = 0;

        self.is_last_level_incomplete = false;

        drop(permit);
    }
}

impl Index {
    /// Current document count: indexed document count - deleted document count
    pub async fn current_doc_count(&self) -> usize {
        let mut current_doc_count = 0;
        for shard in self.shard_vec.iter() {
            let shard_ref = shard.read().await;
            current_doc_count += shard_ref
                .indexed_doc_count
                .saturating_sub(shard_ref.delete_hashset.len());
        }
        current_doc_count
    }

    /// are there uncommitted documents?
    pub async fn uncommitted_doc_count(&self) -> usize {
        let mut uncommitted_doc_count = 0;
        for shard in self.shard_vec.iter() {
            let shard_ref = shard.read().await;
            uncommitted_doc_count += shard_ref
                .indexed_doc_count
                .saturating_sub(shard_ref.committed_doc_count);
        }
        uncommitted_doc_count
    }

    /// Get number of indexed documents.
    pub async fn committed_doc_count(&self) -> usize {
        let mut committed_doc_count = 0;
        for shard in self.shard_vec.iter() {
            committed_doc_count += shard.read().await.committed_doc_count;
        }
        committed_doc_count
    }

    /// Get number of indexed documents.
    pub async fn indexed_doc_count(&self) -> usize {
        let mut indexed_doc_count = 0;
        for shard in self.shard_vec.iter() {
            indexed_doc_count += shard.read().await.indexed_doc_count;
        }
        indexed_doc_count
    }

    /// Get number of indexed vectors.
    pub async fn indexed_vector_count(&self) -> usize {
        let mut indexed_vector_count = 0;
        for shard in self.shard_vec.iter() {
            indexed_vector_count += shard.read().await.indexed_vector_count;
        }
        indexed_vector_count
    }

    /// Get number of indexed clusters.
    pub async fn indexed_cluster_count(&self) -> usize {
        let mut indexed_cluster_count = 0;
        for shard in self.shard_vec.iter() {
            indexed_cluster_count += shard.read().await.indexed_cluster_count;
        }
        indexed_cluster_count
    }

    /// Get number of index levels. One index level comprises 64K documents **per shard**.
    pub async fn level_count(&self) -> usize {
        let mut level_count = 0;
        for shard in self.shard_vec.iter() {
            level_count += shard.read().await.level_index.len();
        }
        level_count
    }

    /// Get number of index shards.
    pub async fn shard_count(&self) -> usize {
        self.shard_number
    }

    /// Get number of facets defined in the index schema.
    pub fn facets_count(&self) -> usize {
        self.facets.len()
    }

    /// Reset the index to empty, while maintaining the schema.
    pub async fn clear_index(&mut self) {
        let mut result_object_list = Vec::new();
        for shard in self.shard_vec.iter() {
            let shard_clone = shard.clone();
            result_object_list.push(tokio::spawn(async move {
                shard_clone.write().await.clear_shard().await;
            }));
        }
        future::join_all(result_object_list).await;

        *self.docid_global.write().await = 0;
        self.indexed_doc_count = 0;
        self.indexed_vector_count = 0;
        self.indexed_cluster_count = 0;
        self.deleted_doc_count = 0;
    }

    /// Delete index from disc and ram
    pub fn delete_index(&mut self) {
        let index_path = Path::new(&self.index_path_string);

        let _ = fs::remove_dir_all(index_path);
    }

}

/// Commit pending changes and flush index files
#[allow(async_fn_in_trait)]
pub trait Close {
    /// Commit pending changes and flush index files
    async fn close(&self);
}

/// Commit pending changes and flush index files
impl Close for IndexArc {
    /// Commit pending changes and flush index files
    async fn close(&self) {
        self.commit().await;

        let mut result_object_list = Vec::new();
        for shard in self.read().await.shard_vec.iter() {
            let shard_clone = shard.clone();
            result_object_list.push(tokio::spawn(async move {
                let mut mmap_options = MmapOptions::new();
                let mmap: MmapMut = mmap_options.len(4).map_anon().unwrap();
                shard_clone.write().await.index_file_mmap = mmap
                    .make_read_only()
                    .expect("Unable to make Mmap read-only");

                let mut mmap_options = MmapOptions::new();
                let mmap: MmapMut = mmap_options.len(4).map_anon().unwrap();
                shard_clone.write().await.docstore_file_mmap = mmap
                    .make_read_only()
                    .expect("Unable to make Mmap read-only");

                shard_clone.write().await.index_option = None;
            }));
        }
        future::join_all(result_object_list).await;
    }
}

/// Delete document from index by document id
/// ⚠️ Use search or get_iterator first to obtain a valid doc_id. Document IDs are not guaranteed to be continuous and gapless!
#[allow(async_fn_in_trait)]
pub trait DeleteDocument {
    /// Delete document from index by document id
    async fn delete_document(&self, docid: u64);
}

/// Delete document from index by document id
///
/// Arguments:
/// * `doc_id`: Document ID that specifies which document to delete from the index.
///   ⚠️ Use search or get_iterator first to obtain a valid doc_id. Document IDs are not guaranteed to be continuous and gapless!
///
/// Immediately effective, independent of commit.
/// Only already allocated docids are accepted; clear_index invalidates old docids and reuses the namespace.
/// Index space used by deleted documents is not reclaimed (until compaction is implemented), but result_count_total is updated.
/// By manually deleting the delete.bin file the deleted documents can be recovered (until compaction).
/// Deleted documents impact performance, especially but not limited to counting (Count, TopKCount). They also increase the size of the index (until compaction is implemented).
/// For minimal query latency delete index and reindexing documents is preferred over deleting documents (until compaction is implemented).
/// BM25 scores are not updated (until compaction is implemented), but the impact is minimal.
impl DeleteDocument for IndexArc {
    async fn delete_document(&self, docid: u64) {
        let docid_global = self.read().await.docid_global.clone();
        let allocated_docid_count = *docid_global.read().await as u64;
        if docid >= allocated_docid_count {
            return;
        }
        let (shard_number, shard) = {
            let index_ref = self.read().await;
            (
                index_ref.shard_number as u64,
                index_ref.shard_vec[(docid % index_ref.shard_number as u64) as usize].clone(),
            )
        };
        let docid_local = docid / shard_number;

        let mut shard_mut = shard.write().await;

        if shard_mut.delete_hashset.insert(docid_local as usize) {
            let mut buffer: [u8; 8] = [0; 8];
            write_u64(docid_local, &mut buffer, 0);
            let _ = shard_mut.delete_file.write(&buffer);
            let _ = shard_mut.delete_file.flush();
        }
    }
}

/// Delete documents from index by document id
#[allow(async_fn_in_trait)]
pub trait DeleteDocuments {
    /// Delete documents from index by document id
    async fn delete_documents(&self, docid_vec: Vec<u64>);
}

/// Delete documents from index by document id
/// Document ID can by obtained by search.
/// Immediately effective, independent of commit.
/// Only already allocated docids are accepted; clear_index invalidates old docids and reuses the namespace.
/// Index space used by deleted documents is not reclaimed (until compaction is implemented), but result_count_total is updated.
/// By manually deleting the delete.bin file the deleted documents can be recovered (until compaction).
/// Deleted documents impact performance, especially but not limited to counting (Count, TopKCount). They also increase the size of the index (until compaction is implemented).
/// For minimal query latency delete index and reindexing documents is preferred over deleting documents (until compaction is implemented).
/// BM25 scores are not updated (until compaction is implemented), but the impact is minimal.
impl DeleteDocuments for IndexArc {
    async fn delete_documents(&self, docid_vec: Vec<u64>) {
        for docid in docid_vec {
            self.delete_document(docid).await;
        }
    }
}

/// Delete documents from index by query
/// Delete and search have identical parameters.
/// It is recommended to test with search prior to delete to verify that only those documents are returned that you really want to delete.
#[allow(clippy::too_many_arguments)]
#[allow(async_fn_in_trait)]
pub trait DeleteDocumentsByQuery {
    /// Delete documents from index by query
    /// Delete and search have identical parameters.
    /// It is recommended to test with search prior to delete to verify that only those documents are returned that you really want to delete.
    async fn delete_documents_by_query(
        &self,
        query_string: String,
        query_type_default: QueryType,
        offset: usize,
        length: usize,
        include_uncommitted: bool,
        field_filter: Vec<String>,
        facet_filter: Vec<FacetFilter>,
        result_sort: Vec<ResultSort>,
    );
}

/// Delete documents from index by query
/// Delete and search have identical parameters.
/// It is recommended to test with search prior to delete to verify that only those documents are returned that you really want to delete.
impl DeleteDocumentsByQuery for IndexArc {
    async fn delete_documents_by_query(
        &self,
        query_string: String,
        query_type_default: QueryType,
        offset: usize,
        length: usize,
        include_uncommitted: bool,
        field_filter: Vec<String>,
        facet_filter: Vec<FacetFilter>,
        result_sort: Vec<ResultSort>,
    ) {
        let rlo = self
            .search(
                query_string.to_owned(),
                None,
                query_type_default,
                SearchMode::Lexical,
                false,
                offset,
                length,
                ResultType::Topk,
                include_uncommitted,
                field_filter,
                facet_filter,
                result_sort,
            )
            .await;

        let document_id_vec: Vec<u64> = rlo
            .results
            .iter()
            .map(|result| result.doc_id as u64)
            .collect();
        self.delete_documents(document_id_vec).await;
    }
}

/// Update document in index
/// Update_document is a combination of delete_document and index_document.
/// All current limitations of delete_document apply.
#[allow(async_fn_in_trait)]
pub trait UpdateDocument {
    /// Update document in index
    /// Update_document is a combination of delete_document and index_document.
    /// All current limitations of delete_document apply.
    async fn update_document(&self, id_document: (u64, Document)) -> Result<(), String>;
}

/// Update document in index
/// Update_document is a combination of delete_document and index_document.
/// All current limitations of delete_document apply.
impl UpdateDocument for IndexArc {
    async fn update_document(&self, id_document: (u64, Document)) -> Result<(), String> {
        crate::vector::validate_document_vectors(&*self.read().await, &id_document.1)?;
        self.delete_document(id_document.0).await;
        enqueue_document(self, id_document.1).await;
        Ok(())
    }
}

/// Update documents in index
/// Update_document is a combination of delete_document and index_document.
/// All current limitations of delete_document apply.
#[allow(async_fn_in_trait)]
pub trait UpdateDocuments {
    /// Update documents in index
    /// Update_document is a combination of delete_document and index_document.
    /// All current limitations of delete_document apply.
    async fn update_documents(&self, id_document_vec: Vec<(u64, Document)>) -> Result<(), String>;
}

/// Update documents in index
/// Update_document is a combination of delete_document and index_document.
/// All current limitations of delete_document apply.
impl UpdateDocuments for IndexArc {
    async fn update_documents(&self, id_document_vec: Vec<(u64, Document)>) -> Result<(), String> {
        let (docid_vec, document_vec): (Vec<_>, Vec<_>) = id_document_vec.into_iter().unzip();
        {
            let index = self.read().await;
            for document in &document_vec {
                crate::vector::validate_document_vectors(&index, document)?;
            }
        }
        self.delete_documents(docid_vec).await;
        for document in document_vec {
            enqueue_document(self, document).await;
        }
        Ok(())
    }
}

/// Indexes a list of documents
#[allow(async_fn_in_trait)]
pub trait IndexDocuments {
    /// Indexes a list of documents
    /// May block, if the threshold of documents indexed in parallel is exceeded.
    async fn index_documents(&self, document_vec: Vec<Document>) -> Result<(), String>;
}

impl IndexDocuments for IndexArc {
    /// Index list of documents (bulk)
    /// May block, if the threshold of documents indexed in parallel is exceeded.
    async fn index_documents(&self, document_vec: Vec<Document>) -> Result<(), String> {
        {
            let index = self.read().await;
            for document in &document_vec {
                crate::vector::validate_document_vectors(&index, document)?;
            }
        }
        for document in document_vec {
            enqueue_document(self, document).await;
        }
        Ok(())
    }
}

/// Indexes a single document
#[allow(async_fn_in_trait)]
pub trait IndexDocument {
    /// Indexes a single document
    /// May block, if the threshold of documents indexed in parallel is exceeded.
    async fn index_document(&self, document: Document) -> Result<(), String>;
}

impl IndexDocument for IndexArc {
    /// Index document
    /// May block, if the threshold of documents indexed in parallel is exceeded.
    async fn index_document(&self, document: Document) -> Result<(), String> {
        crate::vector::validate_document_vectors(&*self.read().await, &document)?;
        enqueue_document(self, document).await;
        Ok(())
    }
}

async fn enqueue_document(index_arc: &IndexArc, document: Document) {
    let shard_number = index_arc.read().await.shard_number;
    let docid_global_arc = index_arc.read().await.docid_global.clone();
    let mut docid_global = docid_global_arc.write().await;
    let docid_global_clone = *docid_global;
    let shard_id = *docid_global % shard_number;

    let shard_arc = index_arc.read().await.shard_vec[shard_id].clone();
    let semaphore = shard_arc.read().await.semaphore.clone();
    let permit = semaphore.acquire_owned().await.unwrap();

    *docid_global += 1;
    drop(docid_global);

    INDEX_RUNTIME.handle().spawn(async move {
        shard_arc
            .index_document_shard(document, docid_global_clone)
            .await;
        drop(permit);
    });
}
/// Indexes a single document
#[allow(async_fn_in_trait)]
pub(crate) trait IndexDocumentShard {
    /// Indexes a single document
    /// May block, if the threshold of documents indexed in parallel is exceeded.
    async fn index_document_shard(&self, document: Document, docid_global: usize);
}

/// Recursively extract strings from a serde_json::Value and append them to a vector of Strings.
/// Only value, not field names.
pub(crate) fn object_values_to_string_vec_recursive(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::String(s) => out.push(s.clone()),
        Value::Array(arr) => {
            for v in arr {
                object_values_to_string_vec_recursive(v, out);
            }
        }
        Value::Object(map) => {
            for v in map.values() {
                object_values_to_string_vec_recursive(v, out);
            }
        }
        _ => {}
    }
}

impl IndexDocumentShard for ShardArc {
    /// Index document
    /// May block, if the threshold of documents indexed in parallel is exceeded.
    async fn index_document_shard(&self, document: Document, docid_global: usize) {
        let shard_arc_clone = self.clone();
        let shard_ref = self.read().await;
        let schema = shard_ref.indexed_schema_vec.clone();
        let ngram_indexing = shard_ref.meta.ngram_indexing;
        let indexed_field_vec_len = shard_ref.indexed_field_vec.len();
        let tokenizer_type = shard_ref.meta.tokenizer;
        let segment_number_mask1 = shard_ref.segment_number_mask1;

        drop(shard_ref);

        let token_per_field_max: u32 = u16::MAX as u32;
        let mut unique_terms: AHashMap<String, TermObject> = AHashMap::new();
        let mut field_vec: Vec<(usize, u8, u32, u32)> = Vec::new();

        let shard_ref2 = shard_arc_clone.read().await;

        for schema_field in schema.iter() {
            if !schema_field.index_lexical {
                continue;
            }

            if let Some(field_value) = document.get(&schema_field.field) {
                let mut non_unique_terms: Vec<NonUniqueTermObject> = Vec::new();
                let mut nonunique_terms_count = 0u32;

                let text = match schema_field.field_type {
                    FieldType::Json => {
                        if matches!(field_value, Value::Object { .. }) {
                            let mut strings_vec: Vec<String> = Vec::new();
                            object_values_to_string_vec_recursive(field_value, &mut strings_vec);
                            strings_vec.join(" ")
                        } else {
                            serde_json::from_value::<String>(field_value.clone())
                                .unwrap_or(field_value.to_string())
                        }
                    }
                    FieldType::Text | FieldType::String16 | FieldType::String32 => {
                        serde_json::from_value::<String>(field_value.clone())
                            .unwrap_or(field_value.to_string())
                    }

                    _ => field_value.to_string(),
                };

                let mut query_type_mut = QueryType::Union;

                tokenizer(
                    &shard_ref2,
                    &text,
                    &mut unique_terms,
                    &mut non_unique_terms,
                    tokenizer_type,
                    segment_number_mask1,
                    &mut nonunique_terms_count,
                    token_per_field_max,
                    MAX_POSITIONS_PER_TERM,
                    false,
                    &mut query_type_mut,
                    ngram_indexing,
                    schema_field.indexed_field_id,
                    indexed_field_vec_len,
                )
                .await;

                let document_length_compressed: u8 = int_to_byte4(nonunique_terms_count);
                let document_length_normalized: u32 =
                    DOCUMENT_LENGTH_COMPRESSION[document_length_compressed as usize];
                field_vec.push((
                    schema_field.indexed_field_id,
                    document_length_compressed,
                    document_length_normalized,
                    nonunique_terms_count,
                ));
            }
        }
        drop(shard_ref2);

        let ngrams: Vec<String> = unique_terms
            .iter()
            .filter(|term| term.1.ngram_type != NgramType::SingleTerm)
            .map(|term| term.1.term.clone())
            .collect();

        for term in ngrams.iter() {
            let ngram = unique_terms.get(term).unwrap();

            match ngram.ngram_type {
                NgramType::SingleTerm => {}
                NgramType::NgramFF | NgramType::NgramFR | NgramType::NgramRF => {
                    let term_ngram1 = ngram.term_ngram_1.clone();
                    let term_ngram2 = ngram.term_ngram_0.clone();

                    for indexed_field_id in 0..indexed_field_vec_len {
                        let positions_count_ngram1 =
                            unique_terms[&term_ngram1].field_positions_vec[indexed_field_id].len();
                        let positions_count_ngram2 =
                            unique_terms[&term_ngram2].field_positions_vec[indexed_field_id].len();
                        let ngram = unique_terms.get_mut(term).unwrap();

                        if positions_count_ngram1 > 0 {
                            ngram
                                .field_vec_ngram1
                                .push((indexed_field_id, positions_count_ngram1 as u32));
                        }
                        if positions_count_ngram2 > 0 {
                            ngram
                                .field_vec_ngram2
                                .push((indexed_field_id, positions_count_ngram2 as u32));
                        }
                    }
                }
                _ => {
                    let term_ngram1 = ngram.term_ngram_2.clone();
                    let term_ngram2 = ngram.term_ngram_1.clone();
                    let term_ngram3 = ngram.term_ngram_0.clone();

                    for indexed_field_id in 0..indexed_field_vec_len {
                        let positions_count_ngram1 =
                            unique_terms[&term_ngram1].field_positions_vec[indexed_field_id].len();
                        let positions_count_ngram2 =
                            unique_terms[&term_ngram2].field_positions_vec[indexed_field_id].len();
                        let positions_count_ngram3 =
                            unique_terms[&term_ngram3].field_positions_vec[indexed_field_id].len();
                        let ngram = unique_terms.get_mut(term).unwrap();

                        if positions_count_ngram1 > 0 {
                            ngram
                                .field_vec_ngram1
                                .push((indexed_field_id, positions_count_ngram1 as u32));
                        }
                        if positions_count_ngram2 > 0 {
                            ngram
                                .field_vec_ngram2
                                .push((indexed_field_id, positions_count_ngram2 as u32));
                        }
                        if positions_count_ngram3 > 0 {
                            ngram
                                .field_vec_ngram3
                                .push((indexed_field_id, positions_count_ngram3 as u32));
                        }
                    }
                }
            }
        }

        let document_item = DocumentItem {
            document,
            unique_terms,
            field_vec,
        };

        shard_arc_clone
            .index_document_shard_2(document_item, docid_global)
            .await;
    }
}

#[allow(async_fn_in_trait)]
pub(crate) trait IndexDocumentShard2 {
    async fn index_document_shard_2(&self, document_item: DocumentItem, docid_global: usize);
}

impl IndexDocumentShard2 for ShardArc {
    async fn index_document_shard_2(&self, document_item: DocumentItem, docid_global: usize) {
        let mut shard_mut = self.write().await;

        let docid_local = docid_global / shard_mut.shard_number;

        shard_mut.indexed_doc_count = docid_local + 1;

        let do_commit = shard_mut.block_id != docid_local >> 16;
        if do_commit {
            if shard_mut.is_vector_indexing {
                shard_mut.commit_vector_shard().await;
            }
            shard_mut.commit_lexical_shard(docid_local).await;

            shard_mut.block_id = docid_local >> 16;
        }

        if shard_mut.is_vector_indexing {
            shard_mut
                .index_vector_shard(docid_local, &document_item.document)
                .await;
        }

        if !shard_mut.facets.is_empty() {
            let facets_size_sum = shard_mut.facets_size_sum;
            for i in 0..shard_mut.facets.len() {
                let facet = &mut shard_mut.facets[i];
                if let Some(field_value) = document_item.document.get(&facet.name) {
                    let address = (facets_size_sum * docid_local) + facet.offset;

                    match facet.field_type {
                        FieldType::U8 => {
                            let value = field_value.as_u64().unwrap_or_default() as u8;

                            shard_mut.facets_file_mmap[address] = value
                        }
                        FieldType::U16 => {
                            let value = field_value.as_u64().unwrap_or_default() as u16;

                            write_u16(value, &mut shard_mut.facets_file_mmap, address)
                        }
                        FieldType::U32 => {
                            let value = field_value.as_u64().unwrap_or_default() as u32;

                            write_u32(value, &mut shard_mut.facets_file_mmap, address)
                        }
                        FieldType::U64 => {
                            let value = field_value.as_u64().unwrap_or_default();

                            write_u64(value, &mut shard_mut.facets_file_mmap, address)
                        }
                        FieldType::I8 => {
                            let value = field_value.as_i64().unwrap_or_default() as i8;

                            write_i8(value, &mut shard_mut.facets_file_mmap, address)
                        }
                        FieldType::I16 => {
                            let value = field_value.as_i64().unwrap_or_default() as i16;

                            write_i16(value, &mut shard_mut.facets_file_mmap, address)
                        }
                        FieldType::I32 => {
                            let value = field_value.as_i64().unwrap_or_default() as i32;

                            write_i32(value, &mut shard_mut.facets_file_mmap, address)
                        }
                        FieldType::I64 => {
                            let value = field_value.as_i64().unwrap_or_default();

                            write_i64(value, &mut shard_mut.facets_file_mmap, address)
                        }
                        FieldType::Timestamp => {
                            let value = field_value.as_i64().unwrap_or_default();

                            write_i64(value, &mut shard_mut.facets_file_mmap, address);
                        }
                        FieldType::F32 => {
                            let value = field_value.as_f64().unwrap_or_default() as f32;

                            write_f32(value, &mut shard_mut.facets_file_mmap, address)
                        }
                        FieldType::F64 => {
                            let value = field_value.as_f64().unwrap_or_default();

                            write_f64(value, &mut shard_mut.facets_file_mmap, address)
                        }
                        FieldType::String16 if facet.values.len() < u16::MAX as usize => {
                            let key = serde_json::from_value::<String>(field_value.clone())
                                .unwrap_or(field_value.to_string());

                            let key_string = key.clone();
                            let key = vec![key];

                            facet.values.entry(key_string.clone()).or_insert(key);

                            let facet_value_id =
                                facet.values.get_index_of(&key_string).unwrap() as u16;
                            write_u16(facet_value_id, &mut shard_mut.facets_file_mmap, address)
                        }

                        FieldType::StringSet16 if facet.values.len() < u16::MAX as usize => {
                            let mut key: Vec<String> =
                                serde_json::from_value(field_value.clone()).unwrap();
                            key.sort();

                            let key_string = key.join("_");
                            let is_new = !facet.values.contains_key(&key_string);
                            let terms = key.clone();
                            facet.values.entry(key_string.clone()).or_insert(key);

                            let facet_value_id =
                                facet.values.get_index_of(&key_string).unwrap() as u16;
                            if is_new {
                                for term in terms {
                                    shard_mut.string_set_to_single_term_id_vec[i]
                                        .entry(term)
                                        .or_default()
                                        .insert(facet_value_id as u32);
                                }
                            }
                            write_u16(facet_value_id, &mut shard_mut.facets_file_mmap, address)
                        }

                        FieldType::String32 if facet.values.len() < u32::MAX as usize => {
                            let key = serde_json::from_value::<String>(field_value.clone())
                                .unwrap_or(field_value.to_string());

                            let key_string = key.clone();
                            let key = vec![key];

                            facet.values.entry(key_string.clone()).or_insert(key);

                            let facet_value_id =
                                facet.values.get_index_of(&key_string).unwrap() as u32;
                            write_u32(facet_value_id, &mut shard_mut.facets_file_mmap, address)
                        }

                        FieldType::StringSet32 if facet.values.len() < u32::MAX as usize => {
                            let mut key: Vec<String> =
                                serde_json::from_value(field_value.clone()).unwrap();
                            key.sort();

                            let key_string = key.join("_");
                            let is_new = !facet.values.contains_key(&key_string);
                            let terms = key.clone();
                            facet.values.entry(key_string.clone()).or_insert(key);

                            let facet_value_id =
                                facet.values.get_index_of(&key_string).unwrap() as u32;
                            if is_new {
                                for term in terms {
                                    shard_mut.string_set_to_single_term_id_vec[i]
                                        .entry(term)
                                        .or_default()
                                        .insert(facet_value_id);
                                }
                            }
                            write_u32(facet_value_id, &mut shard_mut.facets_file_mmap, address)
                        }

                        _ => {}
                    };
                }
            }
        }

        shard_mut.uncommitted = true;

        let mut longest_field_id: usize = 0;
        let mut longest_field_length: u32 = 0;
        for value in document_item.field_vec {
            if docid_local == 0 && value.3 > longest_field_length {
                longest_field_id = value.0;
                longest_field_length = value.3;
            }

            shard_mut.document_length_compressed_array[value.0]
                [docid_local & 0b11111111_11111111] = value.1;
            shard_mut.positions_sum_normalized += value.2 as u64;
            shard_mut.indexed_field_vec[value.0].field_length_sum += value.2 as usize;
        }

        if docid_local == 0 && shard_mut.is_lexical_indexing {
            if !shard_mut.longest_field_auto {
                longest_field_id = shard_mut.longest_field_id;
            }
            shard_mut.longest_field_id = longest_field_id;
            shard_mut.indexed_field_vec[longest_field_id].is_longest_field = true;
            if shard_mut.longest_field_auto && shard_mut.indexed_field_vec.len() > 1 {
                log::debug!(
                    "detect longest field id {} name {} length {}",
                    longest_field_id,
                    shard_mut.indexed_field_vec[longest_field_id].schema_field_name,
                    longest_field_length
                );
            }
        }

        let unique_terms = document_item.unique_terms;

        for term in unique_terms {
            shard_mut.index_posting(term.1, docid_local, false, 0, 0, 0);
        }

        if !shard_mut.stored_field_names.is_empty() {
            shard_mut.store_document(docid_local, document_item.document);
        }

        if do_commit {
            drop(shard_mut);
            warmup(self).await;
        }
    }
}

pub(crate) struct DocumentItem {
    pub document: Document,
    pub unique_terms: AHashMap<String, TermObject>,
    pub field_vec: Vec<(usize, u8, u32, u32)>,
}
