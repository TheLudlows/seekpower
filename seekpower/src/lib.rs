#![crate_type = "lib"]
#![crate_name = "seekpower"]

//! # Seekpower core
//! Lexical, external-vector and hybrid search.
//! Use a path or git dependency on this repository. Models and document parsing run externally.
//! Metadata fields support filtering and sorting; aggregate counts, histograms and extrema
//! are no longer exposed. See the repository's docs/library-examples.md for current examples.

#![warn(
    missing_docs,
    unused_import_braces,
    trivial_casts,
    trivial_numeric_casts,
    unused_qualifications
)]

/// Include the retained library examples in documentation.
#[cfg_attr(doctest, doc = include_str!("../../docs/library-examples.md"))]
pub struct ReadmeDoctests;

/// include FACETED_SEARCH.md in documentation
#[cfg_attr(doctest, doc = include_str!("../../FACETED_SEARCH.md"))]
pub struct ReadmeDoctests2;

use std::sync::LazyLock;

use tokio::runtime::Runtime;
/// Global Tokio runtime for index operations, initialized lazily on first use.
pub static INDEX_RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .thread_name("seekpower-indexer")
        .enable_all()
        .build()
        .unwrap()
});

pub(crate) mod add_result;
pub(crate) mod clustering;
/// Commit moves indexed documents from the intermediate uncompressed data structure in RAM
/// to the final compressed data structure on disk.
pub mod commit;
pub(crate) mod compatible;
pub(crate) mod compress_postinglist;
pub(crate) mod doc_store;
/// Operate the index: reate_index, open_index, clear_index, close_index, delete_index, index_document(s)
pub mod index;
pub(crate) mod index_posting;
/// Collect index information independently of terminal rendering.
pub mod info;
pub(crate) mod intersection;
pub(crate) mod intersection_simd;
/// Iterator over all documents, also for search with empty query.
pub mod iterator;
pub(crate) mod min_heap;
pub(crate) mod realtime_search;
/// Search the index for all indexed documents, both for committed and uncommitted documents.
/// The latter enables true realtime search: documents are available for search in exact the same millisecond they are indexed.
pub mod search;
pub(crate) mod single;
/// Tokenizes text into tokens (words), supports Chinese word segmentation, folds (converts) diacritics, accents, zalgo text, umlaut, bold, italic, full-width UTF-8 characters into their basic representation.
pub(crate) mod tokenizer;
pub(crate) mod union;
/// Utils `truncate()` and `substring()`
pub mod utils;
/// Vector search by indexing vectors and searching for similar vectors based on cosine similarity, inner product, and Euclidean distance.
pub mod vector;
/// Vector quantization and similarity measure definitions for vector search.
pub mod vector_similarity;


mod metadata_filter;
