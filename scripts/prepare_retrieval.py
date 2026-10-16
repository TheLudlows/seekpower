#!/usr/bin/env python3
"""Prepare SciFact or MIRACL-zh with pinned data and multilingual E5 embeddings."""
import argparse
import hashlib
import gzip
import heapq
import json
import os
from pathlib import Path
import time
import urllib.request
import zipfile

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import onnxruntime as ort
import tokenizers

MODEL = 'intfloat/multilingual-e5-small'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
CORPUS_URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
MIRACL_CORPUS = 'd921ec7e349ce0d28daf30b2da9da5ee698bef0d'
MIRACL_LABELS = '5be20db9509754dadad47689368639fcec739c00'


def read_jsonl(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def prepare_miracl(args):
    corpus_base = f'https://huggingface.co/datasets/miracl/miracl-corpus/resolve/{MIRACL_CORPUS}/miracl-corpus-v1.0-zh/'
    labels_base = f'https://huggingface.co/datasets/miracl/miracl/resolve/{MIRACL_LABELS}/miracl-v1.0-zh/'
    for kind in ('topics', 'qrels'):
        name = f'{kind}.miracl-v1.0-zh-dev.tsv'
        download(labels_base + kind + '/' + name, args.cache / name)
    queries = [line.split('\t', 1) for line in (args.cache / 'topics.miracl-v1.0-zh-dev.tsv').read_text().splitlines()]
    judgments = [line.split() for line in (args.cache / 'qrels.miracl-v1.0-zh-dev.tsv').read_text().splitlines()]
    judged = {row[2] for row in judgments}
    if len(queries) != 393 or len(judgments) != 3928:
        raise ValueError('unexpected MIRACL-zh dev query/judgment count')
    if args.documents and args.documents < len(judged):
        raise ValueError(f'subset must retain all {len(judged)} judged documents')
    (args.output / 'queries.jsonl').write_text(''.join(json.dumps({'_id': q, 'text': text}, ensure_ascii=False) + '\n' for q, text in queries))
    (args.output / 'qrels').mkdir(exist_ok=True)
    (args.output / 'qrels/dev.tsv').write_text('query-id\tcorpus-id\tscore\n' + ''.join(f'{q}\t{doc}\t{score}\n' for q, _, doc, score in judgments))
    listing = args.cache / 'corpus-files.json'
    download(f'https://huggingface.co/api/datasets/miracl/miracl-corpus/tree/{MIRACL_CORPUS}/miracl-corpus-v1.0-zh', listing)
    files = json.loads(listing.read_text())
    selected, negatives, seen = {}, [], set()
    count, background = 0, max(0, args.documents - len(judged))
    hashes = {}
    with (args.output / 'corpus.jsonl').open('w') as output:
        for source in sorted(files, key=lambda f: f['path']):
            name = Path(source['path']).name
            path = args.cache / name
            download(corpus_base + name, path)
            hashes[name] = sha256(path)
            if hashes[name] != source['lfs']['oid']:
                raise ValueError(f'official corpus checksum mismatch: {name}')
            with gzip.open(path, 'rt') as stream:
                for line in stream:
                    doc = json.loads(line)
                    doc_id = str(doc['docid'])
                    value = {'_id': doc_id, 'title': doc['title'], 'text': doc['text']}
                    count += 1
                    if doc_id in judged:
                        if doc_id in seen:
                            raise ValueError(f'duplicate judged document: {doc_id}')
                        seen.add(doc_id)
                    if not args.documents:
                        output.write(json.dumps(value, ensure_ascii=False) + '\n')
                    elif doc_id in judged:
                        selected[doc_id] = value
                    elif background:
                        # Bottom-k hashes sample across every shard; never just take the first N documents.
                        key = int.from_bytes(hashlib.sha256(f'20261016:{doc_id}'.encode()).digest(), 'big')
                        if len(negatives) < background:
                            heapq.heappush(negatives, (-key, doc_id, value))
                        elif key < -negatives[0][0]:
                            heapq.heapreplace(negatives, (-key, doc_id, value))
            print(f'Corpus scanned: {count:,}', flush=True)
        if count != 4934368 or seen != judged:
            raise ValueError('incomplete official corpus or missing judged documents')
        if args.documents:
            selected.update((doc_id, doc) for _, doc_id, doc in negatives)
            for doc_id in sorted(selected):
                output.write(json.dumps(selected[doc_id], ensure_ascii=False) + '\n')
    return {'name': 'MIRACL/zh', 'split': 'dev', 'qrels_file': 'qrels/dev.tsv',
            'documents': args.documents or count, 'queries': len(queries), 'source_documents': count,
            'full_corpus': not args.documents, 'tokenizer': 'UnicodeAlphanumericZH',
            'selection': 'full corpus' if not args.documents else 'all dev judged documents (positive AND negative) + lowest SHA256(20261016:docid) background; not an official full-corpus score',
            'judged_documents': len(judged), 'judgments': len(judgments),
            'corpus_revision': MIRACL_CORPUS, 'labels_revision': MIRACL_LABELS,
            'source_sha256': {**hashes, **{name: sha256(args.cache / name) for name in
                                          ('topics.miracl-v1.0-zh-dev.tsv', 'qrels.miracl-v1.0-zh-dev.tsv')}}}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(url, path):
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.part')
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        temporary.rename(path)


def prepare_scifact(args):
    download(CORPUS_URL, args.cache / 'scifact.zip')
    with (args.cache / 'scifact.zip').open('rb') as source:
        if hashlib.file_digest(source, 'md5').hexdigest() != '5f7d1de60b170fc8027bb7898e2efca1':
            raise ValueError('SciFact archive does not match the BEIR published checksum')
    with zipfile.ZipFile(args.cache / 'scifact.zip') as archive:
        for name in ('corpus.jsonl', 'queries.jsonl', 'qrels/test.tsv'):
            path = args.output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read('scifact/' + name))
    ids = {line.split('\t')[0] for line in (args.output / 'qrels/test.tsv').read_text().splitlines()[1:]}
    queries = [q for q in read_jsonl(args.output / 'queries.jsonl') if str(q['_id']) in ids]
    (args.output / 'queries.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in queries))
    return {'name': 'BEIR/SciFact', 'split': 'test', 'qrels_file': 'qrels/test.tsv',
            'full_corpus': True, 'documents': 5183, 'queries': len(queries), 'corpus_url': CORPUS_URL}


def embed(args, metadata):
    for name in ('tokenizer.json', 'onnx/model.onnx'):
        download(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}', args.cache / Path(name).name)
    tokenizer = tokenizers.Tokenizer.from_file(str(args.cache / 'tokenizer.json'))
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(pad_id=1, pad_token='<pad>')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.cache / 'model.onnx'), options, providers=['CPUExecutionProvider'])
    inputs = {value.name for value in session.get_inputs()}
    config = {'model': MODEL, 'revision': REVISION, 'runtime': 'ONNX Runtime CPU FP32',
              'onnxruntime': ort.__version__, 'tokenizers': tokenizers.__version__, 'numpy': np.__version__,
              'max_tokens': 512, 'pooling': 'attention-mask mean, then L2 normalization',
              'document_input': 'passage: {title}\n{text}', 'query_input': 'query: {text}',
              'threads': 8, 'batch_size': 8, 'model_sha256': sha256(args.cache / 'model.onnx'),
              'tokenizer_sha256': sha256(args.cache / 'tokenizer.json')}
    seconds = 0
    for source, target, total, prefix in [('corpus.jsonl', 'vectors.npy', metadata['documents'], 'passage: '),
                                          ('queries.jsonl', 'query_vectors.npy', metadata['queries'], 'query: ')]:
        signature = {**config, 'input_sha256': sha256(args.output / source), 'rows': total}
        progress_path = args.output / (target + '.progress.json')
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'signature': signature, 'rows': 0, 'seconds': 0}
        if progress['signature'] != signature:
            raise ValueError('embedding checkpoint belongs to different input/model; use another output directory')
        vectors = np.lib.format.open_memmap(args.output / target, mode='r+' if progress_path.exists() else 'w+', dtype=np.float32, shape=(total, 384))
        records = iter(read_jsonl(args.output / source))
        for _ in range(progress['rows']):
            next(records)
        for start in range(progress['rows'], total, 256):
            began = time.perf_counter()
            docs = [next(records) for _ in range(min(256, total - start))]
            texts = [prefix + ((d.get('title', '') + '\n' + d['text']).strip() if prefix == 'passage: ' else d['text']) for d in docs]
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            for offset in range(0, len(order), 8):
                ids = order[offset:offset + 8]
                encoded = tokenizer.encode_batch([texts[i] for i in ids])
                batch = {'input_ids': np.array([t.ids for t in encoded], dtype=np.int64),
                         'attention_mask': np.array([t.attention_mask for t in encoded], dtype=np.int64),
                         'token_type_ids': np.array([t.type_ids for t in encoded], dtype=np.int64)}
                hidden = session.run(None, {k: v for k, v in batch.items() if k in inputs})[0]
                if hidden.ndim != 3 or hidden.shape[-1] != 384:
                    raise ValueError(f'unexpected model output shape: {hidden.shape}')
                mask = batch['attention_mask'][..., None]
                pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
                pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
                if not np.isfinite(pooled).all():
                    raise ValueError('nonfinite embeddings')
                vectors[np.array(ids) + start] = pooled
            vectors.flush()
            progress.update(rows=start + len(docs), seconds=progress['seconds'] + time.perf_counter() - began)
            temporary = progress_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(progress))
            temporary.replace(progress_path)
            print(f'{target}: {progress["rows"]}/{total}; completed chunks {progress["seconds"]:.1f}s', flush=True)
        if next(records, None) is not None:
            raise ValueError('unexpected extra input rows')
        del vectors
        seconds += progress['seconds']
    return {**config, 'seconds': seconds, 'timing': 'completed encoding chunks; excludes model setup/download; resumable'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['scifact', 'miracl-zh'], default='scifact')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--documents', type=int, default=0, help='MIRACL only: 0 means full corpus; nonzero creates an explicitly labeled judged-plus-background subset')
    parser.add_argument('--text-only', action='store_true', help='prepare corpus/labels without embeddings, for lexical evaluation')
    args = parser.parse_args()
    if args.documents < 0 or (args.dataset == 'scifact' and args.documents):
        parser.error('document limit must be nonnegative and is only supported for MIRACL')
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'manifest.json').exists():
        parser.error('completed dataset already exists; choose another output directory')
    metadata = prepare_miracl(args) if args.dataset == 'miracl-zh' else prepare_scifact(args)
    metadata['dimensions'] = 0 if args.text_only else 384
    files = ['corpus.jsonl', 'queries.jsonl', metadata['qrels_file']]
    if not args.text_only:
        metadata.update(model=MODEL, revision=REVISION, embedding=embed(args, metadata))
        files.extend(['vectors.npy', 'query_vectors.npy'])
    metadata['files_sha256'] = {name: sha256(args.output / name) for name in files}
    metadata['preparation_sha256'] = sha256(Path(__file__))
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
