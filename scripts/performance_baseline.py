#!/usr/bin/env python3
"""Reproducible Linux HTTP baseline; starts an isolated server and needs NumPy only for data/ground truth."""
import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import http.client
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
import secrets
import socket
import statistics
import subprocess
import tempfile
import threading
import time


def encode(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()


class Client:
    def __init__(self, port, key):
        self.connection = http.client.HTTPConnection('127.0.0.1', port, timeout=180)
        self.headers = {'apikey': key, 'Content-Type': 'application/json'}

    def request(self, method, path, body=None):
        if body is not None and not isinstance(body, bytes):
            body = encode(body)
        start = time.perf_counter()
        self.connection.request(method, path, body, self.headers)
        response = self.connection.getresponse()
        raw = response.read()
        elapsed = time.perf_counter() - start
        if response.status != 200:
            raise RuntimeError(f'{method} {path}: HTTP {response.status}: {raw[:500]!r}')
        return json.loads(raw), elapsed * 1000, len(raw)

    def close(self):
        self.connection.close()


def resource_pid(process, index_argument):
    # /proc can be mounted from an outer PID namespace in containers.
    for path in [Path(f'/proc/{process.pid}/cmdline'), *Path('/proc').glob('[0-9]*/cmdline')]:
        try:
            if index_argument.encode() in path.read_bytes().split(b'\0'):
                return int(path.parent.name)
        except OSError:
            continue
    raise RuntimeError('cannot locate benchmark server in /proc for resource measurement')


def process_usage(pid):
    """CPU seconds across all server threads, and resident bytes (Linux /proc)."""
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    cpu = (int(stat[11]) + int(stat[12])) / os.sysconf('SC_CLK_TCK')
    rss = int(Path(f'/proc/{pid}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    return cpu, rss


class Measurement:
    def __init__(self, pid):
        self.pid = pid
        self.stop = threading.Event()

    def __enter__(self):
        self.cpu, self.peak = process_usage(self.pid)
        self.driver_cpu = time.process_time()
        self.start = time.perf_counter()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.wait(0.05):
            try:
                self.peak = max(self.peak, process_usage(self.pid)[1])
            except FileNotFoundError:
                return

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self.start
        driver_cpu = time.process_time() - self.driver_cpu
        self.stop.set()
        self.thread.join()
        cpu, rss = process_usage(self.pid)
        self.result = {
            'wall_seconds': elapsed,
            'server_cpu_seconds': cpu - self.cpu,
            'server_cpu_cores': (cpu - self.cpu) / elapsed,
            'driver_cpu_cores': driver_cpu / elapsed,
            'server_peak_rss_mib': max(self.peak, rss) / 2**20,
        }


def generate(args):
    # Generation and brute-force reference run before timed server work.
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import numpy as np
    if args.dataset:
        return load_dataset(args)
    rng = np.random.default_rng(args.seed)
    topics = min(64, max(1, args.documents // args.top_k))
    centers = rng.standard_normal((topics, args.dimensions), dtype=np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    vectors = centers[np.arange(args.documents) % topics] + rng.standard_normal(
        (args.documents, args.dimensions), dtype=np.float32) * (0.4 / math.sqrt(args.dimensions))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query_vectors = centers[np.arange(args.query_pool) % topics] + rng.standard_normal(
        (args.query_pool, args.dimensions), dtype=np.float32) * (0.4 / math.sqrt(args.dimensions))
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)
    vocabulary = ('database search storage index vector query document server engine memory disk '
                  'network cache stream transaction table partition cluster replica shard token '
                  'filter range score result text field value record update insert delete read '
                  'write commit log recovery snapshot backup balance tenant resource capacity '
                  'request response latency throughput batch thread queue worker block page '
                  'compression encoding language ranking keyword similarity distance metadata').split()
    text_rng = random.Random(args.seed)
    batches, batch = [], []
    digest = hashlib.sha256()
    for i in range(args.documents):
        batch.append({'record_id': i, 'body': f'topic{i % topics:04d} ' + ' '.join(
            text_rng.choices(vocabulary, k=48)), 'vector': vectors[i].tolist()})
        if len(batch) == args.batch_size or i + 1 == args.documents:
            payload = encode(batch)
            batches.append(payload)
            digest.update(payload)
            batch = []
    # Independent float64 cosine reference over the actual float32 input values.
    exact_vectors = vectors.astype(np.float64)
    exact_vectors /= np.linalg.norm(exact_vectors, axis=1, keepdims=True)
    exact_queries = query_vectors.astype(np.float64)
    exact_queries /= np.linalg.norm(exact_queries, axis=1, keepdims=True)
    scores = exact_queries @ exact_vectors.T
    truth = np.argsort(-scores, axis=1, kind='stable')[:, :args.top_k].tolist()
    payloads = {'lexical': [], 'vector': [], 'hybrid': []}
    for i, vector in enumerate(query_vectors.tolist()):
        text = f'topic{i % topics:04d}'
        for mode, query in [('lexical', {'query': text}), ('vector', {'query_vector': vector}),
                            ('hybrid', {'query': text, 'query_vector': vector})]:
            # Keep the API's count=true and full stored-document projection defaults.
            payloads[mode].append(encode({**query, 'length': args.top_k}))
    return batches, payloads, truth, {
        'numpy_version': np.__version__, 'topics': topics, 'words_per_document': 49,
        'input_json_bytes': sum(map(len, batches)), 'input_sha256': digest.hexdigest(),
        'queries_sha256': hashlib.sha256(b''.join(p for group in payloads.values() for p in group)).hexdigest(),
    }


class DiskBatches:
    """Pre-serialize outside the timed run without retaining all HTTP bodies in RAM."""
    def __init__(self):
        self.file = tempfile.TemporaryFile()
        self.digest = hashlib.sha256()
        self.size = 0

    def append(self, documents):
        payload = encode(documents)
        self.file.write(payload + b'\n')
        self.digest.update(payload)
        self.size += len(payload)

    def __iter__(self):
        self.file.seek(0)
        for line in self.file:
            yield line[:-1]

    def close(self):
        self.file.close()


def exact_neighbors(vectors, queries, k, block_size=4096):
    """Exhaustive float64 cosine, bounded memory, stable document-row tie breaking."""
    import numpy as np
    queries = queries.astype(np.float64)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if not np.isfinite(queries).all() or (norms == 0).any():
        raise ValueError('invalid query embedding')
    queries /= norms
    best_scores = np.empty((len(queries), 0))
    best_ids = np.empty((len(queries), 0), dtype=np.int64)
    for start in range(0, len(vectors), block_size):
        block = np.asarray(vectors[start:start + block_size], dtype=np.float64)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        if not np.isfinite(block).all() or (norms == 0).any():
            raise ValueError('invalid document embedding')
        block /= norms
        scores = queries @ block.T
        local = np.argsort(-scores, axis=1, kind='stable')[:, :k]
        ids = np.concatenate([best_ids, local + start], axis=1)
        candidates = np.concatenate([best_scores, np.take_along_axis(scores, local, axis=1)], axis=1)
        order = np.lexsort((ids, -candidates), axis=1)[:, :k]
        best_ids = np.take_along_axis(ids, order, axis=1)
        best_scores = np.take_along_axis(candidates, order, axis=1)
    return best_ids.tolist()


class TextNormalization:
    """Benchmark-only normalization; keep stored text separate from indexed text."""
    def __init__(self, mode):
        self.convert = lambda text: text
        self.report = {'mode': mode, 'location': 'client preprocessing, excluded from engine timings',
                       'documents': {'count': 0, 'changed': 0, 'conversion_seconds': 0.0},
                       'queries': {'count': 0, 'changed': 0, 'conversion_seconds': 0.0},
                       'query_conversion_ms': []}
        if mode == 't2s':
            import opencc
            version = importlib.metadata.version('OpenCC')
            if version != '1.4.2':
                raise ValueError('normalization experiment requires OpenCC==1.4.2')
            self.convert = opencc.OpenCC('t2s.json').convert
            root = Path(opencc.__file__).parent
            self.report.update(package='OpenCC', version=version, config='t2s.json',
                files_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(root.rglob('*')) if p.suffix in ('.json', '.ocd2', '.so', '.py')})

    def __call__(self, text, kind):
        start = time.perf_counter()
        converted = self.convert(text)
        elapsed = time.perf_counter() - start
        stats = self.report[kind]
        stats['count'] += 1
        stats['changed'] += converted != text
        stats['conversion_seconds'] += elapsed
        if kind == 'queries':
            self.report['query_conversion_ms'].append(elapsed * 1000)
        return converted


def load_dataset(args):
    """Stream a prepared BEIR corpus; mmap embeddings and block exact reference search."""
    import numpy as np
    directory = Path(args.dataset)
    metadata = json.loads((directory / 'manifest.json').read_text())
    for name, expected in metadata['files_sha256'].items():
        with (directory / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'dataset checksum mismatch: {name}')
    queries = [json.loads(line) for line in (directory / 'queries.jsonl').read_text().splitlines()]
    args.documents, args.query_pool = metadata['documents'], len(queries)
    args.dimensions = metadata['dimensions']
    needs_vectors = any(mode != 'lexical' for mode in args.modes)
    normalization = getattr(args, 'text_normalization', None)
    if normalization and (needs_vectors or metadata.get('tokenizer') != 'UnicodeAlphanumericZH'):
        raise ValueError('text normalization experiment requires Chinese lexical-only data')
    normalize = TextNormalization(normalization) if normalization else None
    if needs_vectors and not args.dimensions:
        raise ValueError('text-only dataset requires --modes lexical')
    vectors = np.load(directory / 'vectors.npy', mmap_mode='r', allow_pickle=False) if needs_vectors else None
    query_vectors = np.load(directory / 'query_vectors.npy', allow_pickle=False) if needs_vectors else None
    if (not len(queries) or args.top_k > args.documents or
            (needs_vectors and (vectors.shape != (args.documents, args.dimensions) or
                               query_vectors.shape != (len(queries), args.dimensions)))):
        raise ValueError('corpus/query/embedding dimensions do not match')
    query_rows = {str(query['_id']): i for i, query in enumerate(queries)}
    if len(query_rows) != len(queries):
        raise ValueError('duplicate query IDs')
    labeled = []
    for line in (directory / metadata.get('qrels_file', 'qrels/test.tsv')).read_text().splitlines()[1:]:
        query_id, doc_id, relevance = line.split('\t')
        if query_id in query_rows and float(relevance) > 0:
            labeled.append((query_id, doc_id, float(relevance)))
    relevant_ids = {doc for _, doc, _ in labeled}
    doc_rows, batches, batch = {}, DiskBatches(), []
    count = 0
    with (directory / 'corpus.jsonl').open() as stream:
        for i, line in enumerate(stream):
            doc = json.loads(line)
            doc_id = str(doc['_id'])
            if doc_id in relevant_ids:
                if doc_id in doc_rows:
                    raise ValueError('duplicate relevant document ID')
                doc_rows[doc_id] = i
            document = {'record_id': i, 'body': (doc.get('title', '') + '\n' + doc['text']).strip()}
            if normalize:
                document['search_body'] = normalize(document['body'], 'documents')
            if needs_vectors:
                document['vector'] = vectors[i].tolist()
            batch.append(document)
            count += 1
            if normalize and count % 250000 == 0:
                print(f'Prepared {count:,} documents ({normalization})', flush=True)
            if len(batch) == args.batch_size:
                batches.append(batch)
                batch = []
    if batch:
        batches.append(batch)
    if count != args.documents or relevant_ids != doc_rows.keys():
        raise ValueError('corpus count or relevant documents do not match')
    qrels = [{} for _ in queries]
    for query, doc, relevance in labeled:
        qrels[query_rows[query]][doc_rows[doc]] = relevance
    if not all(qrels):
        raise ValueError('each evaluation query must have positive relevance judgments')
    truth = exact_neighbors(vectors, query_vectors, args.top_k) if needs_vectors else None
    payloads = {mode: [] for mode in args.modes}
    for i, query in enumerate(queries):
        for mode in payloads:
            request = {'operator': 'or', 'length': args.top_k}
            if mode != 'vector':
                request['query'] = normalize(query['text'], 'queries') if normalize else query['text']
            if mode != 'lexical':
                request['query_vector'] = query_vectors[i].tolist()
            payloads[mode].append(encode(request))
    return batches, payloads, truth, {
        **metadata, '_qrels': qrels, 'query_ids': list(query_rows),
        **({'text_normalization': normalize.report} if normalize else {}),
        'input_json_bytes': batches.size, 'input_sha256': batches.digest.hexdigest(),
        'queries_sha256': hashlib.sha256(b''.join(p for group in payloads.values() for p in group)).hexdigest(),
    }


def ranking_metrics(ids, relevant):
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate result IDs')
    dcg = sum(relevant.get(doc, 0) / math.log2(rank + 2) for rank, doc in enumerate(ids[:10]))
    ideal = sum(score / math.log2(rank + 2) for rank, score in
                enumerate(sorted(relevant.values(), reverse=True)[:10]))
    return {'ndcg_at_10': dcg / ideal,
            'recall_at_10': len(set(ids[:10]) & relevant.keys()) / len(relevant),
            'recall_at_100': len(set(ids[:100]) & relevant.keys()) / len(relevant),
            'mrr_at_10': next((1 / rank for rank, doc in enumerate(ids[:10], 1) if doc in relevant), 0)}


def percentiles(values):
    ordered = sorted(values)
    return {f'p{p}': ordered[max(0, math.ceil(len(ordered) * p / 100) - 1)]
            for p in (50, 95, 99)}


def init_worker(port, key, path, payloads, warmup, ready, start, deadline):
    global worker_state
    worker_state = port, key, path, payloads, warmup, ready, start, deadline


def query_worker(number):
    port, key, path, payloads, warmup, ready, start, deadline = worker_state
    client = Client(port, key)
    latencies, engine_ms, response_bytes = [], [], 0
    position = number
    try:
        try:
            for i in range(warmup):
                client.request('POST', path, payloads[(position + i) % len(payloads)])
            ready.wait(timeout=180)
        except Exception:
            ready.abort()
            raise
        start.wait()
        cpu_start = time.process_time()
        while time.perf_counter() < deadline.value:
            result, ms, size = client.request('POST', path, payloads[position % len(payloads)])
            # Never hide HTTP/JSON/shape errors in QPS.
            if not isinstance(result.get('results'), list) or result['count'] != len(result['results']):
                raise RuntimeError(f'invalid query response: {result}')
            latencies.append(ms)
            engine_ms.append(result['time'] / 1e6)
            response_bytes += size
            position += 1
        return latencies, engine_ms, response_bytes, time.process_time() - cpu_start
    finally:
        client.close()


def load(port, key, path, payloads, concurrency, args, pid):
    # Separate Python processes avoid a shared GIL becoming the concurrency limit.
    context = multiprocessing.get_context('fork')
    ready = context.Barrier(concurrency + 1)
    start = context.Event()
    deadline = context.Value('d', 0.0, lock=False)
    with ProcessPoolExecutor(max_workers=concurrency, mp_context=context,
            initializer=init_worker,
            initargs=(port, key, path, payloads, args.warmup, ready, start, deadline)) as pool:
        futures = [pool.submit(query_worker, i) for i in range(concurrency)]
        ready.wait(timeout=180)
        with Measurement(pid) as measurement:
            deadline.value = time.perf_counter() + args.seconds
            start.set()
            results = [future.result() for future in futures]
    latencies = [value for result in results for value in result[0]]
    engine_ms = [value for result in results for value in result[1]]
    if not latencies:
        raise RuntimeError('measurement completed without any requests')
    measurement.result['driver_cpu_cores'] += sum(r[3] for r in results) / measurement.result['wall_seconds']
    return {**measurement.result, 'requests': len(latencies), 'errors': 0,
            'qps': len(latencies) / measurement.result['wall_seconds'],
            'http_latency_ms': percentiles(latencies),
            'reported_search_ms': percentiles(engine_ms),
            'mean_response_bytes': sum(r[2] for r in results) / len(latencies)}


def environment(root, binary):
    def command(*args):
        return subprocess.check_output(args, cwd=root, text=True).strip()
    def read(path):
        p = Path(path)
        return p.read_text().strip() if p.exists() else None
    return {
        'engine_commit': command('git', 'rev-parse', 'HEAD'),
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'rustc': command('rustc', '--version'), 'python': platform.python_version(),
        'platform': platform.platform(), 'cpu': command('lscpu'),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'cgroup_cpu_max': read('/sys/fs/cgroup/cpu.max'),
        'cgroup_memory_max': read('/sys/fs/cgroup/memory.max'),
        'cgroup_cpu_stat_before': read('/sys/fs/cgroup/cpu.stat'),
        'meminfo': read('/proc/meminfo'),
        'filesystem': command('df', '-T', tempfile.gettempdir()),
        'cargo_config': read(root / '.cargo/config.toml'),
    }


def run(args, root, binary):
    preparation_start = time.perf_counter()
    batches, payloads, truth, dataset = generate(args)
    preparation_seconds = time.perf_counter() - preparation_start
    qrels = dataset.pop('_qrels', None)
    payloads = {mode: payloads[mode] for mode in args.modes}
    report = {'format_version': 1, 'started_at': datetime.now(timezone.utc).isoformat(),
              'config': {k: v for k, v in vars(args).items() if k not in ('server', 'output')},
              'environment': environment(root, binary), 'dataset': dataset,
              'preparation_wall_seconds': preparation_seconds,
              'method': {'transport': 'loopback HTTP/1.1 keep-alive, one connection per worker process',
                         'load': 'closed-loop; fixed duration; warmed before every repetition',
                         'http_latency': 'request send through complete response read; excludes response JSON decode',
                         'qps': 'successful completions / wall time, including driver JSON decoding and result collection/drain',
                         'reported_search': 'server time field; excludes stored-document materialization/serialization',
                         'rss': 'server RSS sampled every 50 ms; includes mapped resident pages',
                         'quality': 'vector Recall@K against independent float64 exhaustive cosine; no human relevance labels',
                         'scope': 'synthetic, warm cache, default HTTP settings; no cold-cache/filter/update/saturation claim'},
              'queries': []}
    if qrels is not None:
        report['method'].update({
            'quality': 'human qrels: nDCG@10 (linear gains), Recall@10/100, MRR@10; exact cosine Recall@K when vector mode is measured',
            'scope': 'prepared real corpus, warm cache; operator=or; top K for timing, top 100 for relevance; embedding time excluded',
        })
    secret = secrets.token_hex(24)
    key = base64.b64encode(hashlib.sha256(secret.encode()).digest()).decode()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='seekpower-perf-') as directory:
        directory = Path(directory)
        with (directory / 'server.log').open('wb') as log:
            process = subprocess.Popen([str(binary), 'local_ip=127.0.0.1', f'local_port={port}',
                f'index_path={directory / "index"}'], stdout=log, stderr=log,
                env={**os.environ, 'MASTER_KEY_SECRET': secret})
            client = Client(port, key)
            try:
                deadline = time.monotonic() + 60
                while True:
                    try:
                        client.request('GET', '/api/v1/live')
                        break
                    except (OSError, http.client.HTTPException):
                        client.close()
                        if process.poll() is not None or time.monotonic() >= deadline:
                            raise RuntimeError('benchmark server failed to start')
                        time.sleep(0.1)
                pid = resource_pid(process, f'index_path={directory / "index"}')
                config = {'index_name': 'baseline', 'schema': [
                    {'field': 'record_id', 'field_type': 'U32'},
                    {'field': 'body', 'field_type': 'Text'},
                    {'field': 'vector', 'field_type': 'Json', 'index_vector': True}],
                    'vector': {'dimensions': args.dimensions}}
                if dataset.get('tokenizer'):
                    config['tokenizer'] = dataset['tokenizer']
                if not any(mode != 'lexical' for mode in payloads):
                    config['schema'].pop()
                    del config['vector']
                if dataset.get('text_normalization'):
                    config['schema'][1]['index_lexical'] = False
                    config['schema'].append({'field': 'search_body', 'field_type': 'Text', 'store': False})
                    report['method']['text_normalization'] = (
                        'identity/t2s use identical schemas: body stores the original title+text; '
                        'only search_body is indexed and never returned; queries use the same transform. '
                        'Normalization is precomputed; conversion time is reported separately.')
                index_id, _, _ = client.request('POST', '/api/v1/index', config)
                path = f'/api/v1/index/{index_id}'
                with Measurement(pid) as ingest:
                    submitted, periodic_commits = 0, 0
                    for batch in batches:
                        client.request('POST', path + '/doc', batch)
                        submitted += 1
                        if args.max_rss_mib and process_usage(pid)[1] > args.max_rss_mib * 2**20:
                            raise RuntimeError(f'server RSS exceeded {args.max_rss_mib} MiB after {submitted} batches')
                        if args.commit_every and submitted % args.commit_every == 0:
                            client.request('PATCH', path)
                            periodic_commits += 1
                        if submitted % 100 == 0:
                            print(f'Ingested approximately {min(submitted * args.batch_size, args.documents):,} documents', flush=True)
                with Measurement(pid) as commit:
                    count, _, _ = client.request('PATCH', path)
                if count != args.documents:
                    raise RuntimeError(f'indexed count {count} != {args.documents}')
                total = ingest.result['wall_seconds'] + commit.result['wall_seconds']
                report['ingest'] = {'submit': ingest.result, 'commit': commit.result,
                    'periodic_commits': periodic_commits, 'submit_includes_periodic_commits': bool(args.commit_every),
                    'submit_documents_per_second': args.documents / ingest.result['wall_seconds'],
                    'documents_per_second_including_commit': args.documents / total}
                files = list((directory / 'index').rglob('*'))
                report['index'] = {'create_request': config,
                    'info': client.request('GET', path)[0],
                    'logical_bytes': sum(f.stat().st_size for f in files if f.is_file()),
                    'allocated_bytes': sum(f.stat().st_blocks * 512 for f in files if f.is_file()),
                    'metadata': [{'path': str(p.relative_to(directory / 'index')), 'value': json.loads(p.read_text())}
                                 for p in files if p.name == 'index.json']}
                recall = []
                relevance = {mode: [] for mode in payloads}
                for i in range(args.query_pool):
                    if 'vector' in payloads:
                        result, _, _ = client.request('POST', path + '/query', payloads['vector'][i])
                        ids = [doc['record_id'] for doc in result['results']]
                        if len(ids) != args.top_k or len(set(ids)) != args.top_k:
                            raise RuntimeError('vector result count or unique record IDs do not match top_k')
                        recall.append(len(set(ids) & set(truth[i])) / args.top_k)
                    if qrels is not None:
                        for mode in payloads:
                            request = {**json.loads(payloads[mode][i]), 'length': 100}
                            ranked, _, _ = client.request('POST', path + '/query', request)
                            relevance[mode].append(ranking_metrics([doc['record_id'] for doc in ranked['results']], qrels[i]))
                    elif 'lexical' in payloads:
                        lexical, _, _ = client.request('POST', path + '/query', payloads['lexical'][i])
                        topic = i % dataset['topics']
                        expected = len(range(topic, args.documents, dataset['topics']))
                        if lexical['count_total'] != expected:
                            raise RuntimeError(f'lexical count {lexical["count_total"]} != {expected}')
                report['quality'] = {}
                if recall:
                    report['quality'].update(vector_recall_at_k=statistics.mean(recall),
                        vector_min_recall_at_k=min(recall), vector_recall_per_query=recall)
                if qrels is not None:
                    report['quality']['relevance'] = {
                        mode: {metric: statistics.mean(row[metric] for row in rows) for metric in rows[0]}
                        for mode, rows in relevance.items()}
                    report['quality']['relevance_per_query'] = relevance
                elif 'lexical' in payloads:
                    report['quality']['lexical_count_queries_checked'] = args.query_pool
                print(f'Indexed {args.documents} docs incl. commit: {args.documents / total:.1f} docs/s; '
                      f'quality={report["quality"].get("relevance", {})}', flush=True)
                # Shuffle scenario order per repetition to reduce systematic order bias.
                scenarios = [(mode, c) for mode in payloads for c in args.concurrency]
                for repetition in range(args.repeats):
                    order = scenarios.copy()
                    random.Random(args.seed + repetition).shuffle(order)
                    for mode, concurrency in order:
                        result = load(port, key, path + '/query', payloads[mode], concurrency, args, pid)
                        report['queries'].append({'mode': mode, 'concurrency': concurrency,
                                                  'repetition': repetition + 1, **result})
                        print(f'{mode} c={concurrency} repeat={repetition + 1}: {result["qps"]:.1f} QPS; '
                              f'p95={result["http_latency_ms"]["p95"]:.3f} ms', flush=True)
            except Exception:
                log.flush()
                with (directory / 'server.log').open('rb') as failed_log:
                    failed_log.seek(max(0, (directory / 'server.log').stat().st_size - 4000))
                    print(failed_log.read().decode(errors='replace'), flush=True)
                raise
            finally:
                client.close()
                if isinstance(batches, DiskBatches):
                    batches.close()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    report['environment']['cgroup_cpu_stat_after'] = Path('/sys/fs/cgroup/cpu.stat').read_text()
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    report['summary'] = []
    for mode in payloads:
        for concurrency in args.concurrency:
            runs = [r for r in report['queries'] if r['mode'] == mode and r['concurrency'] == concurrency]
            report['summary'].append({'mode': mode, 'concurrency': concurrency,
                'qps_median': statistics.median(r['qps'] for r in runs),
                'qps_min': min(r['qps'] for r in runs), 'qps_max': max(r['qps'] for r in runs),
                'http_latency_ms_median_of_runs': {p: statistics.median(r['http_latency_ms'][p] for r in runs)
                                                  for p in ('p50', 'p95', 'p99')},
                'server_cpu_cores_median': statistics.median(r['server_cpu_cores'] for r in runs),
                'driver_cpu_cores_median': statistics.median(r['driver_cpu_cores'] for r in runs),
                'server_peak_rss_mib': max(r['server_peak_rss_mib'] for r in runs)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='target/release/seekpower_server')
    parser.add_argument('--output', required=True, help='JSON result file; existing files are not overwritten')
    parser.add_argument('--modes', nargs='+', choices=['lexical', 'vector', 'hybrid'], default=['lexical', 'vector', 'hybrid'])
    parser.add_argument('--commit-every', type=int, default=0, help='commit after this many batches; 0 only commits at the end; periodic commits are included in ingest time')
    parser.add_argument('--max-rss-mib', type=int, default=0, help='abort ingestion above this server RSS; 0 disables the guard')
    parser.add_argument('--dataset', help='prepared BEIR dataset directory; overrides synthetic sizes')
    parser.add_argument('--text-normalization', choices=['identity', 't2s'],
                        help='Chinese lexical-only experiment; keep original stored text and normalize indexed/query text before timing')
    parser.add_argument('--documents', type=int, default=20000)
    parser.add_argument('--dimensions', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--query-pool', type=int, default=64)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=32, help='untimed requests per worker before every run')
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--seed', type=int, default=20261015)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        parser.error('resource measurement requires Linux /proc')
    if args.text_normalization and not args.dataset:
        parser.error('--text-normalization requires --dataset')
    if any(value <= 0 for value in [args.documents, args.dimensions, args.batch_size, args.query_pool,
                                    args.top_k, args.seconds, args.repeats, *args.concurrency]) or args.warmup < 0 or args.commit_every < 0 or args.max_rss_mib < 0:
        parser.error('sizes, durations and concurrency must be positive; warmup must be nonnegative')
    if (not args.dataset and args.top_k > args.documents) or len(set(args.concurrency)) != len(args.concurrency) or len(set(args.modes)) != len(args.modes):
        parser.error('top-k must not exceed document count; concurrency entries must be unique')
    output = Path(args.output)
    if output.exists():
        parser.error('output already exists; choose a new file to preserve previous measurements')
    binary = Path(args.server).resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'seekpower', 'seekpower_server',
                    '.cargo', 'Cargo.toml'], cwd=root, check=True, stdout=subprocess.DEVNULL)
    report = run(args, root, binary)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write('\n')
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()
