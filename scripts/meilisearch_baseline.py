#!/usr/bin/env python3
"""Meilisearch lexical baseline using the SeekPower dataset, metrics and load driver."""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import random
import secrets
import shutil
import socket
import statistics
import subprocess
import time

import performance_baseline as baseline


class Client(baseline.Client):
    def __init__(self, port, key):
        super().__init__(port, key)
        self.headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}

    def request(self, method, path, body=None):
        if body is not None and not isinstance(body, bytes):
            body = baseline.encode(body)
        start = time.perf_counter()
        self.connection.request(method, path, body, self.headers)
        response = self.connection.getresponse()
        raw = response.read()
        elapsed = (time.perf_counter() - start) * 1000
        if not 200 <= response.status < 300:
            raise RuntimeError(f'{method} {path}: HTTP {response.status}: {raw[:500]!r}')
        result = json.loads(raw)
        if path.endswith('/search'):
            if not isinstance(result.get('hits'), list) or result.get('degraded', False):
                raise RuntimeError(f'invalid/degraded search: {result}')
            # Adapt after measuring the HTTP response; keep raw fields for quality checks.
            result.update(results=result['hits'], count=len(result['hits']),
                          time=result['processingTimeMs'] * 1e6)
        return result, elapsed, len(raw)


def guard(pid, directory, args):
    if args.max_rss_mib and baseline.process_usage(pid)[1] > args.max_rss_mib * 2**20:
        raise RuntimeError(f'server RSS exceeded {args.max_rss_mib} MiB')
    if shutil.disk_usage(directory).free < 1024**3:
        raise RuntimeError('less than 1 GiB free disk space')


def wait_tasks(client, uids, pid, directory, args):
    pending, finished = set(uids), []
    deadline = time.monotonic() + 3600
    while pending:
        guard(pid, directory, args)
        rows = client.request('GET', '/tasks?limit=1000&uids=' + ','.join(map(str, sorted(pending))))[0]['results']
        for task in rows:
            if task['status'] in ('failed', 'canceled'):
                raise RuntimeError(f'task failed: {task}')
            if task['status'] == 'succeeded':
                pending.remove(task['uid'])
                finished.append(task)
        if pending:
            if time.monotonic() > deadline:
                raise TimeoutError('tasks did not complete within one hour')
            time.sleep(0.2)
    return finished


def summarize(report):
    return [dict(concurrency=c,
        qps_median=statistics.median(r['qps'] for r in rows),
        qps_min=min(r['qps'] for r in rows), qps_max=max(r['qps'] for r in rows),
        http_latency_ms_median_of_runs={p: statistics.median(r['http_latency_ms'][p] for r in rows)
                                        for p in ('p50', 'p95', 'p99')},
        server_peak_rss_mib=max(r['server_peak_rss_mib'] for r in rows))
        for c in report['config']['concurrency']
        if (rows := [r for r in report['queries'] if r['concurrency'] == c])]


def run(args):
    root = Path(__file__).resolve().parents[1]
    binary = Path(args.server).resolve(strict=True)
    directory = Path(args.work_dir)
    directory.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    batches, payloads, _, dataset = baseline.load_dataset(args)
    qrels = dataset.pop('_qrels')
    first_document = json.loads(next(iter(batches)))[0]
    queries = [json.loads(p)['query'] for p in payloads['lexical']]
    payloads = [baseline.encode({'q': q, 'limit': args.top_k, 'matchingStrategy': 'last'}) for q in queries]
    environment = baseline.environment(root, binary)
    environment['benchmark_commit'] = environment.pop('engine_commit')
    environment['shared_harness_sha256'] = environment.pop('harness_sha256')
    environment['adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    environment.pop('cargo_config')  # This is an official prebuilt Meilisearch binary.
    report = {'format_version': 1, 'engine': 'Meilisearch', 'status': 'running', 'started_at': started,
        'config': {k: v for k, v in vars(args).items() if k not in ('server', 'output', 'work_dir')},
        'environment': environment, 'dataset': dataset, 'preparation_wall_seconds': time.perf_counter() - start,
        'search_payloads_sha256': hashlib.sha256(b''.join(payloads)).hexdigest(),
        'method': {'transport': 'loopback HTTP/1.1 keep-alive, one connection per worker process',
            'load': 'same closed-loop driver; fixed duration; warmup before every run; HTTP send through response read, excluding JSON decode',
            'quality': 'same qrels and ranking_metrics; top 100 for relevance; top 10 for latency',
            'search': 'default ranking rules, typo tolerance, prefix search and automatic language detection; proximityPrecision recorded in settings; matchingStrategy=last; estimatedTotalHits capped by default maxTotalHits, unlike SeekPower exact count=true OR search',
            'indexing': '500 documents per request; wait for every task in each 100-request group; includes asynchronous indexing completion and polling, not just HTTP acceptance',
            'storage': 'same original body and t2s search_body; only search_body searchable; record_id and original body returned. Meilisearch also stores search_body internally',
            'memory': 'RSS includes resident mapped pages; indexing-memory budget is not an RSS hard limit',
            'scope': 'full lexical corpus; no embeddings, tuning on dev labels, cold-cache or saturation claims'},
        'queries': [], 'ingest_groups': []}
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    key = secrets.token_hex(24)
    command = [str(binary), '--db-path', str(directory / 'data.ms'), '--http-addr', f'127.0.0.1:{port}',
               '--env', 'production', '--no-analytics', '--log-level', 'WARN',
               '--max-indexing-memory', args.indexing_memory, '--max-indexing-threads', str(args.indexing_threads)]
    report['server_arguments'] = command[1:]
    with (directory / 'server.log').open('wb') as log:
        process = subprocess.Popen(command, cwd=directory, stdout=log, stderr=log,
                                   env={**{k:v for k,v in os.environ.items() if not k.startswith('MEILI_')}, 'MEILI_MASTER_KEY': key})
        client = Client(port, key)
        try:
            deadline = time.monotonic() + 60
            while True:
                try:
                    client.request('GET', '/health')
                    break
                except (OSError, http.client.HTTPException):
                    client.close()
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('Meilisearch failed to start')
                    time.sleep(0.1)
            pid = baseline.resource_pid(process, str(directory / 'data.ms'))
            report['environment']['version'] = client.request('GET', '/version')[0]
            task = client.request('POST', '/indexes', {'uid': 'baseline', 'primaryKey': 'record_id'})[0]
            wait_tasks(client, [task['taskUid']], pid, directory, args)
            settings = {'searchableAttributes': ['search_body'], 'displayedAttributes': ['record_id', 'body'],
                        'proximityPrecision': args.proximity_precision}
            task = client.request('PATCH', '/indexes/baseline/settings', settings)[0]
            wait_tasks(client, [task['taskUid']], pid, directory, args)
            report['settings'] = client.request('GET', '/indexes/baseline/settings')[0]
            pending, count = [], 0
            with baseline.Measurement(pid) as ingest:
                for number, batch in enumerate(batches, 1):
                    task = client.request('POST', '/indexes/baseline/documents', batch)[0]
                    pending.append(task['taskUid'])
                    guard(pid, directory, args)
                    if number % args.wait_every == 0:
                        tasks = wait_tasks(client, pending, pid, directory, args)
                        count += sum(t['details']['indexedDocuments'] for t in tasks)
                        report['ingest_groups'].append({'documents': count, 'tasks': len(tasks),
                            'elapsed_seconds': time.perf_counter() - ingest.start})
                        Path(args.output).with_suffix('.partial.json').write_text(
                            json.dumps(report, ensure_ascii=False, indent=2) + '\n')
                        pending.clear()
                        print(f'Indexed {count:,}/{args.documents:,} documents', flush=True)
                if pending:
                    tasks = wait_tasks(client, pending, pid, directory, args)
                    count += sum(t['details']['indexedDocuments'] for t in tasks)
            stats = client.request('GET', '/indexes/baseline/stats')[0]
            if count != args.documents or stats['numberOfDocuments'] != args.documents or stats['isIndexing']:
                raise RuntimeError(f'incomplete index: {count}, {stats}')
            stored = client.request('GET', '/indexes/baseline/documents/0')[0]
            if stored.get('body') != first_document['body'] or stored.get('search_body') != first_document['search_body']:
                raise RuntimeError('stored original/normalized text mismatch')
            report['ingest'] = {**ingest.result, 'documents_per_second': args.documents / ingest.result['wall_seconds']}
            files = [p for p in (directory / 'data.ms').rglob('*') if p.is_file()]
            report['index'] = {'stats': stats, 'logical_bytes': sum(p.stat().st_size for p in files),
                'allocated_bytes': sum(p.stat().st_blocks * 512 for p in files),
                'files': [{'path': str(p.relative_to(directory / 'data.ms')), 'logical_bytes': p.stat().st_size,
                           'allocated_bytes': p.stat().st_blocks * 512} for p in files]}
            per_query, top_ids, hit_counts = [], [], []
            with baseline.Measurement(pid) as quality_measurement:
                for i, q in enumerate(queries):
                    result = client.request('POST', '/indexes/baseline/search', {'q': q, 'limit': 100, 'matchingStrategy': 'last'})[0]
                    ids = [h['record_id'] for h in result['hits']]
                    if any(not isinstance(doc, int) or not 0 <= doc < args.documents for doc in ids):
                        raise RuntimeError('invalid result document IDs')
                    if any(set(h) != {'record_id', 'body'} for h in result['hits']):
                        raise RuntimeError('unexpected returned fields')
                    top_ids.append(ids)
                    hit_counts.append(result.get('estimatedTotalHits'))
                    per_query.append(baseline.ranking_metrics(ids, qrels[i]))
            report['quality'] = {'relevance': {'lexical': {k: statistics.mean(r[k] for r in per_query) for k in per_query[0]}},
                'relevance_per_query': {'lexical': per_query}, 'top_100_record_ids': top_ids,
                'estimated_total_hits': hit_counts, 'measurement': quality_measurement.result}
            print(f'Quality: {report["quality"]["relevance"]}', flush=True)
            for repetition in range(args.repeats):
                order = args.concurrency.copy()
                random.Random(args.seed + repetition).shuffle(order)
                for concurrency in order:
                    result = baseline.load(port, key, '/indexes/baseline/search', payloads,
                                           concurrency, args, pid, client_factory=Client)
                    report['queries'].append({'concurrency': concurrency, 'repetition': repetition + 1, **result})
                    print(f'c={concurrency} repeat={repetition+1}: {result["qps"]:.1f} QPS; p95={result["http_latency_ms"]["p95"]:.3f} ms', flush=True)
            report['summary'] = summarize(report)
            report['status'] = 'completed'
        except Exception as error:
            report.update(status='failed', error=str(error))
            raise
        finally:
            client.close()
            batches.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            report['finished_at'] = datetime.now(timezone.utc).isoformat()
            report['environment']['cgroup_cpu_stat_after'] = Path('/sys/fs/cgroup/cpu.stat').read_text()
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            Path(args.output).with_suffix('.partial.json').unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--work-dir', required=True, help='new directory; retains database and log for inspection')
    parser.add_argument('--text-normalization', choices=['identity', 't2s'], default='t2s')
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--wait-every', type=int, default=100)
    parser.add_argument('--indexing-memory', default='4 GiB')
    parser.add_argument('--proximity-precision', choices=['byWord', 'byAttribute'], default='byWord')
    parser.add_argument('--indexing-threads', type=int, default=4)
    parser.add_argument('--max-rss-mib', type=int, default=0,
                        help='optional RSS stop threshold; 0 relies on container memory limit, allowing mapped-page reclaim')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=32)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--seed', type=int, default=20261015)
    args = parser.parse_args()
    args.modes = ['lexical']
    if any(v <= 0 for v in [args.batch_size, args.wait_every, args.indexing_threads, args.top_k,
                            args.seconds, args.repeats, *args.concurrency]) or args.warmup < 0 or args.max_rss_mib < 0:
        parser.error('sizes/durations must be positive; warmup and RSS guard nonnegative')
    if args.wait_every > 1000 or len(set(args.concurrency)) != len(args.concurrency):
        parser.error('wait-every must be <=1000 and concurrency values unique')
    if Path(args.output).exists() or Path(args.output).with_suffix('.partial.json').exists() or Path(args.work_dir).exists():
        parser.error('output and work directory must not already exist')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run(args)
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
