#!/usr/bin/env python3
"""Recompute aggregate measurements from retained per-request data, no fabricated samples."""
from run import *
import collections,statistics
results=[]
for p in sorted((ART/'raw').glob('*/run.json')):
 d=json.loads(p.read_text())
 if d['args']['smoke']:continue
 d['tag']=p.parent.name; d['aggregated']=[]
 for c in [1,4,8]:
  runs=[r for r in d.get('warm',[]) if r['concurrency']==c]
  if not runs:continue
  success=[];errors=collections.Counter();all_count=0
  for r in runs:
   for line in (p.parent/(r['label']+'.jsonl')).open():
    sample=json.loads(line);all_count+=1
    if sample['status']==200:success.append(sample['ms'])
    else:errors[str(sample['status'])]+=1
  d['aggregated'].append({'concurrency':c,'repeats':len(runs),'qps_median':statistics.median(r['qps'] for r in runs),'qps_min':min(r['qps'] for r in runs),'qps_max':max(r['qps'] for r in runs),'successful_latency':stats(success),'requests':all_count,'errors_by_http_status':dict(errors),'error_rate':sum(errors.values())/all_count})
 mixed=p.parent/'mixed-unjudged.json'
 if mixed.exists():
  m=json.loads(mixed.read_text());d['mixed_unjudged']={'queries':len(m),'nonempty':sum(bool(r['ids']) for r in m),'errors':sum(r['status']!=200 for r in m)}
 if 'writes' in d:d['write_errors']=sum(not w['ok'] for w in d['writes'])
 results.append(d)
dump(ART/'summary.json',results)
lines=['# 本地 MIRACL 中文诊断：实测表','', '**不是同语义引擎排名：默认anyseek的40个问句均为空；中文配置未完成1000条装载。** 默认分词长词项、中文分词AND/OR最小复核见REPORT.md；原测量数值未改。', '', '由 `analyze.py` 从原始 JSONL 重算；成功请求延迟不含错误请求，必须同时读错误率。QPS 为三轮中位数。','', '|实验|状态/已接受|构建至可搜索 s|Recall@10|nDCG@10|MRR@10|空查询|RSS峰值 MiB|最终分配磁盘 MiB|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for d in results:
 q=d.get('quality',{});f=lambda x: f'{x:.3f}' if isinstance(x,(int,float)) else '—'
 lines.append('|'+ '|'.join([d['tag'],d['status']+'/'+str(d.get('count',d.get('accepted_count_on_failure','?'))),f(d.get('build_to_searchable_seconds')),f(q.get('recall10')),f(q.get('ndcg10')),f(q.get('mrr10')),str(q.get('empty','—')),f(d.get('peak_rss_kib',0)/1024),f(d.get('disk_final',{}).get('allocated_bytes',0)/2**20)])+'|')
lines+=['','|实验|并发|QPS中位数 [min,max]|P50/P95/P99 ms（成功）|错误率|错误状态计数|','|---|---:|---:|---|---:|---|']
for d in results:
 for a in d['aggregated']:
  l=a['successful_latency'];lines.append(f"|{d['tag']}|{a['concurrency']}|{a['qps_median']:.1f} [{a['qps_min']:.1f}, {a['qps_max']:.1f}]|{l.get('p50_ms',0):.2f} / {l.get('p95_ms',0):.2f} / {l.get('p99_ms',0):.2f}|{a['error_rate']:.2%}|{a['errors_by_http_status']}|")
lines+=['','进程重开、checkpoint、混合写入和各轮详情见 `artifacts/benchmark/summary.json`；所有原始响应和计时见 `raw/`。']
(ROOT/'benchmarks/miracl-meili/RESULTS.md').write_text('\n'.join(lines)+'\n')
