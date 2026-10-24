#!/usr/bin/env python3
"""After all measurement servers stop, independently reopen and verify update outcomes."""
from run import *
os.sched_setaffinity(0,{2,3})
documents=[doc(r) for r in json.loads((ART/'data/corpus-1000.json').read_text())[:100]]
for name in ['anyseek','meili']:
 prior=ART/'raw'/f'{name}-default-write1000';result=json.loads((prior/'run.json').read_text());out=ART/'raw'/f'{name}-verify-updates';out.mkdir(exist_ok=False)
 e=Engine(name,'default',out);e.data=prior/'db';verified=[]
 try:
  e.start();assert e.count()==1000
  for d,w in zip(documents,result['writes']):
   path=f'/api/v1/index/bench/doc/{d["id"]}' if name=='anyseek' else f'/indexes/bench/documents/{d["id"]}'
   value=e.request('GET',path)[1];returned=value['document'] if name=='anyseek' else value
   updated=returned['text']==d['text']+' benchmark_update_marker'
   assert updated or returned['text']==d['text'],returned
   assert not w['ok'] or updated,(w,returned)
   verified.append({'i':w['i'],'acknowledged_success':w['ok'],'persisted_update':updated,'version':value.get('version') if name=='anyseek' else None})
  dump(out/'verification.json',{'records':verified,'acknowledged':sum(w['ok'] for w in result['writes']),'persisted':sum(v['persisted_update'] for v in verified),'unconfirmed_but_persisted':sum(v['persisted_update'] and not v['acknowledged_success'] for v in verified)})
 finally:e.close()
