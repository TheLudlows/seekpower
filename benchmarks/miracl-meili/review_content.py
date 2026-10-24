#!/usr/bin/env python3
"""Read-only HTTP roundtrip of all 1000 documents, separate from measurements."""
from run import *
os.sched_setaffinity(0,{2,3})
documents=[doc(r) for r in json.loads((ART/'data/corpus-1000.json').read_text())]
for name in ['anyseek','meili']:
 out=ART/'raw'/f'review-content-{name}';out.mkdir(exist_ok=False);e=Engine(name,'default',out);e.data=ART/'raw'/f'{name}-default-1000/db'
 try:
  e.start();assert e.count()==1000;verified=[]
  for d in documents:
   path=f'/api/v1/index/bench/doc/{d["id"]}' if name=='anyseek' else f'/indexes/bench/documents/{d["id"]}'
   status,value,ms,size=e.request('GET',path);r=value['document'] if name=='anyseek' else value
   fields=['id','docid','title','text'];assert all(r[k]==d[k] for k in fields),(name,d['id'])
   verified.append({'id':d['id'],'docid':d['docid'],'http_status':status,'original_sha256':hashlib.sha256(json.dumps({k:d[k] for k in fields},ensure_ascii=False,sort_keys=True).encode()).hexdigest(),'returned_sha256':hashlib.sha256(json.dumps({k:r[k] for k in fields},ensure_ascii=False,sort_keys=True).encode()).hexdigest()})
  dump(out/'verification.json',{'count':len(verified),'all_exact':True,'records':verified})
 finally:e.close()
