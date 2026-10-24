#!/usr/bin/env python3
"""Three real positive documents; query-contract diagnosis only, never quality/performance."""
from run import *
os.sched_setaffinity(0,{2,3})
qs=json.loads((ART/'data/queries.json').read_text())[:3];corpus={r['docid']:r for r in json.loads((ART/'data/corpus-1000.json').read_text())}
source=[corpus[next(d for d,g in q['qrels'].items() if g>0)] for q in qs]
for profile in ['default','restricted']:
 out=ART/'raw'/f'review-semantic-{profile}';out.mkdir(exist_ok=False);e=Engine('anyseek',profile,out)
 try:
  e.start();e.create();e.write([doc(r) for r in source]);results=[]
  for q,r in zip(qs,source):
   cases=[]
   for kind,text in [('original',q['text']),('known-title',r['title']),('source-text-query-analysis',r['text'])]:
    for operator in (['and','or'] if kind=='original' else ['or']):
     body={'query':text,'operator':operator,'length':10,'count':True,'fields':['docid','title','text'],'search_fields':['title','text']}
     status,response,ms,size=e.request('POST','/api/v1/index/bench/query',body)
     cases.append({'kind':kind,'request':body,'http_status':status,'response':response})
   results.append({'qid':q['qid'],'positive':r,'qrel':q['qrels'][r['docid']],'cases':cases})
  dump(out/'review.json',results)
 finally:e.close()
