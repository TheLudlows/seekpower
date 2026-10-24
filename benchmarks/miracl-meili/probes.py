#!/usr/bin/env python3
"""Unjudged, deliberately small semantic diagnostics, excluded from performance/quality."""
from run import *
os.sched_setaffinity(0,{2,3})
rows=[{'docid':'p1','title':'alpha','text':'beta apple 中国 软件','date':0}, {'docid':'p2','title':'beta','text':'alpha apples 中國 軟體','date':1}, {'docid':'p3','title':'other','text':'alpha beta application','date':0}]
queries=[('and','alpha beta',None,False),('field','alpha','text',False),('prefix','appl',None,False),('typo','applf',None,False),('simplified','中国',None,False),('traditional','中國',None,False),('filter','alpha',None,True)]
for name in ['anyseek','meili']:
    for profile in ['default','restricted']:
        out=ART/'raw'/f'probes-{name}-{profile}';out.mkdir(exist_ok=False)
        e=Engine(name,profile,out)
        try:
            e.start();e.create();documents=[{**doc(r),'date':r['date']} for r in rows];e.write(documents)
            results=[]
            for label,q,field,filtered in queries:
                status,data,ms,size=e.query(q,field,filtered)
                assert status==200,(label,status,data)
                ids=e.hit_ids(data)
                if label=='filter':assert set(ids)<= {'p1','p3'},ids
                results.append({'label':label,'query':q,'field':field,'filtered':filtered,'ids':ids,'response':data})
            dump(out/'probes.json',{'fixture':rows,'results':results})
        finally:e.close()
