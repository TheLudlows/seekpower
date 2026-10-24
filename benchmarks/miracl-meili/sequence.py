#!/usr/bin/env python3
"""Sequential continuation; never overlaps servers or retries an existing experiment."""
import pathlib,subprocess,json
from run import ROOT,ART,dump
steps=[('meili','default',1000),('anyseek','default',5000),('meili','default',5000)]
results=[]
for engine,profile,n in steps:
 tag=f'{engine}-{profile}-{n}'
 if (ART/'raw'/tag).exists():raise SystemExit('Refusing duplicate '+tag)
 with (ART/'raw'/(tag+'.log')).open('w') as log:
  status=subprocess.call(['python3',str(ROOT/'benchmarks/miracl-meili/run.py'),'--engine',engine,'--profile',profile,'--n',str(n),'--tag',tag],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
 results.append({'tag':tag,'exit_code':status});dump(ART/'raw/sequence.json',results)
 # No 5000 Meili performance run if the matching anyseek corpus could not load.
 if engine=='anyseek' and status:break
