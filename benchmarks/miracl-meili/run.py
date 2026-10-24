#!/usr/bin/env python3
"""Bounded local HTTP experiment. Never modifies production code or raises quotas."""
import argparse, base64, concurrent.futures, hashlib, http.client, json, math, os, pathlib, signal, socket, subprocess, threading, time
ROOT=pathlib.Path(__file__).resolve().parents[2]; ART=ROOT/'artifacts/benchmark'; SECRET='local-benchmark-only-20261024'; MEILI_KEY='local-benchmark-only-master-key-20261024'
def dump(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def op(i): return {'epoch':int(time.time()//86400),'nonce':f'benchmark-{i:08d}'}
def doc(row): return {'id':hashlib.sha256(row['docid'].encode()).hexdigest()[:32], 'docid':row['docid'], 'title':row['title'], 'text':row['text'], 'date':int(hashlib.sha256(row['docid'].encode()).hexdigest()[:4],16)%2}
def disk(root):
    logical=allocated=0
    for p in root.rglob('*'):
        try:
            if p.is_file(): s=p.stat();logical+=s.st_size;allocated+=s.st_blocks*512
        except FileNotFoundError:pass
    return {'logical_bytes':logical,'allocated_bytes':allocated}
def stats(values):
    if not values:return {}
    v=sorted(values)
    return {'n':len(v),'mean_ms':sum(v)/len(v), **{f'p{q}_ms':v[max(0,math.ceil(len(v)*q/100)-1)] for q in [50,95,99]}}
class Engine:
    def __init__(self,name,profile,out):
        self.name=name;self.profile=profile;self.out=out;self.data=out/'db';self.local=threading.local();self.seq=0;self.process=None;self.phase='starting';self.resources=[]
        self.stop_sample=threading.Event();self.writer=(out/'events.jsonl').open('w');self.lock=threading.Lock()
        sock=socket.socket();sock.bind(('127.0.0.1',0));self.port=sock.getsockname()[1];sock.close()
    def event(self,kind,**data):
        with self.lock:self.writer.write(json.dumps({'wall_time':time.time(),'kind':kind,**data},ensure_ascii=False)+'\n');self.writer.flush()
    def request(self,method,path,body=None,required=True):
        if not getattr(self.local,'conn',None):self.local.conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=35)
        payload=None if body is None else json.dumps(body,ensure_ascii=False,separators=(',',':')).encode()
        headers={'Content-Type':'application/json'}
        headers.update({'apikey':base64.b64encode(hashlib.sha256(SECRET.encode()).digest()).decode()} if self.name=='anyseek' else {'Authorization':'Bearer '+MEILI_KEY})
        start=time.perf_counter()
        try:
            self.local.conn.request(method,path,payload,headers);r=self.local.conn.getresponse();raw=r.read();code=r.status;data=json.loads(raw) if raw else {}
        except Exception as e:
            self.local.conn.close();self.local.conn=None;code=0;data={'error':str(e)};raw=b''
        ms=(time.perf_counter()-start)*1000
        if required and not 200<=code<300:raise RuntimeError((method,path,code,data))
        return code,data,ms,len(raw)
    def sample(self):
        while not self.stop_sample.wait(.1):
            p=self.process
            if p is None or p.poll() is not None:continue
            try:
                fields={line.split(':')[0]:line.split(':',1)[1].strip() for line in pathlib.Path(f'/proc/{p.pid}/status').read_text().splitlines() if ':' in line}
                s=pathlib.Path(f'/proc/{p.pid}/stat').read_text().split()
                self.resources.append({'time':time.time(),'phase':self.phase,'rss_kib':int(fields['VmRSS'].split()[0]),'hwm_kib':int(fields['VmHWM'].split()[0]),'cpu_ticks':int(s[13])+int(s[14]),'threads':int(fields['Threads'])})
            except (OSError,KeyError,ValueError):pass
    def start(self):
        self.local=threading.local();binary=ROOT/'target/release/anyseek_server' if self.name=='anyseek' else ART/'bin/meilisearch-v1.55.0'
        args=[str(binary),f'index_path={self.data}',f'local_port={self.port}','local_ip=127.0.0.1','force_shard_number=1'] if self.name=='anyseek' else [str(binary),'--db-path',str(self.data),'--http-addr',f'127.0.0.1:{self.port}','--env','production','--master-key',MEILI_KEY,'--no-analytics','--max-indexing-threads','2']
        env={**os.environ,'MASTER_KEY_SECRET':SECRET,'TOKIO_WORKER_THREADS':'2','RAYON_NUM_THREADS':'2'}
        # Shared available memory is identical; no product-specific RAM cap or cgroup edits.
        log=(self.out/'server.log').open('a');start=time.perf_counter();self.process=subprocess.Popen(['taskset','-c','0,1',*args],cwd=self.out,env=env,stdout=log,stderr=log);log.close()
        self.event('start',pid=self.process.pid,args=args,env_overrides={k:env[k] for k in ['TOKIO_WORKER_THREADS','RAYON_NUM_THREADS']})
        if not hasattr(self,'sampler'):self.sampler=threading.Thread(target=self.sample,daemon=True);self.sampler.start()
        path='/api/v1/live' if self.name=='anyseek' else '/health'
        until=time.monotonic()+60
        while time.monotonic()<until:
            if self.process.poll() is not None:raise RuntimeError('server exited: '+(self.out/'server.log').read_text()[-3000:])
            if self.request('GET',path,required=False)[0]==200:return (time.perf_counter()-start)*1000
            time.sleep(.05)
        raise RuntimeError('startup exceeded 60 seconds')
    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait();self.event('forced_kill')
        self.process=None
    def close(self):
        self.stop();self.stop_sample.set()
        if hasattr(self,'sampler'):self.sampler.join(timeout=2)
        dump(self.out/'resources.json',self.resources);self.writer.close()
    def task(self,data):
        uid=data['taskUid'];start=time.monotonic()
        while time.monotonic()-start<180:
            result=self.request('GET',f'/tasks/{uid}')[1]
            if result['status']=='succeeded':return result
            if result['status'] in ['failed','canceled']:raise RuntimeError(result)
            time.sleep(.01)
        raise RuntimeError('Meili task exceeded 180 seconds')
    def create(self):
        if self.name=='anyseek':
            cfg={'index_name':'benchmark','schema':[{'field':f,'field_type':'Text','index_lexical':f in ['title','text'],'store':True} for f in ['id','docid','title','text']]+[{'field':'date','field_type':'Timestamp','index_lexical':False,'filterable':True,'store':False}]}
            if self.profile=='restricted':cfg['tokenizer']='UnicodeAlphanumericZH'
            self.seq+=1;self.request('POST','/api/v1/index',{'operation_id':op(self.seq),'collection_id':'bench','config':cfg})
            dump(self.out/'settings.json',self.request('GET','/api/v1/index/bench')[1])
        else:
            self.task(self.request('POST','/indexes',{'uid':'bench','primaryKey':'id'})[1])
            settings={'searchableAttributes':['title','text'],'displayedAttributes':['id','docid','title','text'],'filterableAttributes':['date']}
            if self.profile=='restricted':settings.update({'typoTolerance':{'enabled':False},'prefixSearch':'disabled'})
            self.task(self.request('PATCH','/indexes/bench/settings',settings)[1]);dump(self.out/'settings.json',self.request('GET','/indexes/bench/settings')[1])
    def write(self,docs):
        started=time.perf_counter()
        if self.name=='anyseek':
            writes=[]
            for d in docs:
                self.seq+=1;writes.append({'operation_id':op(self.seq),'record_id':d['id'],'routing_key':'benchmark','expected':'Any','document':d})
            code,data,ms,_=self.request('POST','/api/v1/index/bench/batch',{'writes':writes},required=False)
            failures=[r for r in data.get('results',[]) if r.get('code')!='applied']
            accepted=sum(r.get('code')=='applied' for r in data.get('results',[]))
            self.event('write',count=len(docs),accepted=accepted,status=code,ms=ms,result=data)
            if code!=200 or failures or accepted!=len(docs):raise RuntimeError(('write rejected',code,failures or data))
        else:
            code,data,ms,_=self.request('POST','/indexes/bench/documents',docs,required=False)
            self.event('write_submit',count=len(docs),status=code,ms=ms,result=data)
            if code!=202:raise RuntimeError(('write rejected',code,data))
            self.event('write_task',result=self.task(data),count=len(docs))
        return (time.perf_counter()-started)*1000
    def query(self,text,field=None,filtered=False):
        if self.name=='anyseek':
            data={'query':text,'length':10,'count':False,'fields':['id','docid','title','text'],'search_fields':[field] if field else ['title','text']}
            if self.profile=='restricted':data['operator']='and'
            if filtered:data['filter']=[{'Timestamp':{'field':'date','filter':{'start':0,'end':1}}}]
            return self.request('POST','/api/v1/index/bench/query',data,required=False)
        data={'q':text,'limit':10,'attributesToRetrieve':['id','docid','title','text'],'attributesToSearchOn':[field] if field else ['title','text']}
        if self.profile=='restricted':data['matchingStrategy']='all'
        if filtered:data['filter']='date = 0'
        return self.request('POST','/indexes/bench/search',data,required=False)
    def hit_ids(self,data):
        if self.name=='anyseek':return [r['record']['document']['docid'] for r in data.get('hits',[])]
        return [r['docid'] for r in data.get('hits',[])]
    def count(self):
        if self.name=='anyseek':return self.request('POST','/api/v1/index/bench/query',{'query':'','length':0,'count':True})[1]['count_total']
        return self.request('GET','/indexes/bench/stats')[1]['numberOfDocuments']
    def checkpoint(self):
        if self.name=='anyseek':return self.request('PATCH','/api/v1/index/bench')[1]
        return {'note':'no equivalent per-index checkpoint requested; successful asynchronous task and process-reopen verification used'}

def quality(engine,queries,out):
    rows=[]
    for q in queries:
        code,data,ms,size=engine.query(q['text']);ids=engine.hit_ids(data) if code==200 else []
        relevant={d for d,g in q.get('qrels',{}).items() if g>0}
        labels=[q.get('qrels',{}).get(d,0)>0 for d in ids]
        ideal=sum(1/math.log2(i+2) for i in range(min(10,len(relevant))))
        rows.append({'qid':q['qid'],'query':q['text'],'status':code,'ms':ms,'bytes':size,'ids':ids,'response':data,
            'recall10':len(set(ids)&relevant)/len(relevant) if relevant else None,
            'ndcg10':sum(v/math.log2(i+2) for i,v in enumerate(labels))/ideal if ideal else None,
            'mrr10':next((1/(i+1) for i,v in enumerate(labels) if v),0) if relevant else None})
    dump(out,rows)
    return {k:sum(r[k] for r in rows if r[k] is not None)/max(1,sum(r[k] is not None for r in rows)) for k in ['recall10','ndcg10','mrr10']}|{'queries':len(rows),'errors':sum(r['status']!=200 for r in rows),'empty':sum(not r['ids'] for r in rows)}

def load(engine,queries,concurrency,seconds,label):
    barrier=threading.Barrier(concurrency+1);deadline=[0.0];all_rows=[]
    def worker(worker_id):
        rows=[];i=worker_id;barrier.wait()
        while time.perf_counter()<deadline[0]:
            q=queries[i%len(queries)];t=time.perf_counter();code,data,ms,size=engine.query(q['text']);rows.append({'worker':worker_id,'qid':q['qid'],'start':t,'ms':ms,'status':code,'bytes':size,'error':data if code!=200 else None});i+=concurrency
        return rows
    cpu=time.process_time();start=time.perf_counter();deadline[0]=start+seconds
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures=[pool.submit(worker,i) for i in range(concurrency)];barrier.wait()
        for f in futures:all_rows.extend(f.result())
    elapsed=time.perf_counter()-start;cpu=time.process_time()-cpu
    with (engine.out/(label+'.jsonl')).open('w') as f:
        for r in all_rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    success=[r['ms'] for r in all_rows if r['status']==200]
    return {'label':label,'concurrency':concurrency,'requested_seconds':seconds,'elapsed_seconds':elapsed,'requests':len(all_rows),'successful':len(success),'errors':len(all_rows)-len(success),'error_rate':1-len(success)/max(1,len(all_rows)),'qps':len(success)/elapsed,'client_cpu_seconds':cpu,'latency':stats(success),'all_status_latency':stats([r['ms'] for r in all_rows])}

def main():
    p=argparse.ArgumentParser();p.add_argument('--engine',choices=['anyseek','meili'],required=True);p.add_argument('--profile',choices=['default','restricted'],required=True);p.add_argument('--n',type=int,choices=[1000,5000],default=1000);p.add_argument('--tag',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--write-mix',action='store_true');p.add_argument('--seconds',type=float,default=5);p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
    os.sched_setaffinity(0,{2,3});out=ART/'raw'/a.tag;out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((ART/'data'/f'corpus-{a.n}.json').read_text());queries=json.loads((ART/'data/queries.json').read_text());mixed=json.loads((ART/'data/mixed-probes.json').read_text())
    if a.smoke:rows=rows[:20];queries=queries[:3]
    documents=[doc(row) for row in rows];engine=Engine(a.engine,a.profile,out);summary={'args':vars(a),'status':'running','process_restart_is_not_cold_cache':True}
    dump(out/'run.json',summary)
    try:
        summary['initial_startup_ms']=engine.start();engine.create();engine.phase='build';start=time.perf_counter()
        for i in range(0,len(documents),50):
            engine.write(documents[i:i+50]);print(a.tag,'loaded',min(i+50,len(documents)),flush=True)
        assert engine.count()==len(documents);summary['build_to_searchable_seconds']=time.perf_counter()-start;summary['count']=engine.count();summary['disk_after_build']=disk(engine.data)
        engine.phase='checkpoint';start=time.perf_counter();summary['checkpoint']=engine.checkpoint();summary['checkpoint_ms']=(time.perf_counter()-start)*1000
        summary['disk_after_checkpoint']=disk(engine.data)
        # Confirm original title/text through each formal document API.
        for d in [documents[0],documents[len(documents)//2],documents[-1]]:
            path=f'/api/v1/index/bench/doc/{d["id"]}' if a.engine=='anyseek' else f'/indexes/bench/documents/{d["id"]}'
            returned=engine.request('GET',path)[1]
            if a.engine=='anyseek':returned=returned['document']
            assert all(returned[k]==d[k] for k in ['id','docid','title','text']),returned
        summary['content_roundtrip_verified']=3
        engine.phase='quality';summary['quality']=quality(engine,queries,out/'quality.json');quality(engine,mixed,out/'mixed-unjudged.json')
        engine.phase='restart';engine.stop();summary['restart_startup_ms']=engine.start();assert engine.count()==len(documents)
        summary['restart_first_queries']=[]
        for q in queries:
            code,data,ms,size=engine.query(q['text']);summary['restart_first_queries'].append({'qid':q['qid'],'status':code,'ms':ms,'bytes':size})
        for _ in range(2):
            for q in queries:assert engine.query(q['text'])[0]==200
        engine.phase='warm'
        summary['warm']=[]
        if not a.smoke and not a.write_mix:
            for repeat in range(a.repeats):
                for concurrency in [1,4,8]:
                    result=load(engine,queries,concurrency,a.seconds,f'warm-r{repeat}-c{concurrency}');summary['warm'].append(result);dump(out/'run.json',summary);print(a.tag,result,flush=True)
        if a.write_mix:
            assert a.n==1000;engine.phase='mixed-write';writes=[]
            def writer():
                start=time.perf_counter()
                for i in range(100):
                    time.sleep(max(0,start+i*.5-time.perf_counter()));d={**documents[i],'text':documents[i]['text']+' benchmark_update_marker'}
                    t=time.perf_counter()
                    try:ms=engine.write([d]);writes.append({'i':i,'start':t,'ms':ms,'ok':True})
                    except Exception as e:writes.append({'i':i,'start':t,'ok':False,'error':repr(e)})
            thread=threading.Thread(target=writer);thread.start();summary['mixed_query']=load(engine,queries,4,50,'during-writes');thread.join();summary['writes']=writes;summary['final_count']=engine.count()
        summary['status']='completed'
    except Exception as e:
        summary['status']='failed';summary['error']=repr(e)
        try:summary['accepted_count_on_failure']=engine.count()
        except Exception:pass
        raise
    finally:
        summary['disk_final']=disk(engine.data) if engine.data.exists() else {};engine.close();summary['peak_rss_kib']=max((r['rss_kib'] for r in engine.resources),default=0);summary['peak_hwm_kib']=max((r['hwm_kib'] for r in engine.resources),default=0);dump(out/'run.json',summary)
if __name__=='__main__':main()
