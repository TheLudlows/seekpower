#!/usr/bin/env python3
"""Verify the pinned corpus and make nested, relevance-covered diagnostic subsets."""
import gzip, hashlib, heapq, json, pathlib, re
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/benchmark/data'
SEED = 'anyseek-meili-20261024-v1'
def digest(value): return hashlib.sha256((SEED + ':' + value).encode()).hexdigest()
def save(name, value):
    p = OUT / name
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest = json.loads((ROOT/'miracl-corpus-v1.0-zh/manifest.json').read_text())
    topics = dict(line.rstrip('\n').split('\t', 1) for line in (OUT/'topics.miracl-v1.0-zh-dev.tsv').read_text().splitlines())
    judgments = {}
    for line in (OUT/'qrels.miracl-v1.0-zh-dev.tsv').read_text().splitlines():
        q, _, doc, grade = line.split(); judgments.setdefault(q, {})[doc] = int(grade)
    selected = sorted((q for q in topics if any(g>0 for g in judgments.get(q,{}).values())),key=digest)[:40]
    required = {doc for q in selected for doc in judgments[q]}
    positives = {doc for q in selected for doc,g in judgments[q].items() if g>0}
    assert len(required)<1000, len(required)
    retained, heap, checked = {}, [], []
    seen = 0
    for entry in manifest['files']:
        p=ROOT/'miracl-corpus-v1.0-zh'/entry['name']; h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
        assert p.stat().st_size==entry['size_bytes'] and h.hexdigest()==entry['sha256'],p
        count=0
        with gzip.open(p,'rt') as f:
            for line in f:
                row=json.loads(line); doc=row['docid']; count+=1; seen+=1
                if doc in required: retained[doc]=row; continue
                score=int(digest(doc),16)
                if len(heap)<5000: heapq.heappush(heap,(-score,doc,row))
                elif score < -heap[0][0]: heapq.heapreplace(heap,(-score,doc,row))
        assert count==entry['documents']
        checked.append({'name':p.name,'sha256':h.hexdigest(),'records':count})
        print('verified',p.name,count,flush=True)
    assert seen==manifest['total_documents'] and required<=retained.keys()
    distractors=[row for _,_,row in sorted(heap,key=lambda x:-x[0])]
    required_rows=[retained[d] for d in sorted(required,key=digest)]
    hashes={}
    for n in [1000,5000]:
        rows=required_rows+distractors[:n-len(required_rows)]
        # Same deterministic insertion order for both products; 1000 is a subset of 5000.
        hashes[str(n)]=save(f'corpus-{n}.json',rows)
    queries=[{'qid':q,'text':topics[q],'qrels':judgments[q]} for q in selected]
    hashes['queries']=save('queries.json',queries)
    mixed=[]
    for row in required_rows+distractors:
        words=re.findall(r'\b[A-Za-z][A-Za-z0-9-]{2,}\b',row['title']+' '+row['text'])
        chinese=re.search(r'[\u4e00-\u9fff]{2,6}',row['title'])
        if words and chinese:
            mixed.append({'qid':f'probe-{len(mixed)}','text':chinese[0]+' '+words[0], 'source_docid':row['docid'], 'label':'constructed from corpus, unjudged'})
        if len(mixed)==10:break
    hashes['mixed_probes']=save('mixed-probes.json',mixed)
    summary={'seed':SEED,'corpus_revision':manifest['revision'],'topics_qrels_revision':json.loads((OUT/'miracl-api.json').read_text())['sha'],
        'verified_shards':checked,'corpus_total':seen,'all_dev_queries':len(topics),'selected_queries':len(selected),
        'required_judged_docs':len(required),'required_positive_docs':len(positives),'selected_positive_coverage':1.0,
        'selection':'40 dev query IDs with smallest seeded SHA256; include ALL their judged positive and negative documents, then globally smallest seeded doc-ID hashes as distractors; nested subsets',
        'limitations':'Relevance-enriched small-corpus diagnostic, NOT full MIRACL score; unjudged treated nonrelevant for nDCG; quality only official 40 queries; constructed mixed probes never receive qrels.',
        'hashes':hashes,'mixed_probes':len(mixed)}
    for n in [1000,5000]:
        rows=json.loads((OUT/f'corpus-{n}.json').read_text())
        summary[f'corpus_{n}']={'utf8_document_bytes':sum(len(json.dumps(r,ensure_ascii=False).encode()) for r in rows),
            'max_document_bytes':max(len(json.dumps(r,ensure_ascii=False).encode()) for r in rows),
            'documents_containing_latin':sum(bool(re.search('[A-Za-z]',r['title']+' '+r['text'])) for r in rows)}
    save('selection-manifest.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__': main()
