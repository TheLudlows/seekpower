#!/usr/bin/env python3
"""Fetch pinned public inputs; verify binary and qrels before execution."""
import hashlib,json,pathlib,urllib.request
from run import ART
DATA=ART/'data';DATA.mkdir(parents=True,exist_ok=True);(ART/'bin').mkdir(exist_ok=True)
REV='5be20db9509754dadad47689368639fcec739c00'
def fetch(url,path,expected=None):
 req=urllib.request.Request(url,headers={'User-Agent':'anyseek-local-benchmark'})
 with urllib.request.urlopen(req,timeout=90) as r:body=r.read()
 if expected and hashlib.sha256(body).hexdigest()!=expected:raise RuntimeError('Checksum mismatch: '+url)
 path.write_bytes(body)
for folder,sha in [('topics','5b284a9aabf08bb2d1c88ed7ea276025c9d23846457c5350dc8d391b5e0d0a13'),('qrels','5546474d3dc8139014e6571e9f4041848bb0a314ca6d2d4bac708174d71c59eb')]:
 name=folder+'.miracl-v1.0-zh-dev.tsv';fetch(f'https://huggingface.co/datasets/miracl/miracl/resolve/{REV}/miracl-v1.0-zh/{folder}/{name}',DATA/name,sha)
fetch(f'https://huggingface.co/api/datasets/miracl/miracl/revision/{REV}',DATA/'miracl-api.json')
assert json.loads((DATA/'miracl-api.json').read_text())['sha']==REV
fetch(f'https://huggingface.co/datasets/miracl/miracl/resolve/{REV}/README.md',DATA/'miracl-readme.md')
fetch('https://api.github.com/repos/meilisearch/meilisearch/releases/tags/v1.55.0',DATA/'meili-release.json')
release=json.loads((DATA/'meili-release.json').read_text());assert not release['prerelease'] and not release['draft']
asset=next(a for a in release['assets'] if a['name']=='meilisearch-linux-amd64');sha='7e33f5889235f14290074a19a6d3574bd9b10fb2ee8731eabb7dafd0714b0513';assert asset['digest']=='sha256:'+sha
binary=ART/'bin/meilisearch-v1.55.0';fetch(asset['browser_download_url'],binary,sha);binary.chmod(0o700)
for name,dest in [('LICENSE','meili-license.txt'),('LICENSE-MIT','meili-license-mit.txt')]:fetch(f'https://raw.githubusercontent.com/meilisearch/meilisearch/v1.55.0/{name}',DATA/dest)
print('Pinned binary and topics/qrels verified. Existing corpus is independently checked by prepare_data.py.')
