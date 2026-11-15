"""Assemble fetched scholarly metadata and explicit local artifact references."""
import pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[1];REPO=ROOT.parent
s=''
for key in ['nablafuzz','gpufuzz']:
 raw=(ROOT/'data'/f'{key}.original.bib').read_text()
 s+=re.sub(r'(@\w+\s*\{)[^,]+,',lambda m:m[1]+key+',',raw,count=1).replace('1174–1186','1174--1186')+'\n\n'
items=[('rm008','docs/roadmap-008-joint-state-exploration.md'),('rm009','docs/roadmap-009-general-state-research.md'),('rm010','docs/roadmap-010-optimizer-gradient-chain.md')]
for key,path in items:
 title=(REPO/path).read_text().splitlines()[0].lstrip('# ').replace('—','--')
 s+='@misc{'+key+',\n author={{Mirage project}},\n title={'+title+'},\n year={2026},\n note={Repository roadmap; source path and SHA256 in the accompanying evidence index}\n}\n\n'
for key,title,path in [('result008','Roadmap 008: Frozen paired formal comparison','result/joint-state-exploration/778244/comparison-01/comparison.json'),('result009','Roadmap 009: Frozen general-state formal comparison','result/general-state-research/778971/final-comparison-01/comparison.json'),('result010','Roadmap 010: Frozen optimizer-chain formal comparison','result/optimizer-gradient-chain/779954/final-seed0-report/comparison.json'),('addcase','Roadmap 010: Candidate torch.add saved execution summaries','result/optimizer-gradient-chain/779954/final-seed0/candidate/add/summary.json')]:
 assert (REPO/path).is_file()
 s+='@misc{'+key+',\n author={{Mirage project}},\n title={'+title+'},\n year={2026},\n note={Saved repository experiment artifact; path and SHA256 in the accompanying evidence index}\n}\n\n'
(ROOT/'references.bib').write_text(s)
print('Assembled two fetched scholarly citations and seven local artifact references.')
