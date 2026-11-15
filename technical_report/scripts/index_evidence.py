"""Record exact repository provenance and scoped code entry points, read-only."""
import hashlib,json,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[1];REPO=ROOT.parent
sources=json.loads((ROOT/'data/source-fingerprints.json').read_text());protocols=json.loads((ROOT/'data/protocols.json').read_text())
entries=[]
def add(path,kind):
 p=REPO/path
 if not p.is_file():raise FileNotFoundError(p)
 h=hashlib.sha256(p.read_bytes()).hexdigest();sources[path]=h
 entries.append({'path':path,'kind':kind,'sha256':h})
 return p
for p in ['docs/roadmap-008-joint-state-exploration.md','docs/roadmap-009-general-state-research.md','docs/roadmap-010-optimizer-gradient-chain.md','docs/research-009/report.md','docs/research-010/report.md','docs/research-010/novelty-positioning.md','docs/research-009/architecture-gap.md','docs/research-009/certificate-boundaries.md','docs/research-010/final-linear-limitation.md']:
 add(p,'historical documentation')
modules=['workflow.py','generator/state_plan.py','generator/program.py','sampler/policy.py','runtime/stateful.py','runtime/optimizer.py','runtime/worker.py','inspector/stateful.py','inspector/oracles.py','inspector/numerical_gradients.py','rules/catalog.py','backends/pytorch.py','artifacts/report.py','artifacts/corpus.py','contracts/capabilities.py']
code=[]
for m in modules:
 path=str(pathlib.Path(protocols['010']['engines']['candidate']['root']).relative_to(REPO)/'src/mirage'/m);p=add(path,'frozen 010 implementation')
 definitions=[{'line':i,'definition':line.strip()} for i,line in enumerate(p.read_text().splitlines(),1) if re.match(r'^(?:def |class |    def )',line)]
 code.append({'path':path,'entry_points':definitions})
(ROOT/'data/source-fingerprints.json').write_text(json.dumps(sources,indent=2)+'\n')
(ROOT/'data/code-entry-points.json').write_text(json.dumps(code,indent=2)+'\n')
lines=['# Evidence index','', 'All paths below refer to pre-existing repository evidence, read-only. The report does not execute or reclassify the original programs.','', '## Core report citations','']
for e in entries[:9]:lines += ['- ['+e['path']+'](../../'+e['path']+') — SHA256 `'+e['sha256']+'`.']
lines+=['','## Frozen implementation consulted','']
for e in entries[9:]:lines+=['- ['+e['path']+'](../../'+e['path']+') — SHA256 `'+e['sha256']+'`.']
lines+=['','## Quantitative inputs','', 'Every metric in [metrics.csv](metrics.csv) records the original artifact path and JSON field or explicit aggregation rule. [source-fingerprints.json](source-fingerprints.json) includes the three comparisons, all JSON summaries used for aggregation, documentation and code. [protocols.json](protocols.json) retains frozen engine identities and experimental settings.','', 'Original report gates are historical validation results, not newly rerun checks. The report build/static/PDF checks are documented separately in ../validation/validation.md.','', '## Reporting correction during drafting','', 'The first draft chart mistakenly used optimizer-only MR counts for phase010 reference (0), while the total applicable MR count is 6. The corrected dataset retains `mr_applicable=6` and `optimizer_mr_applicable=0` separately. Final charts and the main table use total applicable MR. Draft PDF 781224 is retained only for build history and is superseded.','']
(ROOT/'data/evidence-index.md').write_text('\n'.join(lines))
print('Indexed',len(sources),'protected evidence files and',len(code),'frozen implementation modules')
