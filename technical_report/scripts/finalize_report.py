"""Finalize document validation and Slurm cleanup evidence after visual review."""
import hashlib,json,pathlib,re,subprocess,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
ids=['781204','781224','781232','781237','781239','781240','781247','781249']
x=json.loads((ROOT/'validation/pdf-inspection.json').read_text());assert x['job_id']==ids[-1],x['job_id']
for f in ['main.tex','references.bib','metrics.json','gen_fig_metrics.py']:
 current=ROOT/({'metrics.json':'data/metrics.json','gen_fig_metrics.py':'figures/gen_fig_metrics.py'}.get(f,f));frozen=ROOT/'validation'/('build-'+ids[-1])/f
 assert current.read_bytes()==frozen.read_bytes(),f
log=(ROOT/'build/paper/main.log').read_text()
assert not re.search(r'Overfull|Font Warning|Citation .*undefined|Reference .*undefined|There were undefined',log)
raw=subprocess.check_output(['sacct','-j',','.join(ids),'--format=JobID,JobName,Partition,State,ExitCode,ElapsedRaw,AllocTRES','-P'],text=True)
(ROOT/'validation/slurm-accounting.txt').write_text(raw)
lines=[line.split('|') for line in raw.splitlines()[1:] if line.split('|')[0] in ids]
assert len(lines)==len(ids)
assert all(a[3]=='COMPLETED' and a[4]=='0:0' for a in lines)
queue=subprocess.check_output(['squeue','-h','-j',','.join(ids),'-o','%i %T'],text=True)
assert not queue.strip(),queue
(ROOT/'validation/owned-jobs-final.txt').write_text('No owned report jobs remain in squeue.\n'+queue)
with zipfile.ZipFile(ROOT/'templates/IEEEtran.zip') as z:
 members=[n for n in z.namelist() if not n.endswith('/')]
 assert all((ROOT/'templates'/n).read_bytes()==z.read(n) for n in members)
allocated=sum(int(a[5]) for a in lines)
sha=hashlib.sha256((ROOT/'main.pdf').read_bytes()).hexdigest()
result={'job_ids':ids,'all_top_level_jobs_completed_zero':True,'final_job':ids[-1],'allocated_gpu_seconds':allocated,'allocated_gpu_hours':allocated/3600,'scope':'One allocated GPU per job; includes cold TeX downloads and document build iterations. Allocation is not GPU utilization. No model startup/inference/restarts.','final_pdf_sha256':sha,'final_page_count':len(x['pages']),'final_inputs_match_recorded_snapshot':True,'complete_template_distribution_unchanged_files':len(members),'no_overfull_boxes_missing_fonts_or_unresolved_citations':True,'underfull_box_warnings':len(re.findall(r'Underfull',log)),'cache_retained':'Report-local reproducibility cache; no services or temporary compute processes remain.'}
(ROOT/'validation/completion.json').write_text(json.dumps(result,indent=2)+'\n')
static=json.loads((ROOT/'validation/static-checks.json').read_text())['checks']
text=f'''# Report validation

The final English report is **{len(x['pages'])} pages**, US Letter, IEEEtran conference
mode, built in owned Slurm job **{ids[-1]}**. SHA256: `{sha}`.
This validates the document and its evidence synthesis, not Mirage correctness.

## Checks actually run

- Original unmodified conference skeleton built first, job 781204, exit 0.
- All {len(members)} extracted IEEEtran files still match the full CTAN archive.
- Final report/figure generation, bibliography compilation and every-page PDF
  rendering executed in Slurm. All report jobs completed with 0:0; see
  `slurm-accounting.txt` and individual `.out`, `.err`, `job-ID.json` records.
- All 122 extracted metric records have source paths/fields or explicit original
  JSON aggregation rules. Six core count rows are compared numerically with the
  LaTeX table; original planned-valid counts and ordinary/optimizer distinctions
  are checked. No Torch import, tensor loading or framework replay was used.
- {static['historical_evidence_hashes_preserved']} historical evidence fingerprints
  match. The Git index and production/tests/configuration fingerprints match.
  Four phase011 documentation files changed concurrently; they are listed in
  `static-checks.json`, preserved and excluded from claims of global immutability.
  The report's commands only write under `technical_report/`.
- Nine citation keys and all referenced labels resolve. Two scholarly references
  were fetched as BibTeX via DOI registries and checked against primary arXiv
  records; local references are explicit repository evidence, not invented papers.
- All Python scripts parse; shell scripts pass `bash -n`. Original failed draft
  checks (listing-label parsing and concurrent document detection) were corrected
  before final validation. No runtime regression-suite claim is made.
- Final TeX log has no overfull boxes, missing-font warnings, undefined citation
  or undefined reference warnings. {result['underfull_box_warnings']} underfull
  boxes remain: these are spacing warnings, including narrow table cells. They
  do not indicate text clipping. Body Times-compatible and Courier-compatible
  fonts and the directly inspected page fonts are embedded.
- Three quantitative figures have vector PDF and 300-DPI PNG outputs. Grayscale
  chart copies and all page renders were reviewed. Drafting corrected total MR
  applicability (010 reference 6) versus optimizer MR applicability (0), chart
  label collisions, a split/overlapping listing caption and XeTeX font encoding.
  Final main/metrics/bibliography/figure-script bytes match the recorded snapshot
  in `build-{ids[-1]}/`.

## Limits

No framework experiments, model calls, raw tensor reassessment or defect
confirmations were performed for this report. Existing campaign/fixture results
are historical evidence. One seed per arm/API supports descriptive comparisons;
no confidence intervals or significance test were fabricated. IEEE PDF eXpress
has not been run, and no unspecified venue acceptance/compliance is claimed.

## Resource and cleanup evidence

Eight short owned jobs include the cold template build and report-layout
iterations. Total assigned resources are {allocated} GPU-seconds
({allocated/3600:.6f} allocated GPU-hours), with one GPU/eight CPUs/8 GB per job.
This is allocation, not utilization, and not elapsed task wall time. No vLLM
service was started. All owned jobs/steps are terminal and absent from squeue;
unrelated jobs were left running. All tools, retained caches, inputs, outputs and
validation logs remain under the report directory.
'''
(ROOT/'validation/validation.md').write_text(text)
print(json.dumps(result,indent=2))
