# Report validation

The final English report is **9 pages**, US Letter, IEEEtran conference
mode, built in owned Slurm job **781249**. SHA256: `8e3c80e701a7a4e7df73c254d81bd709df040b497e5216166df2b32c260c64d5`.
This validates the document and its evidence synthesis, not Mirage correctness.

## Checks actually run

- Original unmodified conference skeleton built first, job 781204, exit 0.
- All 36 extracted IEEEtran files still match the full CTAN archive.
- Final report/figure generation, bibliography compilation and every-page PDF
  rendering executed in Slurm. All report jobs completed with 0:0; see
  `slurm-accounting.txt` and individual `.out`, `.err`, `job-ID.json` records.
- All 122 extracted metric records have source paths/fields or explicit original
  JSON aggregation rules. Six core count rows are compared numerically with the
  LaTeX table; original planned-valid counts and ordinary/optimizer distinctions
  are checked. No Torch import, tensor loading or framework replay was used.
- 73 historical evidence fingerprints
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
  or undefined reference warnings. 23 underfull
  boxes remain: these are spacing warnings, including narrow table cells. They
  do not indicate text clipping. Body Times-compatible and Courier-compatible
  fonts and the directly inspected page fonts are embedded.
- Three quantitative figures have vector PDF and 300-DPI PNG outputs. Grayscale
  chart copies and all page renders were reviewed. Drafting corrected total MR
  applicability (010 reference 6) versus optimizer MR applicability (0), chart
  label collisions, a split/overlapping listing caption and XeTeX font encoding.
  Final main/metrics/bibliography/figure-script bytes match the recorded snapshot
  in `build-781249/`.

## Limits

No framework experiments, model calls, raw tensor reassessment or defect
confirmations were performed for this report. Existing campaign/fixture results
are historical evidence. One seed per arm/API supports descriptive comparisons;
no confidence intervals or significance test were fabricated. IEEE PDF eXpress
has not been run, and no unspecified venue acceptance/compliance is claimed.

## Resource and cleanup evidence

Eight short owned jobs include the cold template build and report-layout
iterations. Total assigned resources are 210 GPU-seconds
(0.058333 allocated GPU-hours), with one GPU/eight CPUs/8 GB per job.
This is allocation, not utilization, and not elapsed task wall time. No vLLM
service was started. All owned jobs/steps are terminal and absent from squeue;
unrelated jobs were left running. All tools, retained caches, inputs, outputs and
validation logs remain under the report directory.
