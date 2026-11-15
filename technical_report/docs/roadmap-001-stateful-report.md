# Roadmap 001: English stateful API testing technical report

## Approved scope

The user approved the proposed report on 2026-11-15 and specified English.
Explain how Mirage tests one API in the stateful path, and compare methods and
metrics in roadmaps 008, 009 and 010. Use the IEEE conference template and the
ml-paper-writing and academic-plotting skills. All report material, supporting
scripts, downloads, build outputs, caches and validation logs stay under
technical_report/. This explicit storage instruction overrides the general
root docs/ and sbatch/ locations for this report only. No production, test,
configuration or existing research artifact edits; preserve the Git index.

## Implementation steps

1. Record the original workspace/index fingerprints and source provenance.
2. Read frozen engines, protocols, aggregate results and selected raw evidence.
   Keep phase011 changes outside the historical 008–010 narrative.
3. Acquire the complete official IEEE conference template; retain provenance.
4. Build a source-linked metric dataset and vector/raster quantitative figures.
5. Write an English report, source Markdown, bibliography and build instructions.
   Include a single-API walkthrough, conditional oracle semantics, methods,
   phase-specific comparisons, negative findings and limitations.
6. Validate metric extraction and references, build the template and report in
   Slurm when a prepared engine is available, inspect rendered pages, and verify
   all changed/new files are confined to technical_report/.

## Validation plan

Use lightweight static/read-only checks on the login node. Run plotting and
LaTeX compilation through Slurm; use the default account/partition/resources,
with no silent scheduler overrides. Do not launch a model or rerun framework
experiments for this evidence-synthesis task. Validate source hashes, numerator/
denominator distinctions, missing values, coverage versions and figure inputs.
Check PDF references, layout, embedded fonts and unresolved LaTeX warnings when
build tools permit. Report unavailable checks accurately.

## Progress

- Approved framing: explain single-API stateful testing through runtime evidence,
  conditional MRs and exploration feedback; compare each phase within its own
  experimental protocol.
- English confirmed; implementation starting. No report experiments executed.

### Evidence extraction and first build

- The original IEEE ZIP endpoint returned an empty body; acquired the complete,
  unmodified IEEEtran CTAN distribution instead. Template provenance is retained.
- Template Slurm job 781204 passed (112 seconds, exit 0) before report drafting.
- Parsed historical JSON without Torch or raw tensor loading. Dataset now has
  122 metric records; evidence index covers 73 protected source artifacts.
- Draft review corrected phase010 total MR applicability: reference has six
  ordinary MR comparisons, distinct from zero optimizer MR comparisons.
- Draft report job 781224 passed (39 seconds, exit 0), producing a ten-page PDF
  and all three PDF/PNG figures. It started plotting before the metric correction;
  its stale chart is preserved in validation/draft-781224/ and superseded.
- Improve bar-value spacing and prevent the example listing splitting across
  pages. Rebuild the final report after freezing input hashes. No framework
  experiment, generated-program replay or model inference has been run.
- Default 8gpus/4h scheduler preflight estimated next-day startup. Explicit
  dev/30min/8GB preflight passed after dev limits were inspected. Actual report
  jobs use one GPU/eight CPUs; CPU plotting/TeX does not measure GPU utilization.

### Static checks and final rebuild

- Static metric audit found phase010 reference total ordinary MR count 6 versus
  optimizer MR count 0; the dataset and report retain both explicitly.
- All historical evidence hashes checked; original production/test/config/index
  fingerprints remained unchanged. Four pre-existing phase011 documentation
  files changed concurrently outside this report's commands. Their before/current
  differences are recorded separately, not reverted or claimed report changes.
- Corrected the report-only static validator to recognize the listing label and
  the Markdown exporter to preserve indented code/display math. All static checks
  now pass. This is document validation, not a runtime regression-suite claim.
- Final build submitted as owned Slurm job 781232 with explicit dev/30min/8GB,
  one GPU/eight CPUs. Review its final PDF/figures, logs and job termination.

### Completed

- Final deliverable is the nine-page English IEEEtran conference PDF, main.tex,
  generated report.md, three vector/300-DPI comparison figures, source-linked
  metrics and recorded torch.add walkthrough. All content is inside this directory.
- Final build 781249 passed with exit 0. Every page rendered; Times/Courier-compatible
  body fonts are embedded, and no missing-font, overfull-box or unresolved-reference
  warnings remain. Twenty-three underfull-box spacing warnings remain and were
  reviewed. All nine pages and final chart changes were visually inspected.
- The complete 36-file IEEEtran distribution remains byte-identical to its archive.
  Nine reference keys, all cross-references and the core count table pass static
  checks. All 73 historical source fingerprints are intact. Production/tests/
  configuration and the Git index were preserved; concurrent 011 document updates
  are disclosed in validation/static-checks.json and were not reverted.
- All eight owned document jobs/steps completed 0:0 and are absent from squeue.
  Total allocation is 210 GPU-seconds (0.058333 GPU-hours), not utilization or task
  wall time. No model inference, new framework experiment or replay was performed.
- See validation/validation.md for actual checks, draft corrections, limitations,
  resource accounting and cleanup evidence. IEEE PDF eXpress was not run.
