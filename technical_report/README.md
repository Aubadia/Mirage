# Mirage stateful API testing report

**Start with [main.pdf](main.pdf)** (nine pages, IEEE conference layout), or read the generated
[English Markdown](report.md). The editable typesetting source is [main.tex](main.tex).

The report explains how Mirage tests one API, gives a recorded `torch.add`
walkthrough, and compares methods, definitions, protocols, results and limits
in roadmaps 008–010. It uses frozen experiment evidence. It does not run new
framework experiments or include roadmap 011 results in the comparison.

## Contents

- `main.pdf`, `main.tex`, `report.md`: the report and editable sources.
- `references.bib`: two programmatically fetched scholarly citations and seven
  explicit local evidence references; no placeholder references.
- `data/metrics.csv` and `data/metrics.json`: 122 source-linked metric records.
- `data/evidence-index.md`: original artifact paths, hashes and code provenance.
- `data/phase-*-comparison.json`, `data/protocols.json`: original comparison and
  protocol copies, including frozen engine identities.
- `data/example-add-*`: exact historical source/plans and selected saved JSON
  evidence; the program is not rerun by this report.
- `figures/`: three quantitative comparisons, vector PDF and 300-DPI PNG, plus
  `gen_fig_metrics.py` for reproduction. Charts use color plus hatching.
- `templates/`: complete unmodified IEEEtran distribution, downloaded archive
  and template provenance. No class or page-geometry modifications.
- `scripts/`, `sbatch/validation/`: report-only extraction, build and validation.
- `validation/`: actual check results, Slurm logs, rendered pages and grayscale
  chart reviews; `validation.md` identifies the final build and its limitations.
- `tools/`, `cache/`, `build/`: isolated report tools, downloads, TeX/font caches
  and intermediate outputs. These do not change project Python dependencies.
- `docs/roadmap-001-stateful-report.md`: approved scope, steps and progress.

## Rebuild without changing the codebase

Use a prepared Python with matplotlib, numpy and Pillow. The Nano4 script uses
`/work/envstack/apps/application/biology/Anaconda3/anaconda3_v2026.07/anaconda3/bin/python`.
The supplied standalone Tectonic is version 0.17.0, Linux x86_64 musl. The package
contains report-only `pypdf` and `pypdfium2` for PDF inspection and rendering.

All report caches, temporary files and outputs are directed into this directory.
Run from the repository root. Compilation and plotting must run through Slurm:

```bash
bash -n technical_report/scripts/build.sh \
  technical_report/sbatch/validation/build-report.sbatch
sbatch --test-only technical_report/sbatch/validation/build-report.sbatch template
sbatch --parsable technical_report/sbatch/validation/build-report.sbatch template
# Review the template job before the report build:
sbatch --test-only technical_report/sbatch/validation/build-report.sbatch report
sbatch --parsable technical_report/sbatch/validation/build-report.sbatch report
```

The script retains the default account `ACD115198`, partition `8gpus`, one node,
eight CPUs, one GPU and four hours. For the executed builds, default preflight
estimated next-day startup. Verified explicit overrides were used:

```bash
sbatch --test-only --partition=dev --time=00:30:00 --mem=8G \
  technical_report/sbatch/validation/build-report.sbatch report
sbatch --parsable --partition=dev --time=00:30:00 --mem=8G \
  technical_report/sbatch/validation/build-report.sbatch report
```

They are resource choices for document generation, not framework experiments.
An `srun` step executes the commands, logs start/end/elapsed time and preserves
nonzero failures. No vLLM service is started or reused by this task. The build
records its input files/hashes under `validation/build-JOB_ID/` and renders each
PDF page. Network access can be needed when the isolated TeX cache is cold.

To recreate the source-linked dataset in the original repository, these are
lightweight JSON/text checks and require no Torch execution:

```bash
python3 technical_report/scripts/extract_metrics.py
python3 technical_report/scripts/index_evidence.py
python3 technical_report/scripts/assemble_references.py
python3 technical_report/scripts/export_markdown.py
python3 technical_report/scripts/validate_report.py
```

The original raw JSON files must be present for re-extraction. Existing report
figures/PDF can instead be built from the included extracted data alone.
For a fresh report tools directory, `prepare_tex_tool.py` and
`prepare_pdf_tools.py` download only into this package. `fetch_references.py`
retrieves scholarly BibTeX; run `assemble_references.py` afterward to include
local evidence references. Original fetched BibTeX is retained separately.

## Interpretation

Compare reference and candidate **within each phase**. API panels, budgets, path
policies and endpoints differ across phases. There is one seed per arm/API; no
error bars or statistical superiority claims are supported. Unknown, unsupported,
inapplicable and unmeasured are kept separate from pass and measured zero.
In phase010, reference total applicable MR pairs are **6**, while reference
optimizer MR pairs are **0**. The candidate has 15 optimizer MR passes but zero
numerical-audit agreement records. These are different correctness measurements.

The current worktree contains pre-existing 011 work. The report does not modify
production code, tests, project configuration, the index or historical results.
The workspace audit separately records concurrent changes to 011 documentation;
it does not claim that an otherwise active repository stayed globally static.
