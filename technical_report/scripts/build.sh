#!/usr/bin/env bash
set -euo pipefail
REPORT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPORT_ROOT"
export XDG_CACHE_HOME="$REPORT_ROOT/cache"
export TEXMFHOME="$REPORT_ROOT/cache/texmf"
export TEXINPUTS="$REPORT_ROOT/templates/IEEEtran//:"
export BSTINPUTS="$REPORT_ROOT/templates/IEEEtran/bibtex//:"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$REPORT_ROOT/cache/matplotlib"
export TMPDIR="$REPORT_ROOT/build/tmp"
mkdir -p "$TMPDIR" "$REPORT_ROOT/build/template" "$REPORT_ROOT/build/paper"
if [[ ${1:-report} == template ]]; then
    "$REPORT_ROOT/tools/tectonic" --keep-logs --keep-intermediates --outdir "$REPORT_ROOT/build/template" "$REPORT_ROOT/templates/IEEEtran/bare_conf.tex"
else
    REPORT_BUILD_RECORD="$REPORT_ROOT/validation/build-${SLURM_JOB_ID:-manual}"
    mkdir -p "$REPORT_BUILD_RECORD"
    cp "$REPORT_ROOT/main.tex" "$REPORT_ROOT/references.bib" "$REPORT_ROOT/data/metrics.json" "$REPORT_ROOT/figures/gen_fig_metrics.py" "$REPORT_BUILD_RECORD/"
    sha256sum "$REPORT_BUILD_RECORD/"* > "$REPORT_BUILD_RECORD/input-sha256.txt"
    REPORT_PYTHON=/work/envstack/apps/application/biology/Anaconda3/anaconda3_v2026.07/anaconda3/bin/python
    "$REPORT_PYTHON" "$REPORT_ROOT/figures/gen_fig_metrics.py"
    "$REPORT_ROOT/tools/tectonic" --keep-logs --keep-intermediates --outdir "$REPORT_ROOT/build/paper" "$REPORT_ROOT/main.tex"
    cp "$REPORT_ROOT/build/paper/main.pdf" "$REPORT_ROOT/main.pdf"
    cp "$REPORT_ROOT/main.pdf" "$REPORT_ROOT/build/paper/main.log" "$REPORT_BUILD_RECORD/"
    "$REPORT_PYTHON" "$REPORT_ROOT/scripts/inspect_pdf.py"
fi
