#!/bin/bash
# Source this helper inside the attached testing step, after session.env.

mirage_iteration() {
    : "${SLURM_STEP_ID:?Run inside the allocated testing step}"
    if [[ -f "$MIRAGE_SESSION_DIR/cancellation-unconfirmed" ]]; then
        echo "Previous inference cancellation is unconfirmed; restart the service before reuse." >&2
        return 2
    fi
    if [[ -z "${MIRAGE_TEST_CACHE_DIR:-}" ]]; then
        export MIRAGE_TEST_CACHE_DIR
        MIRAGE_TEST_CACHE_DIR=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/mirage-test-${SLURM_JOB_ID}.XXXXXX")
        printf '%s\n' "$MIRAGE_TEST_CACHE_DIR" > "$MIRAGE_SESSION_DIR/test-cache-path.txt"
    fi
    export TORCHINDUCTOR_CACHE_DIR="$MIRAGE_TEST_CACHE_DIR/inductor"
    export TRITON_CACHE_DIR="$MIRAGE_TEST_CACHE_DIR/triton"
    local iteration status=0 started budget argument
    local arguments=()
    started=$(date +%s)
    iteration=$(mktemp -d "${MIRAGE_SESSION_DIR}/iteration-XXXXXX")
    mkdir -p "$iteration/code"
    cp -a "$MIRAGE_PROJECT_ROOT/src" "$MIRAGE_PROJECT_ROOT/tests" "$MIRAGE_PROJECT_ROOT/sbatch" "$iteration/code/"
    if [[ -d "$MIRAGE_PROJECT_ROOT/benchmarks" ]]; then
        cp -a "$MIRAGE_PROJECT_ROOT/benchmarks" "$iteration/code/"
    fi
    source "$iteration/code/sbatch/lib/environment.sh"
    budget=$(mirage_budget "${MIRAGE_ITERATION_TIMEOUT:-1800}" 300) || return
    for argument in "$@"; do
        argument="${argument/#$MIRAGE_PROJECT_ROOT\/tests/$iteration\/code\/tests}"
        arguments+=("${argument/#$MIRAGE_PROJECT_ROOT\/benchmarks/$iteration\/code\/benchmarks}")
    done
    printf '%q ' "$@" > "$iteration/command.txt"
    printf '\n' >> "$iteration/command.txt"
    PYTHONPATH="$iteration/code/src" timeout --signal=TERM --kill-after=30 \
        "$budget" "$MIRAGE_PYTHON" "${arguments[@]}" \
        > "$iteration/stdout.log" 2> "$iteration/stderr.log" || status=$?
    printf '{"exit_status":%s,"elapsed_seconds":%s}\n' \
        "$status" "$(( $(date +%s) - started ))" > "$iteration/status.json"
    printf 'Iteration: %s; exit status: %s\n' "$iteration" "$status"
    return "$status"
}
