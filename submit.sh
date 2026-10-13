#!/bin/bash
# submit.sh
# Submission wrapper for the real-time evolution sweep (jobscript.sh).
#
# Workflow
# --------
# 1. Edit configs/base.toml with the fixed physics parameters.
# 2. Generate ground state configs and run the ground state sweep:
#        bash submit_gs.sh
#    Wait for all ground state jobs to complete before continuing.
# 3. Generate the real-time sweep configs:
#        python generate_sweep_files.py --base configs/base.toml --out configs/sweep/
# 4. Run:  bash submit.sh
#
# Options
# -------
#   --sweep-dir <path>      directory holding job_list.txt (default configs/sweep)
#   --initial-state <path>  .npy wavefunction used as the initial state for every
#                           real-time task, instead of the ground state looked up
#                           in ground_states/ by matching parameters. Its shape
#                           must match the sweep geometry (n_comp * N_total).
#                           TWA noise (if enabled) is still added per sample.

SWEEP_DIR="configs/sweep"
INITIAL_STATE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sweep-dir)     SWEEP_DIR="$2";     shift 2 ;;
        --initial-state) INITIAL_STATE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

JOB_LIST="${SWEEP_DIR}/job_list.txt"

if [ ! -f "$JOB_LIST" ]; then
    echo "ERROR: $JOB_LIST not found. Run generate_sweep_files.py first."
    exit 1
fi

# Optional .npy used as the initial state instead of the matching ground state.
# Resolve to an absolute path so it survives any cwd change in the batch job.
if [ -n "$INITIAL_STATE" ]; then
    if [ ! -f "$INITIAL_STATE" ]; then
        echo "ERROR: initial state '$INITIAL_STATE' not found."
        exit 1
    fi
    INITIAL_STATE=$(readlink -f "$INITIAL_STATE")
    echo "Using initial state: $INITIAL_STATE"
fi

N=$(( $(wc -l < "$JOB_LIST") - 1 ))

mkdir -p logs

echo "Submitting array job: $((N+1)) tasks (--array=0-${N})"
sbatch --array=0-"${N}" \
    --export=ALL,SWEEP_DIR="${SWEEP_DIR}",INITIAL_STATE="${INITIAL_STATE}" \
    jobscript.sh
