#!/bin/bash
# submit_gs.sh
# Submission wrapper for the ground state sweep (jobscript_gs.sh).
#
# Workflow
# --------
# 1. Edit configs/base.toml with the fixed physics parameters.
# 2. Edit generate_sweep_files.py: set SWEEP to the full real-time parameter
#    space (ramp_time_ms, sample, delta_start_hz, omega_l_start, …).
# 3. Generate all sweep configs:
#        python generate_sweep_files.py --base configs/base.toml --out configs/sweep/
#    This writes two index files:
#      configs/sweep/job_list.txt     — full sweep (used by submit.sh)
#      configs/sweep/gs_job_list.txt  — unique (delta_start_hz, omega_l_start)
#                                       combinations only (used here)
# 4. Run:  bash submit_gs.sh
#    Wait for all ground state jobs to complete before continuing.
# 5. Run:  bash submit.sh
#
# Options
# -------
#   --sweep-dir <path>      directory holding gs_job_list.txt (default configs/sweep)
#   --initial-state <path>  .npy wavefunction used as the first guess for cooling
#                           instead of the default Thomas-Fermi profile. Its shape
#                           must match the sweep geometry (n_comp * N_total). The
#                           same file is used for every task in the array.

SWEEP_DIR="configs/sweep"
INITIAL_STATE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sweep-dir)     SWEEP_DIR="$2";     shift 2 ;;
        --initial-state) INITIAL_STATE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

JOB_LIST="${SWEEP_DIR}/gs_job_list.txt"

if [ ! -f "$JOB_LIST" ]; then
    echo "ERROR: $JOB_LIST not found. Run generate_sweep_files.py first."
    exit 1
fi

# Optional .npy used as the first guess for cooling instead of Thomas-Fermi.
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

echo "Submitting ground state sweep: $((N+1)) tasks (--array=0-${N})"
sbatch --array=0-"${N}" \
    --export=ALL,SWEEP_DIR="${SWEEP_DIR}",INITIAL_STATE="${INITIAL_STATE}" \
    jobscript_gs.sh
