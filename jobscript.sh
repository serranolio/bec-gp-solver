#!/bin/bash
#SBATCH --partition=kamiak
#SBATCH --job-name=kzm_sweep_ramp_times
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=federico.serrano@wsu.edu
#SBATCH --time=0-24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

# Array range is set dynamically by submit.sh — do not call sbatch directly.
## SBATCH --array=0-4

# --- paths ---
SWEEP_DIR="${SWEEP_DIR:-configs/sweep}"
JOB_LIST="${SWEEP_DIR}/job_list.txt"
OUTPUT_DIR="output"
GS_DIR="ground_states"

# --- environment ---
mkdir -p "$OUTPUT_DIR"
source ~/.bashrc
pixi run python --version

# --- map SLURM task ID to config file ---
if [ ! -f "$JOB_LIST" ]; then
    echo "ERROR: $JOB_LIST not found. Run generate_sweep_files.py first."
    exit 1
fi

# --- initial state: explicit file, or matching ground state from GS_DIR ---
# INITIAL_STATE is exported by submit.sh; empty means look up GS_DIR.
EXTRA_ARGS=()
if [ -n "${INITIAL_STATE:-}" ]; then
    echo "Using initial state: ${INITIAL_STATE}"
    EXTRA_ARGS+=(--initial-state "${INITIAL_STATE}")
elif [ ! -d "$GS_DIR" ] || [ -z "$(ls -A "$GS_DIR")" ]; then
    echo "ERROR: $GS_DIR is empty or missing. Run submit_gs.sh first."
    exit 1
fi

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOB_LIST")

if [ -z "$CONFIG" ]; then
    echo "ERROR: No config found for task ID ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: running config ${CONFIG}"

# --- threads: one BLAS/OpenMP thread per allocated core ---
# Without this, OpenBLAS may start a thread for every core on the node in
# each job, and array tasks sharing a node oversubscribe it.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# --- run ---
srun pixi run python run_simulation.py \
    --config     "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --gs-dir     "$GS_DIR" \
    "${EXTRA_ARGS[@]}"
