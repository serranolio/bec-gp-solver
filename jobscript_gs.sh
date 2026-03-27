#!/bin/bash
#SBATCH --partition=kamiak
#SBATCH --job-name=kzm_ground_states
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=federico.serrano@wsu.edu
#SBATCH --time=0-06:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=4G

# Array range is set dynamically by submit_gs.sh — do not call sbatch directly.
##SBATCH --array=0-9

# --- paths ---
SWEEP_DIR="${SWEEP_DIR:-configs/sweep}"
JOB_LIST="${SWEEP_DIR}/gs_job_list.txt"
GS_DIR="ground_states"

# --- environment ---
mkdir -p "$GS_DIR"
source ~/.bashrc
pixi run python --version

# --- map SLURM task ID to config file ---
if [ ! -f "$JOB_LIST" ]; then
    echo "ERROR: $JOB_LIST not found. Run generate_sweep_files.py first."
    exit 1
fi

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOB_LIST")

if [ -z "$CONFIG" ]; then
    echo "ERROR: No config found for task ID ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: computing ground state for ${CONFIG}"

# --- run imaginary-time cooling ---
srun pixi run python run_simulation.py \
    --config     "$CONFIG" \
    --output-dir "$GS_DIR" \
    --imag
