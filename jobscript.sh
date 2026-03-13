#!/bin/bash
#SBATCH --partition=kamiak
#SBATCH --job-name=kzm_sweep_ramp_times
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=federico.serrano@wsu.edu
#SBATCH --time=0-03:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# The array range is set dynamically below.
# To submit: first run generate_sweep.py, then:
#   sbatch --array=0-$(( $(wc -l < configs/sweep/job_list.txt) - 1 )) jobscript.sh
#SBATCH --array=0-19

# --- environment ---
mkdir -p logs
source ~/.bashrc
pixi run python --version   # or: conda activate bec-gp-solver

# --- map SLURM task ID to config file ---
JOB_LIST="configs/sweep/job_list.txt"

if [ ! -f "$JOB_LIST" ]; then
    echo "ERROR: $JOB_LIST not found. Run generate_sweep.py first."
    exit 1
fi

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOB_LIST")

if [ -z "$CONFIG" ]; then
    echo "ERROR: No config found for task ID ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: running config ${CONFIG}"

# --- run ---
srun python run_simulation.py \
    --config     "$CONFIG" \
    --output-dir output \
    --gs-dir     ground_states
