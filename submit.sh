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

JOB_LIST="configs/sweep/job_list.txt"

if [ ! -f "$JOB_LIST" ]; then
    echo "ERROR: $JOB_LIST not found. Run generate_sweep_files.py first."
    exit 1
fi

N=$(( $(wc -l < "$JOB_LIST") - 1 ))

mkdir -p logs

echo "Submitting array job: ${N+1} tasks (--array=0-${N})"
sbatch --array=0-"${N}" jobscript.sh
