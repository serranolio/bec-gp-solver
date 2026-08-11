#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_sweep.py
Generate one TOML config file per simulation job by copying base.toml
and overriding the [sweep] section with the job-specific parameters.

Usage
-----
    python generate_sweep.py --base configs/base.toml --out configs/sweep/

Most sweep parameters live in the [sweep] section of the config. A few
physical parameters live elsewhere (e.g. omega_r in [spin_orbit]); these can
still be swept by declaring their section in SECTION_OF below.

Two index files are written to --out:
  job_list.txt     — one line per config (all jobs, for submit.sh)
  gs_job_list.txt  — one line per unique (delta_start_hz, omega_l_start,
                     omega_r) combination (for submit_gs.sh; avoids redundant
                     ground-state computations when only ramp_time or sample
                     varies, while still recomputing when the ground state
                     itself changes)
"""

import argparse
import shutil
import tomllib
import tomli_w                    # pip install tomli-w
from itertools import product
from pathlib import Path

import logging
from bec_gp_solver._logging import setup_logging
logger = logging.getLogger(__name__)



# =============================================================================
# Define the parameter sweep here
# =============================================================================

SWEEP = {
    'ramp_time_ms'   : [4.0, 10.0, 40.0, 80.0, 150.0],
    'total_time_ms'  : [500.0],
    'delta_start_hz' : [0.0],
    'delta_end_hz'   : [0.0],
    'omega_l_start'  : [0.9],
    'omega_l_end'    : [0.3],
    'sample'         : [1],
    'omega_r'        : [2.7],
}

# Sweep keys default to the [sweep] section. Keys that live in a different
# config section are listed here so the override lands in the right place.
SECTION_OF = {
    'omega_r' : 'spin_orbit',
}

def generate_sweep(base_path, out_dir):
    base_path = Path(base_path)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(base_path, 'rb') as f:
        base_cfg = tomllib.load(f)

    # cartesian product over all sweep axes
    keys   = list(SWEEP.keys())
    values = list(SWEEP.values())

    job_paths = []
    gs_job_paths = []          # one entry per unique (delta_start_hz, omega_l_start)
    seen_gs_keys = set()

    for i, combo in enumerate(product(*values)):
        cfg = _deep_copy(base_cfg)

        # override the relevant section for each swept parameter
        params = dict(zip(keys, combo))
        for key, val in params.items():
            section = SECTION_OF.get(key, 'sweep')
            cfg[section][key] = val

        job_path = out_dir / f'job_{i:04d}.toml'
        with open(job_path, 'wb') as f:
            tomli_w.dump(cfg, f)
        job_paths.append(job_path)

        # Deduplicate ground-state jobs by the parameters that determine the
        # ground state. Read from cfg (not params) so values fall back to the
        # base config when a parameter is not being swept.
        gs_key = (
            cfg['sweep']['delta_start_hz'],
            cfg['sweep']['omega_l_start'],
            cfg['spin_orbit']['omega_r'],
        )
        if gs_key not in seen_gs_keys:
            seen_gs_keys.add(gs_key)
            gs_job_paths.append(job_path)

    # write full job list for submit.sh / jobscript.sh
    job_list = out_dir / 'job_list.txt'
    with open(job_list, 'w') as f:
        for p in job_paths:
            f.write(str(p) + '\n')

    # write deduplicated ground-state job list for submit_gs.sh / jobscript_gs.sh
    gs_job_list = out_dir / 'gs_job_list.txt'
    with open(gs_job_list, 'w') as f:
        for p in gs_job_paths:
            f.write(str(p) + '\n')

    logger.info(f"Generated {len(job_paths)} job configs in {out_dir}/")
    logger.info(f"Job list written to {job_list}")
    logger.info(f"SLURM array range: 0-{len(job_paths)-1}")
    logger.info(
        f"Ground-state job list written to {gs_job_list} "
        f"({len(gs_job_paths)} unique initial condition(s))"
    )
    return job_paths


def _deep_copy(cfg):
    """Deep copy a nested dict of TOML-compatible types."""
    import copy
    return copy.deepcopy(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='configs/base.toml')
    parser.add_argument('--out',  default='configs/sweep')
    args = parser.parse_args()
    setup_logging()

    generate_sweep(args.base, args.out)
