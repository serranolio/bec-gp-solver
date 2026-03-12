#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_sweep.py
Generate one TOML config file per simulation job by copying base.toml
and overriding the [sweep] section with the job-specific parameters.

Usage
-----
    python generate_sweep.py --base configs/base.toml --out configs/sweep/

The script prints the number of jobs generated and the path to a
job_list.txt file that jobscript.sh reads to map SLURM_ARRAY_TASK_ID
to config file paths.
"""

import argparse
import shutil
import tomllib
import tomli_w                    # pip install tomli-w
from itertools import product
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

# =============================================================================
# logging output file 
# =============================================================================

def _setup_logging(log_file=None, verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level,
                        format=fmt,
                        datefmt=datefmt,
                        handlers=handlers)



# =============================================================================
# Define the parameter sweep here
# =============================================================================

SWEEP = {
    'ramp_time_ms'   : list(range(10, 110, 10)),
    'total_time_ms'  : [10.0],
    'delta_start_hz' : [0.0],
    'delta_end_hz'   : [0.0],
    'omega_l_start'  : [0.9],
    'omega_l_end'    : [0.3],
    'sample'         : list(range(10)),
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
    for i, combo in enumerate(product(*values)):
        cfg = _deep_copy(base_cfg)

        # override sweep section
        for key, val in zip(keys, combo):
            cfg['sweep'][key] = val

        job_path = out_dir / f'job_{i:04d}.toml'
        with open(job_path, 'wb') as f:
            tomli_w.dump(cfg, f)
        job_paths.append(job_path)

    # write job list for jobscript.sh
    job_list = out_dir / 'job_list.txt'
    with open(job_list, 'w') as f:
        for p in job_paths:
            f.write(str(p) + '\n')

    logger.info(f"Generated {len(job_paths)} job configs in {out_dir}/")
    logger.info(f"Job list written to {job_list}")
    logger.info(f"SLURM array range: 0-{len(job_paths)-1}")
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
    _setup_logging()

    generate_sweep(args.base, args.out)
