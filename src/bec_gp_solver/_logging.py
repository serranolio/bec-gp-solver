# -*- coding: utf-8 -*-
"""
_logging.py
Shared logging configuration for run_simulation.py and generate_sweep_files.py.

Library modules (geometry, config_loader, gp_equation) only acquire a logger
via logging.getLogger(__name__) and never call this function — configuration
is exclusively the responsibility of the entry-point scripts.
"""

import logging
from pathlib import Path


def setup_logging(log_file=None, verbose=False):
    """
    Configure the root logger for a simulation run.

    Should be called once, at the top of the entry-point script's
    ``if __name__ == '__main__'`` block, before any other code runs.

    Parameters
    ----------
    log_file : str or Path or None
        If given, also write all log output to this file in addition to
        stdout. Intended for SLURM runs: pass a per-job path derived from
        the config filename or SLURM_ARRAY_TASK_ID.
    verbose : bool
        If True, set level to DEBUG. Default is INFO.
    """
    level   = logging.DEBUG if verbose else logging.INFO
    fmt     = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(Path(log_file)))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt,
                        handlers=handlers)
