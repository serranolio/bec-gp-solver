# -*- coding: utf-8 -*-
"""
test_generate_sweep.py
The sweep generator writes one config per combination of its SWEEP axes,
routing each key to the right section, and one ground-state job per unique
ground state (modulation parameters do not change the ground state).

Run with:
    pytest tests/test_generate_sweep.py -v
"""

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import generate_sweep_files as gen   # noqa: E402


@pytest.fixture
def base(tmp_path):
    import tomli_w
    cfg = {'sweep': {'sample': 1, 'ramp_time_ms': 0.0, 'delta_start_hz': 0.0,
                     'delta_end_hz': 0.0, 'omega_l_start': 0.0, 'omega_l_end': 0.0,
                     'total_time_ms': 18.0},
           'spin_orbit': {'omega_r': 2.7}}
    path = tmp_path / 'base.toml'
    with open(path, 'wb') as f:
        tomli_w.dump(cfg, f)
    return path


def test_sweeps_amplitude_and_frequency(base, tmp_path, monkeypatch):
    monkeypatch.setattr(gen, 'SWEEP', {'amplitude_hz': [0.0, 50.0],
                                       'frequency_hz': [100.0, 500.0, 1000.0]})
    out = tmp_path / 'sweep'
    paths = gen.generate_sweep(base, out)

    assert len(paths) == 6
    seen = set()
    for p in paths:
        with open(p, 'rb') as f:
            cfg = tomllib.load(f)
        mod = cfg['detuning_modulation']
        seen.add((mod['amplitude_hz'], mod['frequency_hz']))
        assert cfg['sweep']['total_time_ms'] == 18.0      # untouched base values kept
    assert seen == {(a, f) for a in (0.0, 50.0) for f in (100.0, 500.0, 1000.0)}


def test_modulation_does_not_add_ground_states(base, tmp_path, monkeypatch):
    monkeypatch.setattr(gen, 'SWEEP', {'amplitude_hz': [0.0, 50.0, 100.0],
                                       'frequency_hz': [100.0, 500.0]})
    gen.generate_sweep(base, tmp_path / 'sweep')
    gs_jobs = (tmp_path / 'sweep' / 'gs_job_list.txt').read_text().split()
    assert len(gs_jobs) == 1


def test_default_writes_zero_modulation(base, tmp_path):
    gen.generate_sweep(base, tmp_path / 'sweep')
    with open(tmp_path / 'sweep' / 'job_0000.toml', 'rb') as f:
        cfg = tomllib.load(f)
    assert cfg['detuning_modulation'] == {'amplitude_hz': 0.0, 'frequency_hz': 0.0}
