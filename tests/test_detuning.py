# -*- coding: utf-8 -*-
"""
test_detuning.py
Detuning quench/ramp with an optional sinusoidal modulation A sin(ωt).

  TestBuilder      — ramp and quench values, the sinusoid's amplitude and
                     frequency in recoil units, the start at t = 0, bounds.
  TestSolver       — the stability estimate uses the bound, the time step is
                     capped by the modulation, the output name records it.

Run with:
    pytest tests/test_detuning.py -v
"""

import sys
from pathlib import Path

import numpy as np
from numpy import pi
import pytest

from bec_gp_solver.config_loader import _build_detuning_ramp
from bec_gp_solver.geometry import GeometryCart
from bec_gp_solver.gp_equation import estimate_max_energy
from bec_gp_solver.projector import Projector, soc_dispersion

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_simulation import build_output_stem, _cap_dt_for_modulation  # noqa: E402

F_RECOIL = 1080.0
T_UNIT   = 1.474e-4


def _cfg(start=0.0, end=0.0, ramp_ms=0.0, mod=None):
    cfg = {'constants': {'f_recoil': F_RECOIL},
           'sweep': {'sample': 1, 'delta_start_hz': start, 'delta_end_hz': end,
                     'ramp_time_ms': ramp_ms, 'omega_l_start': 0.0,
                     'omega_l_end': 0.0, 'total_time_ms': 18.0}}
    if mod is not None:
        cfg['detuning_modulation'] = mod
    return cfg


class TestBuilder:

    def test_no_modulation_is_the_ramp(self):
        delta = _build_detuning_ramp(_cfg(start=200.0, end=-100.0, ramp_ms=5.0), T_UNIT)
        t_ramp = 5e-3 / T_UNIT
        assert np.isclose(delta(0.0), 200.0 / F_RECOIL)
        assert np.isclose(delta(t_ramp), -100.0 / F_RECOIL)
        assert np.isclose(delta(10 * t_ramp), -100.0 / F_RECOIL)
        assert delta.modulation_omega == 0.0
        assert np.isclose(delta.bound, 200.0 / F_RECOIL)

    def test_quench_with_modulation(self):
        """ramp_time = 0 is a quench; the sinusoid rides on the new value."""
        mod   = dict(amplitude_hz=50.0, frequency_hz=500.0)
        delta = _build_detuning_ramp(_cfg(start=0.0, end=2000.0, ramp_ms=0.0, mod=mod), T_UNIT)
        A     = 50.0 / F_RECOIL
        w     = 2 * pi * 500.0 * T_UNIT
        assert np.isclose(delta(0.0), 0.0)                       # t = 0: start value
        for t in (1e-3, 0.7, 5.0):
            assert np.isclose(delta(t), 2000.0 / F_RECOIL + A * np.sin(w * t))

    def test_modulation_units(self):
        """A is in Hz (÷ f_recoil) and ω = 2π f t_unit in recoil time."""
        mod   = dict(amplitude_hz=108.0, frequency_hz=1080.0)
        delta = _build_detuning_ramp(_cfg(mod=mod), T_UNIT)
        assert np.isclose(delta.bound, 0.1)
        assert np.isclose(delta.modulation_omega, 2 * pi * 1080.0 * T_UNIT)
        # one full period in physical time: the sinusoid returns to 0
        period = 1 / 1080.0 / T_UNIT
        assert np.isclose(delta(period), 0.0, atol=1e-12)
        assert np.isclose(delta(period / 4), 0.1)

    def test_bound_holds_on_a_dense_sample(self):
        mod   = dict(amplitude_hz=40.0, frequency_hz=300.0)
        delta = _build_detuning_ramp(_cfg(start=-500.0, end=300.0, ramp_ms=3.0, mod=mod), T_UNIT)
        t     = np.linspace(0, 20.0, 200001)
        vals  = np.array([delta(x) for x in t])
        # a safe bound: the ramp's peak plus the amplitude need not coincide
        assert np.abs(vals).max() <= delta.bound + 1e-12

    def test_ground_state_unchanged_by_modulation(self):
        mod  = dict(amplitude_hz=300.0, frequency_hz=50.0)
        base = _build_detuning_ramp(_cfg(start=120.0, end=0.0, ramp_ms=2.0), T_UNIT)
        modu = _build_detuning_ramp(_cfg(start=120.0, end=0.0, ramp_ms=2.0, mod=mod), T_UNIT)
        assert modu(0.0) == base(0.0)


class TestSolver:

    @pytest.fixture
    def system(self):
        geo  = GeometryCart(sizes=(64,), lengths=(20 * pi,))
        sig  = np.array([1.0, -1.0])
        proj = Projector(geo, soc_dispersion(sig), 2.0)
        kw   = dict(n_components=2, sigma_z=sig, q_zeeman=np.zeros(2), omega_r=0.0,
                    g_matrix=np.zeros((2, 2)), n_atoms=1.0)
        psi  = proj(np.ones(128, complex) / np.sqrt(128 * geo.dv[0]))
        return geo, proj, kw, psi

    def test_estimate_uses_the_modulation_bound(self, system):
        """A modulation of amplitude A raises the bound by (diag factor) × A."""
        geo, proj, kw, psi = system
        A   = 80.0 / F_RECOIL
        d0  = _build_detuning_ramp(_cfg(start=0.0, end=0.0), T_UNIT)
        dm  = _build_detuning_ramp(_cfg(start=0.0, end=0.0,
                                        mod=dict(amplitude_hz=80.0, frequency_hz=10.0)),
                                   T_UNIT)
        e0 = estimate_max_energy(geo, psi, (0.0,), proj, detuning=d0, kinetic=False, **kw)
        em = estimate_max_energy(geo, psi, (0.0,), proj, detuning=dm, kinetic=False, **kw)
        assert np.isclose(em - e0, 0.5 * A)                   # diagonal entries ±A/2

    def test_dt_capped_by_modulation(self):
        mod   = dict(amplitude_hz=10.0, frequency_hz=2000.0)
        delta = _build_detuning_ramp(_cfg(mod=mod), T_UNIT)
        period = 2 * pi / delta.modulation_omega
        assert np.isclose(_cap_dt_for_modulation(10.0, delta, None), period / 20)
        assert _cap_dt_for_modulation(0.001, delta, None) == 0.001

    def test_requested_dt_kept_with_warning(self, caplog):
        mod   = dict(amplitude_hz=10.0, frequency_hz=2000.0)
        delta = _build_detuning_ramp(_cfg(mod=mod), T_UNIT)
        assert _cap_dt_for_modulation(0.5, delta, 0.5) == 0.5
        assert 'fewer than 20 steps' in caplog.text

    def test_no_modulation_leaves_dt_alone(self):
        delta = _build_detuning_ramp(_cfg(start=50.0), T_UNIT)
        assert _cap_dt_for_modulation(0.3, delta, None) == 0.3

    def test_output_name(self):
        plain = _cfg(start=100.0, end=0.0, ramp_ms=0.0)
        mod   = _cfg(start=100.0, end=0.0, ramp_ms=0.0,
                     mod=dict(amplitude_hz=50.0, frequency_hz=500.0))
        assert build_output_stem(plain).endswith('_mod_0hz_0hz')
        assert build_output_stem(mod).endswith('_mod_50hz_500hz')
        assert build_output_stem(mod) == build_output_stem(plain)[:-len('_mod_0hz_0hz')] \
            + '_mod_50hz_500hz'
