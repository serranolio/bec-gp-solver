# -*- coding: utf-8 -*-
"""
test_run_simulation.py
Unit tests for helpers in run_simulation.py.

  TestMomentumKick  — the kick splits a state into ±k components of equal
                      weight and keeps it normalised.
  TestTrapRelease   — [trap].release removes the trap for real-time
                      evolution only.

Run with:
    pytest tests/test_run_simulation.py -v
"""

import sys
from pathlib import Path

import numpy as np
from numpy import pi
import pytest

from bec_gp_solver.geometry import GeometryCart, Geometry3DAxial

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_simulation import apply_momentum_kick, _evolution_kwargs   # noqa: E402


def _norm(geo, psi, n_comp):
    return (np.tile(geo.dv, n_comp) * np.abs(psi)**2).sum()


class TestMomentumKick:

    @pytest.fixture
    def gaussian_1d(self):
        geo = GeometryCart(sizes=(512,), lengths=(40 * pi,))
        z,  = geo.grids
        psi = np.exp(-z**2 / (2 * 8.0**2)).astype(complex)
        return geo, psi / np.sqrt(_norm(geo, psi, 1))

    def test_normalised(self, gaussian_1d):
        geo, psi = gaussian_1d
        out = apply_momentum_kick(psi, geo, 1, 2.0)
        assert np.isclose(_norm(geo, out, 1), 1.0)

    def test_two_equal_momentum_peaks(self, gaussian_1d):
        geo, psi = gaussian_1d
        out = apply_momentum_kick(psi, geo, 1, 2.0)
        a2  = np.abs(geo.to_spectral(out))[0]**2
        k,  = geo.spectral_kgrids
        plus, minus = a2[k > 0].sum(), a2[k < 0].sum()
        assert np.isclose(plus, minus, rtol=1e-6)
        assert np.isclose(plus + minus, 1.0, atol=1e-6)
        # the peaks sit at ±2
        assert np.isclose(k[k > 0][np.argmax(a2[k > 0])], 2.0, atol=2 * pi / (40 * pi))
        assert np.isclose(k[k < 0][np.argmax(a2[k < 0])], -2.0, atol=2 * pi / (40 * pi))

    def test_acts_along_z_in_axial_geometry(self):
        geo = Geometry3DAxial(sizes=(16, 256), lengths=(10.0, 40 * pi))
        r, z = geo.grids
        psi  = np.exp(-r**2 / 8 - z**2 / 128).reshape(-1).astype(complex)
        out  = apply_momentum_kick(psi, geo, 1, 2.0).reshape(geo.sizes)
        # the kick multiplies by √2 cos(2z): unchanged radial profile at each z
        ratio = out / psi.reshape(geo.sizes)
        assert np.allclose(ratio, ratio[0:1, :])

    def test_multicomponent(self, gaussian_1d):
        geo, psi = gaussian_1d
        two = np.concatenate([psi, 0.5 * psi]) / np.sqrt(1.25)
        out = apply_momentum_kick(two, geo, 2, 2.0)
        assert np.isclose(_norm(geo, out, 2), 1.0)
        assert np.allclose(out[512:], 0.5 * out[:512])


class TestTrapRelease:

    KW = dict(n_components=1, homogeneous=False)

    def test_release_turns_trap_off(self):
        out = _evolution_kwargs({'trap': {'release': True}}, self.KW)
        assert out['homogeneous'] is True
        assert self.KW['homogeneous'] is False            # not modified in place

    def test_default_keeps_trap(self):
        assert _evolution_kwargs({'trap': {}}, self.KW) is self.KW
