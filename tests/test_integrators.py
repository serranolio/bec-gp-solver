# -*- coding: utf-8 -*-
"""
test_integrators.py
Unit tests for integrators.py.

Strategy
--------
  TestIntegrate  — fourth-order convergence on dψ/dt = -iωψ, frames landing
                   exactly on t_eval, step counting, input validation.
  TestStability  — RK4 stays bounded at dt = RK4_STABILITY/|λ| for real-
                   and imaginary-time eigenvalues; resolve_dt default/check.
  TestRK4IP      — the linear part is exact for any dt, fourth-order
                   convergence of the remainder, LinearPropagator algebra.

Run with:
    pytest tests/test_integrators.py -v
"""

import numpy as np
import pytest

from bec_gp_solver.integrators import (integrate, resolve_dt, RK4_STABILITY,
                                      LinearPropagator)
from bec_gp_solver.geometry import GeometryCart
from bec_gp_solver.projector import mode_energies, soc_dispersion


def _rotation(omega):
    return lambda t, psi: -1j * omega * psi


class TestIntegrate:

    def test_fourth_order_convergence(self):
        omega, T = 2.0, 5.0
        psi0     = np.array([1.0 + 0j])
        exact    = np.exp(-1j * omega * T)
        err = [abs(integrate(_rotation(omega), psi0, [T], dt)[0, -1] - exact)
               for dt in (0.05, 0.025)]
        assert 14 < err[0] / err[1] < 18          # 2⁴ = 16

    def test_output_layout_and_first_frame(self):
        psi0 = np.array([1.0, 2.0j])
        out  = integrate(_rotation(1.0), psi0, np.linspace(0, 1, 5), 0.1)
        assert out.shape == (2, 5)
        assert np.array_equal(out[:, 0], psi0)

    def test_frames_land_on_t_eval(self):
        """Time-dependent rhs dψ/dt = 1 integrates to ψ = t exactly."""
        t_eval = np.array([0.0, 0.3, 0.35, 1.0])
        out    = integrate(lambda t, psi: np.ones_like(psi), np.zeros(1),
                           t_eval, dt=0.1)
        assert np.allclose(out[0].real, t_eval, atol=1e-14)

    def test_step_count(self):
        """Each interval uses ceil(interval/dt) steps, 4 rhs calls per step."""
        calls  = []
        rhs    = lambda t, psi: (calls.append(t), np.zeros_like(psi))[1]
        t_eval = [0.0, 0.25, 1.0]                  # intervals 0.25, 0.75
        integrate(rhs, np.zeros(1), t_eval, dt=0.1)
        assert len(calls) == 4 * (3 + 8)

    def test_steps_never_exceed_dt(self):
        times = []
        rhs   = lambda t, psi: (times.append(t), np.zeros_like(psi))[1]
        integrate(rhs, np.zeros(1), [0.0, 0.37, 1.0], dt=0.1)
        starts = np.array(times[::4])
        assert np.diff(starts).max() <= 0.1 + 1e-12

    def test_nonzero_t0(self):
        out = integrate(lambda t, psi: np.full_like(psi, t), np.zeros(1),
                        [2.0, 3.0], dt=0.1, t0=1.0)
        # ψ(t) = (t² - 1)/2, exact for RK4 on a polynomial of degree ≤ 4
        assert np.allclose(out[0].real, [1.5, 4.0])

    @pytest.mark.parametrize('t_eval', [[1.0, 0.5], [-1.0, 1.0]])
    def test_invalid_t_eval_raises(self, t_eval):
        with pytest.raises(ValueError):
            integrate(_rotation(1.0), np.ones(1), t_eval, dt=0.1)


class TestStability:

    @pytest.mark.parametrize('lam', [-1j, -1.0])   # real time, imaginary time
    def test_bounded_at_stability_limit(self, lam):
        omega = 10.0
        dt    = RK4_STABILITY / omega
        out   = integrate(lambda t, psi: lam * omega * psi, np.ones(1),
                          [1000 * dt], dt)
        assert abs(out[0, -1]) <= 1.0

    def test_unstable_beyond_limit(self):
        omega = 10.0
        dt    = 3.0 / omega                         # beyond 2√2
        out   = integrate(_rotation(omega), np.ones(1), [1000 * dt], dt)
        assert abs(out[0, -1]) > 1e3

    def test_resolve_default(self):
        assert np.isclose(resolve_dt(None, e_max=10.0), 0.5 * RK4_STABILITY / 10)

    def test_resolve_default_accuracy_limited(self):
        """Long runs: (t/dt)(E dt)⁶/72 = damping_tol fixes the default."""
        e_max, t_total, tol = 10.0, 1e4, 0.01
        dt = resolve_dt(None, e_max, t_total=t_total, damping_tol=tol)
        assert dt < 0.5 * RK4_STABILITY / e_max
        assert np.isclose(t_total / dt * (e_max * dt)**6 / 72, tol)

    def test_resolve_warns_above_accuracy_step(self, caplog):
        resolve_dt(0.1, e_max=10.0, t_total=1e4)
        assert 'damping' in caplog.text

    def test_resolve_zero_e_max(self):
        assert resolve_dt(0.3, e_max=0.0) == 0.3
        with pytest.raises(ValueError):
            resolve_dt(None, e_max=0.0)

    def test_resolve_passes_valid(self):
        assert resolve_dt(0.1, e_max=10.0) == 0.1

    def test_resolve_rejects_unstable(self):
        with pytest.raises(ValueError, match="stability"):
            resolve_dt(0.5, e_max=10.0)


class TestRK4IP:

    @pytest.fixture
    def setup(self):
        geo   = GeometryCart(sizes=(128,), lengths=(20 * np.pi,))
        sigma = np.array([1.0, -1.0])
        eps   = mode_energies(geo, soc_dispersion(sigma))
        z,    = geo.grids
        env   = np.exp(-z**2 / 50) * np.exp(1j * 0.7 * z)
        psi0  = np.array([env, 0.5 * env]).reshape(-1)
        return geo, eps, psi0

    def test_propagator_group_property(self, setup):
        geo, eps, psi0 = setup
        D  = LinearPropagator(geo, -1j * eps)
        ab = D(D(psi0, 0.3), 0.5)
        assert np.allclose(ab, D(psi0, 0.8), atol=1e-13)

    def test_linear_part_exact_for_any_dt(self, setup):
        """N = 0: one huge step reproduces exp(-iεt) exactly — no damping."""
        geo, eps, psi0 = setup
        D     = LinearPropagator(geo, -1j * eps)
        T     = 50.0
        out   = integrate(lambda t, psi: np.zeros_like(psi), psi0, [T], dt=T,
                          propagator=D)
        exact = LinearPropagator(geo, -1j * eps)(psi0, T)
        assert np.allclose(out[:, -1], exact, atol=1e-12)

    def test_fourth_order_in_the_remainder(self, setup):
        """N = -i c cos(t) ψ commutes with L: exact phase c·sin(T) on top of exp(-iεT)."""
        geo, eps, psi0 = setup
        c, T  = 3.0, 4.0
        rhs_N = lambda t, psi: -1j * c * np.cos(t) * psi
        exact = LinearPropagator(geo, -1j * eps)(psi0, T) * np.exp(-1j * c * np.sin(T))
        err   = [np.abs(integrate(rhs_N, psi0, [T], dt,
                                  propagator=LinearPropagator(geo, -1j * eps))[:, -1]
                        - exact).max()
                 for dt in (0.1, 0.05)]
        assert 13 < err[0] / err[1] < 19          # 2⁴ = 16
