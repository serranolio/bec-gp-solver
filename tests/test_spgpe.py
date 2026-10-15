# -*- coding: utf-8 -*-
"""
test_spgpe.py
Unit tests for spgpe.py.

Strategy
--------
  TestReservoirRates   — Lerch identity, scalar-BEC limits, equal and
                         unequal scattering lengths.
  TestKernelAndNoise   — ε̃(k) = ℳ/|k|; energy-damping noise is real with
                         per-mode variance 2 k_BT h ε̃(k); growth noise stays
                         in the c-field, with a per-component variance.
  TestExchange         — the general exchange source S_jk reproduces the
                         paper's explicit V_12 (with a_11); Raman drops out;
                         the exchange drift conserves N, moves N_1 - N_2 and
                         lowers the energy.
  TestLimits           — γ = ℳ = 0 reduces to PGPE.
  TestDamping          — at T = 0 number damping lowers E - μN, energy
                         damping lowers E and conserves N, uses each
                         component's own current, and leaves a Raman-dressed
                         ground state alone; with energy noise N is conserved
                         as dt → 0.
  TestEquilibrium      — ideal gas reaches the Rayleigh-Jeans occupations
                         k_BT / (N (ε - μ)) with growth only and with all
                         three processes (fluctuation-dissipation check).
  TestRates            — estimate_max_rate includes |V_j| and |V_jk|.
  TestReservoirConfig  — [reservoir] conversion, first-principles rates,
                         overrides and required keys.

Run with:
    pytest tests/test_spgpe.py -v
"""

import sys
from pathlib import Path

import numpy as np
from numpy import pi
import pytest

from bec_gp_solver.geometry import GeometryCart, Geometry3DAxial
from bec_gp_solver.projector import Projector, soc_dispersion, mode_energies
from bec_gp_solver.gp_equation import get_ip_rhs, estimate_max_energy
from bec_gp_solver.integrators import integrate, LinearPropagator
from bec_gp_solver.spgpe import (get_spgpe_rhs, energy_damping_kernel,
                                 energy_damping_noise, energy_damping_potential,
                                 estimate_max_rate, reservoir_rates, lerch_phi1,
                                 _full_H)

sys.path.insert(0, str(Path(__file__).parent))
from test_gp_equation import _pgpe_setup, _energy, _norm    # noqa: E402


SIGMA = np.array([1.0, -1.0])


def _kicked_setup():
    """_pgpe_setup with a velocity kick, so there is a current to damp."""
    geo, kw, proj, psi = _pgpe_setup()
    z,  = geo.grids
    psi = proj((psi.reshape(2, -1) * np.exp(0.6j * z)).reshape(-1))
    return geo, kw, proj, psi / np.sqrt(_norm(geo, psi))


def _run(geo, kw, proj, psi, T, mu, gamma, M, t_eval, dt, M_x=0.0, seed=1):
    rates, rhs_N, hook = get_spgpe_rhs(geo, proj, np.random.default_rng(seed),
                                       T, mu, gamma, M, M_x, **kw)
    return integrate(rhs_N, psi, t_eval, dt,
                     propagator=LinearPropagator(geo, rates), step_hook=hook)


def _ideal_gas():
    geo  = GeometryCart(sizes=(64,), lengths=(20 * pi,))
    proj = Projector(geo, soc_dispersion(SIGMA), 2.0)
    kw   = dict(n_components=2, sigma_z=SIGMA, detuning=0.0,
                q_zeeman=np.zeros(2), omega_r=0.0,
                g_matrix=np.zeros((2, 2)), n_atoms=1.0)
    return geo, proj, kw


# =============================================================================
# Reservoir rates
# =============================================================================

class TestReservoirRates:

    @pytest.mark.parametrize('z', [0.0, 0.3, 0.9])
    def test_lerch_identity(self, z):
        """Φ(z, 1, 1) = -ln(1 - z)/z, and Φ(z, 1, a) = 1/a + z Φ(z, 1, a + 1)."""
        expected = 1.0 if z == 0 else -np.log1p(-z) / z
        assert np.isclose(lerch_phi1(z, 1.0), expected)
        assert np.isclose(lerch_phi1(z, 2.5), 1 / 2.5 + z * lerch_phi1(z, 3.5))

    def test_n_bar_series(self):
        """Σ_p x^p Φ(x, 1, p) = x / (1 - x): the paper's N̄^s equals N̄."""
        x = np.exp(-1.3)
        p = np.arange(1, 200)
        assert np.isclose((x**p * lerch_phi1(x, p)).sum(), x / (1 - x))

    def test_scalar_limits(self):
        """One component: γ = 8a²Ḡ/λ², ℳ = 16πa² N̄ (Bradley 2008, Rooney 2012)."""
        T, mu, e_cut, a = 0.8, 1.0, 3.0, 0.03
        r     = reservoir_rates(T, mu, e_cut, [[a]])
        n_bar = 1 / np.expm1((e_cut - mu) / T)
        assert np.isclose(r['n_bar'], n_bar)
        assert np.isclose(r['energy_damping'][0], 16 * pi * a**2 * n_bar)
        x, y  = np.exp((mu - e_cut) / T), np.exp((mu - 2 * e_cut) / T)
        rr    = np.arange(200)
        G_bar = x**2 * (y**rr * lerch_phi1(x, rr + 1)**2).sum()
        assert np.isclose(r['gamma'][0], 8 * a**2 * G_bar / (4 * pi / T))
        assert r['exchange_damping'].shape == (1, 1) and r['exchange_damping'][0, 0] == 0

    def test_equal_scattering_lengths(self):
        """ℳ_j = (16π + 4π) a² N̄,  ℳ_12 = 8π a² N̄,  γ_1 = γ_2."""
        a = 0.03
        r = reservoir_rates(1.0, 1.0, 4.0, [[a, a], [a, a]])
        n = r['n_bar']
        assert np.allclose(r['energy_damping'], 20 * pi * a**2 * n)
        assert np.allclose(r['exchange_damping'], 8 * pi * a**2 * n * (1 - np.eye(2)))
        assert np.isclose(r['gamma'][0], r['gamma'][1])

    def test_unequal_scattering_lengths(self):
        a11, a22, a12 = 0.02, 0.04, 0.03
        r = reservoir_rates(1.0, 1.0, 4.0, [[a11, a12], [a12, a22]])
        n = r['n_bar']
        assert np.allclose(r['energy_damping'],
                           [n * (16 * pi * a11**2 + 4 * pi * a12**2),
                            n * (16 * pi * a22**2 + 4 * pi * a12**2)])
        assert np.isclose(r['exchange_damping'][0, 1], 8 * pi * a12**2 * n)
        ratio = r['gamma'][1] / r['gamma'][0]
        assert np.isclose(ratio, (8 * a22**2 + 4 * a12**2) / (8 * a11**2 + 4 * a12**2))

    def test_mu_above_cutoff_raises(self):
        with pytest.raises(ValueError):
            reservoir_rates(1.0, 5.0, 4.0, [[0.03]])


# =============================================================================
# Kernel and noise
# =============================================================================

class TestKernelAndNoise:

    def test_kernel_values(self):
        geo = GeometryCart(sizes=(16, 32), lengths=(5.0, 10.0))
        ker = energy_damping_kernel(geo, 0.3)
        k   = np.sqrt(sum(kk**2 for kk in geo.spectral_kgrids))
        assert ker[0, 0] == 0.0
        nz  = k > 0
        assert np.allclose(ker[nz], 0.3 / k[nz])

    @pytest.mark.parametrize('kind', ['1d', 'axial'])
    def test_energy_noise_spectrum(self, kind):
        """Each spectral mode of dU has variance 2 k_BT h ε̃(k), and dU is real."""
        geo = (GeometryCart(sizes=(64,), lengths=(20.0,)) if kind == '1d'
               else Geometry3DAxial(sizes=(16, 32), lengths=(10.0, 20.0)))
        ker  = energy_damping_kernel(geo, 0.5)
        rng  = np.random.default_rng(0)
        T, h, n = 0.3, 0.01, 4000
        var  = np.zeros(geo.sizes)
        for _ in range(n):
            dU = energy_damping_noise(geo, rng, T, h, np.sqrt(ker))
            assert dU.dtype.kind == 'f'
            var += np.abs(geo.to_spectral(dU))[0]**2
        var /= n
        expected = 2 * T * h * ker
        nz = ker > 0
        ratio = var[nz] / expected[nz]
        assert abs(ratio.mean() - 1) < 0.02
        assert np.all(np.abs(ratio - 1) < 0.25)          # per mode, ~4000 draws

    def test_growth_noise_per_component(self):
        """ψ = 0: rhs_N is pure noise, in the c-field, variance ∝ γ_j per component."""
        geo, proj, kw = _ideal_gas()
        gamma = np.array([0.4, 0.1])
        rates, rhs_N, hook = get_spgpe_rhs(geo, proj, np.random.default_rng(0),
                                           0.5, 0.0, gamma, 0.0, **kw)
        power = np.zeros(2)
        for _ in range(400):
            hook(0.0, 0.01)
            out = rhs_N(0.0, np.zeros(128, complex))
            assert np.max(np.abs(proj(out) - out)) < 1e-12 * np.abs(out).max()
            power += (np.abs(geo.to_spectral(out.reshape(2, -1)))**2
                      * proj.mask).sum(axis=1)
        # per mode: 2 γ_j k_BT h / N, divided by h² for the rate
        expected = 2 * gamma * 0.5 * 0.01 / 0.01**2 * proj.mask.sum(axis=1) * 400
        assert np.allclose(power / expected, 1, atol=0.05)


# =============================================================================
# Particle exchange
# =============================================================================

class TestExchange:

    def _mixture(self, omega_r=0.0):
        geo = GeometryCart(sizes=(128,), lengths=(30.0,))
        z,  = geo.grids
        g   = np.array([[0.04, 0.03], [0.03, 0.06]])
        kw  = dict(n_components=2, sigma_z=np.zeros(2), detuning=0.7,
                   q_zeeman=np.array([0.1, -0.2]), omega_r=omega_r,
                   g_matrix=g, n_atoms=500.0, wz=0.3)
        psi = np.array([np.exp(-(z - 1)**2 / 8 + 0.4j * z),
                        0.6 * np.exp(-(z + 1)**2 / 6 - 0.3j * z + 0.5j)])
        psi = psi / np.sqrt((geo.dv * np.abs(psi)**2).sum())
        return geo, kw, psi

    @staticmethod
    def _S(geo, kw, psi):
        H = _full_H(geo, psi.reshape(-1), **kw)
        return np.imag(psi[1].conj() * H[0]) + np.imag(psi[0].conj() * H[1])

    def test_matches_paper_explicit_form(self):
        """
        Bradley & Blakie Eqs. 157-159: S = -[∇·J^R_12 + (φ2*φ1 - c.c.) V_12 / 2i],
        V_12 = V_2 - V_1 + g22|φ2|² - g12|φ2|² + (g12 - g11)|φ1|² — with a_11;
        the printed a_22 in the last term does not match.
        """
        geo, kw, psi = self._mixture()
        g, N   = kw['g_matrix'], kw['n_atoms']
        z,     = geo.grids
        k,     = geo.kgrids
        d      = lambda f: np.fft.ifft(1j * k * np.fft.fft(f))
        J12    = 1j * (d(psi[1].conj()) * psi[0] - psi[1].conj() * d(psi[0]))
        div_JR = np.real(d(J12 + J12.conj())) / 2
        V_eff  = lambda j: ((2 * j - 1) * kw['detuning'] / 2 + kw['q_zeeman'][j]
                            + (kw['wz'] * z)**2 / 4)
        S      = self._S(geo, kw, psi)

        def mismatch(a_last):
            V12 = (V_eff(1) - V_eff(0) + N * (g[1, 1] - g[0, 1]) * np.abs(psi[1])**2
                   + N * (g[0, 1] - a_last) * np.abs(psi[0])**2)
            bracket = div_JR + ((psi[1].conj() * psi[0] - psi[1] * psi[0].conj())
                                * V12 / 2j).real
            return np.abs(bracket + S).max() / np.abs(S).max()

        assert mismatch(g[0, 0]) < 1e-9
        assert mismatch(g[1, 1]) > 0.1

    def test_raman_drops_out(self):
        geo, kw, psi = self._mixture()
        S0 = self._S(geo, kw, psi)
        S1 = self._S(geo, dict(kw, omega_r=2.7), psi)
        assert np.abs(S1 - S0).max() < 1e-12 * np.abs(S0).max()

    def test_drift_conserves_N_moves_difference_lowers_energy(self):
        geo, kw, proj, psi = _pgpe_setup()
        kw  = dict(kw, omega_r=0.0)
        z,  = geo.grids
        p2  = psi.reshape(2, -1).copy()
        p2[1] *= 1.5 * np.exp(0.8j * z)
        psi = proj(p2.reshape(-1))
        psi /= np.sqrt(_norm(geo, psi))
        out = _run(geo, kw, proj, psi, 0.0, 0.0, 0.0, 0.0,
                   np.linspace(0, 10, 6), 0.0025, M_x=0.5)
        E   = [_energy(geo, out[:, f], kw) for f in range(out.shape[1])]
        n   = [(geo.dv * np.abs(out[:, f].reshape(2, -1))**2).sum(axis=1)
               for f in range(out.shape[1])]
        assert np.all(np.diff(E) < 0)
        assert max(abs(nf.sum() - 1) for nf in n) < 1e-6
        assert abs((n[-1][0] - n[-1][1]) - (n[0][0] - n[0][1])) > 0.1


# =============================================================================
# Limits
# =============================================================================

class TestLimits:

    def test_no_damping_is_pgpe(self):
        geo, kw, proj, psi = _pgpe_setup()
        r_s, n_s, hook = get_spgpe_rhs(geo, proj, np.random.default_rng(0),
                                       0.5, 0.3, 0.0, 0.0, 0.0, **kw)
        r_p, n_p       = get_ip_rhs(geo=geo, mode='real', projector=proj, **kw)
        hook(0.0, 0.01)                  # γ = ℳ = 0: no noise either
        assert np.array_equal(r_s, r_p)
        assert np.allclose(n_s(0.0, psi), n_p(0.0, psi), atol=1e-14)


# =============================================================================
# Damping at T = 0 and conservation laws
# =============================================================================

class TestDamping:

    def test_number_damping_lowers_grand_potential(self):
        geo, kw, proj, psi = _pgpe_setup()
        mu  = 1.0
        out = _run(geo, kw, proj, psi, 0.0, mu, 0.05, 0.0, np.linspace(0, 10, 11), 0.01)
        K = [_energy(geo, out[:, f], kw) - mu * _norm(geo, out[:, f])
             for f in range(out.shape[1])]
        assert np.all(np.diff(K) < 0)

    def test_energy_damping_lowers_energy_and_keeps_norm(self):
        geo, kw, proj, psi = _kicked_setup()
        out = _run(geo, kw, proj, psi, 0.0, 0.0, 0.0, 0.5, np.linspace(0, 10, 11), 0.0025)
        E = [_energy(geo, out[:, f], kw) for f in range(out.shape[1])]
        assert np.all(np.diff(E) < 0)
        assert E[-1] < 0.8 * E[0]
        assert max(abs(_norm(geo, out[:, f]) - 1) for f in range(out.shape[1])) < 1e-5

    def test_energy_damping_uses_own_current(self):
        """
        Without Raman coupling V_j = -N ∫ε ∇·j_j: a component with no current
        gets no energy-damping potential from the other component's current.
        """
        geo, kw, proj, psi = _pgpe_setup()
        kw  = dict(kw, omega_r=0.0)
        z,  = geo.grids
        env = np.exp(-z**2 / (2 * 5.0**2))
        p2  = proj(np.array([env * np.exp(0.6j * z),       # moving
                             env * np.exp(-1j * z)]))      # at its band minimum
        V = energy_damping_potential(geo, p2, proj, 0.5, **kw)
        assert np.abs(V[0]).max() > 1e-3
        # component 1: envelope times e^{iσ_z k_z}, so its kinetic current vanishes
        assert np.abs(V[1]).max() < 1e-6 * np.abs(V[0]).max()

    def test_energy_damping_vanishes_on_dressed_ground_state(self):
        """
        With Raman coupling the dressed ground state has ∇·j_j ≠ 0 per
        component, balanced by Raman conversion; V_j uses the density rate
        under PH, so it vanishes there (up to the ground-state residual),
        while the kinetic-current form does not.
        """
        geo, kw, proj, psi = _pgpe_setup()
        kw = dict(kw, omega_r=2.7)
        rates, rhs_N = get_ip_rhs(geo=geo, mode='imaginary', projector=proj, **kw)
        gs = integrate(rhs_N, psi, [3000.0], 0.02,       # weak trap: slow to converge
                       propagator=LinearPropagator(geo, rates))[:, -1]
        gs = gs / np.sqrt(_norm(geo, gs))
        V  = energy_damping_potential(geo, gs, proj, 0.5, **kw)

        eps   = mode_energies(geo, soc_dispersion(kw['sigma_z']))
        gs2   = gs.reshape(2, -1)
        K_psi = geo.from_spectral(eps * geo.to_spectral(gs2))
        V_kin = -kw['n_atoms'] * 0.5 * np.array(
            [np.fft.ifft(np.fft.fft(-2 * np.imag(gs2[j].conj() * K_psi[j]))
                         * energy_damping_kernel(geo, 1.0)).real for j in range(2)])
        assert np.abs(V_kin).max() > 1e-2
        assert np.abs(V).max() < 1e-3 * np.abs(V_kin).max()

    def test_energy_noise_conserves_norm_as_dt_shrinks(self):
        """i ψ dU is a phase (Stratonovich): the norm error is integration error."""
        geo, kw, proj, psi = _kicked_setup()
        err = [abs(_norm(geo, _run(geo, kw, proj, psi, 0.05, 0.0, 0.0, 0.5,
                                   [5.0], dt)[:, -1]) - 1)
               for dt in (0.01, 0.005)]
        assert err[1] < err[0] / 4


# =============================================================================
# Thermal equilibrium
# =============================================================================

class TestEquilibrium:
    """
    Ideal gas (no interactions, potentials or Raman): every process must
    leave the grand-canonical state invariant, so the occupations reach
    ⟨|a_k|²⟩ = k_BT / (N (ε_k - μ)) whatever the damping rates.
    """

    T, MU = 0.5, -0.5

    def _occupations(self, gamma, M, M_x, t_end, dt):
        geo, proj, kw = _ideal_gas()
        t_eval = np.linspace(0, t_end, int(t_end / 0.5) + 1)
        out    = _run(geo, kw, proj, np.zeros(128, complex), self.T, self.MU,
                      gamma, M, t_eval, dt, M_x=M_x, seed=0)
        frames = range(len(t_eval) // 5, len(t_eval))     # skip relaxation
        occ    = np.mean([np.abs(geo.to_spectral(out[:, f].reshape(2, -1)))**2
                          for f in frames], axis=0)
        eps    = mode_energies(geo, soc_dispersion(SIGMA))
        self.e = eps[proj.mask]
        return (occ / (self.T / (eps - self.MU)))[proj.mask], out, proj, geo

    def _thirds(self, ratio):
        lo = ratio[self.e < np.percentile(self.e, 33)].mean()
        hi = ratio[self.e > np.percentile(self.e, 67)].mean()
        return lo, hi

    def test_rayleigh_jeans_growth(self):
        ratio, out, proj, geo = self._occupations([0.2, 0.05], 0.0, 0.0, 1000, 0.05)
        assert abs(ratio.mean() - 1) < 0.03
        assert np.abs(geo.to_spectral(out[:, -1].reshape(2, -1))[~proj.mask]).max() < 1e-12

    def test_rayleigh_jeans_all_processes(self):
        """
        Weak growth, so energy damping and particle exchange set the shape of
        the distribution (they conserve N): with a wrong noise strength the
        low- and high-energy modes deviate in opposite directions (measured:
        0.70 / 1.27 for an exchange noise twice too strong), even when the
        mean stays close to 1.
        """
        ratio, *_ = self._occupations(0.02, [0.3, 0.1], 1.0, 1000, 0.02)
        lo, hi = self._thirds(ratio)
        assert abs(ratio.mean() - 1) < 0.05
        assert abs(lo - 1) < 0.08 and abs(hi - 1) < 0.08


# =============================================================================
# Step-size estimate
# =============================================================================

class TestRates:

    def test_includes_energy_damping_potential_and_stiffness(self):
        geo, kw, proj, psi = _kicked_setup()
        base   = estimate_max_rate(geo, psi, (0.0,), proj, 0.0, 0.0, 0.0, **kw)
        with_M = estimate_max_rate(geo, psi, (0.0,), proj, 0.0, 0.0, 0.5, **kw)
        V      = energy_damping_potential(geo, psi, proj, 0.5, **kw)
        e_full = estimate_max_energy(geo, psi, (0.0,), proj, **kw)
        stiff  = (2 * kw['n_atoms'] * energy_damping_kernel(geo, 1.0).max()
                  * (np.abs(psi)**2).max() * e_full * 0.5)
        assert np.isclose(with_M - base, np.abs(V).max() + stiff)

    def test_stiffness_counts_on_a_ground_state(self):
        """V ≈ 0 on a stationary state, but the step must still be limited."""
        geo, kw, proj, psi = _pgpe_setup()
        rates, rhs_N = get_ip_rhs(geo=geo, mode='imaginary', projector=proj, **kw)
        gs = integrate(rhs_N, psi, [200.0], 0.02,
                       propagator=LinearPropagator(geo, rates))[:, -1]
        gs = gs / np.sqrt(_norm(geo, gs))
        base   = estimate_max_rate(geo, gs, (0.0,), proj, 0.0, 0.0, 0.0, **kw)
        with_M = estimate_max_rate(geo, gs, (0.0,), proj, 0.0, 0.0, 0.5, **kw)
        V      = energy_damping_potential(geo, gs, proj, 0.5, **kw)
        assert with_M - base > 10 * np.abs(V).max()

    def test_includes_exchange_potential(self):
        geo, kw, proj, psi = _kicked_setup()
        base   = estimate_max_rate(geo, psi, (0.0,), proj, 0.0, 0.0, 0.0, 0.0, **kw)
        with_X = estimate_max_rate(geo, psi, (0.0,), proj, 0.0, 0.0, 0.0, 0.5, **kw)
        assert with_X > base

    def test_number_damping_scaling(self):
        geo, kw, proj, psi = _pgpe_setup()
        e0 = estimate_max_rate(geo, psi, (0.0,), proj, 0.0, 0.0, 0.0, **kw)
        e1 = estimate_max_rate(geo, psi, (0.0,), proj, 0.5, 2.0, 0.0, **kw)
        assert np.isclose(e1, np.sqrt(1.25) * e0 + 0.5 * 2.0)


# =============================================================================
# [reservoir] config section (run_simulation._reservoir)
# =============================================================================

class TestReservoirConfig:

    D = {'f_recoil': 1960.0, 't_unit': 1e-4, 'l_unit': 1.7225e-7}

    @pytest.fixture
    def reservoir_fn(self):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from run_simulation import _reservoir
        return _reservoir

    def _cfg(self, a=((100.4, 100.4), (100.4, 100.4)), e_cut=5.0, kind='3d_axial', **res):
        return {'constants': {'hbar_si': 1.054571800139113e-34,
                              'a_si': 5.2917721067e-11},
                'system': {'a_matrix': [list(r) for r in a]},
                'geometry': {'kind': kind},
                'projector': {'e_cut': e_cut},
                'reservoir': res}

    def test_reduced_dimensions_need_number_damping_only(self, reservoir_fn):
        with pytest.raises(ValueError, match="3D grids"):
            reservoir_fn(self._cfg(kind='1d_cart', temperature_nk=100.0, mu_hz=0.0), self.D)
        res = reservoir_fn(self._cfg(kind='1d_cart', temperature_nk=100.0, mu_hz=0.0,
                                     energy_damping=0.0, exchange_damping=0.0), self.D)
        assert np.all(res['gamma'] > 0)

    def test_unit_conversion(self, reservoir_fn):
        res = reservoir_fn(self._cfg(temperature_nk=50.0, mu_hz=980.0,
                                     equilibrate_ms=2.0), self.D)
        kT  = 1.380649e-23 * 50e-9 / (2 * pi * 1.054571800139113e-34 * 1960.0)
        assert np.isclose(res['temperature'], kT)          # ≈ 0.53 E_R
        assert np.isclose(res['mu'], 0.5)
        assert np.isclose(res['t_equilibrate'], 20.0)

    def test_first_principles_rates(self, reservoir_fn):
        a   = ((100.0, 90.0), (90.0, 110.0))
        res = reservoir_fn(self._cfg(a=a, temperature_nk=100.0, mu_hz=2441.0), self.D)
        a_r = np.array(a) * 5.2917721067e-11 / self.D['l_unit']
        ref = reservoir_rates(res['temperature'], res['mu'], 5.0, a_r)
        for key in ('gamma', 'energy_damping', 'exchange_damping'):
            assert np.allclose(res[key], ref[key])
        assert res['energy_damping'][1] > res['energy_damping'][0]   # a_22 > a_11

    def test_overrides(self, reservoir_fn):
        res = reservoir_fn(self._cfg(temperature_nk=50.0, mu_hz=0.0, gamma=[0.1, 0.2],
                                     energy_damping=0.3, exchange_damping=0.0), self.D)
        assert np.allclose(res['gamma'], [0.1, 0.2])
        assert res['energy_damping'] == 0.3 and res['exchange_damping'] == 0.0

    def test_equilibration_optional(self, reservoir_fn):
        res = reservoir_fn(self._cfg(temperature_nk=50.0, mu_hz=0.0), self.D)
        assert res['t_equilibrate'] == 0.0

    def test_missing_keys_raise(self, reservoir_fn):
        with pytest.raises(ValueError, match="mu_hz"):
            reservoir_fn(self._cfg(temperature_nk=50.0), self.D)
