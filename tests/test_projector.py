# -*- coding: utf-8 -*-
"""
test_projector.py
Unit tests for projector.py.

Strategy
--------
  TestProjectorAlgebra  — P² = P and P† = P (w.r.t. Σ dv) on random fields,
                          for every geometry, to machine precision.
  TestCfieldRegion      — plane waves inside the region are untouched,
                          outside are removed; spin-dependent shifted windows.
  TestSocDispersion     — the dispersion matches the kinetic + spin-orbit
                          part of gp_equation._H_sp on plane waves.
  TestValidation        — aliasing flag and input errors.
  TestDressedMinimum    — Raman-dressed band minimum against the analytic
                          two-component result; flag for a cutoff that
                          cuts its minority spin.

Run with:
    pytest tests/test_projector.py -v
"""

import numpy as np
from numpy import pi
import pytest

from bec_gp_solver.geometry import GeometryCart, Geometry3DAxial
from bec_gp_solver.projector import (Projector, soc_dispersion, alias_free_cutoff,
                                     dressed_band_minimum, check_dressed_minimum)
from bec_gp_solver.gp_equation import _H_sp


SIGMA_Z = np.array([1.0, -1.0])


def _random_field(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _inner(geo, phi, psi):
    """Grid inner product Σ dv φ* ψ, summed over components."""
    return (geo.dv * phi.conj() * psi).sum()


@pytest.fixture(params=['1d', '2d', '3d', 'axial'])
def geo(request):
    if request.param == '1d':
        return GeometryCart(sizes=(64,), lengths=(20*pi,))
    if request.param == '2d':
        return GeometryCart(sizes=(16, 64), lengths=(4*pi, 20*pi))
    if request.param == '3d':
        return GeometryCart(sizes=(8, 8, 64), lengths=(4*pi, 4*pi, 20*pi))
    return Geometry3DAxial(sizes=(32, 64), lengths=(20.0, 20*pi))


@pytest.fixture
def proj(geo):
    return Projector(geo, soc_dispersion(SIGMA_Z), e_cut=2.0)


# =============================================================================
# Algebraic properties
# =============================================================================

class TestProjectorAlgebra:

    def test_idempotent(self, geo, proj):
        rng = np.random.default_rng(0)
        psi = _random_field(rng, (2, len(geo.dv)))
        p1  = proj(psi)
        assert np.max(np.abs(proj(p1) - p1)) < 1e-12 * np.max(np.abs(p1))

    def test_hermitian(self, geo, proj):
        """⟨φ|Pψ⟩ == ⟨Pφ|ψ⟩ with the grid volume element."""
        rng = np.random.default_rng(1)
        phi = _random_field(rng, (2, len(geo.dv)))
        psi = _random_field(rng, (2, len(geo.dv)))
        lhs = _inner(geo, phi, proj(psi))
        rhs = _inner(geo, proj(phi), psi)
        assert abs(lhs - rhs) < 1e-12 * abs(lhs)

    def test_norm_does_not_increase(self, geo, proj):
        rng = np.random.default_rng(2)
        psi = _random_field(rng, (2, len(geo.dv)))
        n0  = _inner(geo, psi, psi).real
        n1  = _inner(geo, proj(psi), proj(psi)).real
        assert n1 <= n0

    def test_flat_shape_preserved(self, geo, proj):
        psi = np.ones(2 * len(geo.dv), dtype=complex)
        assert proj(psi).shape == psi.shape


# =============================================================================
# The c-field region
# =============================================================================

class TestCfieldRegion:
    """
    1D grid with lz = 20π, so k_z = m/10 and k_z = ±1 (= σ_z) is on the grid.
    With σ_z = [+1, -1] and e_cut = 2:
      component 0 keeps |k_z - 1| < √2,  component 1 keeps |k_z + 1| < √2.
    """

    @pytest.fixture
    def geo1d(self):
        return GeometryCart(sizes=(128,), lengths=(20*pi,))

    @pytest.fixture
    def proj1d(self, geo1d):
        return Projector(geo1d, soc_dispersion(SIGMA_Z), e_cut=2.0)

    def _plane_waves(self, geo1d, k0, k1):
        z, = geo1d.grids
        return np.array([np.exp(1j * k0 * z), np.exp(1j * k1 * z)])

    def test_inside_unchanged(self, geo1d, proj1d):
        psi = self._plane_waves(geo1d, 1.0, -1.0)          # band minima
        assert np.allclose(proj1d(psi), psi, atol=1e-12)

    def test_outside_removed(self, geo1d, proj1d):
        psi = self._plane_waves(geo1d, -1.0, 1.0)          # energy 4 > 2
        assert np.allclose(proj1d(psi), 0.0, atol=1e-12)

    def test_windows_are_shifted_per_component(self, proj1d, geo1d):
        k, = geo1d.spectral_kgrids
        kept0 = k[proj1d.mask[0]]
        kept1 = k[proj1d.mask[1]]
        assert np.allclose(kept0.mean(), 1.0)
        assert np.allclose(kept1.mean(), -1.0)
        assert np.all(np.abs(kept0 - 1.0) < np.sqrt(2.0))
        assert np.all(np.abs(kept1 + 1.0) < np.sqrt(2.0))

    def test_n_modes(self, proj1d):
        # |m/10 ∓ 1| < √2  →  m ∈ (10 ∓ 14.14)  →  29 modes per component
        assert proj1d.n_modes == 2 * 29


# =============================================================================
# Consistency with the Hamiltonian
# =============================================================================

class TestSocDispersion:

    @pytest.mark.parametrize('geo_kind', ['2d', 'axial'])
    def test_matches_H_sp(self, geo_kind):
        """
        On a single spectral mode of component i, the kinetic + spin-orbit
        part of _H_sp acts as multiplication by ε_i(k) from soc_dispersion.
        """
        if geo_kind == '2d':
            geo = GeometryCart(sizes=(16, 64), lengths=(4*pi, 20*pi))
            mode = (3, 7)
            tol  = 1e-10
        else:
            geo = Geometry3DAxial(sizes=(32, 64), lengths=(20.0, 20*pi))
            mode = (2, 7)
            tol  = 1e-6       # kinetic() uses the raw (not orthogonalised) DHT
        n_comp  = len(SIGMA_Z)
        zeros   = np.zeros(n_comp)
        kgrids  = geo.spectral_kgrids
        k_perp2 = sum(k**2 for k in kgrids[:-1])
        eps     = soc_dispersion(SIGMA_Z)(k_perp2, kgrids[-1])

        for i in range(n_comp):
            a = np.zeros((n_comp,) + geo.sizes, dtype=complex)
            a[(i,) + mode] = 1.0
            psi   = geo.from_spectral(a)
            H_psi = _H_sp(psi, geo, SIGMA_Z, zeros, zeros,
                          np.zeros((n_comp, n_comp)))
            expected = eps[(i,) + mode] * psi
            assert np.max(np.abs(H_psi - expected)) < tol * np.max(np.abs(expected))


# =============================================================================
# Validation
# =============================================================================

class TestValidation:

    def test_alias_free_for_small_cutoff(self):
        geo  = GeometryCart(sizes=(128,), lengths=(20*pi,))
        proj = Projector(geo, soc_dispersion(SIGMA_Z), e_cut=2.0)
        assert proj.alias_free

    def test_aliasing_flagged_for_large_cutoff(self, caplog):
        geo  = GeometryCart(sizes=(128,), lengths=(20*pi,))      # k_max = 6.4
        proj = Projector(geo, soc_dispersion(SIGMA_Z), e_cut=25.0)
        assert not proj.alias_free
        assert 'alias' in caplog.text

    def test_empty_region_raises(self):
        geo = GeometryCart(sizes=(64,), lengths=(20*pi,))
        with pytest.raises(ValueError):
            Projector(geo, soc_dispersion(SIGMA_Z), e_cut=0.0)

    def test_bad_dispersion_shape_raises(self):
        geo = GeometryCart(sizes=(64,), lengths=(20*pi,))
        with pytest.raises(ValueError):
            Projector(geo, lambda k_perp2, k_z: k_z**2, e_cut=1.0)


# =============================================================================
# Noise restricted to the c-field
# =============================================================================

class TestNoise:

    def test_white_noise_vanishes_outside(self, geo, proj):
        rng   = np.random.default_rng(3)
        noise = proj.white_noise(rng, 1.0)
        a     = geo.to_spectral(noise)
        assert np.abs(a[~proj.mask]).max() < 1e-12 * np.abs(a[proj.mask]).max()

    def test_white_noise_is_already_projected(self, geo, proj):
        rng   = np.random.default_rng(4)
        noise = proj.white_noise(rng, 1.0)
        assert np.max(np.abs(proj(noise) - noise)) < 1e-12 * np.max(np.abs(noise))

    def test_twa_half_quantum_per_mode(self, geo, proj):
        """⟨|a|²⟩ per c-field mode = 1/(2N), averaged over modes and draws."""
        n_atoms, n_draws = 1e4, 200
        rng = np.random.default_rng(5)
        occ = np.zeros(proj.mask.shape)
        for _ in range(n_draws):
            occ += np.abs(geo.to_spectral(proj.twa_noise(rng, n_atoms)))**2
        mean = n_atoms * occ[proj.mask].mean() / n_draws
        # relative statistical error ≈ 1/√(n_modes · n_draws)
        tol = 5 / np.sqrt(proj.n_modes * n_draws)
        assert abs(mean - 0.5) < 0.5 * tol

    def test_twa_total_atoms(self, geo, proj):
        """Vacuum noise adds n_modes/2 atoms on average: N ∫|δψ|² ≈ n_modes/2."""
        n_atoms = 1e4
        rng     = np.random.default_rng(6)
        added   = np.mean([n_atoms * (geo.dv * np.abs(proj.twa_noise(rng, n_atoms))**2).sum()
                           for _ in range(200)])
        assert abs(added / (proj.n_modes / 2) - 1) < 5 / np.sqrt(proj.n_modes * 200)

    def test_seeded_noise_is_reproducible(self, proj):
        a = proj.twa_noise(np.random.default_rng([7, 1]), 1e4)
        b = proj.twa_noise(np.random.default_rng([7, 1]), 1e4)
        c = proj.twa_noise(np.random.default_rng([7, 2]), 1e4)
        assert np.array_equal(a, b)
        assert not np.allclose(a, c)


# =============================================================================
# Alias-free cutoff
# =============================================================================

class TestAliasFreeCutoff:

    def test_cutoff_is_alias_free(self, geo):
        disp  = soc_dispersion(SIGMA_Z)
        e_cut = alias_free_cutoff(geo, disp)
        assert Projector(geo, disp, e_cut).alias_free

    def test_cutoff_is_the_largest(self, geo, caplog):
        """Any larger cutoff admits a mode beyond k_max/2."""
        disp  = soc_dispersion(SIGMA_Z)
        e_cut = alias_free_cutoff(geo, disp)
        assert not Projector(geo, disp, e_cut * (1 + 1e-9)).alias_free

    def test_1d_value(self):
        """1D, k_max = 6.4, σ_z = ±1: first mode beyond 3.2 is at |k| = 3.3,
        lowest energy (3.3 - 1)² = 5.29."""
        geo = GeometryCart(sizes=(128,), lengths=(20*pi,))
        assert np.isclose(alias_free_cutoff(geo, soc_dispersion(SIGMA_Z)), 2.3**2)

    def test_too_coarse_grid_raises(self):
        """k_z,max = 1.6 puts the band minima k_z = ±1 beyond k_max/2."""
        geo = GeometryCart(sizes=(32,), lengths=(20*pi,))
        with pytest.raises(ValueError, match="alias-free"):
            alias_free_cutoff(geo, soc_dispersion(SIGMA_Z))


# =============================================================================
# Mode-count flag
# =============================================================================

class TestModeCount:

    def test_few_modes_ok(self, proj):
        assert proj.check_mode_count(n_atoms=100 * proj.n_modes)

    def test_too_many_modes_flagged(self, proj, caplog):
        assert not proj.check_mode_count(n_atoms=proj.n_modes)
        assert 'TWA' in caplog.text

    def test_threshold(self, proj):
        from bec_gp_solver.projector import MODE_FRACTION_WARN
        n_edge = proj.n_modes / MODE_FRACTION_WARN
        assert not proj.check_mode_count(n_atoms=n_edge)
        assert proj.check_mode_count(n_atoms=n_edge * 1.01)


# =============================================================================
# Raman-dressed band minimum
# =============================================================================

class TestDressedMinimum:
    """
    Two components, σ_z = [-1, 1], δ = 0: the lower band of
    [[(k+1)², Ω/2], [Ω/2, (k-1)²]] has its minima at k = ±k_l with
    k_l = sqrt(1 - (Ω/4)²) and spin polarisation k_l, i.e. minority
    weight (1 - k_l)/2 and minority bare energy (1 + k_l)².
    """

    SIGMA = np.array([-1.0, 1.0])

    @pytest.mark.parametrize('omega_r', [1.0, 2.7, 3.5])
    def test_analytic_two_component(self, omega_r):
        k_l    = np.sqrt(1 - (omega_r / 4)**2)
        k, w   = dressed_band_minimum(self.SIGMA, omega_r)
        assert abs(abs(k) - k_l) < 2e-3
        assert np.isclose(w.min(), (1 - k_l) / 2, atol=2e-3)
        assert np.isclose(w.sum(), 1.0)

    def test_detuning_moves_minimum(self):
        """δ > 0 lowers component 0 (Δ_0 = -δ/2), centred at k_z = σ_0 = -1."""
        k0, _ = dressed_band_minimum(self.SIGMA, 2.7, delta=0.0)
        k2, w = dressed_band_minimum(self.SIGMA, 2.7, delta=2.0)
        assert k2 < 0 and abs(k2) > abs(k0)
        assert w[0] > w[1]

    def test_flags_small_cutoff(self, caplog):
        """e_cut below (1 + k_l)² ≈ 3.02 cuts the minority spin at ω_R = 2.7."""
        assert not check_dressed_minimum(1.32, self.SIGMA, 2.7)
        assert 'Raman-dressed' in caplog.text

    def test_accepts_large_cutoff(self):
        assert check_dressed_minimum(3.1, self.SIGMA, 2.7)

    def test_checks_every_detuning(self):
        """Fine at δ = 0 with e_cut = 3.1, but δ = 2 needs ≈ 3.6."""
        assert check_dressed_minimum(3.1, self.SIGMA, 2.7, deltas=(0.0,))
        assert not check_dressed_minimum(3.1, self.SIGMA, 2.7, deltas=(0.0, 2.0))

    def test_no_raman_is_always_fine(self):
        """Ω = 0: the minimum is a pure spin state, nothing to cut."""
        assert check_dressed_minimum(0.5, self.SIGMA, 0.0)
