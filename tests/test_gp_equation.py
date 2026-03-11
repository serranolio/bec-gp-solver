# -*- coding: utf-8 -*-
"""
test_gp_equation.py
Unit tests for gp_equation.py.

Strategy
--------
Tests are grouped by function/concern:

  TestBuildDetuningDiagonal — static vs callable output, shape and values
  TestBuildRabiMatrix        — shape, tridiagonal structure, scaling, symmetry
  TestBuildTrapPotential     — zero trap, 1D formula, 2D formula, 3d_axial
  TestBuildLatticePotential  — zero, static profile, callable scaling
  TestHPotential             — zero potential, scaling, shape, per-component
  TestHInteraction           — zero g, zero n_atoms, scaling, single-component
  TestHSp                    — kinetic-only limit, diagonal terms, Rabi coupling
  TestGetRhs                 — ValueError, callable, shape, real vs imaginary mode,
                               time-dependent parameters, eigenstate properties

Run with:
    pytest tests/test_gp_equation.py -v
"""

import numpy as np
from numpy import pi
import pytest

from bec_gp_solver.geometry import GeometryCart, Geometry3DAxial
from bec_gp_solver.gp_equation import (
    _build_detuning_diagonal,
    _build_rabi_matrix,
    _build_trap_potential,
    _build_lattice_potential,
    _H_potential,
    _H_interaction,
    _H_sp,
    get_rhs,
)


# =============================================================================
# Helpers
# =============================================================================

def _geo1d(nz=128, lz=10.0):
    return GeometryCart(sizes=(nz,), lengths=(lz,))


def _geo2d(nx=32, nz=64, lx=6.0, lz=10.0):
    return GeometryCart(sizes=(nx, nz), lengths=(lx, lz))


# =============================================================================
# _build_detuning_diagonal
# =============================================================================

class TestBuildDetuningDiagonal:

    def test_single_component_static(self):
        # Δ_0 = (2*0 - 1) * δ/2 = -δ/2
        result = _build_detuning_diagonal(1, 4.0)
        np.testing.assert_allclose(result, [-2.0])

    def test_three_components_static(self):
        # i=0,1,2: (2i - 1) * δ/2 → [-1, 1, 3] for δ=2
        result = _build_detuning_diagonal(3, 2.0)
        np.testing.assert_allclose(result, [-1.0, 1.0, 3.0])

    def test_zero_detuning(self):
        result = _build_detuning_diagonal(5, 0.0)
        np.testing.assert_allclose(result, np.zeros(5))

    def test_static_not_callable(self):
        result = _build_detuning_diagonal(3, 2.0)
        assert not callable(result)

    def test_callable_returns_callable(self):
        result = _build_detuning_diagonal(3, lambda t: t)
        assert callable(result)

    def test_callable_values(self):
        # detuning(t) = 2*t → Δ_i(t) = (2i-1) * t
        delta_fn = _build_detuning_diagonal(3, lambda t: 2.0 * t)
        np.testing.assert_allclose(delta_fn(1.0), [-1.0, 1.0, 3.0])
        np.testing.assert_allclose(delta_fn(0.5), [-0.5, 0.5, 1.5])

    def test_callable_zero_at_zero(self):
        delta_fn = _build_detuning_diagonal(4, lambda t: t)
        np.testing.assert_allclose(delta_fn(0.0), np.zeros(4))

    def test_output_length_matches_n_components(self):
        for n in [1, 2, 3, 5]:
            result = _build_detuning_diagonal(n, 1.0)
            assert len(result) == n


# =============================================================================
# _build_rabi_matrix
# =============================================================================

class TestBuildRabiMatrix:

    def test_shape(self):
        assert _build_rabi_matrix(4, 1.0).shape == (4, 4)

    def test_single_component_is_zero(self):
        # Only one component → no neighbours → zero matrix
        Omega = _build_rabi_matrix(1, 2.0)
        np.testing.assert_allclose(Omega, np.zeros((1, 1)))

    def test_tridiagonal_structure(self):
        Omega = _build_rabi_matrix(4, 1.0)
        expected = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ], dtype=float)
        np.testing.assert_allclose(Omega, expected)

    def test_symmetric(self):
        Omega = _build_rabi_matrix(5, 3.7)
        np.testing.assert_allclose(Omega, Omega.T)

    def test_diagonal_is_zero(self):
        Omega = _build_rabi_matrix(5, 1.0)
        np.testing.assert_allclose(np.diag(Omega), 0.0)

    def test_scaling_with_omega_r(self):
        Omega1 = _build_rabi_matrix(4, 1.0)
        Omega2 = _build_rabi_matrix(4, 2.5)
        np.testing.assert_allclose(Omega2, 2.5 * Omega1)

    def test_zero_omega_r(self):
        Omega = _build_rabi_matrix(4, 0.0)
        np.testing.assert_allclose(Omega, np.zeros((4, 4)))


# =============================================================================
# _build_trap_potential
# =============================================================================

class TestBuildTrapPotential:

    def test_zero_frequencies_gives_zero(self):
        geo = _geo1d()
        V = _build_trap_potential(geo, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(V, 0.0)

    def test_flat_output_shape_1d(self):
        nz = 64
        geo = GeometryCart(sizes=(nz,), lengths=(10.0,))
        V = _build_trap_potential(geo, 0.0, 0.0, 1.0)
        assert V.shape == (nz,)

    def test_flat_output_shape_2d(self):
        geo = _geo2d(nx=32, nz=64)
        V = _build_trap_potential(geo, 1.0, 0.0, 1.0)
        assert V.shape == (32 * 64,)

    def test_1d_harmonic_formula(self):
        nz, lz = 128, 10.0
        geo = GeometryCart(sizes=(nz,), lengths=(lz,))
        wz = 1.5
        V = _build_trap_potential(geo, 0.0, 0.0, wz)
        z, = geo.grids
        expected = (wz * z.reshape(-1))**2 / 4
        np.testing.assert_allclose(V, expected)

    def test_2d_cart_formula(self):
        # For 2D Cartesian (ndim=2), active_freqs = [wx, wy, wz][-2:] = [wy, wz].
        # The first grid axis (x) gets frequency wy; the last axis (z) gets wz.
        geo = _geo2d(nx=32, nz=64, lx=6.0, lz=10.0)
        wy, wz = 2.0, 1.5
        V = _build_trap_potential(geo, 0.0, wy, wz)
        x, z = geo.grids
        expected = ((wy * x)**2 + (wz * z)**2) / 4
        np.testing.assert_allclose(V, expected.reshape(-1))

    def test_3d_axial_uses_geometric_mean(self):
        """In 3d_axial geometry, radial frequency becomes sqrt(wx * wy)."""
        geo = Geometry3DAxial(sizes=(16, 32), lengths=(10.0, 20.0))
        wx, wy, wz = 2.0, 8.0, 1.5
        V = _build_trap_potential(geo, wx, wy, wz)
        r, z = geo.grids
        w_eff = np.sqrt(wx * wy)  # = 4.0
        expected = ((w_eff * r)**2 + (wz * z)**2) / 4
        np.testing.assert_allclose(V, expected.reshape(-1))

    def test_non_negative(self):
        """Harmonic trap potential is always ≥ 0."""
        geo = _geo2d()
        V = _build_trap_potential(geo, 1.0, 0.0, 2.0)
        assert np.all(V >= 0)


# =============================================================================
# _build_lattice_potential
# =============================================================================

class TestBuildLatticePotential:

    def test_zero_strength_gives_zero(self):
        geo = _geo1d()
        V = _build_lattice_potential(geo, 0.0, 1.0)
        np.testing.assert_allclose(V, 0.0)

    def test_flat_output_shape(self):
        geo = _geo1d(nz=64)
        V = _build_lattice_potential(geo, 1.0, pi)
        assert V.shape == (64,)

    def test_static_formula(self):
        nz, lz = 128, 10.0
        k_l = 2.0
        strength = 3.0
        geo = GeometryCart(sizes=(nz,), lengths=(lz,))
        V = _build_lattice_potential(geo, strength, k_l)
        z, = geo.grids
        expected = strength * 2 * np.sin(k_l * z.reshape(-1))**2
        np.testing.assert_allclose(V, expected)

    def test_static_not_callable(self):
        V = _build_lattice_potential(_geo1d(), 1.0, 1.0)
        assert not callable(V)

    def test_callable_returns_callable(self):
        V_fn = _build_lattice_potential(_geo1d(), lambda t: t, 1.0)
        assert callable(V_fn)

    def test_callable_scales_linearly_with_strength(self):
        nz, lz, k_l = 64, 10.0, 1.0
        geo = GeometryCart(sizes=(nz,), lengths=(lz,))
        V_fn = _build_lattice_potential(geo, lambda t: t, k_l)
        z, = geo.grids
        profile = 2 * np.sin(k_l * z.reshape(-1))**2
        np.testing.assert_allclose(V_fn(0.0), 0.0 * profile)
        np.testing.assert_allclose(V_fn(2.0), 2.0 * profile)
        np.testing.assert_allclose(V_fn(5.0), 5.0 * profile)

    def test_non_negative(self):
        """2 * sin² ≥ 0, so V ≥ 0 for positive strength."""
        geo = _geo1d()
        V = _build_lattice_potential(geo, 1.0, 1.5)
        assert np.all(V >= -1e-15)


# =============================================================================
# _H_potential
# =============================================================================

class TestHPotential:

    def test_zero_potential(self):
        psi = np.ones((3, 64), dtype=complex)
        result = _H_potential(psi, np.zeros(64))
        np.testing.assert_allclose(result, 0.0)

    def test_uniform_potential(self):
        rng = np.random.default_rng(0)
        psi = rng.standard_normal((3, 64)) + 1j * rng.standard_normal((3, 64))
        V = 5.0 * np.ones(64)
        np.testing.assert_allclose(_H_potential(psi, V), 5.0 * psi)

    def test_shape_preserved(self):
        psi = np.ones((4, 100), dtype=complex)
        assert _H_potential(psi, np.ones(100)).shape == psi.shape

    def test_components_see_same_potential(self):
        """Each component is multiplied by the same potential profile."""
        V = np.array([1.0, 2.0, 3.0])
        psi = np.array([[1.0 + 0j, 1.0 + 0j, 1.0 + 0j],
                        [2.0 + 0j, 2.0 + 0j, 2.0 + 0j]])
        result = _H_potential(psi, V)
        np.testing.assert_allclose(result[0], V)       # row 0: 1 * V
        np.testing.assert_allclose(result[1], 2.0 * V) # row 1: 2 * V

    def test_linearity_in_psi(self):
        rng = np.random.default_rng(1)
        psi1 = rng.standard_normal((2, 50)) + 1j * rng.standard_normal((2, 50))
        psi2 = rng.standard_normal((2, 50)) + 1j * rng.standard_normal((2, 50))
        V = rng.standard_normal(50)**2  # non-negative
        a, b = 1.3 + 0.7j, -0.4 + 1.1j
        lhs = _H_potential(a * psi1 + b * psi2, V)
        rhs = a * _H_potential(psi1, V) + b * _H_potential(psi2, V)
        np.testing.assert_allclose(lhs, rhs)


# =============================================================================
# _H_interaction
# =============================================================================

class TestHInteraction:

    def test_zero_g_matrix(self):
        rng = np.random.default_rng(2)
        psi = rng.standard_normal((3, 64)) + 1j * rng.standard_normal((3, 64))
        np.testing.assert_allclose(_H_interaction(psi, np.zeros((3, 3)), 1000.0), 0.0)

    def test_zero_n_atoms(self):
        rng = np.random.default_rng(3)
        psi = rng.standard_normal((3, 64)) + 1j * rng.standard_normal((3, 64))
        np.testing.assert_allclose(_H_interaction(psi, np.ones((3, 3)), 0.0), 0.0)

    def test_scales_linearly_with_n_atoms(self):
        rng = np.random.default_rng(4)
        psi = rng.standard_normal((3, 32)) + 1j * rng.standard_normal((3, 32))
        g = np.eye(3)
        r1 = _H_interaction(psi, g, 1.0)
        r2 = _H_interaction(psi, g, 2.5)
        np.testing.assert_allclose(r2, 2.5 * r1)

    def test_shape_preserved(self):
        psi = np.ones((3, 64), dtype=complex)
        assert _H_interaction(psi, np.ones((3, 3)), 100.0).shape == psi.shape

    def test_single_component_formula(self):
        """H_int = n_atoms * g * |psi|² * psi for single component."""
        psi = np.array([[1.0 + 0j, 2.0 + 0j, 3.0 + 0j]])
        g = np.array([[2.0]])
        n = 5.0
        result = _H_interaction(psi, g, n)
        expected = n * g[0, 0] * np.abs(psi)**2 * psi
        np.testing.assert_allclose(result, expected)

    def test_diagonal_g_decouples_components(self):
        """With diagonal g, component i only feels its own density."""
        rng = np.random.default_rng(5)
        psi = rng.standard_normal((3, 32)) + 1j * rng.standard_normal((3, 32))
        g_vals = np.array([1.0, 2.0, 3.0])
        g = np.diag(g_vals)
        result = _H_interaction(psi, g, 1.0)
        for i in range(3):
            expected_i = g_vals[i] * np.abs(psi[i])**2 * psi[i]
            np.testing.assert_allclose(result[i], expected_i)

    def test_real_psi_stays_real(self):
        """For real psi and real g, H_int should be real."""
        psi = np.array([[1.0, 2.0, 3.0]], dtype=complex)
        result = _H_interaction(psi, np.array([[1.0]]), 1.0)
        np.testing.assert_allclose(result.imag, 0.0, atol=1e-15)


# =============================================================================
# _H_sp
# =============================================================================

class TestHSp:

    @pytest.fixture
    def geo1d(self):
        return _geo1d(nz=256, lz=10.0)

    def test_kinetic_only_plane_wave(self, geo1d):
        """sigma_z=0, delta=0, q=0, Omega=0 → H_sp = kinetic(psi) only."""
        nz, lz = 256, 10.0
        kz = 2 * pi / lz * 4
        z, = geo1d.grids
        psi = np.array([np.exp(1j * kz * z).reshape(-1)])

        result = _H_sp(psi, geo1d,
                       sigma_z=np.array([0.0]),
                       delta=np.array([0.0]),
                       q_zeeman=np.array([0.0]),
                       Omega=np.zeros((1, 1)))

        expected = geo1d.kinetic(psi)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_diagonal_terms_uniform_psi(self, geo1d):
        """
        For uniform psi, kinetic and SOC terms are zero.
        Result = (sigma_z² + delta + q) * psi.
        """
        nz = 256
        psi = np.ones((3, nz), dtype=complex)
        sigma_z = np.array([1.0, -1.0, 2.0])
        delta    = np.array([0.5, -0.3, 0.1])
        q_zeeman = np.array([0.2,  0.1, 0.3])
        Omega    = np.zeros((3, 3))

        result = _H_sp(psi, geo1d, sigma_z, delta, q_zeeman, Omega)

        expected_coeff = sigma_z**2 + delta + q_zeeman
        for i in range(3):
            np.testing.assert_allclose(
                result[i].real, expected_coeff[i] * np.ones(nz), atol=1e-10
            )
            np.testing.assert_allclose(result[i].imag, 0.0, atol=1e-10)

    def test_rabi_coupling_uniform_psi(self, geo1d):
        """
        For uniform psi with sigma_z=0, delta=0, q=0:
        H_sp = 0.5 * Omega @ psi  (kinetic and SOC are zero for uniform psi).
        """
        nz = 256
        psi = np.ones((3, nz), dtype=complex)
        sigma_z  = np.zeros(3)
        delta    = np.zeros(3)
        q_zeeman = np.zeros(3)
        Omega    = _build_rabi_matrix(3, 1.0)

        result = _H_sp(psi, geo1d, sigma_z, delta, q_zeeman, Omega)

        # For uniform psi: each component of 0.5*(Omega@psi)[i] = 0.5 * sum_j Omega[i,j]
        expected = 0.5 * (Omega @ np.ones((3, nz)))
        np.testing.assert_allclose(result.real, expected, atol=1e-10)

    def test_shape_preserved(self, geo1d):
        psi = np.ones((2, 256), dtype=complex)
        result = _H_sp(psi, geo1d,
                       sigma_z=np.array([1.0, -1.0]),
                       delta=np.array([0.0, 0.0]),
                       q_zeeman=np.array([0.0, 0.0]),
                       Omega=np.zeros((2, 2)))
        assert result.shape == psi.shape

    def test_single_component_zero_all(self, geo1d):
        """All parameters zero → H_sp psi = kinetic(psi)."""
        nz = 256
        rng = np.random.default_rng(6)
        psi = (rng.standard_normal((1, nz)) + 1j * rng.standard_normal((1, nz)))
        result = _H_sp(psi, geo1d,
                       sigma_z=np.array([0.0]),
                       delta=np.array([0.0]),
                       q_zeeman=np.array([0.0]),
                       Omega=np.zeros((1, 1)))
        np.testing.assert_allclose(result, geo1d.kinetic(psi), atol=1e-10)


# =============================================================================
# get_rhs
# =============================================================================

class TestGetRhs:

    # ------------------------------------------------------------------
    # Basic API
    # ------------------------------------------------------------------

    def test_invalid_mode_raises_value_error(self):
        geo = _geo1d()
        with pytest.raises(ValueError, match="mode"):
            get_rhs(geo, 1, [0.0], 0.0, [0.0], 0.0, [[1.0]], 1.0, mode='invalid')

    def test_returns_callable(self):
        geo = _geo1d()
        rhs = get_rhs(geo, 1, [0.0], 0.0, [0.0], 0.0, [[1.0]], 1.0)
        assert callable(rhs)

    def test_real_mode_is_default(self):
        """Default mode should be 'real' and not raise."""
        geo = _geo1d(nz=32)
        rhs = get_rhs(geo, 1, [0.0], 0.0, [0.0], 0.0, [[0.0]], 0.0)
        psi = np.ones(32, dtype=complex)
        result = rhs(0.0, psi)
        assert result.shape == psi.shape

    # ------------------------------------------------------------------
    # Output shape
    # ------------------------------------------------------------------

    def test_real_mode_output_shape_1d(self):
        nz = 64
        geo = _geo1d(nz=nz)
        rhs = get_rhs(geo, 2, [1.0, -1.0], 0.0, [0.0, 0.0], 1.0,
                      np.eye(2), 1000.0)
        psi = np.ones(2 * nz, dtype=complex)
        assert rhs(0.0, psi).shape == psi.shape

    def test_imaginary_mode_output_shape_1d(self):
        nz = 64
        geo = _geo1d(nz=nz)
        rhs = get_rhs(geo, 2, [1.0, -1.0], 0.0, [0.0, 0.0], 1.0,
                      np.eye(2), 1000.0, mode='imaginary')
        psi = np.ones(2 * nz, dtype=complex)
        assert rhs(0.0, psi).shape == psi.shape

    # ------------------------------------------------------------------
    # Real-time mode properties
    # ------------------------------------------------------------------

    def test_real_mode_free_particle_plane_wave(self):
        """
        No SOC, no detuning, no interactions, no trap.
        For plane wave psi = exp(ikz), geo.kinetic returns k²*psi (full Laplacian),
        so rhs = -i * k² * psi.
        """
        nz, lz = 256, 10.0
        kz = 2 * pi / lz * 3
        geo = GeometryCart(sizes=(nz,), lengths=(lz,))
        z, = geo.grids
        psi_2d = np.exp(1j * kz * z).reshape(-1)

        rhs = get_rhs(geo, 1,
                      sigma_z=[0.0], detuning=0.0, q_zeeman=[0.0],
                      omega_r=0.0, g_matrix=[[0.0]], n_atoms=0.0)

        result   = rhs(0.0, psi_2d)
        expected = -1j * kz**2 * psi_2d
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_real_mode_imaginary_prefactor(self):
        """
        rhs = -i * H * psi. For real H and real psi, rhs should be
        purely imaginary.
        """
        nz, lz = 128, 10.0
        geo = GeometryCart(sizes=(nz,), lengths=(lz,))
        z, = geo.grids

        # Use a real-valued (cosine) wavefunction
        psi = np.cos(2 * pi / lz * z).reshape(-1).astype(complex)

        rhs = get_rhs(geo, 1,
                      sigma_z=[0.0], detuning=2.0, q_zeeman=[0.0],
                      omega_r=0.0, g_matrix=[[0.0]], n_atoms=0.0)

        result = rhs(0.0, psi)
        # Real part should be ≈ 0 since -i * (real) = imaginary
        np.testing.assert_allclose(result.real, 0.0, atol=1e-10)

    def test_real_mode_uniform_eigenstate(self):
        """
        Uniform psi with sigma_z=0 and no SOC/kinetic is an eigenstate.
        H * psi = (delta + q) * psi (since sigma_z=0 → sigma_z²=0).
        rhs = -i * E * psi.
        """
        nz = 64
        geo = _geo1d(nz=nz)
        delta    = 3.0
        q_zeeman = 1.5
        psi = np.ones(nz, dtype=complex)

        rhs = get_rhs(geo, 1,
                      sigma_z=[0.0], detuning=delta, q_zeeman=[q_zeeman],
                      omega_r=0.0, g_matrix=[[0.0]], n_atoms=0.0)

        E = delta / 2 * (2*0 - 1) + q_zeeman  # delta array for i=0: (2*0-1)*delta/2
        # Actually: _build_detuning_diagonal(1, delta)[0] = (2*0-1)*delta/2 = -delta/2
        E = -delta / 2 + q_zeeman
        result   = rhs(0.0, psi)
        expected = -1j * E * psi
        np.testing.assert_allclose(result, expected, atol=1e-10)

    # ------------------------------------------------------------------
    # Imaginary-time mode properties
    # ------------------------------------------------------------------

    def test_imaginary_mode_eigenstate_gives_zero(self):
        """
        For imaginary time, rhs = -(H - mu) * psi.
        An eigenstate of H has mu = E, so rhs should be zero.
        Uniform psi is an eigenstate when sigma_z=0 (kinetic=0, SOC=0).
        """
        nz = 128
        geo = _geo1d(nz=nz)
        psi = np.ones(nz, dtype=complex)

        rhs = get_rhs(geo, 1,
                      sigma_z=[0.0], detuning=1.0, q_zeeman=[0.5],
                      omega_r=0.0, g_matrix=[[0.0]], n_atoms=0.0,
                      mode='imaginary')

        result = rhs(0.0, psi)
        np.testing.assert_allclose(np.abs(result), 0.0, atol=1e-10)

    def test_imaginary_mode_decreases_energy(self):
        """
        Imaginary time evolves toward lower energy.
        Starting from a superposition of two plane-wave eigenstates, a single
        Euler step should lower the mean kinetic energy.
        """
        nz, lz = 256, 10.0
        geo = GeometryCart(sizes=(nz,), lengths=(lz,))
        z, = geo.grids

        kz_lo = 2 * pi / lz * 1   # lower kinetic energy
        kz_hi = 2 * pi / lz * 5   # higher kinetic energy

        # Equal superposition
        psi = (np.exp(1j * kz_lo * z) + np.exp(1j * kz_hi * z)).reshape(-1)
        psi = psi / np.linalg.norm(psi)

        rhs = get_rhs(geo, 1,
                      sigma_z=[0.0], detuning=0.0, q_zeeman=[0.0],
                      omega_r=0.0, g_matrix=[[0.0]], n_atoms=0.0,
                      mode='imaginary')

        dt = 0.01
        psi_new = psi + dt * rhs(0.0, psi)
        psi_new = psi_new / np.linalg.norm(psi_new)

        # Compute mean kinetic energy: <psi | kinetic | psi>
        # using np.vdot which computes sum(a.conj() * b)
        E_before = np.vdot(psi,     geo.kinetic(psi)).real
        E_after  = np.vdot(psi_new, geo.kinetic(psi_new)).real

        assert E_after < E_before, (
            f"Energy should decrease in imaginary time: {E_after:.6f} >= {E_before:.6f}"
        )

    # ------------------------------------------------------------------
    # Time-dependent parameters
    # ------------------------------------------------------------------

    def test_time_dependent_detuning_changes_result(self):
        """
        A callable detuning should produce different results at different times.
        """
        nz = 64
        geo = _geo1d(nz=nz)
        psi = np.ones(nz, dtype=complex)

        rhs = get_rhs(geo, 1,
                      sigma_z=[0.0], detuning=lambda t: t,
                      q_zeeman=[0.0], omega_r=0.0,
                      g_matrix=[[0.0]], n_atoms=0.0)

        result_t0 = rhs(0.0, psi)
        result_t1 = rhs(1.0, psi)

        assert not np.allclose(result_t0, result_t1), \
            "Results at t=0 and t=1 should differ with time-dependent detuning"

    def test_time_dependent_lattice_changes_result(self):
        """
        A callable lattice strength should produce different results at t=0 vs t>0.
        """
        nz, lz = 128, 10.0
        geo = GeometryCart(sizes=(nz,), lengths=(lz,))
        rng = np.random.default_rng(7)
        psi = (rng.standard_normal(nz) + 1j * rng.standard_normal(nz))

        rhs = get_rhs(geo, 1,
                      sigma_z=[0.0], detuning=0.0,
                      q_zeeman=[0.0], omega_r=0.0,
                      g_matrix=[[0.0]], n_atoms=0.0,
                      lattice_strength=lambda t: t, k_l=1.0)

        result_t0 = rhs(0.0, psi)
        result_t2 = rhs(2.0, psi)

        assert not np.allclose(result_t0, result_t2), \
            "Results at t=0 and t=2 should differ with time-dependent lattice"

    def test_imaginary_mode_ignores_time(self):
        """
        In imaginary-time mode, time-dependent detuning is always evaluated at t=0.
        Results at t=0 and t=1 should be the same.
        """
        nz = 64
        geo = _geo1d(nz=nz)
        rng = np.random.default_rng(8)
        psi = (rng.standard_normal(nz) + 1j * rng.standard_normal(nz))

        rhs = get_rhs(geo, 1,
                      sigma_z=[0.0], detuning=lambda t: t,
                      q_zeeman=[0.0], omega_r=0.0,
                      g_matrix=[[0.0]], n_atoms=0.0,
                      mode='imaginary')

        result_t0 = rhs(0.0, psi)
        result_t5 = rhs(5.0, psi)

        np.testing.assert_allclose(result_t0, result_t5)

    # ------------------------------------------------------------------
    # Interaction and trap
    # ------------------------------------------------------------------

    def test_zero_interactions_and_trap(self):
        """Setting n_atoms=0 and wx=wy=wz=0 should give purely sp result."""
        nz, lz = 256, 10.0
        kz = 2 * pi / lz * 2
        geo = GeometryCart(sizes=(nz,), lengths=(lz,))
        z, = geo.grids
        psi = np.exp(1j * kz * z).reshape(-1)

        rhs_full = get_rhs(geo, 1, [0.0], 0.0, [0.0], 0.0,
                           [[5.0]], 0.0, wx=0.0, wy=0.0, wz=0.0)
        rhs_bare = get_rhs(geo, 1, [0.0], 0.0, [0.0], 0.0,
                           [[0.0]], 0.0)

        np.testing.assert_allclose(rhs_full(0.0, psi), rhs_bare(0.0, psi), atol=1e-12)

    def test_multicomponent_output_shape(self):
        """3-component system on 2D geometry."""
        geo = _geo2d(nx=16, nz=32)
        n_comp = 3
        n_total = 16 * 32
        psi = np.ones(n_comp * n_total, dtype=complex)

        rhs = get_rhs(geo, n_comp,
                      sigma_z=[-1.0, 1.0, 3.0],
                      detuning=0.0,
                      q_zeeman=[0.0, 0.0, 0.0],
                      omega_r=1.0,
                      g_matrix=np.eye(n_comp),
                      n_atoms=1000.0)

        assert rhs(0.0, psi).shape == psi.shape
