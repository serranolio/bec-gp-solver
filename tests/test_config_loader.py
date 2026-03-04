# -*- coding: utf-8 -*-
"""
test_config_loader.py
Unit tests for config_loader.py.

Strategy
--------
All tests load the real example_setup.toml via the public API and check
that the derived quantities are correct to within a tight tolerance.
Tests are grouped by concern:

  TestUnits          — unit conversion factors
  TestTrap           — trap frequencies in recoil units
  TestInteraction    — interaction matrix and Thomas-Fermi scales
  TestSpinOrbit      — spin-orbit and lattice wavevector
  TestGeometry       — box sizes, lz quantisation, axis filtering
  TestLoadConfig     — public load_config() return values and geometry object
  TestRampCallables  — detuning and lattice ramp lambdas

Run with:
    pytest tests/test_config_loader.py -v

The example_setup.toml must be accessible at configs/example_setup.toml
relative to the project root, or the TOML_PATH variable below must be
updated.
"""

import tomllib
from pathlib import Path

import numpy as np
from numpy import pi
import pytest

from bec_gp_solver.config_loader import load_config, _compute_derived
from bec_gp_solver.geometry import Geometry3DAxial


# Path to the test config — adjust if your directory layout differs
TOML_PATH = Path(__file__).parent.parent / "configs" / "setup_test_config.toml"


@pytest.fixture(scope="module")
def cfg():
    """Raw config dict, loaded once for the whole module."""
    with open(TOML_PATH, "rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def d(cfg):
    """Derived quantities dict, computed once for the whole module."""
    return _compute_derived(cfg)


@pytest.fixture(scope="module")
def geo_and_kwargs():
    """Geometry and rhs_kwargs from load_config(), built once."""
    return load_config(TOML_PATH)


# =============================================================================
# Unit conversions
# =============================================================================

class TestUnits:

    def test_l_unit(self, d):
        """Length unit in metres."""
        assert abs(d['l_unit'] - 1.7225e-7) / 1.7225e-7 < 1e-4

    def test_t_unit(self, d):
        """Time unit in seconds."""
        assert abs(d['t_unit'] - 8.1202e-5) / 8.1202e-5 < 1e-4

    def test_e_unit(self, d):
        """Energy unit: ħ × 2π × f_recoil."""
        hbar_si  = 1.054571800139113e-34
        f_recoil = 1960.0
        expected = 2 * pi * hbar_si * f_recoil
        assert abs(d['e_unit'] - expected) / expected < 1e-10

    def test_f_recoil_passthrough(self, d):
        assert d['f_recoil'] == 1960.0


# =============================================================================
# Trap frequencies
# =============================================================================

class TestTrap:

    def test_wx(self, d):
        assert abs(d['wx'] - 176.22 / 1960.0) < 1e-10

    def test_wy(self, d):
        assert abs(d['wy'] - 198.53 / 1960.0) < 1e-10

    def test_wz(self, d):
        assert abs(d['wz'] - 27.99 / 1960.0) < 1e-10

    def test_w3_geometric_mean(self, d):
        """w3 = (fx fy fz)^(1/3) / f_recoil."""
        expected = (176.22 * 198.53 * 27.99)**(1/3) / 1960.0
        assert abs(d['w3'] - expected) / expected < 1e-10


# =============================================================================
# Interaction matrix and Thomas-Fermi scales
# =============================================================================

class TestInteraction:

    def test_g_matrix_shape(self, d):
        """g_matrix must be (n_comp, n_comp) even for a single-component config."""
        n = d['n_comp']
        assert d['g_matrix'].shape == (n, n)

    def test_g_matrix_symmetric(self, d):
        """Scattering length matrix is symmetric so g_matrix must be too."""
        assert np.allclose(d['g_matrix'], d['g_matrix'].T)

    def test_g_matrix_3d_axial(self, d):
        """For 3d_axial: g_ij = 8π a_ij (in recoil units)."""
        a_si    = 5.2917721067e-11
        a_ref   = 100.40 * a_si / d['l_unit']
        g_ref   = 8 * pi * a_ref
        assert abs(d['g_matrix'][0, 0] - g_ref) / g_ref < 1e-6

    def test_mu_positive(self, d):
        """Chemical potential must be positive."""
        assert d['mu'] > 0

    def test_thomas_fermi_radii(self, d):
        """rx = sqrt(4μ) / wx,  rz = sqrt(4μ) / wz."""
        rx_expected = np.sqrt(4 * d['mu']) / d['wx']
        rz_expected = np.sqrt(4 * d['mu']) / d['wz']
        assert abs(d['rx'] - rx_expected) / rx_expected < 1e-10
        assert abs(d['rz'] - rz_expected) / rz_expected < 1e-10

    def test_a_matrix_atleast_2d(self, d):
        """a_matrix is always 2D, even when loaded from a flat TOML list."""
        assert d['a_matrix'].ndim == 2


# =============================================================================
# Spin-orbit coupling
# =============================================================================

class TestSpinOrbit:

    def test_omega_r(self, d):
        assert d['omega_r'] == 2.7

    def test_k_l(self, d):
        """k_l = sqrt(1 - (omega_r/4)^2)."""
        expected = np.sqrt(1 - (2.7 / 4)**2)
        assert abs(d['k_l'] - expected) < 1e-10

    def test_k_l_range(self, d):
        """k_l must be in (0, 1] for physical spin-orbit coupling."""
        assert 0 < d['k_l'] <= 1.0


# =============================================================================
# Geometry: box sizes and lz quantisation
# =============================================================================

class TestGeometry:

    def test_lx(self, d, cfg):
        expected = cfg['geometry']['lx_rx_factor'] * d['rx']
        assert abs(d['lx'] - expected) / expected < 1e-10

    def test_lz_is_integer_multiple_of_lattice_period(self, d):
        """
        lz must be an integer multiple of the lattice period 2π/k_l.
        The loader rounds down via floor division so this must hold exactly.
        """
        lattice_period = 2 * pi / d['k_l']
        remainder = d['lz'] % lattice_period
        assert remainder < 1e-10 or abs(remainder - lattice_period) < 1e-10

    def test_lz_fits_at_least_one_period(self, d):
        assert d['lz'] >= 2 * pi / d['k_l']

    def test_ly_zero_for_axial(self, d, cfg):
        """ly_ry_factor = 0 in example_setup.toml so ly must be 0."""
        assert d['ly'] == 0.0

    def test_axis_filtering_sizes(self, geo_and_kwargs, cfg):
        """ny=0 in the TOML so the geometry must have sizes (nx, nz) only."""
        geo, _ = geo_and_kwargs
        assert geo.sizes == (cfg['geometry']['nx'], cfg['geometry']['nz'])

    def test_axis_filtering_removes_ly(self, geo_and_kwargs, d):
        """The geometry lengths tuple must not contain ly=0."""
        geo, _ = geo_and_kwargs
        assert 0.0 not in geo.lengths
        assert len(geo.lengths) == 2


# =============================================================================
# load_config public API
# =============================================================================

class TestLoadConfig:

    def test_returns_two_values(self, geo_and_kwargs):
        assert len(geo_and_kwargs) == 2

    def test_geometry_type(self, geo_and_kwargs, cfg):
        geo, _ = geo_and_kwargs
        assert isinstance(geo, Geometry3DAxial)

    def test_rhs_kwargs_keys(self, geo_and_kwargs):
        """All keys expected by get_rhs() must be present."""
        _, rhs_kwargs = geo_and_kwargs
        required = {
            'n_components', 'sigma_z', 'detuning', 'q_zeeman',
            'omega_r', 'g_matrix', 'n_atoms', 'wx', 'wy', 'wz',
            'lattice_strength', 'k_l',
        }
        assert required.issubset(rhs_kwargs.keys())

    def test_mode_not_in_rhs_kwargs(self, geo_and_kwargs):
        """'mode' must NOT be in rhs_kwargs — it is set at the call site."""
        _, rhs_kwargs = geo_and_kwargs
        assert 'mode' not in rhs_kwargs

    def test_n_components(self, geo_and_kwargs, cfg):
        _, rhs_kwargs = geo_and_kwargs
        assert rhs_kwargs['n_components'] == cfg['system']['n_components']

    def test_sigma_z_values(self, geo_and_kwargs):
        _, rhs_kwargs = geo_and_kwargs
        assert np.allclose(rhs_kwargs['sigma_z'], [-1.0, 1.0])

    def test_g_matrix_in_rhs_kwargs(self, geo_and_kwargs):
        _, rhs_kwargs = geo_and_kwargs
        assert rhs_kwargs['g_matrix'].shape == (2, 2)

    def test_geo_dv_sums_to_box_volume(self, geo_and_kwargs, d):
        """
        ∫ dv over the full grid should approximate the 3D volume 2π ∫r dr dz.
        For the axial geometry: volume ≈ π lx² lz.
        We only check order of magnitude since the DHT quadrature is not
        exact on the full box.
        """
        geo, _ = geo_and_kwargs
        volume_numerical = geo.dv.sum()
        volume_estimate  = pi * d['lx']**2 * d['lz']
        # within a factor of 2 — loose check, exact value depends on quadrature
        assert 0.5 < volume_numerical / volume_estimate < 2.0


# =============================================================================
# Ramp callables
# =============================================================================

class TestRampCallables:

    def test_detuning_at_t0(self, geo_and_kwargs, cfg, d):
        """delta(0) must equal delta_start_hz converted to recoil units."""
        _, rhs_kwargs = geo_and_kwargs
        delta_fn = rhs_kwargs['detuning']
        expected = cfg['sweep']['delta_start_hz'] / d['f_recoil']
        # detuning returns a per-component array: check component 0
        print(delta_fn(0))
        assert abs(float(delta_fn(0.0)) - expected) < 1e-10

    def test_detuning_saturates_after_ramp(self, geo_and_kwargs, cfg, d):
        """delta(t >> t_ramp) must equal delta_end_hz converted to recoil units."""
        _, rhs_kwargs = geo_and_kwargs
        delta_fn = rhs_kwargs['detuning']
        t_large  = 1e10
        expected_end = cfg['sweep']['delta_end_hz'] / d['f_recoil']
        # component 0: (2*0 - 1) * delta_end / 2 = -delta_end / 2
        assert abs(float(delta_fn(t_large)) - expected_end) < 1e-10

    def test_detuning_is_monotone_for_decreasing_sweep(self, geo_and_kwargs, cfg, d):
        """
        In example_setup.toml delta_start == delta_end == 100 Hz so the
        ramp is flat. Check that delta(t) is constant in this case.
        """
        _, rhs_kwargs = geo_and_kwargs
        delta_fn = rhs_kwargs['detuning']
        d_end = cfg['sweep']['delta_end_hz']
        d_start = cfg['sweep']['delta_start_hz']
        t_ramp   = cfg['sweep']['ramp_time_ms'] * 1e-3 / d['t_unit']
        t_axes = np.linspace(0, 1, 20)
        expected_vals = [(d_start + (d_end-d_start)*t) / d['f_recoil'] for t in t_axes]
        vals     = [delta_fn(t * t_ramp) for t in t_axes]
        assert np.allclose(vals, expected_vals)

    def test_lattice_at_t0(self, geo_and_kwargs, cfg):
        """lattice_strength(0) must equal omega_l_start."""
        _, rhs_kwargs = geo_and_kwargs
        lattice_fn = rhs_kwargs['lattice_strength']
        assert abs(lattice_fn(0.0) - cfg['sweep']['omega_l_start']) < 1e-10

    def test_lattice_saturates_after_ramp(self, geo_and_kwargs, cfg):
        """lattice_strength(t >> t_ramp) must equal omega_l_end."""
        _, rhs_kwargs = geo_and_kwargs
        lattice_fn = rhs_kwargs['lattice_strength']
        assert abs(lattice_fn(1e10) - cfg['sweep']['omega_l_end']) < 1e-10

    def test_lattice_callable(self, geo_and_kwargs):
        """lattice_strength must be a callable."""
        _, rhs_kwargs = geo_and_kwargs
        assert callable(rhs_kwargs['lattice_strength'])

    def test_detuning_callable(self, geo_and_kwargs):
        """detuning must be a callable."""
        _, rhs_kwargs = geo_and_kwargs
        assert callable(rhs_kwargs['detuning'])
