# -*- coding: utf-8 -*-
"""
test_geometry.py
Unit tests for the differential operators in geometry.py.

Strategy
--------
Both kinetic() and grad_z() are tested by applying them to functions
whose exact derivatives are known analytically, then comparing numerical
vs analytical results with a tight tolerance.

  GeometryCart   : plane waves are exact eigenfunctions of FFT-based
                   operators, so errors should be at machine precision (~1e-10).

  Geometry3DAxial: Gaussian tested against its known analytic derivatives.
                   Errors are larger (~1e-5) due to the finite Hankel quadrature,
                   but should still be small compared to 1.

Run with:
    pytest test_geometry.py -v
"""

import numpy as np
from numpy import pi
import pytest
from bec_gp_solver.geometry import GeometryCart, Geometry3DAxial, make_geometry


# =============================================================================
# Helpers
# =============================================================================

def relative_error(numerical, analytical):
    """Max absolute error normalised by the max of the analytical solution."""
    return np.max(np.abs(numerical - analytical)) / np.max(np.abs(analytical))


# =============================================================================
# GeometryCart tests
# =============================================================================

class TestGeometryCart:
    """
    Plane wave  ψ(x) = exp(i k·r)  is an exact eigenfunction of both operators:

        -∇²/2  ψ = |k|²/2  ψ       (kinetic)
        -i ∂/∂z ψ =  k_z   ψ       (grad_z)

    With periodic boundary conditions and an integer number of wavelengths
    fitting in the box, FFT reproduces these exactly up to machine precision.
    """

    # ------------------------------------------------------------------
    # 1D
    # ------------------------------------------------------------------

    def test_kinetic_1d(self):
        nz, lz = 256, 10.0
        geo = GeometryCart(sizes=(nz,), lengths=(lz,))

        kz_test = 2 * pi / lz * 3          # 3 full wavelengths in the box
        z,      = geo.grids
        psi     = np.exp(1j * kz_test * z).reshape(-1)

        numerical  = geo.kinetic(psi)
        analytical = (kz_test**2) * psi

        assert relative_error(numerical, analytical) < 1e-10

    def test_grad_z_1d(self):
        nz, lz = 256, 10.0
        geo = GeometryCart(sizes=(nz,), lengths=(lz,))

        kz_test = 2 * pi / lz * 5
        z,      = geo.grids
        psi     = np.exp(1j * kz_test * z).reshape(-1)

        numerical  = geo.grad_z(psi)
        analytical = -1j * kz_test * psi          # -i ∂/∂z e^{ikz} = k e^{ikz}

        assert relative_error(numerical, analytical) < 1e-10

    # ------------------------------------------------------------------
    # 2D
    # ------------------------------------------------------------------

    def test_kinetic_2d(self):
        nx, nz   = 64, 128
        lx, lz   = 8.0, 10.0
        geo      = GeometryCart(sizes=(nx, nz), lengths=(lx, lz))

        kx_test  = 2 * pi / lx * 2
        kz_test  = 2 * pi / lz * 3
        x_, z_   = geo.grids
        psi      = np.exp(1j * (kx_test * x_ + kz_test * z_)).reshape(-1)

        numerical  = geo.kinetic(psi)
        analytical = ((kx_test**2 + kz_test**2)) * psi

        assert relative_error(numerical, analytical) < 1e-10

    def test_grad_z_2d(self):
        nx, nz   = 64, 128
        lx, lz   = 8.0, 10.0
        geo      = GeometryCart(sizes=(nx, nz), lengths=(lx, lz))

        kx_test  = 2 * pi / lx * 1
        kz_test  = 2 * pi / lz * 4
        x_, z_   = geo.grids
        psi      = np.exp(1j * (kx_test * x_ + kz_test * z_)).reshape(-1)

        numerical  = geo.grad_z(psi)
        analytical = -1j * kz_test * psi

        assert relative_error(numerical, analytical) < 1e-10

    # ------------------------------------------------------------------
    # 3D
    # ------------------------------------------------------------------

    def test_kinetic_3d(self):
        nx, ny, nz  = 32, 32, 64
        lx, ly, lz  = 6.0, 6.0, 10.0
        geo         = GeometryCart(sizes=(nx, ny, nz), lengths=(lx, ly, lz))

        kx_test = 2 * pi / lx * 2
        ky_test = 2 * pi / ly * 1
        kz_test = 2 * pi / lz * 3
        x_, y_, z_ = geo.grids
        psi     = np.exp(1j * (kx_test*x_ + ky_test*y_ + kz_test*z_)).reshape(-1)

        numerical  = geo.kinetic(psi)
        analytical = ((kx_test**2 + ky_test**2 + kz_test**2)) * psi

        assert relative_error(numerical, analytical) < 1e-10

    def test_grad_z_3d(self):
        nx, ny, nz  = 32, 32, 64
        lx, ly, lz  = 6.0, 6.0, 10.0
        geo         = GeometryCart(sizes=(nx, ny, nz), lengths=(lx, ly, lz))

        kx_test = 2 * pi / lx * 1
        ky_test = 2 * pi / ly * 2
        kz_test = 2 * pi / lz * 5
        x_, y_, z_ = geo.grids
        psi     = np.exp(1j * (kx_test*x_ + ky_test*y_ + kz_test*z_)).reshape(-1)

        numerical  = geo.grad_z(psi)
        analytical = -1j * kz_test * psi

        assert relative_error(numerical, analytical) < 1e-10

    # ------------------------------------------------------------------
    # multi-component: operators must act independently on each component
    # ------------------------------------------------------------------

    def test_kinetic_multicomponent(self):
        """Each component sees a different plane wave; results must not mix."""
        nz, lz  = 256, 10.0
        geo     = GeometryCart(sizes=(nz,), lengths=(lz,))
        z,      = geo.grids

        k1, k2, k3 = (2*pi/lz * m for m in (2, 5, 8))
        psi = np.array([
            np.exp(1j * k1 * z).reshape(-1),
            np.exp(1j * k2 * z).reshape(-1),
            np.exp(1j * k3 * z).reshape(-1),
        ])

        numerical  = geo.kinetic(psi)
        analytical = np.array([
            (k1**2) * psi[0],
            (k2**2) * psi[1],
            (k3**2) * psi[2],
        ])

        assert relative_error(numerical, analytical) < 1e-10

    def test_linearity(self):
        """kinetic(a*ψ₁ + b*ψ₂) == a*kinetic(ψ₁) + b*kinetic(ψ₂)."""
        nz, lz  = 128, 8.0
        geo     = GeometryCart(sizes=(nz,), lengths=(lz,))
        rng     = np.random.default_rng(0)

        psi1 = rng.standard_normal(nz) + 1j * rng.standard_normal(nz)
        psi2 = rng.standard_normal(nz) + 1j * rng.standard_normal(nz)
        a, b = 1.3 + 0.7j, -0.4 + 1.1j

        lhs = geo.kinetic(a * psi1 + b * psi2)
        rhs = a * geo.kinetic(psi1) + b * geo.kinetic(psi2)

        assert relative_error(lhs, rhs) < 1e-10

    def test_fourier_transform(self):
        """ F(ψ(x)) = ψ(k)"""
        nz, lz = 128, 8.0
        sz = 0.5
        geo    = GeometryCart(sizes=(nz,), lengths=(lz,))
        z,     = geo.grids
        k,     = np.fft.fftshift(geo.kgrids)
        psi    = 1 / np.sqrt(2*np.pi*sz**2) * np.exp(-z**2 / (2*sz**2))
        analytical = 1 / np.sqrt(2*np.pi) * np.exp(-k**2 * sz**2 / 2)
        numerical  = geo.forward_transform(psi)
        
        assert relative_error(numerical, analytical) < 1e-10

    def test_ifourier_transform(self):
        """ iF(ψ(k)) = ψ(x)"""
        nz, lz = 256, 8.0
        sz = 0.1
        geo    = GeometryCart(sizes=(nz,), lengths=(lz,))
        z,     = geo.grids
        k,     = np.fft.fftshift(geo.kgrids)
        psi    = 1 / np.sqrt(2*np.pi) * np.exp(-k**2 * sz**2 / 2)
        analytical = 1 / np.sqrt(2*np.pi*sz**2) * np.exp(-z**2 / (2*sz**2))
        numerical = geo.inverse_transform(psi)

        assert relative_error(numerical, analytical) < 1e-10

    def test_make_geometry(self):
        """ test the make_geometry method"""
        geo = make_geometry("3d_cart", 
                            sizes=(128, 128, 128), 
                            lengths=(1, 2, 3))
        assert geo.dv.sum()==6.0



# =============================================================================
# Geometry3DAxial tests
# =============================================================================

class TestGeometry3DAxial:
    """
    The Hankel transform is not exact for arbitrary functions — it has finite
    quadrature error — so we test with a Gaussian whose analytic derivatives
    are known, and accept a looser tolerance (~1e-4).

    ψ(r, z) = exp(-r²/2σr² - z²/2σz²)

    ∂/∂z ψ  = -(z/σz²) ψ
    ∇²   ψ  = [(r²/σr⁴ - (1 + 1/σr²)/r·∂/∂r terms) + z²/σz⁴ - 1/σz²] ψ

    Rather than expanding the full cylindrical Laplacian analytically
    (which is messy), we use the fact that for a separable Gaussian
    ψ(r, z) = f(r) · g(z), the operators split cleanly:

        T_z  g(z) = -½ g''(z)  via FFT  →  tested independently
        T_r  f(r)              via DHT   →  tested via T applied to f⊗g
    """

    @pytest.fixture
    def geo(self):
        #return Geometry3DAxial(nx=64, nz=128, lx=30, lz=40)
        return Geometry3DAxial(sizes=(64, 128), lengths=(33.8825, 377.3648))

    @pytest.fixture
    def gaussian(self, geo):
        r_, z_ = geo.grids
        sr, sz = 6.0, 25.0 # 2.0, 3.0
        psi    = np.exp(-r_**2 / (2*sr**2) - z_**2 / (2*sz**2))
        return psi, sr, sz

    # ------------------------------------------------------------------
    # grad_z: tested exactly since it's a pure FFT operation
    # -i ∂/∂z ψ = (z/σz²) ψ  (note: -i cancels with i from Gaussian derivative)
    # ------------------------------------------------------------------

    def test_grad_z(self, geo, gaussian):
        psi, sr, sz = gaussian
        r_, z_      = geo.grids

        numerical  = -geo.grad_z(psi.reshape(-1)).reshape(geo.sizes)
        analytical = (-z_ / sz**2) * psi

        # exclude boundary region where Gaussian has decayed to ~0
        # assert np.allclose(numerical.real, analytical)
        interior = np.abs(psi) > 1e-6
        err = np.max(np.abs(numerical[interior] - analytical[interior]))
        assert err < 1e-6

    # ------------------------------------------------------------------
    # kinetic — axial part only: isolate T_z by using ψ = 1 ⊗ g(z)
    # T_z g = -½ g'' = -(z²/σz⁴ - 1/σz²)/2 · g
    # ------------------------------------------------------------------

    def test_kinetic_axial_part(self, geo):
        from scipy.special import jv
        r_, z_ = geo.grids
        sr    = geo._kr[2]
        g_z   = jv(0, sr*r_)     # bessel in r and constant in z

        numerical  = geo.kinetic(g_z.reshape(-1)).reshape(geo.sizes)
        analytical = (sr**2) * g_z

        interior = np.abs(g_z) > 1e-6
        err = np.max(np.abs(numerical[interior] - analytical[interior]))
        assert err < 1e-6

    # ------------------------------------------------------------------
    # kinetic — full 2D Gaussian
    # -½∇²ψ = [-½(∂²/∂r² + 1/r ∂/∂r) - ½∂²/∂z²] ψ
    # For ψ = exp(-r²/2σr²) exp(-z²/2σz²) this equals:
    # [r²/(2σr⁴) - 1/σr² + z²/(2σz⁴) - 1/(2σz²)] ψ
    # ------------------------------------------------------------------

    def test_kinetic_full(self, geo, gaussian):
        psi, sr, sz = gaussian
        r_, z_      = geo.grids

        numerical  = -geo.kinetic(psi.reshape(-1)).reshape(geo.sizes)
        analytical = (r_**2 / (sr**4) - 2 / (sr**2)
                    + z_**2 / (sz**4) - 1 / (sz**2)) * psi

        interior = np.abs(psi) > 1e-4
        err = np.max(np.abs(numerical[interior] - analytical[interior]))
        assert err < 1e-3

    # ------------------------------------------------------------------
    # volume element: norm of a Gaussian should equal its analytic value
    # ∫ 2π r |ψ|² dr dz = 2π · (√π σr)² / 2 · √(2π) σz ... simplified:
    # For ψ = exp(-r²/2σr²-z²/2σz²): norm = π σr² √(2π) σz
    # ------------------------------------------------------------------

    def test_volume_element(self, geo):
        r_, z_ = geo.grids
        sr, sz = 6.0, 25.0
        psi    = np.exp(-r_**2 / (2*sr**2) - z_**2 / (2*sz**2))

        numerical  = (geo.dv * np.abs(psi.reshape(-1))**2).sum()
        analytical = pi * sr**2 * np.sqrt(pi) * sz   # ∫ 2πr |ψ|² dr dz

        assert abs(numerical - analytical) / analytical < 1e-3

    # ------------------------------------------------------------------
    # multi-component: test linearity
    # ------------------------------------------------------------------

    def test_kinetic_multicomponent(self, geo):
        r_, z_ = geo.grids
        sr1, sz1 = 6.0, 25.0
        sr2, sz2 = 3.0, 15.0

        psi = np.array([
            np.exp(-r_**2 / (2*sr1**2) - z_**2 / (2*sz1**2)).reshape(-1),
            np.exp(-r_**2 / (2*sr2**2) - z_**2 / (2*sz2**2)).reshape(-1),
        ])

        numerical  = geo.kinetic(psi)
        analytical = np.array([
            geo.kinetic(psi[0]),
            geo.kinetic(psi[1]),
        ])

        assert relative_error(numerical, analytical) < 1e-10

    def test_transform_roundtrip(self, geo):
        r_, z_ = geo.grids
        sr, sz = 4.0, 20.0
        psi = np.exp(-r_**2 / (2*sr**2) - z_**2 / (2*sz**2)).reshape(-1)

        psi_rt = geo.inverse_transform(geo.forward_transform(psi))

        interior = np.abs(psi) > 1e-6
        err = np.max(np.abs(psi_rt[interior] - psi[interior]))
        assert err < 1e-5

    def test_make_geometry(self):
        try:
            geo = make_geometry("no_name", 
                                sizes=(256, 256),
                                lengths=(2,3))
        except ValueError:
            geo = make_geometry("3d_axial", 
                                sizes=(256, 256), 
                                lengths=(2, 3))
        assert np.allclose(geo.dv.sum(), np.pi*2**2*3, rtol=1e-1)



# =============================================================================
# Spectral basis (to_spectral / from_spectral) — used by the projector
# =============================================================================

def _random_field(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.fixture(params=['1d', '2d', '3d', 'axial'])
def any_geo(request):
    if request.param == '1d':
        return GeometryCart(sizes=(64,), lengths=(10.0,))
    if request.param == '2d':
        return GeometryCart(sizes=(16, 32), lengths=(5.0, 10.0))
    if request.param == '3d':
        return GeometryCart(sizes=(8, 8, 16), lengths=(4.0, 4.0, 8.0))
    return Geometry3DAxial(sizes=(32, 64), lengths=(20.0, 60.0))


class TestSpectral:

    def test_roundtrip_exact(self, any_geo):
        """from_spectral(to_spectral(ψ)) == ψ to machine precision."""
        rng = np.random.default_rng(1)
        psi = _random_field(rng, (2, len(any_geo.dv)))
        out = any_geo.from_spectral(any_geo.to_spectral(psi))
        assert relative_error(out, psi) < 1e-12

    def test_parseval(self, any_geo):
        """to_spectral is unitary: Σ|a|² == Σ dv |ψ|²."""
        rng = np.random.default_rng(2)
        psi = _random_field(rng, len(any_geo.dv))
        a   = any_geo.to_spectral(psi)
        lhs = (np.abs(a)**2).sum()
        rhs = (any_geo.dv * np.abs(psi)**2).sum()
        assert abs(lhs - rhs) / rhs < 1e-12

    def test_shapes(self, any_geo):
        """Flat multi-component input → (n_fields, *sizes) → (n_fields, N_total)."""
        N   = len(any_geo.dv)
        psi = np.ones(3 * N, dtype=complex)
        a   = any_geo.to_spectral(psi)
        assert a.shape == (3,) + tuple(any_geo.sizes)
        assert any_geo.from_spectral(a).shape == (3, N)

    def test_spectral_kgrids_shape(self, any_geo):
        assert len(any_geo.spectral_kgrids) == len(any_geo.sizes)
        for k in any_geo.spectral_kgrids:
            assert k.shape == tuple(any_geo.sizes)

    def test_plane_wave_is_single_mode_cart(self):
        """A plane wave maps onto exactly the mode with the matching momentum."""
        geo   = GeometryCart(sizes=(16, 32), lengths=(5.0, 10.0))
        x, z  = geo.grids
        kx, kz = 2*pi/5.0 * 2, 2*pi/10.0 * -3
        a     = np.abs(geo.to_spectral(np.exp(1j * (kx*x + kz*z))))[0]
        idx   = np.unravel_index(a.argmax(), a.shape)
        assert np.isclose(geo.spectral_kgrids[0][idx], kx)
        assert np.isclose(geo.spectral_kgrids[1][idx], kz)
        assert (a**2).sum() - a[idx]**2 < 1e-20

    def test_bessel_mode_is_single_mode_axial(self):
        """J₀(k_n r) e^{ikz} lands (almost entirely) in mode (n, m)."""
        from scipy.special import jv as _jv
        geo   = Geometry3DAxial(sizes=(32, 64), lengths=(20.0, 60.0))
        r, z  = geo.grids
        n, kz = 3, 2*pi/60.0 * 5
        psi   = _jv(0, geo._kr[n] * r) * np.exp(1j * kz * z)
        a2    = np.abs(geo.to_spectral(psi))[0]**2
        idx   = np.unravel_index(a2.argmax(), a2.shape)
        assert np.isclose(geo.spectral_kgrids[0][idx], geo._kr[n])
        assert np.isclose(geo.spectral_kgrids[1][idx], kz)
        assert a2[idx] / a2.sum() > 1 - 1e-8
