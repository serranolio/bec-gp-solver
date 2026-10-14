# -*- coding: utf-8 -*-
"""
Geometry classes for GP simulations of Bose-Einstein condensates.

Each class encapsulates the grid, volume elements, momentum grids,
kinetic energy operators, and Fourier transforms for a given geometry.

Classes
-------
Geometry        : abstract base class defining the interface
GeometryCart    : n-dimensional Cartesian geometry (pure FFT)
Geometry3DAxial : 3D geometry with axial symmetry (DHT in r, FFT in z)

author: Federico Serrano
Physics and Astronomy Department
Washington State University
"""

import numpy as np
from numpy import pi
from abc import ABC, abstractmethod
from scipy.special import jv
from discrete_hankel_transform import HankelTransform

import logging
logger = logging.getLogger(__name__)


# =============================================================================
# Abstract base class
# =============================================================================

class Geometry(ABC):
    """
    Base class for all geometries.

    Every subclass must expose:
      self.grids        : list of nd-arrays with real-space coordinates,
                          one per spatial dimension, shaped to broadcast
                          over the full grid (output of np.meshgrid).
      self.kgrids       : same for momentum-space coordinates.
      self.dv           : flat array of volume elements, shape (N_total,).
      self.spectral_kgrids : list of nd-arrays, one per axis (last = z),
                          with the momentum of each spectral mode in the
                          ordering used by to_spectral() / from_spectral().

    And implement the abstract methods below.
    """

    @abstractmethod
    def kinetic(self, psi):
        """
        Apply the kinetic energy operator -∇²/2 to psi.

        psi : complex array, shape (N_total,) or (n_components, N_total)
        Returns array of same shape as psi.
        """
        ...

    @abstractmethod
    def grad_z(self, psi):
        """
        Apply the axial gradient operator -i ∂/∂z to psi.
        Used for spin-orbit coupling terms in the Hamiltonian.

        Same shape convention as kinetic().
        """
        ...

    @abstractmethod
    def forward_transform(self, state):
        """
        Transform from real space to momentum space.
        Used for observables, TWA noise generation, and output.
        """
        ...

    @abstractmethod
    def inverse_transform(self, state):
        """Inverse of forward_transform."""
        ...

    @abstractmethod
    def to_spectral(self, psi):
        """
        Expand psi in the orthonormal single-particle mode basis.

        The map is unitary with respect to the grid inner product:
        Σ |a|² = Σ dv |ψ|². Unlike forward_transform() it is unshifted and
        not normalised as a continuous Fourier transform — it is meant for
        projectors and noise sampling, not for observables.

        psi : array whose size is a multiple of N_total, e.g. (N_total,),
              (n_components, N_total) or (n_components * N_total,)
        Returns array of shape (n_fields, *sizes).
        """
        ...

    @abstractmethod
    def from_spectral(self, psi_k):
        """
        Exact inverse of to_spectral().

        psi_k : array of shape (n_fields, *sizes)
        Returns array of shape (n_fields, N_total).
        """
        ...

# =============================================================================
# n-dimensional Cartesian geometry
# =============================================================================

class GeometryCart(Geometry):
    """
    Cartesian geometry for 1D, 2D, or 3D systems. FFT in every direction.

    Parameters
    ----------
    sizes   : tuple of int   — number of grid points per axis, e.g. (nz,)
                               or (nx, nz) or (nx, ny, nz).
    lengths : tuple of float — box sizes in recoil units, same order as sizes.

    Convention: the last axis is always z (the spin-orbit / axial direction).

    Examples
    --------
    >>> geo1d = GeometryCart(sizes=(512,),        lengths=(lz,))
    >>> geo2d = GeometryCart(sizes=(64, 512),     lengths=(lx, lz))
    >>> geo3d = GeometryCart(sizes=(64, 64, 512), lengths=(lx, ly, lz))
    """

    def __init__(self, sizes, lengths):
        assert len(sizes) == len(lengths), \
            "sizes and lengths must have the same number of entries."

        self.ndim        = len(sizes)
        self.sizes       = sizes
        self.grid_points = sizes
        self.lengths     = lengths
        self.basis       = f"{self.ndim}d_cart"

        # grid spacings
        spacings = [l / n for n, l in zip(sizes, lengths)]

        # real-space and momentum-space 1D arrays
        axes  = [np.arange(n) * d - l / 2
                 for n, l, d in zip(sizes, lengths, spacings)]
        kaxes = [2 * pi * np.fft.fftfreq(n, d=d)
                 for n, d in zip(sizes, spacings)]

        # nd meshgrids — exposed so the physics layer can build potentials
        self.grids  = np.meshgrid(*axes,  indexing='ij')
        self.kgrids = np.meshgrid(*kaxes, indexing='ij')

        # volume element (uniform for Cartesian grids)
        self.spacings = np.array(spacings)
        self.dv = np.full(int(np.prod(sizes)), float(np.prod(spacings)))
        self.dvk = np.full(int(np.prod(sizes)), float(np.prod(2*pi / 
                                                              np.array(lengths))))

        # precomputed operators (shaped to broadcast over full grid)
        self._k2 = sum(k**2 for k in self.kgrids)   # |k|² in momentum space
        self._kz = self.kgrids[-1]                   # last axis = z

        # spectral basis = plane waves in FFT ordering
        self.spectral_kgrids = self.kgrids
        self._sqrt_dv        = np.sqrt(np.prod(spacings))

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------

    @property
    def _fft_axes(self):
        return tuple(range(-self.ndim, 0))

    def _fft(self, psi):
        return np.fft.fftn(psi.reshape((-1,) + self.sizes), axes=self._fft_axes)

    def _ifft(self, arr):
        return np.fft.ifftn(arr, axes=self._fft_axes)

    # -------------------------------------------------------------------------
    # interface implementation
    # -------------------------------------------------------------------------

    def kinetic(self, psi):
        shape0 = psi.shape
        return self._ifft(self._k2 * self._fft(psi)).reshape(shape0)

    def grad_z(self, psi):
        shape0 = psi.shape
        return self._ifft((-1j * self._kz) * self._fft(psi)).reshape(shape0)

    def forward_transform(self, state):
        out = np.fft.fftn(np.fft.fftshift(state).reshape((-1,) + self.sizes),
                          axes=self._fft_axes)
        return (np.prod(self.spacings / np.sqrt(2*pi)) *
                np.fft.fftshift(out, axes=self._fft_axes))

    def inverse_transform(self, state):
        return np.prod(np.sqrt(2*pi) / self.spacings) * np.fft.fftshift(
                np.fft.ifftn(np.fft.fftshift(state, axes=self._fft_axes), 
                             axes=self._fft_axes
        ))

    def to_spectral(self, psi):
        return self._sqrt_dv * np.fft.fftn(psi.reshape((-1,) + self.sizes),
                                           axes=self._fft_axes, norm='ortho')

    def from_spectral(self, psi_k):
        out = np.fft.ifftn(psi_k, axes=self._fft_axes, norm='ortho')
        return (out / self._sqrt_dv).reshape(psi_k.shape[0], -1)


# =============================================================================
# 3D axial geometry
# =============================================================================

class Geometry3DAxial(Geometry):
    """
    3D geometry with axial symmetry (cylindrical coordinates).
    Transverse direction r : discrete Hankel transform on [0, lx].
    Axial direction z      : FFT on [-lz/2, lz/2].

    Parameters
    ----------
    sizes : tuple of int   — number of grid points
    lengths : tuploe of float — box size in recoil units

    Notes
    -----
    The radial grid nodes are not uniformly spaced — they coincide with
    the zeros of J₀, as required by the discrete Hankel transform.
    The volume element dv already accounts for the 2π r dr dz factor.
    """

    def __init__(self, sizes, lengths):
        assert len(sizes) == 2 and len(lengths) == 2, \
                "Geometry3DAxial expects sizes=(nx, nz) and lengths=(lx, lz)"
        
        nx, nz = sizes
        lx, lz = lengths

        self.lengths = lengths

        self.ndim = 2
        self.sizes = (nx, nz)
        self.grid_points = (nx, nz)
        self.basis       = "3d_axial"

        # zeros of J₀ needed for Hankel quadrature
        self._ht = HankelTransform(nx, order=0, r_max=lx)

        # coordinate arrays
        r = self._ht.r
        z  = np.arange(nz) * lz / nz - lz / 2

        kr = self._ht.k
        kz = 2 * pi * np.fft.fftfreq(nz, d=lz / nz)


        # 2D meshgrids (indexing='ij': r is axis 0, z is axis 1)
        self.grids  = np.meshgrid(r,  z,  indexing='ij')
        self.kgrids = np.meshgrid(kr, np.fft.fftshift(kz), indexing='ij')

        # volume elements
        # radial weight from Hankel quadrature: 2π r dr integrated by the
        # quadrature rule gives this expression
        self.dx = self._ht.measure
        self.dz  = lz / nz
        self.dv  = np.kron(2 * pi * self.dz * self._ht.measure, np.ones(nz))
        self.dvk = np.kron((2 * pi)**2 / lz / lx**4 * self._ht.measure_k, np.ones(nz))
        self.spacings = np.array([self.dx.min(), self.dz])

        # store for use in operators and transforms
        self._kr = kr
        self._kz = kz

        # spectral basis: Bessel modes J₀(kr r) in r × plane waves in z.
        # The symmetric DHT kernel T is orthogonal only up to ~1e-9, so we
        # replace it by its polar factor (nearest orthogonal matrix): this
        # makes to_spectral exactly unitary and projectors exactly idempotent.
        j, S = self._ht.zeros, self._ht.factor
        J1   = np.abs(jv(1, j))
        T    = 2 * jv(0, np.outer(j, j) / S) / (np.outer(J1, J1) * S)
        u, _, vt = np.linalg.svd(T)
        self._U_r     = u @ vt
        self._sqrt_dv = np.sqrt(self.dv).reshape(nx, nz)
        self.spectral_kgrids = np.meshgrid(kr, kz, indexing='ij')

        # kinetic matrix in r: built once, O(nx²), applied by matmul
        # T_r ψ = iDHT[ kr² · DHT[ψ] ] precomputed as a dense (nx × nx) matrix
        logger.info("Building radial kinetic energy matrix …")
        try:
            self._Tr = self._ht.backward(
                    np.diag(kr**2) @ self._ht.forward(np.eye(nx), axis=0),
                    axis=0
                    )
        except MemoryError as e:
            raise MemoryError(
                    f"Failed to allocate {nx}×{nx} radial kinetic matrix. "
                    f"Required memory: ~{(nx**2 * 16) / (1024**3):.2f} GB. "
                    f"Try reducing nx in your config file."
                    ) from e
        except Exception as e:
            raise RuntimeError(
                    f"Failed to build radial kinetic matrix for nx={nx}: {e}"
                    ) from e
        logger.info("Done.")
        

    # -------------------------------------------------------------------------
    # interface implementation
    # -------------------------------------------------------------------------

    def kinetic(self, psi):
        """T_r via matmul (Hankel), T_z via FFT."""
        shape0 = psi.shape
        nx, nz = self.sizes
        p = psi.reshape((-1, nx, nz))

        Tr_psi = self._Tr @ p                   # matmul along r-axis
        Tz_psi = np.fft.ifft(
            self._kz**2 * np.fft.fft(p, axis=-1), axis=-1
        )
        return (Tr_psi + Tz_psi).reshape(shape0)

    def grad_z(self, psi):
        """-i ∂/∂z via FFT along the axial axis."""
        shape0 = psi.shape
        nx, nz = self.sizes
        p = psi.reshape((-1, nx, nz))
        out = np.fft.ifft((-1j * self._kz[None, :]) * np.fft.fft(p, axis=-1), axis=-1)
        return out.reshape(shape0)

    def forward_transform(self, state):
        """DHT in r, FFT in z."""
        shape0 = state.shape
        nx, nz = self.sizes 
        lx, lz = self.lengths
        s = state.reshape(shape0[:-1] + (nx, nz))

        out = lx**2 * self._ht.forward(s, axis=-2)
        out = (self.dz / np.sqrt(2*pi)) * np.fft.fft(out, axis=-1)
        out = np.fft.fftshift(out, axes=-1)
        return out.reshape(shape0)

    def inverse_transform(self, state):
        """Inverse DHT in r, inverse FFT in z."""
        shape0 = state.shape
        nx, nz = self.sizes
        lx, lz = self.lengths
        s = state.reshape(shape0[:-1] + (nx, nz))

        out = self._ht.backward(s, axis=-2) / lx**2
        out = np.fft.fftshift(out, axes=-1)
        out = (np.sqrt(2*pi) / self.dz) * np.fft.ifft(out, axis=-1)
        return out.reshape(shape0)

    def to_spectral(self, psi):
        """Orthogonal Bessel transform in r, unitary FFT in z."""
        nx, nz = self.sizes
        p = self._sqrt_dv * psi.reshape((-1, nx, nz))
        return np.fft.fft(self._U_r @ p, axis=-1, norm='ortho')

    def from_spectral(self, psi_k):
        p = self._U_r.T @ np.fft.ifft(psi_k, axis=-1, norm='ortho')
        return (p / self._sqrt_dv).reshape(psi_k.shape[0], -1)


# =============================================================================
# factory function
# =============================================================================

def make_geometry(kind, **kwargs):
    """
    Instantiate a geometry by name. Passes all keyword arguments to the
    corresponding class constructor.

    Parameters
    ----------
    kind : str — one of '1d_cart', '2d_cart', '3d_cart', '3d_axial'

    Examples
    --------
    >>> geo = make_geometry('1d_cart'   sizes=(512,),        lengths=(lz,))
    >>> geo = make_geometry('3d_cart',  sizes=(64, 64, 512), lengths=(lx, ly, lz))
    >>> geo = make_geometry('3d_axial', sizes=(64, 512),     lengths=(lx, lz))
    """
    cart_aliases = {'1d_cart', '2d_cart', '3d_cart'}
    logger.info(f"Setting geometry to {kind}")
    if kind in cart_aliases:
        return GeometryCart(**kwargs)
    elif kind == '3d_axial':
        return Geometry3DAxial(**kwargs)
    else:
        raise ValueError(
            f"Unknown geometry '{kind}'. "
            f"Choose from: {sorted(cart_aliases | {'3d_axial'})}"
        )
