# -*- coding: utf-8 -*-
"""
projector.py
Projector onto the low-energy (c-field) region for projected GP methods.

The c-field region is the set of single-particle modes whose energy, given
by a system-dependent dispersion, lies below a cutoff e_cut. The projector

    P ψ = Σ_{modes with ε < e_cut} |mode⟩⟨mode|ψ⟩

is applied in the orthonormal spectral basis of the geometry (plane waves
for Cartesian grids, Bessel × plane waves for the axial geometry). Every
component gets its own region, so spin-dependent dispersions such as the
spin-orbit shifted parabolas (k_z - σ_z)² are handled naturally.

Anything that must be consistent with the c-field region — the projected
equation of motion, TWA and SPGPE noise — should go through this class.

Public API
----------
Projector(geo, dispersion, e_cut)     — callable, P(psi) -> projected psi
    .white_noise(rng, variance)       — Gaussian noise on the c-field modes
    .twa_noise(rng, n_atoms)          — Wigner vacuum: ½ quantum per mode
soc_dispersion(sigma_z)               — dispersion matching gp_equation._H_sp
mode_energies(geo, dispersion)        — dispersion on the spectral grid
alias_free_cutoff(geo, dispersion)    — largest e_cut free of aliasing

author: Federico Serrano
Physics and Astronomy Department
Washington State University
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)


# TWA is only reliable when the vacuum noise (n_modes/2 atoms) is a small
# fraction of the atom number: flag when n_modes >= MODE_FRACTION_WARN * N.
MODE_FRACTION_WARN = 0.1


# =============================================================================
# Dispersions
# =============================================================================

def soc_dispersion(sigma_z):
    """
    Spin-diagonal kinetic dispersion of the spin-orbit-coupled Hamiltonian:

        ε_i(k) = k_⊥² + (k_z - σ_z,i)²

    This is the eigenvalue of the kinetic + spin-orbit part of _H_sp
    (k² + 2σ_z k̂_z + σ_z², with the sign convention of geo.grad_z) on a
    plane wave in component i. Detuning, Zeeman and Rabi terms are left
    out so the region does not change during a detuning ramp.

    sigma_z : 1D array (n_comp,)
    Returns callable(k_perp2, k_z) -> array (n_comp, *sizes)
    """
    sigma_z = np.asarray(sigma_z, dtype=float)

    def dispersion(k_perp2, k_z):
        s = sigma_z.reshape((-1,) + (1,) * np.ndim(k_z))
        return k_perp2 + (k_z - s)**2

    return dispersion


def mode_energies(geo, dispersion):
    """Dispersion evaluated on the spectral grid: (n_comp, *sizes)."""
    kgrids  = geo.spectral_kgrids
    k_perp2 = sum(k**2 for k in kgrids[:-1]) + np.zeros(geo.sizes)
    return np.asarray(dispersion(k_perp2, kgrids[-1]))


def alias_free_cutoff(geo, dispersion):
    """
    Largest e_cut for which the c-field stays within k_max/2 on every axis,
    i.e. the lowest energy of any mode outside that half-box. Using it as
    e_cut gives a region with Projector.alias_free == True.
    """
    energies = mode_energies(geo, dispersion)
    outside  = np.zeros(geo.sizes, dtype=bool)
    for k in geo.spectral_kgrids:
        outside |= np.abs(k) > np.abs(k).max() / 2

    e_cut = float(energies[:, outside].min())
    if not (energies < e_cut).any():
        raise ValueError(
            "No alias-free c-field exists on this grid: the dispersion "
            "minimum lies beyond k_max/2. Refine the grid (smaller spacing)."
        )
    return e_cut


# =============================================================================
# Projector
# =============================================================================

class Projector:
    """
    Projector onto the modes with dispersion below e_cut.

    Parameters
    ----------
    geo        : Geometry instance — must implement to_spectral/from_spectral
    dispersion : callable(k_perp2, k_z) -> array (n_comp, *sizes), the
                 single-particle energy of each spectral mode per component
    e_cut      : float — energy cutoff, same units as the dispersion

    Attributes
    ----------
    mask       : bool array (n_comp, *sizes) — True inside the c-field region
    n_comp     : int
    n_modes    : int — total number of retained modes (all components)
    alias_free : bool — whether the cubic nonlinearity can be evaluated on
                 this grid without aliasing back into the region

    Notes
    -----
    P is exactly idempotent and Hermitian with respect to the grid inner
    product Σ dv ψ* φ, because to_spectral is unitary.
    """

    def __init__(self, geo, dispersion, e_cut):
        energies = mode_energies(geo, dispersion)
        if energies.ndim != len(geo.sizes) + 1:
            raise ValueError(
                f"dispersion must return shape (n_comp, *{geo.sizes}), "
                f"got {energies.shape}"
            )

        self.geo     = geo
        self.e_cut   = e_cut
        self.mask    = energies < e_cut
        self.n_comp  = self.mask.shape[0]
        self.n_modes = int(self.mask.sum())

        if self.n_modes == 0:
            raise ValueError(f"No modes below e_cut={e_cut}; the projector "
                             f"would remove the whole field.")

        self.alias_free = self._check_aliasing(geo.spectral_kgrids)

        logger.info(f"Projector: e_cut={e_cut}, {self.n_modes} modes "
                    f"({self.n_modes / self.mask.size:.1%} of the grid)")

    def __call__(self, psi):
        """
        Apply P to psi.

        psi : complex array, shape (n_comp, N_total) or (n_comp * N_total,)
        Returns array of same shape as psi.
        """
        shape0 = psi.shape
        psi_k  = self.geo.to_spectral(psi.reshape(self.n_comp, -1))
        return self.geo.from_spectral(psi_k * self.mask).reshape(shape0)

    # -------------------------------------------------------------------------
    # noise restricted to the c-field
    # -------------------------------------------------------------------------

    def white_noise(self, rng, variance):
        """
        Complex Gaussian field with ⟨|a|²⟩ = variance on every c-field mode
        and exactly zero outside. Since the spectral basis is orthonormal,
        this is P applied to δ-correlated real-space noise.

        rng      : np.random.Generator
        variance : float — mean |amplitude|² per mode
        Returns complex array, shape (n_comp, N_total).
        """
        scale = np.sqrt(variance / 2)
        a = (rng.normal(scale=scale, size=self.mask.shape)
             + 1j * rng.normal(scale=scale, size=self.mask.shape))
        return self.geo.from_spectral(a * self.mask)

    def twa_noise(self, rng, n_atoms):
        """
        Wigner vacuum noise: half a quantum per c-field mode.

        ψ is normalised to 1 (Ψ = √N ψ), so each mode amplitude of ψ has
        variance 1/(2N).

        Returns complex array, shape (n_comp, N_total).
        """
        return self.white_noise(rng, 1 / (2 * n_atoms))

    def check_mode_count(self, n_atoms):
        """
        Flag a c-field with too many modes for the TWA.

        Vacuum noise adds n_modes/2 atoms on average; the truncated Wigner
        expansion needs this to be small compared with N, i.e.
        n_modes << n_atoms. Logs a warning when
        n_modes >= MODE_FRACTION_WARN * n_atoms.

        Returns True if the mode count is acceptable.
        """
        ratio = self.n_modes / n_atoms
        if ratio >= MODE_FRACTION_WARN:
            logger.warning(
                f"Projector: {self.n_modes} c-field modes for N={n_atoms:.0f} "
                f"atoms (n_modes/N = {ratio:.2f} >= {MODE_FRACTION_WARN}); "
                f"vacuum noise adds ~{self.n_modes / 2:.0f} atoms and the TWA "
                f"is not reliable. Lower e_cut."
            )
            return False
        return True

    # -------------------------------------------------------------------------
    # aliasing
    # -------------------------------------------------------------------------

    def _check_aliasing(self, kgrids):
        """
        For a cubic nonlinearity the product ψ*ψψ reaches 3·k_c; it does not
        alias back into |k| ≤ k_c as long as k_c ≤ k_max/2 on every axis.
        (For the radial Bessel axis this is the analogous heuristic.)
        """
        alias_free = True
        for axis, k in enumerate(kgrids):
            k_grid = np.abs(k).max()
            k_cut  = np.abs(np.broadcast_to(k, self.mask.shape)[self.mask]).max()
            if k_cut > k_grid / 2:
                alias_free = False
                logger.warning(
                    f"Projector: on axis {axis} the c-field reaches "
                    f"|k|={k_cut:.3f} > k_max/2={k_grid / 2:.3f}; the "
                    f"nonlinear term will alias into the c-field. Lower "
                    f"e_cut or refine the grid."
                )
        return alias_free
