# -*- coding: utf-8 -*-
"""
gp_equation.py
Right-hand side of the Gross-Pitaevskii equation for an N-component BEC.

The Hamiltonian acting on component i is:

    H ψ_i = [ -½∇² + 2(σ_z)_i (-i∂/∂z) + (σ_z)_i² + Δ_i + q_i ] ψ_i
           + Ω/2 · ψ   (off-diagonal Rabi coupling, row i)
           + Σ_j g_ij n_j ψ_i
           + [ V_trap + V_lattice ] ψ_i

where:
    σ_z      : 1D array (n_comp,)  — spin-orbit coupling coefficients
    Δ        : 1D array (n_comp,)  — detuning diagonal, built from a single
                                     scalar δ as [-δ/2, δ/2, 3δ/2, 5δ/2, ...]
    q        : 1D array (n_comp,)  — quadratic Zeeman shifts per component
    Ω        : 2D array (n_comp, n_comp) — Rabi coupling matrix
    g_ij     : 2D array (n_comp, n_comp) — interaction matrix
    V_trap   : Σ_k (w_k r_k)² / 4   (harmonic trap, geometry-dependent)
    V_lattice: 2 * lattice_strength * sin²(k_l * z)

Time evolution:
    Real time      — dψ/dt = -i H[ψ]
    Imaginary time — dψ/dt = -(H[ψ] - μ ψ),  μ = <ψ|H|ψ>/<ψ|ψ>

Public API
----------
get_rhs(geo, n_components, sigma_z, detuning, q_zeeman, omega_r,
        g_matrix, n_atoms, wx, wy, wz, lattice_strength, k_l,
        mode) -> callable(t, psi)

    Takes all physical parameters, closes over them, and returns a
    function rhs(t, psi) ready to pass directly to the RK4 integrator.
    All time-dependent parameters are resolved inside the closure with
    no overhead from callable() checks at each timestep.

author: Federico Serrano
Physics and Astronomy Department
Washington State University
"""

import numpy as np


# =============================================================================
# Private builders
# =============================================================================

def _build_detuning_diagonal(n_components, detuning):
    """
    Δ_i = (2i - 1) * δ/2  →  [-δ/2, δ/2, 3δ/2, 5δ/2, ...]
    Returns a 1D array (static) or callable(t) -> 1D array (time-dependent).
    """
    i = np.arange(n_components)
    if callable(detuning):
        return lambda t: (2*i - 1) * detuning(t) / 2
    else:
        return (2*i - 1) * detuning / 2


def _build_trap_potential(geo, wx, wy, wz):
    """
    V_trap = Σ_k (w_k r_k)² / 4  on the geometry grid.
    Aligns (wx, wy, wz) to the last geo.ndim axes, ignoring unused ones.
    Returns a flat 1D array of shape (N_total,).
    """
    if geo.basis=="3d_axial":
        active_freqs = [np.sqrt(wx*wy), wz]
    else:
        active_freqs = [wx, wy, wz][-geo.ndim:]
    U = sum((w * r)**2 / 4 for w, r in zip(active_freqs, geo.grids))
    return U.reshape(-1)


def _build_lattice_potential(geo, lattice_strength, k_l):
    """
    V_lattice = 2 * Ω_l * sin²(k_l * z).
    The spatial profile sin²(k_l * z) is computed once.
    Returns a 1D array (static) or callable(t) -> 1D array (time-dependent).
    """
    profile = (2 * np.sin(k_l * geo.grids[-1])**2).reshape(-1)
    if callable(lattice_strength):
        return lambda t: lattice_strength(t) * profile
    else:
        return lattice_strength * profile


def _build_rabi_matrix(n_components, omega_r):
    """Nearest-neighbour tridiagonal Rabi coupling matrix."""
    ones = np.ones(n_components - 1)
    return omega_r * (np.diag(ones, k=1) + np.diag(ones, k=-1))


# =============================================================================
# Private Hamiltonian terms
# =============================================================================

def _H_sp(psi, geo, sigma_z, delta, q_zeeman, Omega):
    """
    Single-particle term:
        [-½∇² + 2σ_z(-i∂_z) + σ_z² + Δ + q] ψ  +  Ω/2 · ψ
    """
    T_psi  = geo.kinetic(psi)
    Kz_psi = -1j * geo.grad_z(psi)
    diag   = (2 * sigma_z[:, None] * Kz_psi
              + (sigma_z**2 + delta + q_zeeman)[:, None] * psi)
    return T_psi + diag + 0.5 * (Omega @ psi)


def _H_potential(psi, V_static):
    """Static external potential (trap + lattice already summed)."""
    return V_static[None, :] * psi


def _H_interaction(psi, g_matrix, n_atoms):
    """Mean-field interaction: Σ_j g_ij n_atoms |ψ_j|² ψ_i"""
    return n_atoms * (g_matrix @ np.abs(psi)**2) * psi


# =============================================================================
# Public API
# =============================================================================

def get_rhs(geo, n_components, sigma_z, detuning, q_zeeman, omega_r,
            g_matrix, n_atoms, wx=0.0, wy=0.0, wz=0.0,
            lattice_strength=0.0, k_l=0.0, mode='real'):
    """
    Build and return the RHS function rhs(t, psi) for the GP equation.

    All parameters are resolved once here. The returned callable closes
    over everything, so the integrator only ever calls rhs(t, psi) with
    no extra arguments and no runtime overhead from parameter resolution.

    Parameters
    ----------
    geo              : Geometry instance
    n_components     : int
    sigma_z          : 1D array (n_components,) — spin-orbit coefficients
    detuning         : float or callable(t) -> float
    q_zeeman         : 1D array (n_components,) — quadratic Zeeman per component
    omega_r          : float — Rabi frequency
    g_matrix         : 2D array (n_components, n_components)
    n_atoms          : float
    wx, wy, wz       : float — trap frequencies in recoil units
    lattice_strength : float or callable(t) -> float
    k_l              : float — lattice wavevector
    mode             : 'real' or 'imaginary'

    Returns
    -------
    rhs : callable(t, psi) -> flat complex array of shape (n_components * N_total,)
    """
    if mode not in ('real', 'imaginary'):
        raise ValueError(f"mode must be 'real' or 'imaginary', got '{mode}'")

    # --- build all static quantities once ---
    _sigma_z  = np.asarray(sigma_z)
    _q_zeeman = np.asarray(q_zeeman)
    _Omega    = _build_rabi_matrix(n_components, omega_r)
    _g_matrix = np.asarray(g_matrix)
    _V_trap   = _build_trap_potential(geo, wx, wy, wz)
    _dv       = geo.dv

    # --- resolve delta and V_lattice into plain callables ---
    _delta_fn     = _build_detuning_diagonal(n_components, detuning)
    _V_lattice_fn = _build_lattice_potential(geo, lattice_strength, k_l)

    # wrap static arrays in trivial lambdas so the closure body is uniform
    if not callable(_delta_fn):
        _delta_static = _delta_fn
        _delta_fn = lambda t: _delta_static

    if not callable(_V_lattice_fn):
        _V_lattice_static = _V_lattice_fn
        _V_lattice_fn = lambda t: _V_lattice_static

    # --- build the closure ---
    if mode == 'real':
        def rhs(t, psi):
            psi2d = psi.reshape(n_components, -1)
            H_psi = (_H_sp(psi2d, geo, _sigma_z,
                           _delta_fn(t), _q_zeeman, _Omega)
                   + _H_potential(psi2d, _V_trap + _V_lattice_fn(t))
                   + _H_interaction(psi2d, _g_matrix, n_atoms))
            return (-1j * H_psi).reshape(psi.shape)

    else:  # imaginary time
        def rhs(t, psi):
            psi2d = psi.reshape(n_components, -1)
            H_psi = (_H_sp(psi2d, geo, _sigma_z,
                           _delta_fn(0.0), _q_zeeman, _Omega)
                   + _H_potential(psi2d, _V_trap + _V_lattice_fn(0.0))
                   + _H_interaction(psi2d, _g_matrix, n_atoms))
            norm  = (_dv[None, :] * np.abs(psi2d)**2).sum()
            mu    = (_dv[None, :] * (psi2d.conj() * H_psi).real).sum() / norm
            return (-(H_psi - mu * psi2d)).reshape(psi.shape)

    return rhs
