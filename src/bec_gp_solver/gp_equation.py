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

With a projector P (PGPE), H[ψ] is replaced by P H[ψ] in both modes, so
a state in the c-field region stays there.

Public API
----------
get_rhs(geo, n_components, sigma_z, detuning, q_zeeman, omega_r,
        g_matrix, n_atoms, wx, wy, wz, lattice_strength, k_l,
        mode) -> callable(t, psi)

    Takes all physical parameters, closes over them, and returns a
    function rhs(t, psi) ready to pass directly to the RK4 integrator.
    All time-dependent parameters are resolved inside the closure with
    no overhead from callable() checks at each timestep.

get_ip_rhs(geo, ..., mode, projector) -> (rates, rhs_N)

    Same physics split for the interaction-picture integrator: the kinetic
    + spin-orbit part as exponent rates in the spectral basis, and the rest.

estimate_max_energy(geo, psi, times, projector, **rhs_kwargs) -> float

    Upper bound on the Hamiltonian spectrum, used to pick a stable dt.

author: Federico Serrano
Physics and Astronomy Department
Washington State University
"""

import numpy as np

from bec_gp_solver.projector import soc_dispersion, mode_energies


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


def _static_terms(geo, n_components, detuning, q_zeeman, omega_r, g_matrix,
                  wx, wy, wz, lattice_strength, k_l, homogeneous, projector):
    """
    Resolve every parameter once. Time-dependent terms are returned as
    callables of t (static ones wrapped in trivial lambdas) and a missing
    projector as the identity, so closure bodies need no branching.
    """
    delta_fn   = _build_detuning_diagonal(n_components, detuning)
    lattice_fn = _build_lattice_potential(geo, lattice_strength, k_l)
    if not callable(delta_fn):
        delta_static = delta_fn
        delta_fn = lambda t: delta_static
    if not callable(lattice_fn):
        lattice_static = lattice_fn
        lattice_fn = lambda t: lattice_static

    return dict(
        q_zeeman = np.asarray(q_zeeman),
        Omega    = _build_rabi_matrix(n_components, omega_r),
        g_matrix = np.asarray(g_matrix),
        V_trap   = (np.zeros_like(geo.dv) if homogeneous
                    else _build_trap_potential(geo, wx, wy, wz)),
        delta_fn   = delta_fn,
        lattice_fn = lattice_fn,
        project    = projector if projector is not None else (lambda psi: psi),
    )


# =============================================================================
# Public API
# =============================================================================

def get_rhs(geo, n_components, sigma_z, detuning, q_zeeman, omega_r,
            g_matrix, n_atoms, wx=0.0, wy=0.0, wz=0.0,
            lattice_strength=0.0, k_l=0.0, mode='real', homogeneous=False,
            projector=None):
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
    homogeneous      : bool — if True, set the trap potential to zero
                       (lattice is unaffected); default False
    projector        : Projector or None — if given, H[ψ] is projected onto
                       the c-field region (PGPE); default None (plain GPE)

    Returns
    -------
    rhs : callable(t, psi) -> flat complex array of shape (n_components * N_total,)
    """
    if mode not in ('real', 'imaginary'):
        raise ValueError(f"mode must be 'real' or 'imaginary', got '{mode}'")

    # --- build all static quantities once ---
    _sigma_z = np.asarray(sigma_z)
    _dv      = geo.dv
    T = _static_terms(geo, n_components, detuning, q_zeeman, omega_r, g_matrix,
                      wx, wy, wz, lattice_strength, k_l, homogeneous, projector)
    _q_zeeman, _Omega, _g_matrix, _V_trap = (T['q_zeeman'], T['Omega'],
                                             T['g_matrix'], T['V_trap'])
    _delta_fn, _V_lattice_fn, _project   = (T['delta_fn'], T['lattice_fn'],
                                             T['project'])

    # --- build the closure ---
    if mode == 'real':
        def rhs(t, psi):
            psi2d = psi.reshape(n_components, -1)
            H_psi = (_H_sp(psi2d, geo, _sigma_z,
                           _delta_fn(t), _q_zeeman, _Omega)
                   + _H_potential(psi2d, _V_trap + _V_lattice_fn(t))
                   + _H_interaction(psi2d, _g_matrix, n_atoms))
            return (-1j * _project(H_psi)).reshape(psi.shape)

    else:  # imaginary time
        def rhs(t, psi):
            psi2d = psi.reshape(n_components, -1)
            H_psi = (_H_sp(psi2d, geo, _sigma_z,
                           _delta_fn(0.0), _q_zeeman, _Omega)
                   + _H_potential(psi2d, _V_trap + _V_lattice_fn(0.0))
                   + _H_interaction(psi2d, _g_matrix, n_atoms))
            H_psi = _project(H_psi)
            norm  = (_dv[None, :] * np.abs(psi2d)**2).sum()
            mu    = (_dv[None, :] * (psi2d.conj() * H_psi).real).sum() / norm
            return (-(H_psi - mu * psi2d)).reshape(psi.shape)

    return rhs


def get_ip_rhs(geo, n_components, sigma_z, detuning, q_zeeman, omega_r,
               g_matrix, n_atoms, wx=0.0, wy=0.0, wz=0.0,
               lattice_strength=0.0, k_l=0.0, mode='real', homogeneous=False,
               projector=None):
    """
    Split the GP equation for the interaction-picture integrator (RK4IP):

        dψ/dt = L ψ + N(t, ψ)

    L is the kinetic + spin-orbit term, diagonal in the spectral basis of
    geo with eigenvalue ε_i(k) = k_⊥² + (k_z - σ_z,i)² per component
    (soc_dispersion). It is returned as the rates of exp(L τ):
    -iε in real time, -ε in imaginary time. N(t, ψ) is everything else —
    detuning, Zeeman, Rabi, trap, lattice and interactions — projected
    when a projector is given. L and P are diagonal in the same basis, so
    they commute and the split is exact for PGPE.

    L ψ + N(t, ψ) equals get_rhs(...)(t, ψ) for ψ in the c-field (exactly
    for Cartesian grids; to ~1e-9 for the axial grid, where L uses the
    orthogonalised Bessel basis while geo.kinetic() uses the raw DHT).

    Takes the same arguments as get_rhs().

    Returns
    -------
    rates : complex array (n_components, *geo.sizes) — exponent rates of L
    rhs_N : callable(t, psi) -> flat complex array, the remainder N(t, ψ)
    """
    if mode not in ('real', 'imaginary'):
        raise ValueError(f"mode must be 'real' or 'imaginary', got '{mode}'")

    eps = mode_energies(geo, soc_dispersion(sigma_z))
    _dv = geo.dv
    T = _static_terms(geo, n_components, detuning, q_zeeman, omega_r, g_matrix,
                      wx, wy, wz, lattice_strength, k_l, homogeneous, projector)
    _q_zeeman, _Omega, _g_matrix, _V_trap = (T['q_zeeman'], T['Omega'],
                                             T['g_matrix'], T['V_trap'])
    _delta_fn, _V_lattice_fn, _project   = (T['delta_fn'], T['lattice_fn'],
                                             T['project'])

    def _H_rest(t, psi2d):
        """H - (kinetic + spin-orbit): Δ + q + Ω/2 + V + interactions."""
        return ((_delta_fn(t) + _q_zeeman)[:, None] * psi2d
                + 0.5 * (_Omega @ psi2d)
                + _H_potential(psi2d, _V_trap + _V_lattice_fn(t))
                + _H_interaction(psi2d, _g_matrix, n_atoms))

    if mode == 'real':
        rates = -1j * eps

        def rhs_N(t, psi):
            psi2d = psi.reshape(n_components, -1)
            return (-1j * _project(_H_rest(t, psi2d))).reshape(psi.shape)

    else:  # imaginary time: μ needs the kinetic energy, taken spectrally
        rates = -eps.astype(complex)

        def rhs_N(t, psi):
            psi2d  = psi.reshape(n_components, -1)
            N_psi  = _project(_H_rest(0.0, psi2d))
            e_kin  = (eps * np.abs(geo.to_spectral(psi2d))**2).sum()
            norm   = (_dv[None, :] * np.abs(psi2d)**2).sum()
            mu     = (e_kin + (_dv[None, :] * (psi2d.conj() * N_psi).real).sum()) / norm
            return (-(N_psi - mu * psi2d)).reshape(psi.shape)

    return rates, rhs_N


def estimate_max_energy(geo, psi, times=(0.0,), projector=None, *,
                        n_components, sigma_z, detuning, q_zeeman, omega_r,
                        g_matrix, n_atoms, wx=0.0, wy=0.0, wz=0.0,
                        lattice_strength=0.0, k_l=0.0, homogeneous=False,
                        kinetic=True):
    """
    Upper bound on |E| for the Hamiltonian of get_rhs(), used to choose a
    stable time step. Takes the same keyword arguments as get_rhs().

    E_max = max ε(k)                (spin-orbit dispersion; within the
                                     c-field if a projector is given)
          + max |Δ| + max |q| + |Ω_R|
          + max (V_trap + V_lattice)
          + N max_i Σ_j |g_ij| max|ψ_j|²

    Each term bounds the spectral radius of its operator, so the sum bounds
    the whole Hamiltonian. Time-dependent parameters are evaluated at every
    entry of `times` (for linear ramps the start and end suffice). The
    interaction term uses the density of psi; it can grow during the
    evolution, which the safety factor of resolve_dt() should absorb.

    Parameters
    ----------
    geo       : Geometry instance
    psi       : complex array — state used for the interaction term,
                shape (n_components * N_total,) or (n_components, N_total)
    times     : iterable of float — times at which to evaluate the ramps
    projector : Projector or None
    kinetic   : bool — include the dispersion term. Set False for RK4IP,
                where the kinetic part is integrated exactly and only N(t, ψ)
                limits the step.

    Returns
    -------
    e_max : float, recoil energies
    """
    e_kin = 0.0
    if kinetic:
        energies = mode_energies(geo, soc_dispersion(sigma_z))
        e_kin    = (energies[projector.mask] if projector is not None
                    else energies).max()

    delta_fn   = _build_detuning_diagonal(n_components, detuning)
    lattice_fn = _build_lattice_potential(geo, lattice_strength, k_l)
    V_trap     = (np.zeros_like(geo.dv) if homogeneous
                  else _build_trap_potential(geo, wx, wy, wz))

    e_delta = max(np.abs(delta_fn(t) if callable(delta_fn) else delta_fn).max()
                  for t in times)
    e_pot   = max((V_trap + (lattice_fn(t) if callable(lattice_fn)
                             else lattice_fn)).max()
                  for t in times)

    density = np.abs(np.asarray(psi).reshape(n_components, -1))**2
    e_int   = n_atoms * (np.abs(np.asarray(g_matrix))
                         @ density.max(axis=1)).max()

    return float(e_kin + e_delta + np.abs(q_zeeman).max() + abs(omega_r)
                 + e_pot + e_int)
