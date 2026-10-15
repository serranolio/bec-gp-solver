# -*- coding: utf-8 -*-
"""
spgpe.py
Stochastic projected Gross-Pitaevskii equation (SPGPE) for multi-component
condensates, following Bradley & Blakie, PRA 90, 023631 (2014),
arXiv:1406.2029 (Sec. V.C, two-component mixtures; Appendix A rates).

In recoil units (ħ = 1, ψ normalised to 1, Ψ = √N ψ), Stratonovich form,
for each component j (sums over k ≠ j):

  dψ_j = P{ -i (Hψ)_j dt }                                       Hamiltonian
       + P{ γ_j (μ - H)ψ|_j dt + dW_j }                          number damping
       + P{ -i V_j ψ_j dt + i ψ_j dU_j }                         energy damping
       + P{ Σ_k [ -i V_jk ψ_k dt + i ψ_k dU_jk ] }               particle exchange

  ⟨dW_j* dW_k⟩   = (2 γ_j k_BT / N) δ_jk δ(r - r') dt
  V_j(r)         =  N ∫ ε_j(r - r') ṅ_j(r') d³r',         ṅ_j = 2 Im[ψ_j* (PHψ)_j]
  V_jk(r)        =  N ∫ ε_jk(r - r') S_jk(r') d³r',
                    S_jk = Im[ψ_k* (PHψ)_j] + Im[ψ_j* (PHψ)_k]
  ε̃_j(k) = ℳ_j / |k|,   ε̃_jk(k) = ℳ_jk / |k|
  ⟨dU_j dU_l⟩    = 2 k_BT ε_j(r - r') δ_jl dt
  dU_jk = dU_kj,   ⟨dU_jk dU_jk⟩ = k_BT ε_jk(r - r') dt,   pairs independent

Exchange noise — a deliberate departure from the paper. Its Eq. 161 gives
two independent noises dU_12, dU_21, each of variance 2 k_BT ℳ_12. For a
real multiplicative noise i (Gψ) dU with ⟨dU dU⟩ = 2 k_BT κ ε, the
grand-canonical state is stationary only if the drift is
-i (Gψ) · 2κ ∫ ε Im[(Gψ)† Hψ] (energy damping, G = 1 per component,
κ = ℳ_j, satisfies this). Exchange has G = σ_x and drift
-i (σ_x ψ) ∫ ℳ_12 ε S with S = Im[(σ_x ψ)† Hψ], so it needs one shared
noise with κ = ℳ_12/2. The paper's two independent noises are exactly that
shared noise plus an antisymmetric part with no matching drift, which
heats the gas. Checked numerically on an ideal gas (tests/test_spgpe.py):
the independent noises overfill the c-field by ~30 %, a shared noise of
variance 2 k_BT ℳ_12 overfills the high modes and empties the low ones,
and the shared noise of variance k_BT ℳ_12 gives Rayleigh-Jeans.

ṅ_j is the rate of change of n_j under the projected Hamiltonian, the
paper's {L_C ψ_j* ψ_j}/iħ: without Raman coupling it is -∇·j_j (own current
only, Eqs. 151-152), and it vanishes for any stationary state. With Raman
coupling the kinetic current alone does not: a dressed ground state has
∇·j_j ≠ 0, balanced by Raman conversion, and energy damping on -∇·j_j would
act on equilibrium. Using P(Hψ) rather than Hψ keeps the grand-canonical
state of the projected dynamics stationary. S_jk is the paper's
-({L_C ψ_k* ψ_j} + {L_C ψ_j* ψ_k}) / 2i (Eqs. 155-158): Raman coupling drops
out of it identically, and it reproduces the paper's explicit V_12 with the
mean-field difference (a_12 - a_11)|ψ_1|² (the printed Eq. 159 has a_22
there, a typo).

Rates (Appendix A, equal masses, quasi-static reservoir, one μ and one
cutoff for all components; reservoir_rates()):

  γ_j  = (1 / π λ²) Ḡ Σ_k σ_jk                        (Eq. 149)
  ℳ_j  = N̄ (2 σ_jj + Σ_{k≠j} σ_jk)                   (Eq. 153)
  ℳ_jk = 2 σ_jk N̄                                     (Eq. 160)

  σ_jk = 4π a_jk² (1 + δ_jk),   λ² = 2πħ²/(m k_BT) = 4π / k_BT  (recoil units)
  N̄    = 1 / (e^{(ε_cut - μ)/k_BT} - 1)
  Ḡ    = x² Σ_{r≥0} y^r Φ(x, 1, r+1)²,  x = e^{(μ - ε_cut)/k_BT},
         y = e^{(μ - 2 ε_cut)/k_BT},     Φ the Lerch transcendent    (Eq. 147)

The paper writes these for two components; the sums over partner species k
are the natural extension to n components with density-density
interactions. Caveats: the theory is derived without coherent coupling
between components, so applying it with Raman coupling (single μ, bare-basis
reservoir) is an approximation; for effective 1D/2D grids the 3D kernel
1/|k| is applied to the reduced momenta.

Integration: the kinetic + spin-orbit part of -(i + γ_j) H is exact in the
RK4IP propagator (rates); everything else is in rhs_N. All noises are drawn
once per step by step_hook and held fixed across the RK stages
(Stratonovich). Convergence in dt is slower than RK4's dt⁴ once noise is
present: check results by halving dt.

Public API
----------
reservoir_rates(temperature, mu, e_cut, a_matrix)   -> γ_j, ℳ_j, ℳ_jk
get_spgpe_rhs(geo, projector, rng, temperature, mu, gamma, energy_damping,
              exchange_damping, **rhs_kwargs)         -> (rates, rhs_N, step_hook)
energy_damping_kernel(geo, M)                         -> M / |k| on the spectral grid
energy_damping_noise(geo, rng, temperature, h, √ε̃)    -> one real noise increment
energy_damping_potential(geo, psi, projector, energy_damping, **rhs_kwargs) -> V_j
estimate_max_rate(geo, psi, times, projector, gamma, mu, energy_damping,
                  exchange_damping, **rhs_kwargs)     -> bound on |rate| of rhs_N

author: Federico Serrano
Physics and Astronomy Department
Washington State University
"""

import numpy as np
from numpy import pi


from bec_gp_solver.gp_equation import (_static_terms, _make_H_rest,
                                       estimate_max_energy)
from bec_gp_solver.projector import mode_energies, soc_dispersion


# =============================================================================
# Reservoir rates (Bradley & Blakie 2014, Appendix A)
# =============================================================================

def lerch_phi1(z, a):
    """
    Lerch transcendent Φ(z, 1, a) = Σ_{k≥0} z^k / (a + k), for 0 ≤ z < 1.
    a may be an array.
    """
    if not 0 <= z < 1:
        raise ValueError(f"Φ(z, 1, a) needs 0 <= z < 1, got z={z}")
    if z == 0:
        return 1.0 / np.asarray(a, dtype=float)
    n_terms = int(np.ceil(40 / -np.log(z))) + 1
    k = np.arange(n_terms)
    a = np.asarray(a, dtype=float)
    return (z**k / (a[..., None] + k)).sum(axis=-1)


def reservoir_rates(temperature, mu, e_cut, a_matrix):
    """
    First-principles damping rates for a reservoir at temperature k_BT and
    chemical potential μ, above the cutoff e_cut (all recoil energies, same
    energy zero as the Hamiltonian), with scattering lengths a_matrix
    (n_comp × n_comp, units of 1/k_R).

    Returns
    -------
    dict with
      gamma            : array (n_comp,)  — number damping γ_j
      energy_damping   : array (n_comp,)  — ℳ_j, units of 1/k_R²
      exchange_damping : array (n_comp, n_comp), zero diagonal — ℳ_jk
      n_bar            : float — reservoir occupation at the cutoff
    """
    if temperature <= 0:
        raise ValueError("reservoir_rates needs temperature > 0")
    if mu >= e_cut:
        raise ValueError(f"the reservoir μ={mu} must lie below e_cut={e_cut}")

    a     = np.atleast_2d(np.asarray(a_matrix, dtype=float))
    n     = a.shape[0]
    sigma = 4 * pi * a**2 * (1 + np.eye(n))
    beta  = 1 / temperature
    lam2  = 4 * pi * temperature**-1                     # λ_dB² in 1/k_R²

    n_bar = 1 / np.expm1(beta * (e_cut - mu))

    x = np.exp(beta * (mu - e_cut))
    y = np.exp(beta * (mu - 2 * e_cut))
    r = np.arange(int(np.ceil(40 / -np.log(y))) + 1)
    G_bar = x**2 * (y**r * lerch_phi1(x, r + 1)**2).sum()

    off = sigma * (1 - np.eye(n))
    return dict(
        gamma            = G_bar / (pi * lam2) * sigma.sum(axis=1),
        energy_damping   = n_bar * (2 * np.diag(sigma) + off.sum(axis=1)),
        exchange_damping = 2 * n_bar * off,
        n_bar            = float(n_bar),
    )


# =============================================================================
# Kernels, noise and potentials
# =============================================================================

def energy_damping_kernel(geo, energy_damping):
    """
    ε̃(k) = ℳ/|k| on the spectral grid of geo, shape geo.sizes.

    The k = 0 mode is set to zero: it only adds a uniform potential (a
    global phase), and ∇·j has no k = 0 component on a periodic grid.
    """
    k_abs = np.sqrt(sum(k**2 for k in geo.spectral_kgrids))
    out   = np.zeros(geo.sizes)
    np.divide(energy_damping, k_abs, out=out, where=k_abs > 0)
    return out


def _convolve(geo, kernel_k, f):
    """Real field(s) f, (N_total,) or (n, N_total), convolved with a k-space kernel."""
    out = geo.from_spectral(kernel_k * geo.to_spectral(f)).real
    return out[0] if np.ndim(f) == 1 else out


def _density_rate(PH_psi, psi2d):
    """ṅ_j = 2 Im(ψ_j* (PHψ)_j) per component, shape (n_comp, N_total)."""
    return 2 * np.imag(psi2d.conj() * PH_psi)


def energy_damping_noise(geo, rng, temperature, h, sqrt_kernel_k):
    """
    Real noise increment dU for a step h, with
    ⟨dU(r) dU(r')⟩ = 2 k_BT h ε(r - r').

    A δ-correlated real field η (variance 1/dv per point) is convolved with
    √ε̃(k); in the orthonormal spectral basis each mode of dU then has
    variance 2 k_BT h ε̃(k).

    sqrt_kernel_k : √ε̃(k), e.g. np.sqrt(energy_damping_kernel(geo, ℳ))
    Returns real array (N_total,).
    """
    eta = rng.standard_normal(geo.dv.size) / np.sqrt(geo.dv)
    return np.sqrt(2 * temperature * h) * _convolve(geo, sqrt_kernel_k, eta)


def energy_damping_potential(geo, psi, projector, energy_damping, **rhs_kwargs):
    """
    V_j(r) = N ∫ ε_j(r - r') ṅ_j(r') d³r',  ṅ_j = 2 Im(ψ_j* (PHψ)_j),
    for ψ normalised to 1 (equals -N ∫ ε_j ∇·j_j without Raman coupling).

    psi            : complex array, (n_comp, N_total) or (n_comp * N_total,)
    projector      : Projector of the c-field
    energy_damping : ℳ_j, float or array (n_comp,)
    rhs_kwargs     : as get_rhs()
    Returns real array (n_comp, N_total).
    """
    n_comp = rhs_kwargs['n_components']
    psi2d  = np.asarray(psi).reshape(n_comp, -1)
    PH_psi = projector(_full_H(geo, psi2d.reshape(-1), **rhs_kwargs))
    M      = np.broadcast_to(np.asarray(energy_damping, float), (n_comp,))
    unit   = energy_damping_kernel(geo, 1.0)
    return rhs_kwargs['n_atoms'] * M[:, None] * _convolve(
        geo, unit, _density_rate(PH_psi, psi2d))


# =============================================================================
# Right-hand side
# =============================================================================

def _as_per_component(value, n_comp, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_comp, float(arr))
    if arr.shape != (n_comp,):
        raise ValueError(f"{name} must be a scalar or have {n_comp} entries")
    return arr


def _as_exchange_matrix(value, n_comp):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = float(arr) * (1 - np.eye(n_comp))
    if arr.shape != (n_comp, n_comp) or not np.allclose(arr, arr.T):
        raise ValueError(f"exchange_damping must be a scalar or a symmetric "
                         f"{n_comp}×{n_comp} matrix")
    return arr * (1 - np.eye(n_comp))


def get_spgpe_rhs(geo, projector, rng, temperature, mu, gamma, energy_damping,
                  exchange_damping=0.0, *, n_components, sigma_z, detuning,
                  q_zeeman, omega_r, g_matrix, n_atoms, wx=0.0, wy=0.0, wz=0.0,
                  lattice_strength=0.0, k_l=0.0, homogeneous=False):
    """
    Build the multi-component SPGPE for the RK4IP integrator.

    Parameters
    ----------
    geo              : Geometry instance
    projector        : Projector — the c-field region (required)
    rng              : np.random.Generator — source of all noises
    temperature      : float — k_BT of the reservoir, recoil energies
    mu               : float — reservoir chemical potential, recoil energies
    gamma            : γ_j, float or array (n_comp,)
    energy_damping   : ℳ_j, float or array (n_comp,), units of 1/k_R²
    exchange_damping : ℳ_jk, float (all pairs) or symmetric (n_comp, n_comp)
    remaining keyword arguments: as get_rhs() (rhs_kwargs of load_config)

    Returns
    -------
    rates     : complex array (n_comp, *sizes) — -(i + γ_j) ε_j(k), for
                LinearPropagator
    rhs_N     : callable(t, psi) — projected drift of the rest of the
                equation plus the current step's noise
    step_hook : callable(t, h) — draws the noise for a step of length h;
                pass to integrate(step_hook=...)
    """
    n_comp   = n_components
    sigma_z  = np.asarray(sigma_z, dtype=float)
    gamma    = _as_per_component(gamma, n_comp, 'gamma')
    M_e      = _as_per_component(energy_damping, n_comp, 'energy_damping')
    M_x      = _as_exchange_matrix(exchange_damping, n_comp)

    eps      = mode_energies(geo, soc_dispersion(sigma_z))
    terms    = _static_terms(geo, n_comp, detuning, q_zeeman, omega_r,
                             g_matrix, wx, wy, wz, lattice_strength, k_l,
                             homogeneous, projector)
    H_rest   = _make_H_rest(terms, n_atoms)
    unit     = energy_damping_kernel(geo, 1.0)            # 1/|k|
    sqrt_u   = np.sqrt(unit)
    dv       = geo.dv
    pairs    = [(j, k) for j in range(n_comp) for k in range(n_comp)
                if j != k and M_x[j, k] > 0]

    use_energy   = bool((M_e > 0).any())
    growth_noise = temperature > 0 and bool((gamma > 0).any())
    noisy        = temperature > 0

    # noise rates dW/h, dU/h of the current step, set by step_hook
    xi = {'W': np.zeros((n_comp, dv.size), dtype=complex),
          'U': np.zeros((n_comp, dv.size)),
          'X': {pair: np.zeros(dv.size) for pair in pairs}}

    def step_hook(t, h):
        if growth_noise:
            xi['W'] = projector.white_noise(
                rng, 2 * gamma * temperature * h / n_atoms) / h
        if noisy:
            for j in range(n_comp):
                if M_e[j] > 0:
                    xi['U'][j] = energy_damping_noise(
                        geo, rng, temperature, h, np.sqrt(M_e[j]) * sqrt_u) / h
            # one noise per pair, shared by both directions, variance
            # k_BT ℳ_jk ε (see module docstring)
            for (j, k) in pairs:
                if k < j:
                    xi['X'][(j, k)] = xi['X'][(k, j)]
                else:
                    xi['X'][(j, k)] = energy_damping_noise(
                        geo, rng, temperature, h,
                        np.sqrt(M_x[j, k] / 2) * sqrt_u) / h

    def rhs_N(t, psi):
        psi2d = psi.reshape(n_comp, -1)
        Hr    = H_rest(t, psi2d)
        out   = -(1j + gamma[:, None]) * Hr + (gamma * mu)[:, None] * psi2d

        if use_energy or pairs:
            K_psi  = geo.from_spectral(eps * geo.to_spectral(psi2d))
            PH_psi = projector(K_psi + Hr)

        if use_energy:
            V = n_atoms * M_e[:, None] * _convolve(
                geo, unit, _density_rate(PH_psi, psi2d))
            out += -1j * (V - xi['U']) * psi2d

        if pairs:
            for (j, k) in pairs:
                S    = (np.imag(psi2d[k].conj() * PH_psi[j])
                        + np.imag(psi2d[j].conj() * PH_psi[k]))
                V_jk = n_atoms * M_x[j, k] * _convolve(geo, unit, S)
                out[j] += -1j * (V_jk - xi['X'][(j, k)]) * psi2d[k]

        out += xi['W']
        return projector(out).reshape(psi.shape)

    rates = -(1j + gamma.reshape((-1,) + (1,) * len(geo.sizes))) * eps
    return rates, rhs_N, step_hook


def estimate_max_rate(geo, psi, times, projector, gamma, mu, energy_damping,
                      exchange_damping=0.0, **rhs_kwargs):
    """
    Upper bound on the rate |λ| of rhs_N, for resolve_dt():

        √(1 + γ_max²) E_max(N) + γ_max |μ|
      + Σ_{energy, exchange} [ max|V(ψ)| + 2 N ℳ max ε̃₁ max|ψ|² E_max(H) ]

    E_max(N) is estimate_max_energy(kinetic=False) and E_max(H) the bound
    including the c-field kinetic energy; ε̃₁ = 1/|k| on the grid, ℳ the
    largest ℳ_j (energy damping) or row sum of ℳ_jk (exchange). The second
    term bounds the linear response of V to deviations from psi — the
    stiffness, which can be large even when V(psi) itself is small, e.g.
    on a ground state.
    """
    n_comp = rhs_kwargs['n_components']
    gamma  = _as_per_component(gamma, n_comp, 'gamma')
    M_e    = _as_per_component(energy_damping, n_comp, 'energy_damping')
    M_x    = _as_exchange_matrix(exchange_damping, n_comp)

    e_rest = estimate_max_energy(geo, psi, times, projector, kinetic=False,
                                 **rhs_kwargs)
    rate   = np.sqrt(1 + gamma.max()**2) * e_rest + gamma.max() * abs(mu)
    if not ((M_e > 0).any() or (M_x > 0).any()):
        return float(rate)

    N      = rhs_kwargs['n_atoms']
    psi2d  = np.asarray(psi).reshape(n_comp, -1)
    e_full = estimate_max_energy(geo, psi, times, projector, **rhs_kwargs)
    unit   = energy_damping_kernel(geo, 1.0)
    stiff  = 2 * N * unit.max() * (np.abs(psi2d)**2).max() * e_full

    if (M_e > 0).any():
        rate += np.abs(energy_damping_potential(geo, psi, projector, M_e,
                                                **rhs_kwargs)).max()
        rate += stiff * M_e.max()
    if (M_x > 0).any():
        PH    = projector(_full_H(geo, psi, **rhs_kwargs))
        V_max = max(np.abs(N * _convolve(
                    geo, unit, np.imag(psi2d[k].conj() * PH[j])
                    + np.imag(psi2d[j].conj() * PH[k]))).max()
                    for j in range(n_comp) for k in range(n_comp) if j != k)
        rate += (V_max + stiff) * M_x.sum(axis=1).max()
    return float(rate)


def _full_H(geo, psi, **rhs_kwargs):
    """(Hψ) for the full Hamiltonian, (n_comp, N_total)."""
    kw      = dict(rhs_kwargs)
    sigma_z = np.asarray(kw['sigma_z'], dtype=float)
    n_comp  = sigma_z.size
    terms  = _static_terms(geo, n_comp, kw['detuning'], kw['q_zeeman'],
                           kw['omega_r'], kw['g_matrix'], kw.get('wx', 0.0),
                           kw.get('wy', 0.0), kw.get('wz', 0.0),
                           kw.get('lattice_strength', 0.0), kw.get('k_l', 0.0),
                           kw.get('homogeneous', False), None)
    psi2d  = np.asarray(psi).reshape(n_comp, -1)
    eps    = mode_energies(geo, soc_dispersion(sigma_z))
    K_psi  = geo.from_spectral(eps * geo.to_spectral(psi2d))
    return K_psi + _make_H_rest(terms, kw['n_atoms'])(0.0, psi2d)
