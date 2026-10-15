#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_simulation.py
Main simulation driver.

Workflow
--------
1. Load config from a TOML file.
2. Build geometry and RHS function.
3. Look up a cached ground state (shared across samples with same physics).
   If not found, run imaginary-time cooling and cache it.
4. With [simulation].mode = "pgpe" or "spgpe", project the initial state
   onto the c-field region; optionally add TWA quantum noise in that region.
5. Run real-time evolution with fixed-step RK4IP: GPE, PGPE (projector
   applied to the right-hand side) or SPGPE (PGPE plus number damping,
   energy damping and particle exchange with a thermal reservoir, see
   [reservoir]; optionally first equilibrated at the initial parameters).
   The kinetic + spin-orbit term
   is integrated exactly; the step is limited only by the remaining terms.
6. Compute and save the requested observables.

Usage
-----
    python run_simulation.py
    --config configs/sweep/job_0042.toml 
    --output-dir PATH/output/dir
    --gs-dir PATH/input/dir/

"""

import argparse
import hashlib
import tomllib
from pathlib import Path

import numpy as np
from numpy import pi
from bec_gp_solver.config_loader import (load_config, _compute_derived,
                                        build_projector)
from bec_gp_solver.gp_equation   import get_rhs, get_ip_rhs, estimate_max_energy
from bec_gp_solver.integrators   import integrate, resolve_dt, LinearPropagator
from bec_gp_solver.spgpe         import (get_spgpe_rhs, estimate_max_rate,
                                        reservoir_rates)
from bec_gp_solver._logging       import setup_logging
import logging
logger = logging.getLogger(__name__)


# =============================================================================
# Simulation mode
# =============================================================================

SIM_MODES       = ('gpe', 'pgpe', 'spgpe')
PROJECTED_MODES = ('pgpe', 'spgpe')     # evolve only the c-field
K_BOLTZMANN     = 1.380649e-23          # J/K (exact, SI)


def _sim_mode(cfg):
    """[simulation].mode: 'gpe' (default), 'pgpe' or 'spgpe'."""
    mode = cfg['simulation'].get('mode', 'gpe')
    if mode not in SIM_MODES:
        raise ValueError(f"[simulation].mode must be one of {SIM_MODES}, "
                         f"got '{mode}'")
    return mode


def _reservoir(cfg, d):
    """
    SPGPE reservoir from [reservoir], in recoil units. Fixed during the ramp.

        temperature_nk  -> temperature   : k_B T / E_R
        mu_hz           -> mu            : μ / E_R (same energy zero as H)
        equilibrate_ms  -> t_equilibrate : optional SPGPE hold at the t = 0
                                           parameters before the ramp

    Energy damping and particle exchange are only supported on 3D grids; on
    1d_cart/2d_cart they must be overridden to 0 (number damping only).

    The damping rates γ_j, ℳ_j and ℳ_jk are computed from T, μ, the cutoff
    [projector].e_cut and the scattering lengths [system].a_matrix
    (spgpe.reservoir_rates, Bradley & Blakie 2014). Each can be overridden
    with optional keys gamma, energy_damping (scalar or per component) and
    exchange_damping (scalar for all pairs, or a symmetric matrix).
    """
    res      = cfg.get('reservoir')
    required = ('temperature_nk', 'mu_hz')
    missing  = [k for k in required if res is None or k not in res]
    if missing:
        raise ValueError(f"mode = 'spgpe' needs [reservoir] with {required}; "
                         f"missing: {missing}")

    f_recoil    = d['f_recoil']
    h_si        = 2 * pi * cfg['constants']['hbar_si']
    temperature = K_BOLTZMANN * res['temperature_nk'] * 1e-9 / (h_si * f_recoil)
    mu          = res['mu_hz'] / f_recoil

    a_recoil = (np.array(cfg['system']['a_matrix'], dtype=float)
                * cfg['constants']['a_si'] / d['l_unit'])
    rates    = reservoir_rates(temperature, mu, cfg['projector']['e_cut'], a_recoil)
    for key in ('gamma', 'energy_damping', 'exchange_damping'):
        if key in res:
            rates[key] = np.asarray(res[key], dtype=float)

    # energy damping and particle exchange use the 3D kernel 1/|k|; on
    # effective 1D/2D grids that kernel is not the reduced one (it diverges
    # at small k in 1D and misses the transverse-area factor), so only number
    # damping is supported there for now
    reduced = cfg['geometry']['kind'] in ('1d_cart', '2d_cart')
    if reduced and (np.any(rates['energy_damping'] > 0)
                    or np.any(rates['exchange_damping'] > 0)):
        raise ValueError(
            f"SPGPE energy damping and particle exchange are only implemented "
            f"for 3D grids (3d_axial, 3d_cart), not '{cfg['geometry']['kind']}'. "
            f"Set energy_damping = 0.0 and exchange_damping = 0.0 in "
            f"[reservoir] to run with number damping only."
        )

    return dict(
        temperature      = temperature,
        mu               = mu,
        gamma            = rates['gamma'],
        energy_damping   = rates['energy_damping'],
        exchange_damping = rates['exchange_damping'],
        n_bar            = rates['n_bar'],
        t_equilibrate    = res.get('equilibrate_ms', 0.0) * 1e-3 / d['t_unit'],
    )


# =============================================================================
# Ground state
# =============================================================================

def _gs_cache_key(cfg):
    """
    MD5 hash of the parameters that determine the ground state:
    geometry, trap, system, spin_orbit, and initial lattice/detuning.
    Sample number and ramp parameters are excluded — they don't affect
    the ground state.
    """
    sweep   = cfg['sweep']
    physics = {
        'geometry'   : cfg['geometry'],
        'trap'       : cfg['trap'],
        'system'     : cfg['system'],
        'spin_orbit' : cfg['spin_orbit'],
        'omega_l'    : sweep['omega_l_start'],
        'delta_hz'   : sweep['delta_start_hz'],
    }
    # a PGPE ground state depends on the c-field region; GPE keys are left
    # unchanged so existing cached ground states are still found
    # (PGPE and SPGPE share the projected ground state)
    if _sim_mode(cfg) in PROJECTED_MODES:
        physics['e_cut'] = cfg['projector']['e_cut']
    blob = str(sorted(str(physics))).encode()
    return hashlib.md5(blob).hexdigest()[:10]


def _gs_filename(cfg):
    sweep   = cfg['sweep']
    omega_r = cfg['spin_orbit']['omega_r']
    key     = _gs_cache_key(cfg)
    name  = (f"gs"
             f"_delta_{sweep['delta_start_hz']:.2f}hz"
             f"_omegaL_{sweep['omega_l_start']:.2f}"
             f"_omegaR_{omega_r:.2f}"
             f"_{key}.npy")
    return name


def _find_ground_state(cfg, gs_dir):
    """
    Locate a cached ground state in gs_dir by reconstructing its exact filename.

    The filename is derived from the same hash used by run_imaginary_time(),
    so any config with identical initial physics will resolve to the same file.

    Parameters
    ----------
    cfg    : raw config dict
    gs_dir : str or Path

    Returns
    -------
    Path to the ground state file.

    Raises
    ------
    FileNotFoundError : if the expected file does not exist in gs_dir.
    """
    gs_path = Path(gs_dir) / _gs_filename(cfg)
    if not gs_path.exists():
        raise FileNotFoundError(
            f"Ground state not found: '{gs_path}'. "
            f"Run the ground state sweep (submit_gs.sh) first."
        )
    return gs_path


def _thomas_fermi_initial_state(cfg, geo):
    """Thomas-Fermi density profile as initial guess for imaginary-time cooling.

    A Thomas-Fermi profile ``psi0`` is broadcast across the components and
    scaled by per-component weights. Two optional keys in the
    ``[ground_state]`` config section control the spinor structure:

    initial_weights : list of float, length ``n_components`` (default all ones)
        Component ``k`` is seeded with ``initial_weights[k] * psi0``. The
        default ``[1, 1, ...]`` reproduces ``[psi0, psi0, ...]``.
    modulation : bool (default False)
        If True, imprint a plane-wave phase ``e^{-i k_l z}`` on component 0 and
        ``e^{+i k_l z}`` on component 1 (stripe-like guess). Other components
        are left unmodulated.

    The result is not normalised here; ``prepare_initial_state`` renormalises.
    """
    d      = _compute_derived(cfg)
    n_comp = cfg['system']['n_components']

    gs_cfg     = cfg['ground_state']
    weights    = gs_cfg.get('initial_weights', [1.0] * n_comp)
    modulation = gs_cfg.get('modulation', False)

    weights = np.asarray(weights, dtype=complex)
    if weights.size != n_comp:
        raise ValueError(
            f"ground_state.initial_weights has {weights.size} entries but "
            f"n_components={n_comp}."
        )

    #r_, z_ = geo.grids
    if geo.basis=="3d_axial":
        active_freqs = [np.sqrt(d['wx']*d['wy']), d['wz']]
    else:
        active_freqs = [d['wx'], d['wy'], d['wz']][-geo.ndim:]
    trap = sum((w*r)**2/4 for w, r, in zip(active_freqs, geo.grids))
    psi0   = np.sqrt(np.maximum(d['mu'] - trap, 0) / d['g_ref'] / d['n_atoms']) + 0j
    psi0   = psi0.reshape(-1)
    norm   = (geo.dv * np.abs(psi0)**2).sum()
    psi0  /= np.sqrt(norm)

    # broadcast the profile across components, scaled by the weights
    components = weights[:, None] * psi0[None, :]

    # optional plane-wave phase on components 0 and 1 only
    if modulation:
        z = geo.grids[-1].reshape(-1)
        components[0] *= np.exp(-1j * d['k_l'] * z)
        if n_comp > 1:
            components[1] *= np.exp(1j * d['k_l'] * z)

    return components.reshape(-1)


def _renormalise(psi_gs, geo, n_comp):
    psi2d = psi_gs.reshape(n_comp, -1)
    norm  = (geo.dv[None, :] * np.abs(psi2d)**2).sum()
    return (psi2d / np.sqrt(norm)).reshape(-1)


def _load_initial_state(path, geo, n_comp):
    """
    Load a wavefunction from a .npy file and validate its shape.

    Expected shape: (n_comp * N_total,) — the same flat layout used
    throughout the solver.
    """
    psi = np.load(path).astype(np.complex128)
    expected = n_comp * len(geo.dv)
    if psi.size != expected:
        raise ValueError(
            f"Initial state loaded from '{path}' has {psi.size} elements "
            f"but the geometry requires {expected} (n_comp={n_comp}, "
            f"N_total={len(geo.dv)})."
        )
    return psi.reshape(-1)


def prepare_initial_state(cfg, geo, initial_state_path=None):
    """
    Return the initial state for real-time evolution or imaginary-time
    cooling, depending on what the caller provides.

    Priority:
        1. Load from --initial-state path if provided.
        2. Fall back to a Thomas-Fermi profile.

    The returned state is always normalised to unit norm.
    """
    n_comp = cfg['system']['n_components']

    if initial_state_path is not None:
        logger.info(f"Loading initial state from: {initial_state_path}")
        psi0 = _load_initial_state(initial_state_path, geo, n_comp)
    else:
        logger.info("No initial state provided — using Thomas-Fermi profile.")
        psi0 = _thomas_fermi_initial_state(cfg, geo)

    return _renormalise(psi0, geo, n_comp)


def apply_momentum_kick(psi, geo, n_comp, k_kick):
    """
    Split every component into two momentum components ±ħ k_kick along z:

        ψ -> ψ (e^{-i k_kick z} + e^{+i k_kick z}) / √2 = √2 cos(k_kick z) ψ

    k_kick is in units of k_R (k_kick = 2 gives momenta ±2ħk_R). The result
    is renormalised to unit norm.
    """
    z    = geo.grids[-1].reshape(-1)
    kick = (np.exp(-1j * k_kick * z) + np.exp(1j * k_kick * z)) / np.sqrt(2)
    out  = (psi.reshape(n_comp, -1) * kick[None, :]).reshape(-1)
    return _renormalise(out, geo, n_comp)


def _evolution_kwargs(cfg, rhs_kwargs):
    """
    rhs_kwargs for real-time evolution: with [trap].release = true the trap
    is switched off for the evolution only (the ground state is still
    cooled in the trap).
    """
    if cfg['trap'].get('release', False):
        logger.info("Trap released: real-time evolution without the trap.")
        return dict(rhs_kwargs, homogeneous=True)
    return rhs_kwargs


# =============================================================================
# Imaginary-time cooling
# =============================================================================

def run_imaginary_time(cfg, geo, rhs_kwargs, psi0, output_dir, projector=None):
    """
    Run imaginary-time evolution to find the ground state.

    The Hamiltonian is time-independent in this mode: all time-dependent
    parameters (detuning ramp, lattice ramp) are evaluated at t=0 by
    the 'imaginary' mode of get_ip_rhs().

    Integrated with fixed-step RK4IP, dt = [ground_state].step_size, which
    is checked against the stability limit. With a projector (PGPE) the
    cooling stays in the c-field region and gives the projected ground state.

    The result is normalised, saved to out_dir, and returned.

    Parameters
    ----------
    cfg        : raw config dict
    geo        : Geometry instance
    rhs_kwargs : dict — as returned by load_config(), passed to get_ip_rhs()
    psi0       : complex array — initial guess, shape (n_comp * N_total,)
    out_dir    : str or Path — directory to save the ground state .npy file
    projector  : Projector or None — PGPE projector, None for plain GPE
    """
    gs_cfg = cfg['ground_state']
    t_end  = gs_cfg['steps'] * gs_cfg['step_size']

    logger.info(f"Running imaginary-time cooling: {gs_cfg['steps']} steps "
                f"× dt={gs_cfg['step_size']} = t_end={t_end:.1f} (recoil units)")

    e_max        = estimate_max_energy(geo, psi0, (0.0,), projector,
                                       kinetic=False, **rhs_kwargs)
    dt           = resolve_dt(gs_cfg['step_size'], e_max)
    rates, rhs_N = get_ip_rhs(geo=geo, mode='imaginary', projector=projector,
                              **rhs_kwargs)
    psi_t        = integrate(rhs_N, psi0, t_eval=[t_end], dt=dt,
                             propagator=LinearPropagator(geo, rates))

    n_comp = cfg['system']['n_components']
    psi_gs = _renormalise(psi_t[:, -1], geo, n_comp)

    # chemical potential μ = ⟨ψ|H|ψ⟩, e.g. to set [reservoir].mu_hz for SPGPE
    H_psi = 1j * get_rhs(geo=geo, mode='real', projector=projector,
                         **rhs_kwargs)(0.0, psi_gs)
    mu_gs = (np.tile(geo.dv, n_comp) * (psi_gs.conj() * H_psi).real).sum()
    f_rec = cfg['constants']['f_recoil']
    logger.info(f"Ground-state chemical potential: mu = {mu_gs:.6f} E_R "
                f"= {mu_gs * f_rec:.2f} Hz")

    # convergence: a stationary state has (P)Hψ = μψ
    residual = np.abs(H_psi - mu_gs * psi_gs).max() / np.abs(mu_gs * psi_gs).max()
    logger.info(f"Ground-state residual max|(P)Hψ - μψ| / max|μψ| = {residual:.2e}")
    if residual > 1e-3:
        logger.warning("Ground state not converged (residual > 1e-3): "
                       "increase [ground_state].steps.")

    # save with a content-based filename so it can be used as --initial-state
    # in subsequent real-time runs
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    gs_path  = out_path / _gs_filename(cfg)
    np.save(gs_path, psi_gs)
    logger.info(f"Ground state saved: {gs_path}")

    return psi_gs


# =============================================================================
# TWA noise
# =============================================================================

def add_twa_noise(psi_gs, projector, n_atoms, rng):
    """
    Add Wigner vacuum noise (TWA): half a quantum per c-field mode.

    The noise lives exactly in the c-field region defined by the projector
    (spin-orbit dispersion below e_cut), so its mode content matches the
    projected dynamics. psi_gs itself is not projected here.

    Parameters
    ----------
    psi_gs    : complex array, shape (n_comp * N_total,)
    projector : Projector — from config_loader.build_projector()
    n_atoms   : float
    rng       : np.random.Generator
    """
    noise = projector.twa_noise(rng, n_atoms)
    return psi_gs + noise.reshape(psi_gs.shape)

# =============================================================================
# Observables
# =============================================================================

def compute_observables(psi_t, geo, cfg):
    """
    Compute the requested observables from the time-evolved wavefunction.

    Parameters
    ----------
    psi_t : complex array, shape (n_comp * N_total, n_frames)
    geo   : Geometry instance
    cfg   : full config dict

    Returns
    -------
    dict mapping observable name -> array
    """
    requested = cfg['simulation']['observables']
    n_comp    = cfg['system']['n_components']
    n_frames  = psi_t.shape[1]
    N_total   = len(geo.dv)

    psi3d = psi_t.reshape(n_comp, N_total, n_frames)
    n_i   = (geo.dv[None, :, None] * np.abs(psi3d)**2).sum(axis=1)  # (n_comp, n_frames)

    results = {}

    if 'populations' in requested:
        results['populations'] = n_i

    if 'polarization' in requested:
        results['polarization'] = (n_i[0] - n_i[1]) / (n_i[0] + n_i[1])

    if 'wavefunction' in requested:
        results['wavefunction'] = psi_t

    if 'momentum_density' in requested:
        rho_k = np.zeros((n_comp, N_total, n_frames), dtype=float)
        for f in range(n_frames):
            psi_k = geo.forward_transform(psi3d[:, :, f])
            rho_k[:, :, f] = np.abs(psi_k)**2
        results['momentum_density'] = rho_k

    unknown = set(requested) - set(results)
    if unknown:
        raise ValueError(
            f"Unknown observables: {unknown}. "
            f"Supported: populations, polarization, wavefunction, momentum_density"
        )

    return results


# =============================================================================
# Output filename
# =============================================================================

def build_output_stem(cfg):
    s = cfg['sweep']
    return (f"sample_{s['sample']}"
            f"_ramp_{s['ramp_time_ms']:.0f}ms"
            f"_di_{s['delta_start_hz']:.0f}hz"
            f"_df_{s['delta_end_hz']:.0f}hz"
            f"_omli_{s['omega_l_start']:.2f}"
            f"_omlf_{s['omega_l_end']:.2f}")


# =============================================================================
# Main
# =============================================================================

def run(config_path,
        output_dir='output',
        gs_dir=None,
        initial_state_path=None,
        imag=False):
    """
    Main entry point.

    Parameters
    ----------
    config_path        : str or Path — TOML config file
    output_dir         : str or Path — directory for all output files
    gs_dir             : str or Path or None
                         Directory containing cached ground states produced by
                         --imag runs. If provided and initial_state_path is
                         None, the ground state is located automatically by
                         reconstructing its filename from the config.
    initial_state_path : str or Path or None
                         Explicit path to a .npy wavefunction. Takes priority
                         over gs_dir. For --imag runs this is the first guess
                         for cooling. If both are None, a Thomas-Fermi profile
                         is used.
    imag               : bool
                         If True, run imaginary-time cooling and save the
                         ground state to output_dir, then exit.
                         If False (default), run real-time evolution.
    """
    config_path = Path(config_path)

    # --- load config ---
    with open(config_path, 'rb') as f:
        cfg = tomllib.load(f)

    geo, rhs_kwargs = load_config(config_path)

    d = _compute_derived(cfg)

    # --- resolve initial state ---
    if initial_state_path is None and gs_dir is not None and not imag:
        initial_state_path = _find_ground_state(cfg, gs_dir)

    # --- prepare initial state ---
    psi0 = prepare_initial_state(cfg, geo, initial_state_path)

    # --- c-field projector: needed for PGPE/SPGPE and for TWA noise ---
    # built for every sample so a missing e_cut or a flagged cutoff shows
    # up in the sample-0 job as well
    sim       = cfg['simulation']
    mode      = _sim_mode(cfg)
    projector = None
    if mode in PROJECTED_MODES or sim['twa_noise']:
        projector = build_projector(cfg, geo)

    # PGPE/SPGPE evolve only the c-field: start from the projected state
    rhs_projector = projector if mode in PROJECTED_MODES else None

    # --- imaginary-time mode: cool and exit ---
    if imag:
        if rhs_projector is not None:
            psi0 = _renormalise(projector(psi0), geo, d['n_comp'])
        run_imaginary_time(cfg, geo, rhs_kwargs, psi0, output_dir,
                           projector=rhs_projector)
        return

    # --- real-time mode ---

    # optional momentum kick of the initial state, then projection
    k_kick = cfg.get('initial_state', {}).get('momentum_kick')
    if k_kick:
        logger.info(f"Momentum kick: ψ -> ψ (e^(-i {k_kick} z) + e^(+i {k_kick} z))/√2")
        psi0 = apply_momentum_kick(psi0, geo, d['n_comp'], k_kick)
    if rhs_projector is not None:
        psi0 = _renormalise(projector(psi0), geo, d['n_comp'])

    rhs_kwargs = _evolution_kwargs(cfg, rhs_kwargs)

    # one random stream per run (TWA and SPGPE noise); with
    # [simulation].seed set, each sample gets a reproducible stream
    sample = cfg['sweep']['sample']
    seed   = sim.get('seed')
    rng    = np.random.default_rng(None if seed is None else [seed, sample])

    # --- optional TWA noise (sample 0 is the noiseless mean-field run) ---
    if sim['twa_noise'] and sample > 0:
        psi0 = add_twa_noise(psi0, projector, d['n_atoms'], rng)

    sweep     = cfg['sweep']
    t_total   = sweep['total_time_ms'] * 1e-3 / d['t_unit']
    t_eval    = np.linspace(0.0, t_total, sim['t_frames'])

    logger.info(f"Real-time evolution ({mode}): t_total={t_total:.2f}, "
                f"frames={sim['t_frames']}")

    # RK4IP: kinetic + spin-orbit exact, so only the rest limits dt.
    # Ramps are linear, so their extremes are at the start and the end.
    step_hook = None
    if mode == 'spgpe':
        res = _reservoir(cfg, d)
        logger.info(f"Reservoir (recoil units): k_BT={res['temperature']:.4g}, "
                    f"mu={res['mu']:.4g}, occupation at cutoff={res['n_bar']:.3g}, "
                    f"equilibration t={res['t_equilibrate']:.4g}")
        logger.info(f"  gamma_j={np.round(res['gamma'], 8).tolist()}, "
                    f"M_j={np.round(res['energy_damping'], 8).tolist()}, "
                    f"M_jk={np.round(res['exchange_damping'], 8).tolist()}")
        e_max = estimate_max_rate(geo, psi0, (0.0, t_total), projector,
                                  res['gamma'], res['mu'], res['energy_damping'],
                                  res['exchange_damping'], **rhs_kwargs)
        dt    = resolve_dt(sim.get('dt'), e_max,
                           t_total=t_total + res['t_equilibrate'])
        rates, rhs_N, step_hook = get_spgpe_rhs(
            geo, projector, rng, res['temperature'], res['mu'], res['gamma'],
            res['energy_damping'], res['exchange_damping'], **rhs_kwargs)

        # thermalise with the reservoir at the t = 0 parameters
        if res['t_equilibrate'] > 0:
            psi0 = integrate(lambda t, psi: rhs_N(0.0, psi), psi0,
                             t_eval=[res['t_equilibrate']], dt=dt,
                             propagator=LinearPropagator(geo, rates),
                             step_hook=step_hook)[:, -1]
    else:
        e_max        = estimate_max_energy(geo, psi0, (0.0, t_total), rhs_projector,
                                           kinetic=False, **rhs_kwargs)
        dt           = resolve_dt(sim.get('dt'), e_max, t_total=t_total)
        rates, rhs_N = get_ip_rhs(geo=geo, mode='real', projector=rhs_projector,
                                  **rhs_kwargs)

    psi_t = integrate(rhs_N, psi0, t_eval=t_eval, dt=dt,
                      propagator=LinearPropagator(geo, rates),
                      step_hook=step_hook)
    # psi_t: complex array, shape (n_comp * N_total, n_frames)

    # --- compute and save observables ---
    observables = compute_observables(psi_t, geo, cfg)

    stem     = build_output_stem(cfg)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for name, data in observables.items():
        fpath = out_path / f"{stem}_{name}.npy"
        np.save(fpath, data)
        logger.info(f"Saved: {fpath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='GP equation solver for spin-orbit-coupled BECs.',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # imaginary-time cooling from Thomas-Fermi guess
  python run_simulation.py --config base.toml --imag --output-dir ground_states/

  # imaginary-time cooling from a previous result
  python run_simulation.py --config base.toml --imag --initial-state
  prev_gs.npy --output-dir ground_states/

  # real-time evolution from a ground state
  python run_simulation.py --config job_0042.toml --initial-state
  ground_states/gs_delta_5000hz_omega_0.20_a3f82c9d1b.npy --output-dir output/

  # real-time evolution from Thomas-Fermi (no prior ground state)
  python run_simulation.py --config job_0042.toml --output-dir output/
        """
            )
    parser.add_argument(
        '--config', required=True,
        help='Path to TOML config file.'
    )
    parser.add_argument(
        '--initial-state', default=None, dest='initial_state',
        help='Explicit path to a .npy wavefunction to use as the initial '
             'state. Takes priority over --gs-dir. For --imag runs this is '
             'the first guess for cooling. If omitted, a Thomas-Fermi profile '
             'is used.'
    )
    parser.add_argument(
        '--gs-dir', default=None, dest='gs_dir',
        help='Directory containing cached ground states produced by --imag '
             'runs. The matching file is found automatically from the config '
             'parameters. Ignored if --initial-state is provided or --imag '
             'is set.'
    )
    parser.add_argument(
        '--output-dir', default='output', dest='output_dir',
        help='Directory for output files (default: output/).'
    )
    parser.add_argument(
        '--imag', action='store_true',
        help='Run imaginary-time cooling instead of real-time evolution. '
             'Saves the ground state to --out-dir and exits.'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable DEBUG-level logging. Default is INFO'
    )
    args = parser.parse_args()
    setup_logging(
        log_file = Path(args.output_dir) / f"run_{Path(args.config).stem}.log",
        verbose  = args.verbose,
    )

    run(
        config_path        = args.config,
        output_dir         = args.output_dir,
        gs_dir             = args.gs_dir,
        initial_state_path = args.initial_state,
        imag               = args.imag,
    )
