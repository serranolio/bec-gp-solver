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
   If not found, run imaginary-time cooling with solve_ivp and cache it.
4. Optionally add TWA quantum noise.
5. Run real-time evolution with solve_ivp (RK45, adaptive step size).
6. Compute and save the requested observables.

Usage
-----
    python run_simulation.py --config configs/sweep/job_0042.toml 
    --output-dir PATH/output/dir
    --gs-dir PATH/input/dir/


"""

import argparse
import hashlib
import tomllib
from pathlib import Path

import numpy as np
from numpy import pi
from scipy.integrate import solve_ivp

from bec_gp_solver.config_loader import load_config, _compute_derived
from bec_gp_solver.gp_equation   import get_rhs


# =============================================================================
# solve_ivp wrapper
# =============================================================================

def _solve(rhs, psi0, t_span, t_eval, rtol=1e-6, atol=1e-9):
    """
    Run solve_ivp with RK45 on a complex-valued problem.

    solve_ivp supports complex y0 natively as long as y0 is a complex
    array — no real/imaginary splitting needed.

    Parameters
    ----------
    rhs    : callable(t, psi) -> complex array
    psi0   : complex 1D array — initial state
    t_span : (t_start, t_end)
    t_eval : 1D array of output times (or None for final state only)
    rtol, atol : solver tolerances

    Returns
    -------
    sol : scipy OdeSolution result
    """
    sol = solve_ivp(
        fun    = rhs,
        t_span = t_span,
        y0     = psi0.astype(np.complex128),   # ensure complex dtype
        method = 'RK45',
        t_eval = t_eval,
        rtol   = rtol,
        atol   = atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol


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
    blob = str(sorted(str(physics))).encode()
    return hashlib.md5(blob).hexdigest()[:10]


def _gs_filename(cfg, gs_dir):
    sweep = cfg['sweep']
    key   = _gs_cache_key(cfg)
    name  = (f"gs"
             f"_delta_{sweep['delta_start_hz']:.0f}hz"
             f"_omega_{sweep['omega_l_start']:.2f}"
             f"_{key}.npy")
    return Path(gs_dir) / name


def _thomas_fermi_initial_state(cfg, geo):
    """Thomas-Fermi density profile as initial guess for imaginary-time cooling."""
    d      = _compute_derived(cfg)
    n_comp = cfg['system']['n_components']

    r_, z_ = geo.grids
    trap   = ((d['wx'] * r_)**2 + (d['wz'] * z_)**2) / 4
    psi0   = np.sqrt(np.maximum(d['mu'] - trap, 0) / d['g_ref'] / d['n_atoms']) + 0j
    psi0   = psi0.reshape(-1)
    norm   = (geo.dv * np.abs(psi0)**2).sum()
    psi0  /= np.sqrt(norm)

    # distribute equally across first two components, third empty
    components = ([psi0 / np.sqrt(2)] * min(2, n_comp)
                + [np.zeros_like(psi0)] * max(0, n_comp - 2))
    return np.array(components).reshape(-1)


def _renormalise(psi_gs, geo, n_comp):
    psi2d = psi_gs.reshape(n_comp, -1)
    norm  = (geo.dv[None, :] * np.abs(psi2d)**2).sum()
    return (psi2d / np.sqrt(norm)).reshape(-1)


def load_or_compute_ground_state(cfg, geo, rhs_kwargs, gs_dir):
    """
    Return the ground state wavefunction, loading from cache if available.
    If not found, run imaginary-time evolution with solve_ivp and cache it.

    The cache is shared across samples — sample index and ramp parameters
    do not affect the ground state.
    """
    gs_path = _gs_filename(cfg, gs_dir)

    if gs_path.exists():
        print(f"Ground state found: {gs_path.name}")
        return np.load(gs_path)

    print("Ground state not found. Running imaginary-time cooling ...")

    gs_cfg = cfg['ground_state']
    t_end  = gs_cfg['steps'] * gs_cfg['step_size']   # total imaginary time

    # imaginary-time RHS with fixed (t=0) detuning and lattice
    rhs_imag = get_rhs(geo=geo, mode='imaginary', **rhs_kwargs)

    psi0 = _thomas_fermi_initial_state(cfg, geo)

    sol = _solve(rhs_imag, psi0, t_span=(0.0, t_end), t_eval=[t_end])

    psi_gs = _renormalise(sol.y[:, -1], geo, cfg['system']['n_components'])

    Path(gs_dir).mkdir(parents=True, exist_ok=True)
    np.save(gs_path, psi_gs)
    print(f"Ground state saved: {gs_path.name}")
    return psi_gs


# =============================================================================
# TWA noise
# =============================================================================

def add_twa_noise(psi_gs, geo, n_atoms, n_comp):
    """Add half-quantum of vacuum noise per mode (TWA approximation)."""
    scale = 1 / np.sqrt(4 * n_atoms * np.tile(geo.dv, n_comp))
    noise = (np.random.normal(scale=scale, size=psi_gs.shape)
           + 1j * np.random.normal(scale=scale, size=psi_gs.shape))
    return psi_gs + noise


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
            f"_ol_{s['omega_l_start']:.2f}")


# =============================================================================
# Main
# =============================================================================

def run(config_path, output_dir='output', gs_dir='ground_states'):
    config_path = Path(config_path)

    # --- load config ---
    with open(config_path, 'rb') as f:
        cfg = tomllib.load(f)

    geo, rhs_kwargs = load_config(config_path)

    d = _compute_derived(cfg)

    # --- ground state ---
    psi_gs = load_or_compute_ground_state(cfg, geo, rhs_kwargs, gs_dir)

    # --- optional TWA noise ---
    if cfg['simulation']['twa_noise'] and cfg['sweep']['sample'] > 0:
        psi_gs = add_twa_noise(psi_gs, geo, d['n_atoms'], d['n_comp'])

    # --- real-time evolution ---
    sim       = cfg['simulation']
    sweep     = cfg['sweep']
    t_total   = sweep['total_time_ms'] * 1e-3 / d['t_unit']
    t_eval    = np.linspace(0.0, t_total, sim['t_frames'])

    print(f"Real-time evolution: t_total={t_total:.2f}, frames={sim['t_frames']}")

    rhs_real = get_rhs(geo=geo, mode='real', **rhs_kwargs)

    sol   = _solve(rhs_real, psi_gs, t_span=(0.0, t_total), t_eval=t_eval)
    psi_t = sol.y   # complex array, shape (n_comp * N_total, n_frames)

    # --- compute and save observables ---
    observables = compute_observables(psi_t, geo, cfg)

    stem     = build_output_stem(cfg)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for name, data in observables.items():
        fpath = out_path / f"{stem}_{name}.npy"
        np.save(fpath, data)
        print(f"Saved: {fpath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a GP simulation from a TOML config.')
    parser.add_argument('--config',
                        required=True,
                        help='Path to TOML config file')
    parser.add_argument('--output-dir', 
                        default='output',
                        help='Directory for output files')
    parser.add_argument('--gs-dir',
                        default='ground_states', 
                        help='Directory for ground state cache')
    args = parser.parse_args()

    run(args.config, args.output_dir, args.gs_dir)
