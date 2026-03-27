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
from scipy.integrate import solve_ivp

from bec_gp_solver.config_loader import load_config, _compute_derived
from bec_gp_solver.gp_equation   import get_rhs
from bec_gp_solver._logging       import setup_logging
import logging
logger = logging.getLogger(__name__)


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


def _gs_filename(cfg):
    sweep = cfg['sweep']
    key   = _gs_cache_key(cfg)
    name  = (f"gs"
             f"_delta_{sweep['delta_start_hz']:.0f}hz"
             f"_omega_{sweep['omega_l_start']:.2f}"
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
    """Thomas-Fermi density profile as initial guess for imaginary-time cooling."""
    d      = _compute_derived(cfg)
    n_comp = cfg['system']['n_components']

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

    # distribute equally across first two components, third empty
    components = ([psi0 / np.sqrt(2)] * min(2, n_comp)
                + [np.zeros_like(psi0)] * max(0, n_comp - 2))
    return np.array(components).reshape(-1)


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


# =============================================================================
# Imaginary-time cooling
# =============================================================================

def run_imaginary_time(cfg, geo, rhs_kwargs, psi0, output_dir):
    """
    Run imaginary-time evolution to find the ground state.

    The Hamiltonian is time-independent in this mode: all time-dependent
    parameters (detuning ramp, lattice ramp) are evaluated at t=0 by
    the 'imaginary' mode of get_rhs().

    The result is normalised, saved to out_dir, and returned.

    Parameters
    ----------
    cfg        : raw config dict
    geo        : Geometry instance
    rhs_kwargs : dict — as returned by load_config(), passed to get_rhs()
    psi0       : complex array — initial guess, shape (n_comp * N_total,)
    out_dir    : str or Path — directory to save the ground state .npy file
    """
    gs_cfg = cfg['ground_state']
    t_end  = gs_cfg['steps'] * gs_cfg['step_size']

    logger.info(f"Running imaginary-time cooling: {gs_cfg['steps']} steps "
                f"× dt={gs_cfg['step_size']} = t_end={t_end:.1f} (recoil units)")

    rhs_imag = get_rhs(geo=geo, mode='imaginary', **rhs_kwargs)
    sol      = _solve(rhs_imag, psi0, t_span=(0.0, t_end), t_eval=[t_end])

    n_comp = cfg['system']['n_components']
    psi_gs = _renormalise(sol.y[:, -1], geo, n_comp)

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

def add_twa_noise(psi_gs, geo, n_atoms, n_comp, sample):
    """
    Add half-quantum of vacuum noise per mode (TWA approximation).

    Noise is generated in momentum space with amplitude 1/sqrt(4 N dvk),
    then transformed back to real space. A circular de-aliasing mask
    retains only modes within 2/3 of the maximum |k| (standard 2/3 rule),
    which prevents aliasing errors from the nonlinear interaction term.

    sample == 0 is treated as the mean-field trajectory: noise is set to
    zero so the ground state is evolved without any stochastic fluctuations.

    Parameters
    ----------
    psi_gs : complex array, shape (n_comp * N_total,)
    geo    : Geometry instance — must expose kgrids, dv, dvk
    n_atoms: float
    n_comp : int
    sample : int — noise realisation index; 0 means no noise
    """
    N_total  = geo.dv.shape[0]
    kr_, kz_ = geo.kgrids                          # both shape (nx, nz)

    # de-aliasing mask: keep modes inside 2/3 of the maximum |k| radius
    k2   = kr_**2 + kz_**2
    mask = (k2 <= k2.max() * (2.0 / 3.0)**2).reshape(N_total)

    # generate noise in k-space, one row per component
    noise_k = (
        np.random.normal(scale=1 / np.sqrt(4 * n_atoms * geo.dvk),
                         size=(n_comp, N_total))
      + 1j * np.random.normal(scale=1 / np.sqrt(4 * n_atoms * geo.dvk),
                               size=(n_comp, N_total))
    )

    # apply mask and transform back to real space
    noise = geo.inverse_transform(
        noise_k * mask[None, :]
    ).reshape(n_comp * N_total)

    if sample == 0:
        noise = np.zeros_like(noise)

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

    # --- imaginary-time mode: cool and exit ---
    if imag:
        run_imaginary_time(cfg, geo, rhs_kwargs, psi0, output_dir)
        return

    # --- real-time mode ---

    # --- optional TWA noise ---
    if cfg['simulation']['twa_noise'] and cfg['sweep']['sample'] > 0:
        psi0 = add_twa_noise(psi0,
                             geo,
                             d['n_atoms'],
                             d['n_comp'],
                             cfg['sweep']['sample'])

    sim       = cfg['simulation']
    sweep     = cfg['sweep']
    t_total   = sweep['total_time_ms'] * 1e-3 / d['t_unit']
    t_eval    = np.linspace(0.0, t_total, sim['t_frames'])

    logger.info(f"Real-time evolution: t_total={t_total:.2f}, frames={sim['t_frames']}")

    rhs_real = get_rhs(geo=geo, mode='real', **rhs_kwargs)
    sol   = _solve(rhs_real, psi0, t_span=(0.0, t_total), t_eval=t_eval)
    psi_t = sol.y   # complex array, shape (n_comp * N_total, n_frames)

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
