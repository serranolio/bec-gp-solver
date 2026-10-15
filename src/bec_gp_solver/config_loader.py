# -*- coding: utf-8 -*-
"""
config_loader.py
Load a TOML configuration file and build all inputs needed by get_rhs().

The TOML file is structured as:
    [constants]   — physical constants
    [system]      — n_atoms, n_components, scattering lengths, sigma_z, q_zeeman
    [trap]        — trap frequencies in Hz; homogeneous = true removes the
                    trap everywhere, release = true only for real-time
                    evolution (ground state cooled in the trap); optional
                    v_max caps the trap potential (recoil energies)
    [initial_state] — (optional) momentum_kick: split the real-time initial
                    state into ±momentum_kick ħk_R along z
    [spin_orbit]  — omega_r
    [geometry]    — grid sizes, box size factors
    [lattice]     — ramp duration in ms
    [detuning]    — ramp duration in ms
    [ground_state]— imaginary-time cooling parameters
    [simulation]  — real-time parameters and list of observables to save
    [projector]   — e_cut of the c-field region in recoil energies
                    (required when twa_noise = true or mode = "pgpe"/"spgpe")
    [reservoir]   — SPGPE thermal reservoir: temperature, mu, damping rates
    [sweep]       — job-specific values: sample, ramp_time_ms, delta_start/end,
                    omega_l_start/end

The ramp start/end values live in [sweep] so the sweep script can override
them per job. The ramp durations live in [lattice] and [detuning] and are
shared across the sweep.

Public API
----------
load_config(path)           -> (geo, rhs_kwargs)
build_projector(cfg, geo)   -> Projector onto the c-field region
_compute_derived(cfg)       -> dict of derived physical quantities
                               (used by run_simulation.py)

Usage
-----
    from bec_gp_solver.config_loader import load_config
    from bec_gp_solver.gp_equation   import get_rhs

    geo, rhs_kwargs = load_config("configs/sweep/job_0007.toml")
    rhs = get_rhs(geo=geo, **rhs_kwargs)

author: Federico Serrano
Physics and Astronomy Department
Washington State University
"""

import tomllib
import numpy as np
from numpy import pi
from pathlib import Path

from bec_gp_solver.geometry  import make_geometry
from bec_gp_solver.projector import (Projector, soc_dispersion,
                                     alias_free_cutoff, check_dressed_minimum)
import logging
logger = logging.getLogger(__name__)

# =============================================================================
# Derived quantities — separated so run_simulation.py can reuse them
# =============================================================================

def _compute_derived(cfg):
    """
    Compute all derived physical quantities from the raw config dict.

    Returns a flat dict containing unit conversions, trap frequencies,
    interaction matrix, Thomas-Fermi scales, and geometry box sizes.
    Everything is in recoil units unless noted.

    Parameters
    ----------
    cfg : dict — raw config as loaded by tomllib

    Returns
    -------
    dict with keys:
        m_si, e_unit, l_unit, t_unit,
        fx, fy, fz, wx, wy, wz, w3,
        n_atoms, n_comp, sigma_z, q_zeeman,
        a_matrix, g_ref, g_matrix, mu,
        rx, ry, rz, lx, ly, lz,
        omega_r, k_l
    """
    # ------------------------------------------------------------------
    # 1. Unit conversions
    # ------------------------------------------------------------------
    c        = cfg['constants']
    m_ua     = c['m_ua']
    hbar_si  = c['hbar_si']
    a_si     = c['a_si']
    f_recoil = c['f_recoil']

    m_si   = m_ua / 6.022140857e23 / 1000
    e_unit = 2 * pi * hbar_si * f_recoil
    l_unit = hbar_si / np.sqrt(2 * m_si * e_unit)
    t_unit = hbar_si / e_unit

    # ------------------------------------------------------------------
    # 2. Trap frequencies  (Hz → recoil units)
    # ------------------------------------------------------------------
    trap = cfg['trap']
    fx, fy, fz = trap['fx'], trap['fy'], trap['fz']
    wx = fx / f_recoil
    wy = fy / f_recoil
    wz = fz / f_recoil
    w3 = (fx * fy * fz)**(1/3) / f_recoil

    # ------------------------------------------------------------------
    # 3. System
    # ------------------------------------------------------------------
    sys     = cfg['system']
    n_atoms = sys['n_atoms']
    n_comp  = sys['n_components']
    sigma_z = np.array(sys['sigma_z'])
    q_zeeman = np.array(sys['q_zeeman_hz']) / f_recoil

    # ------------------------------------------------------------------
    # 4. Interaction matrix (geometry-dependent)
    # ------------------------------------------------------------------
    a_matrix      = np.atleast_2d(np.array(sys['a_matrix']) * a_si / l_unit)
    geometry_kind = cfg['geometry']['kind']
    g_ref         = 8 * pi * 100.4 * a_si / l_unit   # reference coupling
    mu            = (15 * w3**3 * n_atoms * g_ref / (64 * pi))**(2/5)

    if geometry_kind in ('3d_axial', '3d_cart'):
        g_matrix = 8 * pi * a_matrix

    elif geometry_kind == '2d_cart':
        ratio    = a_matrix / (100.4 * a_si / l_unit)
        g_matrix = 4/3 * (mu * ratio**(2/5))**2 / (wx * wy) / n_atoms

    elif geometry_kind == '1d_cart':
        ratio    = a_matrix / (100.4 * a_si / l_unit)
        g_matrix = 8/3 * (mu * ratio**(2/5))**(3/2) / wz / n_atoms

    else:
        raise ValueError(f"Unknown geometry kind '{geometry_kind}'")

    # ------------------------------------------------------------------
    # 5. Spin-orbit coupling
    # ------------------------------------------------------------------
    omega_r = cfg['spin_orbit']['omega_r']
    # For omega_r >= 4 the single-particle dispersion has a single minimum at
    # k = 0, so the lattice wavevector vanishes.
    k_l     = np.sqrt(1 - (omega_r / 4)**2) if omega_r < 4 else 0.0

    # ------------------------------------------------------------------
    # 6. Box sizes from Thomas-Fermi radii
    # ------------------------------------------------------------------
    geo_cfg = cfg['geometry']
    rx = np.sqrt(4 * mu) / wx
    ry = np.sqrt(4 * mu) / wy
    rz = np.sqrt(4 * mu) / wz
    
    lx = geo_cfg['lx_rx_factor'] * rx
    ly = geo_cfg['ly_ry_factor'] * ry

    if k_l > 0:
        # Commensurate box: lz is an integer number of lattice periods 2*pi/k_l.
        lattice_period = 2*pi / k_l
        # At least one full period, so a short Thomas-Fermi radius cannot
        # floor the box down to lz = 0.
        m_z = max((geo_cfg['lz_rz_factor'] * rz) // lattice_period, 1.0)
        lz  = lattice_period * m_z
    else:
        # No lattice modulation (omega_r >= 4): nothing to commensurate with.
        lz = geo_cfg['lz_rz_factor'] * rz

    return dict(
        # units
        f_recoil = f_recoil,
        m_si     = m_si,
        e_unit   = e_unit,
        l_unit   = l_unit,
        t_unit   = t_unit,
        # trap
        fx=fx, fy=fy, fz=fz,
        wx=wx, wy=wy, wz=wz, w3=w3,
        # system
        n_atoms  = n_atoms,
        n_comp   = n_comp,
        sigma_z  = sigma_z,
        q_zeeman = q_zeeman,
        # interaction
        a_matrix = a_matrix,
        g_ref    = g_ref,
        g_matrix = g_matrix,
        mu       = mu,
        # geometry
        rx=rx, ry=ry, rz=rz, lx=lx, ly=ly, lz=lz,
        # spin-orbit
        omega_r  = omega_r,
        k_l      = k_l,
    )


# =============================================================================
# Ramp builders — read start/end from [sweep], duration from [lattice]/[detuning]
# =============================================================================

def _linear_ramp(start, end, t_ramp):
    """
    Linear ramp start → end over t_ramp, saturating afterwards.

    t_ramp == 0 is a sudden quench: the value is `start` at t <= 0 (so the
    imaginary-time ground state, evaluated at t=0, uses the initial
    parameters) and `end` for any t > 0.
    """
    if t_ramp < 0:
        raise ValueError(f"ramp time must be non-negative, got {t_ramp}")
    if t_ramp == 0:
        return lambda t: start if t <= 0 else end

    return lambda t: start + (end - start) * np.clip(t / t_ramp, 0.0, 1.0)


def _build_lattice_ramp(cfg, t_unit):
    """
    Linear ramp: omega_l_start → omega_l_end over t_ramp_ms.
    Start/end come from [sweep].
    Saturates after t_ramp; t_ramp_ms = 0 is a sudden quench.
    """
    sweep         = cfg['sweep']
    omega_l_start = sweep['omega_l_start']
    omega_l_end   = sweep['omega_l_end']
    t_ramp        = sweep['ramp_time_ms'] * 1e-3 / t_unit

    return _linear_ramp(omega_l_start, omega_l_end, t_ramp)


def _build_detuning_ramp(cfg, t_unit):
    """
    Linear ramp: delta_start_hz → delta_end_hz over t_ramp_ms.
    Start/end come from [sweep] in Hz and are converted to recoil units.
    Saturates after t_ramp; t_ramp_ms = 0 is a sudden quench.
    """
    sweep       = cfg['sweep']
    f_recoil    = cfg['constants']['f_recoil']
    delta_start = sweep['delta_start_hz'] / f_recoil
    delta_end   = sweep['delta_end_hz']   / f_recoil
    t_ramp      = sweep['ramp_time_ms'] * 1e-3 / t_unit

    return _linear_ramp(delta_start, delta_end, t_ramp)


# =============================================================================
# Public API
# =============================================================================

def load_config(path):
    """
    Load a TOML config file and return a geometry instance and the keyword
    arguments for get_rhs().

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    geo        : Geometry instance
    rhs_kwargs : dict — unpack directly into get_rhs():
                   n_components, sigma_z, detuning, q_zeeman, omega_r,
                   g_matrix, n_atoms, wx, wy, wz, lattice_strength, k_l
                 Note: 'mode' is NOT included — set it explicitly when
                 calling get_rhs() since it differs between ground state
                 (imaginary) and real-time runs.
    """
    path = Path(path)
    with open(path, 'rb') as f:
        cfg = tomllib.load(f)

    d = _compute_derived(cfg)

    # build geometry
    geo_cfg = cfg['geometry']

    all_sizes   = (geo_cfg['nx'], geo_cfg['ny'], geo_cfg['nz'])
    all_lengths = (d['lx'], d['ly'], d['lz'])

    sizes   = tuple(n for n, l in zip(all_sizes, all_lengths) if n != 0)
    lengths = tuple(l for n, l in zip(all_sizes, all_lengths) if n != 0)

    geo = make_geometry(geo_cfg['kind'], sizes=sizes, lengths=lengths)

    # build ramp callables
    lattice_strength = _build_lattice_ramp(cfg, d['t_unit'])
    detuning         = _build_detuning_ramp(cfg, d['t_unit'])

    # print summary
    sweep = cfg['sweep']
    logger.info(f"Config loaded: {path.name}")
    logger.info(f"  geometry    : {geo_cfg['kind']}  "
                f"nx={geo_cfg['nx']}, ny={geo_cfg['ny']}, nz={geo_cfg['nz']}")
    logger.info(f"  box sizes   : lx={d['lx']:.5f}, ly={d['ly']:.5f}, lz={d['lz']:.5f}  (recoil units)")
    logger.info(f"  TF radii    : rx={d['rx']:.5f}, ry={d['ry']:.5f}, rz={d['rz']:.5f}")
    logger.info(f"  mu          : {d['mu']:.4f}")
    logger.info(f"  sample      : {sweep['sample']}")
    logger.info(f"  ramp time   : {sweep['ramp_time_ms']} ms")
    logger.info(f"  delta       : {sweep['delta_start_hz']} → {sweep['delta_end_hz']} Hz")
    logger.info(f"  omega_l     : {sweep['omega_l_start']} → {sweep['omega_l_end']}")
    logger.info(f"  omega_r     : {d['omega_r']}")

    v_max = cfg['trap'].get('v_max')
    if v_max is not None and v_max < 2 * d['mu']:
        logger.warning(f"[trap].v_max = {v_max} E_R is below 2 mu ≈ {2 * d['mu']:.2f}: "
                       f"the cap reaches into the cloud and changes its shape.")

    rhs_kwargs = dict(
        n_components     = d['n_comp'],
        sigma_z          = d['sigma_z'],
        detuning         = detuning,
        q_zeeman         = d['q_zeeman'],
        omega_r          = d['omega_r'],
        g_matrix         = d['g_matrix'],
        n_atoms          = d['n_atoms'],
        wx               = d['wx'],
        wy               = d['wy'],
        wz               = d['wz'],
        lattice_strength = lattice_strength,
        k_l              = d['k_l'],
        homogeneous      = cfg['trap'].get('homogeneous', False),
        v_trap_max       = cfg['trap'].get('v_max'),
    )

    return geo, rhs_kwargs


def build_projector(cfg, geo):
    """
    Build the projector onto the c-field region for this config.

    The region uses the spin-orbit dispersion k_⊥² + (k_z - σ_z)² with
    σ_z from [system], below the cutoff [projector].e_cut (recoil energies),
    which is required. Logged warnings flag a cutoff that aliases the
    nonlinear term, a mode count not small compared with the atom number,
    and a cutoff that cuts the spin admixture of the Raman-dressed band
    minimum (checked at the start and end detuning of the sweep).

    Parameters
    ----------
    cfg : raw config dict
    geo : Geometry instance from load_config()

    Returns
    -------
    Projector

    Raises
    ------
    ValueError : if [projector].e_cut is not set (it is required for
                 TWA noise, PGPE and SPGPE).
    """
    dispersion = soc_dispersion(np.array(cfg['system']['sigma_z']))
    e_cut      = cfg.get('projector', {}).get('e_cut')
    if e_cut is None:
        try:
            hint = (f"the largest alias-free cutoff on this grid is "
                    f"{alias_free_cutoff(geo, dispersion):.4f}")
        except ValueError:
            hint = "no alias-free cutoff exists on this grid"
        raise ValueError(
            f"[projector].e_cut must be set when twa_noise = true or "
            f"mode = 'pgpe' / 'spgpe' ({hint})."
        )

    projector = Projector(geo, dispersion, e_cut)
    projector.check_mode_count(cfg['system']['n_atoms'])

    # the Raman-dressed ground state must fit in the c-field at both ends
    # of the detuning ramp
    d      = _compute_derived(cfg)
    sweep  = cfg['sweep']
    deltas = sorted({sweep['delta_start_hz'] / d['f_recoil'],
                     sweep['delta_end_hz']   / d['f_recoil']})
    check_dressed_minimum(e_cut, d['sigma_z'], d['omega_r'], deltas,
                          d['q_zeeman'])
    return projector
