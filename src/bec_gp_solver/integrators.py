# -*- coding: utf-8 -*-
"""
integrators.py
Fixed-step time integration for GPE, PGPE and (later) SPGPE.

A fixed step is required for stochastic equations (the noise increment is
drawn once per step) and is used for all modes so results are comparable.

Two schemes share one driver, integrate():
  RK4    — classical Runge-Kutta on the full right-hand side.
  RK4IP  — interaction-picture RK4 (Caradoc-Davies): for dψ/dt = Lψ + N(t,ψ)
           with L diagonal in the spectral basis, exp(Lτ) is applied exactly
           and RK4 only sees N. The kinetic term — the stiffest one — then
           neither limits the step nor suffers RK4's numerical damping.

The step is a maximum: within every output interval it is reduced so the
interval holds an integer number of steps, and the saved frames land
exactly on t_eval with no interpolation.

Stability
---------
Classical RK4 is stable for eigenvalues λ of the right-hand side with
|λ dt| ≲ 2.83 on the imaginary axis (real time, λ = -iE) and ≲ 2.79 on the
negative real axis (imaginary time). With E_max an upper bound on the
Hamiltonian spectrum, dt < RK4_STABILITY / E_max. For RK4IP, E_max only
needs to cover N.

Accuracy
--------
Stable is not accurate over long runs: per step RK4 multiplies a mode of
energy E by |R(iE dt)|² ≈ 1 - (E dt)⁶/72, so over a run of length t the
fastest mode loses a fraction ≈ (t/dt)(E dt)⁶/72 of its norm. resolve_dt()
keeps this below damping_tol when choosing a default step.

Public API
----------
integrate(rhs, psi0, t_eval, dt, t0=0.0, propagator=None, step_hook=None)
                                          -> array (psi0.size, len(t_eval))
LinearPropagator(geo, rates)              -> ψ ↦ exp(rates·τ) ψ, for RK4IP
resolve_dt(dt, e_max, t_total=None, ...)  -> validated or default step

author: Federico Serrano
Physics and Astronomy Department
Washington State University
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)


RK4_STABILITY = 2.78   # min of the imaginary- and real-axis limits, rounded down


def resolve_dt(dt, e_max, t_total=None, safety=0.5, damping_tol=0.01):
    """
    Return the time step to use, checked against stability and accuracy.

    Parameters
    ----------
    dt          : float or None — requested step; None picks a default
    e_max       : float — upper bound on |E| of the operator RK4 sees
                  (the full Hamiltonian for RK4, only N for RK4IP)
    t_total     : float or None — run length; enables the accuracy check
    safety      : float — fraction of the stability limit for the default
    damping_tol : float — maximum norm fraction the fastest mode may lose
                  over the whole run, (t/dt)(E dt)⁶/72 ≤ damping_tol

    The default is the smaller of safety × stability limit and the
    accuracy step. A requested dt above the stability limit is an error;
    above the accuracy step it only logs a warning.

    Raises
    ------
    ValueError : if dt exceeds the stability limit, or if dt is None and
                 e_max == 0 (nothing to base a default on).
    """
    if e_max <= 0:
        if dt is None:
            raise ValueError("No terms limit the step (e_max = 0): set dt "
                             "explicitly.")
        return dt

    dt_max = RK4_STABILITY / e_max
    dt_acc = (np.inf if t_total is None
              else (72 * damping_tol / (e_max**6 * t_total))**(1 / 5))

    if dt is None:
        dt = min(safety * dt_max, dt_acc)
        logger.info(f"No dt set — using {dt:.4g} (stability limit {dt_max:.4g}, "
                    f"accuracy step {dt_acc:.4g}, E_max≈{e_max:.4g})")
    elif dt > dt_max:
        raise ValueError(
            f"dt={dt} exceeds the RK4 stability limit {dt_max:.4g} "
            f"(E_max≈{e_max:.4g} recoil energies). Use dt ≤ {dt_max:.4g}, "
            f"or omit dt to choose one automatically."
        )
    elif dt > dt_acc:
        logger.warning(
            f"dt={dt} is above the accuracy step {dt_acc:.4g}: the fastest "
            f"mode (E≈{e_max:.4g}) may lose more than {damping_tol:.0%} of "
            f"its norm to RK4 damping over t={t_total:.4g}."
        )
    return dt


class LinearPropagator:
    """
    ψ ↦ exp(rates · τ) ψ, applied in the spectral basis of geo.

    rates : complex array (n_comp, *geo.sizes) — e.g. -iε(k) in real time
    The exponential for the last τ is cached; within an output interval the
    step (and so τ) is constant.
    """

    def __init__(self, geo, rates):
        self.geo     = geo
        self.rates   = rates
        self._n_comp = rates.shape[0]
        self._tau    = None
        self._factor = None

    def __call__(self, psi, tau):
        if tau != self._tau:
            self._tau, self._factor = tau, np.exp(self.rates * tau)
        a = self.geo.to_spectral(psi.reshape(self._n_comp, -1)) * self._factor
        return self.geo.from_spectral(a).reshape(psi.shape)


def _rk4_step(rhs, t, psi, dt):
    k1 = rhs(t, psi)
    k2 = rhs(t + dt / 2, psi + dt / 2 * k1)
    k3 = rhs(t + dt / 2, psi + dt / 2 * k2)
    k4 = rhs(t + dt, psi + dt * k3)
    return psi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _rk4ip_step(rhs_N, D, t, psi, dt):
    """One RK4IP step; D(ψ, τ) = exp(Lτ)ψ, the frame is moved by dt/2."""
    h2    = dt / 2
    psi_I = D(psi, h2)
    k1    = D(rhs_N(t, psi), h2)
    k2    = rhs_N(t + h2, psi_I + h2 * k1)
    k3    = rhs_N(t + h2, psi_I + h2 * k2)
    k4    = rhs_N(t + dt, D(psi_I + dt * k3, h2))
    return D(psi_I + dt / 6 * (k1 + 2 * k2 + 2 * k3), h2) + dt / 6 * k4


def integrate(rhs, psi0, t_eval, dt, t0=0.0, propagator=None, step_hook=None):
    """
    Integrate dψ/dt = rhs(t, ψ) with fixed-step RK4, or with RK4IP when a
    propagator is given (then dψ/dt = Lψ + rhs(t, ψ), exp(Lτ) = propagator).

    Parameters
    ----------
    rhs        : callable(t, psi) -> array of the same shape as psi; the full
                 right-hand side (RK4) or the remainder N (RK4IP)
    psi0       : complex array — state at t0
    t_eval     : 1D array of output times, non-decreasing, all >= t0
    dt         : float — maximum step; reduced per output interval so that
                 each interval is an integer number of steps
    t0         : float — initial time
    propagator : callable(psi, tau) -> exp(Lτ)ψ, e.g. LinearPropagator;
                 None for plain RK4
    step_hook  : callable(t, h) or None — called once before every step.
                 Stochastic right-hand sides use it to draw their noise
                 increments, which then stay fixed for all stages of the
                 step (Stratonovich interpretation).

    Returns
    -------
    psi_t : complex array, shape (psi0.size, len(t_eval)) — same layout as
            solve_ivp's sol.y
    """
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval[0] < t0 or np.any(np.diff(t_eval) < 0):
        raise ValueError("t_eval must be non-decreasing and start at or after t0.")

    if propagator is None:
        step = lambda t, psi, h: _rk4_step(rhs, t, psi, h)
    else:
        step = lambda t, psi, h: _rk4ip_step(rhs, propagator, t, psi, h)

    psi   = np.array(psi0, dtype=np.complex128).reshape(-1)
    out   = np.empty((psi.size, len(t_eval)), dtype=np.complex128)
    t     = t0
    steps = 0

    for i, t_next in enumerate(t_eval):
        interval = t_next - t
        if interval > 0:
            n  = int(np.ceil(interval / dt - 1e-9))
            h  = interval / n
            for j in range(n):
                if step_hook is not None:
                    step_hook(t + j * h, h)
                psi = step(t + j * h, psi, h)
            steps += n
        t = t_next
        out[:, i] = psi

    scheme = 'RK4' if propagator is None else 'RK4IP'
    logger.info(f"{scheme}: {steps} steps (dt ≤ {dt:.4g}) up to t={t:.4g}")
    return out
