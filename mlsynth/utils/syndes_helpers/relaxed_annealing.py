"""Simulated-annealing primitives for the relaxed SYNDES D-step.

The relaxed solver decouples the discrete assignment search from the
convex weight solve. This module contains the three components that drive
the discrete chain:

- :func:`temperature_schedule` — adaptive temperature controller
- :func:`propose_swap` — Metropolis proposal generator
- :func:`d_step_annealed` — inner Metropolis sweep with weight re-solves
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .relaxed_formulation import compute_energy, solve_weights_global
from .relaxed_structures import RelaxedSwapLog


def temperature_schedule(
    it: int,
    Y: np.ndarray,
    delta_history: Optional[list] = None,
    T0: Optional[float] = None,
    decay: float = 0.97,
    target_accept: float = 0.4,
) -> float:
    """Adaptive simulated-annealing temperature.

    Uses a fixed geometric decay during a short warm-up phase and switches
    to an adaptive schedule driven by the median uphill ``delta`` once
    sufficient history has accumulated.

    Parameters
    ----------
    it : int
        Current outer iteration index (0-based).
    Y : np.ndarray
        Outcome matrix; its standard deviation is used as the default
        starting temperature.
    delta_history : list, optional
        Running history of accepted/rejected proposal energy deltas. When
        more than 20 entries are available the schedule switches to an
        adaptive regime.
    T0 : float, optional
        Override for the starting temperature. Defaults to ``np.std(Y)``.
    decay : float, optional
        Geometric decay factor used during warm-up and as fallback.
    target_accept : float, optional
        Target uphill acceptance probability used by the adaptive branch.

    Returns
    -------
    float
        Temperature to use at iteration ``it``.
    """

    if T0 is None:
        T0 = float(np.std(Y))

    if it < 5:
        return T0 * (decay ** it)

    if delta_history is not None and len(delta_history) > 20:
        deltas = np.array(delta_history[-50:])
        uphill = deltas[deltas > 0]
        if len(uphill) > 5:
            scale = float(np.median(uphill))
            T_adapt = scale / (np.log(1 / target_accept + 1e-8))
            return max(T_adapt, 1e-8)

    return T0 * (decay ** it)


def propose_swap(
    D: np.ndarray,
    T: float,
    max_m: int = 5,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Generate a Metropolis swap proposal of size scaled by temperature.

    Parameters
    ----------
    D : np.ndarray
        Current binary assignment vector of shape ``(N,)``.
    T : float
        Current temperature; larger values produce larger swap blocks.
    max_m : int, optional
        Maximum number of treated/control pairs to swap simultaneously.

    Returns
    -------
    tuple
        ``(D_new, (treated_to_flip, control_to_flip))`` where ``D_new`` is
        the proposed assignment and the second element holds the swapped
        index arrays.
    """

    treated = np.where(D == 1)[0]
    control = np.where(D == 0)[0]

    m = int(1 + (T / (T + 1e-8)) * (max_m - 1))
    m = min(m, len(treated), len(control))

    i_idx = np.random.choice(treated, size=m, replace=False)
    j_idx = np.random.choice(control, size=m, replace=False)

    D_new = D.copy()
    D_new[i_idx] = 0
    D_new[j_idx] = 1
    return D_new, (i_idx, j_idx)


def d_step_annealed(
    Y: np.ndarray,
    D: np.ndarray,
    w: np.ndarray,
    K: int,
    T: float,
    lam: float,
    n_proposals: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, RelaxedSwapLog]:
    """Run one outer iteration of the Metropolis D-step.

    Generates ``n_proposals`` swap proposals, re-solves the convex w-step
    at each candidate assignment, and accepts proposals according to the
    Metropolis criterion at temperature ``T``. The best state visited
    during the sweep is returned.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix of shape ``(T, N)``.
    D : np.ndarray
        Current assignment vector of shape ``(N,)``.
    w : np.ndarray
        Current combined weight vector of shape ``(N,)``.
    K : int
        Number of treated units (preserved by the swap proposal).
    T : float
        Current annealing temperature.
    lam : float
        Ridge penalty used by both the w-step and the energy evaluator.
    n_proposals : int, optional
        Number of proposals per outer iteration. Defaults to ``N``.

    Returns
    -------
    tuple
        ``(best_D, best_w, log)`` — best assignment and weights visited in
        this sweep along with a :class:`RelaxedSwapLog` of diagnostics.
    """

    del K  # currently unused; swap proposals preserve cardinality

    N = len(D)
    n_proposals = N if n_proposals is None else n_proposals

    base_E = compute_energy(Y, D, w, lam)

    best_D = D.copy()
    best_w = w.copy()
    best_E = base_E

    n_accepted = 0
    n_uphill = 0
    n_uphill_accepted = 0
    delta_history: list[float] = []

    for _ in range(n_proposals):
        D_cand, _ = propose_swap(D, T, max_m=5)

        w_cand = solve_weights_global(Y, D_cand, lam=0.0)
        E_cand = compute_energy(Y, D_cand, w_cand, lam)

        delta = E_cand - base_E
        delta_history.append(delta)

        if delta > 0:
            n_uphill += 1

        accept = (delta <= 0) or (np.random.rand() < np.exp(-delta / max(T, 1e-8)))

        if accept:
            D, w = D_cand, w_cand
            base_E = E_cand
            n_accepted += 1
            if delta > 0:
                n_uphill_accepted += 1
            if E_cand < best_E:
                best_E = E_cand
                best_D = D.copy()
                best_w = w.copy()

    log = RelaxedSwapLog(
        n_proposals=n_proposals,
        n_accepted=n_accepted,
        n_uphill=n_uphill,
        n_uphill_accepted=n_uphill_accepted,
        delta_history=delta_history,
    )
    return best_D, best_w, log
