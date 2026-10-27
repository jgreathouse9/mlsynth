"""Simulation of model (1) and its channel-off counterfactuals.

A linear Hawkes process is a branching process: immigrants of type ``e`` arrive
as a Poisson process with rate ``mu_e``, and each type-``k`` event begets
``Poisson(alpha[k, e])`` type-``e`` children at lags drawn from ``psi_k``
(an exponential with rate ``beta_k``, since ``psi_k`` integrates to one).
Firm-initiated types are immigrants only: their columns of ``alpha`` are zero,
which makes them the exogenous Poisson inputs of the paper's Sec. 5.1.

The branching form gives the ground truth of Sec. 5 exactly. Switching channel
``z`` off removes its events and every descendant of them, and leaves all other
draws unchanged, so ``CCC_z`` (the conversions lost) is the number of
conversions with a channel-``z`` event in their ancestry. That is the paper's
"run the simulator again with the same seed", with a coupling that holds
event by event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .fit import GPPParams
from .paths import Path


@dataclass(frozen=True)
class Simulation:
    paths: list[Path]
    conversions: int
    ccc: dict[str, int]  # conversions lost when each channel is switched off

    @property
    def ccc_shares(self) -> dict[str, float]:
        total = sum(self.ccc.values())
        return {ch: v / total for ch, v in self.ccc.items()}


def simulate(
    mu_all: Sequence[float],
    alpha: np.ndarray,
    decay,
    T,
    n: int,
    channels: Mapping[str, Sequence[int]] | None = None,
    seed=None,
) -> Simulation:
    """Simulate ``n`` independent paths on ``[0, T]``.

    ``mu_all`` holds a baseline rate for every type (customer- and
    firm-initiated); ``alpha`` is ``(p, p)`` with ``alpha[k, e]`` the edge
    ``k -> e``.
    """
    rng = np.random.default_rng(seed)
    mu_all = np.asarray(mu_all, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    p = mu_all.size
    if alpha.shape != (p, p):
        raise ValueError("alpha must be (p, p)")
    beta = np.broadcast_to(np.asarray(decay, dtype=float), (p,))
    Tn = np.broadcast_to(np.asarray(T, dtype=float), (n,)).copy()
    channels = dict(channels or {})
    bit = np.zeros(p, dtype=np.int64)
    for z, (ch, types) in enumerate(channels.items()):
        bit[list(types)] |= 1 << z

    counts = rng.poisson(mu_all[None, :] * Tn[:, None])  # (n, p)
    path_id = np.repeat(np.repeat(np.arange(n), p), counts.ravel())
    etype = np.tile(np.arange(p), n).repeat(counts.ravel())
    time = rng.uniform(0.0, 1.0, path_id.size) * Tn[path_id]
    mask = bit[etype]
    gens = [(path_id, time, etype, mask)]
    while gens[-1][0].size:
        pid, t, ty, msk = gens[-1]
        kids = rng.poisson(alpha[ty])  # (m, p) children per parent and type
        par = np.repeat(np.arange(ty.size), kids.sum(axis=1))
        ktype = np.tile(np.arange(p), ty.size).repeat(kids.ravel())  # parent-major, like par
        ktime = t[par] + rng.exponential(1.0 / beta[ty[par]])
        keep = ktime <= Tn[pid[par]]
        par, ktype, ktime = par[keep], ktype[keep], ktime[keep]
        gens.append((pid[par], ktime, ktype, msk[par] | bit[ktype]))

    pid = np.concatenate([g[0] for g in gens])
    t = np.concatenate([g[1] for g in gens])
    ty = np.concatenate([g[2] for g in gens])
    msk = np.concatenate([g[3] for g in gens])
    order = np.lexsort((t, pid))
    pid, t, ty, msk = pid[order], t[order], ty[order], msk[order]
    bounds = np.searchsorted(pid, np.arange(n + 1))
    paths = [Path(t[a:b], ty[a:b], Tn[j]) for j, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
    conv = ty == 0
    ccc = {ch: int(np.sum(conv & ((msk >> z) & 1).astype(bool))) for z, ch in enumerate(channels)}
    return Simulation(paths, int(conv.sum()), ccc)


def simulate_params(params: GPPParams, firm_rates: Sequence[float], T, n, channels=None, seed=None):
    """Simulate from fitted or true :class:`GPPParams` plus firm-initiated Poisson rates."""
    mu_all = np.r_[params.mu, np.asarray(firm_rates, dtype=float)]
    if mu_all.size != params.p:
        raise ValueError("need one rate per firm-initiated type")
    return simulate(mu_all, params.alpha_full, params.decay, T, n, channels, seed)


def compensator_increments(params: GPPParams, paths: Sequence[Path], e: int) -> np.ndarray:
    """Time-rescaled inter-arrival times of type ``e`` (``e < q``).

    By the time-rescaling theorem, ``Lambda_e`` maps each path's type-``e``
    events to a unit-rate Poisson process on ``[0, Lambda_e(T_j)]``. Laying the
    rescaled paths end to end gives one unit-rate process, whose gaps are
    i.i.d. Exp(1) if the model is right; this is a goodness-of-fit check for
    fitted parameters and simulators alike. (Gaps measured within paths alone
    are censored at ``T_j``, which biases them low for rare events.)
    """
    if not 0 <= e < params.q:
        raise ValueError("e must be a customer-initiated type")
    a = params.alpha_full[:, e]
    beta = params.decay

    def Lam(path, at):
        lag = at[:, None] - path.times[None, :]
        ty = path.types
        contrib = np.where(lag > 0, a[ty] * -np.expm1(-beta[ty] * np.maximum(lag, 0.0)), 0.0)
        return params.mu[e] * at + contrib.sum(axis=1)

    offset = 0.0
    rescaled = []
    for path in paths:
        hits = path.times[path.types == e]
        rescaled.append(offset + Lam(path, hits))
        offset += Lam(path, np.array([path.T]))[0]
    return np.diff(np.r_[0.0, np.concatenate(rescaled)])
