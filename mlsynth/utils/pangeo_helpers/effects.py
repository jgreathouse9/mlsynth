"""Realized ATT estimation for a PANGEO design with post-period data.

PANGEO is a *design* method: with only pre-treatment history it returns the
supergeo pairs and the treatment/control assignment. If the experiment has
since run -- i.e. the panel carries a ``post_col`` marking post-treatment
periods -- the **same design** (built on the pre-period alone) is scored
against the realized post outcomes here.

Under the pre-period balance the design enforces (parallel treatment and
control supergeo trajectories), the within-pair estimator is the textbook
difference-in-differences. For a pair with population-weighted supergeo
means :math:`\\bar Y^T_t, \\bar Y^C_t` and gap
:math:`g_t = \\bar Y^T_t - \\bar Y^C_t`,

.. math::

   \\hat\\tau_p
   = \\underbrace{\\overline{g}_{\\text{post}}}_{\\text{post gap}}
   - \\underbrace{\\delta_p}_{\\text{pre gap (counterfactual)}},
   \\qquad \\delta_p = \\overline{g}_{\\text{pre}} .

where the counterfactual level :math:`\\delta_p` is the gap mean over the
**held-out blank window** (the recent pre periods the split was *not*
optimised on) -- a local anchor, so per-geo level differences and slow
drift are removed rather than compared against a stale level. The arm and
**program** ATTs are population-weighted averages of the pair effects
(treated-supergeo weights); the program number is the headline.

Inference is by **moving-block bootstrap of the held-out gap increments**
-- the integrated-series analogue of moving-block conformal inference
(Chernozhukov, Wuthrich & Zhu 2021). The supergeo gap is typically
near-integrated: the residual factor loading that survives imperfect
balancing rides the latent random walk, so the gap *drifts* over the post
horizon and the dominant uncertainty is that future drift -- which the
*levels* of a short pre/blank window cannot reveal, but their (stationary)
**first differences** can. The blank-window increments are moving-block
resampled and cumulated into synthetic ``n_level + n_post`` segments; the
null statistic ``mean(post) - mean(trailing level)`` is read off each
segment, so both the level-estimation noise and the forward drift come from
one coherent path. The 95% interval and p-value follow at the **arm** and
**program** levels. (A stationary residual reservoir, by contrast,
under-covers badly here because it is blind to the random-walk drift.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .parallelism import _wavg


@dataclass(frozen=True)
class AttEstimate:
    """A difference-in-differences ATT (with bootstrap CI) at one level.

    Attributes
    ----------
    level : str
        ``"program"`` or an arm label.
    att : float
        DiD ATT on the (population-weighted) outcome scale.
    att_pct : float
        ATT as a percentage of the pre-period treated baseline level.
    baseline : float
        Pre-period treated-group level used for ``att_pct``.
    ci_lower, ci_upper : float
        Moving-block-bootstrap confidence interval for the absolute ATT.
    ci_lower_pct, ci_upper_pct : float
        The same interval expressed as a percentage of baseline.
    p_value : float
        Bootstrap p-value for the null of no effect.
    n_post : int
        Number of post-treatment periods averaged.
    """

    level: str
    att: float
    att_pct: float
    baseline: float
    ci_lower: float
    ci_upper: float
    ci_lower_pct: float
    ci_upper_pct: float
    p_value: float
    n_post: int


@dataclass(frozen=True)
class PangeoEffects:
    """Realized ATT for a PANGEO design scored against post-period data.

    Attributes
    ----------
    program : AttEstimate
        Headline program-level ATT (pooled across all arms).
    arms : dict
        ``{arm_label: AttEstimate}``.
    pair_att : dict
        ``{arm_label: [per-pair ATT, ...]}`` (point estimates).
    n_post : int
        Number of post periods.
    weighted : bool
        Whether a population weight was used in the aggregation.
    alpha : float
        Significance level for the conformal intervals.
    """

    program: AttEstimate
    arms: Dict[Any, AttEstimate]
    pair_att: Dict[Any, List[float]]
    n_post: int
    weighted: bool
    alpha: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Tidy table of the program and per-arm ATT estimates."""
        rows = [_att_row(self.program)]
        rows += [_att_row(c) for c in self.arms.values()]
        return pd.DataFrame(rows).set_index("level")


def _att_row(e: AttEstimate) -> Dict[str, Any]:
    return {
        "level": e.level, "att": e.att, "att_pct": e.att_pct,
        "ci_lower": e.ci_lower, "ci_upper": e.ci_upper,
        "ci_lower_pct": e.ci_lower_pct, "ci_upper_pct": e.ci_upper_pct,
        "p_value": e.p_value, "n_post": e.n_post,
    }


def _rw_null(
    G_pre: np.ndarray, n_post: int, n_level: int, block: int, n_rep: int,
    rng: np.random.Generator,
) -> "tuple[np.ndarray, float]":
    """Null distribution of the level-adjusted ATT under a near-integrated gap.

    The supergeo gap is typically near-integrated (the residual factor
    loading rides the latent random walk), so the dominant uncertainty is the
    *future drift* over the post horizon -- which a short pre or blank window
    cannot reveal directly, but the (stationary) gap **increments** can. This
    moving-block bootstraps the held-out blank-window first differences,
    cumulates them into synthetic ``n_post``-step forward paths, and adds the
    deviation of the last pre value from the counterfactual level. Returns
    ``(null_draws, delta)``.
    """
    T0 = G_pre.size
    delta = float(G_pre[T0 - n_level:].mean())
    diffs = np.diff(G_pre[T0 - n_level:])      # held-out increments (honest)
    if diffs.size < block + 1:                 # too few: use full-pre diffs
        diffs = np.diff(G_pre)
    nb = diffs.size
    starts = nb - block + 1
    span = n_level + n_post
    null = np.empty(n_rep)
    for r in range(n_rep):
        seq = []
        while len(seq) < span:
            i = int(rng.integers(0, starts))
            seq.extend(diffs[i:i + block])
        # One coherent random-walk segment: trailing level window + post
        # window, so both the level-estimation noise and the forward drift
        # are drawn from the same path.
        path = np.cumsum(np.asarray(seq[:span]))
        null[r] = path[n_level:].mean() - path[:n_level].mean()
    return null, delta


def _pair_records(arm_design, pos, Y_post, weights):
    """Per-pair full-pre gap, post gap, baseline and treated weight."""
    recs = []
    for p in arm_design.pairs:
        t_idx = [pos[u] for u in p.treatment]
        c_idx = [pos[u] for u in p.control]
        gap_pre = np.asarray(p.treatment_mean) - np.asarray(p.control_mean)
        gap_post = _wavg(Y_post, t_idx, weights) - _wavg(Y_post, c_idx, weights)
        baseline = float(np.asarray(p.treatment_mean).mean())
        w = float(weights[t_idx].sum()) if weights is not None else len(t_idx)
        recs.append({"gap_pre": gap_pre, "gap_post": gap_post,
                     "baseline": baseline, "weight": w})
    return recs


def _aggregate(level, recs, alpha, n_level, n_rep, seed) -> AttEstimate:
    w = np.array([r["weight"] for r in recs], dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    G_pre = np.sum([w[i] * recs[i]["gap_pre"] for i in range(len(recs))],
                   axis=0)
    G_post = np.sum([w[i] * recs[i]["gap_post"] for i in range(len(recs))],
                    axis=0)
    baseline = float(w @ np.array([r["baseline"] for r in recs]))
    n_post = len(G_post)
    T0 = G_pre.size
    n_level = n_level if 2 <= n_level <= T0 else T0
    block = max(2, int(round(np.sqrt(n_post))))

    rng = np.random.default_rng(seed)
    null, delta = _rw_null(G_pre, n_post, n_level, block, n_rep, rng)
    att = float(G_post.mean()) - delta
    med = float(np.median(null))
    dev = np.abs(null - med)
    q = float(np.quantile(dev, 1.0 - alpha))
    p_value = float((np.sum(dev >= abs(att - med)) + 1) / (null.size + 1))
    ci_lo, ci_hi = att - q, att + q

    pct = (lambda x: x / baseline * 100.0) if abs(baseline) > 1e-12 \
        else (lambda x: float("nan"))
    return AttEstimate(
        level=level, att=att, att_pct=pct(att), baseline=baseline,
        ci_lower=ci_lo, ci_upper=ci_hi,
        ci_lower_pct=pct(ci_lo), ci_upper_pct=pct(ci_hi),
        p_value=p_value, n_post=int(n_post),
    )


def compute_pangeo_effects(
    results, inputs, Y_post: np.ndarray, *, alpha: float = 0.05,
    n_boot: int = 2000, seed: int = 0,
) -> PangeoEffects:
    """Difference-in-differences ATT (with conformal CIs) for a design scored
    on post outcomes.

    Parameters
    ----------
    results : PangeoResults
        The frozen design (pairs + assignment) built on the pre-period.
    inputs : PangeoInputs
        Pre-period inputs (supplies unit order and population weights).
    Y_post : np.ndarray
        Post-period outcomes, shape ``(N, T_post)``, rows aligned with
        ``inputs.unit_names``.
    alpha : float
        Significance level for the conformal CIs / p-values.
    """
    pos = {u: i for i, u in enumerate(inputs.unit_names)}
    weights = inputs.weights
    n_post = int(Y_post.shape[1])
    # Local counterfactual-level window = the held-out blank window.
    n_level = int(results.metadata.get("n_holdout", 0)) or None

    all_recs: List[dict] = []
    by_arm: Dict[Any, List[dict]] = {}
    pair_att: Dict[Any, List[float]] = {}
    for arm, d in results.arm_designs.items():
        recs = _pair_records(d, pos, Y_post, weights)
        by_arm[arm] = recs
        all_recs.extend(recs)
        # per-pair point ATT (local level over the blank window)
        for r in recs:
            gp = r["gap_pre"]
            nl = n_level if (n_level and 2 <= n_level <= gp.size) else gp.size
            pair_att.setdefault(arm, []).append(
                float(r["gap_post"].mean()) - float(gp[gp.size - nl:].mean()))

    nl = n_level if n_level else 0
    program = _aggregate("program", all_recs, alpha, nl, n_boot, seed)
    arms = {arm: _aggregate(str(arm), recs, alpha, nl, n_boot, seed + 1)
            for arm, recs in by_arm.items()}

    return PangeoEffects(
        program=program, arms=arms, pair_att=pair_att, n_post=n_post,
        weighted=weights is not None, alpha=alpha,
        metadata={"estimator": "population-weighted DiD; program = "
                               "treated-weighted average of pair ATTs",
                  "inference": "moving-block bootstrap of held-out gap "
                               "increments (near-integrated gap), per arm "
                               "and program"},
    )
