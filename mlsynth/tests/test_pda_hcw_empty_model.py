"""HCW's best-subset search must not return the intercept-only model.

Hsiao, Ching & Wan (2012) Section 5 selects among subsets of the control units,
and the reference implementation this method is checked against -- ``pampe``,
whose search is ``leaps::regsubsets`` -- enumerates model sizes ``1..nvmax``.
Size zero is not in its search space, so the empty subset is never a candidate.

mlsynth's search seeded its incumbent with the empty subset scored at one
parameter, which admits a counterfactual built from the treated unit's own
pre-period mean and no control at all. That is not a panel data approach
counterfactual: equation (6) of Wan, Xie & Hsiao (2018) is
``Y(0)_1t = b0 + Xtilde_0 b + u_1t``, and dropping ``Xtilde_0`` leaves a
constant extrapolated across the post window.

It only bites when ``T0`` is small, which is why it survived. AICc's
small-sample correction is ``2K(K+1)/(n-K-1)`` with ``K = donors + 2``: at
``T0 = 5`` that is 6.0 for one donor against 1.33 for none, so the empty model
wins outright more than half the time. At ``T0 = 20`` it is 0.75 against 0.35
and essentially never wins.

Measured against ``pampe`` on identical R-generated panels from Wan, Xie &
Hsiao's own simulation designs, the two agreed to three decimals at
``T0 in {20, 40}`` and diverged by 6.9x at ``T0 = 5`` (Design 6a: 6.238 against
0.910, the paper's published cell). Restricted to the replications where
mlsynth did select a donor, it returned 0.949 against pampe's 0.910.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.pda_helpers.hcw.estimation import (
    _gram,
    _subset_rss,
    best_subset_select,
    info_criterion,
)


def _panel(seed: int, n_donors: int = 5, T0: int = 5, T: int = 15):
    """Wan, Xie & Hsiao (2018) Design 6a: one drifting walk, unit loadings.

    Every unit is the same random walk plus N(0, 0.25) noise, so every donor is
    a valid predictor and a search that selects none of them is wrong on the
    merits, not merely different from the reference.
    """
    rng = np.random.default_rng(seed)
    eta = rng.chisquare(1)
    lam = np.zeros(T)
    for s in range(1, T):
        lam[s] = eta + lam[s - 1] + rng.normal(0.0, 0.5)
    y = lam[:, None] + rng.normal(0.0, 0.25, size=(T, n_donors + 1))
    return y[:, 0], y[:, 1:], T0


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("backend", ["fw"])
def test_selection_is_never_empty(seed, backend):
    """At least one donor, on every draw, at the sample size where it bit."""
    y, X, T0 = _panel(seed)
    chosen = best_subset_select(y, X, T0, criterion="AICc", backend=backend)
    assert len(chosen) >= 1, (
        f"seed {seed}: the search returned the intercept-only model, which "
        f"pampe's regsubsets cannot select"
    )


def test_the_empty_model_would_have_won_without_the_fix():
    """The positive control: this is not vacuous at T0 = 5.

    If the empty model never beat the best single donor on AICc here, the test
    above would pass for a reason unrelated to the defect.
    """
    beaten = 0
    for seed in range(25):
        y, X, T0 = _panel(seed)
        G, Zty, yty = _gram(np.asarray(y)[:T0], np.asarray(X)[:T0])
        ic_empty = info_criterion(_subset_rss(G, Zty, yty, ()), T0, 1, "AICc")
        ic_best1 = min(info_criterion(_subset_rss(G, Zty, yty, (j,)), T0, 2, "AICc")
                       for j in range(X.shape[1]))
        beaten += int(ic_empty < ic_best1)
    assert beaten >= 8, (
        f"the empty model beat every single-donor model on only {beaten}/25 "
        f"draws; this design no longer exercises the defect"
    )


def test_selected_subset_is_the_best_of_its_size():
    """The fix must not change which non-empty subset wins.

    Excluding size zero may not perturb the ranking among sizes >= 1.
    """
    for seed in range(15):
        y, X, T0 = _panel(seed)
        chosen = best_subset_select(y, X, T0, criterion="AICc", backend="fw")
        G, Zty, yty = _gram(np.asarray(y)[:T0], np.asarray(X)[:T0])
        got = info_criterion(_subset_rss(G, Zty, yty, tuple(chosen)), T0,
                             len(chosen) + 1, "AICc")
        from itertools import combinations
        n_donors = X.shape[1]
        r_max = min(n_donors, max(T0 - 2, 0))
        best = min(
            info_criterion(_subset_rss(G, Zty, yty, c), T0, len(c) + 1, "AICc")
            for r in range(1, r_max + 1)
            for c in combinations(range(n_donors), r))
        assert got == pytest.approx(best, rel=1e-9), (
            f"seed {seed}: chose {chosen} at AICc {got:.6f}, but the best "
            f"non-empty subset scores {best:.6f}")


def test_large_pre_period_selection_is_unchanged():
    """The defect was invisible at T0 = 20; the fix must keep it that way.

    Three of the four cells cross-validated against pampe already agreed to
    three decimals. Whatever the search returns there must not move.
    """
    for seed in range(10):
        y, X, T0 = _panel(seed, n_donors=5, T0=20, T=30)
        chosen = best_subset_select(y, X, T0, criterion="AICc", backend="fw")
        assert len(chosen) >= 1
        G, Zty, yty = _gram(np.asarray(y)[:T0], np.asarray(X)[:T0])
        ic_empty = info_criterion(_subset_rss(G, Zty, yty, ()), T0, 1, "AICc")
        ic_got = info_criterion(_subset_rss(G, Zty, yty, tuple(chosen)), T0,
                                len(chosen) + 1, "AICc")
        assert ic_got < ic_empty, (
            "at T0 = 20 the chosen model should beat intercept-only outright, "
            "so excluding size zero changes nothing here")


def test_backends_agree():
    """Exhaustive and Furnival-Wilson must return the same subset."""
    from mlsynth.utils.pda_helpers.hcw.estimation import _best_subset_exhaustive
    for seed in range(15):
        y, X, T0 = _panel(seed)
        G, Zty, yty = _gram(np.asarray(y)[:T0], np.asarray(X)[:T0])
        n_donors = X.shape[1]
        r_max = min(n_donors, max(T0 - 2, 0))
        ex = _best_subset_exhaustive(G, Zty, yty, n_donors, T0, r_max, "AICc")
        fw = best_subset_select(y, X, T0, criterion="AICc", backend="fw")
        assert len(ex) >= 1 and sorted(ex) == sorted(fw), (
            f"seed {seed}: exhaustive {ex} against Furnival-Wilson {fw}")
