"""Choosing the factor number needs a parsimony guard, not a bare argmin.

Xu (2017) Algorithm 1 scores each candidate rank by leave-one-pre-period-out
mean squared prediction error. What it does with those scores is the part
mlsynth had left out. gsynth walks the ranks upward and adopts a larger one
only when it beats the running minimum by more than ``tol``, which defaults to
0.001 (``R/core.R:351``, ``R/default.R:52``):

    if ((min(CV.out[,"MSPE"]) - MSPE) > tol * min(CV.out[,"MSPE"]))

mlsynth took ``min(mspe, key=...)``, which adopts a larger rank on any
improvement at all. The MSPE surface is not convex in ``r`` and its minimum is
often a near-tie, so on a flat surface the bare argmin buys a factor that
explains nothing and spends a degree of freedom on it.

The constant is gsynth's because Algorithm 1 is gsynth's criterion. ``fect``,
which ``gsynth >= 1.4`` forwards to, guards the same step at 0.01
(``R/cv.R:614``) but over a different fold scheme, and taking its constant here
costs five of the sixteen rank agreements in
``benchmarks/cases/gsynth_av_laws.py``, measured against a pinned live gsynth
1.2.1 run. That benchmark is what caught the wrong constant.

Measured on the Ronczewski (2026) cannabis-alcohol panel, where the surface is
flat and multi-modal (#483):

    r        0         1         2         3         4         5
    mspe  6.73e-4   4.14e-4   4.63e-4   4.90e-4   3.89e-4   4.17e-4
    att   0.00435   0.01792   0.02597   0.03983   0.00694  -0.00573

The guard does not change that panel's answer -- r=4 beats r=1 by 6%, which
clears 0.1% comfortably -- and the last test here pins that, because a guard
credited with fixing a disagreement it does not fix would be a story instead of
a finding.
The published gsynth number for that panel comes from fect's cross-validation,
a different fold scheme, and that remains open in #483.

The rule is tested as a function of the scores. A panel engineered to produce a
near-tie would be testing the fold arithmetic and the rule at once, and would
break whenever either moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import GSYNTH
from mlsynth.utils.gsynth_helpers.pipeline import select_rank, CV_GUARD

# The cannabis panel's measured surface (#483), to two more digits than the
# issue table carries.
CANNABIS = {0: 0.00067276, 1: 0.00041379, 2: 0.00046305,
            3: 0.00048987, 4: 0.00038902, 5: 0.00041734}


class TestTheGuard:
    def test_a_gain_under_the_threshold_does_not_buy_a_factor(self):
        """0.04% better at r=2. The bare argmin takes it; the reference does not."""
        assert select_rank({0: 1.0, 1: 0.5, 2: 0.4998}) == 1

    def test_a_gain_over_the_threshold_does(self):
        assert select_rank({0: 1.0, 1: 0.5, 2: 0.49}) == 2

    def test_the_threshold_is_a_tenth_of_a_percent(self):
        """Either side of it, with the margin written out.

        The constant is ``gsynth``'s ``tol`` default, 0.001. A tenth of a
        percent of 0.5 is 5e-04, so 0.49949 clears the bar by 1e-05 and
        0.49951 misses it by the same.

        Pinned because the constant is the whole rule and it is easy to reach
        for the wrong one: ``fect`` guards the same step at 0.01, and taking
        that constant costs five of the sixteen rank agreements in
        ``benchmarks/cases/gsynth_av_laws.py`` against a live gsynth 1.2.1.
        """
        assert CV_GUARD == pytest.approx(0.001)
        assert select_rank({0: 1.0, 1: 0.5, 2: 0.49949}) == 2
        assert select_rank({0: 1.0, 1: 0.5, 2: 0.49951}) == 1

    def test_the_comparison_is_against_the_running_minimum(self):
        """Not against the preceding rank, which is the easy way to write it.

        Here r=2 is much worse than r=1 and r=3 recovers to just under it. A
        rule comparing consecutive ranks sees 0.9 -> 0.499, a 45% gain, and
        buys two factors. Against the running minimum of 0.5 the gain is
        0.04% and buys none.
        """
        assert select_rank({0: 1.0, 1: 0.5, 2: 0.9, 3: 0.4998}) == 1

    def test_a_later_rank_is_measured_against_the_best_so_far_even_if_it_wins(self):
        """The running minimum keeps falling as ranks are adopted."""
        assert select_rank({0: 1.0, 1: 0.5, 2: 0.4, 3: 0.3999}) == 2
        assert select_rank({0: 1.0, 1: 0.5, 2: 0.4, 3: 0.39}) == 3


class TestTheDegenerateScores:
    def test_a_single_rank_is_that_rank(self):
        assert select_rank({0: 1.0}) == 0

    def test_a_flat_surface_keeps_the_smallest_rank(self):
        assert select_rank({0: 0.5, 1: 0.5, 2: 0.5}) == 0

    def test_a_rising_surface_keeps_the_smallest_rank(self):
        assert select_rank({0: 0.4, 1: 0.5, 2: 0.6}) == 0

    def test_the_walk_starts_from_the_smallest_rank_offered(self):
        """The scores need not start at zero -- a rank ceiling can push the
        floor up -- and the first one offered is adopted unconditionally.

        The 0.05% gain is deliberately off the boundary. A gain of exactly
        0.1% lands on the comparison itself, where ``1.0 - 0.999`` is
        0.0010000000000000009 in binary and the assertion would be about
        float representation.
        """
        assert select_rank({2: 1.0, 3: 0.9995}) == 2
        assert select_rank({2: 1.0, 3: 0.9}) == 3

    def test_the_ranks_are_walked_in_order_and_not_in_dict_order(self):
        assert select_rank({3: 0.39, 0: 1.0, 2: 0.4, 1: 0.5}) == 3

    def test_a_non_finite_score_is_never_adopted(self):
        """A rank whose loading regression was singular scores ``inf``."""
        assert select_rank({0: 1.0, 1: float("inf"), 2: 0.5}) == 2

    def test_no_scores_at_all_is_refused(self):
        with pytest.raises(ValueError):
            select_rank({})


class TestTheGuardDoesNotExplainTheGsynthDisagreement:
    def test_the_cannabis_surface_still_selects_the_larger_rank(self):
        """r=4 beats the running minimum by 6%, so the guard passes it through.

        #483's disagreement with the published r=1 is therefore about the MSPE
        values, which come from a different fold scheme in fect, and not about
        what is done with them.
        """
        best = min(CANNABIS[0], CANNABIS[1])
        gain = (best - CANNABIS[4]) / best
        assert gain > CV_GUARD
        assert gain == pytest.approx(0.0599, abs=1e-3)
        assert select_rank(CANNABIS) == 4

    def test_a_bare_argmin_agrees_on_this_surface(self):
        """Which is why the panel could not have found the missing guard."""
        assert min(CANNABIS, key=lambda r: CANNABIS[r]) == select_rank(CANNABIS)


# --------------------------------------------------------------------------- #
# the rule is the one the estimator uses
# --------------------------------------------------------------------------- #
def factor_panel(n_units=30, n_per=24, r_true=2, onsets=(18, 20), k=2,
                 noise=0.05, seed=0):
    """A panel whose outcome is driven by ``r_true`` factors above the noise."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n_per, r_true))
    L = rng.normal(size=(n_units, r_true))
    assign = {u: g for i, g in enumerate(onsets) for u in range(i * k, (i + 1) * k)}
    rows = []
    for u in range(n_units):
        base = rng.normal(0, 1)
        g = assign.get(u)
        for t in range(n_per):
            on = g is not None and t >= g
            y = (base + F[t] @ L[u] + noise * rng.normal() + 2.0 * on)
            rows.append((u, t, y, int(on)))
    return pd.DataFrame(rows, columns=["id", "time", "y", "d"])


def fit(df, **over):
    return GSYNTH({"df": df, "outcome": "y", "treat": "d", "unitid": "id",
                   "time": "time", "display_graphs": False, "inference": False,
                   "force": "two-way", **over}).fit()


class TestTheEstimatorUsesIt:
    def test_the_selected_rank_is_what_the_rule_returns(self):
        """Wiring, asserted against the scores the fit itself reports, so it
        holds whatever the fold arithmetic produces on this panel."""
        res = fit(factor_panel(), r_max=5)
        cv = res.design.cv
        assert cv.r_selected == select_rank(cv.mspe)
        assert res.method_details.parameters_used["r"] == cv.r_selected
        assert res.method_details.parameters_used["r_source"] == "cv"

    def test_a_fixed_rank_skips_the_rule_entirely(self):
        res = fit(factor_panel(), r=3)
        assert res.method_details.parameters_used["r"] == 3
        assert res.method_details.parameters_used["r_source"] == "user"
        assert res.design.cv is None

    def test_the_criterion_finds_a_rank_it_can_see(self):
        """With two factors well above the noise, the scores must fall from
        r=0 to r=2 and stop paying for more."""
        res = fit(factor_panel(r_true=2, noise=0.05), r_max=5)
        mspe = res.design.cv.mspe
        assert mspe[2] < mspe[0]
        assert res.design.cv.r_selected >= 2
