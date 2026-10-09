"""The design search's gates and contracts, asserted where the suite was blind.

Every test here exists because a mutant survived without it. The search half
was well covered on the shape of what it returns -- one outcome per size, the
groups disjoint, the menu measured -- and uncovered on whether the individual
gates and terms compute what they are named for. A gate that passes everything
it is shown still produces a plausible design menu, so nothing downstream
noticed.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from scipy import special

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.design import objective as obj
from mlsynth.utils.tbr_helpers.design.search import (CONTROL, TREATMENT,
                                                     UNASSIGNED, _match,
                                                     _read_eligibility,
                                                     _Scorer, greedy_search)

FULL = frozenset({TREATMENT, CONTROL, UNASSIGNED})


def panel(seed: int = 0, n: int = 60, n_geos: int = 14, r: int = 2,
          noise: float = 2.0) -> np.ndarray:
    """An untreated panel with ``r`` common factors."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(n, r)), axis=0) + 100.0
    load = rng.uniform(0.6, 1.4, (r, n_geos))
    lev = rng.uniform(10, 25, n_geos)
    return lev[None, :] + 3.0 * (f @ load) + rng.normal(0, noise, (n, n_geos))


# ---------------------------------------------------------------- the CUSUM

def test_the_cusum_reads_the_largest_excursion_and_not_the_endpoint():
    """A path that wanders far and returns has to be caught.

    The residuals of a fit carrying an intercept sum to zero, so the end of the
    cumulative path is zero whatever happened along it. Reading the statistic
    there instead of at the supremum is invisible to any test that only asks
    for a probability in the unit interval.
    """
    resid = np.concatenate([np.ones(10), -np.ones(10)])
    assert np.cumsum(resid)[-1] == pytest.approx(0.0)      # why it hides
    assert np.max(np.abs(np.cumsum(resid))) == pytest.approx(10.0)
    assert obj.cusum_pvalue(resid, 1.0) < 0.05


def test_the_cusum_statistic_is_scaled_by_the_window_length():
    """The same excursion is more forgivable in a longer window.

    Without the square root of n the statistic is the raw supremum, and two
    windows that reached the same excursion score identically however much
    history separates them.
    """
    short = np.concatenate([np.ones(6), -np.ones(6)])
    long = np.concatenate([np.ones(6), -np.ones(6), np.zeros(12)])
    assert np.max(np.abs(np.cumsum(short))) == np.max(np.abs(np.cumsum(long)))
    p_short = obj.cusum_pvalue(short, 1.0)
    p_long = obj.cusum_pvalue(long, 1.0)
    assert 0.0 < p_short < p_long < 1.0


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(-5, 5, allow_nan=False), min_size=4, max_size=40),
       st.floats(0.01, 10.0, allow_nan=False))
def test_the_cusum_probability_stays_a_probability(values, sigma):
    p = obj.cusum_pvalue(np.asarray(values, dtype=float), sigma)
    assert 0.0 <= p <= 1.0


# ------------------------------------------------------- the Brownian bridge

def test_the_bridge_boundary_is_aligned_with_the_first_excursion():
    """A breach in the opening period is a breach.

    The boundary is tightest at the ends of the window, so the first period is
    where a small excursion can cross it. Comparing the path against the
    boundary one position out of step drops exactly that period, and the
    remaining points sit under a boundary that has already widened.
    """
    n = 20
    resid = np.empty(n)
    resid[0] = 3.5
    resid[1] = -1.0
    resid[2:] = -2.5 / (n - 2)
    envelope = obj.brownian_bridge_envelope(n)
    path = np.abs(np.cumsum(resid))
    assert path[0] > envelope[0]                   # the opening period breaches
    assert np.all(path[1:-1] <= envelope[1:])      # and nothing later does
    assert obj.brownian_bridge_ok(resid, 1.0) is False


# ------------------------------------------------------------ the A/A test

def test_the_false_positive_rate_is_taken_at_the_near_interval_bound():
    """The reported rate is the most forgiving one the interval allows.

    The function promises a lower bound on how often a design like this one
    would call a null window significant, which is the rate at the interval
    bound nearest zero. Taking the far bound reports a number that is larger,
    and the A/A gate then refuses designs on their worst case alone.
    """
    fit = obj.HoldoutFit(estimate=8.0, half_width=5.0, sigma=10.0,
                         n_pretest=30, df=28)
    n_test = 10

    def rate_at(true_mean: float) -> float:
        tq_sig = fit.half_width / fit.sigma
        scale = fit.sigma * np.sqrt(1.0 / fit.n_pretest + 1.0 / n_test)
        return float(1.0 - special.stdtr(fit.df, tq_sig - true_mean / scale)
                     + special.stdtr(fit.df, -tq_sig - true_mean / scale))

    near, far = 3.0, 13.0                      # the interval is (3, 13)
    assert fit.estimate - fit.half_width == near
    assert fit.estimate + fit.half_width == far
    got = obj.false_positive_probability(fit, n_test)
    assert got == pytest.approx(rate_at(near), rel=1e-12)
    # the test discriminates: the two bounds are far apart here
    assert rate_at(far) - rate_at(near) > 0.1


def test_the_aa_gate_reads_the_held_out_window_and_not_the_fit():
    """The A/A estimate has to come from periods the fit never saw.

    Estimated on its own fitting window the cumulative effect is near zero by
    construction, the interval covers zero, and every candidate passes.
    """
    P = panel(0)
    y, x = P[:, 0], P[:, 1:].sum(axis=1)
    n_test = 14
    held = obj.holdout_fit(y, x, n_test)
    assert held.n_pretest == y.size - n_test

    # The same fit read on its own pretest window: the estimate collapses
    # toward zero, which is what makes a gate reading it vacuous.
    from mlsynth.utils.tbr_helpers.posterior import (cumulative_posterior,
                                                     fit_pretest)
    f = fit_pretest(y[:held.n_pretest], x[:held.n_pretest])
    loc_in, _ = cumulative_posterior(f, y[:held.n_pretest], x[:held.n_pretest])
    assert abs(float(loc_in[-1])) < abs(held.estimate)


# -------------------------------------------------------- the power term

def test_the_detectable_impact_carries_the_power_term():
    """Eqn 6 solved for the smallest detectable effect adds both quantiles.

    The significance quantile alone answers "how wide is the interval"; the
    power quantile is what turns that into "how large an effect would clear
    it four times in five". A detectable impact missing it reports a
    sensitivity the experiment does not have, and the search ranks on exactly
    that number.
    """
    sigma, n, n_test = 7.5, 60, 14
    c = obj.window_constants(n, n_test)
    sq = np.sqrt(c.phi * (n + 1) / (n * n_test * (n - 1)) + 1.0 / n + 1.0 / n_test)
    assert obj.required_impact(sigma, n, n_test) == pytest.approx(
        (c.tq_sig + c.tq_pow) * n_test * sq * sigma, rel=1e-12)
    # both quantiles are positive at these levels, so dropping one is visible
    assert c.tq_pow > 0.0 and c.tq_sig > 0.0


@settings(max_examples=50, deadline=None)
@given(st.floats(0.1, 100.0, allow_nan=False), st.floats(0.1, 100.0, allow_nan=False))
def test_the_detectable_impact_is_increasing_in_the_residual_scale(a, b):
    lo, hi = sorted((a, b))
    n, n_test = 60, 14
    got_lo = obj.required_impact(lo, n, n_test)
    got_hi = obj.required_impact(hi, n, n_test)
    assert got_lo > 0.0
    assert got_lo <= got_hi


# ------------------------------------------------- where the control walk starts

def _toggleable(elig, treatment, pools):
    return [j for j in pools["control"]
            if j not in treatment and UNASSIGNED in elig[j]]


@pytest.mark.parametrize("start", ["carried", "pool"])
def test_each_control_start_begins_where_it_is_documented_to(start):
    """The two starts are told apart by the group they begin the match from.

    ``carried`` hands the previous size's matched control group forward, less
    whatever geo just joined the treatment group. ``pool`` re-derives the start
    from every control-eligible geo the treatment group leaves free. Both
    return a matched group of the same shape, so a suite that only checks the
    shape cannot tell them apart -- or notice them exchanged.
    """
    P = panel(0)
    elig = [FULL] * P.shape[1]
    pools = _read_eligibility(elig)
    n_test = 14

    walks = {s: greedy_search(P, elig, max_treatment_size=2, n_test=n_test,
                              objective="reference", control_start=s)
             for s in ("carried", "pool")}
    first, second = walks[start]

    # The panel has to be one where the two starts disagree, or this asserts
    # nothing: an exchanged pair of starts would pass unnoticed.
    assert set(walks["carried"][1].control) != set(walks["pool"][1].control)

    joined = set(second.treatment) - set(first.treatment)
    assert len(joined) == 1
    begins_from = (set(first.control) - joined if start == "carried"
                   else set(pools["control"]) - set(second.treatment))

    scorer = _Scorer(P, "reference", n_test)
    expected, _, _ = _match(scorer, list(second.treatment), begins_from,
                            _toggleable(elig, second.treatment, pools))
    assert set(second.control) == expected


def test_a_treatment_size_that_would_leave_no_control_geo_is_refused():
    """K has to leave a control aggregate behind, and the refusal says so.

    When every control-eligible geo is also treatment eligible, one of them has
    to stay out of the treatment group, so the largest attainable K is one
    below the pool. Admitting that K does not produce a wrong answer -- the
    augmentation step runs out of candidates and raises too -- so the check is
    pinned on which refusal arrives. The feasibility message means it was
    refused before anything was scored; the augmentation message means an
    exact precondition was reached by climbing, which is the branch marked
    unreachable.
    """
    P = panel(1, n_geos=5)
    elig = [frozenset({TREATMENT, CONTROL})] * 5
    with pytest.raises(MlsynthDataError, match="is not attainable"):
        greedy_search(P, elig, max_treatment_size=5, n_test=14,
                      objective="reference")
