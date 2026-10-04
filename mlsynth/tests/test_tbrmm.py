"""TBRMM: the matched-markets hill climb (Au 2018).

Test-first, per ``agents/agents_tests.md``: written before the estimator exists
and RED until it lands.

TBRMM chooses which geos to treat before an experiment runs, so it returns a
:class:`~mlsynth.config_models.DesignResult` and not an effect. Algorithm 1
alternates two routines until the treatment group reaches its allowed maximum:

* matching -- given the treatment group, toggle the single control-group
  membership that most improves the objective, and stop when none does;
* augmentation -- given the control group, add the one treatment-eligible geo
  that most improves the objective, then match again.

The result is one recommended pair per treatment size ``k``, which is what the
advertiser chooses between.

Two objectives, because the paper and the reference implementation specify
different ones and the difference was measured, not assumed. Au section
3.1 states ``f = min(CUSUM p, Breusch-Godfrey p, R^2)``. The reference scores
``(corr_test, aa_test, bb_test, dw_test, corr, 1/required_impact)``
lexicographically. On 400 random splits of the GeoLift markets the two rank
designs at Spearman +0.54 with no overlap in their top five, because Au's ``f``
carries no power term at all while the reference's ends in one: the reference's
top ten are 2.5x better on detectable impact. Au's own section 3.1 asks for a
power term and then omits it from ``f``, so ``objective="reference"`` is the
default and ``objective="paper"`` is there for fidelity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import TBR
from mlsynth.config_models import TBRConfig, DesignResult
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.tbr_helpers.design.search import CONTROL as CONTROL_ROLE
from mlsynth.utils.tbr_helpers.design.objective import required_impact, score_split
from mlsynth.utils.tbr_helpers.posterior import fit_pretest


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def market_panel(n_units=12, T=40, seed=0, size_log_sd=0.8, common_sd=0.6,
                 idio_sd=0.12):
    """Geos of differing size sharing a daily factor, so the group aggregates of
    any split track each other the way real markets do."""
    rng = np.random.default_rng(seed)
    size = rng.lognormal(0.0, size_log_sd, n_units)
    size = 100.0 * size / size.mean()
    common = 1.0 + common_sd * rng.normal(size=(T, 1))
    idio = 1.0 + idio_sd * rng.normal(size=(T, n_units))
    y = size * (0.5 * common + 0.5 * idio)
    rows = []
    for j in range(n_units):
        for t in range(T):
            rows.append(dict(geo=f"m{j:02d}", date=t, Y=float(y[t, j])))
    return pd.DataFrame(rows)


def base_config(df, **over):
    kw = dict(df=df, unitid="geo", time="date", outcome="Y",
              max_treatment_size=3, n_test=10)
    kw.update(over)
    return TBRConfig(**kw)


def with_eligibility(df, forced_treatment=(), no_treatment=(), no_control=(),
                     no_unassigned=()):
    """Attach Au's ``A_i`` as three per-unit boolean columns."""
    out = df.copy()
    out["can_treat"] = (~out.geo.isin(no_treatment)).astype(int)
    out["can_control"] = (~out.geo.isin(no_control)).astype(int)
    out["can_exclude"] = (~out.geo.isin(no_unassigned)).astype(int)
    for g in forced_treatment:                 # A_i = {treatment}
        out.loc[out.geo == g, ["can_treat", "can_control", "can_exclude"]] = [1, 0, 0]
    return out


def elig_config(df, **over):
    kw = dict(treatment_eligible_col="can_treat",
              control_eligible_col="can_control",
              unassigned_eligible_col="can_exclude")
    kw.update(over)
    return base_config(df, **kw)


# --------------------------------------------------------------------------- #
# Layer 4: smoke
# --------------------------------------------------------------------------- #
def test_fits_and_returns_a_design_result():
    res = TBR(base_config(market_panel())).fit()
    assert isinstance(res, DesignResult)
    assert res.report is None          # a design resolves to a report later


def test_one_recommended_pair_per_treatment_size():
    K = 4
    res = TBR(base_config(market_panel(), max_treatment_size=K)).fit()
    assert [d.k for d in res.designs] == list(range(1, K + 1))


def test_the_recommendation_is_one_of_the_designs():
    res = TBR(base_config(market_panel(), max_treatment_size=4)).fit()
    assert res.recommended in res.designs


# --------------------------------------------------------------------------- #
# Layer 2: what Algorithm 1 guarantees
# --------------------------------------------------------------------------- #
def test_a_design_of_size_k_treats_exactly_k_geos():
    res = TBR(base_config(market_panel(), max_treatment_size=5)).fit()
    for d in res.designs:
        assert len(d.treatment_units) == d.k


def test_the_three_groups_partition_the_panel():
    df = market_panel()
    res = TBR(base_config(df, max_treatment_size=4)).fit()
    units = set(df.geo.unique())
    for d in res.designs:
        t, c, u = set(d.treatment_units), set(d.control_units), set(d.unassigned_units)
        assert t & c == set() and t & u == set() and c & u == set()
        assert t | c | u == units


def test_no_design_has_an_empty_control_group():
    res = TBR(base_config(market_panel(), max_treatment_size=5)).fit()
    for d in res.designs:
        assert len(d.control_units) >= 1


def test_the_objective_recorded_is_the_objective_of_those_groups():
    """A design carries its own score, so recomputing it must agree."""
    from mlsynth.utils.tbr_helpers.design.objective import score_split
    df = market_panel()
    res = TBR(base_config(df, max_treatment_size=3)).fit()
    wide = df.pivot_table(index="date", columns="geo", values="Y")
    for d in res.designs:
        again = score_split(
            wide[list(d.treatment_units)].sum(axis=1).to_numpy(),
            wide[list(d.control_units)].sum(axis=1).to_numpy(),
            objective="reference", n_test=10)
        assert again.value == pytest.approx(d.objective_value, rel=1e-12)


def test_matching_never_accepts_a_worse_control_group():
    """Hill climbing: the objective along the trace is non-decreasing."""
    res = TBR(base_config(market_panel(), max_treatment_size=4)).fit()
    for d in res.designs:
        trace = list(d.matching_trace)
        assert trace, "a design must record the climb that produced it"
        assert all(b >= a for a, b in zip(trace, trace[1:])), trace


def test_the_climb_stops_when_no_toggle_improves():
    """The last matching step is the one that failed to improve, so the final
    value repeats or the trace has length one."""
    res = TBR(base_config(market_panel(), max_treatment_size=2)).fit()
    for d in res.designs:
        assert d.matching_converged is True


# --------------------------------------------------------------------------- #
# eligibility: Au's A_i
# --------------------------------------------------------------------------- #
def test_a_forced_treatment_geo_is_in_every_treatment_group():
    df = with_eligibility(market_panel(), forced_treatment=["m00", "m01"])
    res = TBR(elig_config(df, max_treatment_size=4)).fit()
    assert [d.k for d in res.designs] == [2, 3, 4]      # k starts at k0 = 2
    for d in res.designs:
        assert {"m00", "m01"} <= set(d.treatment_units)


def test_a_geo_ineligible_for_treatment_is_never_treated():
    df = with_eligibility(market_panel(), no_treatment=["m02", "m03", "m04"])
    res = TBR(elig_config(df, max_treatment_size=5)).fit()
    for d in res.designs:
        assert {"m02", "m03", "m04"}.isdisjoint(d.treatment_units)


def test_a_geo_ineligible_for_control_is_never_a_control():
    df = with_eligibility(market_panel(), no_control=["m05", "m06"])
    res = TBR(elig_config(df, max_treatment_size=3)).fit()
    for d in res.designs:
        assert {"m05", "m06"}.isdisjoint(d.control_units)


def test_a_geo_that_cannot_be_unassigned_stays_assigned():
    df = with_eligibility(market_panel(), no_unassigned=["m07"])
    res = TBR(elig_config(df, max_treatment_size=3)).fit()
    for d in res.designs:
        assert "m07" in set(d.treatment_units) | set(d.control_units)


def test_k0_forced_larger_than_one_skips_the_smaller_sizes():
    df = with_eligibility(market_panel(), forced_treatment=["m00", "m01", "m02"])
    res = TBR(elig_config(df, max_treatment_size=4)).fit()
    assert [d.k for d in res.designs] == [3, 4]


# --------------------------------------------------------------------------- #
# the two objectives
# --------------------------------------------------------------------------- #
def test_both_objectives_run_and_record_which_was_used():
    df = market_panel(n_units=14, seed=3)
    for obj in ("reference", "paper"):
        res = TBR(base_config(df, objective=obj, max_treatment_size=3)).fit()
        assert res.objective == obj
        assert all(np.isfinite(d.objective_value) for d in res.designs)


def test_the_paper_objective_reports_its_three_components():
    df = market_panel(n_units=14, seed=4)
    res = TBR(base_config(df, objective="paper", max_treatment_size=2)).fit()
    d = res.designs[-1]
    assert 0.0 <= d.detail["p_cusum"] <= 1.0
    assert 0.0 <= d.detail["p_bg"] <= 1.0
    assert 0.0 <= d.detail["r2"] <= 1.0
    # f is the minimum of the three
    assert d.objective_value == pytest.approx(
        min(d.detail["p_cusum"], d.detail["p_bg"], d.detail["r2"]), rel=1e-12)


def test_the_reference_objective_reports_its_gates_and_power():
    df = market_panel(n_units=14, seed=5)
    res = TBR(base_config(df, objective="reference", max_treatment_size=2)).fit()
    d = res.designs[-1]
    for key in ("corr_test", "aa_test", "bb_test", "dw_test"):
        assert isinstance(d.detail[key], bool)
    assert d.detail["required_impact"] > 0.0
    assert -1.0 <= d.detail["corr"] <= 1.0


def test_the_objective_scores_the_model_tbr_will_actually_fit():
    """One definition of the pretest relation, shared with TBR.

    Both objectives are functionals of eqn 1's fit: the CUSUM and
    Breusch-Godfrey gates read its residuals, the correlation term is its fit
    quality, and ``required_impact`` is eqn 6's half-width solved for the
    smallest detectable effect. A search that scored a different regression from
    the one the experiment is later analysed with would rank designs by a
    quantity no estimator computes, so the residual scale the objective uses is
    the residual scale :func:`mlsynth.utils.tbr_helpers.posterior.fit_pretest`
    returns.
    """
    from mlsynth.utils.tbr_helpers.posterior import fit_pretest
    from mlsynth.utils.tbr_helpers.design.objective import _pretest_fit

    rng = np.random.default_rng(11)
    for _ in range(8):
        n = int(rng.integers(12, 60))
        x = rng.normal(100.0, 20.0, n)
        y = 3.0 + 0.8 * x + rng.normal(0.0, 4.0, n)
        _, resid, sigma, r2 = _pretest_fit(y, x)
        fit = fit_pretest(y, x)
        assert sigma == pytest.approx(np.sqrt(fit.sigma_sq), rel=1e-12)
        assert resid == pytest.approx(y - (fit.alpha + fit.beta * x), abs=1e-9)
        assert r2 == pytest.approx(float(np.corrcoef(y, x)[0, 1]) ** 2, rel=1e-10)


def test_the_objective_refuses_a_pretest_too_short_for_a_residual_scale():
    """Two pretest periods leave no degrees of freedom, so eqn 1 has no residual
    scale and every gate downstream is a division by zero. A design scored on
    such a window would come back as a number, which is the one outcome a
    malformed panel must not produce."""
    from mlsynth.utils.tbr_helpers.design.objective import _pretest_fit

    with pytest.raises(MlsynthDataError, match="degrees of freedom"):
        _pretest_fit(np.array([1.0, 2.0]), np.array([1.0, 3.0]))


def test_the_two_objectives_can_disagree_on_the_recommendation():
    """Measured on the GeoLift markets: Spearman +0.54, no top-five overlap. A
    build that could not express the disagreement would be hiding it."""
    df = market_panel(n_units=16, seed=6)
    a = TBR(base_config(df, objective="reference", max_treatment_size=4)).fit()
    b = TBR(base_config(df, objective="paper", max_treatment_size=4)).fit()
    assert set(a.recommended.treatment_units) != set(b.recommended.treatment_units)


# --------------------------------------------------------------------------- #
# invariances
# --------------------------------------------------------------------------- #
def test_relabelling_the_geos_selects_the_same_markets():
    df = market_panel(n_units=12, seed=8)
    renamed = df.copy()
    mapping = {g: f"z{i:02d}" for i, g in enumerate(sorted(df.geo.unique())[::-1])}
    renamed["geo"] = renamed.geo.map(mapping)
    a = TBR(base_config(df, max_treatment_size=3)).fit()
    b = TBR(base_config(renamed, max_treatment_size=3)).fit()
    assert {mapping[g] for g in a.recommended.treatment_units} == \
           set(b.recommended.treatment_units)


def test_scaling_the_outcome_selects_the_same_markets():
    df = market_panel(n_units=12, seed=9)
    big = df.copy(); big["Y"] = big["Y"] * 1000.0
    a = TBR(base_config(df, max_treatment_size=3)).fit()
    b = TBR(base_config(big, max_treatment_size=3)).fit()
    assert set(a.recommended.treatment_units) == set(b.recommended.treatment_units)


def test_the_search_is_deterministic():
    df = market_panel(n_units=14, seed=10)
    a = TBR(base_config(df, max_treatment_size=4)).fit()
    b = TBR(base_config(df, max_treatment_size=4)).fit()
    assert [d.treatment_units for d in a.designs] == \
           [d.treatment_units for d in b.designs]
    assert [d.objective_value for d in a.designs] == \
           [d.objective_value for d in b.designs]


# --------------------------------------------------------------------------- #
# Layer 3: edges
# --------------------------------------------------------------------------- #
def test_max_treatment_size_of_one():
    res = TBR(base_config(market_panel(), max_treatment_size=1)).fit()
    assert len(res.designs) == 1 and res.designs[0].k == 1


def test_the_smallest_workable_panel():
    """Three geos: one treated, one control, one free."""
    res = TBR(base_config(market_panel(n_units=3, T=20),
                            max_treatment_size=1)).fit()
    assert len(res.designs[0].treatment_units) == 1


def test_treating_every_eligible_geo_but_one():
    df = market_panel(n_units=6, T=25)
    res = TBR(base_config(df, max_treatment_size=5)).fit()
    assert res.designs[-1].k == 5
    assert len(res.designs[-1].control_units) == 1


def test_a_post_column_restricts_scoring_to_the_pre_rows():
    df = market_panel(n_units=10, T=40, seed=11)
    df["post"] = (df.date >= 30).astype(int)
    a = TBR(base_config(df, post_col="post", max_treatment_size=2)).fit()
    b = TBR(base_config(df[df.date < 30].drop(columns=["post"]),
                          max_treatment_size=2)).fit()
    assert [d.objective_value for d in a.designs] == \
           pytest.approx([d.objective_value for d in b.designs], rel=1e-12)


# --------------------------------------------------------------------------- #
# Layer 1: the objective on degenerate input
# --------------------------------------------------------------------------- #
def test_every_gate_fails_a_split_with_no_residual_scale():
    """A control aggregate reproducing the treatment aggregate exactly leaves
    sigma = 0. No gate can be evaluated against a scale of zero, so each reports
    failure instead of dividing by it -- a split that cannot be tested is not a
    split that passed."""
    from mlsynth.utils.tbr_helpers.design import objective as obj

    resid = np.zeros(20)
    assert obj.cusum_pvalue(resid, 0.0) == 0.0
    assert obj.brownian_bridge_ok(resid, 0.0) is False
    passed, stat = obj.durbin_watson_ok(resid)
    assert passed is False and np.isnan(stat)


def test_the_cusum_tail_is_one_where_the_statistic_carries_no_information():
    from mlsynth.utils.tbr_helpers.design.objective import _kolmogorov_sf

    assert _kolmogorov_sf(0.0) == 1.0
    assert _kolmogorov_sf(float("nan")) == 1.0


def test_breusch_godfrey_declines_a_window_too_short_for_its_lag():
    """Four residuals and one lag leave nothing to regress; the test reports no
    evidence of autocorrelation instead of a value from an empty fit."""
    from mlsynth.utils.tbr_helpers.design.objective import breusch_godfrey_pvalue

    assert breusch_godfrey_pvalue(np.arange(4.0), np.arange(4.0)) == 1.0


def _stable_pair(n=60, seed=2):
    """Two series in a stable linear relation, which is what TBR assumes."""
    rng = np.random.default_rng(seed)
    x = np.linspace(100.0, 200.0, n) + rng.normal(0.0, 1.0, n)
    return 5.0 + 1.5 * x + rng.normal(0.0, 1.0, n), x


def test_the_aa_test_passes_a_pretest_that_predicts_its_own_last_window():
    """TBR run on the pretest against itself. The held-out window carries no
    intervention, so an interval covering zero is the design declining to find
    an effect where there is none."""
    from mlsynth.utils.tbr_helpers.design.objective import aa_test_ok

    y, x = _stable_pair()
    assert aa_test_ok(y, x, 12) is True


def test_the_aa_test_fails_a_pretest_that_breaks_before_its_last_window():
    """A level shift in the held-out window is an effect the design would report
    out of a period where nothing happened, which is its false-positive rate
    showing before the experiment is run."""
    from mlsynth.utils.tbr_helpers.design.objective import aa_test_ok

    y, x = _stable_pair(seed=3)
    y = y.copy()
    y[-12:] += 400.0
    assert aa_test_ok(y, x, 12) is False


def test_the_false_positive_probability_is_read_from_the_holdout_fit():
    """The probability is reached only when the interval excludes zero, and it
    is taken at the bound nearest zero, so it is a lower bound on how often the
    design would cry wolf."""
    from mlsynth.utils.tbr_helpers.design.objective import (
        false_positive_probability, holdout_fit)

    y, x = _stable_pair(seed=4)
    y = y.copy()
    y[-12:] += 400.0
    fit = holdout_fit(y, x, 12)
    assert fit.n_pretest == 48 and fit.df == 46
    assert abs(fit.estimate) - fit.half_width > 0      # the interval excludes zero
    assert 0.0 <= false_positive_probability(fit, 12) <= 1.0


def test_the_aa_test_refuses_a_window_leaving_no_pretest_to_fit():
    """The A/A test fits on what the held-out window leaves behind, so a test
    length that consumes the pretest has nothing to fit on."""
    from mlsynth.utils.tbr_helpers.design.objective import aa_test_ok

    with pytest.raises(MlsynthDataError, match="A/A test"):
        aa_test_ok(np.linspace(1.0, 10.0, 10), np.linspace(2.0, 11.0, 10), 9)


def test_a_score_orders_by_its_key_and_not_its_scalar():
    """The reference objective's scalar readout can fall on a step that gains a
    gate, so the comparison is the key."""
    from mlsynth.utils.tbr_helpers.design.objective import SplitScore

    weaker = SplitScore(key=(0, 9.0), value=9.0)
    stronger = SplitScore(key=(1, 0.1), value=0.1)
    assert weaker < stronger


def test_the_window_constants_are_a_function_of_the_window_alone():
    """Every quantile the score needs depends on the window length and the test
    length, not on which geos a candidate puts where, so two different splits of
    one panel share them. Caching them is what makes that reuse explicit."""
    from mlsynth.utils.tbr_helpers.design.objective import window_constants

    a = window_constants(90, 14)
    b = window_constants(90, 14)
    assert a is b                       # the cache hands back the same object
    assert a.holdout_df == 90 - 14 - 2
    assert a.tq_sig > 0 and a.tq_pow > 0 and a.phi > 0
    assert window_constants(60, 14) != a


def test_the_holdout_quantile_is_absent_where_the_window_cannot_support_one():
    """A test length that consumes the window leaves the A/A fit no degrees of
    freedom. The quantile is reported as nan; holdout_fit refuses such a window
    before it would be used."""
    from mlsynth.utils.tbr_helpers.design.objective import window_constants

    const = window_constants(10, 9)
    assert const.holdout_df <= 0
    assert np.isnan(const.tq_holdout)


def test_the_brownian_bridge_envelope_is_shared_and_read_only():
    """The cache hands one array to every caller, so a caller that wrote to it
    would change the boundary every later split is tested against."""
    from mlsynth.utils.tbr_helpers.design.objective import brownian_bridge_envelope

    env = brownian_bridge_envelope(90)
    assert env.shape == (89,)
    assert brownian_bridge_envelope(90) is env
    with pytest.raises(ValueError):
        env[0] = 0.0


# --------------------------------------------------------------------------- #
# Layer 1: the search's own contracts
# --------------------------------------------------------------------------- #
def _flat_panel(n_periods=20, n_units=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(100.0, 5.0, (n_periods, n_units))


def test_the_search_refuses_a_role_it_does_not_know():
    from mlsynth.utils.tbr_helpers.design.search import greedy_search

    elig = [frozenset({"treatment", "control", "unassigned"})] * 3 + [frozenset({"donor"})]
    with pytest.raises(MlsynthDataError, match="role"):
        greedy_search(_flat_panel(), elig, max_treatment_size=1, n_test=4,
                      objective="reference")


def test_the_search_refuses_a_window_that_is_not_periods_by_geos():
    from mlsynth.utils.tbr_helpers.design.search import greedy_search

    with pytest.raises(MlsynthDataError, match="periods by geos"):
        greedy_search(np.arange(12.0), [frozenset({CONTROL_ROLE})],
                      max_treatment_size=1, n_test=2, objective="reference")


def test_eligibility_has_to_cover_every_geo_in_the_window():
    from mlsynth.utils.tbr_helpers.design.search import greedy_search

    with pytest.raises(MlsynthDataError, match="eligibility covers"):
        greedy_search(_flat_panel(n_units=4), [frozenset({CONTROL_ROLE})] * 3,
                      max_treatment_size=1, n_test=4, objective="reference")


def test_augmentation_rebuilds_a_control_group_the_climb_pruned_to_one():
    """A climb can leave a single control geo. Treating it would empty the
    control aggregate, so the pool is rebuilt from every control-eligible geo the
    new treatment group leaves free, and the next size is still reachable."""
    from mlsynth.utils.tbr_helpers.design.search import _Scorer, _augment

    scorer = _Scorer(_flat_panel(n_units=4, seed=3), "reference", 4)
    pools = {"forced": [], "treatment": [0, 1, 2, 3], "control": [1, 2]}
    treatment, control = _augment(scorer, [0], {1}, pools)
    assert len(treatment) == 2 and control


def test_augmentation_skips_a_geo_no_rebuild_can_leave_a_control_for():
    """Geo 1 is the only control-eligible geo, so treating it leaves nothing to
    regress on however the pool is rebuilt; the candidate is passed over and a
    geo that does leave a control group is taken."""
    from mlsynth.utils.tbr_helpers.design.search import _Scorer, _augment

    scorer = _Scorer(_flat_panel(n_units=3, seed=4), "reference", 4)
    pools = {"forced": [], "treatment": [0, 1, 2], "control": [1]}
    treatment, control = _augment(scorer, [0], {1}, pools)
    assert treatment == [0, 2] and control == {1}


# --------------------------------------------------------------------------- #
# failure tests
# --------------------------------------------------------------------------- #
def test_a_max_treatment_size_of_zero_is_refused():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        base_config(market_panel(), max_treatment_size=0)


def test_a_non_positive_test_length_is_refused():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        base_config(market_panel(), n_test=0)


def test_an_unknown_objective_is_refused():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        base_config(market_panel(), objective="cusum_only")


def test_a_max_treatment_size_beyond_the_eligible_geos_raises():
    df = with_eligibility(market_panel(n_units=6),
                          no_treatment=["m02", "m03", "m04", "m05"])
    with pytest.raises(MlsynthDataError, match="eligible"):
        TBR(elig_config(df, max_treatment_size=4)).fit()


def test_no_control_eligible_geo_raises():
    df = with_eligibility(market_panel(n_units=5),
                          no_control=["m00", "m01", "m02", "m03", "m04"])
    with pytest.raises(MlsynthDataError, match="control"):
        TBR(elig_config(df, max_treatment_size=2)).fit()


def test_a_geo_eligible_for_nothing_raises():
    df = with_eligibility(market_panel(n_units=5))
    df.loc[df.geo == "m02", ["can_treat", "can_control", "can_exclude"]] = 0
    with pytest.raises(MlsynthDataError, match="eligible for no"):
        TBR(elig_config(df, max_treatment_size=2)).fit()


def test_an_eligibility_column_varying_within_unit_raises():
    df = with_eligibility(market_panel(n_units=6))
    df.loc[(df.geo == "m01") & (df.date > 10), "can_treat"] = 0
    with pytest.raises(MlsynthDataError, match="constant"):
        TBR(elig_config(df, max_treatment_size=2)).fit()


def test_a_missing_eligibility_column_raises():
    with pytest.raises(MlsynthDataError, match="can_treat"):
        TBR(elig_config(market_panel(), max_treatment_size=2)).fit()


def test_too_few_periods_to_fit_raises():
    with pytest.raises(MlsynthDataError):
        TBR(base_config(market_panel(n_units=6, T=2),
                          max_treatment_size=2)).fit()


def test_a_repeated_unit_period_cell_is_refused():
    """Group aggregates sum across units, so a repeated cell has no one reading:
    a duplicate row and two records meant to be added together are the same
    input. The base configuration owns this invariant; the test pins that TBRMM
    inherits it and never reaches the search."""
    df = market_panel(n_units=5, T=20)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(MlsynthDataError, match="[Dd]uplicate"):
        TBR(base_config(df, max_treatment_size=2)).fit()


def test_a_gap_in_the_scoring_window_is_refused():
    """A missing cell drops that geo from whichever aggregate it lands in for
    those periods, which changes the aggregate without changing its length."""
    df = market_panel(n_units=5, T=20)
    df = df[~((df.geo == "m02") & (df.date == 7))]
    with pytest.raises(MlsynthDataError, match="missing"):
        TBR(base_config(df, max_treatment_size=2)).fit()


def test_a_max_treatment_size_below_the_forced_count_is_refused():
    """Three geos are eligible for treatment alone, so every design contains
    them; a K of two describes no partition."""
    df = with_eligibility(market_panel(n_units=6),
                          forced_treatment=["m00", "m01", "m02"])
    with pytest.raises(MlsynthDataError, match="forced"):
        TBR(elig_config(df, max_treatment_size=2)).fit()


def test_treating_every_geo_leaves_no_control_and_is_refused():
    """Five geos, all eligible for everything: K of five leaves nothing to
    regress on, and the refusal comes before any climbing."""
    df = market_panel(n_units=5, T=25)
    with pytest.raises(MlsynthDataError, match="eligible"):
        TBR(base_config(df, max_treatment_size=5)).fit()


def test_a_blank_column_name_is_refused():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        base_config(market_panel(), post_col="   ", max_treatment_size=2)


def test_the_dict_path_builds_the_config():
    df = market_panel(n_units=6, T=25)
    res = TBR(dict(df=df, unitid="geo", time="date", outcome="Y",
                     max_treatment_size=2, n_test=6)).fit()
    assert res.recommended is not None


def test_an_invalid_dict_is_translated_to_a_config_error():
    df = market_panel(n_units=6, T=25)
    with pytest.raises(MlsynthConfigError, match="TBRConfig"):
        TBR(dict(df=df, unitid="geo", time="date", outcome="Y",
                   max_treatment_size=0, n_test=6))


def test_a_non_config_input_raises_a_config_error():
    with pytest.raises(MlsynthConfigError, match="TBRConfig"):
        TBR(17)


# --------------------------------------------------------------------------- #
# the score reads the fit's sums, and takes no second pass
# --------------------------------------------------------------------------- #
def test_the_correlation_is_the_pearson_correlation():
    """Position five of the key is ``round(corr, 2)``, so ``corr`` is pinned.

    Derived from the fit's centred sums as ``S_xy / sqrt(S_xx S_yy)`` in place
    of a second pass through ``np.corrcoef``. The referee is ``np.corrcoef``.
    """
    rng = np.random.default_rng(21)
    for _ in range(25):
        x = 30000.0 + 4000.0 * rng.standard_normal(80)
        y = 0.3 * x + 500.0 * rng.standard_normal(80)
        got = score_split(y, x, objective="reference", n_test=12).detail["corr"]
        want = float(np.corrcoef(y, x)[0, 1])
        assert got == pytest.approx(want, rel=1e-12)


def test_required_impact_is_the_reference_formula_through_the_residual_scale():
    """``std(y, ddof=2) sqrt(1 - corr^2)`` is ``sqrt(sigma_sq)``.

    The reference writes eqn 6's scale the first way and the fit already holds
    the second, so the identity is what lets the impact read it off the fit.
    Asserted here against the reference's own arrangement.
    """
    rng = np.random.default_rng(22)
    for _ in range(25):
        x = 8000.0 + 900.0 * rng.standard_normal(70)
        y = 1.2 * x + 200.0 * rng.standard_normal(70)
        fit = fit_pretest(y, x)
        corr = float(np.corrcoef(y, x)[0, 1])
        want = float(np.std(y, ddof=2)) * np.sqrt(max(1.0 - corr ** 2, 0.0))
        assert np.sqrt(fit.sigma_sq) == pytest.approx(want, rel=1e-10)
        assert (score_split(y, x, objective="reference", n_test=10)
                .detail["required_impact"]
                == pytest.approx(required_impact(np.sqrt(fit.sigma_sq), y.size, 10),
                                 rel=1e-12))


# --------------------------------------------------------------------------- #
# where the control group search starts
# --------------------------------------------------------------------------- #
def _design_key(design):
    """The lexicographic key the climb maximises, restated from the result.

    Rebuilt from the reported diagnostics and not read off the search, so a test
    comparing two designs compares them on the objective as documented.
    """
    d = design.detail
    return (int(d["corr_test"]), int(d["aa_test"]), int(d["bb_test"]),
            int(d["dw_test"]), round(d["corr"], 2), 1.0 / d["required_impact"])


def _by_size(df, control_start):
    res = TBR(base_config(df, control_start=control_start)).fit()
    return {d.k: d for d in res.designs}


def test_the_control_group_search_starts_from_the_carried_group_by_default():
    """Algorithm 1 hands each size's matched control group to the next size.

    That is what the reference does and what the benchmark pins, so it stays the
    default and naming it explicitly changes nothing.
    """
    df = market_panel(n_units=12, T=40, seed=1)
    assert base_config(df).control_start == "carried"
    default = _by_size(df, "carried")
    named = {d.k: d for d in TBR(base_config(df)).fit().designs}
    for k in default:
        assert sorted(default[k].treatment_units) == sorted(named[k].treatment_units)
        assert sorted(default[k].control_units) == sorted(named[k].control_units)


def test_the_pool_start_reaches_a_different_partition():
    """Matching is a single-toggle climb, so its answer depends on where it starts.

    Re-deriving the starting control group from the whole pool at each size lands
    on a different local optimum, which is the point of the option.
    """
    df = market_panel(n_units=12, T=40, seed=1)
    carried, pool = _by_size(df, "carried"), _by_size(df, "pool")
    differing = [k for k in carried
                 if sorted(carried[k].treatment_units) != sorted(pool[k].treatment_units)
                 or sorted(carried[k].control_units) != sorted(pool[k].control_units)]
    assert differing, "the two starts agree everywhere; this panel cannot separate them"


def test_best_returns_whichever_of_the_two_scores_higher():
    """Neither start dominates, so ``best`` runs both and takes the winner.

    Seed 7 is chosen because the two disagree in both directions on it: the
    carried walk wins one size and the pool walk another, so a rule that simply
    preferred one of them would fail here.
    """
    df = market_panel(n_units=12, T=40, seed=7)
    carried, pool, best = (_by_size(df, s) for s in ("carried", "pool", "best"))
    verdicts = set()
    for k in carried:
        a, b = _design_key(carried[k]), _design_key(pool[k])
        assert _design_key(best[k]) == max(a, b)
        verdicts.add("carried" if a > b else "pool" if b > a else "tie")
    assert {"carried", "pool"} <= verdicts, "this seed no longer disagrees both ways"


@pytest.mark.parametrize("seed", [0, 1, 3, 6, 7, 8])
def test_best_is_never_worse_than_the_reference_walk(seed):
    """The option can only cost time.

    ``best`` takes the maximum over a set containing the carried walk's design,
    so its score is at least that one's at every treatment size.
    """
    df = market_panel(n_units=12, T=40, seed=seed)
    carried, best = _by_size(df, "carried"), _by_size(df, "best")
    for k in carried:
        assert _design_key(best[k]) >= _design_key(carried[k])


def test_a_tie_between_the_two_starts_returns_the_reference_walk():
    """An even contest gives back the design the reference would have chosen."""
    df = market_panel(n_units=12, T=40, seed=2)
    carried, pool, best = (_by_size(df, s) for s in ("carried", "pool", "best"))
    tied = [k for k in carried if _design_key(carried[k]) == _design_key(pool[k])]
    assert tied, "this panel no longer ties; the test has lost its power"
    for k in tied:
        assert sorted(best[k].treatment_units) == sorted(carried[k].treatment_units)
        assert sorted(best[k].control_units) == sorted(carried[k].control_units)


def test_every_start_returns_a_control_group_no_single_toggle_improves():
    """Whatever the start, matching runs to convergence."""
    for start in ("carried", "pool", "best"):
        res = TBR(base_config(market_panel(n_units=12, T=40, seed=1),
                                control_start=start)).fit()
        for d in res.designs:
            assert d.matching_converged is True


def test_the_three_starts_are_the_only_ones_accepted():
    """The accepted set is asserted alongside the refusal.

    Refusing an unknown value alone would pass on any build without the option
    at all, since the base configuration forbids unknown fields, so the check
    has to name what the option does admit.
    """
    df = market_panel(n_units=12, T=40, seed=0)
    for start in ("carried", "pool", "best"):
        assert base_config(df, control_start=start).control_start == start
    with pytest.raises((MlsynthConfigError, ValueError)):
        base_config(df, control_start="fresh")


def test_the_search_refuses_an_unknown_control_start_of_its_own():
    """The configuration's ``Literal`` never lets a bad value reach the search,
    so the search states its own admissible set for a caller who imports it."""
    from mlsynth.utils.tbr_helpers.design.search import greedy_search
    y = np.asarray([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 5.0, 2.0],
                    [4.0, 3.0, 6.0]])
    elig = [frozenset({"treatment", "control", "unassigned"})] * 3
    with pytest.raises(MlsynthDataError, match="control_start"):
        greedy_search(y, elig, max_treatment_size=1, n_test=1,
                      objective="reference", control_start="fresh")
