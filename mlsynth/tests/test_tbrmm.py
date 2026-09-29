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
different ones and the difference was measured rather than assumed. Au section
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

from mlsynth import TBRMM
from mlsynth.config_models import TBRMMConfig, DesignResult
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError


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
              max_treatment_size=3, n_test=10, display_graphs=False)
    kw.update(over)
    return TBRMMConfig(**kw)


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
    res = TBRMM(base_config(market_panel())).fit()
    assert isinstance(res, DesignResult)
    assert res.report is None          # a design resolves to a report later


def test_one_recommended_pair_per_treatment_size():
    K = 4
    res = TBRMM(base_config(market_panel(), max_treatment_size=K)).fit()
    assert [d.k for d in res.designs] == list(range(1, K + 1))


def test_the_recommendation_is_one_of_the_designs():
    res = TBRMM(base_config(market_panel(), max_treatment_size=4)).fit()
    assert res.recommended in res.designs


# --------------------------------------------------------------------------- #
# Layer 2: what Algorithm 1 guarantees
# --------------------------------------------------------------------------- #
def test_a_design_of_size_k_treats_exactly_k_geos():
    res = TBRMM(base_config(market_panel(), max_treatment_size=5)).fit()
    for d in res.designs:
        assert len(d.treatment_units) == d.k


def test_the_three_groups_partition_the_panel():
    df = market_panel()
    res = TBRMM(base_config(df, max_treatment_size=4)).fit()
    units = set(df.geo.unique())
    for d in res.designs:
        t, c, u = set(d.treatment_units), set(d.control_units), set(d.unassigned_units)
        assert t & c == set() and t & u == set() and c & u == set()
        assert t | c | u == units


def test_no_design_has_an_empty_control_group():
    res = TBRMM(base_config(market_panel(), max_treatment_size=5)).fit()
    for d in res.designs:
        assert len(d.control_units) >= 1


def test_the_objective_recorded_is_the_objective_of_those_groups():
    """A design carries its own score, so recomputing it must agree."""
    from mlsynth.utils.tbrmm_helpers.objective import score_split
    df = market_panel()
    res = TBRMM(base_config(df, max_treatment_size=3)).fit()
    wide = df.pivot_table(index="date", columns="geo", values="Y")
    for d in res.designs:
        again = score_split(
            wide[list(d.treatment_units)].sum(axis=1).to_numpy(),
            wide[list(d.control_units)].sum(axis=1).to_numpy(),
            objective="reference", n_test=10)
        assert again.value == pytest.approx(d.objective_value, rel=1e-12)


def test_matching_never_accepts_a_worse_control_group():
    """Hill climbing: the objective along the trace is non-decreasing."""
    res = TBRMM(base_config(market_panel(), max_treatment_size=4)).fit()
    for d in res.designs:
        trace = list(d.matching_trace)
        assert trace, "a design must record the climb that produced it"
        assert all(b >= a for a, b in zip(trace, trace[1:])), trace


def test_the_climb_stops_when_no_toggle_improves():
    """The last matching step is the one that failed to improve, so the final
    value repeats or the trace has length one."""
    res = TBRMM(base_config(market_panel(), max_treatment_size=2)).fit()
    for d in res.designs:
        assert d.matching_converged is True


# --------------------------------------------------------------------------- #
# eligibility: Au's A_i
# --------------------------------------------------------------------------- #
def test_a_forced_treatment_geo_is_in_every_treatment_group():
    df = with_eligibility(market_panel(), forced_treatment=["m00", "m01"])
    res = TBRMM(elig_config(df, max_treatment_size=4)).fit()
    assert [d.k for d in res.designs] == [2, 3, 4]      # k starts at k0 = 2
    for d in res.designs:
        assert {"m00", "m01"} <= set(d.treatment_units)


def test_a_geo_ineligible_for_treatment_is_never_treated():
    df = with_eligibility(market_panel(), no_treatment=["m02", "m03", "m04"])
    res = TBRMM(elig_config(df, max_treatment_size=5)).fit()
    for d in res.designs:
        assert {"m02", "m03", "m04"}.isdisjoint(d.treatment_units)


def test_a_geo_ineligible_for_control_is_never_a_control():
    df = with_eligibility(market_panel(), no_control=["m05", "m06"])
    res = TBRMM(elig_config(df, max_treatment_size=3)).fit()
    for d in res.designs:
        assert {"m05", "m06"}.isdisjoint(d.control_units)


def test_a_geo_that_cannot_be_unassigned_stays_assigned():
    df = with_eligibility(market_panel(), no_unassigned=["m07"])
    res = TBRMM(elig_config(df, max_treatment_size=3)).fit()
    for d in res.designs:
        assert "m07" in set(d.treatment_units) | set(d.control_units)


def test_k0_forced_larger_than_one_skips_the_smaller_sizes():
    df = with_eligibility(market_panel(), forced_treatment=["m00", "m01", "m02"])
    res = TBRMM(elig_config(df, max_treatment_size=4)).fit()
    assert [d.k for d in res.designs] == [3, 4]


# --------------------------------------------------------------------------- #
# the two objectives
# --------------------------------------------------------------------------- #
def test_both_objectives_run_and_record_which_was_used():
    df = market_panel(n_units=14, seed=3)
    for obj in ("reference", "paper"):
        res = TBRMM(base_config(df, objective=obj, max_treatment_size=3)).fit()
        assert res.objective == obj
        assert all(np.isfinite(d.objective_value) for d in res.designs)


def test_the_paper_objective_reports_its_three_components():
    df = market_panel(n_units=14, seed=4)
    res = TBRMM(base_config(df, objective="paper", max_treatment_size=2)).fit()
    d = res.designs[-1]
    assert 0.0 <= d.detail["p_cusum"] <= 1.0
    assert 0.0 <= d.detail["p_bg"] <= 1.0
    assert 0.0 <= d.detail["r2"] <= 1.0
    # f is the minimum of the three
    assert d.objective_value == pytest.approx(
        min(d.detail["p_cusum"], d.detail["p_bg"], d.detail["r2"]), rel=1e-12)


def test_the_reference_objective_reports_its_gates_and_power():
    df = market_panel(n_units=14, seed=5)
    res = TBRMM(base_config(df, objective="reference", max_treatment_size=2)).fit()
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
    from mlsynth.utils.tbrmm_helpers.objective import _pretest_fit

    rng = np.random.default_rng(11)
    for _ in range(8):
        n = int(rng.integers(12, 60))
        x = rng.normal(100.0, 20.0, n)
        y = 3.0 + 0.8 * x + rng.normal(0.0, 4.0, n)
        resid, sigma, r2 = _pretest_fit(y, x)
        fit = fit_pretest(y, x)
        assert sigma == pytest.approx(np.sqrt(fit.sigma_sq), rel=1e-12)
        assert resid == pytest.approx(y - (fit.alpha + fit.beta * x), abs=1e-9)
        assert r2 == pytest.approx(float(np.corrcoef(y, x)[0, 1]) ** 2, rel=1e-10)


def test_the_objective_refuses_a_pretest_too_short_for_a_residual_scale():
    """Two pretest periods leave no degrees of freedom, so eqn 1 has no residual
    scale and every gate downstream is a division by zero. A design scored on
    such a window would come back as a number, which is the one outcome a
    malformed panel must not produce."""
    from mlsynth.utils.tbrmm_helpers.objective import _pretest_fit

    with pytest.raises(MlsynthDataError, match="degrees of freedom"):
        _pretest_fit(np.array([1.0, 2.0]), np.array([1.0, 3.0]))


def test_the_two_objectives_can_disagree_on_the_recommendation():
    """Measured on the GeoLift markets: Spearman +0.54, no top-five overlap. A
    build that could not express the disagreement would be hiding it."""
    df = market_panel(n_units=16, seed=6)
    a = TBRMM(base_config(df, objective="reference", max_treatment_size=4)).fit()
    b = TBRMM(base_config(df, objective="paper", max_treatment_size=4)).fit()
    assert set(a.recommended.treatment_units) != set(b.recommended.treatment_units)


# --------------------------------------------------------------------------- #
# invariances
# --------------------------------------------------------------------------- #
def test_relabelling_the_geos_selects_the_same_markets():
    df = market_panel(n_units=12, seed=8)
    renamed = df.copy()
    mapping = {g: f"z{i:02d}" for i, g in enumerate(sorted(df.geo.unique())[::-1])}
    renamed["geo"] = renamed.geo.map(mapping)
    a = TBRMM(base_config(df, max_treatment_size=3)).fit()
    b = TBRMM(base_config(renamed, max_treatment_size=3)).fit()
    assert {mapping[g] for g in a.recommended.treatment_units} == \
           set(b.recommended.treatment_units)


def test_scaling_the_outcome_selects_the_same_markets():
    df = market_panel(n_units=12, seed=9)
    big = df.copy(); big["Y"] = big["Y"] * 1000.0
    a = TBRMM(base_config(df, max_treatment_size=3)).fit()
    b = TBRMM(base_config(big, max_treatment_size=3)).fit()
    assert set(a.recommended.treatment_units) == set(b.recommended.treatment_units)


def test_the_search_is_deterministic():
    df = market_panel(n_units=14, seed=10)
    a = TBRMM(base_config(df, max_treatment_size=4)).fit()
    b = TBRMM(base_config(df, max_treatment_size=4)).fit()
    assert [d.treatment_units for d in a.designs] == \
           [d.treatment_units for d in b.designs]
    assert [d.objective_value for d in a.designs] == \
           [d.objective_value for d in b.designs]


# --------------------------------------------------------------------------- #
# Layer 3: edges
# --------------------------------------------------------------------------- #
def test_max_treatment_size_of_one():
    res = TBRMM(base_config(market_panel(), max_treatment_size=1)).fit()
    assert len(res.designs) == 1 and res.designs[0].k == 1


def test_the_smallest_workable_panel():
    """Three geos: one treated, one control, one free."""
    res = TBRMM(base_config(market_panel(n_units=3, T=20),
                            max_treatment_size=1)).fit()
    assert len(res.designs[0].treatment_units) == 1


def test_treating_every_eligible_geo_but_one():
    df = market_panel(n_units=6, T=25)
    res = TBRMM(base_config(df, max_treatment_size=5)).fit()
    assert res.designs[-1].k == 5
    assert len(res.designs[-1].control_units) == 1


def test_a_post_column_restricts_scoring_to_the_pre_rows():
    df = market_panel(n_units=10, T=40, seed=11)
    df["post"] = (df.date >= 30).astype(int)
    a = TBRMM(base_config(df, post_col="post", max_treatment_size=2)).fit()
    b = TBRMM(base_config(df[df.date < 30].drop(columns=["post"]),
                          max_treatment_size=2)).fit()
    assert [d.objective_value for d in a.designs] == \
           pytest.approx([d.objective_value for d in b.designs], rel=1e-12)


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
        TBRMM(elig_config(df, max_treatment_size=4)).fit()


def test_no_control_eligible_geo_raises():
    df = with_eligibility(market_panel(n_units=5),
                          no_control=["m00", "m01", "m02", "m03", "m04"])
    with pytest.raises(MlsynthDataError, match="control"):
        TBRMM(elig_config(df, max_treatment_size=2)).fit()


def test_a_geo_eligible_for_nothing_raises():
    df = with_eligibility(market_panel(n_units=5))
    df.loc[df.geo == "m02", ["can_treat", "can_control", "can_exclude"]] = 0
    with pytest.raises(MlsynthDataError, match="eligible for no"):
        TBRMM(elig_config(df, max_treatment_size=2)).fit()


def test_an_eligibility_column_varying_within_unit_raises():
    df = with_eligibility(market_panel(n_units=6))
    df.loc[(df.geo == "m01") & (df.date > 10), "can_treat"] = 0
    with pytest.raises(MlsynthDataError, match="constant"):
        TBRMM(elig_config(df, max_treatment_size=2)).fit()


def test_a_missing_eligibility_column_raises():
    with pytest.raises(MlsynthDataError, match="can_treat"):
        TBRMM(elig_config(market_panel(), max_treatment_size=2)).fit()


def test_too_few_periods_to_fit_raises():
    with pytest.raises(MlsynthDataError):
        TBRMM(base_config(market_panel(n_units=6, T=2),
                          max_treatment_size=2)).fit()


def test_a_non_config_input_raises_a_config_error():
    with pytest.raises(MlsynthConfigError, match="TBRMMConfig"):
        TBRMM(17)
