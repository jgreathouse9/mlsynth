"""Inference across a registry of designs, read from each design's own placebo.

Run with ``inference="placebo"``, every design's native VanillaSC fit carries
its in-space placebo: each unit's gaps, donor weights and ratio
(``inference.details``). ``placebo_includes_treated=True`` keeps the treated
unit in the placebo units' pools, as Abadie, Diamond and Hainmueller (2010)
and Ferman, Pinto and Possebom (2020) do. The result's inference methods rank
the treated unit among those placebo fits and combine the designs.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.sensitivity import SpecificationStudy
from mlsynth.utils.sensitivity_helpers.inference import (
    donohue,
    dube_zipperer,
    joint_pvalue,
    permutation_pvalue,
    ratio_statistic,
    synthetic_gaps,
)
from mlsynth.utils.sensitivity_helpers.placebo import stack

T, N, T0 = 14, 7, 10
ATT = list(range(T0, T))
UNITS = [f"u{u}" for u in range(N)]


def panel(effect=3.0, seed=7, labels=UNITS):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, (T, 2)).cumsum(axis=0)
    y = 20 + f @ rng.uniform(0.5, 1.5, (N, 2)).T + rng.normal(0, 0.3, (T, N))
    y[T0:, 0] += effect
    rows = [(labels[u], t, y[t, u], int(u == 0 and t >= T0))
            for u in range(N) for t in range(T)]
    return pd.DataFrame(rows, columns=["unit", "time", "y", "treated"])


def base(df=None, inference="placebo", **kw):
    extra = {"placebo_includes_treated": True} if inference == "placebo" else {}
    return dict(df=panel() if df is None else df, unitid="unit", time="time",
                outcome="y", treat="treated", backend="outcome-only",
                inference=inference, display_graphs=False, **extra, **kw)


def study(specs=(), df=None, **kw):
    return SpecificationStudy(VanillaSC, base(df, **kw), specs, ATT)


@pytest.fixture(scope="module")
def two_designs():
    return study([{"name": "late", "fit_window": (4, 9)}]).fit()


def details(result, spec):
    return result.fits[spec].inference.details


def gaps_of(result, spec, units=UNITS):
    """``(T, N)`` gaps from the native details; missing where a refit was skipped."""
    gaps = details(result, spec)["unit_gaps"]
    return np.column_stack([np.asarray(gaps.get(u, [np.nan] * T), float) for u in units])


def skipping(unit, periods):
    """Patches under which ``unit``'s placebo refit on ``periods`` raises.

    Without the batched leave-one-out solve every placebo is refit one by one.
    """
    from mlsynth.utils.bilevel import BilevelSCM

    target = panel().query("unit == @unit").sort_values("time").y.to_numpy()[list(periods)]
    real = BilevelSCM.fit

    def failing(engine, y_pre, Y0_pre, *a, **kw):
        if np.array_equal(np.asarray(y_pre, float), target):
            raise MlsynthEstimationError("placebo solver failure")
        return real(engine, y_pre, Y0_pre, *a, **kw)

    return (patch("mlsynth.utils.solvers.minnorm.solve_simplex_loo_exact",
                  side_effect=RuntimeError("no batch")),
            patch.object(BilevelSCM, "fit", failing))


def test_every_design_carries_every_units_placebo(two_designs):
    for spec in ("baseline", "late"):
        assert set(details(two_designs, spec)["unit_gaps"]) == set(UNITS)
    table = two_designs.placebo_test()
    assert list(table.index) == ["baseline", "late"]
    assert ((table.p_value > 0) & (table.p_value <= 1)).all()
    assert (table.n_units == N).all()


def test_the_treated_units_placebo_entry_is_its_main_fit(two_designs):
    for spec in ("baseline", "late"):
        main = two_designs.paths[two_designs.paths.spec_id == spec].gap.to_numpy()
        np.testing.assert_allclose(details(two_designs, spec)["unit_gaps"]["u0"], main)


@pytest.mark.parametrize("spec, window", [("late", (4, 9)), ("baseline", None)])
def test_a_placebo_is_a_standalone_fit_with_that_unit_treated(two_designs, spec, window):
    df = panel()
    df["treated"] = ((df.unit == "u3") & (df.time >= T0)).astype(int)
    cfg = base(df, inference=False)
    if window:
        cfg["fit_window"] = window
    alone = VanillaSC(cfg).fit()
    counterfactual = np.asarray(alone.time_series.counterfactual_outcome, float).ravel()
    observed = np.asarray(alone.time_series.observed_outcome, float).ravel()
    np.testing.assert_allclose(details(two_designs, spec)["unit_gaps"]["u3"],
                               observed - counterfactual, atol=1e-6)
    assert "u0" in details(two_designs, spec)["unit_weights"]["u3"]


def test_placebo_test_ranks_the_ratio_statistic(two_designs):
    pre, post = np.arange(T) < T0, np.arange(T) >= T0
    table = two_designs.placebo_test()
    for spec in ("baseline", "late"):
        stat = ratio_statistic(gaps_of(two_designs, spec), pre, post)
        assert table.loc[spec, "statistic"] == pytest.approx(stat[0])
        assert table.loc[spec, "p_value"] == pytest.approx(permutation_pvalue(stat, 0))
    rooted = two_designs.placebo_test(root=True)
    np.testing.assert_allclose(rooted.statistic, np.sqrt(table.statistic))
    np.testing.assert_allclose(rooted.p_value, table.p_value)


def test_placebo_test_uses_the_requested_windows(two_designs):
    pre, post = np.arange(T) < 6, np.isin(np.arange(T), [12, 13])
    table = two_designs.placebo_test(pre_periods=range(6), post_periods=[12, 13])
    stat = ratio_statistic(gaps_of(two_designs, "baseline"), pre, post)
    assert table.loc["baseline", "statistic"] == pytest.approx(stat[0])
    assert table.loc["baseline", "p_value"] == pytest.approx(permutation_pvalue(stat, 0))


def test_joint_test_combines_the_designs(two_designs):
    pre, post = np.arange(T) < T0, np.arange(T) >= T0
    stats = np.column_stack([ratio_statistic(gaps_of(two_designs, s), pre, post)
                             for s in ("baseline", "late")])
    out = two_designs.joint_test()
    assert out["p_value"] == pytest.approx(joint_pvalue(stats, 0))
    assert out["n_units"] == N and out["n_designs"] == 2
    one = two_designs.joint_test(designs=["late"])
    assert one["p_value"] == pytest.approx(two_designs.placebo_test().loc["late", "p_value"])


def test_choice_rules(two_designs):
    G = np.stack([gaps_of(two_designs, s) for s in ("baseline", "late")])
    post = np.arange(T) >= T0
    dz = two_designs.choose("dube-zipperer")
    np.testing.assert_allclose(dz.loc[["baseline", "late"]], dube_zipperer(G, post, 0))
    assert list(dz) == sorted(dz)
    window = [7, 8, 9]
    daw = two_designs.choose("donohue", window=window)
    np.testing.assert_allclose(daw.loc[["baseline", "late"]],
                               donohue(G, np.isin(np.arange(T), window), 0))
    with pytest.raises(MlsynthConfigError, match="window"):
        two_designs.choose("donohue")
    with pytest.raises(MlsynthConfigError, match="rule"):
        two_designs.choose("best")


def test_confidence_band_brackets_the_average_effect(two_designs):
    band = two_designs.confidence_band(significance=2 / N, precision=8)
    assert list(band.index) == list(range(T))
    average = np.mean([gaps_of(two_designs, s)[:, 0] for s in ("baseline", "late")], axis=0)
    np.testing.assert_allclose(band.average, average)
    assert band.loc[: T0 - 1, ["lower", "upper"]].isna().all().all()
    post = band.loc[T0:]
    assert (post.lower <= post.upper).all()
    np.testing.assert_allclose(post.upper - post.average, (post.upper - post.average).iloc[0])


@pytest.mark.parametrize("inference", [False, "lto"])
def test_inference_without_the_in_space_placebo_is_reported(inference):
    res = study(inference=inference).fit()
    for call in (res.placebo_test, res.joint_test, res.confidence_band,
                 lambda: res.choose("dube-zipperer")):
        with pytest.raises(MlsynthConfigError, match="inference='placebo'"):
            call()


def test_joint_inference_needs_designs_with_one_unit_set():
    res = study([{"name": "pool", "donors": ["u1", "u2", "u3"]}]).fit()
    assert res.placebo_test().loc["pool", "n_units"] == 4
    with pytest.raises(MlsynthConfigError, match="unit set"):
        res.joint_test()
    assert res.joint_test(designs=["pool"])["n_units"] == 4


def test_a_placebo_that_could_not_be_refit_is_left_out():
    batch, refit = skipping("u2", range(T0))
    with batch, refit:
        res = study().fit()
    assert "u2" in details(res, "baseline")["skipped_placebos"]
    assert res.placebo_test().loc["baseline", "n_units"] == N - 1
    assert res.summary.status.tolist() == ["warning"]


def test_a_unit_skipped_in_one_design_still_counts_in_the_joint_test():
    batch, refit = skipping("u2", range(T0))  # the baseline's fit periods, not late's
    with batch, refit:
        res = study([{"name": "late", "fit_window": (4, 9)}]).fit()
    assert "u2" in details(res, "baseline")["skipped_placebos"]
    assert "u2" in details(res, "late")["unit_gaps"]
    pre, post = np.arange(T) < T0, np.arange(T) >= T0
    stats = np.column_stack([ratio_statistic(gaps_of(res, s), pre, post)
                             for s in ("baseline", "late")])
    out = res.joint_test()
    assert out["n_units"] == N
    assert out["p_value"] == pytest.approx(joint_pvalue(stats, 0))


def test_the_stacked_weights_rebuild_the_stacked_gaps(two_designs):
    s = stack(two_designs.fits, panel(), "unit", "time", "y", "u0", ["baseline", "late"])
    assert s.units == UNITS and s.times == list(range(T))
    for d in range(2):
        np.testing.assert_allclose(synthetic_gaps(s.outcomes, s.weights[d]), s.gaps[d],
                                   atol=1e-8)


def test_numeric_unit_ids_are_matched_to_the_placebo_labels(two_designs):
    res = study(df=panel(labels=list(range(N)))).fit()
    expected = two_designs.placebo_test().loc["baseline"]
    assert res.placebo_test().loc["baseline", "p_value"] == pytest.approx(expected.p_value)
    band = res.confidence_band(significance=2 / N, precision=8)
    assert np.isfinite(band.loc[T0:, ["lower", "upper"]].to_numpy()).all()


def test_a_design_too_small_for_a_placebo_is_left_out_and_reported():
    res = study([{"name": "one_donor", "donors": ["u1"]}]).fit()
    assert list(res.placebo_test().index) == ["baseline"]
    with pytest.raises(MlsynthConfigError, match="No placebo runs"):
        res.joint_test(designs=["one_donor"])
    status = res.summary.set_index("spec_id").status
    assert status["one_donor"] == "warning"
    logged = res.warnings[res.warnings.spec_id == "one_donor"].message
    assert logged.str.contains("needs >=2 donors").any()


@settings(max_examples=4, deadline=None)
@example(order=list(reversed(UNITS)))  # the treated unit's label sorts last
@given(st.permutations(UNITS))
def test_placebo_tests_do_not_depend_on_unit_labels(two_designs, order):
    renamed = study(df=panel(labels=order)).fit().placebo_test().loc["baseline"]
    original = two_designs.placebo_test().loc["baseline"]
    assert renamed.p_value == pytest.approx(original.p_value)
    assert renamed.statistic == pytest.approx(original.statistic, rel=1e-6)


def test_inference_reports_unknown_designs_periods_and_empty_selections(two_designs):
    with pytest.raises(MlsynthConfigError, match="No placebo"):
        two_designs.joint_test(designs=["nope"])
    with pytest.raises(MlsynthConfigError, match="No designs"):
        two_designs.joint_test(designs=[])
    with pytest.raises(MlsynthConfigError, match="time axis"):
        two_designs.placebo_test(pre_periods=[99])


def test_undefined_treated_statistic_is_reported_as_missing(two_designs):
    real = ratio_statistic

    def undefined_for_treated(*args, **kwargs):
        out = real(*args, **kwargs)
        out[0] = np.nan
        return out

    with patch("mlsynth.utils.sensitivity_helpers.structures.ratio_statistic",
               undefined_for_treated):
        table = two_designs.placebo_test()
    assert table.p_value.isna().all() and table.statistic.isna().all()
    assert (table.n_units == N - 1).all()


# --- identification ---------------------------------------------------------


def convex_panel():
    rng = np.random.default_rng(10)
    donors = rng.normal(10, 2, (12, 3))
    treated = donors @ np.array([0.2, 0.3, 0.5])
    treated[8:] += 2.0
    y = np.column_stack([treated, donors])
    return pd.DataFrame([(f"u{u}", t, y[t, u], int(u == 0 and t >= 8))
                         for u in range(4) for t in range(12)],
                        columns=["unit", "time", "y", "treated"])


def test_identification_marks_designs_that_leave_the_effect_open():
    res = SpecificationStudy(VanillaSC, base(convex_panel(), inference=False),
                             [{"name": "one_period", "fit_window": (7, 7)}],
                             range(8, 12)).fit()
    ident = res.identification.set_index("spec_id")
    assert bool(ident.loc["baseline", "identified"])
    assert ident.loc["baseline", "att_max"] - ident.loc["baseline", "att_min"] < 1e-6
    assert not bool(ident.loc["one_period", "identified"])
    for _, row in ident.iterrows():
        assert row.att_min - 1e-7 <= row.att <= row.att_max + 1e-7


def test_identification_is_computed_when_donors_get_zero_weight():
    res = study(inference=False).fit()
    assert len(res.fits["baseline"].weights.donor_weights) < N - 1  # some donor at zero
    row = res.identification.iloc[0]
    assert np.isfinite(row.att_min) and np.isfinite(row.att_max)
    assert row.att_min - 1e-7 <= row.att <= row.att_max + 1e-7
    assert res.warnings.empty


def test_identification_uses_the_weighted_predictors_with_covariates():
    df = convex_panel()
    df["x"] = df.unit.str[1:].astype(float) + df.time / 20
    cfg = dict(base(df, inference=False), backend="malo", covariates=["x", "y"],
               covariate_windows={"x": (0, 7), "y": (0, 5)})
    res = SpecificationStudy(VanillaSC, cfg, [], range(8, 12)).fit()
    row = res.identification.set_index("spec_id").loc["baseline"]
    assert row.att_min - 1e-7 <= row.att <= row.att_max + 1e-7
    assert isinstance(row.identified, (bool, np.bool_))
    assert row.att == pytest.approx(res.summary.set_index("spec_id").loc["baseline", "att"])


def test_identification_weights_the_predictors_by_v():
    from mlsynth.utils.sensitivity_helpers.inference import identified_effect_range
    from mlsynth.utils.vanillasc_helpers.pipeline import _predictors

    rng = np.random.default_rng(4)
    df = panel()
    df["x"] = df.unit.map({u: rng.normal() for u in UNITS}) + rng.normal(0, 0.1, len(df))
    df["z"] = df.unit.map({u: rng.normal() for u in UNITS})
    cfg = dict(base(df, inference=False), backend="malo", covariates=["x", "z", "y"],
               covariate_windows={"x": (0, 9), "z": (0, 9), "y": (6, 9)})
    res = SpecificationStudy(VanillaSC, cfg, [], ATT).fit()
    fitted = res.fits["baseline"]
    donors = list(fitted.additional_outputs["donor_names"])
    v = fitted.weights.summary_stats["predictor_weights"]
    _, Xs = _predictors(VanillaSC(dict(cfg, df=df)).config, ["u0", *donors], list(range(T0)))
    A = np.sqrt([v[c] for c in ("x", "z", "y")])[:, None] * Xs[:, 1:]
    w = np.array([fitted.weights.donor_weights.get(d, 0.0) for d in donors])
    wide = df.pivot(index="time", columns="unit", values="y")
    lo, hi = identified_effect_range(A, w, wide.loc[ATT, "u0"], wide.loc[ATT, donors])
    row = res.identification.iloc[0]
    assert row.att_min == pytest.approx(lo) and row.att_max == pytest.approx(hi)


def test_identification_failure_is_reported_and_keeps_the_estimate():
    with patch("mlsynth.sensitivity.identified_effect_range",
               side_effect=RuntimeError("program failed")):
        res = study(inference=False).fit()
    row = res.identification.iloc[0]
    assert np.isnan(row.att_min) and row.identified is None
    assert res.summary.status.tolist() == ["ok"]
    assert "identification not computed: program failed" in res.warnings.message.tolist()
