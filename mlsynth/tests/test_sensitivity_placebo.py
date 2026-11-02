"""Placebo runs over a registry of designs, and the inference built on them.

With ``placebo=True`` every design that fits the treated unit is refit once per
unit of its pool with that unit treated and the actually treated unit kept as
an ordinary donor (Abadie, Diamond and Hainmueller 2010; Ferman, Pinto and
Possebom 2020). The result's inference methods rank the treated unit among
those placebo fits.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.sensitivity import SpecificationStudy
from mlsynth.utils.sensitivity_helpers.inference import (
    donohue,
    dube_zipperer,
    joint_pvalue,
    permutation_pvalue,
    ratio_statistic,
)
from mlsynth.utils.sensitivity_helpers.structures import PlaceboRuns

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


def base(df=None):
    return dict(df=panel() if df is None else df, unitid="unit", time="time",
                outcome="y", treat="treated", backend="outcome-only",
                inference=False, display_graphs=False)


def study(specs=(), df=None, **kw):
    return SpecificationStudy(VanillaSC, base(df), specs, ATT, **kw)


@pytest.fixture(scope="module")
def two_designs():
    return study([{"name": "late", "fit_window": (4, 9)}], placebo=True).fit()


def gaps_of(result, spec, units=UNITS):
    p = result.placebo.paths
    p = p[p.spec_id == spec]
    return np.column_stack([
        p[p.unit == u].set_index("time").gap.reindex(range(T)).to_numpy(float)
        for u in units])


def test_placebo_runs_cover_every_unit_of_every_design(two_designs):
    res = two_designs
    assert isinstance(res.placebo, PlaceboRuns)
    for spec in ("baseline", "late"):
        assert set(res.placebo.paths.loc[res.placebo.paths.spec_id == spec, "unit"]) == set(UNITS)
    assert res.placebo.failures.empty
    table = res.placebo_test()
    assert list(table.index) == ["baseline", "late"]
    assert ((table.p_value > 0) & (table.p_value <= 1)).all()
    assert (table.n_units == N).all()


def test_treated_rows_are_the_main_fit(two_designs):
    res = two_designs
    for spec in ("baseline", "late"):
        main = res.paths[res.paths.spec_id == spec].reset_index(drop=True)
        rows = res.placebo.paths
        rows = rows[(rows.spec_id == spec) & (rows.unit == "u0")].reset_index(drop=True)
        np.testing.assert_allclose(rows.synthetic, main.synthetic)


def test_placebo_fit_is_a_standalone_fit_with_that_unit_treated(two_designs):
    df = panel()
    df["treated"] = ((df.unit == "u3") & (df.time >= T0)).astype(int)
    alone = VanillaSC(dict(base(df), fit_window=(4, 9))).fit()
    rows = two_designs.placebo.paths
    rows = rows[(rows.spec_id == "late") & (rows.unit == "u3")]
    np.testing.assert_allclose(rows.synthetic, alone.time_series.counterfactual_outcome)
    w = two_designs.placebo.weights
    w = w[(w.spec_id == "late") & (w.unit == "u3")]
    assert "u0" in set(w.donor)  # the treated unit is in the placebo's pool


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
    np.testing.assert_allclose(dz.loc[["baseline", "late"]],
                               dube_zipperer(G, post, 0))
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


def test_inference_without_placebo_runs_is_reported():
    res = study().fit()
    assert res.placebo is None
    for call in (res.placebo_test, res.joint_test, res.confidence_band,
                 lambda: res.choose("dube-zipperer")):
        with pytest.raises(MlsynthConfigError, match="placebo"):
            call()


def test_placebo_flag_must_be_boolean():
    with pytest.raises(MlsynthConfigError, match="placebo"):
        study(placebo="yes")


def test_joint_inference_needs_designs_with_one_unit_set():
    res = study([{"name": "pool", "donors": ["u1", "u2", "u3"]}], placebo=True).fit()
    assert res.placebo_test().loc["pool", "n_units"] == 4
    with pytest.raises(MlsynthConfigError, match="unit"):
        res.joint_test()
    assert res.joint_test(designs=["pool"])["n_units"] == 4


def test_failed_placebo_fit_is_logged_and_left_out():
    fit = VanillaSC.fit

    def failing(self):
        df, cfg = self.config.df, self.config
        if set(df.loc[df[cfg.treat] == 1, cfg.unitid]) == {"u2"}:
            raise RuntimeError("placebo solver failure")
        return fit(self)

    with patch.object(VanillaSC, "fit", failing):
        res = study(placebo=True).fit()
    assert res.placebo.failures[["spec_id", "unit"]].values.tolist() == [["baseline", "u2"]]
    assert res.placebo.failures.message.iloc[0] == "placebo solver failure"
    assert res.placebo_test().loc["baseline", "n_units"] == N - 1


@settings(max_examples=4, deadline=None)
@given(st.permutations(UNITS[1:]))
def test_placebo_pvalues_do_not_depend_on_unit_labels(order):
    labels = ["u0", *order]
    renamed = study(df=panel(labels=labels), placebo=True).fit().placebo_test()
    original = study(placebo=True).fit().placebo_test()
    assert renamed.loc["baseline", "p_value"] == pytest.approx(original.loc["baseline", "p_value"])


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
    res = SpecificationStudy(VanillaSC, base(convex_panel()),
                             [{"name": "one_period", "fit_window": (7, 7)}],
                             range(8, 12)).fit()
    ident = res.identification.set_index("spec_id")
    assert bool(ident.loc["baseline", "identified"])
    assert ident.loc["baseline", "att_max"] - ident.loc["baseline", "att_min"] < 1e-6
    assert not bool(ident.loc["one_period", "identified"])
    for spec, row in ident.iterrows():
        assert row.att_min - 1e-7 <= row.att <= row.att_max + 1e-7


def test_identification_uses_the_weighted_predictors_with_covariates():
    df = convex_panel()
    df["x"] = df.unit.str[1:].astype(float) + df.time / 20
    cfg = dict(base(df), backend="malo", covariates=["x", "y"],
               covariate_windows={"x": (0, 7), "y": (0, 5)})
    res = SpecificationStudy(VanillaSC, cfg, [], range(8, 12)).fit()
    row = res.identification.set_index("spec_id").loc["baseline"]
    assert row.att_min - 1e-7 <= row.att <= row.att_max + 1e-7
    assert isinstance(row.identified, (bool, np.bool_))
    assert row.att == pytest.approx(res.summary.set_index("spec_id").loc["baseline", "att"])


def test_identification_failure_is_reported_and_keeps_the_estimate():
    with patch("mlsynth.sensitivity.identified_effect_range",
               side_effect=RuntimeError("program failed")):
        res = study().fit()
    row = res.identification.iloc[0]
    assert np.isnan(row.att_min) and row.identified is None
    assert res.summary.status.tolist() == ["ok"]
    assert "identification not computed: program failed" in res.warnings.message.tolist()


def test_placebo_solver_warnings_are_logged_by_unit():
    from mlsynth.utils.bilevel import BilevelSCM

    fit = BilevelSCM.fit

    def warned(self, *args, **kwargs):
        import warnings

        warnings.warn("primary solver did not converge", RuntimeWarning)
        return fit(self, *args, **kwargs)

    with patch.object(BilevelSCM, "fit", warned):
        res = study(placebo=True).fit()
    logged = res.placebo.warnings
    assert set(logged.unit) == set(UNITS[1:])
    assert (logged.message == "primary solver did not converge").all()


def test_inference_reports_unknown_designs_periods_and_empty_selections(two_designs):
    with pytest.raises(MlsynthConfigError, match="No placebo runs"):
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


def test_identification_is_computed_when_donors_get_zero_weight():
    res = study().fit()
    assert len(res.fits["baseline"].weights.donor_weights) < N - 1  # some donor at zero
    row = res.identification.iloc[0]
    assert np.isfinite(row.att_min) and np.isfinite(row.att_max)
    assert row.att_min - 1e-7 <= row.att <= row.att_max + 1e-7
    assert res.warnings.empty


def test_placebo_test_uses_the_requested_windows(two_designs):
    pre, post = np.arange(T) < 6, np.isin(np.arange(T), [12, 13])
    table = two_designs.placebo_test(pre_periods=range(6), post_periods=[12, 13])
    stat = ratio_statistic(gaps_of(two_designs, "baseline"), pre, post)
    assert table.loc["baseline", "statistic"] == pytest.approx(stat[0])
    assert table.loc["baseline", "p_value"] == pytest.approx(permutation_pvalue(stat, 0))


def test_identification_weights_the_predictors_by_v():
    from mlsynth.utils.sensitivity_helpers.inference import identified_effect_range
    from mlsynth.utils.vanillasc_helpers.pipeline import _predictors

    rng = np.random.default_rng(4)
    df = panel()
    df["x"] = df.unit.map({u: rng.normal() for u in UNITS}) + rng.normal(0, 0.1, len(df))
    df["z"] = df.unit.map({u: rng.normal() for u in UNITS})
    cfg = dict(base(df), backend="malo", covariates=["x", "z", "y"],
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
