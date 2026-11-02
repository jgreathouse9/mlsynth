"""The registry wraps existing fits without changing the estimand or sample."""

import warnings
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.sensitivity import SensitivityResult, SpecificationStudy


def config():
    rng = np.random.default_rng(10)
    donors = rng.normal(10, 2, (12, 3))
    treated = donors @ np.array([0.2, 0.3, 0.5])
    treated[8:] += np.arange(1, 5)
    y = np.column_stack([treated, donors])
    panel = pd.DataFrame(
        [
            (f"u{u}", t, y[t, u], int(u == 0 and t >= 8), float(u) + t / 20)
            for u in range(4)
            for t in range(12)
        ],
        columns=["unit", "time", "y", "treated", "x"],
    )
    return dict(
        df=panel,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated",
        backend="outcome-only",
        inference=False,
        display_graphs=False,
    )


def study(specs=(), base=None, **kw):
    return SpecificationStudy(
        estimator=VanillaSC,
        base_config=config() if base is None else base,
        specifications=specs,
        att_periods=[8, 9, 10, 11],
        **kw,
    )


@given(st.sampled_from(["2", b"2", ["missing"], [2]]))
def test_exclusions_require_a_collection_of_known_unit_ids(excluded):
    cfg = config()
    cfg["df"]["unit"] = cfg["df"].unit.str[-1]
    with pytest.raises(MlsynthConfigError, match="excluded_donors"):
        study(base=cfg, excluded_donors=excluded)


@given(st.booleans(), st.sampled_from([False, True, "sensitivity.png"]))
@settings(max_examples=6, deadline=None)
def test_batch_fits_never_display_or_save_graphs(display, save):
    cfg = config()
    cfg.update(display_graphs=display, save=save)
    with (
        patch("matplotlib.pyplot.show") as show,
        patch("matplotlib.figure.Figure.savefig") as savefig,
    ):
        result = study(base=cfg).fit()
    assert result.failures.empty
    show.assert_not_called()
    savefig.assert_not_called()
    assert result.registry["baseline"]["display_graphs"] is False
    assert result.registry["baseline"]["save"] is False
    assert result.baseline["display_graphs"] is display
    assert result.baseline["save"] == save


@given(st.booleans())
def test_source_checkout_without_distribution_metadata_retains_results(missing):
    from importlib.metadata import PackageNotFoundError

    with patch("mlsynth.sensitivity.version") as version:
        if missing:
            version.side_effect = PackageNotFoundError("mlsynth")
        else:
            version.return_value = "test-version"
        result = study().fit()
    assert result.failures.empty
    assert np.isfinite(result.summary.att.iloc[0])
    assert result.metadata["version"] == (None if missing else "test-version")


@given(st.one_of(st.none(), st.integers(0, 3)), st.one_of(st.none(), st.integers(4, 7)))
@settings(max_examples=8, deadline=None)
def test_open_fitting_window_endpoints_match_native_estimator(start, end):
    cfg = config()
    window = (start, end)
    native = VanillaSC({**cfg, "fit_window": window}).fit()
    result = study([{"name": "window", "fit_window": window}], base=cfg).fit()
    assert result.failures.empty
    np.testing.assert_allclose(
        result.paths.loc[result.paths.spec_id == "window", "synthetic"],
        native.time_series.counterfactual_outcome,
    )


@given(st.sampled_from(["unused", "u1"]), st.integers(12, 100))
@settings(max_examples=6, deadline=None)
def test_unselected_units_do_not_expand_the_common_trajectory(unit, time):
    cfg = config()
    original = study(base=cfg).fit()
    extra = pd.DataFrame([dict(unit=unit, time=time, y=100, treated=0, x=1)])
    cfg["df"] = pd.concat([cfg["df"], extra], ignore_index=True)
    cfg["donors"] = ["u1", "u2", "u3"]
    result = study(base=cfg).fit()
    pd.testing.assert_frame_equal(result.paths, original.paths)


def test_baseline_matches_standalone_and_weighted_outcomes():
    cfg = config()
    standalone = VanillaSC(cfg).fit()
    result = study(base=cfg).fit()
    assert isinstance(result, SensitivityResult)
    assert result.summary.att.iloc[0] == pytest.approx(2.5, abs=1e-6)
    assert result.summary.att.iloc[0] == pytest.approx(standalone.effects.att)
    np.testing.assert_allclose(
        result.paths.synthetic, standalone.time_series.counterfactual_outcome
    )
    w = result.weights.set_index("donor").weight
    wide = cfg["df"].pivot(index="time", columns="unit", values="y")
    np.testing.assert_allclose(result.paths.synthetic, wide[w.index] @ w)
    assert result.metadata["treatment_time"] == 8
    assert result.registry["baseline"]["backend"] == "outcome-only"
    assert (
        result.fits["baseline"].weights.donor_weights
        == standalone.weights.donor_weights
    )


@settings(max_examples=6, deadline=None)
@given(st.permutations(["u1", "u2", "u3"]))
def test_donor_order_invariance(donors):
    result = study([{"name": "permuted", "donors": donors}]).fit()
    assert result.summary.att.iloc[0] == pytest.approx(
        result.summary.att.iloc[1], abs=1e-10
    )


@settings(max_examples=6, deadline=None)
@given(st.floats(min_value=-100, max_value=100, allow_nan=False))
def test_post_outcomes_never_change_fitted_weights(delta):
    cfg = config()
    original = study(base=cfg).fit()
    cfg["df"].loc[(cfg["df"].unit == "u0") & (cfg["df"].time >= 8), "y"] += delta
    modified = study(base=cfg).fit()
    np.testing.assert_array_equal(original.weights.weight, modified.weights.weight)
    assert modified.summary.att.iloc[0] == pytest.approx(
        original.summary.att.iloc[0] + delta
    )


@pytest.mark.parametrize(
    "override",
    [
        {"outcome": "x"},
        {"backend": "mscmt"},
        {"seed": 12},
        {"df": pd.DataFrame()},
        {"treat": "other"},
        {"fit_window": (2, 8)},
        {"fit_window": (1,)},
        {"fit_window": (1, 7, 8)},
        {"fit_exclude": [8]},
        {"covariate_windows": {"x": (2, 8)}},
        {"donors": ["u0", "u1"]},
        {"donors": []},
        {"donors": ["u1", "u1"]},
        {"donors": ["absent"]},
        {"fit_window": (20, 30)},
        {"fit_exclude": list(range(8))},
        {"covariates": ["x"]},
    ],
)
def test_invalid_design_logged_without_stopping(override):
    result = study(
        [{"name": "invalid", **override}, {"name": "late", "fit_window": (4, 7)}]
    ).fit()
    assert result.summary.status.tolist() == ["ok", "failed", "ok"]
    assert result.failures.spec_id.tolist() == ["invalid"]
    assert result.failures.exception_type.iloc[0] in {
        "MlsynthConfigError",
        "MlsynthDataError",
    }
    assert np.isnan(result.summary.att.iloc[1])
    assert len(result.paths) == 24


def test_missing_outcome_and_contamination_preserve_failure():
    cfg = config()
    cfg["df"].loc[(cfg["df"].unit == "u3") & (cfg["df"].time == 9), "y"] = np.nan
    result = study([{"name": "complete", "donors": ["u1", "u2"]}], base=cfg).fit()
    assert result.summary.status.tolist() == ["failed", "ok"]
    result = study(
        [{"name": "eligible", "donors": ["u1", "u2"]}], excluded_donors=["u3"]
    ).fit()
    assert result.summary.status.tolist() == ["failed", "ok"]


@pytest.mark.parametrize("periods", [[], [7, 8], [8, 8], [20]])
def test_invalid_common_att_periods_raise(periods):
    with pytest.raises(MlsynthConfigError):
        SpecificationStudy(
            estimator=VanillaSC,
            base_config=config(),
            specifications=[],
            att_periods=periods,
        )


@pytest.mark.parametrize(
    "change", ["none", "multiple", "reversal", "nonbinary", "missing", "duplicate"]
)
def test_ambiguous_treatment_or_panel_rejected(change):
    cfg = config()
    frame = cfg["df"]
    if change == "none":
        frame["treated"] = 0
    elif change == "multiple":
        frame.loc[(frame.unit == "u1") & (frame.time >= 9), "treated"] = 1
    elif change == "reversal":
        frame.loc[(frame.unit == "u0") & (frame.time == 10), "treated"] = 0
    elif change == "nonbinary":
        frame.loc[0, "treated"] = 2
    elif change == "missing":
        frame.loc[0, "time"] = np.nan
    else:
        cfg["df"] = pd.concat([frame, frame.iloc[:1]])
    with pytest.raises((MlsynthConfigError, MlsynthDataError)):
        study(base=cfg)


def test_registry_snapshot_and_fixed_auto_backend():
    cfg = config()
    cfg["backend"] = "auto"
    specs = [{"name": "short", "fit_window": (4, 7)}]
    runner = study(specs, cfg)
    cfg["df"].loc[:, "y"] = 0
    specs[0]["fit_window"] = (9, 10)
    result = runner.fit()
    assert result.summary.status.tolist() == ["ok", "ok"]
    assert result.metadata["backend"] == "outcome-only"
    assert result.baseline["backend"] == "auto"
    assert result.registry["short"]["fit_window"] == (4, 7)


def test_warnings_are_retained_with_estimates():
    fit = VanillaSC.fit

    def warned(self):
        warnings.warn("solver did not converge", RuntimeWarning)
        return fit(self)

    with patch.object(VanillaSC, "fit", warned):
        result = study().fit()
    assert result.summary.status.tolist() == ["warning"]
    assert result.warnings.message.tolist() == ["solver did not converge"]
    assert result.dispersion()["with_warnings"] == 1
    assert np.isfinite(result.summary.att.iloc[0])


def test_all_failures_plot_and_counts():
    result = study(excluded_donors=["u1", "u2", "u3"]).fit()
    assert result.dispersion()["failed"] == 1
    assert np.isnan(result.dispersion()["positive_share"])
    fig = result.plot()
    assert len(fig.axes[0].collections) == 0
    assert [t.get_text() for t in fig.axes[0].texts] == ["failed"]
    plt.close(fig)


def test_common_pre_rmse_and_subset_att():
    runner = SpecificationStudy(
        estimator=VanillaSC,
        base_config=config(),
        specifications=[{"name": "late", "fit_window": (4, 7)}],
        att_periods=[9, 11],
    )
    result = runner.fit()
    for row in result.summary.itertuples():
        paths = result.paths[result.paths.spec_id == row.spec_id].set_index("time")
        assert row.att == pytest.approx(paths.loc[[9, 11], "gap"].mean())
        assert row.pre_rmse == pytest.approx(
            np.sqrt(np.mean(paths.loc[range(8), "gap"] ** 2))
        )


def test_dimension_summary_retains_failures():
    result = study(
        [{"name": "late", "fit_window": (4, 7)}, {"name": "bad", "fit_window": (4, 10)}]
    ).fit()
    table = result.by_dimension("fit_window")
    assert table.attempted.sum() == 3
    assert table.failed.sum() == 1
    with pytest.raises(MlsynthConfigError):
        result.by_dimension("backend")


def test_covariate_matching_and_plot():
    cfg = config()
    cfg.update(
        backend="malo",
        covariates=["x", "y"],
        covariate_windows={"x": (0, 7), "y": (0, 5)},
    )
    result = study(
        [{"name": "late", "covariate_windows": {"x": (3, 7), "y": (3, 7)}}], cfg
    ).fit()
    assert result.failures.empty
    assert len(result.fits) == 2
    fig = result.plot()
    assert len(fig.axes[0].collections) == 2
    plt.close(fig)


def test_unsupported_estimator_and_variant_raise():
    with pytest.raises(MlsynthConfigError, match="Only VanillaSC"):
        SpecificationStudy(object, config(), [], [8])
    cfg = config()
    cfg["augment"] = "ridge"
    with pytest.raises(MlsynthConfigError, match="classical"):
        study(base=cfg)


def test_duplicate_registry_and_changed_baseline():
    with pytest.raises(MlsynthConfigError, match="names"):
        study([{"name": "a"}, {"name": "a"}])
    result = study([{"name": "baseline", "fit_window": (4, 7)}]).fit()
    assert result.summary.status.tolist() == ["failed"]


@pytest.mark.parametrize("change", ["times", "nonfinite", "overflow", "exception"])
def test_invalid_estimator_output_is_logged(change):
    result = VanillaSC(config()).fit()
    if change == "times":
        series = result.time_series.model_copy(
            update={"time_periods": list(range(1, 13))}
        )
        result = result.model_copy(update={"time_series": series})
    elif change == "nonfinite":
        series = result.time_series.model_copy(
            update={"counterfactual_outcome": np.full(12, np.nan)}
        )
        result = result.model_copy(update={"time_series": series})
    elif change == "overflow":
        series = result.time_series.model_copy(
            update={
                "observed_outcome": np.full(12, 1e308),
                "counterfactual_outcome": np.full(12, -1e308),
            }
        )
        result = result.model_copy(update={"time_series": series})
    if change == "exception":
        with patch.object(VanillaSC, "fit", side_effect=RuntimeError("solver failure")):
            actual = study().fit()
    else:
        with patch.object(VanillaSC, "fit", return_value=result):
            actual = study().fit()
    assert actual.summary.status.tolist() == ["failed"]
    assert len(actual.failures) == 1
    assert actual.paths.empty


@settings(max_examples=8, deadline=None)
@given(st.lists(st.integers(min_value=8, max_value=11), min_size=1, unique=True))
def test_common_att_periods_average_exactly_the_requested_gaps(periods):
    runner = SpecificationStudy(
        VanillaSC, config(), [{"name": "late", "fit_window": (4, 7)}], periods
    )
    result = runner.fit()
    for row in result.summary.itertuples():
        paths = result.paths[result.paths.spec_id == row.spec_id].set_index("time")
        assert row.att == pytest.approx(paths.loc[periods, "gap"].mean())


def test_missing_treatment_indicator_is_rejected():
    cfg = config()
    cfg["df"]["treated"] = cfg["df"]["treated"].astype(float)
    cfg["df"].loc[0, "treated"] = np.nan
    with pytest.raises(MlsynthDataError, match="must not be missing"):
        study(base=cfg)


@settings(max_examples=10, deadline=None)
@given(st.integers(min_value=8, max_value=100))
def test_every_post_treatment_fitting_window_is_a_failed_attempt(end):
    result = study([{"name": "invalid", "fit_window": (0, end)}]).fit()
    assert result.summary.status.tolist() == ["ok", "failed"]
    assert result.failures.spec_id.tolist() == ["invalid"]


@settings(max_examples=6, deadline=None)
@given(st.lists(st.sampled_from(["u1", "u2", "u3"]), min_size=1, unique=True))
def test_every_declared_contaminated_pool_fails(excluded):
    result = study(excluded_donors=excluded).fit()
    assert result.summary.status.tolist() == ["failed"]
    assert result.failures.spec_id.tolist() == ["baseline"]


def test_warnings_suppressed_inside_native_pipeline_are_preserved():
    from mlsynth.utils.bilevel import BilevelSCM

    fit = BilevelSCM.fit

    def warned(self, *args, **kwargs):
        warnings.warn("primary solver did not converge", RuntimeWarning)
        return fit(self, *args, **kwargs)

    with patch.object(BilevelSCM, "fit", warned):
        result = study().fit()
    assert result.summary.status.tolist() == ["warning"]
    assert result.warnings.message.tolist() == ["primary solver did not converge"]
    assert result.warnings.warning_type.tolist() == ["RuntimeWarning"]
    assert result.fits["baseline"].additional_outputs["solver_warnings"] == [
        {"warning_type": "RuntimeWarning", "message": "primary solver did not converge"}
    ]


@pytest.mark.parametrize(
    "gaps, expected",
    [
        ([1e308] * 4, 1e308),
        ([1e308, 1e308, -1e308, -1e308], 0.0),
        ([1e308, -1e308, 1e-200, 1e-200], 5e-201),
        ([0.0] * 4, 0.0),
        ([1e-200] * 4, 1e-200),
    ],
)
def test_extreme_post_gaps_keep_finite_att(gaps, expected):
    cfg = config()
    frame = cfg["df"]
    frame.loc[frame.time >= 8, "y"] = 0.0
    frame.loc[(frame.unit == "u0") & (frame.time >= 8), "y"] = gaps
    result = study(base=cfg).fit()
    assert result.failures.empty
    assert result.summary.att.iloc[0] == expected
    assert np.isfinite(result.summary[["att", "pre_rmse", "fit_rmse"]]).all().all()
    assert result.dispersion()["estimated"] == 1


def test_large_pre_gap_outside_fit_window():
    cfg = config()
    cfg["fit_window"] = (4, 7)
    frame = cfg["df"]
    frame.loc[(frame.unit == "u0") & (frame.time == 0), "y"] = 1e200
    result = study(base=cfg).fit()
    assert result.failures.empty
    assert result.summary.pre_rmse.iloc[0] / 1e200 == pytest.approx(1 / np.sqrt(8))
    assert np.isfinite(result.summary.fit_rmse.iloc[0])


def test_rms_scales_and_summary_guard():
    from mlsynth.utils.sensitivity_helpers.numerics import rms

    for scale in [0.0, np.nextafter(0.0, 1.0), 1e-200, 1e200, 1e308]:
        assert rms([scale, -scale]) == scale
    assert rms([3e200, 4e200]) / 1e200 == pytest.approx(np.sqrt(12.5))
    with patch("mlsynth.sensitivity.rms", return_value=np.inf):
        result = study().fit()
    assert result.summary.status.tolist() == ["failed"]
    assert result.summary[["att", "pre_rmse", "fit_rmse"]].isna().all().all()
    assert result.paths.empty
    assert "nonfinite summary" in result.failures.message.iloc[0]


def test_dispersion_uses_finite_estimates_and_stable_statistics():
    result = study().fit()
    summary = pd.DataFrame(
        {
            "spec_id": ["baseline"] * 5,
            "att": [1e308, 1e308, np.inf, -np.inf, np.nan],
            "status": ["ok"] * 5,
        }
    )
    result = result.model_copy(update={"summary": summary})
    stats = result.dispersion()
    assert stats["estimated"] == 2
    assert stats["positive_share"] == 1
    assert stats["median"] == 1e308
    assert stats["std"] == 0
    group = result.by_dimension("donors").iloc[0]
    assert group.estimated == 2
    assert group["median"] == 1e308
    summary.loc[:, "att"] = [1e200, -1e200, np.nan, np.nan, np.nan]
    assert result.dispersion()["std"] / 1e200 == pytest.approx(np.sqrt(2))
    summary.loc[:, "att"] = [1.7e308, -1.7e308, np.nan, np.nan, np.nan]
    assert np.isnan(result.dispersion()["std"])
