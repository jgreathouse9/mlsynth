import pytest
import numpy as np
import pandas as pd

from mlsynth.utils.geolift_helpers.marketselect.helpers.windows import (
    lookback_pre_periods,
    lookback_treatment_window,
)
from mlsynth.utils.geolift_helpers.marketselect.helpers.shaping import (
    aggregate_treated,
    donor_matrix,
)
from mlsynth.utils.geolift_helpers.marketselect.helpers.diagnostics import (
    scaled_l2_imbalance,
)
from mlsynth.utils.geolift_helpers.marketselect.helpers.fit import (
    fit_intercept,
    counterfactual,
    fit_augsynth_once,
)
from mlsynth.utils.datautils import geoex_dataprep
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError


def _panel():
    """A small balanced wide panel (6 periods, 4 units)."""
    t = np.arange(1, 7, dtype=float)
    Ywide = pd.DataFrame(
        {"A": t, "B": 2 * t, "C": t + 5, "D": -t},
        index=pd.Index(range(1, 7), name="time"),
    )
    Ywide.columns.name = "unit"
    return Ywide


# === lookback_pre_periods ===

def test_pre_periods_smoke():
    assert lookback_pre_periods(100, 4, 1) == 96


def test_pre_periods_matches_geolift_formula():
    """Faithful to GeoLift's max_time - tp - sim + 2 (in 0-indexed counts)."""
    for n, d, s in [(100, 4, 1), (100, 4, 2), (50, 7, 3), (20, 1, 5)]:
        assert lookback_pre_periods(n, d, s) == n - d - s + 1


def test_pre_periods_shrinks_one_per_sim():
    base = lookback_pre_periods(100, 4, 1)
    assert lookback_pre_periods(100, 4, 2) == base - 1
    assert lookback_pre_periods(100, 4, 3) == base - 2


def test_pre_periods_minimal_one():
    assert lookback_pre_periods(5, 4, 1) == 1            # 5 - 4 - 1 + 1


def test_pre_periods_runs_off_start_raises():
    with pytest.raises(MlsynthConfigError, match="runs off the start"):
        lookback_pre_periods(4, 4, 1)                    # 4 - 4 - 1 + 1 = 0
    with pytest.raises(MlsynthConfigError, match="runs off the start"):
        lookback_pre_periods(5, 4, 2)                    # 5 - 4 - 2 + 1 = 0


@pytest.mark.parametrize("bad", [0, -1])
def test_pre_periods_nonpositive_args_raise(bad):
    with pytest.raises(MlsynthConfigError, match="positive integer"):
        lookback_pre_periods(bad, 4, 1)
    with pytest.raises(MlsynthConfigError, match="positive integer"):
        lookback_pre_periods(100, bad, 1)
    with pytest.raises(MlsynthConfigError, match="positive integer"):
        lookback_pre_periods(100, 4, bad)


def test_pre_periods_rejects_bool():
    with pytest.raises(MlsynthConfigError, match="positive integer"):
        lookback_pre_periods(True, 4, 1)


# === lookback_treatment_window ===

def test_treatment_window_smoke():
    assert lookback_treatment_window(100, 4, 1) == (96, 99)


def test_treatment_window_slides_back_one_per_sim():
    assert lookback_treatment_window(100, 4, 1) == (96, 99)
    assert lookback_treatment_window(100, 4, 2) == (95, 98)
    assert lookback_treatment_window(100, 4, 3) == (94, 97)


def test_treatment_window_length_equals_duration():
    for n, d, s in [(100, 4, 1), (50, 7, 3), (30, 2, 5), (12, 1, 1)]:
        start, end = lookback_treatment_window(n, d, s)
        assert end - start + 1 == d


def test_treatment_window_sim1_ends_at_last_period():
    n = 40
    _, end = lookback_treatment_window(n, 6, 1)
    assert end == n - 1                                  # 0-indexed last period


def test_treatment_window_inherits_off_start_guard():
    with pytest.raises(MlsynthConfigError, match="runs off the start"):
        lookback_treatment_window(5, 4, 2)


# === aggregate_treated ===

def test_aggregate_treated_sum_smoke():
    Y = _panel()
    s = aggregate_treated(Y, frozenset(["A", "B"]))
    assert isinstance(s, pd.Series)
    assert len(s) == 6
    assert s.index.equals(Y.index)
    np.testing.assert_allclose(s.to_numpy(), (Y["A"] + Y["B"]).to_numpy())


def test_aggregate_treated_mean():
    Y = _panel()
    s = aggregate_treated(Y, frozenset(["A", "B"]), how="mean")
    np.testing.assert_allclose(s.to_numpy(), ((Y["A"] + Y["B"]) / 2).to_numpy())


def test_aggregate_treated_sum_equals_mean_times_k():
    Y = _panel()
    cand = frozenset(["A", "B", "C"])
    s = aggregate_treated(Y, cand, how="sum")
    m = aggregate_treated(Y, cand, how="mean")
    np.testing.assert_allclose(s.to_numpy(), m.to_numpy() * len(cand))


def test_aggregate_treated_single_unit_sum_eq_mean():
    Y = _panel()
    s = aggregate_treated(Y, frozenset(["C"]), how="sum")
    m = aggregate_treated(Y, frozenset(["C"]), how="mean")
    np.testing.assert_allclose(s.to_numpy(), Y["C"].to_numpy())
    np.testing.assert_allclose(m.to_numpy(), Y["C"].to_numpy())


def test_aggregate_treated_invalid_how_raises():
    with pytest.raises(MlsynthConfigError, match="sum.*mean"):
        aggregate_treated(_panel(), frozenset(["A"]), how="median")


def test_aggregate_treated_empty_candidate_raises():
    with pytest.raises(MlsynthConfigError, match="at least one unit"):
        aggregate_treated(_panel(), frozenset())


def test_aggregate_treated_unknown_unit_raises():
    with pytest.raises(MlsynthDataError, match="not found"):
        aggregate_treated(_panel(), frozenset(["A", "Z"]))


# === donor_matrix ===

def test_donor_matrix_smoke():
    Y = _panel()
    D = donor_matrix(Y, frozenset(["A", "B"]))
    assert isinstance(D, pd.DataFrame)
    assert list(D.columns) == ["C", "D"]
    assert D.shape == (6, 2)
    assert D.index.equals(Y.index)


def test_donor_matrix_excludes_candidate_preserves_order():
    Y = _panel()
    D = donor_matrix(Y, frozenset(["C"]))
    assert list(D.columns) == ["A", "B", "D"]   # panel order, C removed


def test_donor_matrix_no_donors_raises():
    Y = _panel()
    with pytest.raises(MlsynthDataError, match="No donor"):
        donor_matrix(Y, frozenset(["A", "B", "C", "D"]))


def test_donor_matrix_unknown_unit_raises():
    with pytest.raises(MlsynthDataError, match="not found"):
        donor_matrix(_panel(), frozenset(["Z"]))


def test_donor_matrix_empty_candidate_raises():
    with pytest.raises(MlsynthConfigError, match="at least one unit"):
        donor_matrix(_panel(), frozenset())


# === composition with geoex_dataprep (the real entry path) ===

def test_shaping_composes_with_geoex_dataprep():
    """Shaping + window helpers consume geoex_dataprep's canonical output."""
    long_df = pd.DataFrame(
        {
            "unit": ["A"] * 6 + ["B"] * 6 + ["C"] * 6 + ["D"] * 6,
            "time": list(range(6)) * 4,
            "outcome": list(np.arange(1, 7))
            + list(2 * np.arange(1, 7))
            + list(np.arange(1, 7) + 5)
            + list(-np.arange(1, 7.0)),
        }
    )
    prep = geoex_dataprep(long_df, "unit", "time", "outcome")
    Ywide, n_periods = prep["Ywide"], prep["n_periods"]

    treated = aggregate_treated(Ywide, frozenset(["A", "B"]), how="mean")
    donors = donor_matrix(Ywide, frozenset(["A", "B"]))

    assert len(treated) == n_periods
    assert list(donors.columns) == ["C", "D"]
    # the window helper keys off the same n_periods
    assert lookback_pre_periods(n_periods, 2, 1) == n_periods - 2


# === scaled_l2_imbalance ===

def test_scaled_l2_equals_one_at_uniform_weights():
    X0 = np.array([[1.0, 3], [2, 5], [4, 9]])
    X1 = np.array([2.0, 4, 7])
    assert scaled_l2_imbalance(X1, X0, [0.5, 0.5]) == pytest.approx(1.0)


def test_scaled_l2_zero_at_perfect_fit():
    X0 = np.array([[1.0, 3], [2, 5], [4, 9]])
    w = np.array([0.25, 0.75])
    X1 = X0 @ w
    assert scaled_l2_imbalance(X1, X0, w) == pytest.approx(0.0)


def test_scaled_l2_matches_formula():
    X0 = np.array([[1.0, 3], [2, 5], [4, 9]])
    X1 = np.array([2.0, 4, 7])
    w = np.array([1.0, 0.0])
    num = float(np.linalg.norm(X0 @ w - X1))
    den = float(np.linalg.norm(X0 @ np.full(2, 0.5) - X1))
    assert scaled_l2_imbalance(X1, X0, w) == pytest.approx(num / den)


def test_scaled_l2_degenerate_denominator_is_nan():
    X0 = np.array([[1.0, 3], [2, 5], [4, 9]])
    X1 = X0 @ np.full(2, 0.5)                 # uniform average exactly -> denom 0
    assert np.isnan(scaled_l2_imbalance(X1, X0, [1.0, 0.0]))


def test_scaled_l2_weights_shape_mismatch_raises():
    X0 = np.array([[1.0, 3], [2, 5], [4, 9]])
    X1 = np.array([2.0, 4, 7])
    with pytest.raises(MlsynthConfigError, match="weights"):
        scaled_l2_imbalance(X1, X0, [1.0, 0.0, 0.0])


def test_scaled_l2_treated_length_mismatch_raises():
    X0 = np.array([[1.0, 3], [2, 5], [4, 9]])
    with pytest.raises(MlsynthConfigError, match="treated_pre"):
        scaled_l2_imbalance(np.array([2.0, 4]), X0, [0.5, 0.5])


def test_scaled_l2_donor_not_2d_raises():
    with pytest.raises(MlsynthConfigError, match="2-D"):
        scaled_l2_imbalance(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]), [1.0])


# === fit_intercept ===

def test_fit_intercept_matches_mean_residual():
    Y0 = np.array([[1.0, 2], [3, 4], [5, 6]])
    w = np.array([0.5, 0.5])
    y = np.array([2.0, 4, 7])
    assert fit_intercept(y, Y0, w) == pytest.approx(float(np.mean(y - Y0 @ w)))


def test_fit_intercept_zero_when_matched():
    Y0 = np.array([[1.0, 2], [3, 4], [5, 6]])
    w = np.array([0.25, 0.75])
    y = Y0 @ w                                  # exact -> no level gap
    assert fit_intercept(y, Y0, w) == pytest.approx(0.0)


def test_fit_intercept_recovers_pure_level_shift():
    Y0 = np.array([[1.0, 2], [3, 4], [5, 6]])
    w = np.array([0.5, 0.5])
    y = Y0 @ w + 10.0
    assert fit_intercept(y, Y0, w) == pytest.approx(10.0)


# === counterfactual ===

def test_counterfactual_is_intercept_plus_yw():
    Y0 = np.array([[1.0, 2], [3, 4]])
    w = np.array([0.5, 0.5])
    np.testing.assert_allclose(counterfactual(Y0, w, 10.0), 10.0 + Y0 @ w)


def test_counterfactual_default_intercept_zero():
    Y0 = np.array([[1.0, 2], [3, 4]])
    w = np.array([0.5, 0.5])
    np.testing.assert_allclose(counterfactual(Y0, w), Y0 @ w)


# === fit_augsynth_once ===

def _synthesizable(T=12, level=0.0, seed=0):
    rng = np.random.default_rng(seed)
    Y0 = rng.normal(size=(T, 3)) + np.arange(T)[:, None] * 0.5
    w = np.array([0.5, 0.3, 0.2])
    y = level + Y0 @ w
    return y, Y0, w


def test_fit_augsynth_once_ridge_smoke():
    y, Y0, _ = _synthesizable()
    fit = fit_augsynth_once(y, Y0, augment="ridge")
    assert fit.weights.shape == (3,)
    assert np.all(np.isfinite(fit.weights))
    assert fit.intercept == 0.0
    assert fit.pre_rmspe >= 0.0
    cf = fit.predict(Y0)
    assert np.sqrt(np.mean((y - cf) ** 2)) < 0.5


def test_fit_augsynth_once_simplex_smoke():
    y, Y0, _ = _synthesizable()
    fit = fit_augsynth_once(y, Y0, augment=None)
    assert fit.weights.shape == (3,)
    assert np.all(fit.weights >= -1e-8)
    assert abs(fit.weights.sum() - 1.0) < 1e-6
    cf = fit.predict(Y0)
    assert np.sqrt(np.mean((y - cf) ** 2)) < 0.5


def test_fit_augsynth_once_simplex_intercept_recovers_level_shift():
    """Plain simplex can't reach a big level offset; the intercept does."""
    y, Y0, _ = _synthesizable(level=100.0)
    fit = fit_augsynth_once(y, Y0, augment=None)
    assert fit.intercept == pytest.approx(100.0, abs=2.0)
    cf = fit.predict(Y0)
    assert np.sqrt(np.mean((y - cf) ** 2)) < 1.0


def test_fit_augsynth_once_returns_scaled_l2_and_rmse():
    y, Y0, _ = _synthesizable()
    fit = fit_augsynth_once(y, Y0, augment="ridge")
    assert np.isfinite(fit.scaled_l2) and fit.scaled_l2 >= 0.0
    assert np.isfinite(fit.pre_rmspe)


def test_fit_augsynth_once_invalid_augment_raises():
    y, Y0, _ = _synthesizable()
    with pytest.raises(MlsynthConfigError, match="augment"):
        fit_augsynth_once(y, Y0, augment="bogus")


def test_fit_augsynth_once_shape_mismatch_raises():
    y, Y0, _ = _synthesizable()
    with pytest.raises(MlsynthConfigError, match="Y0_pre"):
        fit_augsynth_once(y[:-1], Y0, augment="ridge")
