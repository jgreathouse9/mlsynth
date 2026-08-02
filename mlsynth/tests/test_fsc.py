"""Tests for FSC -- functional synthetic control (Okano & Kurisu 2026).

Layered as the contract asks: a smoke test that the thing runs end to end on
minimal input, invariant tests that pin the properties the paper proves rather
than the floats a solver happens to produce, edge cases on degenerate panels,
and failure tests asserting every bad input is *reported* through the right
translated exception rather than swallowed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import FSC
from mlsynth.exceptions import (MlsynthConfigError, MlsynthDataError,
                                MlsynthEstimationError)
from mlsynth.utils.fsc_helpers.basis import cubic_bspline_basis
from mlsynth.utils.fsc_helpers.config import FSCConfig
from mlsynth.utils.fsc_helpers.project import (project_increasing,
                                               project_psd, vech_to_matrix)
from mlsynth.utils.fsc_helpers.structures import FSCResults


# ---------------------------------------------------------------------------
# panel builders
# ---------------------------------------------------------------------------
def curve_panel(n_units=6, n_times=12, n_grid=15, n_pre=8, seed=0,
                effect=0.0, weights=None):
    """Factor-model panel of curves; unit 0 is treated from ``n_pre`` on.

    Each unit's curve is a loading-weighted combination of smooth factors, so
    the treated unit lies near the donor hull and the pre-fit is informative.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_grid)
    factors = np.stack([np.ones_like(x), np.sin(np.pi * x), np.cos(2 * np.pi * x)])
    loads = rng.uniform(0.2, 1.0, size=(n_units, factors.shape[0]))
    drift = rng.uniform(0.9, 1.1, size=(n_times, factors.shape[0]))
    cube = np.einsum("if,tf,fm->itm", loads, drift, factors)
    cube += rng.normal(scale=0.01, size=cube.shape)
    if weights is not None:                       # treated = exact donor mix
        cube[0] = np.einsum("j,jtm->tm", np.asarray(weights), cube[1:])
    cube[0, n_pre:] += effect
    return _to_long(cube, n_pre), cube


def _to_long(cube, n_pre, argument="x", value="y"):
    n_units, n_times, n_grid = cube.shape
    ui, ti, ai = np.meshgrid(np.arange(n_units), np.arange(n_times),
                             np.arange(n_grid), indexing="ij")
    return pd.DataFrame({
        "unit": [f"u{i}" for i in ui.ravel()],
        "time": ti.ravel(),
        argument: ai.ravel().astype(float) / (n_grid - 1),
        value: cube.ravel(),
        "treat": ((ui.ravel() == 0) & (ti.ravel() >= n_pre)).astype(int),
    })


def quantile_panel(n_units=5, n_times=10, n_grid=20, n_pre=6, seed=1):
    """Panel of quantile functions -- increasing in the argument by construction."""
    rng = np.random.default_rng(seed)
    p = np.linspace(0.02, 0.98, n_grid)
    base = np.sqrt(2.0) * np.vectorize(lambda q: float(np.sqrt(2) * q))(p)
    cube = np.empty((n_units, n_times, n_grid))
    for i in range(n_units):
        loc, scale = rng.uniform(0, 1), rng.uniform(0.8, 1.4)
        for t in range(n_times):
            cube[i, t] = loc + scale * (base + 0.05 * t * p)
    return _to_long(cube, n_pre, argument="p", value="q"), cube


def matrix_panel(n_units=5, n_times=10, dim=3, n_pre=6, seed=2):
    """Panel of symmetric PSD matrices stored as row-major lower triangles."""
    rng = np.random.default_rng(seed)
    rows, cols = np.tril_indices(dim)
    cube = np.empty((n_units, n_times, rows.size))
    for i in range(n_units):
        A = rng.normal(size=(dim, dim))
        for t in range(n_times):
            M = A @ A.T + (0.1 * t + 0.5) * np.eye(dim)
            cube[i, t] = M[rows, cols]
    return _to_long(cube, n_pre, argument="cell", value="cov"), cube


def base_config(df, **kw):
    cfg = dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
               argument="x", display_graphs=False)
    cfg.update(kw)
    return cfg


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------
def test_fit_returns_populated_results():
    df, _ = curve_panel()
    res = FSC(base_config(df)).fit()

    assert isinstance(res, FSCResults)
    assert len(res.curves) == 12
    assert np.isfinite(res.effects.att)
    assert np.isfinite(res.pre_treatment_fit)
    assert res.method_details.method_name == "FSC"
    for curve in res.curves:
        assert curve.counterfactual.shape == curve.observed.shape
        assert np.all(np.isfinite(curve.counterfactual))
        assert np.allclose(curve.effect, curve.observed - curve.counterfactual)


def test_time_series_gap_is_the_difference_of_the_reported_series():
    """The scalar series is a linear functional, so the identity must hold."""
    df, _ = curve_panel()
    ts = FSC(base_config(df)).fit().time_series
    assert np.allclose(ts.estimated_gap,
                       ts.observed_outcome - ts.counterfactual_outcome)
    assert ts.time_periods.size == ts.observed_outcome.size


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------
def test_plain_fsc_weights_lie_on_the_simplex():
    df, _ = curve_panel()
    res = FSC(base_config(df, augment=False)).fit()
    w = res.fsc_weights
    assert w.min() >= -1e-9
    assert res.augmented_weights is None
    assert np.isclose(w.sum(), 1.0)


def test_augmented_weights_sum_to_one_and_may_leave_the_simplex():
    """Eq. (10) constrains the sum, not the sign."""
    df, _ = curve_panel()
    res = FSC(base_config(df)).fit()
    assert np.isclose(res.augmented_weights.sum(), 1.0)
    assert np.isclose(res.fsc_weights.sum(), 1.0)


def test_augmentation_weakly_improves_the_pre_treatment_fit():
    df, _ = curve_panel()
    res = FSC(base_config(df)).fit()
    assert res.pre_treatment_fit <= res.pre_treatment_fit_fsc + 1e-9


def test_perfect_pre_fit_leaves_the_weights_unaugmented():
    """r_1 = r_0' gamma makes the correction in eq. (9) vanish identically."""
    w = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    df, _ = curve_panel(weights=w)
    res = FSC(base_config(df)).fit()
    assert res.pre_treatment_fit_fsc < 1e-6
    assert np.allclose(res.fsc_weights, res.augmented_weights, atol=1e-6)


def test_recovers_a_known_donor_mix():
    w = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    df, _ = curve_panel(weights=w)
    res = FSC(base_config(df)).fit()
    assert np.allclose(res.fsc_weights, w, atol=1e-3)
    assert max(np.abs(c.effect).max() for c in res.curves) < 1e-4


def test_constant_shift_is_recovered_as_the_effect():
    w = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    df, _ = curve_panel(weights=w, effect=0.75)
    res = FSC(base_config(df)).fit()
    post = [c for c in res.curves if c.is_post]
    assert len(post) == 4
    for curve in post:
        assert np.allclose(curve.effect, 0.75, atol=1e-3)
    assert np.isclose(res.effects.att, 0.75, atol=1e-3)


def test_weights_are_invariant_to_relabelling_the_grid():
    """The stacked design only sees the values, so a monotone relabel is inert."""
    df, _ = curve_panel()
    shifted = df.copy()
    shifted["x"] = shifted["x"] * 7.0 + 3.0
    a = FSC(base_config(df)).fit()
    b = FSC(base_config(shifted)).fit()
    assert np.allclose(a.fsc_weights, b.fsc_weights, atol=1e-8)


def test_distribution_space_returns_an_increasing_counterfactual():
    df, _ = quantile_panel()
    res = FSC(base_config(df, outcome="q", argument="p",
                          space="distribution")).fit()
    for curve in res.curves:
        assert np.all(np.diff(curve.counterfactual) >= -1e-9)


def test_matrix_space_returns_a_psd_counterfactual():
    df, _ = matrix_panel()
    res = FSC(base_config(df, outcome="cov", argument="cell",
                          space="matrix")).fit()
    for curve in res.curves:
        M = vech_to_matrix(curve.counterfactual, 3)
        assert np.linalg.eigvalsh(M).min() >= -1e-8


def test_matrix_space_skips_the_basis_expansion():
    """H is Euclidean there, so the standard basis is already orthonormal."""
    df, _ = matrix_panel()
    res = FSC(base_config(df, outcome="cov", argument="cell",
                          space="matrix")).fit()
    assert res.method_details.parameters_used["n_basis"] is None


# ---------------------------------------------------------------------------
# penalty selection
# ---------------------------------------------------------------------------
def test_explicit_lambda_is_used_verbatim():
    df, _ = curve_panel()
    res = FSC(base_config(df, ridge_lambda=2.5)).fit()
    assert res.ridge_lambda == pytest.approx(2.5)
    assert res.lambda_selected_by_cv is False


def test_cross_validated_lambda_lies_inside_the_bounds():
    df, _ = curve_panel()
    res = FSC(base_config(df, lambda_bounds=(0.0, 4.0))).fit()
    assert 0.0 <= res.ridge_lambda <= 4.0
    assert res.lambda_selected_by_cv is True


def test_large_lambda_collapses_augmentation_onto_the_simplex_fit():
    """As lambda grows the correction in eq. (9) vanishes."""
    df, _ = curve_panel()
    res = FSC(base_config(df, ridge_lambda=1e12)).fit()
    assert np.allclose(res.fsc_weights, res.augmented_weights, atol=1e-6)


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
def test_conformal_band_brackets_the_counterfactual():
    df, _ = curve_panel()
    res = FSC(base_config(df, inference="conformal")).fit()
    for curve in res.curves:
        if curve.is_post:
            assert np.all(curve.lower <= curve.counterfactual + 1e-12)
            assert np.all(curve.upper >= curve.counterfactual - 1e-12)
            assert np.all(curve.upper - curve.lower > 0)


def test_conformal_band_widens_with_a_smaller_alpha():
    df, _ = curve_panel()
    wide = FSC(base_config(df, conformal_alpha=0.05)).fit()
    narrow = FSC(base_config(df, conformal_alpha=0.40)).fit()
    w = [c for c in wide.curves if c.is_post][0]
    n = [c for c in narrow.curves if c.is_post][0]
    assert np.mean(w.upper - w.lower) >= np.mean(n.upper - n.lower)


def test_placebo_p_values_are_ranks_over_the_units():
    df, _ = curve_panel()
    res = FSC(base_config(df, inference="placebo")).fit()
    n_units = 6
    for p in res.placebo_p_values:
        assert 1 / n_units - 1e-12 <= p <= 1.0
        assert np.isclose(p * n_units, round(p * n_units))


def test_inference_none_leaves_the_band_and_p_values_absent():
    df, _ = curve_panel()
    res = FSC(base_config(df, inference="none")).fit()
    assert res.placebo_p_values is None
    assert all(c.lower is None for c in res.curves)
    assert res.time_series.has_prediction_interval is False


def test_inference_both_populates_band_and_p_values():
    df, _ = curve_panel()
    res = FSC(base_config(df, inference="both")).fit()
    assert res.placebo_p_values is not None
    assert res.time_series.has_prediction_interval is True


# ---------------------------------------------------------------------------
# basis and projection units
# ---------------------------------------------------------------------------
def test_bspline_basis_is_a_partition_of_unity():
    B = cubic_bspline_basis(np.linspace(0, 1, 30), 12)
    assert B.shape == (30, 12)
    assert np.allclose(B.sum(axis=1), 1.0)
    assert B.min() >= 0.0


def test_bspline_basis_spans_the_grid_span():
    """A basis that vanishes on part of the grid would drop information."""
    B = cubic_bspline_basis(np.linspace(-2.0, 5.0, 40), 9)
    assert np.all(B.sum(axis=1) > 0)


def test_project_increasing_is_idempotent_on_increasing_input():
    x = np.linspace(0, 1, 25)
    v = np.sqrt(x)
    assert np.allclose(project_increasing(x, v), v, atol=1e-8)


def test_project_increasing_sorts_and_respects_bounds():
    x = np.linspace(0, 1, 25)
    v = np.linspace(1.0, 0.0, 25)                     # decreasing
    out = project_increasing(x, v, bounds=(0.25, 0.75))
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0.25 - 1e-12 and out.max() <= 0.75 + 1e-12


def test_project_psd_clips_negative_eigenvalues_only():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    M = A + A.T                                        # symmetric, indefinite
    rows, cols = np.tril_indices(4)
    out = vech_to_matrix(project_psd(M[rows, cols], 4), 4)
    assert np.linalg.eigvalsh(out).min() >= -1e-10
    good = vech_to_matrix(project_psd((M @ M.T)[rows, cols], 4), 4)
    assert np.allclose(good, M @ M.T, atol=1e-8)


def test_vech_roundtrip():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(5, 5)); M = A + A.T
    rows, cols = np.tril_indices(5)
    assert np.allclose(vech_to_matrix(M[rows, cols], 5), M)


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
def test_single_donor_is_forced_to_weight_one():
    df, _ = curve_panel(n_units=2)
    res = FSC(base_config(df)).fit()
    assert res.fsc_weights.shape == (1,)
    assert np.isclose(res.fsc_weights[0], 1.0)


def test_single_pre_period_runs():
    df, _ = curve_panel(n_times=6, n_pre=1)
    res = FSC(base_config(df, ridge_lambda=1.0)).fit()
    assert np.isfinite(res.effects.att)


def test_single_post_period_runs():
    df, _ = curve_panel(n_times=9, n_pre=8)
    res = FSC(base_config(df)).fit()
    assert sum(c.is_post for c in res.curves) == 1


def test_collinear_donors_do_not_blow_up():
    df, cube = curve_panel(n_units=5)
    dup = cube.copy()
    dup[2] = dup[1]                                    # exact duplicate donor
    res = FSC(base_config(_to_long(dup, 8))).fit()
    assert np.all(np.isfinite(res.augmented_weights))
    assert np.isclose(res.augmented_weights.sum(), 1.0)


def test_two_grid_points_runs_with_a_small_basis():
    df, _ = curve_panel(n_grid=2)
    res = FSC(base_config(df, n_basis=4)).fit()
    assert np.isfinite(res.pre_treatment_fit)


def test_unsorted_input_rows_do_not_change_the_answer():
    df, _ = curve_panel()
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a = FSC(base_config(df)).fit()
    b = FSC(base_config(shuffled)).fit()
    assert np.allclose(a.augmented_weights, b.augmented_weights, atol=1e-8)


# ---------------------------------------------------------------------------
# failures -- every one must be reported, not swallowed
# ---------------------------------------------------------------------------
def test_missing_argument_column_is_reported():
    df, _ = curve_panel()
    with pytest.raises(MlsynthDataError, match="argument"):
        FSC(base_config(df, argument="not_a_column")).fit()


def test_argument_column_colliding_with_a_reserved_column_is_reported():
    df, _ = curve_panel()
    with pytest.raises(MlsynthConfigError, match="distinct"):
        FSC(base_config(df, argument="time"))


def test_ragged_grid_is_reported():
    df, _ = curve_panel()
    ragged = df.drop(df.index[[3]])
    with pytest.raises(MlsynthDataError, match="grid"):
        FSC(base_config(ragged)).fit()


def test_non_numeric_outcome_is_reported():
    df, _ = curve_panel()
    df["y"] = df["y"].astype(str)
    with pytest.raises(MlsynthDataError, match="numeric"):
        FSC(base_config(df)).fit()


def test_non_finite_values_are_reported():
    df, _ = curve_panel()
    df.loc[0, "y"] = np.nan
    with pytest.raises(MlsynthDataError, match="finite|NaN"):
        FSC(base_config(df)).fit()


def test_no_treated_unit_is_reported():
    df, _ = curve_panel()
    df["treat"] = 0
    with pytest.raises(MlsynthDataError, match="treated"):
        FSC(base_config(df)).fit()


def test_two_treated_units_are_reported():
    df, _ = curve_panel()
    df.loc[(df["unit"] == "u1") & (df["time"] >= 8), "treat"] = 1
    with pytest.raises(MlsynthDataError, match="single treated"):
        FSC(base_config(df)).fit()


def test_no_donors_is_reported():
    df, _ = curve_panel(n_units=1)
    with pytest.raises(MlsynthDataError, match="donor"):
        FSC(base_config(df)).fit()


def test_treatment_from_the_first_period_is_reported():
    df, _ = curve_panel(n_pre=0)
    with pytest.raises(MlsynthDataError, match="pre-treatment"):
        FSC(base_config(df)).fit()


def test_matrix_space_with_a_non_triangular_grid_is_reported():
    df, _ = curve_panel(n_grid=15)                    # 15 = vech of 5x5, fine
    bad, _ = curve_panel(n_grid=8)                    # 8 is not m(m+1)/2
    with pytest.raises(MlsynthDataError, match="triangle"):
        FSC(base_config(bad, space="matrix")).fit()
    FSC(base_config(df, space="matrix")).fit()        # 15 is a valid vech


def test_unknown_space_is_reported():
    df, _ = curve_panel()
    with pytest.raises(MlsynthConfigError):
        FSCConfig(**base_config(df, space="hyperbolic"))


def test_bad_alpha_is_reported():
    df, _ = curve_panel()
    with pytest.raises(MlsynthConfigError):
        FSCConfig(**base_config(df, conformal_alpha=1.5))


def test_alpha_below_the_conformal_resolution_is_reported():
    """The band needs alpha > 1 / (T0 + 1); below that the test never rejects."""
    df, _ = curve_panel(n_times=6, n_pre=3)
    with pytest.raises(MlsynthEstimationError, match="alpha"):
        FSC(base_config(df, conformal_alpha=0.01)).fit()


def test_too_few_basis_functions_is_reported():
    df, _ = curve_panel()
    with pytest.raises(MlsynthConfigError):
        FSCConfig(**base_config(df, n_basis=3))


def test_inverted_lambda_bounds_are_reported():
    df, _ = curve_panel()
    with pytest.raises(MlsynthConfigError, match="lambda_bounds"):
        FSCConfig(**base_config(df, lambda_bounds=(5.0, 1.0)))


def test_negative_lambda_is_reported():
    df, _ = curve_panel()
    with pytest.raises(MlsynthConfigError):
        FSCConfig(**base_config(df, ridge_lambda=-1.0))


def test_extra_config_field_is_forbidden():
    df, _ = curve_panel()
    with pytest.raises(MlsynthConfigError):
        FSC(base_config(df, not_a_field=1))


def test_config_must_be_a_config_or_dict():
    with pytest.raises(MlsynthConfigError, match="FSCConfig"):
        FSC("not a config")


def test_value_bounds_only_apply_to_the_distribution_space():
    df, _ = curve_panel()
    with pytest.raises(MlsynthConfigError, match="value_bounds"):
        FSCConfig(**base_config(df, space="function", value_bounds=(0.0, 1.0)))


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def test_plot_smoke(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    df, _ = curve_panel()
    FSC(base_config(df, display_graphs=True)).fit()


def test_plot_save_to_file(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    df, _ = curve_panel()
    target = tmp_path / "fsc"
    FSC(base_config(df, display_graphs=True, save=str(target))).fit()
    assert target.with_suffix(".png").exists()
