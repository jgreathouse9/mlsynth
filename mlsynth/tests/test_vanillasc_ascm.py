"""Coverage of VanillaSC's Augmented-SCM features and its failure reporting.

Exercises the ridge-augmentation modes (parallel / residualized covariates),
the conformal prediction intervals, the plotting path, and every edge / error
branch of the vanillasc pipeline so the module is fully covered. Smoke tests
confirm the happy path; edge-case tests push unusual inputs; failure tests
assert the estimator raises the right typed error *and* reports why.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless: no display during tests

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError

pytest.importorskip("cvxpy")  # the exact simplex QP base


def _panel(n_units=8, T=18, t0=13, seed=0, with_cov=True, effect=3.0):
    """Long panel: unit 'u0' treated from period t0, with a planted effect."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(T, 2))
    loads = rng.uniform(0.2, 1.0, size=(n_units, 2))
    rows = []
    for i in range(n_units):
        base = factor @ loads[i] * 5 + 20 + rng.normal(0, 0.1, size=T)
        x1 = loads[i, 0] + rng.normal(0, 0.01, size=T)
        for t in range(T):
            y = base[t] - (effect if (i == 0 and t >= t0) else 0.0)
            rows.append((f"u{i}", t, y, int(i == 0 and t >= t0), x1[t]))
    cols = ["unit", "time", "y", "treated", "x1"]
    df = pd.DataFrame(rows, columns=cols)
    return df if with_cov else df.drop(columns="x1")


_BASE = {"outcome": "y", "treat": "treated", "unitid": "unit", "time": "time",
         "display_graphs": False}


# --------------------------------------------------------------------------- #
# Smoke: the new Augmented-SCM modes run and return finite, standardized output
# --------------------------------------------------------------------------- #
def test_ridge_augment_smoke():
    res = VanillaSC({"df": _panel(), **_BASE, "augment": "ridge", "inference": False}).fit()
    assert np.isfinite(res.effects.att)
    assert res.method_details.parameters_used["augment"] == "ridge"
    # ridge augmentation leaves the simplex: weights need not sum to one
    assert res.fit_diagnostics.rmse_pre is not None


def test_ridge_with_covariates_parallel_and_residualized():
    for residualize in (False, True):
        res = VanillaSC({"df": _panel(), **_BASE, "covariates": ["x1"],
                         "augment": "ridge", "residualize": residualize,
                         "inference": False}).fit()
        assert np.isfinite(res.effects.att)


def test_conformal_inference_bands_contain_counterfactual():
    res = VanillaSC({"df": _panel(seed=3), **_BASE, "augment": "ridge",
                     "inference": "conformal", "scpi_sims": 40, "alpha": 0.1}).fit()
    inf = res.inference
    assert "conformal" in inf.method.lower()
    assert inf.p_value is not None and 0.0 <= inf.p_value <= 1.0
    cf = res.time_series.counterfactual_outcome
    lo = np.asarray(inf.details["counterfactual_lower"], dtype=float)
    hi = np.asarray(inf.details["counterfactual_upper"], dtype=float)
    post = ~np.isnan(lo)
    assert post.any()
    # the band brackets the plotted counterfactual at every post period
    assert np.all(lo[post] - 1e-9 <= cf[post]) and np.all(cf[post] <= hi[post] + 1e-9)


def test_fixed_lambda_skips_cv():
    res = VanillaSC({"df": _panel(), **_BASE, "augment": "ridge",
                     "ridge_lambda": 5.0, "inference": False}).fit()
    assert np.isfinite(res.effects.att)


# --------------------------------------------------------------------------- #
# Plotting: the Plotter path runs and the prediction-interval band is drawn
# --------------------------------------------------------------------------- #
def test_plot_saved_with_conformal_band(tmp_path):
    out = tmp_path / "vsc.png"
    VanillaSC({"df": _panel(seed=5), **_BASE, "augment": "ridge",
               "inference": "conformal", "scpi_sims": 30,
               "display_graphs": False, "save": str(out)}).fit()
    assert out.exists() and out.stat().st_size > 0


def test_plot_saved_default_name_without_inference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    VanillaSC({"df": _panel(), **_BASE, "inference": False, "save": True}).fit()
    assert any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_display_graphs_show(monkeypatch):
    shown = {"n": 0}
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.__setitem__("n", shown["n"] + 1))
    VanillaSC({"df": _panel(), **_BASE, "inference": False, "display_graphs": True}).fit()
    assert shown["n"] == 1


# --------------------------------------------------------------------------- #
# Other inference modes (scpi, lto, placebo) + their edge knobs
# --------------------------------------------------------------------------- #
def test_scpi_gaussian_and_empirical():
    for e_method in ("gaussian", "empirical"):
        res = VanillaSC({"df": _panel(seed=7), **_BASE, "inference": "scpi",
                         "scpi_sims": 40, "scpi_e_method": e_method}).fit()
        assert res.inference.ci_lower is not None


def test_lto_runs_and_subsamples():
    res = VanillaSC({"df": _panel(n_units=7, seed=2), **_BASE,
                     "inference": "lto", "lto_max_pairs": 3}).fit()
    assert res.inference.p_value is not None
    assert res.inference.details["subsampled"] is True


def test_placebo_with_covariates():
    res = VanillaSC({"df": _panel(seed=1), **_BASE, "covariates": ["x1"],
                     "backend": "mscmt", "inference": "placebo",
                     "mscmt_maxiter": 5, "mscmt_popsize": 5}).fit()
    assert res.inference.p_value is not None


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_per_covariate_windows_use_independent_means():
    # two covariates with *different* windows -> the joint-na.omit path can't
    # apply, so means are taken independently (per-covariate fallback).
    df = _panel(seed=3).copy()
    df["x2"] = df["x1"] + 0.5
    res = VanillaSC({"df": df, **_BASE, "covariates": ["x1", "x2"],
                     "covariate_windows": {"x1": (0, 6), "x2": (0, 10)},
                     "backend": "mscmt", "mscmt_maxiter": 5, "mscmt_popsize": 5,
                     "inference": False}).fit()
    assert np.isfinite(res.effects.att)


def test_covariate_window_out_of_range_falls_back():
    # a window outside the pre-period range -> no years match -> fall back to all
    res = VanillaSC({"df": _panel(), **_BASE, "covariates": ["x1"],
                     "covariate_windows": {"x1": (900, 999)},
                     "backend": "mscmt", "mscmt_maxiter": 5, "mscmt_popsize": 5,
                     "inference": False}).fit()
    assert np.isfinite(res.effects.att)


def test_lto_skipped_when_too_few_donors():
    # J < 3: the LTO branch is skipped (no triple to form); fit still succeeds.
    res = VanillaSC({"df": _panel(n_units=3, seed=0), **_BASE,
                     "inference": "lto"}).fit()
    assert np.isfinite(res.effects.att)


# --------------------------------------------------------------------------- #
# Failure reporting: the estimator raises the right typed error, with a message
# --------------------------------------------------------------------------- #
def test_non_config_input_raises_config_error():
    with pytest.raises(MlsynthConfigError, match="VanillaSCConfig"):
        VanillaSC(42)


def test_missing_outcome_column_raises():
    with pytest.raises((MlsynthDataError, MlsynthConfigError)):
        VanillaSC({"df": _panel(), **{**_BASE, "outcome": "nope"}}).fit()


def test_unknown_covariate_raises_data_error():
    with pytest.raises(MlsynthDataError, match="not in DataFrame"):
        VanillaSC({"df": _panel(), **_BASE, "covariates": ["ghost"],
                   "backend": "mscmt"}).fit()


def test_nan_covariate_means_raises_data_error():
    df = _panel()
    df.loc[df["unit"] == "u3", "x1"] = np.nan        # one unit has no covariate
    with pytest.raises(MlsynthDataError, match="NaN"):
        VanillaSC({"df": df, **_BASE, "covariates": ["x1"],
                   "backend": "mscmt"}).fit()


def test_multiple_treated_units_raises_data_error():
    df = _panel()
    df.loc[(df["unit"] == "u1") & (df["time"] >= 13), "treated"] = 1
    with pytest.raises(MlsynthDataError):
        VanillaSC({"df": df, **_BASE, "inference": False}).fit()


def test_scpi_plot_band_padding(tmp_path):
    # SCPI stores post-only bands -> the plotter pads them to the full axis.
    out = tmp_path / "scpi.png"
    VanillaSC({"df": _panel(seed=9), **_BASE, "inference": "scpi",
               "scpi_sims": 30, "display_graphs": False, "save": str(out)}).fit()
    assert out.exists()


# --------------------------------------------------------------------------- #
# Direct helper coverage: scpi / lto numerical edge branches
# --------------------------------------------------------------------------- #
def test_lto_f_clamps_negative_discriminant():
    from mlsynth.utils.vanillasc_helpers.lto import lto_f
    # a large alpha drives the discriminant negative -> clamp to 0
    assert np.isfinite(lto_f(5, 0.95))


def test_rmspe_ratio_resid_perfect_prefit_is_inf():
    from mlsynth.utils.vanillasc_helpers.lto import _rmspe_ratio_resid
    y = np.array([1.0, 2.0, 3.0, 4.0])
    cf = y.copy()                       # perfect pre-fit -> pre_r == 0 -> inf
    assert _rmspe_ratio_resid(y, cf, pre=2) == float("inf")


def test_regularization_rho_degenerate_uses_rho_max():
    from mlsynth.utils.vanillasc_helpers.scpi import _regularization_rho, _RHO_MAX
    rng = np.random.default_rng(0)
    u = np.full(40, 1.0) + 1e-9 * rng.normal(size=40)   # ~constant residuals
    B = 50.0 * rng.normal(size=(40, 4))                 # large donor spread
    rho = _regularization_rho(u, B, d0=4)
    assert rho == pytest.approx(_RHO_MAX)


def test_mat_regularize_zero_matrix_returns_none():
    from mlsynth.utils.vanillasc_helpers.scpi import _mat_regularize
    scale, Qreg = _mat_regularize(np.zeros((3, 3)))
    assert scale == 0.0 and Qreg is None


def test_out_of_sample_ls_method():
    from mlsynth.utils.vanillasc_helpers.scpi import _out_of_sample
    rng = np.random.default_rng(1)
    u = rng.normal(size=20)
    Xe0 = np.column_stack([rng.normal(size=20), np.ones(20)])
    Xe1 = np.column_stack([rng.normal(size=3), np.ones(3)])
    lb, ub = _out_of_sample(u, Xe0, Xe1, e_alpha=0.1, e_method="ls")
    assert np.all(np.asarray(lb) <= np.asarray(ub) + 1e-9)


def test_lto_loss_branch_with_no_effect():
    # no planted effect -> the treated unit does not dominate every triple, so
    # the loss-counting branch fires (p-value strictly above 0).
    res = VanillaSC({"df": _panel(n_units=7, seed=4, effect=0.0), **_BASE,
                     "inference": "lto"}).fit()
    assert res.inference.p_value > 0.0


def test_scpi_direct_no_misspecification_and_degenerate_donors():
    from mlsynth.utils.vanillasc_helpers.scpi import scpi_intervals
    rng = np.random.default_rng(2)
    T, pre, J = 16, 12, 4
    # near-constant donors -> zero-variance design: rho exceeds every weight
    # (idxw fallback) and Q is degenerate (the quadratic constraint drops out).
    Y0 = 5.0 + 1e-10 * rng.normal(size=(T, J))
    y = 5.0 + rng.normal(0, 0.02, size=T); y[pre:] -= 0.5
    W = np.full(J, 1.0 / J)
    # u_missp=False also exercises the zero-mean residual branch.
    sc = scpi_intervals(y, Y0, pre, W, sims=30, u_missp=False, seed=0)
    assert sc.lower.shape == sc.upper.shape


# --------------------------------------------------------------------------- #
# augsynth Kansas ladder -- reproduced through the PUBLIC API (not the engine).
# This is the definition of done for the augsynth port: all four cells must
# match augsynth via VanillaSC(...).fit(), with the user log-transforming the
# covariates beforehand (mlsynth's covariate convention -- plain column names).
# --------------------------------------------------------------------------- #
_KANSAS = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "kansas_ascm.csv")


@pytest.mark.skipif(not os.path.exists(_KANSAS), reason="Kansas data not present")
def test_augsynth_kansas_ladder_public_api():
    df = pd.read_csv(_KANSAS)
    # augsynth Kansas spec: lngdpcapita + log(revstate/revlocal/avgwkly) +
    # estabs + emplvl. The user logs the three columns before input.
    for c in ("revstatecapita", "revlocalcapita", "avgwklywagecapita"):
        df[c] = np.log(df[c])
    covs = ["lngdpcapita", "revstatecapita", "revlocalcapita",
            "avgwklywagecapita", "estabscapita", "emplvlcapita"]
    base = dict(df=df, outcome="lngdpcapita", treat="treated", unitid="fips",
                time="year_qtr", display_graphs=False, inference=False)

    def att(cfg):
        return float(VanillaSC({**base, **cfg}).fit().effects.att)

    scm = att({})
    ridge = att({"augment": "ridge"})
    cov = att({"augment": "ridge", "covariates": covs})
    rez = att({"augment": "ridge", "covariates": covs, "residualize": True})

    assert abs(scm - (-0.0294)) < 0.003, f"SCM {scm}"
    assert abs(ridge - (-0.0401)) < 0.003, f"ridge {ridge}"
    assert abs(cov - (-0.0629)) < 0.004, f"covariate {cov}"
    assert abs(rez - (-0.0572)) < 0.004, f"residualized {rez}"
    # the de-biasing ladder is monotone in |ATT|
    assert abs(scm) < abs(ridge) < abs(cov)
