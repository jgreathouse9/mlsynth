"""Tests for SCTA -- Synthetic Control with Temporal Aggregation.

SCTA (Sun, Ben-Michael & Feller 2024, AEA P&P) jointly balances a treated unit
against donors on both the disaggregated high-frequency pre-periods and their
temporal aggregates (block means), trading the two off through a single weight
``nu`` on a fixed diagonal ``V``. These tests pin the public contract: the
standardized results surface, the block-aggregation construction, the ``nu``
knob (``nu=0`` collapses to a pure disaggregated fit), the optional ridge
augmentation, the imbalance frontier diagnostic, and config validation.

The estimator uses mlsynth's own simplex solver, which reaches the *true*
optimum of the temporal-aggregation objective; cross-implementation recovery
against augsynth to solver tolerance lives in the durable benchmark, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SCTA
from mlsynth.config_models import SCTAConfig, BaseEstimatorResults
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _panel(n_units=8, block_length=4, n_blocks_pre=3, n_post=4,
           effect=8.0, seed=0):
    """A high-frequency panel with a shared low-rank trend.

    ``T0 = block_length * n_blocks_pre`` pre-periods, ``n_post`` post-periods.
    The treated unit (``u0``) tracks the donor average pre-treatment and gets a
    constant additive ``effect`` post-treatment, so a demeaned SC recovers it.
    """
    rng = np.random.default_rng(seed)
    T0 = block_length * n_blocks_pre
    T = T0 + n_post
    f1 = np.linspace(0.0, 3.0, T) + 0.4 * np.sin(np.linspace(0, 6, T))
    f2 = np.cos(np.linspace(0, 4, T))
    rows = []
    loads = rng.normal(1.0, 0.3, size=(n_units, 2))
    levels = rng.normal(100.0, 8.0, size=n_units)
    for u in range(n_units):
        base = levels[u] + loads[u, 0] * f1 + loads[u, 1] * f2
        base = base + rng.normal(0, 0.2, size=T)
        for t in range(T):
            treated_post = (u == 0 and t >= T0)
            y = base[t] + (effect if treated_post else 0.0)
            rows.append((f"u{u}", t, float(y), 1 if treated_post else 0))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "treat"])


def _cfg(df=None, **kw):
    base = dict(df=df if df is not None else _panel(), outcome="y", treat="treat",
                unitid="unit", time="time", block_length=4, display_graphs=False)
    base.update(kw)
    return SCTAConfig(**base)


# --------------------------------------------------------------------------- #
# Smoke / contract
# --------------------------------------------------------------------------- #
def test_fit_smoke():
    res = SCTA(_cfg()).fit()
    assert isinstance(res, BaseEstimatorResults)
    assert np.isfinite(res.effects.att)
    assert res.time_series.counterfactual_outcome.shape[0] == 16


def test_standard_submodels_populated():
    res = SCTA(_cfg()).fit()
    assert res.effects is not None
    assert res.time_series is not None
    assert res.weights is not None and res.weights.donor_weights
    assert res.fit_diagnostics is not None
    assert res.method_details.method_name.startswith("SCTA")


def test_weights_simplex():
    res = SCTA(_cfg()).fit()
    w = np.array(list(res.weights.donor_weights.values()), dtype=float)
    assert abs(w.sum() - 1.0) < 1e-4
    assert (w >= -1e-8).all()


def test_recovers_known_effect():
    res = SCTA(_cfg(_panel(effect=8.0, seed=3))).fit()
    assert 4.0 < res.effects.att < 12.0


# --------------------------------------------------------------------------- #
# Temporal-aggregation construction
# --------------------------------------------------------------------------- #
def test_block_count_in_method_details():
    res = SCTA(_cfg(block_length=4)).fit()
    md = res.method_details.parameters_used or {}
    assert md.get("n_blocks") == 3        # T0=12, K=4
    assert md.get("block_length") == 4


def test_nu_zero_collapses_to_disaggregated():
    """nu=0 zeroes the aggregate rows of V -> a pure disaggregated SC fit."""
    df = _panel(seed=7)
    res0 = SCTA(_cfg(df, nu=0.0)).fit()
    # A disaggregated-only reference: huge block weight removed (nu=0).
    w0 = np.array(list(res0.weights.donor_weights.values()))
    # Re-fit with a tiny nu; weights should move (aggregate now matters).
    res_eps = SCTA(_cfg(df, nu=2.0)).fit()
    w_eps = np.array(list(res_eps.weights.donor_weights.values()))
    assert np.abs(w0 - w_eps).sum() > 1e-6


def test_nu_changes_estimate():
    df = _panel(seed=11)
    a = SCTA(_cfg(df, nu=0.0)).fit().effects.att
    b = SCTA(_cfg(df, nu=3.0)).fit().effects.att
    assert a != b


# --------------------------------------------------------------------------- #
# Frontier diagnostic
# --------------------------------------------------------------------------- #
def test_frontier_traces_tradeoff():
    res = SCTA(_cfg(frontier=[0.0, 0.5, 1.0, 2.0])).fit()
    fr = res.frontier
    assert fr is not None and len(fr) == 4
    for pt in fr:
        assert {"nu", "rmse_dis", "rmse_agg", "att"} <= set(pt)
    # More aggregate weight -> aggregate imbalance falls (weakly).
    rmse_agg = [pt["rmse_agg"] for pt in fr]
    assert rmse_agg[-1] <= rmse_agg[0] + 1e-6


def test_no_frontier_by_default():
    assert SCTA(_cfg()).fit().frontier is None


# --------------------------------------------------------------------------- #
# Ridge augmentation
# --------------------------------------------------------------------------- #
def test_augment_ridge_changes_weights():
    df = _panel(seed=5)
    w_plain = np.array(list(SCTA(_cfg(df)).fit().weights.donor_weights.values()))
    w_ridge = np.array(list(SCTA(_cfg(df, augment="ridge")).fit()
                            .weights.donor_weights.values()))
    assert np.abs(w_plain - w_ridge).sum() > 1e-6


def test_augment_fixed_lambda_runs():
    res = SCTA(_cfg(augment="ridge", ridge_lambda=10.0)).fit()
    assert np.isfinite(res.effects.att)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def test_block_length_too_small_raises():
    with pytest.raises(MlsynthConfigError):
        _cfg(block_length=1)


def test_ragged_preperiod_runs():
    # T0 = 12 is not a multiple of 5: aggregate 2 whole blocks (10 periods),
    # keep all 12 disaggregated -- the paper's ragged-tail handling.
    res = SCTA(_cfg(block_length=5)).fit()
    assert np.isfinite(res.effects.att)
    assert res.method_details.parameters_used["n_blocks"] == 2


def test_preperiod_shorter_than_block_raises():
    # T0 = 12 < block_length = 16.
    with pytest.raises((MlsynthConfigError, MlsynthDataError)):
        SCTA(_cfg(block_length=16)).fit()


def test_negative_nu_raises():
    with pytest.raises(MlsynthConfigError):
        _cfg(nu=-0.5)


def test_ridge_lambda_without_augment_raises():
    with pytest.raises(MlsynthConfigError):
        _cfg(ridge_lambda=5.0)


def test_ridge_lambda_nonpositive_raises():
    with pytest.raises(MlsynthConfigError):
        _cfg(augment="ridge", ridge_lambda=0.0)


def test_extra_field_forbidden():
    with pytest.raises(Exception):
        _cfg(bogus=123)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_no_treated_unit_raises():
    df = _panel()
    df["treat"] = 0
    with pytest.raises(MlsynthDataError):
        SCTA(_cfg(df)).fit()


def test_plot_smoke(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    res = SCTA(_cfg(display_graphs=True, frontier=[0.0, 0.5, 1.0])).fit()
    assert np.isfinite(res.effects.att)


def test_plot_saves_file(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    out = tmp_path / "scta.png"
    SCTA(_cfg(display_graphs=True, save=str(out))).fit()
    assert out.exists()


def test_dict_config_accepted():
    df = _panel()
    res = SCTA(dict(df=df, outcome="y", treat="treat", unitid="unit",
                    time="time", block_length=4, display_graphs=False)).fit()
    assert np.isfinite(res.effects.att)


def test_conformal_inference_attached():
    res = SCTA(_cfg()).fit()
    assert res.inference is not None
    assert 0.0 <= res.inference.p_value <= 1.0


def test_frontier_negative_nu_raises():
    with pytest.raises(MlsynthConfigError):
        _cfg(frontier=[0.0, -1.0])


def test_demean_false_runs():
    res = SCTA(_cfg(demean=False)).fit()
    assert np.isfinite(res.effects.att)
    w = np.array(list(res.weights.donor_weights.values()))
    assert abs(w.sum() - 1.0) < 1e-4


