"""Tests for COMPSC (Compositional Synthetic Controls).

The estimator is validated three ways here:

* recovery -- on a factor DGP satisfying the paper's Assumptions 1-3 exactly,
  the fitted weights return the convex combination that generated the treated
  unit;
* closed-form oracle -- the counterfactual is checked against the closure of the
  weighted geometric mean of the donor compositions (paper eq. 16), computed
  independently of the pipeline;
* invariants -- effects sum to zero, the counterfactual is a valid composition,
  and the ``clr`` geometry is invariant to relabelling the categories while the
  paper's ``alr`` geometry is not.

The full Path A replication (Pennsylvania AEPS, Boussim 2026 Tables 1-2) lives
in the benchmark suite, not here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mlsynth import COMPSC
from mlsynth.config_models import COMPSCConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.compsc_helpers import (
    aitchison_distance,
    alr,
    closure,
    clr,
)

CATS = ["c1", "c2", "c3"]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _panel(J=9, T=12, K=3, T0=8, w_true=None, seed=3, treated_noise=0.0):
    """Compositional panel from a factor model on relative utilities.

    With ``w_true`` supplied the treated unit's log-odds are exactly the convex
    combination of the donors', so Assumption 3 holds and the weights are
    identified (the donors carry idiosyncratic shocks, keeping them full rank).
    """
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, 3, K - 1))
    mu = rng.standard_normal((J, 3))
    V = np.einsum("tfk,jf->jtk", F, mu) + 0.3 * rng.standard_normal((J, T, K - 1))
    if w_true is not None:
        V[0] = np.tensordot(w_true, V[1:], axes=(0, 0))
        if treated_noise:
            V[0] = V[0] + treated_noise * rng.standard_normal((T, K - 1))
    P = closure(np.exp(np.concatenate([V, np.zeros((J, T, 1))], axis=-1)))

    recs = []
    for j in range(J):
        for t in range(T):
            rec = {"unit": f"u{j:02d}", "time": t,
                   "treat": int(j == 0 and t >= T0)}
            for k in range(K):
                rec[CATS[k]] = float(P[j, t, k])
            recs.append(rec)
    return pd.DataFrame(recs), P, T0


def _fit(df=None, **over):
    if df is None:
        df, _, _ = _panel()
    cfg = {"df": df, "outcomes": CATS, "treat": "treat", "unitid": "unit",
           "time": "time", "display_graphs": False}
    cfg.update(over)
    return COMPSC(cfg).fit()


# --------------------------------------------------------------------------- #
# recovery
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("geometry", ["clr", "alr"])
def test_recovers_generating_weights(geometry):
    """Assumption 3 holding exactly => the fitted weights are w*."""
    w_true = np.zeros(8)
    w_true[[0, 3, 5]] = [0.5, 0.3, 0.2]
    df, _, _ = _panel(w_true=w_true)
    res = _fit(df, geometry=geometry)
    w = np.array([res.donor_weights[f"u{j:02d}"] for j in range(1, 9)])
    assert np.abs(w - w_true).max() < 1e-6


def test_perfect_fit_gives_zero_pre_treatment_distance():
    w_true = np.zeros(8)
    w_true[[1, 2]] = [0.6, 0.4]
    df, _, T0 = _panel(w_true=w_true)
    res = _fit(df)
    assert res.aitchison_distance[:T0].max() < 1e-6


# --------------------------------------------------------------------------- #
# closed-form oracle: eq (16)
# --------------------------------------------------------------------------- #
def test_counterfactual_is_closure_of_weighted_geometric_mean():
    """Independent recomputation of paper eq. (16)."""
    df, P, _ = _panel()
    res = _fit(df)
    w = np.array([res.donor_weights[f"u{j:02d}"] for j in range(1, 9)])
    expected = closure(np.prod(P[1:] ** w[:, None, None], axis=0))
    assert np.allclose(res.counterfactual_composition, expected, atol=1e-8)


def test_log_ratio_effects_match_definition():
    """LRTE_{kl,t} = log(pi_k/pi_l) - log(cf_k/cf_l), paper eq. (3)."""
    df, _, _ = _panel()
    res = _fit(df)
    obs, cf = res.observed_composition, res.counterfactual_composition
    for k, l in [(0, 1), (0, 2), (2, 1)]:
        want = np.log(obs[:, k] / obs[:, l]) - np.log(cf[:, k] / cf[:, l])
        assert np.allclose(res.log_ratio_effects[:, k, l], want, atol=1e-12)
    # antisymmetric, zero on the diagonal
    assert np.allclose(res.log_ratio_effects, -res.log_ratio_effects.transpose(0, 2, 1))
    assert np.allclose(np.diagonal(res.log_ratio_effects, axis1=1, axis2=2), 0.0)


# --------------------------------------------------------------------------- #
# structural invariants
# --------------------------------------------------------------------------- #
def test_counterfactual_is_a_valid_composition():
    res = _fit()
    cf = res.counterfactual_composition
    assert (cf > 0).all()
    assert np.abs(cf.sum(axis=1) - 1.0).max() < 1e-12


def test_share_effects_sum_to_zero():
    """What one category gains the others jointly lose (paper eq. 2)."""
    res = _fit()
    assert np.abs(res.share_effects.sum(axis=1)).max() < 1e-12
    assert abs(res.sum_constraint) < 1e-12


def test_weights_are_on_the_simplex():
    res = _fit()
    w = np.array(list(res.donor_weights.values()))
    assert (w >= -1e-12).all()
    assert abs(w.sum() - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_clr_weights_invariant_to_category_relabelling():
    """clr is the isometry for the Aitchison metric, so the fit cannot depend
    on the order or the choice of a reference category."""
    df, _, _ = _panel()
    base = _fit(df, geometry="clr")
    shuffled = df.rename(columns={"c1": "c3", "c3": "c1"})
    perm = _fit(shuffled, outcomes=["c3", "c2", "c1"], geometry="clr")
    for j in range(1, 9):
        assert base.donor_weights[f"u{j:02d}"] == pytest.approx(
            perm.donor_weights[f"u{j:02d}"], abs=1e-8)


def test_alr_weights_depend_on_the_baseline_category():
    """The paper's alr map is not the Aitchison isometry: with imperfect
    pre-treatment fit the weights move with the arbitrary baseline choice.
    Pinned so the diagnostic below stays meaningful."""
    w_true = np.zeros(8)
    w_true[[0, 3, 5]] = [0.5, 0.3, 0.2]
    df, _, _ = _panel(w_true=w_true, treated_noise=0.8)
    ws = []
    for b in CATS:
        r = _fit(df, geometry="alr", baseline=b)
        ws.append([r.donor_weights[f"u{j:02d}"] for j in range(1, 9)])
    spread = np.ptp(np.asarray(ws), axis=0).max()
    assert spread > 1e-3


def test_clr_fits_the_aitchison_metric_at_least_as_well_as_alr():
    """clr minimizes the pre-treatment Aitchison distance by construction."""
    df, _, _ = _panel(treated_noise=0.5)
    a = _fit(df, geometry="clr").pre_treatment_aitchison_rmspe
    for b in CATS:
        assert a <= _fit(df, geometry="alr", baseline=b
                         ).pre_treatment_aitchison_rmspe + 1e-9


def test_binary_composition_has_no_baseline_sensitivity():
    """With two categories there is a single log-ratio, so every baseline gives
    the same fit and the diagnostic is not defined."""
    df, _, _ = _panel(K=2)
    res = _fit(df, outcomes=["c1", "c2"])
    assert res.baseline_sensitivity is None
    assert len(res.categories) == 2


def test_counterfactual_is_the_same_under_both_geometries_given_weights():
    """Given w, eq. (16) is symmetric in the categories; only the estimated
    weights are baseline-dependent."""
    w_true = np.zeros(8)
    w_true[[2, 4]] = [0.7, 0.3]
    df, _, _ = _panel(w_true=w_true)
    a = _fit(df, geometry="clr").counterfactual_composition
    b = _fit(df, geometry="alr", baseline="c2").counterfactual_composition
    assert np.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def test_aitchison_distance_matches_the_pairwise_definition():
    """Paper eq. (4): the double sum over all log-ratio pairs."""
    rng = np.random.default_rng(0)
    x, y = rng.dirichlet(np.ones(5), 2)
    p = len(x)
    want = math.sqrt(sum(
        (math.log(x[i] / x[j]) - math.log(y[i] / y[j])) ** 2
        for i in range(p) for j in range(p)) / (2 * p))
    assert aitchison_distance(x, y) == pytest.approx(want, rel=1e-12)


def test_alr_is_not_an_isometry_but_clr_is():
    """Documents the correction: ||alr(x)-alr(y)|| != d_A(x,y) in general."""
    rng = np.random.default_rng(1)
    x, y = rng.dirichlet(np.ones(5), 2)
    d = aitchison_distance(x, y)
    assert np.linalg.norm(clr(x) - clr(y)) == pytest.approx(d, rel=1e-12)
    assert not np.isclose(np.linalg.norm(alr(x) - alr(y)), d, rtol=1e-6)


def test_closure_renormalizes():
    x = np.array([2.0, 3.0, 5.0])
    assert np.allclose(closure(x), [0.2, 0.3, 0.5])


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def test_placebo_inference_reports_a_valid_p_value():
    res = _fit(inference="placebo")
    p = res.inference.p_value
    assert 0.0 < p <= 1.0
    assert res.placebo.n_retained >= 1
    assert res.placebo.treated_ratio > 0
    # p-value is a proportion over the retained set
    assert res.placebo.n_retained * p == pytest.approx(
        round(res.placebo.n_retained * p), abs=1e-9)


def test_inference_none_skips_the_placebo():
    res = _fit(inference="none")
    assert res.placebo is None
    assert res.inference is None


def test_placebo_screen_drops_poorly_fitting_donors():
    loose = _fit(inference="placebo", placebo_screen=1e9).placebo.n_retained
    tight = _fit(inference="placebo", placebo_screen=1.0).placebo.n_retained
    assert tight <= loose


# --------------------------------------------------------------------------- #
# result contract
# --------------------------------------------------------------------------- #
def test_result_populates_the_effect_contract():
    from mlsynth.config_models import BaseEstimatorResults

    res = _fit()
    assert isinstance(res, BaseEstimatorResults)
    assert res.effects is not None and res.time_series is not None
    assert res.weights is not None and res.fit_diagnostics is not None
    assert res.method_details.method_name.startswith("COMPSC")
    assert np.isfinite(res.att)
    assert len(res.counterfactual) == len(res.observed_composition)


def test_target_drives_the_flat_accessors():
    res = _fit(target="c2")
    k = CATS.index("c2")
    assert res.target == "c2"
    assert np.allclose(res.counterfactual, res.counterfactual_composition[:, k])


def test_categories_are_reported_in_outcome_order():
    res = _fit()
    assert [c.name for c in res.categories] == CATS
    for k, c in enumerate(res.categories):
        assert np.allclose(c.counterfactual, res.counterfactual_composition[:, k])


# --------------------------------------------------------------------------- #
# config failures
# --------------------------------------------------------------------------- #
def test_rejects_fewer_than_two_categories():
    df, _, _ = _panel()
    with pytest.raises(MlsynthConfigError):
        COMPSC({"df": df, "outcomes": ["c1"], "treat": "treat",
                "unitid": "unit", "time": "time"})


def test_rejects_unknown_category_column():
    df, _, _ = _panel()
    with pytest.raises(MlsynthConfigError):
        COMPSC({"df": df, "outcomes": ["c1", "nope"], "treat": "treat",
                "unitid": "unit", "time": "time"})


def test_rejects_duplicate_categories():
    df, _, _ = _panel()
    with pytest.raises(MlsynthConfigError):
        COMPSC({"df": df, "outcomes": ["c1", "c1", "c2"], "treat": "treat",
                "unitid": "unit", "time": "time"})


def test_rejects_baseline_outside_the_composition():
    df, _, _ = _panel()
    with pytest.raises(MlsynthConfigError):
        COMPSC({"df": df, "outcomes": CATS, "treat": "treat", "unitid": "unit",
                "time": "time", "geometry": "alr", "baseline": "c9"})


def test_rejects_target_outside_the_composition():
    df, _, _ = _panel()
    with pytest.raises(MlsynthConfigError):
        COMPSC({"df": df, "outcomes": CATS, "treat": "treat", "unitid": "unit",
                "time": "time", "target": "zzz"})


def test_rejects_unknown_geometry():
    df, _, _ = _panel()
    with pytest.raises(MlsynthConfigError):
        COMPSC({"df": df, "outcomes": CATS, "treat": "treat", "unitid": "unit",
                "time": "time", "geometry": "euclidean"})


def test_rejects_extra_fields():
    df, _, _ = _panel()
    with pytest.raises(MlsynthConfigError):
        COMPSC({"df": df, "outcomes": CATS, "treat": "treat", "unitid": "unit",
                "time": "time", "not_a_field": 1})


def test_rejects_negative_placebo_screen():
    df, _, _ = _panel()
    with pytest.raises(MlsynthConfigError):
        COMPSC({"df": df, "outcomes": CATS, "treat": "treat", "unitid": "unit",
                "time": "time", "placebo_screen": -1.0})


# --------------------------------------------------------------------------- #
# data failures
# --------------------------------------------------------------------------- #
def test_rejects_multiple_treated_units():
    df, _, T0 = _panel()
    df = df.copy()
    df.loc[(df.unit == "u01") & (df.time >= T0), "treat"] = 1
    with pytest.raises(MlsynthDataError, match="single treated unit"):
        _fit(df)


def test_rejects_treatment_in_the_first_period():
    df, _, _ = _panel()
    df = df.copy()
    df.loc[df.unit == "u00", "treat"] = 1
    with pytest.raises(MlsynthDataError):
        _fit(df)


def test_rejects_negative_shares():
    df, _, _ = _panel()
    df = df.copy()
    df.loc[0, "c1"] = -0.1
    with pytest.raises(MlsynthDataError, match="non-negative"):
        _fit(df)


def test_rejects_zero_shares_when_replacement_is_disabled():
    df, _, _ = _panel()
    df = df.copy()
    df.loc[0, "c1"] = 0.0
    with pytest.raises(MlsynthDataError, match="zero"):
        _fit(df, zero_replacement=0.0)


def test_zero_replacement_rescues_a_structural_zero():
    """Assumption 1 relaxed the standard CoDA way (Algorithm 1, Step 1)."""
    df, _, _ = _panel()
    df = df.copy()
    df.loc[0, "c1"] = 0.0
    res = _fit(df, zero_replacement=1e-6)
    assert np.isfinite(res.att)
    assert (res.counterfactual_composition > 0).all()


def test_rejects_a_row_that_is_entirely_zero():
    df, _, _ = _panel()
    df = df.copy()
    df.loc[0, CATS] = 0.0
    with pytest.raises(MlsynthDataError):
        _fit(df)


def test_rejects_missing_values():
    df, _, _ = _panel()
    df = df.copy()
    df.loc[0, "c1"] = np.nan
    with pytest.raises(MlsynthDataError, match="Missing values"):
        _fit(df)


def test_rejects_non_finite_shares():
    df, _, _ = _panel()
    df = df.copy()
    df.loc[0, "c1"] = np.inf
    with pytest.raises(MlsynthDataError, match="non-finite"):
        _fit(df)


def test_rejects_a_panel_with_no_donors():
    df, _, _ = _panel(J=1)
    with pytest.raises(MlsynthDataError):
        _fit(df)


def test_unnormalized_rows_are_closed_automatically():
    """Compositions are scale-free: doubling a unit's row changes nothing."""
    df, _, _ = _panel()
    scaled = df.copy()
    mask = scaled.unit == "u03"
    scaled.loc[mask, CATS] = scaled.loc[mask, CATS] * 7.0
    a = _fit(df).donor_weights
    b = _fit(scaled).donor_weights
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-8)


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def test_plotting_smoke(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    res = _fit(display_graphs=True)
    assert res is not None


def test_plot_labels_the_alr_baseline(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    res = _fit(display_graphs=True, geometry="alr", baseline="c2")
    assert res.baseline == "c2"


def test_plot_saves_to_a_path(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    out = tmp_path / "compsc.png"
    _fit(display_graphs=True, save=str(out))
    assert out.exists()
