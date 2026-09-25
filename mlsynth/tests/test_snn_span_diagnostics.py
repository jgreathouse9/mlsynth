"""SNN's two span diagnostics: the linear span and subspace inclusion tests.

SNN identifies an entry from a local anchor cross under two span conditions.
Assumption 3 asks the target row's latent factor to lie in the span of the
anchor rows'; Assumption 7 asks the target column's to lie in the span of the
anchor columns'. Neither is testable directly, because the factors are latent,
but both have an empirical counterpart the reference implementation
(``deshen24/syntheticNN``) computes and reports per entry:

* the linear span statistic, the normalized training error
  :math:`\\|S^\\top \\beta - q\\|^2 / \\|q\\|^2`, which measures whether the
  target row can be reconstructed from the anchor rows at all;
* the subspace inclusion statistic,
  :math:`\\|(I - V^\\top V) x\\|^2 / \\|x\\|^2` for :math:`V` the retained
  right singular directions of the anchor block, which measures whether the
  target column lies in the span the weights were fit on.

The reference calls an entry feasible when both fall at or below their
thresholds (0.1 each by default). These tests pin mlsynth's port of both
statistics against the reference's own formulas, on panels where the answer is
known by construction: a low-rank panel where both should vanish, and two
planted violations where one statistic has to fire and the other should not.

Reporting is separate from gating. The statistics are recorded; an entry that
fails them is still imputed, and ``feasible`` keeps its existing meaning (an
anchor cross existed and the value was finite). A test below pins that, because
silently dropping cells would change every ATT mlsynth already reports.

Layered per agents/agents_tests.md:

* smoke -- the fields exist, with the right shapes and support.
* unit invariants -- each statistic equals the reference formula; both vanish
  on an exactly low-rank panel; each planted violation fires its own statistic.
* edge -- a rank-1 block and a zero target column.
* failure -- invalid thresholds are refused by the config, and adding the
  diagnostics does not move the imputed values.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import SNN
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.pcr import hsvt, pcr_weights
from mlsynth.utils.snn_helpers.completion import snn_complete, snn_predict

T0, T1 = 14, 6
N_DONORS = 18


# ------------------------------------------------------------- reference forms
def _ref_span_error(S, q, beta):
    """``deshen24/syntheticNN`` ``_train_error(X1.T, y1, beta)``."""
    delta = np.linalg.norm(S.T @ beta - q)
    return float((delta / np.linalg.norm(q)) ** 2)


def _ref_subspace_stat(Vt_r, x):
    """``deshen24/syntheticNN`` ``_subspace_inclusion(v_rank, X2)``."""
    delta = (np.eye(Vt_r.shape[1]) - Vt_r.T @ Vt_r) @ x
    return float((np.linalg.norm(delta) / np.linalg.norm(x)) ** 2)


def _panel(n_donors=N_DONORS, t0=T0, t1=T1, r=3, effect=2.0, noise=0.0, seed=5):
    rng = np.random.default_rng(seed)
    N, T = n_donors + 1, t0 + t1
    Y = (rng.standard_normal((N, r)) @ rng.standard_normal((r, T))
         + rng.standard_normal((N, T)) * noise)
    Y[0, t0:] += effect
    rows = [{"unit": f"u{i}", "time": t, "y": float(Y[i, t]),
             "treat": int(i == 0 and t >= t0)}
            for i in range(N) for t in range(T)]
    return pd.DataFrame(rows), Y


def _cfg(df, **kw):
    return {"df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
            "time": "time", "display_graphs": False, **kw}


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SNN(_cfg(df, **kw)).fit()


# --------------------------------------------------------------------- smoke
def test_the_result_carries_both_statistics_on_the_imputed_cells():
    df, _ = _panel()
    res = _fit(df, max_rank=3)
    for mat in (res.span_error_matrix, res.subspace_stat_matrix):
        assert mat.shape == (N_DONORS + 1, T0 + T1)
    imputed = res.inputs.D > 0
    assert np.isfinite(res.span_error_matrix[imputed]).all()
    assert np.isfinite(res.subspace_stat_matrix[imputed]).all()
    # Observed cells carry no statistic: nothing was fit for them.
    assert np.isnan(res.span_error_matrix[~imputed]).all()
    assert np.isnan(res.subspace_stat_matrix[~imputed]).all()


def test_both_statistics_are_non_negative():
    df, _ = _panel(noise=0.2)
    res = _fit(df, max_rank=3)
    imputed = res.inputs.D > 0
    assert (res.span_error_matrix[imputed] >= 0).all()
    assert (res.subspace_stat_matrix[imputed] >= 0).all()


def test_method_details_summarize_the_span_tests():
    df, _ = _panel()
    res = _fit(df, max_rank=3)
    d = res.method_details.parameters_used
    for key in ("max_span_error", "max_subspace_stat", "n_span_test_failures",
                "linear_span_eps", "subspace_eps"):
        assert key in d, key


# ----------------------------------------------------------- unit invariants
def test_the_span_statistic_matches_the_reference_formula():
    df, Y = _panel(noise=0.3)
    res = _fit(df, max_rank=3)
    S, q = Y[1:, :T0], Y[0, :T0]
    beta = pcr_weights(S.T, q, 3)
    assert res.span_error_matrix[0, T0] == pytest.approx(
        _ref_span_error(S, q, beta), rel=1e-10)


def test_the_subspace_statistic_matches_the_reference_formula():
    df, Y = _panel(noise=0.3)
    res = _fit(df, max_rank=3)
    S = Y[1:, :T0]
    _, _, _, Vt_r = hsvt(S.T, 3)
    for offset, t in enumerate(range(T0, T0 + T1)):
        assert res.subspace_stat_matrix[0, t] == pytest.approx(
            _ref_subspace_stat(Vt_r, Y[1:, t]), rel=1e-10), offset


def test_both_statistics_vanish_on_an_exactly_low_rank_panel():
    """Noiseless rank 3, rank 3 retained: the cross explains the target exactly."""
    df, _ = _panel(r=3, noise=0.0)
    res = _fit(df, max_rank=3)
    imputed = res.inputs.D > 0
    assert res.span_error_matrix[imputed].max() < 1e-18
    assert res.subspace_stat_matrix[imputed].max() < 1e-18
    assert res.span_tests_passed[imputed].all()


def test_a_target_row_outside_the_anchor_row_span_fires_the_span_statistic():
    """Plant a pre-period the donors cannot reproduce, and only that test fails.

    The target column is untouched, so the subspace statistic must stay small:
    the two statistics have to separate the two assumptions, not co-move.
    """
    df, Y = _panel(r=2, noise=0.0)
    Y2 = Y.copy()
    rng = np.random.default_rng(11)
    Y2[0, :T0] += 3.0 * rng.standard_normal(T0)      # target row off the span
    rows = [{"unit": f"u{i}", "time": t, "y": float(Y2[i, t]),
             "treat": int(i == 0 and t >= T0)}
            for i in range(Y2.shape[0]) for t in range(Y2.shape[1])]
    res = _fit(pd.DataFrame(rows), max_rank=2)
    assert res.span_error_matrix[0, T0] > 0.1
    assert res.subspace_stat_matrix[0, T0] < 1e-12
    assert not res.span_tests_passed[0, T0]


def test_a_target_column_outside_the_anchor_column_span_fires_the_subspace_statistic():
    """Plant a post-period direction the anchor block never saw.

    The pre-period is untouched, so the span statistic must stay small.
    """
    df, Y = _panel(r=2, noise=0.0)
    Y2 = Y.copy()
    rng = np.random.default_rng(12)
    Y2[1:, T0] = rng.standard_normal(Y2.shape[0] - 1) * 5.0   # off-span column
    rows = [{"unit": f"u{i}", "time": t, "y": float(Y2[i, t]),
             "treat": int(i == 0 and t >= T0)}
            for i in range(Y2.shape[0]) for t in range(Y2.shape[1])]
    res = _fit(pd.DataFrame(rows), max_rank=2)
    assert res.subspace_stat_matrix[0, T0] > 0.1
    assert res.span_error_matrix[0, T0] < 1e-12
    assert not res.span_tests_passed[0, T0]


def test_the_thresholds_are_configurable():
    df, _ = _panel(noise=0.4)
    loose = _fit(df, max_rank=2, linear_span_eps=1.0, subspace_eps=1.0)
    tight = _fit(df, max_rank=2, linear_span_eps=1e-12, subspace_eps=1e-12)
    imputed = loose.inputs.D > 0
    assert loose.span_tests_passed[imputed].sum() >= tight.span_tests_passed[imputed].sum()
    assert not tight.span_tests_passed[imputed].any()


def test_snn_complete_can_return_the_diagnostics():
    """The engine exposes them for general matrix completion, not only panels."""
    _, Y = _panel()
    X = Y.copy()
    X[0, T0:] = np.nan
    completed, feasible, span, sub = snn_complete(X, max_rank=3,
                                                  return_diagnostics=True)
    assert span.shape == sub.shape == X.shape
    assert np.isfinite(span[0, T0:]).all()
    assert np.isnan(span[~np.isnan(X)]).all()
    assert completed.shape == X.shape and feasible.shape == X.shape


# ----------------------------------------------------------------------- edge
def test_a_rank_one_block_reports_both_statistics():
    df, _ = _panel(r=1, noise=0.05)
    res = _fit(df, max_rank=1)
    imputed = res.inputs.D > 0
    assert np.isfinite(res.span_error_matrix[imputed]).all()
    assert np.isfinite(res.subspace_stat_matrix[imputed]).all()


def test_a_zero_target_column_yields_a_defined_subspace_statistic():
    """``x = 0`` makes the reference's ratio 0/0; the port reports 0, not NaN."""
    _, Y = _panel(r=2, noise=0.0)
    X = Y.copy()
    X[1:, T0] = 0.0                       # anchor rows all zero at the target column
    X[0, T0:] = np.nan
    _, _, span, sub = snn_complete(X, max_rank=2, return_diagnostics=True)
    assert np.isfinite(sub[0, T0])
    assert sub[0, T0] == 0.0


def test_an_infeasible_entry_leaves_the_statistics_missing():
    _, Y = _panel()
    X = Y.copy()
    X[0, :] = np.nan                      # no anchor columns for the target row
    value, ok, diag = snn_predict(X, (~np.isnan(X)).astype(int), 0, T0,
                                  return_diagnostics=True)
    assert ok is False and np.isnan(value)
    assert np.isnan(diag["span_error"]) and np.isnan(diag["subspace_stat"])


# -------------------------------------------------------------------- failure
def test_a_non_positive_threshold_is_refused():
    df, _ = _panel()
    for bad in ({"linear_span_eps": 0.0}, {"subspace_eps": -1.0}):
        with pytest.raises(MlsynthConfigError):
            SNN(_cfg(df, **bad))


def test_reporting_the_statistics_does_not_change_the_imputed_values():
    """The guard: diagnostics are reported, never gated on.

    An entry failing both tests is still imputed and still counted in the ATT.
    If this ever changes, every ATT mlsynth reports moves, including the ones
    the snn_prop99 benchmark pins.
    """
    df, _ = _panel(noise=0.4)
    tight = _fit(df, max_rank=2, linear_span_eps=1e-12, subspace_eps=1e-12)
    loose = _fit(df, max_rank=2, linear_span_eps=1e6, subspace_eps=1e6)
    assert not tight.span_tests_passed[tight.inputs.D > 0].any()
    assert tight.att == pytest.approx(loose.att, rel=0, abs=0)
    np.testing.assert_array_equal(tight.counterfactual_matrix,
                                  loose.counterfactual_matrix)
    np.testing.assert_array_equal(tight.feasible, loose.feasible)


def test_a_failing_span_test_is_warned_about_not_swallowed():
    """A caller who ignores the fields still learns the assumption failed."""
    _, Y = _panel(r=2, noise=0.0)
    Y2 = Y.copy()
    rng = np.random.default_rng(13)
    Y2[0, :T0] += 5.0 * rng.standard_normal(T0)
    rows = [{"unit": f"u{i}", "time": t, "y": float(Y2[i, t]),
             "treat": int(i == 0 and t >= T0)}
            for i in range(Y2.shape[0]) for t in range(Y2.shape[1])]
    with pytest.warns(UserWarning, match="span"):
        SNN(_cfg(pd.DataFrame(rows), max_rank=2)).fit()
