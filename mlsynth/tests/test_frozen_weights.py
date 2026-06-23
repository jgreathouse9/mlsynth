"""Tests for freeze / apply_frozen_weights -- the commission/refresh mechanic.

Test-first per CLAUDE.md. A fitted SCM result is frozen to a small serializable
artifact (donor-weight matrix + pre-period residuals); applying it to an extended
panel reproduces the counterfactual and re-runs inference WITHOUT a refit, and
returns a standard EffectResult so a refresh flows through the same tooling as a
fit. For the pre-period-weighting family (MSQRT) the reapplication is exact.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth import MSQRT
from mlsynth.config_models import BaseEstimatorResults
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.frozen_weights import (
    FrozenWeights,
    apply_frozen_weights,
    freeze,
)


# --------------------------------------------------------------------------- #
# A factor panel where treated units live in the donors' span (so SCM fits)
# --------------------------------------------------------------------------- #
def _panel(n_c=20, n_t=3, T0=30, post=12, tau=-6.0, seed=1):
    rng = np.random.default_rng(seed)
    T = T0 + post
    F = np.cumsum(rng.standard_normal((T, 3)), axis=0) * 0.5
    L_c = rng.uniform(0.2, 1.2, size=(n_c, 3))
    L_t = rng.dirichlet(np.ones(n_c), size=n_t) @ L_c
    Y = np.vstack([L_c @ F.T + 0.25 * rng.standard_normal((n_c, T)),
                   L_t @ F.T + 0.25 * rng.standard_normal((n_t, T))])
    Y[n_c:, T0:] += tau
    units = [f"c{i:02d}" for i in range(n_c)] + [f"t{j}" for j in range(n_t)]
    rows = [{"geo": u, "week": t, "sales": Y[i, t],
             "treat": int(u.startswith("t") and t >= T0)}
            for i, u in enumerate(units) for t in range(T)]
    return pd.DataFrame(rows), T0, T


MAP = dict(outcome="sales", unitid="geo", time="week")


def _fit(df):
    return MSQRT({**MAP, "df": df, "treat": "treat",
                 "inference": True, "display_graphs": False}).fit()


@pytest.fixture(scope="module")
def commissioned():
    """Fit on pre + first 4 post weeks; return (frozen, full panel, T0, T)."""
    df, T0, T = _panel()
    commission = df[df["week"] <= T0 + 3]            # pre + 4 post weeks
    frozen = freeze(_fit(commission))
    return frozen, df, T0, T


# --------------------------------------------------------------------------- #
# freeze
# --------------------------------------------------------------------------- #
def test_freeze_returns_artifact(commissioned):
    frozen, _, T0, _ = commissioned
    assert isinstance(frozen, FrozenWeights)
    assert frozen.estimator == "MSQRT"
    assert frozen.weights.shape == (20, 3)           # (n_donors, m_treated)
    assert frozen.donor_names[:1] == ["c00"]
    assert frozen.treated_names == ["t0", "t1", "t2"]
    assert frozen.pre_residuals.shape == (T0, 3)


def test_freeze_serializable_roundtrip(commissioned):
    frozen, _, _, _ = commissioned
    blob = frozen.model_dump_json()
    back = FrozenWeights.model_validate_json(blob)
    assert np.allclose(back.weights, frozen.weights)
    assert np.allclose(back.pre_residuals, frozen.pre_residuals)
    assert back.donor_names == frozen.donor_names


def test_freeze_rejects_non_msqrt():
    with pytest.raises(MlsynthConfigError):
        freeze(BaseEstimatorResults())               # no weights/inputs


# --------------------------------------------------------------------------- #
# apply: returns a standard EffectResult, no refit
# --------------------------------------------------------------------------- #
def test_apply_returns_effect_result(commissioned):
    frozen, df, T0, T = commissioned
    res = apply_frozen_weights(frozen, df, **MAP)
    assert isinstance(res, BaseEstimatorResults)
    assert isinstance(res.att, float) and np.isfinite(res.att)
    cf = np.asarray(res.counterfactual)
    assert cf.shape[0] == T                           # full timeline
    assert res.att_ci is not None and res.att_ci[0] <= res.att_ci[1]


def test_apply_matches_manual_gap(commissioned):
    frozen, df, T0, T = commissioned
    res = apply_frozen_weights(frozen, df, **MAP)
    wide = df.pivot(index="geo", columns="week", values="sales").sort_index(axis=1)
    X = wide.loc[frozen.donor_names].to_numpy().T
    Y = wide.loc[frozen.treated_names].to_numpy().T
    synth = X @ frozen.weights
    post = np.arange(T) >= T0
    assert res.att == pytest.approx(float((Y - synth)[post].mean()))


def test_apply_is_exact_vs_refit(commissioned):
    """Frozen reapply == a full refit at the same horizon (weights are pre-only)."""
    frozen, df, T0, T = commissioned
    frozen_res = apply_frozen_weights(frozen, df, **MAP)
    refit = _fit(df)                                  # refit on the full panel
    assert frozen_res.att == pytest.approx(float(refit.att), rel=1e-6)


def test_apply_missing_donor_raises(commissioned):
    frozen, df, _, _ = commissioned
    dropped = df[df["geo"] != frozen.donor_names[0]]
    with pytest.raises(MlsynthDataError):
        apply_frozen_weights(frozen, dropped, **MAP)


def test_apply_no_post_period_raises(commissioned):
    frozen, df, T0, _ = commissioned
    pre_only = df[df["week"] < T0]                    # nothing at/after intervention
    with pytest.raises(MlsynthDataError):
        apply_frozen_weights(frozen, pre_only, **MAP)


def test_apply_missing_column_raises(commissioned):
    frozen, df, _, _ = commissioned
    with pytest.raises(MlsynthDataError):
        apply_frozen_weights(frozen, df.rename(columns={"sales": "amt"}), **MAP)


def test_refresh_history_keeps_weights_fixed(commissioned):
    """Refresh at growing horizons reuses identical weights; CI tightens."""
    frozen, df, T0, T = commissioned
    widths = []
    for h in (4, 8, 12):
        sub = df[df["week"] <= T0 + h - 1]
        res = apply_frozen_weights(frozen, sub, **MAP)
        lo, hi = res.att_ci
        widths.append(hi - lo)
    assert widths[0] > widths[-1]                     # more post periods -> tighter
