"""SNN's anchor cross nests the SI and PCR-RSC estimators.

SNN (Agarwal, Dahleh, Shah & Shen), SI (Agarwal, Shah & Shen) and RSC (Amjad,
Shah & Shen) share one kernel: truncate the SVD of a fully observed block,
regress the target onto it, apply the weights out of sample. They differ only
in which block the design hands them. On the block missingness of a
comparative case study the three blocks coincide -- anchor rows are the donor
pool, anchor columns are the pre-periods -- so at a matched truncation rank the
three estimators are one estimator and must return the same counterfactual.

These tests pin that identity inside mlsynth, on a planted low-rank panel and
without a network. The empirical companion is the ``snn_nesting`` benchmark,
which runs the same comparison on the three Shen-Ding-Sekhon-Yu case studies
against those authors' own PCR code.

An equality test that cannot fail establishes nothing, so the controls below
carry equal weight: perturbing the anchor cross, or the rank, has to break the
identity. Each control asserts a floor on the disagreement it creates.

Layered per agents/agents_tests.md:

* smoke -- the three estimators fit the panel and return finite paths.
* unit invariants -- equality at matched rank; the row-side/column-side
  identity; the anchor search recovering the donor x pre-period block.
* edge -- a single donor, and a pre-period barely long enough to identify the
  rank.
* failure -- a target row with no observed anchor columns is reported
  infeasible, not silently imputed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import CLUSTERSC, SI, SNN
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.pcr import pcr_weights
from mlsynth.utils.snn_helpers.completion import _find_anchors, snn_predict

RANK = 3
N_DONORS = 20
T0 = 15
T1 = 8


def _panel(n_donors=N_DONORS, T0=T0, T1=T1, r=RANK, effect=3.0, noise=0.05, seed=7):
    """One treated unit, block missingness, a planted rank-``r`` signal."""
    rng = np.random.default_rng(seed)
    N, T = n_donors + 1, T0 + T1
    Y = (rng.standard_normal((N, r)) @ rng.standard_normal((r, T))
         + rng.standard_normal((N, T)) * noise)
    Y[0, T0:] += effect                      # unit 0 is the treated unit
    units = [f"u{i}" for i in range(N)]
    rows = [
        {"unit": units[i], "time": t, "y": float(Y[i, t]),
         "treat": int(i == 0 and t >= T0), "control_arm": int(i > 0)}
        for i in range(N) for t in range(T)
    ]
    return pd.DataFrame(rows), Y


def _cfg(df):
    return {"df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
            "time": "time", "display_graphs": False}


def _paths(df, rank=RANK):
    """The three estimators' post-period counterfactuals at a matched rank."""
    cfg = _cfg(df)
    obs = (df[(df.unit == "u0") & (df.treat == 1)]
           .sort_values("time")["y"].to_numpy(float))

    snn = SNN({**cfg, "max_rank": rank, "clip": False}).fit()
    post_times = sorted(df.loc[df.treat == 1, "time"].unique())
    cf_snn = obs - np.array([snn.att_by_period[t] for t in post_times])

    si = SI({**cfg, "inters": ["control_arm"], "rank_method": "fixed",
             "rank": rank, "bias_correct": False}).fit()
    cf_si = np.asarray(si.arms["control_arm"].counterfactual, float)[-len(obs):]

    rsc = CLUSTERSC({**cfg, "method": "pcr", "clustering": False,
                     "pcr_objective": "OLS", "rank": rank,
                     "rank_method": "fixed", "standardize_for_rank": False}).fit()
    cf_rsc = np.asarray(rsc.counterfactual, float)[-len(obs):]
    return cf_snn, cf_si, cf_rsc


def _rel(a, b):
    """Max absolute gap, scaled by the level of the paths being compared."""
    scale = max(np.abs(a).mean(), np.abs(b).mean(), 1e-12)
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / scale)


# --------------------------------------------------------------------- smoke
def test_the_three_estimators_fit_and_return_finite_paths():
    df, _ = _panel()
    for path in _paths(df):
        assert path.shape == (T1,)
        assert np.isfinite(path).all()


# ----------------------------------------------------------- unit invariants
def test_snn_equals_si_at_a_matched_rank():
    """SI's cross (donor pool, pre-periods) is the cross SNN finds itself."""
    df, _ = _panel()
    cf_snn, cf_si, _ = _paths(df)
    assert _rel(cf_snn, cf_si) < 1e-10


def test_snn_equals_pcr_rsc_at_a_matched_rank():
    df, _ = _panel()
    cf_snn, _, cf_rsc = _paths(df)
    assert _rel(cf_snn, cf_rsc) < 1e-10


def test_the_identity_holds_across_seeds_and_shapes():
    """Not a property of one draw: vary the seed, the donor count and T0."""
    for seed, n_donors, t0 in [(1, 12, 10), (2, 30, 20), (3, 8, 25)]:
        df, _ = _panel(n_donors=n_donors, T0=t0, seed=seed)
        cf_snn, cf_si, cf_rsc = _paths(df)
        assert _rel(cf_snn, cf_si) < 1e-10, (seed, n_donors, t0)
        assert _rel(cf_snn, cf_rsc) < 1e-10, (seed, n_donors, t0)


def test_row_side_and_column_side_syntheses_agree():
    """SNN section 3.1: <Y_{AR,j}, beta> = <Y_{i,AC}, alpha>, citing SDSY23.

    The row side synthesises the target unit from the anchor rows; the column
    side synthesises the target period from the anchor columns. Same rank, same
    block, same number.
    """
    _, Y = _panel()
    S = Y[1:, :T0]                       # anchor block: donors x pre-periods
    q = Y[0, :T0]                        # target row on the anchor columns
    for t in range(T0, T0 + T1):
        x = Y[1:, t]                     # target column on the anchor rows
        row_side = float(x @ pcr_weights(S.T, q, RANK))
        col_side = float(q @ pcr_weights(S, x, RANK))
        assert abs(row_side - col_side) < 1e-8 * max(abs(row_side), 1.0)


def test_the_anchor_search_recovers_the_donor_by_pre_period_block():
    """On block missingness SNN's own search returns the synthetic-control cross."""
    _, Y = _panel()
    mask = np.ones(Y.shape, dtype=int)
    mask[0, T0:] = 0
    AR, AC = _find_anchors(mask, 0, T0)
    assert sorted(AR) == list(range(1, N_DONORS + 1))
    assert sorted(AC) == list(range(T0))


# ------------------------------------------------- controls (falsifiability)
def test_shrinking_the_anchor_cross_breaks_the_identity():
    """Drop rows and columns from the cross and the estimate has to move.

    Without this the equality tests would pass on an estimator that ignored its
    anchor sets entirely.
    """
    _, Y = _panel()
    S, q = Y[1:, :T0], Y[0, :T0]
    keep_r, keep_c = np.arange(N_DONORS - 4), np.arange(4, T0)
    gaps = []
    for t in range(T0, T0 + T1):
        x = Y[1:, t]
        full = float(x @ pcr_weights(S.T, q, RANK))
        shrunk = float(x[keep_r] @ pcr_weights(S[np.ix_(keep_r, keep_c)].T,
                                               q[keep_c], RANK))
        gaps.append(abs(full - shrunk))
    assert max(gaps) > 1e-6


def test_a_mismatched_rank_breaks_the_identity():
    """The equality is at a *matched* rank; it is not rank-free."""
    df, _ = _panel()
    cf_snn, _, _ = _paths(df)
    obs = (df[(df.unit == "u0") & (df.treat == 1)]
           .sort_values("time")["y"].to_numpy(float))
    si_wrong = SI({**_cfg(df), "inters": ["control_arm"], "rank_method": "fixed",
                   "rank": RANK + 2, "bias_correct": False}).fit()
    cf_wrong = np.asarray(si_wrong.arms["control_arm"].counterfactual,
                          float)[-len(obs):]
    assert _rel(cf_snn, cf_wrong) > 1e-6


# ----------------------------------------------------------------------- edge
def test_the_smallest_donor_pool_snn_accepts_still_nests():
    """|AR| = 2, the narrowest cross SNN ingests, and the identity survives."""
    df, _ = _panel(n_donors=2, r=1)
    cf_snn, cf_si, cf_rsc = _paths(df, rank=1)
    assert _rel(cf_snn, cf_si) < 1e-10
    assert _rel(cf_snn, cf_rsc) < 1e-10


def test_a_pre_period_as_short_as_the_rank_still_nests():
    """T0 = r: the anchor block is exactly identified, with no slack."""
    df, _ = _panel(T0=RANK, T1=4)
    cf_snn, cf_si, cf_rsc = _paths(df)
    assert _rel(cf_snn, cf_si) < 1e-10
    assert _rel(cf_snn, cf_rsc) < 1e-10


# -------------------------------------------------------------------- failure
def test_a_target_row_with_no_anchor_columns_is_reported_infeasible():
    """No observed anchor column means no cross, so no estimate exists.

    The failure has to surface: ``snn_predict`` returns ``feasible=False`` and a
    NaN, never a number derived from an empty block.
    """
    _, Y = _panel()
    mask = np.ones(Y.shape, dtype=int)
    mask[0, :] = 0                       # target row observed nowhere
    value, feasible = snn_predict(Y, mask, 0, T0)
    assert feasible is False
    assert np.isnan(value)


def test_a_lone_donor_is_refused_at_ingestion_not_nested_silently():
    """SNN needs three units; a two-unit panel raises instead of estimating.

    The nesting is a statement about the shared kernel, not a promise that every
    entry point accepts every panel SI and PCR-RSC accept. SI and PCR-RSC both
    fit a one-donor panel; SNN declines it, and the decline is a translated
    ``MlsynthDataError``, not a counterfactual built from one row.
    """
    df, _ = _panel(n_donors=1, r=1)
    with pytest.raises(MlsynthDataError, match="at least 3 units"):
        SNN({**_cfg(df), "max_rank": 1, "clip": False}).fit()


def test_an_empty_anchor_block_yields_empty_anchor_sets():
    """The search reports the absence instead of returning a partial block."""
    mask = np.ones((5, 5), dtype=int)
    mask[:, 2] = 0                       # nobody observed at the target column
    AR, AC = _find_anchors(mask, 0, 2)
    assert AR.size == 0 and AC.size == 0
