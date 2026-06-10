"""End-to-end: LEXSCM spillover-aware selection on the bundled Nielsen DMA map.

Validates the ``basedata/markets/`` artifact *and* the spillover feature together
on real US market-area contiguity. We restrict to the Southeast, synthesise a
spatially-correlated panel over that real border structure, and check the two
guarantees LEXSCM must enforce when given an ``adjacency`` matrix:

* no two **treated** DMAs share a border (Stage-1 "No interference");
* no **donor** borders any treated DMA (Stage-2 "exclusion restriction").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import LEXSCM

_MARKETS = Path(__file__).resolve().parents[2] / "basedata" / "markets"
_SE = {"FL", "GA", "AL", "MS", "SC", "NC", "TN"}


@pytest.fixture(scope="module")
def dma_adjacency():
    adj = pd.read_csv(_MARKETS / "dma_adjacency.csv", index_col=0)
    return adj


@pytest.fixture(scope="module")
def dma_metadata():
    return pd.read_csv(_MARKETS / "dma_metadata.csv")


def test_bundled_dma_matrix_is_valid_contiguity(dma_adjacency):
    A = dma_adjacency
    assert A.shape[0] == A.shape[1] == 206
    assert list(A.index) == list(A.columns)          # name-indexed, aligned
    V = A.values
    assert set(np.unique(V)).issubset({0, 1})         # 0/1
    assert (V == V.T).all()                           # symmetric
    assert int(np.trace(V)) == 0                      # no self-borders
    assert V.sum(1).min() >= 1                         # fully connected (patched)
    # a known adjacency holds
    assert A.loc["Atlanta, GA", "Macon, GA"] == 1


def _se_panel(adj, meta, seed=3, m=4, n_cand=14, T=40, T_post=10):
    se = [n for n in meta.loc[meta.state.isin(_SE), "dma_name"] if n in adj.index]
    A = adj.loc[se, se]
    W = A.values.astype(float)
    n = len(se)
    rng = np.random.default_rng(seed)
    r = 3
    lam = rng.normal(size=(n, r))
    for _ in range(6):                                 # neighbours -> similar loadings
        lam = 0.4 * lam + 0.6 * (W @ lam) / np.maximum(W.sum(1, keepdims=True), 1)
    F = np.cumsum(rng.normal(size=(T, r)), 0)
    Y = 100 + rng.normal(0, 5, n) + F @ lam.T + rng.normal(0, 1.0, (T, n))
    cand = sorted(rng.choice(n, n_cand, replace=False).tolist())
    df = pd.DataFrame([
        {"unitid": nm, "time": t, "sales": Y[t, j],
         "candidate": int(j in cand), "post": int(t >= T - T_post)}
        for j, nm in enumerate(se) for t in range(T)
    ])
    return df, A


def test_spillover_aware_design_on_real_dma_borders(dma_adjacency, dma_metadata):
    df, A = _se_panel(dma_adjacency, dma_metadata)
    res = LEXSCM({
        "df": df, "outcome": "sales", "unitid": "unitid", "time": "time",
        "candidate_col": "candidate", "m": 4, "post_col": "post",
        "adjacency": A, "top_K": 10, "verbose": False,
    }).fit()

    treated = list(res.selected_units)
    donors = list(res.design_weights.donor_weights)
    assert len(treated) == 4

    # Guarantee 1: no two treated DMAs border each other.
    for i, a in enumerate(treated):
        for b in treated[i + 1:]:
            assert A.loc[a, b] == 0, f"treated DMAs {a!r} and {b!r} share a border"

    # Guarantee 2: no donor borders any treated DMA.
    treated_neighbours = {x for t in treated for x in A.columns[A.loc[t] == 1]}
    assert treated_neighbours.isdisjoint(donors), \
        "a donor DMA borders a treated DMA"
