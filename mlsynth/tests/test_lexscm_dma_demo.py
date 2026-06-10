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


# =====================================================================
# South & Midwest grouped-factor design: coverage / quota / size / budget,
# validating the worked docs example (incl. the budget interaction).
# =====================================================================

from mlsynth.exceptions import MlsynthConfigError

_DIVISION = {
    "FL": "S. Atlantic", "GA": "S. Atlantic", "NC": "S. Atlantic",
    "SC": "S. Atlantic", "VA": "S. Atlantic", "WV": "S. Atlantic", "MD": "S. Atlantic",
    "KY": "E.S. Central", "TN": "E.S. Central", "MS": "E.S. Central", "AL": "E.S. Central",
    "AR": "W.S. Central", "LA": "W.S. Central", "OK": "W.S. Central", "TX": "W.S. Central",
    "OH": "E.N. Central", "IN": "E.N. Central", "IL": "E.N. Central",
    "MI": "E.N. Central", "WI": "E.N. Central",
    "MN": "W.N. Central", "IA": "W.N. Central", "MO": "W.N. Central", "ND": "W.N. Central",
    "SD": "W.N. Central", "NE": "W.N. Central", "KS": "W.N. Central",
}


def _south_midwest(dma_adjacency, dma_metadata, seed=11, T=60, T_post=12):
    meta = dma_metadata[dma_metadata.state.isin(_DIVISION)].copy()
    meta["division"] = meta.state.map(_DIVISION)
    names = [n for n in meta.dma_name if n in dma_adjacency.index]
    meta = meta[meta.dma_name.isin(names)].reset_index(drop=True)
    A = dma_adjacency.loc[names, names]
    rng = np.random.default_rng(seed)
    n, r = len(names), 4
    div = meta.set_index("dma_name").loc[names, "division"].values
    Lam_div = {d: rng.normal(size=r) for d in sorted(set(div))}   # deterministic order
    Lam = np.array([Lam_div[d] for d in div]) + 0.15 * rng.normal(size=(n, r))
    F = np.cumsum(rng.normal(size=(T, r)), 0)
    pop = np.round(rng.lognormal(12.5, 0.8, n)).astype(int)
    Y = 100 + rng.normal(0, 5, n) + F @ Lam.T + rng.normal(0, 1.0, (T, n))
    cand = sorted(rng.choice(n, 18, replace=False).tolist())
    df = pd.DataFrame([
        {"market": names[j], "week": t, "sales": Y[t, j], "eligible": int(j in cand),
         "post": int(t >= T - T_post), "division": div[j],
         "population": int(pop[j]), "cost": float(pop[j] * 5.0)}
        for j in range(n) for t in range(T)
    ])
    return df, A, pop


class TestSouthMidwestBudget:
    _BASE = dict(outcome="sales", unitid="market", time="week",
                 candidate_col="eligible", post_col="post", top_K=8, verbose=False)

    def _divmap(self, df):
        return df.groupby("market")["division"].first()

    def test_budget_alone_respected(self, dma_adjacency, dma_metadata):
        df, _A, _pop = _south_midwest(dma_adjacency, dma_metadata)
        res = LEXSCM({"df": df, **self._BASE, "m": 4,
                      "unit_cost_col": "cost", "budget": 10_000_000}).fit()
        cmap = df.groupby("market")["cost"].first()
        assert sum(cmap.loc[u] for u in res.selected_units) <= 10_000_000

    def test_budget_plus_coverage(self, dma_adjacency, dma_metadata):
        df, _A, _pop = _south_midwest(dma_adjacency, dma_metadata)
        res = LEXSCM({"df": df, **self._BASE, "m": 5, "stratum_col": "division",
                      "min_per_stratum": 1, "unit_cost_col": "cost",
                      "budget": 10_000_000}).fit()
        cmap = df.groupby("market")["cost"].first()
        dmap = self._divmap(df)
        assert sum(cmap.loc[u] for u in res.selected_units) <= 10_000_000
        assert len({dmap.loc[u] for u in res.selected_units}) == 5     # all divisions

    def test_budget_size_coverage_conflict_raises(self, dma_adjacency, dma_metadata):
        df, _A, pop = _south_midwest(dma_adjacency, dma_metadata)
        # five above-median (expensive) markets cannot fit a $10M cap
        with pytest.raises(MlsynthConfigError):
            LEXSCM({"df": df, **self._BASE, "m": 5, "stratum_col": "division",
                    "min_per_stratum": 1, "size_col": "population",
                    "min_size": int(np.median(pop)), "unit_cost_col": "cost",
                    "budget": 10_000_000}).fit()

    def test_coverage_quota_compose(self, dma_adjacency, dma_metadata):
        # cover all five divisions (min_per=1) AND at most two per division
        # (max_per=2) at m=6 -> exactly one division carries two.
        df, _A, _pop = _south_midwest(dma_adjacency, dma_metadata)
        res = LEXSCM({"df": df, **self._BASE, "m": 6, "stratum_col": "division",
                      "min_per_stratum": 1, "max_per_stratum": 2}).fit()
        dmap = self._divmap(df)
        counts = pd.Series([dmap.loc[u] for u in res.selected_units]).value_counts()
        assert len(counts) == 5                            # every division covered
        assert counts.max() <= 2                           # quota respected

    def test_size_band_drops_a_division_from_coverage(self, dma_adjacency, dma_metadata):
        # the size band can remove a division's candidates -- coverage then only
        # requires the divisions that still HAVE an eligible (large) market.
        df, _A, pop = _south_midwest(dma_adjacency, dma_metadata)
        res = LEXSCM({"df": df, **self._BASE, "m": 4, "stratum_col": "division",
                      "min_per_stratum": 1, "size_col": "population",
                      "min_size": int(np.median(pop))}).fit()
        pmap = df.groupby("market")["population"].first()
        assert all(pmap.loc[u] >= int(np.median(pop)) for u in res.selected_units)
