"""GEOLIFT spillover-aware selection on the bundled Nielsen DMA map.

The parallel of ``test_lexscm_dma_demo`` for GEOLIFT: validate the
``basedata/markets/`` real US market-area contiguity together with GeoLift's
``adjacency`` / ``cluster_col`` constraints. We synthesise a division-driven,
border-smoothed panel over a real DMA subset and check the two guarantees the
conflict graph must enforce — on **real borders**:

* no two **treated** DMAs share a border (Stage-1 independent-set rule);
* no **donor** borders any treated DMA (Stage-2 spillover exclusion).

There is a genuine tension worth recording: GeoLift nominates candidates by
*correlation*, and a border-smoothed panel makes bordering DMAs the most
correlated, so the independent-set filter removes many nominees — the test
confirms admissible regions still remain and every one honours the borders.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import GEOLIFT

_MARKETS = Path(__file__).resolve().parents[2] / "basedata" / "markets"

# A handful of census divisions, enough DMAs that border-free regions exist.
_DIVISION = {
    "FL": "S. Atlantic", "GA": "S. Atlantic", "NC": "S. Atlantic", "SC": "S. Atlantic",
    "TN": "E.S. Central", "MS": "E.S. Central", "AL": "E.S. Central",
    "OH": "E.N. Central", "IN": "E.N. Central", "IL": "E.N. Central",
    "MI": "E.N. Central", "WI": "E.N. Central",
    "MN": "W.N. Central", "MO": "W.N. Central",
}


@pytest.fixture(scope="module")
def dma_adjacency():
    return pd.read_csv(_MARKETS / "dma_adjacency.csv", index_col=0)


@pytest.fixture(scope="module")
def dma_metadata():
    return pd.read_csv(_MARKETS / "dma_metadata.csv")


def _geo_panel(adj, meta, seed=7, T=40):
    """Division-driven factor panel with mild border smoothing over real DMAs."""
    meta = meta[meta.state.isin(_DIVISION)].copy()
    meta["division"] = meta.state.map(_DIVISION)
    names = [n for n in meta.dma_name if n in adj.index]
    meta = meta[meta.dma_name.isin(names)].reset_index(drop=True)
    A = adj.loc[names, names]
    W = A.values.astype(float)
    n, r = len(names), 4
    rng = np.random.default_rng(seed)
    div = meta.set_index("dma_name").loc[names, "division"].values
    Lam_div = {d: rng.normal(size=r) for d in sorted(set(div))}
    lam = np.array([Lam_div[d] for d in div]) + 0.15 * rng.normal(size=(n, r))
    lam = 0.85 * lam + 0.15 * (W @ lam) / np.maximum(W.sum(1, keepdims=True), 1)
    F = np.cumsum(rng.normal(size=(T, r)), 0)
    Y = 100 + rng.normal(0, 5, n) + F @ lam.T + rng.normal(0, 1.0, (T, n))
    df = pd.DataFrame([
        {"location": names[j], "date": t, "Y": float(Y[t, j]),
         "division": div[j]}
        for j in range(n) for t in range(T)
    ])
    return df, A


def test_geolift_spillover_on_real_dma_borders(dma_adjacency, dma_metadata):
    df, A = _geo_panel(dma_adjacency, dma_metadata)
    res = GEOLIFT({
        "df": df, "outcome": "Y", "unitid": "location", "time": "date",
        "treatment_size": 3, "durations": [8], "effect_sizes": [0.0, 0.1],
        "lookback_window": 1, "how": "mean", "ns": 30, "seed": 0,
        "adjacency": A, "display_graphs": False,
    }).fit()

    assert len(res.search.candidates) >= 1                  # the filter leaves regions
    for cd in res.search.candidates:
        members = [str(u) for u in cd.candidate]
        # Guarantee 1: no two treated DMAs border each other.
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                assert A.loc[a, b] == 0, f"treated {a!r} and {b!r} share a border"
        # Guarantee 2: no donor borders any treated DMA.
        neighbours = {x for t in members for x in A.columns[A.loc[t] == 1]}
        assert neighbours.isdisjoint(cd.weights.donor_weights)

    # the recommended design honours the borders too
    if res.selected_units is not None:
        treated = list(res.selected_units)
        for i, a in enumerate(treated):
            for b in treated[i + 1:]:
                assert A.loc[a, b] == 0


def test_geolift_adjacency_forbids_a_known_border(dma_adjacency, dma_metadata):
    """A specific real border (Atlanta, GA ~ Macon, GA) is never co-treated."""
    df, A = _geo_panel(dma_adjacency, dma_metadata)
    assert A.loc["Atlanta, GA", "Macon, GA"] == 1            # the border exists
    res = GEOLIFT({
        "df": df, "outcome": "Y", "unitid": "location", "time": "date",
        "treatment_size": 3, "durations": [8], "effect_sizes": [0.0, 0.1],
        "lookback_window": 1, "how": "mean", "ns": 30, "seed": 0,
        "adjacency": A, "display_graphs": False,
    }).fit()
    for cd in res.search.candidates:
        members = {str(u) for u in cd.candidate}
        assert not {"Atlanta, GA", "Macon, GA"} <= members


def test_geolift_cluster_col_state_no_two_treated_same_state(dma_adjacency, dma_metadata):
    """cluster_col on real state labels: at most one treated DMA per state."""
    df, A = _geo_panel(dma_adjacency, dma_metadata)
    state = (dma_metadata.set_index("dma_name")["state"]
             .loc[[c for c in A.index]])
    df["state"] = df["location"].map(state)
    res = GEOLIFT({
        "df": df, "outcome": "Y", "unitid": "location", "time": "date",
        "treatment_size": 3, "durations": [8], "effect_sizes": [0.0, 0.1],
        "lookback_window": 1, "how": "mean", "ns": 30, "seed": 0,
        "cluster_col": "state", "display_graphs": False,
    }).fit()
    smap = state.to_dict()
    for cd in res.search.candidates:
        states = [smap[str(u)] for u in cd.candidate]
        assert len(states) == len(set(states))              # one per state
