"""TDD for SYNDES donor-side restrictions (region-matched / non-bordering donors).

The treated-side restrictions constrain *who is treated* (the assignment ``D``).
This module's feature constrains *who may serve as a donor for a treated unit* --
a coupling between ``D`` and the control weights. The flexible primitive is a
donor-exclusion relation ``B[i, j]`` ("if ``i`` is treated, ``j`` may not be its
donor"), enforced in every mode:

* one_way_global  (control vector ``c``):    ``c[j]      <= 1 - D[i]``
* two_way_global  (control ``w - q``):       ``w[j]-q[j] <= 1 - D[i]``
* per_unit        (row ``i``):               ``w[i, j]   == 0``

``B`` is filled by ``donor_region_col`` (a donor must share the treated unit's
region), ``exclude_bordering_donors`` (a treated unit's spillover neighbours are
dropped from its donor pool -- the same conflict graph the treated-side uses), or
an explicit ``donor_exclusion`` matrix (the escape hatch).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.fast_scm_helpers.structure import IndexSet
from mlsynth.utils.syndes_helpers.restrictions import (
    build_restrictions,
    donor_constraints,
)

_MARKETS = Path(__file__).resolve().parents[2] / "basedata" / "markets"

# CDC/NCHS Census-region grouping of US states
# (https://www.cdc.gov/nchs/hus/sources-definitions/geographic-region.htm).
_CDC = {
    **{s: "Northeast" for s in
       ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"]},
    **{s: "Midwest" for s in
       ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"]},
    **{s: "South" for s in
       ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS",
        "TN", "AR", "LA", "OK", "TX"]},
    **{s: "West" for s in
       ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR",
        "WA"]},
}


_REGION = {f"u{j}": ("A" if j < 4 else "B") for j in range(8)}


def _panel(N=8, T=14, n_post=4, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2))
    L = rng.uniform(0.3, 1.0, (N, 2))
    lvl = rng.uniform(8.0, 12.0, N)
    Y = lvl + F @ L.T + rng.normal(scale=0.3, size=(T, N))
    rows = [{"unit": f"u{j}", "time": t, "Y": float(Y[t, j]),
             "post": int(t >= T - n_post), "region": _REGION[f"u{j}"]}
            for j in range(N) for t in range(T)]
    return pd.DataFrame(rows)


def _uindex(N=8):
    return IndexSet.from_labels([f"u{j}" for j in range(N)])


# ----------------------------------------------------------------------
# Layer 1 -- build the donor-exclusion relation
# ----------------------------------------------------------------------

class TestBuildDonorExclusion:
    def test_region_col_excludes_cross_region(self):
        r = build_restrictions(_panel(), "unit", _uindex(),
                               donor_region_col="region")
        expected = {(i, j) for i in range(8) for j in range(8)
                    if i != j and (i < 4) != (j < 4)}
        assert set(r.donor_exclusion) == expected

    def test_exclude_bordering_uses_conflict_both_directions(self):
        labels = [f"u{j}" for j in range(8)]
        A = pd.DataFrame(0.0, index=labels, columns=labels)
        A.loc["u0", "u1"] = A.loc["u1", "u0"] = 1.0
        r = build_restrictions(_panel(), "unit", _uindex(),
                               adjacency=A, spillover_threshold=0.5,
                               exclude_bordering_donors=True)
        assert {(0, 1), (1, 0)} <= set(r.donor_exclusion)

    def test_explicit_matrix(self):
        labels = [f"u{j}" for j in range(8)]
        M = pd.DataFrame(0.0, index=labels, columns=labels)
        M.loc["u0", "u3"] = 1.0                       # u3 forbidden donor for u0
        r = build_restrictions(_panel(), "unit", _uindex(), donor_exclusion=M)
        assert (0, 3) in r.donor_exclusion
        assert (3, 0) not in r.donor_exclusion        # direction matters

    def test_explicit_ndarray_matrix(self):
        M = np.zeros((8, 8))
        M[0, 3] = 1.0
        r = build_restrictions(_panel(), "unit", _uindex(), donor_exclusion=M)
        assert (0, 3) in r.donor_exclusion

    def test_explicit_ndarray_wrong_shape_raises(self):
        with pytest.raises(MlsynthConfigError):
            build_restrictions(_panel(), "unit", _uindex(),
                               donor_exclusion=np.zeros((3, 3)))

    def test_explicit_matrix_missing_unit_raises(self):
        from mlsynth.exceptions import MlsynthDataError
        M = pd.DataFrame(0.0, index=["u0", "u1"], columns=["u0", "u1"])
        with pytest.raises(MlsynthDataError):
            build_restrictions(_panel(), "unit", _uindex(), donor_exclusion=M)

    def test_empty_without_donor_rules(self):
        r = build_restrictions(_panel(), "unit", _uindex())
        assert not r.donor_exclusion


# ----------------------------------------------------------------------
# Layer 1 -- per-mode constraint emission
# ----------------------------------------------------------------------

class TestDonorConstraints:
    def _vars(self):
        import cvxpy as cp
        return cp.Variable(5, boolean=True), cp.Variable(5, nonneg=True)

    def test_one_way_global(self):
        D, c = self._vars()
        cons = donor_constraints("global_equal_weights", {"c": c}, D,
                                 [(0, 3), (1, 4)])
        assert len(cons) == 2

    def test_two_way_global(self):
        import cvxpy as cp
        D = cp.Variable(5, boolean=True)
        cons = donor_constraints("global_2way",
                                 {"w": cp.Variable(5), "q": cp.Variable(5)},
                                 D, [(0, 3)])
        assert len(cons) == 1

    def test_per_unit(self):
        import cvxpy as cp
        D = cp.Variable(5, boolean=True)
        cons = donor_constraints("per_unit",
                                 {"w": cp.Variable((5, 5)),
                                  "q": cp.Variable((5, 5))}, D, [(0, 3)])
        assert len(cons) == 1

    def test_empty_pairs_no_constraints(self):
        D, c = self._vars()
        assert donor_constraints("global_equal_weights", {"c": c}, D, []) == []


# ----------------------------------------------------------------------
# Layer 3 -- the solved design honours donor restrictions, every mode
# ----------------------------------------------------------------------

def _control_idx(res, n_units):
    cw = getattr(res.design, "control_weights", None)
    if cw is not None:
        return np.flatnonzero(np.abs(np.asarray(cw, float).reshape(-1)) > 1e-6)
    tw = np.asarray(res.design.treated_weights, float)   # per_unit (K, N)
    return np.flatnonzero(np.abs(tw).sum(axis=0) > 1e-6)


class TestSYNDESDonorRestrictionsEnforced:
    _BASE = dict(outcome="Y", unitid="unit", time="time", post_col="post",
                 run_inference=False, solver="SCIP", gap_limit=0.2,
                 time_limit=10.0)

    @pytest.mark.parametrize("mode", ["one_way_global", "two_way_global",
                                      "per_unit"])
    def test_same_region_donors(self, mode):
        # Core invariant (all modes): every donor used for treated unit i shares
        # i's region. In the global modes one shared donor vector serves every
        # treated unit, so this also forces the treated set into a single region;
        # in per_unit each treated unit draws its own same-region donors, so the
        # treated set may span regions.
        res = SYNDES({"df": _panel(), **self._BASE, "mode": mode, "K": 2,
                      "donor_region_col": "region"}).fit()
        labels = [f"u{j}" for j in range(8)]
        treated_idx = list(np.asarray(res.design.selected_unit_indices).tolist())
        if mode == "per_unit":
            q = np.asarray(res.design.treated_weights, float)        # (N, N)
            for i in treated_idx:
                donors = np.flatnonzero(q[i] > 1e-6)
                assert all(_REGION[labels[d]] == _REGION[labels[i]]
                           for d in donors)
        else:
            cw = np.asarray(res.design.control_weights, float).reshape(-1)
            donors = np.flatnonzero(cw > 1e-6)
            for i in treated_idx:
                assert all(_REGION[labels[d]] == _REGION[labels[i]]
                           for d in donors)
            assert len({_REGION[labels[i]] for i in treated_idx}) == 1

    def test_non_bordering_donors_one_way(self):
        labels = [f"u{j}" for j in range(8)]
        A = pd.DataFrame(0.0, index=labels, columns=labels)
        # u0 borders u1,u2 ; u4 borders u5
        for a, b in [("u0", "u1"), ("u0", "u2"), ("u4", "u5")]:
            A.loc[a, b] = A.loc[b, a] = 1.0
        res = SYNDES({"df": _panel(), **self._BASE, "mode": "one_way_global",
                      "K": 2, "adjacency": A, "spillover_threshold": 0.5,
                      "exclude_bordering_donors": True}).fit()
        treated_idx = set(np.asarray(res.design.selected_unit_indices).tolist())
        donor_idx = set(_control_idx(res, 8).tolist())
        Ab = A.to_numpy()
        # no donor borders any treated unit
        for i in treated_idx:
            assert not any(Ab[i, j] > 0 for j in donor_idx)


# ----------------------------------------------------------------------
# Config / failure semantics
# ----------------------------------------------------------------------

class TestSYNDESDonorRestrictionsConfig:
    _BASE = dict(outcome="Y", unitid="unit", time="time", post_col="post",
                 mode="two_way_global", run_inference=False)

    def _make(self, **over):
        return SYNDES({"df": _panel(), **self._BASE, **over})

    def test_region_col_must_exist(self):
        with pytest.raises(MlsynthConfigError):
            self._make(K=2, donor_region_col="nope")

    def test_exclude_bordering_needs_conflict_source(self):
        with pytest.raises(MlsynthConfigError):
            self._make(K=2, exclude_bordering_donors=True)   # no cluster/adjacency

    def test_donor_restriction_rejects_annealed(self):
        with pytest.raises(MlsynthConfigError):
            self._make(K=2, mode="two_way_global_annealed",
                       donor_region_col="region")

    def test_donor_restriction_rejects_arm(self):
        with pytest.raises(MlsynthConfigError):
            self._make(K=2, arm="region", donor_region_col="region")


# ----------------------------------------------------------------------
# Real geography: the user's literal scenario on bundled DMA borders +
# CDC regions -- a Midwest treated unit may borrow from non-bordering
# Midwest DMAs, but never from a bordering Northeast (e.g. PA) one.
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def dma():
    adj = pd.read_csv(_MARKETS / "dma_adjacency.csv", index_col=0)
    meta = pd.read_csv(_MARKETS / "dma_metadata.csv")
    meta = meta.assign(cdc=meta["state"].map(_CDC))
    return adj, meta


def _geo_panel(units, cdc_of, T=16, n_post=4, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2))
    L = rng.uniform(0.3, 1.0, (len(units), 2))
    lvl = rng.uniform(8.0, 12.0, len(units))
    Y = lvl + F @ L.T + rng.normal(scale=0.3, size=(T, len(units)))
    rows = [{"market": u, "t": t, "Y": float(Y[t, j]),
             "post": int(t >= T - n_post), "cdc": cdc_of[u]}
            for j, u in enumerate(units) for t in range(T)]
    return pd.DataFrame(rows)


class TestSYNDESDonorRestrictionsRealDMA:
    _BASE = dict(outcome="Y", unitid="market", time="t", post_col="post",
                 run_inference=False, solver="SCIP", gap_limit=0.2,
                 time_limit=15.0)

    def test_region_matched_non_bordering_donors_per_unit(self, dma):
        adj, meta = dma
        # A Midwest + Northeast border zone with real cross-region borders
        # (e.g. Erie, PA <-> Youngstown, OH).
        zone = meta[meta["state"].isin(["MI", "OH", "IL", "IN", "PA", "NY"])]
        units = zone["dma_name"].tolist()
        cdc_of = meta.set_index("dma_name")["cdc"].to_dict()
        df = _geo_panel(units, cdc_of)
        res = SYNDES({"df": df, **self._BASE, "mode": "per_unit", "K": 3,
                      "adjacency": adj, "spillover_threshold": 0.5,
                      "donor_region_col": "cdc",
                      "exclude_bordering_donors": True}).fit()
        labels = list(np.asarray(res.inputs.unit_index.labels))
        A = adj.reindex(index=labels, columns=labels).to_numpy()
        q = np.asarray(res.design.treated_weights, float)        # (N, N)
        treated = list(np.asarray(res.design.selected_unit_indices).tolist())
        assert treated, "expected a non-empty treated set"
        for i in treated:
            for d in np.flatnonzero(q[i] > 1e-6):
                assert cdc_of[labels[d]] == cdc_of[labels[i]]     # same region
                assert A[i, d] == 0                               # not bordering
