"""Tests for the optional SCIP mixed-integer backend of the HCW best subset.

The SCIP backend solves the Bertsimas-King-Mazumder (2016) cardinality-
constrained least-squares problem to provable optimality with a modern MIQP
solver (formulation 2.1: binary indicators, an SOS-1 coupling forcing
``z_i = 0 => beta_i = 0``, and ``sum z_i <= k``), once per model size, then picks
the size by the information criterion -- the same two-step as the exact
Furnival-Wilson search, via a different engine. These tests are skipped when
``pyscipopt`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyscipopt")

from mlsynth.utils.pda_helpers.hcw.estimation import (
    _best_subset_exhaustive,
    _best_subset_fw,
    _gram,
    _subset_rss,
    best_subset_select,
    info_criterion,
)
from mlsynth.utils.pda_helpers.hcw.scip import best_subset_scip

import os
import pandas as pd
from mlsynth import PDA

_HK = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "HongKong.csv")
_HCW_CANDS = ["China", "Indonesia", "Japan", "Korea", "Malaysia",
              "Philippines", "Singapore", "Taiwan", "Thailand", "United States"]


def _ic(G, Zty, yty, n, sel, crit):
    return info_criterion(_subset_rss(G, Zty, yty, sel), n, len(sel) + 1, crit)


class TestScipBackend:

    @pytest.mark.parametrize("criterion", ["AICc", "AIC", "BIC"])
    @pytest.mark.parametrize("seed", range(4))
    def test_scip_reaches_exhaustive_optimum(self, criterion, seed):
        rng = np.random.default_rng(seed)
        T0, N = 26, 7
        X = rng.standard_normal((T0, N))
        k_true = int(rng.integers(1, 4))
        support = rng.choice(N, size=k_true, replace=False)
        y = X[:, support] @ rng.standard_normal(k_true) + 0.4 * rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        r_max = min(N, T0 - 2)
        ex = _best_subset_exhaustive(G, Zty, yty, N, T0, r_max, criterion)
        sc = best_subset_scip(G, Zty, yty, N, T0, r_max, criterion)
        # Same objective value (ties allowed); the MIQP solves to optimality.
        assert abs(_ic(G, Zty, yty, T0, sc, criterion)
                   - _ic(G, Zty, yty, T0, ex, criterion)) < 1e-4

    def test_scip_recovers_strong_support_exactly(self):
        # High SNR, unique optimum: the MIQP must land on the exhaustive set.
        rng = np.random.default_rng(11)
        T0, N = 40, 8
        X = rng.standard_normal((T0, N))
        y = X[:, [1, 5]] @ np.array([3.0, -2.5]) + 0.02 * rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        r_max = min(N, T0 - 2)
        ex = _best_subset_exhaustive(G, Zty, yty, N, T0, r_max, "AICc")
        sc = best_subset_scip(G, Zty, yty, N, T0, r_max, "AICc")
        assert sorted(sc) == sorted(ex)

    def test_scip_reports_certified_and_gap(self):
        rng = np.random.default_rng(3)
        T0, N = 24, 6
        X = rng.standard_normal((T0, N))
        y = X[:, [0, 2]] @ np.array([1.5, -1.0]) + 0.3 * rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        stats: dict = {}
        best_subset_scip(G, Zty, yty, N, T0, min(N, T0 - 2), "AICc", stats=stats)
        assert stats["certified"] is True
        assert stats["optimality_gap"] >= 0.0
        assert stats["backend"] == "scip"

    def test_scip_respects_nvmax(self):
        rng = np.random.default_rng(5)
        T0, N = 28, 8
        X = rng.standard_normal((T0, N))
        y = X @ rng.standard_normal(N) + 0.2 * rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        sc = best_subset_scip(G, Zty, yty, N, T0, 2, "AIC")
        assert len(sc) <= 2

    def test_scip_collinear_pool_matches_objective(self):
        rng = np.random.default_rng(7)
        T0 = 24
        x = rng.standard_normal((T0, 1))
        X = np.column_stack([x[:, 0], x[:, 0], rng.standard_normal(T0),
                             rng.standard_normal(T0)])
        y = 1.5 * x[:, 0] + 0.05 * rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        N, r_max = X.shape[1], min(X.shape[1], T0 - 2)
        ex = _best_subset_exhaustive(G, Zty, yty, N, T0, r_max, "AICc")
        sc = best_subset_scip(G, Zty, yty, N, T0, r_max, "AICc")
        assert abs(_ic(G, Zty, yty, T0, sc, "AICc")
                   - _ic(G, Zty, yty, T0, ex, "AICc")) < 1e-4

    def test_scip_intercept_only_when_donors_useless(self):
        # Donors uncorrelated with a near-constant treated series: no donor earns
        # its penalty, so both engines fall back to the intercept-only model.
        rng = np.random.default_rng(11)
        T0, N = 15, 4
        X = rng.standard_normal((T0, N))
        y = 5.0 + 0.001 * rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        r_max = min(N, T0 - 2)
        ex = _best_subset_exhaustive(G, Zty, yty, N, T0, r_max, "AICc")
        sc = best_subset_scip(G, Zty, yty, N, T0, r_max, "AICc")
        assert sc == sorted(ex)

    def test_scip_honours_time_limit(self):
        # A per-size time limit is accepted and still returns a valid selection.
        rng = np.random.default_rng(6)
        T0, N = 24, 6
        X = rng.standard_normal((T0, N))
        y = X[:, [0, 3]] @ np.array([1.0, -1.0]) + 0.3 * rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        sc = best_subset_scip(G, Zty, yty, N, T0, min(N, T0 - 2), "AICc",
                              time_limit=5.0)
        assert isinstance(sc, list) and len(sc) <= min(N, T0 - 2)

    def test_best_subset_select_scip_backend_matches_fw(self):
        rng = np.random.default_rng(2)
        T0, N = 30, 8
        X = rng.standard_normal((T0, N))
        y = X[:, [1, 3, 6]] @ np.array([1.0, -1.5, 2.0]) + 0.3 * rng.standard_normal(T0)
        G, Zty, yty = _gram(y[:T0], X[:T0])
        r_max = min(N, T0 - 2)
        fw = _best_subset_fw(G, Zty, yty, N, T0, r_max, "AICc")
        sc = best_subset_select(y, X, T0, criterion="AICc", backend="scip")
        assert abs(_ic(G, Zty, yty, T0, sc, "AICc")
                   - _ic(G, Zty, yty, T0, fw, "AICc")) < 1e-4

    def test_estimator_scip_backend_matches_default_on_hk(self):
        # End-to-end: the SCIP backend selects HCW's Table XVI donors, same as
        # the default Furnival-Wilson engine.
        d = pd.read_csv(os.path.abspath(_HK))
        sub = d[d["Country"].isin(["Hong Kong"] + _HCW_CANDS)].copy()
        common = dict(outcome="GDP", treat="Integration", unitid="Country",
                      time="Time", method="hcw", display_graphs=False)
        fw = PDA({"df": sub, **common}).fit()
        sc = PDA({"df": sub, "hcw_backend": "scip", **common}).fit()
        assert sorted(sc.fits["hcw"].selected_donors) == \
            sorted(fw.fits["hcw"].selected_donors)
        assert sc.fits["hcw"].metadata["certified_optimal"] is True
