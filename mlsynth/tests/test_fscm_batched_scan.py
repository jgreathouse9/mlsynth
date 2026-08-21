"""The FSCM forward scan, batched -- and the proof it changed nothing.

Forward SCM scores every remaining donor at every step, and each score needs a
simplex-constrained solve. The released scan called the solver once per
candidate, so a step cost N solver calls and the path cost ~N^2/2 -- 43s on
Prop 99's 38 donors, and the measured cost per call was Python overhead, not
flops.

What makes the scan batchable is that at step k every candidate design
``[X_S, x_j]`` shares its first k columns, so every candidate Gram is a
``(k+1) x (k+1)`` submatrix of the one donor Gram ``X'X``. FISTA runs on
``(A'A, A'b)`` alone, so all candidates advance under one Python loop.

This is an acceleration, not a method change, so the contract is equality:
against a naive reference loop written out in this file, the batched scan must
choose the same donor and report the same RMSPE. The reference lives here rather
than in the library so there is no second implementation to keep alive.

Frank-Wolfe candidate screening was prototyped alongside this and rejected --
see ``agents/future_integrations.md``. It is not tested here because it is not
in the library.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.bilevel.simplex import simplex_lstsq
from mlsynth.utils.fscm_helpers.estimation import (
    candidate_grams,
    rolling_origin_rmspe_batched,
    scan_candidates,
)

BASEDATA = "basedata"


# --------------------------------------------------------------- reference
def reference_step(X_pre, y_pre, selected, candidates):
    """The released scan: one solver call per candidate, best RMSPE wins."""
    best = (None, np.inf, None)
    for j in candidates:
        idx = list(selected) + [j]
        w = simplex_lstsq(X_pre[:, idx], y_pre)
        rmspe = float(np.sqrt(np.mean((y_pre - X_pre[:, idx] @ w) ** 2)))
        if rmspe < best[1]:
            best = (j, rmspe, w)
    return best


def reference_rolling(Y, y, idx, origins):
    """The released expanding-window CV: one solver call per origin."""
    errs = []
    for t in origins:
        w = simplex_lstsq(Y[:t][:, idx], y[:t])
        errs.append((y[t] - Y[t, idx] @ w) ** 2)
    return float(np.sqrt(np.mean(errs)))


def _panel(N=12, T0=20, r=3, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T0, r))
    L = np.abs(rng.normal(size=(N, r)))
    X = F @ L.T + 0.3 * rng.normal(size=(T0, N))
    w = np.zeros(N)
    w[rng.choice(N, min(3, N), replace=False)] = 1.0 / min(3, N)
    return X, X @ w + 0.3 * rng.normal(size=T0)


# ------------------------------------------------------------------- smoke
class TestItRuns:
    def test_a_first_step_on_a_minimal_panel(self):
        X, y = _panel(N=3, T0=6)
        j, rmspe, w = scan_candidates(X.T @ X, X.T @ y, float(y @ y),
                                      [], [0, 1, 2], X.shape[0])
        assert j in (0, 1, 2)
        assert np.isfinite(rmspe) and rmspe >= 0.0
        assert w.shape == (1,) and w[0] == pytest.approx(1.0)


# -------------------------------------------------------------------- unit
class TestTheCandidateGrams:
    @pytest.mark.parametrize("n_sel", [0, 1, 3])
    def test_they_are_the_designs_own_grams(self, n_sel):
        X, y = _panel(N=10, T0=18, seed=n_sel)
        sel = list(range(n_sel))
        cand = [j for j in range(10) if j not in sel]
        AtA, Atb = candidate_grams(X.T @ X, X.T @ y, sel, cand)
        for m, j in enumerate(cand):
            A = X[:, sel + [j]]
            assert np.allclose(AtA[m], A.T @ A, atol=1e-10)
            assert np.allclose(Atb[m], A.T @ y, atol=1e-10)

    def test_the_new_donor_is_the_last_coordinate(self):
        X, y = _panel(N=6, T0=14, seed=1)
        AtA, _ = candidate_grams(X.T @ X, X.T @ y, [0, 1], [4])
        assert AtA[0, -1, -1] == pytest.approx(X[:, 4] @ X[:, 4])
        assert AtA[0, 0, 0] == pytest.approx(X[:, 0] @ X[:, 0])


class TestTheScanMatchesTheReference:
    @pytest.mark.parametrize("n_sel", [0, 1, 2, 4])
    def test_same_donor_and_same_rmspe(self, n_sel):
        X, y = _panel(N=14, T0=22, seed=10 + n_sel)
        sel = list(range(n_sel))
        cand = [j for j in range(14) if j not in sel]
        want_j, want_r, want_w = reference_step(X, y, sel, cand)
        got_j, got_r, got_w = scan_candidates(X.T @ X, X.T @ y, float(y @ y),
                                              sel, cand, X.shape[0])
        assert got_j == want_j
        assert got_r == pytest.approx(want_r, abs=1e-9)
        assert np.allclose(got_w, want_w, atol=1e-7)

    def test_a_whole_path_matches_step_for_step(self):
        X, y = _panel(N=16, T0=24, seed=99)
        G, c, yy = X.T @ X, X.T @ y, float(y @ y)
        sel_ref, sel_fast, rem_ref, rem_fast = [], [], list(range(16)), list(range(16))
        for _ in range(8):
            jr, rr, _ = reference_step(X, y, sel_ref, rem_ref)
            jf, rf, _ = scan_candidates(G, c, yy, sel_fast, rem_fast, X.shape[0])
            assert jf == jr
            assert rf == pytest.approx(rr, abs=1e-9)
            sel_ref.append(jr); rem_ref.remove(jr)
            sel_fast.append(jf); rem_fast.remove(jf)


class TestTheRollingOriginCV:
    @pytest.mark.parametrize("k", [1, 2, 5])
    def test_it_matches_the_per_origin_loop(self, k):
        X, y = _panel(N=9, T0=30, seed=20 + k)
        idx = list(range(k))
        origins = np.arange(12, 30, 3)
        want = reference_rolling(X, y, idx, origins)
        got = rolling_origin_rmspe_batched(X, y, idx, origins)
        assert got == pytest.approx(want, abs=1e-9)


# ------------------------------------------------------------- invariants
class TestPathInvariants:
    def test_the_train_rmspe_never_rises_along_the_path(self):
        """The k-donor simplex sits inside the (k+1)-donor simplex (set the new
        weight to zero), so admitting a donor cannot worsen the optimum."""
        X, y = _panel(N=15, T0=25, seed=4)
        G, c, yy = X.T @ X, X.T @ y, float(y @ y)
        sel, rem, path = [], list(range(15)), []
        for _ in range(10):
            j, r, _ = scan_candidates(G, c, yy, sel, rem, X.shape[0])
            sel.append(j); rem.remove(j); path.append(r)
        assert np.all(np.diff(path) <= 1e-9)

    def test_relabelling_the_donors_picks_the_same_one(self):
        X, y = _panel(N=13, T0=21, seed=6)
        T0 = X.shape[0]
        j0, r0, _ = scan_candidates(X.T @ X, X.T @ y, float(y @ y),
                                    [], list(range(13)), T0)
        p = np.random.default_rng(6).permutation(13)
        Xp = X[:, p]
        jp, rp, _ = scan_candidates(Xp.T @ Xp, Xp.T @ y, float(y @ y),
                                    [], list(range(13)), T0)
        assert p[jp] == j0
        assert rp == pytest.approx(r0, abs=1e-9)

    def test_the_returned_weights_are_on_the_simplex(self):
        X, y = _panel(N=11, T0=19, seed=8)
        _, _, w = scan_candidates(X.T @ X, X.T @ y, float(y @ y),
                                  [0, 1], [2, 3, 4, 5], X.shape[0])
        assert (w >= -1e-12).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------- edge cases
class TestDegenerateInput:
    def test_one_candidate_is_selected_by_default(self):
        X, y = _panel(N=4, T0=12, seed=2)
        j, r, w = scan_candidates(X.T @ X, X.T @ y, float(y @ y),
                                  [0, 1], [3], X.shape[0])
        assert j == 3
        want = simplex_lstsq(X[:, [0, 1, 3]], y)
        assert np.allclose(w, want, atol=1e-7)

    def test_two_pre_periods_still_scan(self):
        X, y = _panel(N=5, T0=2, seed=3)
        j, r, _ = scan_candidates(X.T @ X, X.T @ y, float(y @ y),
                                  [], [0, 1, 2, 3, 4], 2)
        assert np.isfinite(r)

    def test_a_duplicated_donor_does_not_break_the_scan(self):
        X, y = _panel(N=6, T0=16, seed=5)
        X[:, 3] = X[:, 1]
        j, r, w = scan_candidates(X.T @ X, X.T @ y, float(y @ y),
                                  [1], [0, 2, 3, 4, 5], 16)
        assert np.isfinite(r)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_a_constant_donor_does_not_produce_nan(self):
        X, y = _panel(N=6, T0=16, seed=13)
        X[:, 2] = 0.0
        _, r, w = scan_candidates(X.T @ X, X.T @ y, float(y @ y),
                                  [], [0, 1, 2, 3], 16)
        assert np.isfinite(r)
        assert np.isfinite(w).all()


# ---------------------------------------------------------------- failures
class TestBadInputIsReported:
    def test_an_empty_candidate_list_is_rejected(self):
        X, y = _panel(N=4, T0=10, seed=14)
        with pytest.raises(MlsynthEstimationError, match="no candidate"):
            scan_candidates(X.T @ X, X.T @ y, float(y @ y), [0], [], 10)

    def test_a_candidate_already_selected_is_rejected(self):
        X, y = _panel(N=4, T0=10, seed=15)
        with pytest.raises(MlsynthEstimationError, match="already selected"):
            scan_candidates(X.T @ X, X.T @ y, float(y @ y), [1], [1, 2], 10)

    def test_no_origins_is_rejected(self):
        X, y = _panel(N=4, T0=10, seed=16)
        with pytest.raises(MlsynthEstimationError, match="origin"):
            rolling_origin_rmspe_batched(X, y, [0], np.array([], dtype=int))


# --------------------------------------------------------- end-to-end pins
def _fscm(cfg):
    from mlsynth import FSCM
    return FSCM(dict(cfg)).fit()


@pytest.mark.parametrize("name,att,donors", [
    ("prop99", -20.15022693,
     {"Montana": 0.416182, "Nevada": 0.255050, "Utah": 0.328768}),
    ("basque", -0.70149469,
     {"Cataluna": 0.839587, "Madrid (Comunidad De)": 0.160413}),
])
def test_the_public_estimator_is_unchanged(name, att, donors):
    """Values captured from the pre-batching implementation. An acceleration
    that moved them would be a method change, not an acceleration."""
    if name == "prop99":
        d = pd.read_csv(f"{BASEDATA}/P99data.csv")
        d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)
        cfg = dict(df=d, outcome="cigsale", treat="treated", unitid="state",
                   time="year", display_graphs=False)
    else:
        d = pd.read_csv(f"{BASEDATA}/basque_data.csv")[
            ["regionname", "year", "gdpcap"]].dropna()
        d["treat"] = ((d.regionname == "Basque Country (Pais Vasco)")
                      & (d.year >= 1975)).astype(int)
        cfg = dict(df=d, outcome="gdpcap", treat="treat", unitid="regionname",
                   time="year", display_graphs=False)

    res = _fscm(cfg)
    assert res.effects.att == pytest.approx(att, abs=1e-6)
    got = {k: v for k, v in (res.weights.donor_weights or {}).items()
           if abs(v) > 1e-6}
    assert set(got) == set(donors)
    for k, v in donors.items():
        assert got[k] == pytest.approx(v, abs=1e-5)
