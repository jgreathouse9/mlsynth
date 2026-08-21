"""The FSCM forward scan, solved exactly.

Forward SCM scores every remaining donor at every step, and each score needs a
simplex-constrained solve. The released scan called the FISTA primitive once per
candidate, which was slow -- 43s on Prop 99's 38 donors -- and, it turned out,
not accurate enough for what the scan does with the answer.

FISTA exhausts its 2000-iteration budget on these designs and returns an
optimum slightly too high, by up to 9.3e-05 in RMSPE on Prop 99. That is
invisible while the fit is still improving, because the true improvement at each
step is larger than the error. Once the fit saturates there is no true
improvement left, and the reported path *rises* -- which the real optimum cannot
do, since the k-donor simplex is the face of the (k+1)-donor simplex where the
new weight is zero.

The scan now solves each candidate exactly with the active-set QP, warm-started
from the incumbent weights padded with a zero for the new donor. A forward scan
is a chain of neighbouring problems, so that seed is both feasible by
construction and the pattern the active set collapses on: the exact solve is
about 50x faster end to end than the FISTA it replaces.

Being exact, it cannot be checked against the loop it replaced -- that loop is
the thing that was wrong -- and checking it against the active set would be
circular. The oracle here is cvxpy, a different solver entirely, and the
tolerance is set by *its* accuracy rather than by the code under test.

The invariants that decide correctness are searched rather than pinned; see
``test_fscm_scan_properties``. Two earlier attempts on this scan -- Frank-Wolfe
candidate screening and an objective-based stopping rule -- were prototyped and
rejected on measurement; see ``agents/future_integrations.md``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.bilevel.simplex import simplex_lstsq
from mlsynth.utils.fscm_helpers.estimation import (
    rolling_origin_rmspe_exact,
    scan_candidates,
)

BASEDATA = "basedata"


# --------------------------------------------------------------- references
def cvxpy_weights(A, b):
    """An independent exact solve, used as the oracle.

    The scan is exact, so it cannot be checked against the FISTA loop it
    replaced -- that loop is the thing that was wrong. Checking it against the
    active-set QP would be circular, so the oracle is a different solver
    entirely.
    """
    import cvxpy as cp

    w = cp.Variable(A.shape[1], nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(A @ w - b)), [cp.sum(w) == 1]).solve(
        solver="CLARABEL")
    return np.asarray(w.value, dtype=float).ravel()


def fista_step(X_pre, y_pre, selected, candidates):
    """The scan as it was before this change: one FISTA call per candidate.

    Kept to document the direction of the difference, not as a contract. FISTA
    exhausts its iteration budget on these designs and lands above the optimum,
    so the exact scan may pick a different donor and must never report a worse
    fit for the donor they agree on.
    """
    best = (None, np.inf, None)
    for j in candidates:
        idx = list(selected) + [j]
        w = simplex_lstsq(X_pre[:, idx], y_pre)
        rmspe = float(np.sqrt(np.mean((y_pre - X_pre[:, idx] @ w) ** 2)))
        if rmspe < best[1]:
            best = (j, rmspe, w)
    return best


def reference_rolling(Y, y, idx, origins):
    """The expanding-window CV, one independent exact solve per origin."""
    errs = []
    for t in origins:
        w = cvxpy_weights(Y[:t][:, idx], y[:t])
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
        j, rmspe, w = scan_candidates(X, y, [], [0, 1, 2])
        assert j in (0, 1, 2)
        assert np.isfinite(rmspe) and rmspe >= 0.0
        assert w.shape == (1,) and w[0] == pytest.approx(1.0)


# -------------------------------------------------------------------- unit
class TestTheScanIsExact:
    @pytest.mark.parametrize("n_sel", [0, 1, 2, 4])
    def test_it_finds_the_optimum_an_independent_solver_finds(self, n_sel):
        X, y = _panel(N=14, T0=22, seed=10 + n_sel)
        sel = list(range(n_sel))
        cand = [j for j in range(14) if j not in sel]
        got_j, got_r, got_w = scan_candidates(X, y, sel, cand)
        want = cvxpy_weights(X[:, sel + [got_j]], y)
        want_r = float(np.sqrt(np.mean((y - X[:, sel + [got_j]] @ want) ** 2)))
        assert got_r == pytest.approx(want_r, abs=1e-8)
        # and no other candidate beats it under the same independent solver
        for j in cand:
            w = cvxpy_weights(X[:, sel + [j]], y)
            r = float(np.sqrt(np.mean((y - X[:, sel + [j]] @ w) ** 2)))
            assert r >= got_r - 1e-8

    def test_it_is_never_worse_than_the_fista_scan_it_replaced(self):
        """Direction of the change, on the panel family that exposed it."""
        X, y = _saturating_panel(seed=5)
        sel = [0, 1, 2, 3]
        cand = [j for j in range(X.shape[1]) if j not in sel]
        _, r_exact, _ = scan_candidates(X, y, sel, cand)
        _, r_fista, _ = fista_step(X, y, sel, cand)
        assert r_exact <= r_fista + 1e-12

    def test_a_whole_path_is_monotone_and_matches_the_oracle_at_each_step(self):
        X, y = _panel(N=16, T0=24, seed=99)
        sel, rem, warm, path = [], list(range(16)), None, []
        for _ in range(8):
            j, r, warm = scan_candidates(X, y, sel, rem, warm=warm)
            w = cvxpy_weights(X[:, sel + [j]], y)
            assert r == pytest.approx(
                float(np.sqrt(np.mean((y - X[:, sel + [j]] @ w) ** 2))), abs=1e-8)
            sel.append(j); rem.remove(j); path.append(r)
        assert np.all(np.diff(path) <= 1e-9)


class TestTheRollingOriginCV:
    @pytest.mark.parametrize("k", [1, 2, 5])
    def test_it_matches_an_independent_solve_at_every_origin(self, k):
        """Tolerance is set by the oracle, not by the code under test.

        cvxpy solves each window by interior point to roughly 1e-08, while the
        active set is a direct method; across 40 designs it attains the strictly
        lower objective 25 times to cvxpy's 11. So the residual disagreement is
        the oracle's, and asserting tighter than its accuracy would be pinning
        its error.
        """
        X, y = _panel(N=9, T0=30, seed=20 + k)
        idx = list(range(k))
        origins = np.arange(12, 30, 3)
        want = reference_rolling(X, y, idx, origins)
        got = rolling_origin_rmspe_exact(X, y, idx, origins)
        assert got == pytest.approx(want, abs=1e-7)

    def test_it_never_reports_a_worse_fit_than_the_oracle(self):
        """The direction is one-sided: the active set may beat cvxpy, never
        lose to it by more than cvxpy's own tolerance."""
        X, y = _panel(N=9, T0=30, seed=77)
        origins = np.arange(12, 30, 3)
        for k in (1, 3, 5):
            idx = list(range(k))
            assert (rolling_origin_rmspe_exact(X, y, idx, origins)
                    <= reference_rolling(X, y, idx, origins) + 1e-7)


# ------------------------------------------------------------- invariants
def _saturating_panel(N=30, T0=10, r=3, collinearity=0.999, noise=1e-3, seed=0):
    """Ill-conditioned donors whose fit saturates -- the Prop 99 shape.

    Two ingredients are needed to expose an inexact simplex solve, and the
    obvious generator supplies neither. Saturation: the treated unit is an exact
    convex combination of three donors, so from the fourth donor on the true
    optimum stops moving and there is no genuine improvement left to mask solver
    error. Collinearity: the donors are pushed toward a common direction, which
    is what drives ``cond(A)`` up with ``k`` and slows a first-order solver.

    Drawing donors independently with more pre-periods than factors gives a
    strictly decreasing path whatever the solver does, because the true
    improvement at every step stays larger than the error. That is why the
    original monotonicity property never fired.
    """
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T0, r))
    L = np.abs(rng.normal(size=(N, r)))
    X = F @ L.T
    X = collinearity * X.mean(axis=1, keepdims=True) + (1 - collinearity) * X
    X = X + noise * rng.normal(size=(T0, N))
    w = np.zeros(N)
    w[[0, 1, 2]] = [0.5, 0.3, 0.2]
    return X, X @ w


def _scan_path(X, y, steps):
    sel, rem, path, warm = [], list(range(X.shape[1])), [], None
    for _ in range(steps):
        j, r, warm = scan_candidates(X, y, sel, rem, warm=warm)
        sel.append(j); rem.remove(j); path.append(r)
    return sel, np.asarray(path)


class TestPathInvariants:
    def test_the_train_rmspe_never_rises_on_prop99(self):
        """The regression this ladder was built from.

        Adding a donor cannot raise the optimum, because the k-donor simplex is
        the face of the (k+1)-donor simplex where the new weight is zero. On
        Prop 99 the fit saturates at six donors and the FISTA scan's path then
        rose by up to 3.5e-05, sitting 9.3e-05 above the true optimum.
        """
        d = pd.read_csv(f"{BASEDATA}/P99data.csv")
        d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)
        from mlsynth.utils.datautils import dataprep
        prep = dataprep(d, "state", "year", "cigsale", "treated")
        T0 = prep["pre_periods"]
        y = np.asarray(prep["y"], float).ravel()[:T0]
        X = np.asarray(prep["donor_matrix"], float)[:T0]
        _, path = _scan_path(X, y, 12)
        rise = float(np.diff(path).max())
        assert rise <= 1e-9, f"path rose by {rise:.3e}: {np.round(path, 8)}"

    def test_the_train_rmspe_never_rises_on_a_saturating_panel(self):
        """The same invariant on generated data that carries both ingredients."""
        X, y = _saturating_panel()
        _, path = _scan_path(X, y, 10)
        rise = float(np.diff(path).max())
        assert rise <= 1e-9, f"path rose by {rise:.3e}: {np.round(path, 9)}"

    def test_admitting_a_donor_cannot_raise_the_optimum(self):
        """Dominance, per solve and needing no reference solver: the optimum
        over ``S + [j]`` is bounded by the optimum over ``S``."""
        X, y = _saturating_panel(seed=1)
        sel = [0, 1, 2, 3, 4]
        base = float(np.sum((y - X[:, sel] @ scan_candidates(X, y, sel[:-1], [sel[-1]])[2]) ** 2))
        for j in (6, 7, 8, 9, 10):
            w = scan_candidates(X, y, sel, [j])[2]
            full = float(np.sum((y - X[:, sel + [j]] @ w) ** 2))
            assert full <= base + 1e-9, (
                f"donor {j} raised the optimum by {full - base:.3e}")

    def test_the_train_rmspe_never_rises_along_the_path(self):
        """The k-donor simplex sits inside the (k+1)-donor simplex (set the new
        weight to zero), so admitting a donor cannot worsen the optimum."""
        X, y = _panel(N=15, T0=25, seed=4)
        sel, rem, path = [], list(range(15)), []
        for _ in range(10):
            j, r, _ = scan_candidates(X, y, sel, rem)
            sel.append(j); rem.remove(j); path.append(r)
        assert np.all(np.diff(path) <= 1e-9)

    def test_relabelling_the_donors_picks_the_same_one(self):
        X, y = _panel(N=13, T0=21, seed=6)
        T0 = X.shape[0]
        j0, r0, _ = scan_candidates(X, y, [], list(range(13)))
        p = np.random.default_rng(6).permutation(13)
        Xp = X[:, p]
        jp, rp, _ = scan_candidates(Xp, y, [], list(range(13)))
        assert p[jp] == j0
        assert rp == pytest.approx(r0, abs=1e-9)

    def test_the_returned_weights_are_on_the_simplex(self):
        X, y = _panel(N=11, T0=19, seed=8)
        _, _, w = scan_candidates(X, y, [0, 1], [2, 3, 4, 5])
        assert (w >= -1e-12).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------- edge cases
class TestDegenerateInput:
    def test_one_candidate_is_selected_by_default(self):
        X, y = _panel(N=4, T0=12, seed=2)
        j, r, w = scan_candidates(X, y, [0, 1], [3])
        assert j == 3
        want = simplex_lstsq(X[:, [0, 1, 3]], y)
        assert np.allclose(w, want, atol=1e-7)

    def test_two_pre_periods_still_scan(self):
        X, y = _panel(N=5, T0=2, seed=3)
        j, r, _ = scan_candidates(X, y, [], [0, 1, 2, 3, 4])
        assert np.isfinite(r)

    def test_a_duplicated_donor_does_not_break_the_scan(self):
        X, y = _panel(N=6, T0=16, seed=5)
        X[:, 3] = X[:, 1]
        j, r, w = scan_candidates(X, y, [1], [0, 2, 3, 4, 5])
        assert np.isfinite(r)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_a_constant_donor_does_not_produce_nan(self):
        X, y = _panel(N=6, T0=16, seed=13)
        X[:, 2] = 0.0
        _, r, w = scan_candidates(X, y, [], [0, 1, 2, 3])
        assert np.isfinite(r)
        assert np.isfinite(w).all()


# ---------------------------------------------------------------- failures
class TestBadInputIsReported:
    def test_an_empty_candidate_list_is_rejected(self):
        X, y = _panel(N=4, T0=10, seed=14)
        with pytest.raises(MlsynthEstimationError, match="no candidate"):
            scan_candidates(X, y, [0], [])

    def test_a_candidate_already_selected_is_rejected(self):
        X, y = _panel(N=4, T0=10, seed=15)
        with pytest.raises(MlsynthEstimationError, match="already selected"):
            scan_candidates(X, y, [1], [1, 2])

    def test_no_origins_is_rejected(self):
        X, y = _panel(N=4, T0=10, seed=16)
        with pytest.raises(MlsynthEstimationError, match="origin"):
            rolling_origin_rmspe_exact(X, y, [0], np.array([], dtype=int))


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
