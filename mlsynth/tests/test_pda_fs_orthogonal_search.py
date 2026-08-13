"""The forward-selection search scores candidates by orthogonal projection.

Forward selection asks the same question at every step: which not-yet-selected
donor, added to the current set, leaves the smallest pre-period residual
variance? The definition of that question is a scan -- refit OLS once per
candidate and take the minimum -- which costs ``N`` least-squares solves per
step and dominates the run.

The answer is available without the scan. With ``r`` the current pre-period
residual and ``z_j`` donor ``j`` orthogonalised against the columns already
selected, adding ``j`` lowers the residual sum of squares by exactly

    gain_j = (x_j' r)^2 / ||z_j||^2,

because ``r`` is already orthogonal to the selected span. One matrix-vector
product ranks every candidate at once, and one rank-one downdate carries the
orthogonalisation to the next step.

These tests pin the identity. The oracle here is the definitional scan, frozen
as ``_scan_select`` below; the search under test must return its selection, its
coefficients and its counterfactual on ordinary panels and on the degenerate
ones -- duplicate donors, a zero column, an exactly collinear block, an
interpolating fit -- where the two rules could disagree on a tie. It must reach
them having asked for a bounded number of least-squares solves per step, not one
per donor.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
import pytest

from mlsynth.utils.pda_helpers.fs import forward_select
from mlsynth.utils.pda_helpers.fs import estimation as fs_est
from mlsynth.utils.pda_helpers.fs.estimation import _ols_sigma2


# ======================================================================
# The oracle: forward selection as defined, one OLS refit per candidate
# ======================================================================
def _scan_select(
    y: np.ndarray, X: np.ndarray, T0: int, intercept: bool = False,
) -> Tuple[List[int], np.ndarray, float, np.ndarray]:
    """Definitional forward selection, kept here as the reference to match.

    A frozen copy of the scan the fast search replaces: at each step refit OLS
    for every remaining donor, keep the one with the smallest ``mean(e**2)``,
    stop at the first step whose modified IC fails to improve.
    """
    y_pre, X_pre = y[:T0], X[:T0]
    N = X.shape[1]
    B = np.log(np.log(max(N, 3))) * np.log(T0) / T0
    IC = float(np.log(np.var(y_pre, ddof=1))) if intercept else float(np.log(np.mean(y_pre ** 2)))

    def _design(cols):
        return np.column_stack([np.ones(T0), X_pre[:, cols]]) if intercept else X_pre[:, cols]

    selected: List[int] = []
    remaining = list(range(N))
    for _ in range(max(0, T0 - 1 - int(intercept))):
        if not remaining:
            break
        best_j, best_s2 = None, np.inf
        for j in remaining:
            s2 = fs_est._ols_sigma2(y_pre, _design(selected + [j]))
            if s2 < best_s2:
                best_s2, best_j = s2, j
        IC_new = np.log(best_s2) + B * (len(selected) + 1)
        if IC_new < IC:
            IC = IC_new
            selected.append(best_j)
            remaining.remove(best_j)
        else:
            break

    if not selected:
        intercept_val = float(np.mean(y_pre)) if intercept else 0.0
        beta_full = np.zeros(N)
        return [], beta_full, intercept_val, np.full(X.shape[0], intercept_val)

    coef, *_ = np.linalg.lstsq(_design(selected), y_pre, rcond=None)
    beta_full = np.zeros(N)
    if intercept:
        intercept_val, beta_full[selected] = float(coef[0]), coef[1:]
    else:
        intercept_val, beta_full[selected] = 0.0, coef
    return selected, beta_full, intercept_val, X @ beta_full + intercept_val


def _assert_same_fit(fast, scan):
    sel_f, beta_f, icpt_f, cf_f = fast
    sel_s, beta_s, icpt_s, cf_s = scan
    assert sel_f == sel_s
    np.testing.assert_allclose(beta_f, beta_s, rtol=0, atol=1e-12)
    assert icpt_f == pytest.approx(icpt_s, abs=1e-12)
    np.testing.assert_allclose(cf_f, cf_s, rtol=1e-10, atol=1e-10)


def _factor_panel(T, T0, N, k=5, seed=0, noise=0.5):
    """A factor panel with ``k`` donors carrying the treated unit's signal."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 4))
    L = rng.normal(size=(4, N))
    X = F @ L + rng.normal(size=(T, N))
    y = X[:, :k] @ rng.normal(size=k) + noise * rng.normal(size=T)
    return y, X


# ======================================================================
# Smoke
# ======================================================================
class TestSmoke:

    def test_returns_the_scan_selection_on_a_sparse_panel(self):
        y, X = _factor_panel(60, 40, 30, k=3, seed=1)
        out = forward_select(y, X, T0=40)
        _assert_same_fit(out, _scan_select(y, X, T0=40))
        sel, beta, _, cf = out
        assert len(sel) >= 1
        assert beta.shape == (30,) and cf.shape == (60,)
        assert np.all(np.isfinite(cf))

    def test_counterfactual_is_the_ols_extrapolation_of_the_selected_set(self):
        y, X = _factor_panel(60, 40, 30, k=3, seed=2)
        sel, beta, icpt, cf = forward_select(y, X, T0=40)
        np.testing.assert_allclose(cf, X[:, sel] @ beta[sel] + icpt, atol=1e-10)


# ======================================================================
# The identity: same selection as the scan
# ======================================================================
class TestMatchesTheScan:

    @pytest.mark.parametrize("intercept", [False, True])
    @pytest.mark.parametrize("T,T0,N", [(60, 30, 40), (40, 20, 120), (25, 12, 60),
                                        (120, 80, 200)])
    @pytest.mark.parametrize("seed", range(6))
    def test_factor_panels(self, T, T0, N, seed, intercept):
        y, X = _factor_panel(T, T0, N, seed=seed)
        _assert_same_fit(forward_select(y, X, T0, intercept=intercept),
                         _scan_select(y, X, T0, intercept=intercept))

    @pytest.mark.parametrize("intercept", [False, True])
    @pytest.mark.parametrize("seed", range(8))
    def test_pure_noise_runs_deep_and_still_agrees(self, seed, intercept):
        """No signal: selection runs many steps on an ill-conditioned fit.

        This is where accumulated error in the orthogonalisation would show up,
        because each accepted donor is chosen on noise and the residual shrinks
        toward interpolation.
        """
        rng = np.random.default_rng(1000 + seed)
        T0, N = 10, 30
        y, X = rng.standard_normal(T0 + 5), rng.standard_normal((T0 + 5, N))
        _assert_same_fit(forward_select(y, X, T0, intercept=intercept),
                         _scan_select(y, X, T0, intercept=intercept))

    @pytest.mark.parametrize("intercept", [False, True])
    def test_more_donors_than_pre_periods(self, intercept):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((30, 200))
        y = X[:, :3].sum(axis=1) + 0.01 * rng.standard_normal(30)
        _assert_same_fit(forward_select(y, X, 12, intercept=intercept),
                         _scan_select(y, X, 12, intercept=intercept))


# ======================================================================
# Degenerate donor pools: where a tie has to break the same way
# ======================================================================
class TestDegenerateDonorPools:

    @staticmethod
    def _case(name, seed=7):
        rng = np.random.default_rng(seed)
        if name == "duplicate":
            X = rng.normal(size=(40, 6))
            X[:, 3] = X[:, 1]                      # an exact duplicate donor
            return 2.0 * X[:, 1] + 0.1 * rng.normal(size=40), X, 20
        if name == "zero_column":
            X = rng.normal(size=(40, 5))
            X[:, 2] = 0.0                          # a donor with no variation
            return X[:, 0] + 0.3 * rng.normal(size=40), X, 20
        if name == "collinear_block":
            A = rng.normal(size=(40, 2))           # six donors spanning two dims
            X = np.column_stack([A, A @ rng.normal(size=(2, 6))])
            return A[:, 0] + 0.2 * rng.normal(size=40), X, 20
        if name == "exact_fit":
            X = rng.normal(size=(40, 5))
            return 3.0 * X[:, 0], X, 20            # residual collapses to zero
        if name == "constant_outcome":
            return np.full(40, 2.0), rng.normal(size=(40, 5)), 20
        if name == "all_zero_donors":
            return rng.normal(size=40), np.zeros((40, 5)), 20
        if name == "single_donor":
            X = rng.normal(size=(40, 1))
            return X[:, 0] + 0.2 * rng.normal(size=40), X, 20
        if name == "tiny_pre_period":
            X = rng.normal(size=(10, 4))
            return X[:, 0] + 0.1 * rng.normal(size=10), X, 3
        raise AssertionError(name)                 # pragma: no cover

    _NAMES = ["duplicate", "zero_column", "collinear_block", "exact_fit",
              "constant_outcome", "all_zero_donors", "single_donor",
              "tiny_pre_period"]

    @pytest.mark.parametrize("intercept", [False, True])
    @pytest.mark.parametrize("name", _NAMES)
    def test_ties_break_where_the_scan_breaks_them(self, name, intercept):
        y, X, T0 = self._case(name)
        with np.errstate(divide="ignore", invalid="ignore"):
            _assert_same_fit(forward_select(y, X, T0, intercept=intercept),
                             _scan_select(y, X, T0, intercept=intercept))

    @pytest.mark.parametrize("intercept", [False, True])
    @pytest.mark.parametrize("name", ["zero_column", "collinear_block",
                                      "all_zero_donors", "duplicate"])
    def test_rank_deficient_donors_do_not_divide_by_zero(self, name, intercept):
        """A donor with no variation left has gain 0/0; the search must not ask.

        The projection score divides by ``||z_j||^2``, which is zero for a donor
        already in the selected span. Scoring those columns raises a numpy
        warning and yields a NaN that could win an ``argmax``.
        """
        y, X, T0 = self._case(name)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(divide="raise", invalid="raise"):
                sel, beta, icpt, cf = forward_select(y, X, T0, intercept=intercept)
        assert np.all(np.isfinite(beta)) and np.all(np.isfinite(cf))
        assert len(set(sel)) == len(sel)           # no donor selected twice

    @pytest.mark.parametrize("intercept", [False, True])
    @pytest.mark.parametrize("n_donors", [8, 39])
    def test_a_rank_one_pool_picks_an_interchangeable_donor(self, n_donors, intercept):
        """Every donor a multiple of one series: any of them is the same fit.

        The scan's winner here is whichever candidate its least-squares solve
        rounded smallest, which is not the lowest index and carries no meaning --
        all ``n_donors`` regressions have the same ``R^2``. Up to
        ``_SHORTLIST_MAX`` equivalent donors the exact comparison settles it the
        same way; past that the search may name a different member of the set,
        and the fit is what has to agree.
        """
        rng = np.random.default_rng(21)
        T, T0 = 60, 40
        base = rng.normal(size=(T, 1))
        X = base @ rng.normal(size=(1, n_donors))
        y = base[:, 0] + 0.1 * rng.normal(size=T)
        sel_f, _, _, cf_f = forward_select(y, X, T0, intercept=intercept)
        sel_s, _, _, cf_s = _scan_select(y, X, T0, intercept=intercept)

        assert len(sel_f) == len(sel_s) == 1
        np.testing.assert_allclose(cf_f, cf_s, rtol=1e-8, atol=1e-8)
        if n_donors <= fs_est._SHORTLIST_MAX:
            assert sel_f == sel_s

    @pytest.mark.parametrize("intercept", [False, True])
    @pytest.mark.parametrize("seed", range(6))
    def test_an_interpolating_panel_agrees_on_the_fit_not_the_dust(self, seed, intercept):
        """Noiseless donors: the two rules agree on the fit, not on the extras.

        When ``y`` lies exactly in the donor span the residual collapses to
        ~1e-16 after the donors that explain it, and both rules keep going: the
        IC carries ``log(sigma^2)``, which is still falling. On one such panel
        the path runs ``sigma^2`` = 8.5, 3.1e-1, 6.4e-31, 4.4e-31 -- so from the
        third step on, the accept test compares a ratio of two rounding
        artefacts against ``exp(-B)``. Which side it lands on is set by the BLAS
        that computed them, for the scan as much as for the projection score, so
        neither the extras nor *how many* extras there are is a property of the
        rule. Asserting the step count here is what made this test fail on CI's
        numpy while passing locally.

        What survives is the fit: the donors that explain the outcome, the
        counterfactual, and the extras carrying no weight.
        """
        rng = np.random.default_rng(400 + seed)
        T, T0, N = 50, 30, 36
        X = rng.normal(size=(T, N))
        y = X[:, :2] @ rng.normal(size=2)               # exact, no noise
        sel_f, beta_f, _, cf_f = forward_select(y, X, T0, intercept=intercept)
        sel_s, _, _, cf_s = _scan_select(y, X, T0, intercept=intercept)

        # The donors that explain y are columns 0 and 1; both rules find them,
        # in the same order, before the residual becomes dust.
        assert [j for j in sel_f if j < 2] == [j for j in sel_s if j < 2]
        scale = float(np.max(np.abs(cf_s)))
        np.testing.assert_allclose(cf_f, cf_s, rtol=0, atol=1e-10 * scale)
        extras = [j for j in sel_f if j >= 2]
        assert np.max(np.abs(beta_f[extras]), initial=0.0) < 1e-10

    def test_a_donor_with_no_variation_is_never_selected_over_a_useful_one(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(40, 4))
        X[:, 0] = 0.0
        y = X[:, 2] + 0.05 * rng.normal(size=40)
        sel, _, _, _ = forward_select(y, X, T0=40)
        assert sel[0] == 2


# ======================================================================
# The cost: least-squares solves per step, not per donor
# ======================================================================
class TestSearchCost:

    @staticmethod
    def _count_ols(monkeypatch):
        calls = {"n": 0}
        real = fs_est._ols_sigma2

        def counted(y, Z):
            calls["n"] += 1
            return real(y, Z)

        monkeypatch.setattr(fs_est, "_ols_sigma2", counted)
        return calls

    @pytest.mark.parametrize("N", [40, 400])
    def test_cost_does_not_grow_with_the_donor_pool(self, monkeypatch, N):
        """Ten-fold more donors, same number of candidate refits.

        The scan asks for one refit per remaining donor per step, so its cost is
        proportional to ``N``. The projection score ranks the pool with a single
        matrix-vector product and refits only the step's winner.
        """
        y, X = _factor_panel(200, 150, N, k=3, seed=4)
        calls = self._count_ols(monkeypatch)
        sel, _, _, _ = forward_select(y, X, T0=150)
        assert len(sel) >= 1
        # One verified candidate per step, plus the step that stops the search.
        assert calls["n"] <= 2 * (len(sel) + 1)

    def test_the_scan_is_the_thing_being_avoided(self, monkeypatch):
        """The oracle's cost is ``N`` per step, which is what motivates this."""
        y, X = _factor_panel(200, 150, 400, k=3, seed=4)
        calls = self._count_ols(monkeypatch)
        sel, _, _, _ = _scan_select(y, X, T0=150)
        assert calls["n"] > 300 * len(sel)

    def test_selection_still_honours_a_patched_sigma2(self, monkeypatch):
        """The IC path reads ``_ols_sigma2``, so the existing patches still bite."""
        monkeypatch.setattr(fs_est, "_ols_sigma2", lambda y, Z: 1e9)
        y, X = _factor_panel(40, 20, 30, seed=6)
        sel, beta, icpt, cf = forward_select(y, X, T0=20)
        assert sel == [] and np.allclose(cf, 0.0)


# ======================================================================
# Invariants that hold whatever the search does internally
# ======================================================================
class TestInvariants:

    @pytest.mark.parametrize("intercept", [False, True])
    @pytest.mark.parametrize("seed", range(5))
    def test_residual_variance_falls_at_every_accepted_step(self, seed, intercept):
        """Each accepted donor lowers ``mean(e**2)``; that is what it is chosen for."""
        y, X = _factor_panel(60, 40, 50, seed=seed)
        sel, _, _, _ = forward_select(y, X, T0=40, intercept=intercept)
        y_pre, X_pre = y[:40], X[:40]

        def s2(cols):
            Z = (np.column_stack([np.ones(40), X_pre[:, cols]]) if intercept
                 else X_pre[:, cols])
            if Z.shape[1] == 0:
                return float(np.mean(y_pre ** 2))
            return _ols_sigma2(y_pre, Z)

        path = [s2(sel[:k]) for k in range(len(sel) + 1)]
        assert all(b <= a + 1e-12 for a, b in zip(path, path[1:]))

    @pytest.mark.parametrize("intercept", [False, True])
    @pytest.mark.parametrize("seed", range(5))
    def test_the_step_winner_is_the_best_remaining_donor(self, seed, intercept):
        """Greedy optimality: no unselected donor beats the first pick."""
        y, X = _factor_panel(60, 40, 50, seed=seed)
        sel, _, _, _ = forward_select(y, X, T0=40, intercept=intercept)
        if not sel:
            pytest.skip("no donor selected on this panel")
        y_pre, X_pre = y[:40], X[:40]

        def s2(j):
            Z = (np.column_stack([np.ones(40), X_pre[:, [j]]]) if intercept
                 else X_pre[:, [j]])
            return _ols_sigma2(y_pre, Z)

        chosen = s2(sel[0])
        assert all(s2(j) >= chosen - 1e-12 for j in range(X.shape[1]))

    @pytest.mark.parametrize("intercept", [False, True])
    def test_selection_is_capped_below_the_pre_period(self, intercept):
        rng = np.random.default_rng(0)
        T0, N = 12, 40
        y, X = rng.standard_normal(T0 + 5), rng.standard_normal((T0 + 5, N))
        sel, _, _, cf = forward_select(y, X, T0, intercept=intercept)
        assert len(sel) + int(intercept) <= T0 - 1
        assert np.mean((y[:T0] - cf[:T0]) ** 2) > 1e-12

    @pytest.mark.parametrize("intercept", [False, True])
    def test_selection_is_invariant_to_donor_rescaling(self, intercept):
        """Scaling a donor rescales its coefficient and leaves the fit alone."""
        y, X = _factor_panel(60, 40, 40, seed=8)
        scale = np.exp(np.random.default_rng(9).normal(size=40))
        sel_a, beta_a, _, cf_a = forward_select(y, X, 40, intercept=intercept)
        sel_b, beta_b, _, cf_b = forward_select(y, X * scale, 40, intercept=intercept)
        assert sel_a == sel_b
        np.testing.assert_allclose(beta_b * scale, beta_a, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(cf_b, cf_a, rtol=1e-8, atol=1e-8)

    def test_donor_order_does_not_change_the_fitted_counterfactual(self):
        """Permuting the pool permutes the support; the counterfactual holds."""
        y, X = _factor_panel(60, 40, 40, seed=10)
        perm = np.random.default_rng(11).permutation(40)
        sel_a, beta_a, _, cf_a = forward_select(y, X, 40)
        sel_b, beta_b, _, cf_b = forward_select(y, X[:, perm], 40)
        assert set(perm[sel_b]) == set(sel_a)
        np.testing.assert_allclose(cf_b, cf_a, rtol=1e-8, atol=1e-8)
