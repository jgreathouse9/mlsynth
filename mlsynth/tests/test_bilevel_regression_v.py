"""The regression-based predictor weights: Stata ``synth``'s default V.

``synth`` only runs the nested optimisation when the caller passes ``nested``;
otherwise (``synth2.sthlp`` lines 157-162) it uses "a data-driven regression
based method to obtain the variable weights contained in the V-matrix". Since
``allsynth`` forwards ``nested`` as an ordinary optional flag, most published
``synth`` results were produced by this rule, not by the bilevel search --
so reproducing them needs it.

The Mata subroutines are shipped compiled (``l/lsynth2_mata_subr.mlib``); the
package has no source for them. What the library's symbol table does fix:

    normalize:  storemat  xtrout xcoout  quadvariance  diag  xtrmat xcomat
    regsval:    sval  xtrout xcoout ztrout zcoout  beta  trace  diag  vmat

and the only Mata builtins referenced anywhere in it are ``diag``,
``quadvariance`` and ``trace``. Hence: scale by the pooled treated-plus-control
standard deviation with no centring (no ``mean`` is referenced), regress the
pre-treatment outcomes of *both* blocks on the normalised predictors, and set
``V = diag(bb') / trace(bb')``.

Two things the symbols do NOT fix, and which are therefore options rather than
constants here: whether ``beta`` comes from one pooled regression or one per
pre-period (the solve is an opcode, not a name), and the tie-breaking among
equally-optimal ``V``. Both are exposed; neither is guessed silently.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------- fixtures
def _panel(T0=6, J=8, K=4, seed=0, scale=None):
    """Donor block, treated path and a predictor block on very different scales.

    ``scale`` deliberately gives the predictors wildly different units, which is
    what makes the normalisation step observable.
    """
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T0, 2))
    L = rng.normal(size=(J, 2))
    Y0 = F @ L.T + 0.2 * rng.normal(size=(T0, J))
    y1 = F @ rng.normal(size=2) + 0.2 * rng.normal(size=T0)
    X0 = rng.normal(size=(K, J))
    if scale is None:
        scale = np.logspace(0, 4, K)          # 1 ... 10000
    X0 = X0 * scale[:, None]
    X1 = X0 @ rng.dirichlet(np.ones(J)) + 0.01 * rng.normal(size=K) * scale
    return y1, Y0, X1, X0


def _prob(**kw):
    from mlsynth.utils.bilevel import BilevelProblem

    y1, Y0, X1, X0 = _panel(**kw)
    return BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0,
                          predictor_names=[f"x{k}" for k in range(X0.shape[0])])


# ---------------------------------------------------------------- smoke
class TestSmoke:
    def test_it_solves_and_returns_the_contract_type(self):
        from mlsynth.utils.bilevel import BilevelSolution, solve_bilevel

        sol = solve_bilevel(_prob(), method="regression")
        assert isinstance(sol, BilevelSolution)
        assert np.isfinite(sol.upper_loss)

    def test_the_weights_are_on_both_simplices(self):
        from mlsynth.utils.bilevel import solve_bilevel

        sol = solve_bilevel(_prob(), method="regression")
        for name, v in (("V", sol.V), ("W", sol.W)):
            v = np.asarray(v, float)
            assert (v >= -1e-9).all(), name
            assert v.sum() == pytest.approx(1.0, abs=1e-8), name

    def test_it_is_registered_as_a_backend(self):
        from mlsynth.utils.bilevel.engine import _BACKENDS

        assert "regression" in _BACKENDS


# ---------------------------------------------------------------- the rule
class TestTheRule:
    """Properties that follow from the recovered algorithm."""

    def test_normalising_before_regressing_is_not_the_same_as_after(self):
        """The load-bearing ordering.

        ``normalize`` runs on the predictors *before* ``regsval`` sees them, so
        the coefficients come from normalised regressors. Scaling afterwards --
        at the weighting step -- is a different estimator. On the Wiltshire
        panel that distinction moved the reported effect from -1.43 to -1.78
        against a target of -1.77, so it is pinned rather than left to taste.
        """
        from mlsynth.utils.bilevel.regression_v import regression_v

        y1, Y0, X1, X0 = _panel()
        v_before = regression_v(X1, X0, y1, Y0, normalize=True)
        v_after = regression_v(X1, X0, y1, Y0, normalize=False)
        assert not np.allclose(v_before, v_after, atol=1e-6)

    def test_the_treated_unit_enters_the_regression(self):
        """``regsval(Xtr_norm, Xco_norm, Ztr, Zco)`` takes both blocks."""
        from mlsynth.utils.bilevel.regression_v import regression_v

        y1, Y0, X1, X0 = _panel()
        with_treated = regression_v(X1, X0, y1, Y0)
        donors_only = regression_v(X1, X0, y1, Y0, include_treated=False)
        assert not np.allclose(with_treated, donors_only, atol=1e-6)

    def test_it_is_invariant_to_predictor_rescaling(self):
        """Normalising by the pooled standard deviation means the units a
        predictor happens to be measured in cannot change ``V``."""
        from mlsynth.utils.bilevel.regression_v import regression_v

        y1, Y0, X1, X0 = _panel()
        v1 = regression_v(X1, X0, y1, Y0)
        f = np.array([1.0, 1e3, 1e-2, 7.0])[: X0.shape[0]]
        v2 = regression_v(X1 * f, X0 * f[:, None], y1, Y0)
        np.testing.assert_allclose(v1, v2, atol=1e-8)

    def test_pooled_and_per_period_are_both_reachable(self):
        """The symbol table does not determine which one Stata runs, so the
        choice is exposed rather than hard-coded. They must differ, else the
        option would be vacuous."""
        from mlsynth.utils.bilevel.regression_v import regression_v

        y1, Y0, X1, X0 = _panel()
        a = regression_v(X1, X0, y1, Y0, pooled=True)
        b = regression_v(X1, X0, y1, Y0, pooled=False)
        assert not np.allclose(a, b, atol=1e-6)

    def test_a_predictor_uncorrelated_with_the_outcome_gets_little_weight(self):
        """``v_h`` is proportional to a squared regression coefficient, so a
        predictor that does not explain the pre-treatment outcome is
        down-weighted relative to one that does."""
        from mlsynth.utils.bilevel.regression_v import regression_v

        rng = np.random.default_rng(3)
        T0, J = 8, 10
        signal = rng.normal(size=J)
        Y0 = np.outer(np.ones(T0), signal) + 0.01 * rng.normal(size=(T0, J))
        y1 = np.full(T0, signal.mean())
        X0 = np.vstack([signal, rng.normal(size=J)])        # row 0 informative
        X1 = np.array([signal.mean(), 0.0])
        v = regression_v(X1, X0, y1, Y0)
        assert v[0] > v[1]


# ---------------------------------------------------------------- edges
class TestEdgeCases:
    def test_a_constant_predictor_does_not_divide_by_zero(self):
        from mlsynth.utils.bilevel.regression_v import regression_v

        y1, Y0, X1, X0 = _panel()
        X0[1, :] = 3.0
        X1[1] = 3.0
        v = regression_v(X1, X0, y1, Y0)
        assert np.isfinite(v).all()
        assert v.sum() == pytest.approx(1.0)

    def test_a_single_predictor_takes_all_the_weight(self):
        from mlsynth.utils.bilevel.regression_v import regression_v

        y1, Y0, X1, X0 = _panel(K=1)
        v = regression_v(X1, X0, y1, Y0)
        np.testing.assert_allclose(v, [1.0], atol=1e-12)

    def test_a_single_donor_gets_all_the_donor_weight(self):
        from mlsynth.utils.bilevel import solve_bilevel

        sol = solve_bilevel(_prob(J=1), method="regression")
        np.testing.assert_allclose(np.asarray(sol.W, float), [1.0], atol=1e-8)

    def test_degenerate_coefficients_fall_back_to_uniform(self):
        """If every coefficient is zero the normalisation is 0/0; the rule must
        degrade to equal weights rather than emit NaN."""
        from mlsynth.utils.bilevel.regression_v import regression_v

        rng = np.random.default_rng(5)
        T0, J, K = 5, 6, 3
        Y0 = np.zeros((T0, J))
        y1 = np.zeros(T0)
        X0 = rng.normal(size=(K, J))
        X1 = rng.normal(size=K)
        v = regression_v(X1, X0, y1, Y0)
        np.testing.assert_allclose(v, np.full(K, 1.0 / K), atol=1e-12)


# ---------------------------------------------------------------- failures
class TestFailuresAreReported:
    def test_an_unknown_method_still_names_the_valid_ones(self):
        from mlsynth.utils.bilevel import solve_bilevel

        with pytest.raises(ValueError, match="regression"):
            solve_bilevel(_prob(), method="not-a-backend")

    def test_the_backend_requires_covariates(self):
        """Like ``malo``/``mscmt``, a predictor-weight rule is meaningless with
        no predictors, and must say so rather than silently degrade."""
        from mlsynth.utils.bilevel import BilevelSCM

        y1, Y0, _, _ = _panel()
        with pytest.raises(ValueError, match="covariates"):
            BilevelSCM(backend="regression").fit(y1, Y0)

    def test_vanillasc_accepts_it_as_a_backend(self):
        import pandas as pd

        from mlsynth import VanillaSC

        rng = np.random.default_rng(0)
        rows = []
        for u in range(6):
            base = float(rng.normal()) * 3
            for t in range(10):
                rows.append({"unit": f"u{u}", "t": t,
                             "y": float(base + 0.4 * t + rng.normal() * 0.2),
                             "x1": float(rng.normal()),
                             "treat": int(u == 0 and t >= 7)})
        df = pd.DataFrame(rows)
        res = VanillaSC({"df": df, "outcome": "y", "treat": "treat",
                         "unitid": "unit", "time": "t", "covariates": ["x1"],
                         "backend": "regression", "inference": False,
                         "display_graphs": False}).fit()
        assert np.isfinite(res.effects.att)
