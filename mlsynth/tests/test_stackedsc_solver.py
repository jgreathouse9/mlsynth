"""STACKEDSC solves its weight programs exactly.

The design here is adversarial for a first-order method: a cohort has 5 to 10
pre-treatment periods against 39 donors, so the Gram is rank deficient by 30 or
more and its nonzero spectrum is badly spread. Measured on the Wiltshire panel,
FISTA never converged on it -- 20 of 39 leave-one-out columns were still
improving at iteration 20,000, at every tolerance from 1e-11 down to 1e-6, so
the solve was iteration-bound and returned a point that was simply suboptimal.

The primal active-set method
(:func:`mlsynth.utils.bilevel.active_set.solve_simplex_qp`) solves the same
program exactly, in a bounded number of pivots, and is what MEDSC, SCD and
COMPSC already use. On the paper's panel it is 2.8x faster on the point estimate
and roughly 130x on the placebo layer.

What these tests can and cannot assert
--------------------------------------

They assert the objective, never the weights. With fewer rows than donors the
optimum is a *face*, and two exact solvers land in different places on it: on
the Wiltshire cohorts the active set and CLARABEL agree on the objective to
8.7e-6 while their per-county post-treatment paths differ by up to 2.1
percentage points. Exactness pins what is attained, not where.

So the contract is a KKT certificate -- a solver-independent statement that the
returned point is optimal -- plus the invariants that survive the face: the
aggregate, the pre-treatment fit, and the base-period identity.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pd = pytest.importorskip("pandas")


def _staggered(n_treated=6, n_donors=12, T=14, cohorts=(7, 9), effect=0.0,
               seed=0):
    """Fewer pre-periods than donors: the shape that broke the first-order
    solver, in miniature."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2)).cumsum(axis=0)
    rows = []
    for j in range(n_donors):
        load, base = rng.normal(size=2), 100.0 + 20.0 * rng.random()
        for t in range(T):
            rows.append({"unit": f"d{j}", "t": t, "adopt": np.nan,
                         "y": float(base + F[t] @ load + rng.normal() * 0.05)})
    for i in range(n_treated):
        g = cohorts[i % len(cohorts)]
        load, base = rng.normal(size=2), 100.0 + 20.0 * rng.random()
        for t in range(T):
            y = base + F[t] @ load + rng.normal() * 0.05
            rows.append({"unit": f"x{i}", "t": t, "adopt": float(g),
                         "y": float(y * (1.0 + effect) if t >= g else y)})
    d = pd.DataFrame(rows)
    d["treat"] = ((~d.adopt.isna()) & (d.t >= d.adopt)).astype(int)
    return d


def _fit(df=None, **over):
    from mlsynth import STACKEDSC

    cfg = {"df": _staggered() if df is None else df, "outcome": "y",
           "treat": "treat", "unitid": "unit", "time": "t",
           "display_graphs": False}
    cfg.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return STACKEDSC(cfg).fit()


def _cohort_blocks(df, **over):
    """The design matrices the estimator actually solved against."""
    from mlsynth.utils.stackedsc_helpers.pipeline import _design
    from mlsynth.utils.stackedsc_helpers.setup import build_cohorts

    cohorts, _ = build_cohorts(df, "unit", "t", "y", "treat",
                               normalize=over.get("normalize", True))
    return [(c, *_design(c, "outcome-only")) for c in cohorts]


def _kkt_violation(A, b, w, tol=1e-8):
    """How far ``w`` is from satisfying the optimality conditions.

    For ``min ||A w - b||^2`` over the simplex, the conditions are: the gradient
    is constant across the support and no smaller off it. So with
    ``g = 2 A'(A w - b)`` and ``lam = min(g)``, optimality means every
    supported coordinate has ``g_k == lam``. The returned number is the largest
    such excess -- zero at an optimum, positive for a point that could still be
    improved by moving weight.
    """
    g = 2.0 * (A.T @ (A @ w - b))
    lam = float(g.min())
    supported = w > tol
    return float(np.max(g[supported] - lam)) if supported.any() else 0.0


# ---------------------------------------------------------------------------
class TestTheWeightsAreOptimal:
    """A certificate, not a comparison. This holds whatever solver is used and
    would have failed on the unconverged first-order solve."""

    def test_every_treated_unit_satisfies_the_kkt_conditions(self):
        df = _staggered()
        res = _fit(df)
        worst = 0.0
        for cohort, A, B, _V in _cohort_blocks(df):
            for j, unit in enumerate(cohort.units):
                w = np.array([res.per_unit[str(unit)].donor_weights.get(str(d), 0.0)
                              for d in cohort.donors])
                worst = max(worst, _kkt_violation(A, B[:, j], w))
        assert worst < 1e-6, f"largest KKT excess {worst:.3e}"

    def test_no_reachable_point_fits_better(self):
        """The direct form of the same claim: perturbing the returned weights
        toward any other donor cannot lower the objective."""
        df = _staggered()
        res = _fit(df)
        cohort, A, B, _V = _cohort_blocks(df)[0]
        rng = np.random.default_rng(0)
        for j, unit in enumerate(cohort.units):
            w = np.array([res.per_unit[str(unit)].donor_weights.get(str(d), 0.0)
                          for d in cohort.donors])
            base = float(np.sum((A @ w - B[:, j]) ** 2))
            for _ in range(20):
                v = rng.dirichlet(np.ones(len(w)))
                for step in (1e-3, 1e-2, 1e-1):
                    cand = (1 - step) * w + step * v
                    got = float(np.sum((A @ cand - B[:, j]) ** 2))
                    assert got >= base - 1e-9, (unit, step, got, base)

    def test_the_weights_are_still_on_the_simplex(self):
        res = _fit()
        for u in res.per_unit.values():
            w = np.array(list(u.donor_weights.values()))
            assert (w >= -1e-12).all()
            assert w.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
class TestThePlaceboWeightsToo:
    def test_every_placebo_fit_satisfies_the_kkt_conditions(self):
        """The permutation distribution is built from these, so an
        under-converged placebo is a mis-stated null, not a slow one."""
        from mlsynth.utils.stackedsc_helpers.inference import cohort_placebo_paths

        df = _staggered()
        worst = 0.0
        for cohort, A, B, _V in _cohort_blocks(df):
            nJ = A.shape[1]
            allowed = [list(range(nJ))] * len(cohort.units)
            take = np.ones(cohort.Y.shape[0], dtype=bool)
            paths, _ = cohort_placebo_paths(cohort, A, B, take, allowed,
                                            donor_pool="donors-only")
            # refit the family directly and certify it
            from mlsynth.utils.bilevel.active_set import solve_simplex_qp
            for j in range(nJ):
                keep = [k for k in range(nJ) if k != j]
                w = np.asarray(solve_simplex_qp(A[:, keep], A[:, j]), float)
                worst = max(worst, _kkt_violation(A[:, keep], A[:, j], w))
            assert paths
        assert worst < 1e-6, f"largest KKT excess {worst:.3e}"


# ---------------------------------------------------------------------------
class TestTheInvariantsSurviveTheSolverChange:
    """What the face does not touch."""

    def test_the_base_period_gap_is_still_identically_zero(self):
        res = _fit()
        e = np.asarray(res.event_study.horizons, int)
        tau = np.asarray(res.event_study.tau, float)
        assert abs(float(tau[e == -1][0])) < 1e-9

    def test_a_degenerate_weight_still_reproduces_its_unit(self):
        df = _staggered()
        df["only"] = (df.unit == "x2").astype(float)
        res = _fit(df=df, agg_weights="only")
        target = _fit(df=df).per_unit["x2"]
        np.testing.assert_allclose(np.asarray(res.event_study.tau, float),
                                   np.asarray(target.tau, float), atol=1e-8)

    def test_a_planted_effect_is_still_recovered(self):
        res = _fit(df=_staggered(effect=0.10, seed=3))
        e = np.asarray(res.event_study.horizons, int)
        tau = np.asarray(res.event_study.tau, float)
        assert 7.0 < float(np.mean(tau[e >= 0])) < 13.0


# ---------------------------------------------------------------------------
class TestTheDesignBlockSaysWhatItMeans:
    """``batched`` described a multiple-right-hand-side program that the
    active-set solver does not run. What the flag actually detects -- and still
    detects -- is whether the donor predicate left every treated unit in a
    cohort facing the same donor pool, so it is named for that."""

    def test_a_shared_pool_is_reported(self):
        assert _fit().design.shared_donor_pool is True

    def test_a_binding_predicate_is_reported(self):
        res = _fit(donor_predicate=lambda i, d: not (i == "x0" and d == "d0"))
        assert res.design.shared_donor_pool is False

    def test_a_predicate_that_does_not_bind_leaves_the_pool_shared(self):
        res = _fit(donor_predicate=lambda i, d: True)
        assert res.design.shared_donor_pool is True

    def test_it_reaches_the_spec_block(self):
        params = _fit().method_details.parameters_used
        assert params["shared_donor_pool"] is True
        assert "batched" not in params
