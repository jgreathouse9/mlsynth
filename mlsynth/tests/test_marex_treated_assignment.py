"""Which group MAREX treats is decided by the program, not inferred from it.

MAREX solves for treated weights ``w``, control weights ``v`` and a binary
selection ``z``. The three are not interchangeable: ``w <= z`` and
``sum(z) == m_eq`` put the cardinality on ``w``, the budget prices ``w``
(``sum(c * w) <= B``), and the geographic restrictions act on ``z`` and ``v``.
Swapping the two weight vectors is therefore not a symmetry of the feasible set
whenever any of those is active -- the solver's ``w`` is the treated group, and
there is nothing left to work out.

The reporting worked it out anyway. It relabelled the two groups so the treated
one was whichever had the smaller support, ties broken by the earlier first
index, citing Abadie & Zhao's preference for treating few units. On a 12-market
panel that convention overrode the solver: asked for ``m_eq=6`` treated markets
it reported 3, and they were the control synthetic's markets. The number of
markets an experimenter is told to treat was then neither the number they asked
for nor the answer to the program that was solved.

The convention is not wrong everywhere. The convention is right where the labelling really is free, and that case is
reachable: under the symmetric ``standard`` objective, bounds as loose as
``m_min=1, m_max=N-1`` admit the complement as a treated set too, so
``(w, v, z)`` and ``(v, w, 1 - z)`` are both optimal. It is kept there -- it is
what the authors' code does, and the Section 5 Monte Carlo benchmark compares
that cell against their numbers.
:class:`TestTheAmbiguousCaseKeepsTheConvention` pins both halves.

:class:`TestAgreesWithLexscmAndBruteForce` is the other reason this matters. Put
on the same fit window and the same standardisation, MAREX's treated subproblem
and LEXSCM's Stage-1 search are the same program in the same metric, solved two
entirely different ways -- a mixed-integer quadratic program in SCIP against an
enumeration over subsets with Wolfe's active set inside. Agreeing is a
cross-validation of both. They did not agree while the labels could be
overridden.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mlsynth import MAREX
from mlsynth.estimators import lexscm as lexscm_mod
from mlsynth.estimators.lexscm import LEXSCM
from mlsynth.utils.fast_scm_helpers.lexsearch import select_treated_designs
from mlsynth.utils.marex_helpers import orchestration as mx_orch

N_UNITS, T0, T_POST, FRAC_E = 12, 20, 6, 0.7
# LEXSCM fits on the first ``frac_E`` of the pre-period and holds the rest
# blank; MAREX takes the same split as a count of blank periods.
BLANK = T0 - int(T0 * FRAC_E)


def _panel(seed: int, n=N_UNITS) -> pd.DataFrame:
    """A three-factor panel with a pre/post split, deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    T = T0 + T_POST
    F = np.cumsum(rng.normal(0, 1, (T, 3)), axis=0)
    L = rng.normal(0, 1, (n, 3))
    Y = 100 + F @ L.T + rng.normal(0, 1.0, (T, n))
    return pd.DataFrame(
        [{"unit": f"u{j:02d}", "time": t, "Y": float(Y[t, j]),
          "elig": True, "post": int(t >= T0)}
         for j in range(n) for t in range(T)])


def _fit_capturing_solver(df, **over):
    """Run MAREX and return ``(result, w_support, v_support)`` from the solve.

    The supports come from ``solve_design``'s own return, before any reporting
    touches them, so the test compares what was reported against what was
    solved for.
    """
    cfg = dict(df=df, outcome="Y", unitid="unit", time="time", post_col="post",
               design="weakly_targeted", standardize=True,
               blank_periods=BLANK, verbose=False, display_graph=False)
    cfg.update(over)
    # ``beta`` ties the control synthetic to the treated one and the config
    # accepts it only for the weakly-targeted design. Near zero it leaves the
    # treated term alone, which is what makes that design comparable to
    # LEXSCM's Stage 1.
    if cfg["design"] == "weakly_targeted":
        cfg.setdefault("beta", 1e-9)
    else:
        cfg.pop("beta", None)
    cap = {}
    original = mx_orch.solve_design

    def recording(*args, **kwargs):
        out = original(*args, **kwargs)
        cap["raw"] = out
        return out

    mx_orch.solve_design = recording
    try:
        res = MAREX(cfg).fit()
    finally:
        mx_orch.solve_design = original
    labels = sorted({str(u) for u in df["unit"].unique()})
    w = np.asarray(cap["raw"]["w_opt"]).sum(axis=1)
    v = np.asarray(cap["raw"]["v_opt"]).sum(axis=1)
    sup = lambda x: sorted(labels[i] for i in np.where(x > 1e-8)[0])
    return res, sup(w), sup(v)


# =========================================================================
# 1. The solver's w is the treated group
# =========================================================================

class TestTheSolverDecidesWhoIsTreated:
    """What is reported as treated is what the program treated."""

    @pytest.mark.parametrize("seed,m", [(3, 4), (1, 6), (5, 6), (2, 7)])
    def test_reported_treated_is_the_solvers_w(self, seed, m):
        res, w_sup, v_sup = _fit_capturing_solver(_panel(seed), m_eq=m)
        reported = sorted(str(u) for u in res.selected_units)
        assert reported == w_sup, (
            f"reported treated {reported} is not the solver's treated weights "
            f"{w_sup} (it is the control synthetic {v_sup})"
            if reported == v_sup else
            f"reported treated {reported} != solver's {w_sup}")

    @pytest.mark.parametrize("seed", [1, 3, 5])
    @pytest.mark.parametrize("m", [3, 6])
    def test_cardinality_is_what_was_asked_for(self, seed, m):
        """``m_eq`` treated markets were requested, so ``m_eq`` come back."""
        res, _, _ = _fit_capturing_solver(_panel(seed), m_eq=m)
        assert len(res.selected_units) == m

    def test_treated_and_control_are_disjoint(self):
        res, _, _ = _fit_capturing_solver(_panel(3), m_eq=4)
        cluster = list(res.clusters.values())[0]
        treated = {str(k) for k in cluster.unit_weight_map["Treated"]}
        control = {str(k) for k in cluster.unit_weight_map["Control"]}
        assert treated and control
        assert not (treated & control)

    def test_the_synthetic_treated_series_is_built_from_the_solvers_w(self):
        """The labels are not cosmetic: the effect is built from them.

        ``synthetic_treated`` is the treated weights against the outcome panel
        and the estimated effect is it minus ``synthetic_control``, so
        exchanging the two vectors negates the effect exactly. On this panel
        the post-period mean gap read -0.5995 with the solver's assignment and
        +0.5995 with the groups exchanged -- same magnitude, opposite sign.
        """
        df = _panel(3)
        res, w_sup, _ = _fit_capturing_solver(df, m_eq=4)
        wide = df.pivot(index="unit", columns="time", values="Y").sort_index()
        weights = list(res.clusters.values())[0].unit_weight_map["Treated"]
        expected = sum(float(wt) * wide.loc[str(u)].to_numpy()
                       for u, wt in weights.items())
        assert sorted(str(u) for u in weights) == w_sup
        assert np.allclose(np.asarray(res.globres.synthetic_treated), expected,
                           rtol=1e-8, atol=1e-8)

    def test_the_budget_prices_the_group_that_is_reported_treated(self):
        """The cost bound constrains ``w``, so it must bind what is called treated.

        MAREX's bound is :math:`\\sum_j c_j w_j \\le B` on the treated weights,
        which since they sum to one is a cap on the weighted mean cost of the
        treated group. Nothing prices ``v``, so if the control synthetic were
        reported as treated the returned design would carry no cost guarantee
        at all.
        """
        df = _panel(3)
        units = sorted(df["unit"].unique())
        costs = [1.0 + 3.0 * (i % 4) for i in range(len(units))]
        budget = 5          # below the unweighted mean cost, so it binds
        res, w_sup, _ = _fit_capturing_solver(
            df, m_eq=4, costs=costs, budget=budget)
        price = dict(zip(units, costs))
        treated = list(res.clusters.values())[0].unit_weight_map["Treated"]
        weighted = sum(price[str(u)] * float(wt) for u, wt in treated.items())
        assert sorted(str(u) for u in treated) == w_sup
        assert weighted <= budget + 1e-9, (
            f"reported treated group has weighted cost {weighted:.3f} against "
            f"a {budget} bound")


# =========================================================================
# 2. Where the labelling really is free, keep the convention
# =========================================================================

class TestTheAmbiguousCaseKeepsTheConvention:
    """Where either labelling is feasible at the same cost, pick one and say so.

    ``m_min=1, m_max=N-1`` bounds the treated size without pinning it: the
    complement of any admissible set is admissible too. Under the symmetric
    ``standard`` objective ``(w, v, z)`` and ``(v, w, 1 - z)`` are then both
    optimal, so a convention is needed and Abadie & Zhao's is the one to use --
    it is what their own code does, and ``benchmarks/cases/marex_section5_mc``
    compares this cell against their Monte Carlo.
    """

    def test_loose_bounds_report_the_smaller_group_as_treated(self):
        res, w_sup, v_sup = _fit_capturing_solver(
            _panel(3), design="standard", m_min=1, m_max=N_UNITS - 1)
        reported = sorted(str(u) for u in res.selected_units)
        assert len(reported) <= len(w_sup) or len(reported) <= len(v_sup)
        assert len(reported) == min(len(w_sup), len(v_sup))

    def test_a_cardinality_constraint_is_still_required(self):
        """Omitting the bounds entirely is rejected, so the search space is
        always bounded even where the labelling is free."""
        from mlsynth.exceptions import MlsynthConfigError
        with pytest.raises(MlsynthConfigError, match="m_eq|m_min|m_max"):
            _fit_capturing_solver(_panel(3), design="standard")

    def test_bounds_that_exclude_the_complement_pin_the_labels(self):
        """``m_max`` below the complement's size makes the swap infeasible."""
        res, w_sup, _ = _fit_capturing_solver(
            _panel(3), design="standard", m_min=1, m_max=4)
        reported = sorted(str(u) for u in res.selected_units)
        assert reported == w_sup
        assert 1 <= len(reported) <= 4

    def test_an_asymmetric_objective_pins_the_labels(self):
        """Only ``standard`` matches both synthetics to the mean; under
        ``weakly_targeted`` only ``w`` is targeted, so swapping changes the
        objective even with the bounds wide open."""
        res, w_sup, _ = _fit_capturing_solver(
            _panel(3), m_min=1, m_max=N_UNITS - 1)
        assert sorted(str(u) for u in res.selected_units) == w_sup


# =========================================================================
# 3. Cross-validation: MAREX against LEXSCM against brute force
# =========================================================================

def _simplex_min(Q):
    """``min_{w in simplex} w'Qw`` by enumerating the faces of the simplex."""
    m = Q.shape[0]
    best = np.inf
    for r in range(1, m + 1):
        for T_ in combinations(range(m), r):
            T_ = list(T_)
            QT = Q[np.ix_(T_, T_)]
            K = np.zeros((r + 1, r + 1))
            K[:r, :r] = QT
            K[:r, r] = -1.0
            K[r, :r] = 1.0
            rhs = np.zeros(r + 1)
            rhs[r] = 1.0
            sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
            if np.linalg.norm(K @ sol - rhs) > 1e-9:
                continue
            w = sol[:r]
            if w.min() < -1e-11:
                continue
            best = min(best, float(w @ QT @ w))
    return best


def _lexscm_gram(df, m):
    """The Gram LEXSCM builds, captured on the way into its Stage-1 search."""
    grab = {}

    class Stop(Exception):
        pass

    def recording(*args, **kwargs):
        grab.update(G=np.asarray(kwargs["G"], dtype=float),
                    cand=list(kwargs["candidate_idx"]),
                    labels=[str(x) for x in kwargs["unit_index"].labels])
        raise Stop

    original = lexscm_mod.select_treated_designs
    lexscm_mod.select_treated_designs = recording
    try:
        LEXSCM(dict(df=df, outcome="Y", unitid="unit", time="time",
                    candidate_col="elig", post_col="post", m=m, top_K=3,
                    display_graph=False, verbose=False)).fit()
    except Stop:
        pass
    finally:
        lexscm_mod.select_treated_designs = original
    return grab["G"], grab["cand"], grab["labels"]


class TestAgreesWithLexscmAndBruteForce:
    """Two solvers and an enumeration on one program, in one metric."""

    @pytest.mark.parametrize("seed,m", [(0, 3), (3, 4), (4, 3)])
    def test_all_three_pick_the_same_treated_set(self, seed, m):
        df = _panel(seed)
        G, cand, labels = _lexscm_gram(df, m)

        lex = select_treated_designs(G, cand, m=m, top_K=1, method="enumerate")
        assert lex["stats"]["termination"]["status"] == "OPTIMAL"
        lex_set = sorted(labels[i] for i in lex["top_designs"][0].indices)

        triples = list(combinations(range(len(labels)), m))
        losses = [_simplex_min(G[np.ix_(list(S), list(S))]) for S in triples]
        brute_set = sorted(labels[i] for i in triples[int(np.argmin(losses))])

        res, _, _ = _fit_capturing_solver(df, m_eq=m)
        marex_set = sorted(str(u) for u in res.selected_units)

        assert brute_set == lex_set, "brute force and LEXSCM must agree"
        assert marex_set == lex_set, (
            f"MAREX {marex_set} != LEXSCM/brute force {lex_set} on the same "
            f"program in the same metric")

    def test_the_two_metrics_really_are_the_same(self):
        """The agreement above is only meaningful if the Grams coincide.

        Aligned (same fit window, both standardised by the cross-unit spread
        per period, both centred on the unweighted population mean), MAREX's
        treated term ``||Xbar - X'w||^2`` and LEXSCM's ``w'Gw`` are one matrix.
        Without this the test would be comparing two different problems and
        agreement would be luck.
        """
        from mlsynth.utils.marex_helpers import optimization as mx_opt
        df = _panel(0)
        G, _, _ = _lexscm_gram(df, 3)
        cap = {}
        aug, means = mx_opt._augment_fit, mx_opt.compute_cluster_means_members

        def rec_aug(*a, **k):
            out = aug(*a, **k)
            cap["X"] = np.asarray(out, dtype=float)
            return out

        def rec_means(*a, **k):
            Xb, mem = means(*a, **k)
            cap["Xbar"] = np.asarray(Xb[0], dtype=float)
            return Xb, mem

        mx_opt._augment_fit, mx_opt.compute_cluster_means_members = rec_aug, rec_means
        try:
            _fit_capturing_solver(df, m_eq=3)
        finally:
            mx_opt._augment_fit = aug
            mx_opt.compute_cluster_means_members = means

        R = (cap["X"] - cap["Xbar"][None, :]).T
        G_marex = R.T @ R
        assert np.abs(G_marex - G).max() < 1e-9 * max(np.abs(G).max(), 1.0)
