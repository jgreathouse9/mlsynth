"""Does LEXSCM's ``OPTIMAL`` termination status mean what it says?

Stage 1 of LEXSCM picks the treated set. When it enumerates
(:func:`~mlsynth.utils.fast_scm_helpers.lexsearch.select_treated_designs` with
``C(M, m) <= enumerate_max``) it reports termination status ``OPTIMAL``, and the
claim behind that word is

    the returned ``top_designs`` are the ``top_K`` feasible treated
    :math:`m`-tuples of smallest imbalance
    :math:`L(S) = \\min_{w \\in \\Delta(S)} w' G_{SS} w` over the entire feasible
    region -- the candidate pool intersected with the budget, spillover,
    stratum-quota and forced-unit constraints -- and each design's weights are
    the exact minimiser of its own inner QP.

The search does not check that. ``exact = True`` is set because the enumeration
branch was taken, and the status is read off ``exact``; the word is a statement
about the algorithm, not a quantity measured on the instance. The claim rests on
three things holding together, and each is checked here against a reference that
shares no code with the search:

1. The inner QP is solved exactly (:class:`TestInnerWeightCertificate`). Twice
   over: the returned weights satisfy the KKT conditions of the simplex program,
   computed from the Gram alone and so independent of whatever produced them;
   and the loss agrees with a solver that enumerates every face of the simplex
   instead of running Wolfe's active set.
2. The feasible region is covered (:class:`TestFeasibleRegionIsCovered`). The
   count of tuples scored equals the count an independent ``itertools``
   enumeration with plain-Python constraint predicates finds feasible.
3. The right ``K`` come back (:class:`TestGlobalTopK`,
   :class:`TestDegenerateGrams`). Returned tuples and losses match the reference
   brute force under every constraint regime, under the weak-targeting ridge,
   and on Grams that are rank deficient or carry duplicate donors.

Between 1 and 3 sits the step that could break either silently: the enumeration
ranks with the batched solver and then re-solves the surviving ``K`` at high
precision, so a batched loss that is merely close would seat the wrong tuples in
a pool whose reported losses are all correct. :class:`TestBatchedRankingIsExact`
measures that gap on the panel geometry that provokes it.

:class:`TestCertificateCheckHasPower` checks the checks. With the enumeration
deliberately broken -- the optimum dropped, the ranking perturbed, a constraint
skipped -- the comparisons here have to fail. A verification that passes on a
broken search verifies nothing.

``OPTIMAL`` is the only status carrying the claim, so :class:`TestStatusDiscipline`
pins the discipline: the heuristic path reports ``FEASIBLE``, an empty feasible
region reports ``INFEASIBLE``, and the advisory convex lower bound is never
turned into an optimality gap.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers import lexsearch as ls
from mlsynth.utils.fast_scm_helpers.lexsearch import (
    _afw_single,
    _losses_for,
    select_treated_designs,
)

# Agreement tolerances. Measured worst case between the two solvers over the
# instances below is ~1e-14 relative, so these are loose by five orders.
RTOL = 1e-9
ATOL = 1e-12
# A gap this small between two losses is a tie, not an ordering.
TIE = 1e-10


# =========================================================================
# Reference implementations -- no code shared with the search
# =========================================================================

def _face_min(Q: np.ndarray) -> tuple[float, np.ndarray]:
    """``min_{w in simplex} w'Qw`` by enumerating the faces of the simplex.

    A minimiser of a PSD quadratic form on the simplex lies in the relative
    interior of the face spanned by its own support, where it is a stationary
    point of the equality-constrained problem on that face's affine hull:
    ``Q_TT w_T = mu 1``, ``1'w_T = 1``. Enumerating the ``2^m - 1`` faces and
    keeping the smallest value attained at a nonnegative stationary point
    returns the global minimum exactly. Every accepted point is feasible, so the
    result is an upper bound; the true minimiser's own face supplies it, so the
    bound is attained.

    Exponential in ``m`` and useless inside a search. Here ``m <= 4``, and the
    method has nothing in common with Wolfe's active set, which is the point.
    """
    Q = np.asarray(Q, dtype=float)
    m = Q.shape[0]
    best, best_w = np.inf, None
    for r in range(1, m + 1):
        for T in combinations(range(m), r):
            T = list(T)
            QT = Q[np.ix_(T, T)]
            K = np.zeros((r + 1, r + 1))
            K[:r, :r] = QT
            K[:r, r] = -1.0          # -mu, the sum-to-one multiplier
            K[r, :r] = 1.0
            rhs = np.zeros(r + 1)
            rhs[r] = 1.0
            sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
            if np.linalg.norm(K @ sol - rhs) > 1e-9:
                continue             # singular face, no stationary point
            w_T = sol[:r]
            if w_T.min() < -1e-11:
                continue             # stationary point is off this face
            val = float(w_T @ QT @ w_T)
            if val < best:
                w = np.zeros(m)
                w[T] = w_T
                best, best_w = val, w
    assert best_w is not None, "every vertex face is solvable, so this cannot happen"
    return best, best_w


def _kkt_violation(Q: np.ndarray, w: np.ndarray) -> float:
    """Normalised violation of simplex-QP optimality at ``w``, from ``Q`` alone.

    With ``g = Qw`` and ``nu = w'Qw``, the KKT conditions of
    ``min_{w in simplex} w'Qw`` are ``g_j >= nu`` for every ``j``, with equality
    wherever ``w_j > 0``. The equality half is implied: ``nu = sum_j w_j g_j``, so
    if every ``g_j >= nu`` on a simplex point then ``g_j = nu`` on the support.
    One inequality is therefore the whole certificate, and it is computed from
    the Gram, so it holds whatever solver produced ``w``. This is the Gram-form
    twin of :func:`mlsynth.utils.solvers.minnorm.simplex_point_is_optimal`.
    """
    g = np.asarray(Q, dtype=float) @ np.asarray(w, dtype=float)
    nu = float(np.asarray(w) @ g)
    scale = 1.0 + float(np.abs(g).max())
    return float(nu - g.min()) / scale


def _is_feasible(S, *, costs=None, budget=None, conflict=None, strata=None,
                 min_per=None, max_per=None, required=(), forced=()):
    """Feasibility of one tuple, written out in plain Python.

    A second implementation of the constraint set the search applies as
    vectorised numpy masks, so a mask that drops the wrong rows shows up as a
    disagreement instead of being reproduced.
    """
    s = set(int(i) for i in S)
    if not set(int(f) for f in forced) <= s:
        return False
    if costs is not None and budget is not None and not np.isinf(budget):
        if float(np.asarray(costs)[list(S)].sum()) > budget:
            return False
    if conflict is not None:
        if any(conflict[i, j] for i, j in combinations(sorted(s), 2)):
            return False
    if strata is not None:
        cnt = Counter(int(strata[i]) for i in S if strata[i] >= 0)
        if max_per is not None and any(v > max_per for v in cnt.values()):
            return False
        if min_per is not None and any(cnt.get(int(k), 0) < min_per for k in required):
            return False
    return True


def _brute_force(G, cand, m, top_K, *, gamma=0.0, costs=None, budget=None,
                 conflict=None, strata=None, min_per=None, max_per=None,
                 forced=()):
    """The global top-K, by scoring every feasible tuple with :func:`_face_min`.

    Returns ``(ranked, n_feasible)`` where ``ranked`` is the ``top_K`` smallest
    ``(loss, tuple)`` pairs, ordered by loss and then lexicographically by the
    tuple so the order is total.
    """
    G = np.asarray(G, dtype=float)
    Gs = G + gamma * np.eye(G.shape[0]) if gamma else G
    cand = sorted(int(c) for c in cand)
    required = (sorted({int(c) for c in np.asarray(strata)[cand] if c >= 0})
                if strata is not None else ())
    rows = []
    for S in combinations(cand, m):
        if not _is_feasible(S, costs=costs, budget=budget, conflict=conflict,
                            strata=strata, min_per=min_per, max_per=max_per,
                            required=required, forced=forced):
            continue
        rows.append((_face_min(Gs[np.ix_(S, S)])[0], S))
    rows.sort(key=lambda row: (row[0], row[1]))
    return rows[:top_K], len(rows)


# =========================================================================
# Instances
# =========================================================================

def _gram(n=11, T=8, seed=0, *, duplicates=0, factor=0.0):
    """A Gram of f-centred predictors over ``n`` units.

    Centring across units puts the population target at the origin and so inside
    the convex hull of the candidates, which is the geometry the search is built
    for and the reason a relaxation bound cannot prune. ``duplicates`` copies
    donors onto each other (exact ties, a face-valued inner optimum);
    ``factor`` mixes in one dominant common component, the near-collinear geo
    panel that broke the previous fixed-budget Frank-Wolfe.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, n))
    if factor:
        X = factor * np.outer(rng.normal(size=T), rng.uniform(0.8, 1.2, n)) + 0.05 * X
    for d in range(duplicates):
        X[:, n - 1 - d] = X[:, d]
    X = X - X.mean(axis=1, keepdims=True)
    return X.T @ X


@pytest.fixture(scope="module")
def instance():
    """One panel with a full set of constraints hung off it."""
    n = 11
    rng = np.random.default_rng(0)
    conflict = np.zeros((n, n), dtype=bool)
    for i, j in [(0, 1), (2, 3), (4, 5), (1, 7)]:
        conflict[i, j] = conflict[j, i] = True
    return {
        "n": n,
        "G": _gram(n, 8, seed=7),
        "costs": rng.uniform(1.0, 5.0, size=n),
        "conflict": conflict,
        # three strata plus one unit with no stratum (code -1, exempt from quotas)
        "strata": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, -1]),
    }


# The constraint regimes the certificate has to survive. Each is
# ``(name, kwargs)`` where kwargs name the constraint arguments by the brute
# force's spelling; :func:`_run_and_compare` translates them for the search.
REGIMES = [
    ("unconstrained", {}),
    ("budget", {"budget": 9.0}),
    ("spillover", {"conflict": True}),
    ("stratum max quota", {"strata": True, "max_per": 1}),
    ("stratum min coverage", {"strata": True, "min_per": 1}),
    ("forced unit", {"forced": (2,)}),
    ("weak-targeting ridge", {"gamma": 0.05}),
    ("heavy ridge", {"gamma": 1.0}),
    ("budget and spillover", {"budget": 12.0, "conflict": True}),
    ("everything at once", {"budget": 13.0, "conflict": True, "strata": True,
                            "max_per": 2, "gamma": 0.01, "forced": (9,)}),
]


def _run_and_compare(inst, m=3, top_K=5, G=None, **regime):
    """Run the enumeration path and the brute force on the same instance.

    Returns ``(out, ranked, n_feasible)``.
    """
    G = inst["G"] if G is None else G
    n = G.shape[0]
    costs = inst["costs"][:n] if regime.get("budget") is not None else None
    conflict = inst["conflict"][:n, :n] if regime.get("conflict") else None
    strata = inst["strata"][:n] if regime.get("strata") else None
    shared = dict(
        gamma=regime.get("gamma", 0.0), costs=costs, budget=regime.get("budget"),
        conflict=conflict, strata=strata, min_per=regime.get("min_per"),
        max_per=regime.get("max_per"), forced=regime.get("forced", ()),
    )
    ranked, n_feasible = _brute_force(G, range(n), m, top_K, **shared)
    out = select_treated_designs(
        G, candidate_idx=list(range(n)), m=m, top_K=top_K, method="enumerate",
        unit_costs=costs, budget=regime.get("budget"),
        targeting_penalty=regime.get("gamma", 0.0),
        conflict=conflict, strata=strata,
        min_per_stratum=regime.get("min_per"), max_per_stratum=regime.get("max_per"),
        forced=list(regime.get("forced", ())),
    )
    return out, ranked, n_feasible


def _assert_is_global_top_k(out, ranked, n_feasible, G, gamma=0.0):
    """The returned pool is the global top-K of the feasible region.

    Identity of the tuples is asserted only where the losses separate. With
    duplicate donors many tuples are exactly tied and any ``K`` of the tied
    class is a correct answer, so there the assertion is on the loss vector and
    on each returned tuple attaining the loss of its rank.
    """
    designs = out["top_designs"]
    losses = np.array([d.loss for d in designs])
    ref_losses = np.array([r[0] for r in ranked])
    ref_tuples = [r[1] for r in ranked]

    expected_k = min(out["stats"]["problem"]["top_K"], n_feasible)
    assert len(designs) == len(ranked) == expected_k, (
        f"pool size {len(designs)} != {expected_k} designs available")
    assert np.allclose(losses, ref_losses, rtol=RTOL, atol=ATOL), (
        f"top-{len(designs)} losses differ from brute force:\n"
        f"  search {losses}\n  brute  {ref_losses}")
    assert np.all(np.diff(losses) >= -ATOL), "pool is not sorted by loss"

    # Each returned tuple is distinct, and its reported loss is the loss it
    # actually has -- re-solved by the reference solver on its own Gram.
    Gs = G + gamma * np.eye(G.shape[0]) if gamma else G
    seen = set()
    for d, ref_loss in zip(designs, ref_losses):
        key = tuple(d.indices)
        assert key not in seen, f"duplicate tuple {key} in the pool"
        seen.add(key)
        recomputed, _ = _face_min(Gs[np.ix_(d.indices, d.indices)])
        assert recomputed == pytest.approx(d.loss, rel=RTOL, abs=ATOL), (
            f"design {key} reports loss {d.loss} but attains {recomputed}")
        assert recomputed == pytest.approx(ref_loss, rel=RTOL, abs=ATOL), (
            f"design {key} attains {recomputed}, not the rank's {ref_loss}")

    separated = (len(ref_losses) < n_feasible
                 and np.all(np.diff(ref_losses) > TIE))
    if separated:
        assert [tuple(d.indices) for d in designs] == ref_tuples, (
            "losses separate, so the tuples must match exactly")


# =========================================================================
# 0. The reference solver itself
# =========================================================================

class TestReferenceSolver:
    """The brute force is only a reference if it is right on known answers."""

    def test_identity_gram_is_uniform(self):
        """``min w'w`` on the simplex is ``1/m`` at the uniform weights."""
        for m in (1, 2, 3, 5):
            val, w = _face_min(np.eye(m))
            assert val == pytest.approx(1.0 / m)
            assert w == pytest.approx(np.full(m, 1.0 / m))

    def test_rank_one_ones_gram_is_flat(self):
        """``Q = 11'`` gives ``w'Qw = (1'w)^2 = 1`` everywhere on the simplex."""
        val, w = _face_min(np.ones((4, 4)))
        assert val == pytest.approx(1.0)
        assert w.min() >= -1e-12 and w.sum() == pytest.approx(1.0)

    def test_diagonal_gram_is_the_harmonic_mean(self):
        """``min sum a_j w_j^2`` on the simplex is ``1 / sum(1/a_j)``."""
        a = np.array([1.0, 4.0, 9.0, 0.5])
        val, w = _face_min(np.diag(a))
        assert val == pytest.approx(1.0 / np.sum(1.0 / a))
        assert w == pytest.approx((1.0 / a) / np.sum(1.0 / a))

    def test_antipodal_donors_reach_the_origin(self):
        """Two donors at -1 and +1 in one dimension: the hull contains 0."""
        R = np.array([[-1.0, 1.0]])
        val, w = _face_min(R.T @ R)
        assert val == pytest.approx(0.0, abs=1e-14)
        assert w == pytest.approx([0.5, 0.5])

    def test_vertex_optimum_is_found(self):
        """An interior stationary point outside the simplex leaves a vertex."""
        Q = np.array([[1.0, 1.5], [1.5, 4.0]])   # 2t^2 - 5t + 4, minimised at t=1
        val, w = _face_min(Q)
        assert val == pytest.approx(1.0)
        assert w == pytest.approx([1.0, 0.0])

    def test_agrees_with_wolfe_on_random_grams(self):
        """Two exact solvers of the same program, one active set and one not."""
        worst = 0.0
        for seed in range(40):
            G = _gram(9, 7, seed=seed)
            rng = np.random.default_rng(seed)
            S = sorted(rng.choice(9, 4, replace=False).tolist())
            Q = G[np.ix_(S, S)]
            wolfe, _, _ = _afw_single(Q)
            face, _ = _face_min(Q)
            worst = max(worst, abs(wolfe - face) / max(1.0, abs(face)))
        assert worst < 1e-10, f"solvers disagree by {worst:.2e}"


# =========================================================================
# 1. The inner QP
# =========================================================================

class TestInnerWeightCertificate:
    """Every returned design's weights solve its own simplex QP, certifiably."""

    @pytest.mark.parametrize("seed", range(6))
    def test_returned_weights_satisfy_kkt(self, seed):
        G = _gram(11, 8, seed=300 + seed)
        out = select_treated_designs(G, candidate_idx=list(range(11)), m=4,
                                     top_K=6, method="enumerate")
        assert out["top_designs"]
        for d in out["top_designs"]:
            Q = G[np.ix_(d.indices, d.indices)]
            assert d.weights.min() >= -1e-12
            assert d.weights.sum() == pytest.approx(1.0, abs=1e-9)
            assert _kkt_violation(Q, d.weights) < 1e-9

    def test_kkt_holds_on_the_penalised_program_under_a_ridge(self):
        """With ``gamma > 0`` the weights solve ``min w'(G + gamma I)w``."""
        G = _gram(10, 8, seed=42)
        gamma = 0.2
        out = select_treated_designs(G, candidate_idx=list(range(10)), m=3,
                                     top_K=4, method="enumerate",
                                     targeting_penalty=gamma)
        for d in out["top_designs"]:
            Q = G[np.ix_(d.indices, d.indices)] + gamma * np.eye(3)
            assert _kkt_violation(Q, d.weights) < 1e-9
            # the reported imbalance stays the untargeted distance
            raw = G[np.ix_(d.indices, d.indices)]
            assert d.imbalance == pytest.approx(
                np.sqrt(max(d.weights @ raw @ d.weights, 0.0)), rel=1e-9)

    def test_reported_imbalance_is_the_root_of_the_loss(self):
        G = _gram(9, 7, seed=5)
        out = select_treated_designs(G, candidate_idx=list(range(9)), m=3,
                                     top_K=5, method="enumerate")
        for d in out["top_designs"]:
            assert d.imbalance == pytest.approx(np.sqrt(d.loss), rel=1e-9)

    def test_a_suboptimal_weight_vector_fails_the_kkt_check(self):
        """The certificate discriminates: uniform weights are not the optimum."""
        G = _gram(9, 7, seed=5)
        S = [0, 3, 6]
        Q = G[np.ix_(S, S)]
        _, w_star = _face_min(Q)
        uniform = np.full(3, 1.0 / 3)
        assert _kkt_violation(Q, w_star) < 1e-9
        assert _kkt_violation(Q, uniform) > 1e-6


# =========================================================================
# 2. The feasible region
# =========================================================================

class TestFeasibleRegionIsCovered:
    """Enumeration scores every feasible tuple and nothing infeasible."""

    @pytest.mark.parametrize("name,regime", REGIMES, ids=[r[0] for r in REGIMES])
    def test_subsets_evaluated_equals_the_feasible_count(self, instance, name, regime):
        out, _, n_feasible = _run_and_compare(instance, **regime)
        assert n_feasible > 0, "regime is vacuous -- nothing is feasible"
        assert out["stats"]["search"]["subsets_evaluated"] == n_feasible
        assert out["stats"]["search"]["method"] == "enumeration"

    @pytest.mark.parametrize("name,regime", REGIMES, ids=[r[0] for r in REGIMES])
    def test_every_returned_design_is_feasible(self, instance, name, regime):
        out, _, _ = _run_and_compare(instance, **regime)
        n = instance["G"].shape[0]
        costs = instance["costs"][:n] if regime.get("budget") is not None else None
        strata = instance["strata"][:n] if regime.get("strata") else None
        required = (sorted({int(c) for c in strata if c >= 0})
                    if strata is not None else ())
        for d in out["top_designs"]:
            assert _is_feasible(
                d.indices, costs=costs, budget=regime.get("budget"),
                conflict=instance["conflict"][:n, :n] if regime.get("conflict") else None,
                strata=strata, min_per=regime.get("min_per"),
                max_per=regime.get("max_per"), required=required,
                forced=regime.get("forced", ()))

    def test_budget_presolve_drops_only_impossible_units(self, instance):
        """The presolve is sound: no dropped unit sits in a feasible tuple."""
        n = instance["G"].shape[0]
        costs, budget, m = instance["costs"][:n], 9.0, 3
        kept = set(ls._budget_feasible_candidates(np.arange(n), m, costs, budget)
                   .tolist())
        reachable = {i for S in combinations(range(n), m)
                     if costs[list(S)].sum() <= budget for i in S}
        assert reachable <= kept


# =========================================================================
# 3. The global top-K
# =========================================================================

class TestGlobalTopK:
    """The pool is the brute-force optimum, constraint regime by regime."""

    @pytest.mark.parametrize("name,regime", REGIMES, ids=[r[0] for r in REGIMES])
    def test_matches_brute_force(self, instance, name, regime):
        out, ranked, n_feasible = _run_and_compare(instance, **regime)
        _assert_is_global_top_k(out, ranked, n_feasible, instance["G"],
                                gamma=regime.get("gamma", 0.0))
        assert out["stats"]["termination"]["status"] == "OPTIMAL"

    @pytest.mark.parametrize("seed", range(8))
    def test_matches_brute_force_across_panels(self, instance, seed):
        G = _gram(10, 7, seed=1000 + seed)
        out, ranked, n_feasible = _run_and_compare(instance, G=G)
        _assert_is_global_top_k(out, ranked, n_feasible, G)

    @pytest.mark.parametrize("m", [1, 2, 3, 4])
    def test_matches_brute_force_across_m(self, instance, m):
        out, ranked, n_feasible = _run_and_compare(instance, m=m, top_K=5)
        _assert_is_global_top_k(out, ranked, n_feasible, instance["G"])

    def test_single_feasible_tuple(self, instance):
        """Forcing ``m`` units leaves one tuple, which is trivially the optimum."""
        out, ranked, n_feasible = _run_and_compare(
            instance, m=3, top_K=5, forced=(1, 4, 8))
        assert n_feasible == 1
        _assert_is_global_top_k(out, ranked, n_feasible, instance["G"])

    def test_top_k_larger_than_the_feasible_region(self, instance):
        """Asking for more designs than exist returns all of them, still ranked."""
        G = instance["G"][:6, :6]
        out, ranked, n_feasible = _run_and_compare(instance, m=4, top_K=999, G=G)
        assert n_feasible == 15
        assert len(out["top_designs"]) == 15
        _assert_is_global_top_k(out, ranked, n_feasible, G)


# =========================================================================
# 4. Degenerate Grams
# =========================================================================

class TestDegenerateGrams:
    """Ties and rank deficiency: where an exact claim is easiest to overstate."""

    def test_duplicate_donors_return_a_tied_global_optimum(self):
        """With donors duplicated the optimum is a tie class, and any K of it do.

        Three donors copied onto three others makes eight tuples attain the same
        minimum. The search and the brute force break that tie differently --
        both are correct, and the test asserts the losses, not the labels.
        """
        G = _gram(10, 8, seed=3, duplicates=3)
        ranked, n_feasible = _brute_force(G, range(10), 3, 5)
        got = select_treated_designs(G, candidate_idx=list(range(10)), m=3,
                                     top_K=5, method="enumerate")
        _assert_is_global_top_k(got, ranked, n_feasible, G)
        # the tie is real, not an artefact of the tolerance
        all_losses = sorted(_face_min(G[np.ix_(S, S)])[0]
                            for S in combinations(range(10), 3))
        n_tied = sum(abs(v - all_losses[0]) < TIE for v in all_losses)
        assert n_tied >= 5, f"expected a tie class, found {n_tied} at the optimum"

    def test_rank_deficient_gram(self):
        """Fewer estimation periods than units: the Gram is singular."""
        G = _gram(10, 4, seed=5)
        assert np.linalg.matrix_rank(G) < 10
        ranked, n_feasible = _brute_force(G, range(10), 3, 5)
        got = select_treated_designs(G, candidate_idx=list(range(10)), m=3,
                                     top_K=5, method="enumerate")
        _assert_is_global_top_k(got, ranked, n_feasible, G)

    def test_one_dominant_factor(self):
        """The near-collinear geo panel: 97% of variance in one component."""
        G = _gram(11, 8, seed=9, factor=1.0)
        ranked, n_feasible = _brute_force(G, range(11), 3, 5)
        got = select_treated_designs(G, candidate_idx=list(range(11)), m=3,
                                     top_K=5, method="enumerate")
        _assert_is_global_top_k(got, ranked, n_feasible, G)

    def test_exactly_attainable_zero_imbalance(self):
        """When a tuple's hull contains the origin the optimum is zero."""
        n, T = 9, 6
        rng = np.random.default_rng(17)
        X = rng.normal(size=(T, n))
        X[:, 1] = -X[:, 0]                      # units 0 and 1 straddle the origin
        G = X.T @ X
        ranked, n_feasible = _brute_force(G, range(n), 3, 4)
        got = select_treated_designs(G, candidate_idx=list(range(n)), m=3,
                                     top_K=4, method="enumerate")
        assert got["top_designs"][0].loss == pytest.approx(0.0, abs=1e-18)
        _assert_is_global_top_k(got, ranked, n_feasible, G)


# =========================================================================
# 5. The ranking step between the two solvers
# =========================================================================

class TestBatchedRankingIsExact:
    """The pool is seated on batched losses and reported on re-solved ones.

    Enumeration ranks every tuple with :func:`_losses_for` (the batched Wolfe
    active set) and keeps the ``top_K`` by ``argsort``; only those ``K`` are then
    re-solved by :func:`_afw_single`. A batched loss that is close but wrong
    seats the wrong tuples, and the re-solve makes their reported losses
    correct, so the pool would look self-consistent and be wrong. The gap
    between the two solvers is therefore the quantity the certificate needs at
    zero, and it is measured on the geometry that once broke it: a fixed
    80-iteration Frank-Wolfe left a 9% median error on a one-factor panel.
    """

    @pytest.mark.parametrize("factor", [0.0, 1.0])
    def test_batched_equals_single_solve(self, factor):
        worst = 0.0
        for seed in range(6):
            G = _gram(11, 8, seed=200 + seed, factor=factor)
            subsets = np.array([list(S) for S in combinations(range(11), 4)])
            batched = _losses_for(G, subsets)
            single = np.array([_afw_single(G[np.ix_(list(S), list(S))])[0]
                               for S in subsets])
            scale = max(1e-16, float(single.max()))
            worst = max(worst, float(np.abs(batched - single).max()) / scale)
        assert worst < 1e-10, f"batched ranking is off by {worst:.2e} relative"

    def test_the_ranking_solver_is_the_one_that_seats_the_pool(self, instance):
        """The ``K`` returned are the ``K`` smallest batched losses."""
        G = instance["G"]
        n, m, K = G.shape[0], 3, 5
        subsets = np.array([list(S) for S in combinations(range(n), m)])
        batched = _losses_for(G, subsets)
        expected = {tuple(subsets[i]) for i in np.argsort(batched)[:K]}
        out = select_treated_designs(G, candidate_idx=list(range(n)), m=m,
                                     top_K=K, method="enumerate")
        assert {tuple(d.indices) for d in out["top_designs"]} == expected


# =========================================================================
# 6. Status discipline
# =========================================================================

class TestStatusDiscipline:
    """``OPTIMAL`` is claimed when, and only when, the region was enumerated."""

    def test_enumeration_claims_optimal(self, instance):
        out = select_treated_designs(instance["G"], candidate_idx=list(range(11)),
                                     m=3, top_K=5, method="enumerate")
        assert out["stats"]["termination"]["status"] == "OPTIMAL"

    def test_heuristic_never_claims_optimal(self, instance):
        out = select_treated_designs(instance["G"], candidate_idx=list(range(11)),
                                     m=3, top_K=5, method="heuristic",
                                     n_starts=4, random_state=0)
        assert out["stats"]["termination"]["status"] == "FEASIBLE"
        assert out["stats"]["search"]["method"] == "multistart_local_search"
        assert out["stats"]["search"]["consensus"] is not None

    def test_heuristic_reports_feasible_even_when_it_finds_the_optimum(self, instance):
        """Landing on the optimum is not the same as certifying it.

        The local search often does find the global minimiser on a pool this
        small. The status stays ``FEASIBLE`` because the search has no way to
        know that it did.
        """
        G = instance["G"]
        ranked, _ = _brute_force(G, range(11), 3, 1)
        out = select_treated_designs(G, candidate_idx=list(range(11)), m=3,
                                     top_K=1, method="heuristic", n_starts=16,
                                     random_state=0)
        assert out["top_designs"][0].loss == pytest.approx(ranked[0][0], rel=1e-8)
        assert out["stats"]["termination"]["status"] == "FEASIBLE"

    def test_auto_switches_on_enumerate_max(self, instance):
        """``auto`` claims the certificate exactly when it pays for it."""
        G = instance["G"]                                   # C(11, 3) = 165
        assert select_treated_designs(
            G, candidate_idx=list(range(11)), m=3, top_K=3, method="auto",
            enumerate_max=165)["stats"]["termination"]["status"] == "OPTIMAL"
        assert select_treated_designs(
            G, candidate_idx=list(range(11)), m=3, top_K=3, method="auto",
            enumerate_max=164, n_starts=4)["stats"]["termination"]["status"] == "FEASIBLE"

    def test_an_empty_region_raises_before_a_status_is_reached(self, instance):
        """A budget no tuple can meet is an itemised error, not a status.

        The presolve audit runs before the search, so the usual way of emptying
        the feasible region never produces a result object at all. There is no
        optimum, so there is nothing to certify and nothing to return.
        """
        from mlsynth.exceptions import MlsynthConfigError
        n = 11
        with pytest.raises(MlsynthConfigError, match="budget"):
            select_treated_designs(instance["G"], candidate_idx=list(range(n)),
                                   m=3, top_K=5, method="enumerate",
                                   unit_costs=np.full(n, 10.0), budget=29.0)

    def test_a_region_emptied_after_presolve_is_infeasible_not_optimal(self):
        """Forcing in an unaffordable unit empties the region past the audit.

        The audit prices the ``m`` cheapest eligible markets, which are
        affordable here, so it passes; the forced unit then puts every tuple
        over the budget. Enumeration completes with nothing feasible, and the
        status is ``INFEASIBLE``. Enumerating an empty region is not a
        certificate of optimality over it.
        """
        n = 11
        G = _gram(n, 8, seed=7)
        costs = np.array([1.0] * (n - 1) + [100.0])
        out = select_treated_designs(G, candidate_idx=list(range(n)), m=3,
                                     top_K=5, method="enumerate",
                                     unit_costs=costs, budget=10.0, forced=[n - 1])
        assert out["top_designs"] == []
        assert out["stats"]["termination"]["status"] == "INFEASIBLE"
        assert out["stats"]["search"]["subsets_evaluated"] == 0
        assert out["stats"]["incumbent"]["objective"] is None

    def test_relaxation_bound_is_advisory_and_not_a_gap(self, instance):
        """The hull bound collapses to ~0 and is reported as informational.

        The population target is the f-weighted centroid of the candidates, so it
        lies inside their hull and the cardinality-free relaxation has value
        zero. The stats carry the bound and no gap field, which is the honest
        reading: it cannot prune, so it cannot certify.
        """
        out = select_treated_designs(instance["G"], candidate_idx=list(range(11)),
                                     m=3, top_K=5, method="enumerate")
        relaxation = out["stats"]["relaxation"]
        assert relaxation["lower_bound_imbalance"] < 1e-6
        assert "not an optimality gap" in relaxation["note"]
        assert "gap" not in out["stats"]["termination"]
        assert relaxation["lower_bound_imbalance"] <= out["stats"]["incumbent"]["imbalance"] + 1e-12


# =========================================================================
# 7. The checks have power
# =========================================================================

class TestCertificateCheckHasPower:
    """Break the search three ways; the comparisons above have to notice."""

    def _compare(self, instance, **regime):
        out, ranked, n_feasible = _run_and_compare(instance, **regime)
        _assert_is_global_top_k(out, ranked, n_feasible, instance["G"],
                                gamma=regime.get("gamma", 0.0))

    def test_dropping_the_optimum_is_caught(self, instance, monkeypatch):
        """An enumeration returning the 2nd..(K+1)th best fails the check.

        The pool is still ``K`` designs long and every reported loss is the loss
        that design really has, so only a comparison against the region's own
        optimum can see the miss.
        """
        real = ls._enumerate

        def missing_the_best(G, cand, m, top_K, *a, **kw):
            raw, n = real(G, cand, m, top_K + 1, *a, **kw)
            return raw[1:], n

        monkeypatch.setattr(ls, "_enumerate", missing_the_best)
        with pytest.raises(AssertionError, match="losses differ from brute force"):
            self._compare(instance)

    def test_a_perturbed_ranking_is_caught(self, instance, monkeypatch):
        """Seat the pool on noisy losses; the re-solve cannot hide it.

        This is the failure mode the batched solver would produce: the reported
        losses of the returned designs stay correct, because they are re-solved,
        but they are the losses of the wrong tuples.
        """
        real = ls._losses_for

        def noisy(G, subsets, iters=70):
            out = real(G, subsets, iters=iters)
            rng = np.random.default_rng(0)
            return out * (1.0 + 0.1 * rng.random(out.shape))

        monkeypatch.setattr(ls, "_losses_for", noisy)
        with pytest.raises(AssertionError, match="losses differ from brute force"):
            self._compare(instance)

    def test_a_skipped_constraint_is_caught(self, instance, monkeypatch):
        """An enumeration that ignores the budget filter fails the check."""
        real_enumerate = ls._enumerate

        def ignoring_budget(G, cand, m, top_K, unit_costs, budget, iters, **kw):
            return real_enumerate(G, cand, m, top_K, None, None, iters, **kw)

        monkeypatch.setattr(ls, "_enumerate", ignoring_budget)
        with pytest.raises(AssertionError,
                           match="losses differ from brute force|pool size"):
            self._compare(instance, budget=9.0)

    def test_the_unbroken_search_passes_the_same_comparison(self, instance):
        """The control arm: without a mutant the three comparisons pass."""
        self._compare(instance)
        self._compare(instance, budget=9.0)


# =========================================================================
# 8. End to end, through the estimator
# =========================================================================

class TestEstimatorCarriesTheCertificate:
    """The claim has to survive the trip from the search up to ``fit()``."""

    @staticmethod
    def _panel(n=9, T=24, seed=4):
        import pandas as pd
        rng = np.random.default_rng(seed)
        base = np.arange(T) * 0.3
        rows = []
        for i in range(n):
            s = base + rng.normal(scale=1.0, size=T) + i
            rows.extend({"unit": f"u{i}", "time": t, "Y": float(s[t]), "elig": True}
                        for t in range(T))
        return pd.DataFrame(rows)

    def test_fit_reports_a_certificate_that_holds_on_its_own_gram(self, monkeypatch):
        """Capture the Gram ``fit()`` built, and brute-force the same problem.

        The estimator builds ``G`` from f-centred standardised predictors over
        the estimation window, so the only honest way to check its Stage-1 answer
        is against the matrix it actually used. The search is wrapped to record
        that matrix on the way past.
        """
        from mlsynth.estimators import lexscm as lexscm_mod
        from mlsynth.estimators.lexscm import LEXSCM

        captured = {}
        real = lexscm_mod.select_treated_designs

        def recording(G, candidate_idx, m, top_K, **kw):
            captured.update(G=np.asarray(G, dtype=float),
                            cand=list(candidate_idx), m=m, top_K=top_K, kw=kw)
            return real(G, candidate_idx, m, top_K, **kw)

        monkeypatch.setattr(lexscm_mod, "select_treated_designs", recording)

        res = LEXSCM(dict(df=self._panel(), outcome="Y", unitid="unit", time="time",
                          candidate_col="elig", m=3, top_K=5, display_graph=False,
                          verbose=False)).fit()

        stats = res.search.selection["stats"]
        assert stats["termination"]["status"] == "OPTIMAL"

        G, m, top_K = captured["G"], captured["m"], captured["top_K"]
        ranked, n_feasible = _brute_force(G, captured["cand"], m, top_K)
        designs = res.search.selection["top_tuples"]
        assert [d.loss for d in designs] == pytest.approx(
            [r[0] for r in ranked], rel=1e-8, abs=1e-14)
        assert stats["search"]["subsets_evaluated"] == n_feasible
        for d in designs:
            assert _kkt_violation(G[np.ix_(d.indices, d.indices)], d.weights) < 1e-9
