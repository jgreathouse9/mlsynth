"""The simplex-QP classification is an allowlist, not a count.

``tools/simplex_qp_audit.py`` says which cvxpy problems could move onto
``solve_simplex_qp``. A count alone rots: a new cvxpy simplex problem would
join the eligible set with nobody deciding that it should, and a site that
gains a constraint would leave it just as silently.

So the eligible set is pinned by name here. Adding a cvxpy problem with a
sum-to-one constraint fails this test, and the fix is to classify it -- either
add it to the allowlist, having checked that its feasible set really is
``{w >= 0, sum(w) = 1}``, or record why it is not eligible.

The keys are ``(file, variable)`` with a count, not line numbers, so ordinary
edits above a site do not break the pin.
"""
from __future__ import annotations

import collections

import pytest

from tools.simplex_qp_audit import audit

# Sites that are the probability simplex AND minimise ||A - Bw||^2, so
# solve_simplex_qp solves the same program they do.
ELIGIBLE = {
    ("bilevel/ridge_augment.py", "w"): 1,
    ("clustersc_helpers/spannability.py", "w"): 1,
    ("drosc_helpers/estimation.py", "w"): 1,
    ("dsc_helpers/weights.py", "w"): 2,
    ("dtwsc_helpers/pipeline.py", "w"): 1,
    ("inferutils.py", "w"): 1,
    ("masc_helpers/estimation.py", "w"): 1,
    ("mlsc_helpers/optimization.py", "w"): 1,
    ("orthsc_helpers/gmm_sce/solver.py", "w"): 1,
    ("scmo_helpers/estimation.py", "lam"): 1,
    ("scmo_helpers/solvers.py", "w"): 1,
    ("spotsynth_helpers/sc.py", "w"): 1,
    ("ssc_helpers/weights.py", "b"): 1,
}

# Sites this library has already moved onto ``solve_simplex_qp``. They are not
# in ELIGIBLE because they are no longer cvxpy problems at all, and the audit
# cannot see them. Pinned so a migration cannot be undone without a test
# saying so -- reintroducing cvxpy at one of these fails
# ``test_a_migrated_site_stays_migrated``.
MIGRATED = {
    ("iscm_helpers/weights.py", "w"),
    ("spillsynth_helpers/cd/scm_core.py", "w"),
}

# The probability simplex, but minimising something else. Swapping the solver
# here would drop the extra term, not speed it up. Three carry a Gram form
# (``quad_form``) which may be the same program after substituting
# ``Q = B'B``; that has to be checked per site, not assumed.
WRONG_OBJECTIVE = {
    ("bilevel/penalized.py", "w"): "a penalty term lam * (d2 @ w)",
    ("clustersc_helpers/pcr/convex.py", "w"): "cp.norm(..., 2), not its square",
    ("cscm_helpers/engine.py", "W"): "a V-weighted Gram form",
    ("dscar_helpers/weights.py", "w"): "a composite loss built elsewhere",
    ("fast_scm_helpers/fast_scm_bb_helpers.py", "w"): "quad_form(w, Q)",
    ("fast_scm_helpers/fast_scm_control_helpers.py", "v_control"): "built elsewhere",
    ("hsc_helpers/formulation.py", "omega"): "Gram plus a linear term",
    ("laxscm_helpers/crossval.py", "w"): "an infinity norm, which is the method",
    ("mlsc_helpers/crossval.py", "w"): "least squares plus a ridge floor",
    ("mlsc_helpers/crossval.py", "omega"): "least squares plus lambda",
    ("mlsc_helpers/optimization.py", "omega"): "a sum of objective terms",
    ("spsydid_helpers/weights.py", "omega"): "built elsewhere",
}

# Sites that carry a sum-to-one constraint and are NOT the probability simplex,
# with the reason. Swapping the solver at any of these changes the answer.
INELIGIBLE = {
    ("drosc_helpers/estimation.py", "b"):
        "two-sided moment bounds Sig @ b in [lo, hi]",
    ("drosc_helpers/inference.py", "self.b"):
        "the same moment bounds, on the inference path",
    ("laxscm_helpers/fast_solve.py", "w"):
        "an infinity-norm residual cap, which is the method",
    ("orthsc_helpers/regularized.py", "d2"):
        "balance constraints on M @ d2",
    ("spcd_helpers/weights_exact.py", "w"):
        "two sum-to-one constraints, on disjoint treated and control subsets",
    ("helperutils.py", "time_weights_variable"):
        "time weights, which are not constrained non-negative",
    ("musc_helpers/estimation.py", "W"):
        "a matrix with a unit diagonal and off-diagonals in [-1, 0]",
    ("orthsc_helpers/regularized.py", "?"):
        "eta, pinned at its last entry instead of summing to one",
    ("pangeo_helpers/mip.py", "?"):
        "an assignment constraint M @ x == 1, not a simplex",
    ("shc_helpers/kernels.py", "w"):
        "kernel weights, declared without non-negativity",
}


@pytest.fixture(scope="module")
def sites():
    return audit()


def _key(site):
    return (site.path.replace("mlsynth/utils/", ""), site.variable)


def test_every_site_gets_one_of_the_three_verdicts(sites):
    assert sites, "the audit found no cvxpy problems at all"
    assert {s.verdict for s in sites} <= {
        "eligible", "wrong-objective", "extra-constraints", "no-nonnegativity"}


def test_the_eligible_set_is_the_one_that_was_classified(sites):
    found = collections.Counter(
        _key(s) for s in sites if s.verdict == "eligible")
    assert dict(found) == ELIGIBLE, (
        "the set of cvxpy problems on the probability simplex has changed. "
        "A new one is not automatically eligible for solve_simplex_qp: check "
        "that its feasible set is exactly {w >= 0, sum(w) = 1} and add it "
        "here, or record in INELIGIBLE why it is not."
    )


def test_a_simplex_site_with_another_objective_is_not_eligible(sites):
    """Constraints are half of eligibility; the objective is the other half.

    ``solve_simplex_qp`` minimises ``||A - Bw||^2``. LAXSCM minimises an
    infinity norm, MLSC adds a ridge floor, BILEVEL's penalized path adds
    ``lam * (d2 @ w)``. Reading only the constraint set counted 29 of these
    as eligible; 14 of them are not.
    """
    found = {_key(s) for s in sites if s.verdict == "wrong-objective"}
    assert found == set(WRONG_OBJECTIVE), (
        "a simplex site's objective changed shape. solve_simplex_qp solves "
        "||A - Bw||^2 and nothing else."
    )


def test_the_ineligible_sites_keep_their_reasons(sites):
    found = {_key(s) for s in sites
             if s.verdict in ("extra-constraints", "no-nonnegativity")}
    assert found == set(INELIGIBLE), (
        "a site stopped being the probability simplex, or started being one. "
        "Either way the classification is a decision, not a count."
    )


def test_a_redundant_upper_bound_does_not_disqualify(sites):
    """``w <= 1`` follows from ``w >= 0`` and ``sum(w) == 1``.

    Both MASC and DSC write it explicitly. Reading it as an extra constraint
    would exclude two sites that are the probability simplex.
    """
    by = {_key(s): s for s in sites}
    for k in (("masc_helpers/estimation.py", "w"), ("dsc_helpers/weights.py", "w")):
        assert by[k].verdict == "eligible"


def test_non_negativity_is_read_from_the_variable_too(sites):
    """Most of this library writes ``cp.Variable(n, nonneg=True)``.

    An audit that reads only the constraint list calls those sites ineligible
    and halves the count, which is the error this test exists to prevent.
    """
    by = {_key(s): s for s in sites}
    # dtwsc declares nonneg on the Variable and lists only the sum constraint.
    s = by[("dtwsc_helpers/pipeline.py", "w")]
    assert s.verdict == "eligible"
    assert ">= 0" not in s.constraints


def test_a_migrated_site_stays_migrated(sites):
    """A site already on ``solve_simplex_qp`` must not reappear as a cvxpy problem.

    The audit reads cvxpy call sites, so a migrated module is invisible to it
    and its absence from ELIGIBLE says nothing on its own. This asserts the
    absence directly: if someone reintroduces a cvxpy simplex solve at one of
    these, it shows up here instead of restoring the fourth solver path
    this branch exists to remove.
    """
    present = {_key(s) for s in sites}
    assert MIGRATED.isdisjoint(present), (
        f"these were migrated onto solve_simplex_qp and are cvxpy again: "
        f"{sorted(MIGRATED & present)}"
    )
