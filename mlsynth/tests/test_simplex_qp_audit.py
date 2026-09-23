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

# Sites whose feasible set is exactly the probability simplex.
ELIGIBLE = {
    ("bilevel/penalized.py", "w"): 2,
    ("bilevel/ridge_augment.py", "w"): 1,
    ("clustersc_helpers/pcr/convex.py", "w"): 1,
    ("cscm_helpers/engine.py", "W"): 1,
    ("drosc_helpers/estimation.py", "w"): 1,
    ("dsc_helpers/weights.py", "w"): 2,
    ("dscar_helpers/weights.py", "w"): 1,
    ("dtwsc_helpers/pipeline.py", "w"): 1,
    ("fast_scm_helpers/fast_scm_bb_helpers.py", "w"): 1,
    ("fast_scm_helpers/fast_scm_control_helpers.py", "v_control"): 1,
    ("hsc_helpers/formulation.py", "omega"): 1,
    ("inferutils.py", "w"): 1,
    ("iscm_helpers/weights.py", "w"): 1,
    ("laxscm_helpers/crossval.py", "w"): 1,
    ("masc_helpers/estimation.py", "w"): 1,
    ("mlsc_helpers/crossval.py", "w"): 1,
    ("mlsc_helpers/crossval.py", "omega"): 1,
    ("mlsc_helpers/optimization.py", "w"): 1,
    ("mlsc_helpers/optimization.py", "omega"): 1,
    ("orthsc_helpers/gmm_sce/solver.py", "w"): 1,
    ("scmo_helpers/estimation.py", "lam"): 1,
    ("scmo_helpers/solvers.py", "w"): 1,
    ("spillsynth_helpers/cd/scm_core.py", "w"): 1,
    ("spotsynth_helpers/sc.py", "w"): 1,
    ("spsydid_helpers/weights.py", "omega"): 2,
    ("ssc_helpers/weights.py", "b"): 1,
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
        "eta, pinned at its last entry rather than summing to one",
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
        "eligible", "extra-constraints", "no-nonnegativity"}


def test_the_eligible_set_is_the_one_that_was_classified(sites):
    found = collections.Counter(
        _key(s) for s in sites if s.verdict == "eligible")
    assert dict(found) == ELIGIBLE, (
        "the set of cvxpy problems on the probability simplex has changed. "
        "A new one is not automatically eligible for solve_simplex_qp: check "
        "that its feasible set is exactly {w >= 0, sum(w) = 1} and add it "
        "here, or record in INELIGIBLE why it is not."
    )


def test_the_ineligible_sites_keep_their_reasons(sites):
    found = {_key(s) for s in sites if s.verdict != "eligible"}
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
    # iscm declares nonneg on the Variable and lists only the sum constraint.
    s = by[("iscm_helpers/weights.py", "w")]
    assert s.verdict == "eligible"
    assert ">= 0" not in s.constraints
