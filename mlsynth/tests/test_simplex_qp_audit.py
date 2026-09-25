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

from tools.simplex_qp_audit import VERDICTS, _classify_objective, audit

# Sites that are the probability simplex AND minimise ||A - Bw||^2, so
# solve_simplex_qp solves the same program they do -- some of them only after
# the caller reshapes the data, which TRANSFORMS below records per site.
#
# The reshapings are algebra, not approximation. A non-squared norm has the
# same minimiser as its square. A metric-weighted residual is a row scaling. A
# Gram form is a factorisation. A ridge is extra design rows carrying no target
# (Zou and Hastie 2005, Lemma 1). An intercept is profiled out by centring. A
# penalty on a constraint the program already enforces is a constant.
#
# The solver also carries a term linear in the weights directly, as its
# `linear` argument, and that needs no reshaping at all: Abadie and L'Hour's
# penalised SCM is an L1 penalty, which is linear on the non-negative orthant,
# and folding it into the target is not available when the donor block is wide.
ELIGIBLE = {
    ("bilevel/penalized.py", "w"): 1,
    ("solvers/ridge_augment.py", "w"): 1,
    ("dscar_helpers/weights.py", "w"): 1,
    ("fast_scm_helpers/fast_scm_bb_helpers.py", "w"): 1,
    ("hsc_helpers/formulation.py", "omega"): 1,
    ("masc_helpers/estimation.py", "w"): 1,
    ("mlsc_helpers/crossval.py", "omega"): 1,
    ("mlsc_helpers/crossval.py", "w"): 1,
    ("mlsc_helpers/optimization.py", "omega"): 1,
    ("mlsc_helpers/optimization.py", "w"): 1,
    ("orthsc_helpers/gmm_sce/solver.py", "w"): 1,
    ("spsydid_helpers/weights.py", "lam"): 1,
    ("spsydid_helpers/weights.py", "omega"): 1,
    ("ssc_helpers/weights.py", "b"): 1,
}

# Eligible, on cvxpy, and still to swap: the sites the audit reports as work.
REMAINING = {
    ("fast_scm_helpers/fast_scm_bb_helpers.py", "w"),
    ("hsc_helpers/formulation.py", "omega"),
    ("spsydid_helpers/weights.py", "lam"),
    ("spsydid_helpers/weights.py", "omega"),
}

# Eligible and staying on cvxpy, with the reason. Eligibility says the active
# set solves the same program; it does not say the cvxpy call should go. One is
# the reference the native path is checked against, five are escape hatches a
# caller reaches by naming a solver, one is the warm start such a hatch seeds
# itself with, one has no caller, and two are not identified on the panel their
# replication uses. Without this split, "eligible" reads as a to-do list when
# most of it is settled, and the next reader re-derives which.
#
# The two non-identified sites, SSC and DSC, are the same finding twice: where
# the argmin is a face, which point comes back is a property of the solver's
# pivot order, and a replication that matches a published number is matching
# that choice. On DSC's Beijing panel (Zheng and Chen 2024, Section 5) the
# active set is certified optimal by simplex_point_is_optimal in 72 of 72
# per-period solves and strictly better on the objective in 34 with none
# worse, so it solves the program; simplex_optimum_is_unique still rejects
# uniqueness in 33 of the 72. rank[B; 1'] is 6 against 74 donors, which bounds
# the face at dimension 68, but the fit is sparse -- a median of 5 donors carry
# weight, and the face at the returned point has dimension 1 in the median and
# 13 at most. A one-dimensional ambiguity is enough: swapping the solver moves
# the orange-alert ATT from the paper's -33.8 to -35.46 micrograms per cubic
# metre. Migrating either site means deciding which point of the face the
# library should return, which is an econometrics question and not a solver
# one, so both wait for that decision.
KEPT_ON_CVXPY = {
    ("bilevel/penalized.py", "w"):
        "the Gram form, whose linear term carries the data fit; no caller",
    ("solvers/ridge_augment.py", "w"):
        "the cvxpy reference the active-set path is checked against",
    ("masc_helpers/estimation.py", "w"):
        "reached only by naming a non-Clarabel solver; the default is native",
    ("mlsc_helpers/crossval.py", "omega"):
        "the cvxpy escape hatch's Parameter grid sweep",
    ("mlsc_helpers/crossval.py", "w"):
        "the escape hatch's lambda = 0 branch",
    ("mlsc_helpers/optimization.py", "omega"):
        "the cvxpy escape hatch",
    ("mlsc_helpers/optimization.py", "w"):
        "the warm start the escape hatch seeds its solver with",
    ("orthsc_helpers/gmm_sce/solver.py", "w"):
        "reached only by naming a non-Clarabel solver; the default is native",
    ("dscar_helpers/weights.py", "w"):
        "not identified on the authors' panel: the argmin is a face in 33 of "
        "the 72 per-period solves, so the Path-A number selects a point of it. "
        "See the note below",
    ("ssc_helpers/weights.py", "b"):
        "not identified on the authors' panel; the Path-A replication matches "
        "the reference solver's choice among a continuum of exact fits",
}

# What a caller does to the data before the swap, for the sites that need
# something. Every one of these was checked against the cvxpy program it
# replaces on random panels; the agreement is in
# ``agents/agents_simplex_audit.md``. An empty transform means the call site
# swaps with no reshaping, and those keys are absent here.
TRANSFORMS = {
    ("dscar_helpers/weights.py", "w"):
        "scale the rows by the square root of the metric; "
        "the sum-to-one penalty is zero on the feasible set",
    ("fast_scm_helpers/fast_scm_bb_helpers.py", "w"):
        "factor the Gram as R'R and take B = R",
    ("hsc_helpers/formulation.py", "omega"):
        "factor the Gram as R'R and take B = R, "
        "recovering the target from the linear term",
    ("mlsc_helpers/crossval.py", "omega"):
        "augment the design with the penalty's square-root factor",
    ("mlsc_helpers/crossval.py", "w"):
        "augment the design with a multiple of the identity",
    ("mlsc_helpers/optimization.py", "omega"):
        "augment the design with the penalty's square-root factor",
    ("spsydid_helpers/weights.py", "lam"):
        "centre the design and the target to profile out the intercept",
    ("spsydid_helpers/weights.py", "omega"):
        "centre the design and the target to profile out the intercept; "
        "augment the design with a multiple of the identity",
}

MIGRATED = {
    ("cscm_helpers/engine.py", "W"),
    ("clustersc_helpers/pcr/convex.py", "w"),
    ("clustersc_helpers/spannability.py", "w"),
    ("drosc_helpers/estimation.py", "w"),
    ("dsc_helpers/weights.py", "w"),
    ("dtwsc_helpers/pipeline.py", "w"),
    ("inferutils.py", "w"),
    ("iscm_helpers/weights.py", "w"),
    ("scmo_helpers/estimation.py", "lam"),
    ("scmo_helpers/solvers.py", "w"),
    ("spillsynth_helpers/cd/scm_core.py", "w"),
    ("spotsynth_helpers/sc.py", "w"),
    ("tssc_helpers/estimation.py", "w"),
}

# ``dsc_helpers/weights.py`` keys appear in MIGRATED and in INELIGIBLE, for
# the same reason they used to appear in ELIGIBLE and INELIGIBLE: the module
# has two weight options and only ``_refine_exact`` takes the simplex.
#
# ``bilevel/penalized.py`` is in both ELIGIBLE and MIGRATED and that is the point. Its two
# programs are the same shape and only one of them was migrated:
# ``penalized_weights`` takes the residual form and is on the active set, while
# ``_simplex_qp`` takes the Gram form, whose linear term carries the data fit
# and not only the penalty, and has no caller in the library. The count above is
# what remains visible to the audit.

# The probability simplex, but minimising something else. The objective is
# read as a sum of terms and each term is matched against what the solver can
# carry: a squared residual, a term linear in the weights, a ridge the design
# absorbs as extra rows, a penalty that is zero on the feasible set. A site
# lands here when a term is none of those.
WRONG_OBJECTIVE = {
    ("laxscm_helpers/crossval.py", "w"): "an infinity norm, which is the method",
}

# The objective is not an expression at the call site, so the audit cannot say
# what is minimised. This is a third answer, not a variant of the second: the
# sites above were checked and are a different program, while these were not
# checked at all. Filing them as wrong-objective asserted something nobody
# measured, and three of the four sites that carried that label turned out to
# be the simplex least-squares program.
UNREADABLE = {
    ("fast_scm_helpers/fast_scm_control_helpers.py", "?"):
        "a generic solve helper; both arguments arrive from the caller",
}

# Sites that carry a sum-to-one constraint and are NOT the probability simplex,
# with the reason. Swapping the solver at any of these changes the answer.
INELIGIBLE = {
    ("drosc_helpers/estimation.py", "b"):
        "two-sided moment bounds Sig @ b in [lo, hi]",
    ("drosc_helpers/inference.py", "self.b"):
        "the same moment bounds, on the inference path",
    ("dsc_helpers/weights.py", "w"):
        "sum-to-one without non-negativity, which is the other weight option",
    ("helperutils.py", "donor_weights_variable"):
        "donor weights, which are not constrained non-negative",
    ("helperutils.py", "time_weights_variable"):
        "time weights, which are not constrained non-negative",
    ("laxscm_helpers/fast_solve.py", "w"):
        "an infinity-norm residual cap, which is the method",
    ("musc_helpers/estimation.py", "W"):
        "a matrix with a unit diagonal and off-diagonals in [-1, 0]",
    ("orthsc_helpers/regularized.py", "d"):
        "balance constraints on Zs @ YJs.T @ d, with lam minimised",
    ("orthsc_helpers/regularized.py", "d2"):
        "balance constraints on M @ d2",
    ("orthsc_helpers/regularized.py", "?"):
        "eta, pinned at its last entry instead of summing to one",
    ("pangeo_helpers/mip.py", "x"):
        "an assignment constraint M @ x == 1, not a simplex",
    ("shc_helpers/kernels.py", "w"):
        "non-negativity appended under an if, so it holds on one branch only",
    ("spcd_helpers/weights_exact.py", "w"):
        "two sum-to-one constraints, on disjoint treated and control subsets",
}

# ``dsc_helpers/weights.py`` is in both ELIGIBLE and INELIGIBLE, as
# ``bilevel/penalized.py`` is in ELIGIBLE and MIGRATED. Its two solves are the
# module's two weight options: ``_refine_exact`` takes the simplex, and
# ``solve_sum_to_one_weights`` takes ``sum(w) == 1`` alone, which is a
# different feasible set. Reading the whole module for the ``nonneg=True`` flag
# found the first function's declaration and reported the second as eligible --
# the audit saying "swap the solver here" about weights that may go negative.

@pytest.fixture(scope="module")
def sites():
    return audit()


def _key(site):
    return (site.path.replace("mlsynth/utils/", ""), site.variable)


def test_every_site_gets_one_of_the_verdicts(sites):
    assert sites, "the audit found no cvxpy problems at all"
    assert {s.verdict for s in sites} <= set(VERDICTS)
    assert set(VERDICTS) == {"eligible", "wrong-objective", "unreadable",
                             "extra-constraints", "no-nonnegativity"}


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
    infinity norm of a residual, which is the method and not a reshaping of
    least squares. Reading only the constraint set counted 29 sites as
    eligible; this one is not.
    """
    found = {_key(s) for s in sites if s.verdict == "wrong-objective"}
    assert found == set(WRONG_OBJECTIVE), (
        "a simplex site's objective changed shape. solve_simplex_qp solves "
        "||A - Bw||^2 and nothing else."
    )


def test_an_objective_the_audit_cannot_read_says_so(sites):
    """An unread objective is not a different objective.

    ``_solve_qp`` in fast_scm takes ``objective`` as an argument, so the call
    site minimises a name with no binding to follow. Calling that
    wrong-objective asserts the program is something else, which nobody
    measured -- and when the four sites carrying that label were measured,
    three of them were the simplex least-squares program.
    """
    found = {_key(s) for s in sites if s.verdict == "unreadable"}
    assert found == set(UNREADABLE)


def test_each_eligible_site_names_what_the_caller_must_do(sites):
    """The transform is the finding, and a count does not carry it.

    Ten of these sites are the simplex least-squares program written in
    another algebra -- a Gram form, a ridge, an intercept, a metric. Saying
    "eligible" without saying "factor the Gram" leaves the next reader to
    re-derive it, which is how six of them came to be filed under
    wrong-objective in the first place.
    """
    found = {_key(s): s.transform for s in sites if s.verdict == "eligible"}
    assert {k: v for k, v in found.items() if v} == TRANSFORMS


def test_the_ineligible_sites_keep_their_reasons(sites):
    found = {_key(s) for s in sites
             if s.verdict in ("extra-constraints", "no-nonnegativity")}
    assert found == set(INELIGIBLE), (
        "a site stopped being the probability simplex, or started being one. "
        "Either way the classification is a decision, not a count."
    )


def test_a_redundant_upper_bound_does_not_disqualify(sites):
    """``w <= 1`` follows from ``w >= 0`` and ``sum(w) == 1``.

    MASC writes it explicitly. Reading it as an extra constraint would
    exclude a site that is the probability simplex. DSC used to be the second
    example here and is now on the active set.
    """
    by = {_key(s): s for s in sites}
    s = by[("masc_helpers/estimation.py", "w")]
    assert s.verdict == "eligible"
    assert "<= 1" in s.constraints


def test_non_negativity_is_read_from_the_variable_too(sites):
    """Most of this library writes ``cp.Variable(n, nonneg=True)``.

    An audit that reads only the constraint list calls those sites ineligible
    and halves the count, which is the error this test exists to prevent.
    """
    by = {_key(s): s for s in sites}
    # SpSyDiD declares nonneg on the Variable and lists only the sum
    # constraint. DTWSC used to be this example and is now on the active set.
    s = by[("spsydid_helpers/weights.py", "omega")]
    assert s.verdict == "eligible"
    assert ">= 0" not in s.constraints


def test_every_eligible_site_is_either_work_or_a_decision(sites):
    """"Eligible" is not a to-do list until the finished ones are named.

    A site can solve the same program the active set solves and still keep its
    cvxpy call: two of these are the reference that path is checked against,
    four are escape hatches a caller reaches by naming a solver, and one has
    no caller. Asserting the split here means a new eligible site is work
    until someone writes down why it is not.
    """
    eligible = {_key(s) for s in sites if s.verdict == "eligible"}
    assert set(KEPT_ON_CVXPY) <= eligible, (
        "a site kept on cvxpy on purpose is no longer eligible: it changed "
        "shape, or it was migrated and the reason should go with it."
    )
    assert eligible - set(KEPT_ON_CVXPY) == REMAINING, (
        "an eligible site is neither listed as remaining work nor given a "
        "reason for staying on cvxpy. Both are decisions; a count is not."
    )


def test_a_migrated_site_stays_migrated(sites):
    """A site already on ``solve_simplex_qp`` must not reappear as a cvxpy problem.

    The audit reads cvxpy call sites, so a migrated module is invisible to it
    and its absence from ELIGIBLE says nothing on its own. This asserts the
    absence directly: if someone reintroduces a cvxpy simplex solve at one of
    these, it shows up here instead of restoring the fourth solver path
    this branch exists to remove.

    Only the sites the audit reads as the probability simplex count. A key is
    ``(file, variable)`` and a file can hold more than one solve under the
    same variable name: DSC keeps ``solve_sum_to_one_weights`` on cvxpy, which
    is a different feasible set and is pinned as such in INELIGIBLE.
    """
    simplex = ("eligible", "wrong-objective", "unreadable")
    present = {_key(s) for s in sites if s.verdict in simplex}
    assert MIGRATED.isdisjoint(present), (
        f"these were migrated onto solve_simplex_qp and are cvxpy again: "
        f"{sorted(MIGRATED & present)}"
    )


# --------------------------------------------------------------------------- #
# The classifier itself. The pins above say which sites are eligible; these say
# why, on objectives written out here instead of read off the library. The
# distinction matters because the classifier's failure mode is reading syntax
# and calling it algebra: ``quad_form``, a non-squared norm and a ridge term
# all look unlike ``sum_squares`` and are the same program.
# --------------------------------------------------------------------------- #

def _kind(text: str, var: str = "w"):
    return _classify_objective(text, var)


def test_a_squared_residual_needs_no_transform():
    assert _kind("cp.Minimize(cp.sum_squares(A - B @ w))") == ("least-squares", "")


def test_a_non_squared_norm_is_the_same_program():
    """``||r||`` and ``||r||^2`` have the same minimiser; the square is monotone.

    TSSC's four variants were migrated on exactly this reading, so a
    classifier that rejects it contradicts a migration already in the tree.
    """
    kind, transform = _kind("cp.Minimize(cp.norm(A - B @ w, 2))")
    assert kind == "least-squares"
    assert "same minimiser" in transform


def test_an_unsquared_norm_of_another_order_is_not_least_squares():
    """Only the Euclidean norm shares a minimiser with the squared residual."""
    assert _kind("cp.Minimize(cp.norm(A - B @ w, 1))")[0] == "other"
    assert _kind("cp.Minimize(cp.norm_inf(A - B @ w))")[0] == "other"


def test_a_metric_weighted_residual_is_a_row_scaling():
    """``r'Vr`` is ``||V^{1/2} r||^2``, so the metric moves onto the rows."""
    kind, transform = _kind(
        "cp.Minimize(cp.quad_form(X1 - X0 @ w, cp.psd_wrap(np.diag(V))))")
    assert kind == "least-squares"
    assert "square root of the metric" in transform


def test_a_gram_form_on_the_variable_is_a_factorisation():
    kind, transform = _kind("cp.Minimize(cp.quad_form(w, Q))")
    assert kind == "least-squares"
    assert "factor the Gram" in transform


def test_a_gram_form_with_a_linear_term_recovers_the_target():
    """``w'Hw - 2f'w`` is ``||Rw - R^{-T}f||^2`` up to a constant, with R'R = H."""
    kind, transform = _kind(
        "cp.Minimize(cp.quad_form(w, cp.psd_wrap(H)) - 2.0 * f @ w)")
    assert kind == "least-squares"
    assert "factor the Gram" in transform and "linear term" in transform


def test_a_ridge_on_the_weights_is_a_design_augmentation():
    """Zou and Hastie (2005), Lemma 1: the L2 term becomes rows carrying no target."""
    kind, transform = _kind(
        "cp.Minimize(cp.sum_squares(A - B @ w) + 1e-8 * cp.sum_squares(w))")
    assert kind == "least-squares"
    assert "multiple of the identity" in transform


def test_a_generalised_ridge_augments_with_its_square_root_factor():
    kind, transform = _kind(
        "cp.Minimize(cp.sum_squares(A - B @ w)"
        " + lambd * sigma_y2 * cp.quad_form(w, cp.psd_wrap(Q)))")
    assert kind == "least-squares"
    assert "square-root factor" in transform


def test_a_penalty_on_a_constraint_the_program_enforces_is_a_constant():
    """``(1'w - 1)^2`` is zero at every feasible point, so it moves nothing."""
    kind, transform = _kind(
        "cp.Minimize(cp.sum_squares(cp.multiply(v, Z1 - Z0 @ w))"
        " + cp.square(cp.sum(w) - 1.0))")
    assert kind == "least-squares"
    assert "zero on the feasible set" in transform


def test_a_term_linear_in_the_weights_needs_no_transform():
    """The solver carries it as its ``linear`` argument."""
    assert _kind("cp.Minimize(cp.sum_squares(R @ w) + c @ w)") == ("least-squares", "")


def test_a_ridge_is_not_read_as_a_linear_term():
    """``sum_squares(w)`` is quadratic; folding it into ``linear`` drops the curvature."""
    _, transform = _kind("cp.Minimize(cp.sum_squares(A - B @ w) + c * cp.sum_squares(w))")
    assert transform, "a ridge that reports no transform was read as a linear term"


def test_a_negative_ridge_is_not_a_design_augmentation():
    """A subtracted L2 term is concave; there is no real square root to stack."""
    assert _kind("cp.Minimize(cp.sum_squares(A - B @ w) - c * cp.sum_squares(w))")[0] \
        == "other"


def test_an_objective_with_no_squared_residual_is_not_least_squares():
    assert _kind("cp.Minimize(c @ w)")[0] == "other"


def test_a_name_with_no_binding_is_unreadable_not_wrong():
    assert _kind("cp.Minimize(objective)")[0] == "unreadable"
    assert _kind("cp.Minimize(loss)")[0] == "unreadable"


_TWO_FUNCTIONS = '''
import cvxpy as cp

def fit_time_weights(X, y):
    lam = cp.Variable(4, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(X @ lam - y))
    constraints = [cp.sum(lam) == 1]
    return cp.Problem(objective, constraints)

def fit_unit_weights(X, y):
    omega = cp.Variable(4)
    objective = cp.Minimize(cp.norm_inf(X @ omega - y))
    constraints = [cp.sum(omega) == 1, X @ omega >= 0]
    return cp.Problem(objective, constraints)
'''


def test_a_name_is_resolved_inside_its_own_function(tmp_path):
    """Two functions, each binding ``objective`` and ``constraints``.

    Following a name by walking the whole module and keeping the last
    assignment gives both sites the second function's program. SpSyDiD is
    written this way, and the audit read its time-weight solve as the
    unit-weight one: wrong variable, wrong constraint list, wrong objective,
    and the verdict happened to agree, which is what made it invisible.
    """
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "two.py").write_text(_TWO_FUNCTIONS)
    by = {s.variable: s for s in audit(tmp_path)}

    assert set(by) == {"lam", "omega"}
    assert by["lam"].verdict == "eligible"
    assert by["omega"].verdict == "extra-constraints"


_CONDITIONAL = '''
import cvxpy as cp

def fit(X, y, use_augmented):
    w = cp.Variable(4)
    objective = cp.Minimize(cp.sum_squares(X @ w - y))
    constraints = [cp.sum(w) == 1]
    if not use_augmented:
        constraints.append(w >= 0)
    return cp.Problem(objective, constraints)

class _CachedProblem:
    def __init__(self, n, J, kind):
        self.w = cp.Variable(n, nonneg=True)

def solve(n, J, kind):
    prob = _CachedProblem(n, J, kind)
    return prob
'''


def test_a_conditional_constraint_does_not_establish_non_negativity(tmp_path):
    """``constraints.append(w >= 0)`` under an ``if`` holds on one branch only.

    SHC writes its kernel weights this way. A list assembled in pieces has to
    be followed or the constraint set is unreadable, but a piece added on a
    branch is not a property of the program: reading it as one would report a
    site as the probability simplex on the strength of a flag's default.
    """
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "cond.py").write_text(_CONDITIONAL)
    sites = {s.variable: s for s in audit(tmp_path)}
    assert sites["w"].verdict == "no-nonnegativity"


def test_a_class_whose_name_ends_in_problem_is_not_a_cvxpy_problem(tmp_path):
    """``_PenalizedProblem(n, J, kind)`` is a constructor, not ``cp.Problem``.

    LAXSCM caches compiled problems behind two such classes. Matching any
    callable whose name ends in ``Problem`` put both constructors in the audit
    with no objective and no constraints to read.
    """
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "cond.py").write_text(_CONDITIONAL)
    found = audit(tmp_path)
    assert [s.variable for s in found] == ["w"], found


def test_a_maximisation_is_not_a_least_squares_minimisation():
    assert _kind("cp.Maximize(cp.sum_squares(A - B @ w))")[0] == "other"


def test_a_negated_objective_is_concave_and_not_carried():
    """``-||A - Bw||^2`` is a maximisation written with a sign."""
    assert _kind("cp.Minimize(-cp.sum_squares(A - B @ w))")[0] == "other"


def test_a_signed_operand_does_not_open_a_new_term():
    """``c * -d @ w`` is one term; splitting at that ``-`` would make it two."""
    assert _kind("cp.Minimize(cp.sum_squares(A - B @ w) + c * -d @ w)") \
        == ("least-squares", "")


def test_a_float_exponent_is_not_a_subtraction():
    """``1e-8`` carries a ``-`` that is part of the literal, not a term break."""
    assert _kind("cp.Minimize(cp.sum_squares(A - B @ w) + 1e-8 * cp.sum_squares(w))") \
        == ("least-squares", "augment the design with a multiple of the identity")


_KEYWORD_CONSTRAINTS = '''
import cvxpy as cp

def fit(X, y):
    w = cp.Variable(4, nonneg=True)
    return cp.Problem(cp.Minimize(cp.sum_squares(X @ w - y)),
                      constraints=[cp.sum(w) == 1])

def opaque(X, y, objective):
    w = cp.Variable(3, nonneg=True)
    return cp.Problem(cp.Minimize(objective), [cp.sum(w) == 1])
'''


def test_the_constraints_keyword_is_read_like_the_positional_argument(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "kw.py").write_text(_KEYWORD_CONSTRAINTS)
    by = {s.lineno: s for s in audit(tmp_path)}
    assert [s.verdict for s in by.values()] == ["eligible", "unreadable"]


def test_a_simplex_site_can_have_an_objective_the_audit_cannot_read(tmp_path):
    """Constraints readable, objective not: the verdict follows the objective.

    The site is the probability simplex, so it clears the constraint half of
    eligibility, and the audit still cannot say what is minimised.
    """
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "kw.py").write_text(_KEYWORD_CONSTRAINTS)
    site = [s for s in audit(tmp_path) if s.variable == "w"][-1]
    assert (site.verdict, site.objective_kind) == ("unreadable", "unreadable")
    assert not site.extras


def test_two_squared_residuals_are_not_read_as_one_fit():
    """``||A - Bw||^2 + ||C - Dw||^2`` is expressible, by stacking the designs.

    No site in the library writes it, so the classifier answers conservatively
    instead of naming a transform nothing exercises. The alternative -- taking
    the first residual as the fit and the second as a ridge -- would hand the
    solver half the objective.
    """
    assert _kind("cp.Minimize(cp.sum_squares(A - B @ w) + cp.sum_squares(C - D @ w))") \
        == ("other", "")


_CONDITIONAL_SIMPLEX = '''
import cvxpy as cp

def fit(X, y, normalise):
    w = cp.Variable(4, nonneg=True)
    constraints = []
    if normalise:
        constraints.append(cp.sum(w) == 1)
    return cp.Problem(cp.Minimize(cp.sum_squares(X @ w - y)), constraints)
'''


def test_a_conditional_sum_to_one_is_not_the_probability_simplex(tmp_path):
    """The mirror of the SHC case, and the one that would cost more.

    Non-negativity read off a branch reports a site as the simplex; so does a
    sum-to-one read off a branch, and there the weights need not sum to
    anything. The site is still listed -- dropping it would hide a solve -- and
    the conditional constraint is what disqualifies it.
    """
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "cond.py").write_text(_CONDITIONAL_SIMPLEX)
    site, = audit(tmp_path)
    assert site.verdict == "extra-constraints"
    assert "conditional" in " ".join(site.extras)
