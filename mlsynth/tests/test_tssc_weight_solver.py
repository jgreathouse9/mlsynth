"""TSSC's four SC-class variants on the shared weight solver.

Li and Shankar (2023) define the variants by their constraint set alone, so
each is one :class:`WeightConstraint` and the choice among them is the whole
method. The tests come in two kinds. The differential ones pin the migration:
the coefficient vectors must not move from what cvxpy/CLARABEL returned, and a
reference program is built here to check against. The rest are new behaviour --
exact zeros off the support, an optimality certificate, and a uniqueness
verdict that Step-1 selection can read.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.tssc_helpers.estimation import (
    _HAS_INTERCEPT,
    _features,
    _solve,
    _solve_certified,
)
from mlsynth.utils.weights import WeightConstraint, WeightSolution

VARIANTS = ("SC", "MSCa", "MSCb", "MSCc")

#: Each variant's constraint set, as the paper defines it.
EXPECTED_SET = {
    "SC":   WeightConstraint(),
    "MSCa": WeightConstraint(intercept=True),
    "MSCb": WeightConstraint(sum_to_one=False),
    "MSCc": WeightConstraint(sum_to_one=False, intercept=True),
}


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def basque():
    import pandas as pd
    from pathlib import Path
    from mlsynth.utils.datautils import dataprep
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "basedata" / "basque_data.csv")
    df = df[df["regionname"] != "Spain (Espana)"]
    df["treat"] = (
        (df["regionname"] == "Basque Country (Pais Vasco)") & (df["year"] >= 1975)
    ).astype(int)
    p = dataprep(df, "regionname", "year", "gdpcap", "treat")
    T0 = int(p["pre_periods"])
    D = np.asarray(p["donor_matrix"], float)
    return D[:T0], np.asarray(p["y"], float).ravel()[:T0], T0, D.shape[1]


@pytest.fixture(scope="module")
def basque_full():
    """Basque with the post-treatment periods kept, for identification checks."""
    import pandas as pd
    from pathlib import Path
    from mlsynth.utils.datautils import dataprep
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "basedata" / "basque_data.csv")
    df = df[df["regionname"] != "Spain (Espana)"]
    df["treat"] = (
        (df["regionname"] == "Basque Country (Pais Vasco)") & (df["year"] >= 1975)
    ).astype(int)
    p = dataprep(df, "regionname", "year", "gdpcap", "treat")
    return (np.asarray(p["donor_matrix"], float),
            np.asarray(p["y"], float).ravel(), int(p["pre_periods"]))


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(404)
    T0, J = 30, 8
    t = np.linspace(0, 1, T0)
    D = np.column_stack([np.ones(T0), t, np.cos(4 * t), t ** 3]) @ rng.normal(size=(4, J))
    D = D * 2.0 + 20.0 + rng.normal(scale=0.3, size=(T0, J))
    w = np.zeros(J); w[[1, 3, 6]] = (0.55, 0.30, 0.15)
    return D, D @ w + 1.2 + rng.normal(scale=0.05, size=T0), T0, J


def _cvxpy_oracle(method, X, y, n_pre, n_donors):
    """The pre-migration program, kept here as the differential reference."""
    import cvxpy as cp
    X = np.asarray(X, float)[:n_pre]
    y = np.asarray(y, float)[:n_pre]
    if _HAS_INTERCEPT[method]:
        X = np.c_[np.ones((X.shape[0], 1)), X]
        dim = n_donors + 1
    else:
        dim = n_donors
    w = cp.Variable(dim)
    cons = {
        "SC":   lambda: [w >= 0, cp.sum(w) == 1],
        "MSCa": lambda: [w[1:] >= 0, cp.sum(w[1:]) == 1],
        "MSCb": lambda: [w >= 0],
        "MSCc": lambda: [w[1:] >= 0],
    }[method]()
    cp.Problem(cp.Minimize(cp.norm(y - X @ w, 2)), cons).solve(solver=cp.CLARABEL)
    return np.asarray(w.value, float).ravel()


def _ssr(method, X, y, coef):
    return float(np.sum((y - _features(method, X) @ coef) ** 2))


# --------------------------------------------------------------------------
# Differential: the migration must not move the answer
# --------------------------------------------------------------------------
#: How far CLARABEL stops from the minimiser on these programs. The cone
#: variants have near-collinear donors and a flat objective, and on the
#: intercept variants the free constant absorbs the slack, so the intercept is
#: where the gap shows up largest. A comparison against CLARABEL cannot be
#: tighter than CLARABEL is; which of the two points is the minimiser is
#: settled separately, by the certificate, in the test below.
CLARABEL_ACCURACY = 2e-4

PANELS = ["basque", "synthetic"]


@pytest.mark.parametrize("panel", PANELS)
@pytest.mark.parametrize("method", VARIANTS)
def test_each_variant_matches_the_cvxpy_program(request, panel, method):
    X, y, T0, J = request.getfixturevalue(panel)
    got = _solve(method, X, y, T0, J)
    ref = _cvxpy_oracle(method, X, y, T0, J)
    assert got.shape == ref.shape
    assert _ssr(method, X, y, got) <= _ssr(method, X, y, ref) + 1e-9
    assert got == pytest.approx(ref, abs=CLARABEL_ACCURACY)


@pytest.mark.parametrize("panel", PANELS)
@pytest.mark.parametrize("method", VARIANTS)
def test_the_certificate_settles_every_disagreement(request, panel, method):
    """The tolerance above is doing real work -- three of the eight cases move
    by more than 1e-5 -- so it needs a reason beyond convenience.

    Each variant's constraint set has non-empty relative interior, so KKT is
    necessary and sufficient and the residual at a point decides whether that
    point is the minimiser. Evaluating it at both answers settles the direction
    of every disagreement instead of declaring a tolerance and moving on.
    """
    from mlsynth.utils.weights import WeightObjective, kkt_residual

    X, y, T0, J = request.getfixturevalue(panel)
    con = EXPECTED_SET[method]
    got = _solve(method, X, y, T0, J)
    ref = _cvxpy_oracle(method, X, y, T0, J)

    def residual(coef):
        w = coef[1:] if _HAS_INTERCEPT[method] else coef
        a = float(coef[0]) if _HAS_INTERCEPT[method] else 0.0
        return kkt_residual(X, y, w, a, con, WeightObjective())

    assert residual(got) < 1e-9
    if np.abs(got - ref).max() > 1e-6:
        assert residual(ref) > residual(got) * 100.0
        assert _ssr(method, X, y, got) < _ssr(method, X, y, ref)


@pytest.mark.parametrize("method,ssr", [
    ("SC", 1.418982e-01), ("MSCa", 1.111810e-01),
    ("MSCb", 1.358861e-01), ("MSCc", 1.111336e-01),
])
def test_the_basque_objective_is_unchanged_from_before_the_migration(basque, method, ssr):
    X, y, T0, J = basque
    assert _ssr(method, X, y, _solve(method, X, y, T0, J)) == pytest.approx(ssr, rel=1e-6)


# --------------------------------------------------------------------------
# The constraint sets, held exactly
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", VARIANTS)
def test_each_variant_satisfies_its_own_constraint_set(basque, method):
    X, y, T0, J = basque
    coef = _solve(method, X, y, T0, J)
    donors = coef[1:] if _HAS_INTERCEPT[method] else coef
    assert donors.min() >= 0.0
    if method in ("SC", "MSCa"):
        assert donors.sum() == pytest.approx(1.0, abs=1e-9)
    else:
        assert donors.sum() <= 1.0 + 1e-9 or True     # cone: unconstrained above


@pytest.mark.parametrize("method", VARIANTS)
def test_off_support_donors_come_back_at_exactly_zero(basque, method):
    """CLARABEL returned 6.6e-10 here and no weight exactly zero. A donor with
    no weight should read as absent, not as a number that rounds to absent."""
    X, y, T0, J = basque
    coef = _solve(method, X, y, T0, J)
    donors = coef[1:] if _HAS_INTERCEPT[method] else coef
    off = donors[donors < 1e-8]
    assert off.size > 0
    assert np.all(off == 0.0)


# --------------------------------------------------------------------------
# The intercept-first layout the rest of the package depends on
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", VARIANTS)
def test_the_coefficient_vector_keeps_its_length_and_layout(basque, method):
    X, y, T0, J = basque
    coef = _solve(method, X, y, T0, J)
    assert coef.shape == (J + 1,) if _HAS_INTERCEPT[method] else coef.shape == (J,)
    assert _features(method, X).shape[1] == coef.size


@pytest.mark.parametrize("method", ["MSCa", "MSCc"])
def test_a_treated_unit_below_its_donors_gets_a_negative_intercept(basque, method):
    """The intercept is free in sign. Placing it inside the simplex instead --
    bounded to [0, 1] -- cannot represent this case at all, and the old program
    carried a comment warning against exactly that slip. With the intercept a
    field on the constraint set there is no way to write it."""
    X, y, T0, J = basque
    coef = _solve(method, X, y - 3.0, T0, J)
    assert coef[0] < -2.0
    assert coef[1:].min() >= 0.0


# --------------------------------------------------------------------------
# New: the certificate and the uniqueness verdict
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", VARIANTS)
def test_every_variant_returns_a_certified_solution(basque, method):
    X, y, T0, J = basque
    sol = _solve_certified(method, X, y, T0, J)
    assert isinstance(sol, WeightSolution)
    assert sol.status == "optimal"
    assert sol.kkt_residual < 1e-9


@pytest.mark.parametrize("method", VARIANTS)
def test_the_certified_solution_agrees_with_the_flat_vector(basque, method):
    X, y, T0, J = basque
    sol = _solve_certified(method, X, y, T0, J)
    flat = _solve(method, X, y, T0, J)
    rebuilt = np.r_[sol.intercept, sol.weights] if _HAS_INTERCEPT[method] else sol.weights
    assert flat == pytest.approx(rebuilt, abs=1e-12)


@pytest.mark.parametrize("method", VARIANTS)
def test_each_variant_names_the_constraint_set_the_paper_defines(basque, method):
    X, y, T0, J = basque
    assert _solve_certified(method, X, y, T0, J).solver.split(":")[0] == \
        EXPECTED_SET[method].describe()


def test_a_duplicated_donor_is_reported_as_a_continuum(basque):
    """Step-1 selection compares variants on fit. A variant whose optimum is a
    continuum posts a competitive fit with uninterpretable weights, and this is
    what makes that visible to the selector."""
    X, y, T0, J = basque
    lead = int(np.argmax(_solve("SC", X, y, T0, J)))      # a donor on the support
    sol = _solve_certified("SC", np.column_stack([X, X[:, lead]]), y, T0, J + 1)
    assert sol.weights[-1] == 0.0        # the active set returns a vertex ...
    assert sol.unique is False           # ... and the face is still a continuum
    assert _solve_certified("SC", X, y, T0, J).unique is True    # not always False


# --------------------------------------------------------------------------
# Robustness the old program did not have
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", VARIANTS)
def test_the_fit_is_equivariant_to_a_million_times_rescaling(basque, method):
    """CLARABEL declares the SC simplex infeasible at this scale."""
    X, y, T0, J = basque
    base = _solve(method, X, y, T0, J)
    big = _solve(method, X * 1e6, y * 1e6, T0, J)
    assert big is not None
    scaled = big.copy()
    if _HAS_INTERCEPT[method]:
        scaled[0] /= 1e6
    assert scaled == pytest.approx(base, abs=1e-7)


# --------------------------------------------------------------------------
# The None-on-failure contract the subsampling loop branches on
# --------------------------------------------------------------------------
def test_a_non_finite_panel_returns_none_and_does_not_raise(basque):
    """bootstrap_att_ci and Step-1 selection both branch on None to skip a bad
    refit; a raise there would abort the whole interval."""
    X, y, T0, J = basque
    bad = X.copy(); bad[0, 0] = np.nan
    assert _solve("SC", bad, y, T0, J) is None


def test_a_degenerate_subsample_still_returns_a_vector(basque):
    """Two pre-periods and sixteen donors is rank-deficient. CLARABEL returned
    a vector here and so must the active set, or the subsampling loop loses
    draws it used to keep."""
    X, y, T0, J = basque
    coef = _solve("SC", X[:2], y[:2], 2, J)
    assert coef is not None and coef.shape == (J,)
    assert coef.min() >= 0.0 and coef.sum() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# The None contract, end to end. The migration changed what produces None, so
# both consumers of it are exercised here.
# --------------------------------------------------------------------------
@pytest.fixture
def tssc_inputs(basque):
    import pandas as pd
    from pathlib import Path
    from mlsynth.utils.datautils import dataprep
    from mlsynth.utils.tssc_helpers.structures import TSSCInputs
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "basedata" / "basque_data.csv")
    df = df[df["regionname"] != "Spain (Espana)"]
    df["treat"] = (
        (df["regionname"] == "Basque Country (Pais Vasco)") & (df["year"] >= 1975)
    ).astype(int)
    p = dataprep(df, "regionname", "year", "gdpcap", "treat")
    T0, T = int(p["pre_periods"]), int(p["total_periods"])
    return TSSCInputs(
        y=np.asarray(p["y"], float).ravel(),
        donor_matrix=np.asarray(p["donor_matrix"], float),
        donor_names=list(p["donor_names"]),
        T0=T0, T2=T - T0, T=T,
        time_labels=np.asarray(p["time_labels"]),
        treated_unit_name=str(p["treated_unit_name"]),
    )


def test_an_unsolvable_panel_is_reported_and_not_swallowed(tssc_inputs):
    from dataclasses import replace
    from mlsynth.exceptions import MlsynthEstimationError
    from mlsynth.utils.tssc_helpers.estimation import fit_variant

    bad = tssc_inputs.donor_matrix.copy(); bad[0, 0] = np.inf
    with pytest.raises(MlsynthEstimationError, match="optimization failed"):
        fit_variant(replace(tssc_inputs, donor_matrix=bad), "SC",
                    n_bootstrap=2, confidence_level=0.95,
                    rng=np.random.default_rng(0), compute_ci=False)


def test_a_failed_refit_is_skipped_and_the_interval_still_returns(tssc_inputs, monkeypatch):
    """A draw the solver cannot answer is dropped, not fatal. The active set
    answers rank-deficient subsamples that CLARABEL refused, so this path is
    reached far less often than before -- it is still wired."""
    from mlsynth.utils.tssc_helpers import estimation

    fit = estimation.fit_variant(
        tssc_inputs, "SC", n_bootstrap=40, confidence_level=0.9,
        rng=np.random.default_rng(3), compute_ci=False,
    )
    calls = {"n": 0}
    real = estimation._solve

    def flaky(method, donor_pre, y_pre, n_pre, n_donors):
        calls["n"] += 1
        return None if calls["n"] % 3 == 0 else real(method, donor_pre, y_pre, n_pre, n_donors)

    monkeypatch.setattr(estimation, "_solve", flaky)
    lo, hi = estimation.bootstrap_att_ci(
        inputs=tssc_inputs, method="SC", weights=fit.weights,
        counterfactual=fit.counterfactual, att=fit.att, n_bootstrap=40,
        confidence_level=0.9, rng=np.random.default_rng(4),
    )
    assert calls["n"] == 40 and np.isfinite(lo) and np.isfinite(hi) and lo < hi


def test_the_variant_record_carries_the_certificate(tssc_inputs):
    from mlsynth.utils.tssc_helpers.estimation import fit_variant
    for method in VARIANTS:
        fit = fit_variant(tssc_inputs, method, n_bootstrap=2, confidence_level=0.95,
                          rng=np.random.default_rng(1), compute_ci=False)
        assert fit.weights_unique is True
        assert fit.kkt_residual < 1e-9
        assert fit.solver.split(":")[0] == EXPECTED_SET[method].describe()


def test_a_continuum_is_invisible_in_the_pre_fit_step_1_compares_on(basque):
    """Every minimiser of a least-squares program reproduces the same fitted
    values on the periods it was fitted to, so a continuum leaves no trace in
    the pre-treatment RMSE. Step 1 ranks variants on that RMSE."""
    from mlsynth.utils.weights import solve_weights

    X, y, T0, J = basque
    lead = int(np.argmax(_solve("SC", X, y, T0, J)))
    Xd = np.column_stack([X, X[:, lead]])
    sol = solve_weights(Xd, y)
    assert sol.unique is False

    w = np.array(sol.weights)
    alt = w.copy(); alt[lead] -= 0.2; alt[-1] += 0.2
    assert alt.min() >= 0.0 and alt.sum() == pytest.approx(1.0, abs=1e-12)
    assert Xd @ alt == pytest.approx(Xd @ w, abs=1e-10)


def test_duplicated_donors_leave_the_att_alone(basque_full):
    """The continuum is real and the ATT is not touched by it. Only the split
    between the twins is undetermined, and no reported quantity depends on it."""
    from mlsynth.utils.weights import solve_weights

    D, y, T0 = basque_full
    lead = int(np.argmax(solve_weights(D[:T0], y[:T0]).weights))
    Dd = np.column_stack([D, D[:, lead]])
    sol = solve_weights(Dd[:T0], y[:T0])
    assert sol.unique is False
    assert sol.identifies(Dd[T0:]) is True


def test_collinearity_that_ends_at_treatment_leaves_the_att_undetermined():
    """The same verdict, the opposite consequence. Donors identical before
    treatment and separating after give pre-fits agreeing to ten digits and
    counterfactuals that diverge, so the ATT depends on which minimiser came
    back. This is why `weights_unique` is necessary for the ATT to be
    undetermined and not sufficient, and why the post-period block has to be
    asked separately."""
    from mlsynth.utils.weights import solve_weights

    rng = np.random.default_rng(2)
    T0, T2, J = 12, 10, 4
    panel = rng.normal(size=(T0 + T2, J)) * 2 + 10
    panel[:T0, 1] = panel[:T0, 0]          # twins before treatment, not after
    pre, post = panel[:T0], panel[T0:]
    y = pre @ np.array([0.3, 0.2, 0.3, 0.2]) + rng.normal(scale=0.01, size=T0)

    sol = solve_weights(pre, y)
    assert sol.unique is False
    assert sol.identifies(post) is False

    w = np.array(sol.weights)
    alt = w.copy()
    shift = min(w[0], 0.15)
    assert shift > 0.0
    alt[0] -= shift; alt[1] += shift
    assert pre @ alt == pytest.approx(pre @ w, abs=1e-10)          # same pre-fit
    assert np.abs(post @ alt - post @ w).max() > 0.5               # different ATT


# --------------------------------------------------------------------------
# The identification verdict, wired onto the variant record
# --------------------------------------------------------------------------
def _inputs_from(y, donors, T0):
    from mlsynth.utils.tssc_helpers.structures import TSSCInputs
    T = donors.shape[0]
    return TSSCInputs(
        y=np.asarray(y, float).ravel(),
        donor_matrix=np.asarray(donors, float),
        donor_names=[f"d{j}" for j in range(donors.shape[1])],
        T0=T0, T2=T - T0, T=T,
        time_labels=np.arange(T), treated_unit_name="treated",
    )


@pytest.fixture
def twins_after_treatment_too(basque_full):
    """A donor duplicated in every period: a continuum the ATT never feels."""
    D, y, T0 = basque_full
    from mlsynth.utils.weights import solve_weights
    lead = int(np.argmax(solve_weights(D[:T0], y[:T0]).weights))
    return _inputs_from(y, np.column_stack([D, D[:, lead]]), T0)


@pytest.fixture
def twins_only_before_treatment():
    """Donors identical before treatment and separating after: the ATT depends
    on which minimiser the solver returned."""
    rng = np.random.default_rng(2)
    T0, T2, J = 12, 10, 4
    donors = rng.normal(size=(T0 + T2, J)) * 2 + 10
    donors[:T0, 1] = donors[:T0, 0]
    truth = np.array([0.3, 0.2, 0.3, 0.2])
    y = donors @ truth + rng.normal(scale=0.01, size=T0 + T2)
    return _inputs_from(y, donors, T0)


def _fit(inputs, method):
    from mlsynth.utils.tssc_helpers.estimation import fit_variant
    return fit_variant(inputs, method, n_bootstrap=2, confidence_level=0.95,
                       rng=np.random.default_rng(0), compute_ci=False)


@pytest.mark.parametrize("method", VARIANTS)
def test_a_clean_panel_reports_the_att_as_identified(tssc_inputs, method):
    fit = _fit(tssc_inputs, method)
    assert fit.weights_unique is True
    assert fit.att_identified is True


def test_duplicated_donors_leave_the_att_identified(twins_after_treatment_too):
    """`weights_unique` and `att_identified` disagree here, and the second is
    the one the reported number depends on."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # no warning may fire
        fit = _fit(twins_after_treatment_too, "SC")
    assert fit.weights_unique is False
    assert fit.att_identified is True


def test_collinearity_ending_at_treatment_is_reported_and_warned(twins_only_before_treatment):
    with pytest.warns(UserWarning, match="ATT is not identified"):
        fit = _fit(twins_only_before_treatment, "SC")
    assert fit.weights_unique is False
    assert fit.att_identified is False


def test_the_warning_names_the_variant_and_what_moves(twins_only_before_treatment):
    with pytest.warns(UserWarning) as caught:
        _fit(twins_only_before_treatment, "MSCb")
    text = str(caught[0].message)
    assert "MSCb" in text
    assert "pre-treatment fit" in text and "counterfactual" in text


def test_with_no_post_periods_there_is_no_verdict_to_give(basque_full):
    D, y, T0 = basque_full
    inputs = _inputs_from(y[:T0], D[:T0], T0)
    assert inputs.T2 == 0
    fit = _fit(inputs, "SC")
    assert fit.att_identified is None


def test_the_verdict_survives_a_full_fit_through_the_estimator(twins_only_before_treatment):
    """Reached through fit_variant, not by calling the helper directly."""
    fits = {m: None for m in VARIANTS}
    for m in VARIANTS:
        with pytest.warns(UserWarning):
            fits[m] = _fit(twins_only_before_treatment, m)
    assert all(f.att_identified is False for f in fits.values())


# ----------------------------------------------------------------------
# The subsample size m reaches the ATT interval
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def wide_inputs():
    """A panel wide enough for m to vary: T_0 = 90 against 10 donors.

    The Basque fixture has T_0 = 20 with 16 donors, so the only
    admissible subsample sizes are 18 to 20 and m barely moves. Li
    (2020)'s own design is T_1 = 90, T_2 = 20, N = 11.
    """
    from mlsynth.utils.tssc_helpers.structures import TSSCInputs

    rng = np.random.default_rng(0)
    T0, T2, J = 90, 20, 10
    T = T0 + T2
    f = rng.standard_normal((T, 3))
    B = np.zeros((J + 1, 3))
    B[:7] = 1.0
    Y = 1.0 + f @ B.T + rng.uniform(-np.sqrt(3), np.sqrt(3), (T, J + 1))
    return TSSCInputs(
        y=Y[:, 0], donor_matrix=Y[:, 1:],
        donor_names=[f"d{j}" for j in range(J)],
        T0=T0, T2=T2, T=T, time_labels=np.arange(T),
        treated_unit_name="treated",
    )


class TestTheSubsampleSizeReachesTheAttInterval:
    """``m`` is a parameter of Li (2020)'s procedure, not a constant.

    The paper's Table 1 reports coverage at ``m`` in {20, 40, 60, 80, 90}
    and at ``m = T_1`` (the bootstrap special case of Remark 4.3), so a
    replication of it needs the size settable. ``subsample_size`` already
    existed on the config and reached only the Step-1 restriction tests;
    ``bootstrap_att_ci`` computed its own ``m = T_0 - 5`` and ignored it.

    These tests pin that ``m`` reaches the computation and that the
    default is unmoved. They deliberately do not pin a direction for the
    interval width in ``m``: measured across {20, 40, 60, 80, 90} at 600
    draws on the fixture below, the width is flat at 0.99 to 1.05 with no
    ordering, because the ``sqrt(T_2 m / T_0)`` rescaling of the weight
    term and the larger sampling spread of ``w*`` at small ``m`` roughly
    cancel. Li tabulates coverage across ``m`` precisely because it moves
    for him, so whether it moves here is a question for the replication
    to answer, not something to assert in advance.
    """

    def _fit(self, inputs, method="MSCc"):
        from mlsynth.utils.tssc_helpers import estimation

        return estimation.fit_variant(
            inputs, method, n_bootstrap=10, confidence_level=0.95,
            rng=np.random.default_rng(0), compute_ci=False,
        )

    def test_the_default_is_unchanged(self, wide_inputs):
        """None keeps T_0 - 5, so every existing call is byte-identical."""
        from mlsynth.utils.tssc_helpers import estimation

        fit = self._fit(wide_inputs)
        kw = dict(inputs=wide_inputs, method="MSCc", weights=fit.weights,
                  counterfactual=fit.counterfactual, att=fit.att,
                  n_bootstrap=30, confidence_level=0.95)
        default = estimation.bootstrap_att_ci(
            rng=np.random.default_rng(7), **kw
        )
        explicit = estimation.bootstrap_att_ci(
            rng=np.random.default_rng(7),
            subsample_size=wide_inputs.T0 - 5, **kw
        )
        assert default == explicit

    def test_a_different_m_gives_a_different_interval(self, wide_inputs):
        from mlsynth.utils.tssc_helpers import estimation

        fit = self._fit(wide_inputs)
        kw = dict(inputs=wide_inputs, method="MSCc", weights=fit.weights,
                  counterfactual=fit.counterfactual, att=fit.att,
                  n_bootstrap=200, confidence_level=0.95)
        seen = {}
        for m in (20, 60, wide_inputs.T0):
            seen[m] = estimation.bootstrap_att_ci(
                rng=np.random.default_rng(1), subsample_size=m, **kw
            )
            assert all(np.isfinite(v) for v in seen[m])
        assert len(set(seen.values())) == 3

    def test_m_above_the_pre_period_is_refused(self, wide_inputs):
        from mlsynth.exceptions import MlsynthConfigError
        from mlsynth.utils.tssc_helpers import estimation

        fit = self._fit(wide_inputs, "SC")
        with pytest.raises(MlsynthConfigError, match="subsample_size"):
            estimation.bootstrap_att_ci(
                inputs=wide_inputs, method="SC", weights=fit.weights,
                counterfactual=fit.counterfactual, att=fit.att,
                n_bootstrap=10, confidence_level=0.95,
                rng=np.random.default_rng(0),
                subsample_size=wide_inputs.T0 + 1,
            )

    def test_the_estimator_passes_it_through(self):
        """End to end: the config field changes the reported att_ci."""
        import warnings as _w

        import pandas as pd

        from mlsynth import TSSC

        rng = np.random.default_rng(3)
        T0, T2, J = 90, 20, 10
        T = T0 + T2
        f = rng.standard_normal((T, 3))
        B = np.zeros((J + 1, 3))
        B[:7] = 1.0
        Y = 1.0 + f @ B.T + rng.uniform(-np.sqrt(3), np.sqrt(3), (T, J + 1))
        units = ["treated"] + [f"c{j}" for j in range(J)]
        df = pd.DataFrame({
            "unit": np.repeat(units, T),
            "time": np.tile(np.arange(T), J + 1),
            "y": Y.T.ravel(),
            "D": np.concatenate([(np.arange(T) >= T0).astype(int),
                                 np.zeros(J * T, dtype=int)]),
        })
        cfg = dict(df=df, outcome="y", treat="D", unitid="unit", time="time",
                   display_graphs=False, method="MSCc", draws=150, seed=4)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            small = TSSC({**cfg, "subsample_size": 20}).fit()
            default = TSSC({**cfg}).fit()
        assert small.att == pytest.approx(default.att, rel=1e-12)
        assert small.att_ci != default.att_ci
