"""``solve_exact``: the exact backend as a bilevel solver, the solution VanillaSC reads.

It takes the engine's ``BilevelProblem`` (validation outcomes and predictors) and
returns a ``BilevelSolution``: the donor weights of the paper's search, the predictor
weights V that select them (the log-det maximiser, as shares summing to one), the
validation loss, and, under ``metadata["exact"]``, the typed details: the case the
search took, the diagonal and positive definite rationalisers with their centres and
optional ranges, the facet of a positive definite optimum, and optional face counts.
The authors' answers come from ``fixtures/exact/designs_exterior.json`` and from the
captured ``run_all.py`` (``benchmarks/reference/vanillasc_exact``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from _exact_support import EXTERIOR, INTERIOR, exterior_design, inputs, interior_design
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.bilevel import BilevelProblem, BilevelSolution, solve_bilevel
from mlsynth.utils.bilevel.exact import ExactDetails, lsq, solve_exact
from mlsynth.utils.solvers.ridge_augment import simplex_qp

RUN_ALL = json.loads((Path(__file__).resolve().parents[2] / "benchmarks" / "reference" / "vanillasc_exact"
                      / "reference.json").read_text())["values"]


def _problem(case, names=None):
    X1, X0, Z1, Z0 = inputs(case)
    return BilevelProblem(y1_pre=Z1, Y0_pre=Z0, X1=X1, X0=X0, predictor_names=names or [])


def _prop99(**kw):
    case = EXTERIOR["prop99"]
    return case, solve_exact(_problem(case, case["V"]["predictors"]), **kw)


# --------------------------------------------------------------------------- smoke
def test_solve_exact_returns_a_bilevel_solution_with_typed_details():
    case, sol = _prop99()
    assert isinstance(sol, BilevelSolution) and sol.stage == "exact" and sol.iterations == 0
    d = sol.metadata["exact"]
    assert isinstance(d, ExactDetails)
    assert d.v_class == "diagonal" and d.regime == "exterior"
    assert d.predictor_names == tuple(case["V"]["predictors"])
    assert np.isfinite(sol.upper_loss) and sol.W.shape == (38,) and sol.V.shape == (7,)


def test_the_bilevel_dispatch_reaches_the_exact_backend():
    case = EXTERIOR["germany"]
    sol = solve_bilevel(_problem(case), method="exact", v_class="pd")
    assert sol.stage == "exact" and sol.metadata["exact"].v_class == "pd"


# --------------------------------------------------------------------------- unit: the authors' answers
@pytest.mark.parametrize("name", ["prop99", "germany", "prop99_leakage_free"])
def test_donor_weights_and_losses_match_the_authors(name):
    case = EXTERIOR[name]
    for cls in ("diag", "pd"):
        sol = solve_exact(_problem(case), v_class="diagonal" if cls == "diag" else "pd")
        assert abs(sol.upper_loss - case[cls]["loss"]) <= 1e-9 * max(1.0, case[cls]["loss"])
        assert np.abs(sol.W - np.array(case[cls]["w"])).max() <= 1e-7
        assert sol.metadata["exact"].validation_loss == sol.upper_loss


def test_diagonal_predictor_weights_are_the_authors_log_det_maximiser():
    """Table 2: the share on each predictor, and the centre, as the authors' code computes them."""
    case, sol = _prop99()
    V = case["V"]
    d = sol.metadata["exact"].diagonal
    assert np.abs(sol.V - np.array(V["diag_logdet"])).max() <= 1e-4
    assert np.array_equal(sol.V, d.logdet) and abs(sol.V.sum() - 1.0) <= 1e-12
    assert np.abs(d.center - np.array(V["diag_center"])).max() <= 1e-4
    assert [V["predictors"][k] for k in d.forced_zero] == V["diag_forced_zero"]
    assert sorted(d.support + d.forced_zero) == list(range(7)) and d.n_implicit == 33


def test_positive_definite_rationalisers_of_the_diagonal_optimum_are_reported():
    """Table 4: the PD class's log-det maximiser and centre of the diagonal optimum, and its ranges."""
    case, sol = _prop99(ranges=True)
    V = case["V"]
    p = sol.metadata["exact"].pd
    assert np.abs(np.diag(p.logdet) - np.array(V["pd_logdet_diag"])).max() <= 1e-6
    assert np.abs(np.diag(p.center) - np.array(V["pd_center_diag"])).max() <= 1e-4
    assert np.abs(p.range_lo - np.array(V["pd_lo"])).max() <= 1e-6
    assert np.abs(p.range_hi - np.array(V["pd_hi"])).max() <= 1e-6
    assert abs(p.logdet_min_eig - V["pd_logdet_min_eig"]) <= 1e-4 * V["pd_logdet_min_eig"]
    assert abs(np.trace(p.logdet) - 1.0) <= 1e-9 and abs(np.trace(p.center) - 1.0) <= 1e-9


def test_ranges_are_computed_only_on_request():
    _, sol = _prop99()
    assert sol.metadata["exact"].pd.range_lo is None and sol.metadata["exact"].pd.range_hi is None


def test_the_lower_bound_is_the_unconstrained_fit():
    """The bound the predictor constraint is measured against: the best simplex fit of the validation
    outcomes with no V at all, 58.46 against a smaller number on Proposition 99."""
    case, sol = _prop99()
    _, _, Z1, Z0 = inputs(case)
    w = simplex_qp(Z0, Z1)
    assert np.isclose(sol.lower_bound, np.sum((Z1 - Z0 @ w) ** 2), rtol=1e-10)
    assert sol.lower_bound < 0.9 * sol.upper_loss


def test_face_counts_are_computed_on_request_outside_the_hull():
    case = EXTERIOR["germany"]
    d = solve_exact(_problem(case), face_counts=True).metadata["exact"]
    assert d.face_counts is not None and d.face_counts.total == (16, 104, 341, 583, 495, 165)
    assert solve_exact(_problem(case)).metadata["exact"].face_counts is None
    inner = solve_exact(_problem(INTERIOR["random_interior_K3_J12_seed4007"]), face_counts=True)
    assert inner.metadata["exact"].face_counts is None


@pytest.mark.parametrize("name,tag,n_facets", [("prop99", "california", 4), ("germany", "germany", 1)])
def test_a_positive_definite_optimum_reports_the_facets_that_hold_it(name, tag, n_facets):
    """Tables D.1 and D.2: the visible facets, and the facet run_all.py reports among those holding the
    optimum. California's optimum lies on a face of five donors shared by four visible facets, all with
    the same fit; the authors' search keeps whichever its solver's rounding put lowest, so the test asks
    only that their facet is one of the four."""
    case = EXTERIOR[name]
    sol = solve_exact(_problem(case), v_class="pd")
    d = sol.metadata["exact"]
    assert d.n_visible_facets == RUN_ALL[f"{tag}.pd.n_visible_facets"]
    want = tuple(sorted(case["donors"].index(k[len(tag) + 10:-1])
                        for k in RUN_ALL if k.startswith(f"{tag}.pd.facet[")))
    assert len(d.facets) == n_facets and want in d.facets
    _, _, Z1, Z0 = inputs(case)
    support = set(np.flatnonzero(sol.W))
    for F in d.facets:
        val, _ = lsq.simplex_ls(Z1, Z0[:, list(F)])
        assert support <= set(F) and np.isclose(val, sol.upper_loss, rtol=1e-9)
    assert d.diagonal is None
    assert np.allclose(sol.V, np.diag(d.pd.logdet) / np.trace(d.pd.logdet), atol=1e-12)


def test_the_interior_case_is_reported_as_such():
    case = INTERIOR["random_interior_K3_J12_seed4007"]
    for cls in ("diagonal", "pd"):
        d = solve_exact(_problem(case), v_class=cls).metadata["exact"]
        assert d.regime == "interior" and d.n_visible_facets is None and d.facets is None


# --------------------------------------------------------------------------- invariants
def _lower_level_optimal(sol, X1, X0):
    """W minimises ||X1 - X0 w||_V over the simplex under the reported V (value, not argmin)."""
    s = np.sqrt(sol.V)
    w = simplex_qp(s[:, None] * X0, s * X1)
    best = float(np.sum(sol.V * (X1 - X0 @ w) ** 2))
    return sol.lower_loss <= best + 1e-9 * max(1.0, best)


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(3, 4), extra=st.integers(4, 8), interior=st.booleans())
def test_property_the_solution_is_consistent(seed, K, extra, interior):
    X1, X0, Z1, Z0 = (interior_design if interior else exterior_design)(seed, K, K + extra)
    prob = BilevelProblem(y1_pre=Z1, Y0_pre=Z0, X1=X1, X0=X0)
    sol = solve_exact(prob)
    assert sol.W.min() >= 0.0 and abs(sol.W.sum() - 1.0) <= 1e-12
    assert sol.V.min() >= 0.0 and abs(sol.V.sum() - 1.0) <= 1e-9
    assert np.isclose(sol.upper_loss, np.sum((Z1 - Z0 @ sol.W) ** 2), rtol=1e-12)
    assert sol.lower_bound <= sol.upper_loss * (1 + 1e-9) + 1e-12
    assert np.isclose(sol.lower_loss, np.sum(sol.V * (X1 - X0 @ sol.W) ** 2), rtol=1e-9, atol=1e-15)
    assert _lower_level_optimal(sol, X1, X0)
    pd = solve_exact(prob, v_class="pd")
    assert pd.upper_loss <= sol.upper_loss * (1 + 1e-9) + 1e-12


# --------------------------------------------------------------------------- failure
def test_an_unknown_class_is_refused():
    with pytest.raises(MlsynthConfigError, match="v_class"):
        solve_exact(_problem(EXTERIOR["germany"]), v_class="full")


def test_a_problem_without_predictors_is_refused():
    case = EXTERIOR["germany"]
    _, _, Z1, Z0 = inputs(case)
    prob = BilevelProblem(y1_pre=Z1, Y0_pre=Z0, X1=np.empty(0), X0=np.empty((0, Z0.shape[1])))
    with pytest.raises(MlsynthDataError, match="predictors"):
        solve_exact(prob)


# --------------------------------------------------------------------------- the typed details' helpers
@settings(max_examples=25, deadline=None)
@given(K=st.integers(1, 6), seed=st.integers(0, 10_000))
def test_property_shares_and_names_line_up(K, seed):
    from mlsynth.utils.bilevel.exact import PDRationalizers
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(K, K))
    V = L @ L.T + 0.1 * np.eye(K)
    p = PDRationalizers(logdet=V, center=2.0 * V, logdet_min_eig=float(np.linalg.eigvalsh(V).min()))
    assert np.isclose(p.logdet_shares.sum(), 1.0) and np.allclose(p.logdet_shares, p.center_shares)
    names = tuple(f"p{k}" for k in range(K))
    d = ExactDetails(v_class="pd", regime="exterior", predictor_names=names, validation_loss=1.0, pd=p)
    assert d.by_name(p.logdet_shares) == {n: float(v) for n, v in zip(names, p.logdet_shares)}
    with pytest.raises(AttributeError):
        d.regime = "interior"                                # frozen
