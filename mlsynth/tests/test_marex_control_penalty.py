"""MAREX's control-side Abadie and L'Hour penalty (Vives-i-Bastida 2022, program 1).

Vives-i-Bastida's design adds to the weakly-targeted objective a penalty that
charges each control market its distance from the synthetic treated unit,

    ||A - B w||^2 + beta ||B w - B v||^2 + beta lam sum_j v_j ||B w - X_j||^2,

which makes the synthetic control sparse and local (Abadie and L'Hour 2021).
The term is bilinear in ``w`` and ``v``. On the control simplex it equals
``||B w||^2 - 2 w'B'B v + sum_j v_j ||X_j||^2``, and the whole objective is
convex exactly when ``beta lam (1 + lam) <= 1``. Inside that bound both
backends solve it: the MIQP through a sum-of-squares form CVXPY accepts, the
exact backend through the face-Newton coupled fit with the controls priced by
a simplex fit plus a linear term.

References: a general-purpose conic solve of the fixed-support problem for the
oracle, brute-force enumeration and the MIQP for the search, and the stated
objective evaluated directly for every value reported. Levels: smoke,
differential, unit, edge, failure, property.
"""
from __future__ import annotations

import itertools
import warnings

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.marex_helpers.config import MAREXConfig
from mlsynth.utils.marex_helpers.exact import coupled_fit, search_cluster
from mlsynth.utils.marex_helpers.formulation import (
    build_objective,
    init_cvxpy_variables,
    validate_scm_inputs,
)
from mlsynth.utils.syndes_helpers.exact import SubsetFilter
from mlsynth.tests.test_marex_exact_backend import (
    _assert_same_design,
    _both,
    _cfg,
    _fit,
    _panel,
    _problem,
    _treated,
)

warnings.filterwarnings("ignore")


def _objective(B, A, S, C, w, v, beta, lam):
    """Program (1) evaluated as written, with the sum over controls kept."""
    Bs, Bc = B[:, list(S)], B[:, list(C)]
    t, c = Bs @ w, Bc @ v
    pen = sum(v[j] * float((t - Bc[:, j]) @ (t - Bc[:, j]))
              for j in range(len(C)))
    return float((A - t) @ (A - t) + beta * ((t - c) @ (t - c))
                 + beta * lam * pen)


def _joint(B, A, S, C, beta, lam):
    """The fixed-support problem for a conic solver.

    Written from scratch in centred coordinates, ``u = B_S w - A`` and
    ``s = B_C v - A``: the objective is ``a ||u - (b / a) s||^2 + k ||s||^2 +
    beta lam d'v`` with ``a = 1 + beta (1 + lam)``, ``b = beta (1 + lam)``,
    ``k = beta - b^2 / a`` and ``d_j = ||X_j - A||^2``.
    """
    Bc_ = B - A[:, None]
    Bs, Bc = Bc_[:, list(S)], Bc_[:, list(C)]
    w = cp.Variable(len(S), nonneg=True)
    v = cp.Variable(len(C), nonneg=True)
    a = 1.0 + beta * (1.0 + lam)
    b = beta * (1.0 + lam)
    k = beta - b * b / a
    d = (Bc ** 2).sum(axis=0)
    u, s = Bs @ w, Bc @ v
    obj = (a * cp.sum_squares(u - (b / a) * s) + k * cp.sum_squares(s)
           + beta * lam * (d @ v))
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(w) == 1, cp.sum(v) == 1])
    prob.solve(solver=cp.CLARABEL)
    return float(prob.value), np.asarray(w.value), np.asarray(v.value)


def _brute_force(B, A, m, beta, lam):
    n = B.shape[1]
    best = (np.inf, None)
    for S in itertools.combinations(range(n), m):
        C = [j for j in range(n) if j not in S]
        value = _joint(B, A, S, C, beta, lam)[0]
        if value < best[0]:
            best = (value, tuple(S))
    return best


def _support(B, A, seed):
    rng = np.random.default_rng(seed)
    n = B.shape[1]
    S = sorted(rng.choice(n, size=2, replace=False).tolist())
    C = [j for j in range(n) if j not in S]
    return S, C


#: (beta, lam) pairs inside the bound, including Jaume's lam = 0.1 and the edge.
PAIRS = ((1e-6, 0.1), (0.1, 0.1), (1.0, 0.1), (1.0, 0.5), (0.5, 1.0),
         (1.0, (np.sqrt(5.0) - 1.0) / 2.0))


# --------------------------------------------------------------------- smoke
def test_marex_fits_with_a_control_penalty():
    df, T0 = _panel()
    res = _fit(df, T0, "exact", m_eq=2, design="weakly_targeted", beta=1.0,
               control_penalty=0.1)
    assert len(_treated(res)) == 2
    assert res.globres.control_weights_agg.sum() == pytest.approx(1.0)
    assert np.isfinite(res.globres.objective)


# -------------------------------------------------------------- the oracle
@pytest.mark.parametrize("beta,lam", PAIRS)
def test_the_coupled_fit_matches_a_conic_solve(beta, lam):
    B, A = _problem(3)
    S, C = _support(B, A, 3)
    value, w, v = coupled_fit(B, A, S, C, beta, control_penalty=lam)
    ref = _joint(B, A, S, C, beta, lam)[0]
    assert value == pytest.approx(ref, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("beta,lam", PAIRS)
def test_the_value_reported_is_the_objective_of_the_weights(beta, lam):
    B, A = _problem(4)
    S, C = _support(B, A, 4)
    value, w, v = coupled_fit(B, A, S, C, beta, control_penalty=lam)
    assert value == pytest.approx(_objective(B, A, S, C, w, v, beta, lam),
                                  rel=1e-10, abs=1e-12)
    assert w.min() >= 0 and v.min() >= 0
    assert w.sum() == pytest.approx(1.0) and v.sum() == pytest.approx(1.0)


def test_zero_penalty_is_the_weakly_targeted_fit():
    B, A = _problem(5)
    S, C = _support(B, A, 5)
    a = coupled_fit(B, A, S, C, 1.0)
    b = coupled_fit(B, A, S, C, 1.0, control_penalty=0.0)
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])


def test_the_certificate_brackets_the_optimum():
    B, A = _problem(6)
    S, C = _support(B, A, 6)
    ref = _joint(B, A, S, C, 1.0, 0.5)[0]
    value, w, v, info = coupled_fit(B, A, S, C, 1.0, control_penalty=0.5,
                                    return_info=True)
    assert value >= ref - 1e-7 * (1.0 + ref)
    assert value <= ref + 1e-7 * (1.0 + ref)
    assert info["iterations"] <= 50


@pytest.mark.parametrize("beta", (0.01, 1.0, 100.0))
def test_the_penalised_fit_certifies_in_a_few_newton_steps(beta):
    """Measured over 40 random problems per setting, from beta = 0.01 to 100
    and up to 0.99 of the bound: at most four Newton steps and no sweeps."""
    lam = 0.9 * _inside(beta, 1.0)
    B, A = _problem(5, n=9)
    S, C = [0, 1], list(range(2, 9))
    value, w, v, info = coupled_fit(B, A, S, C, beta, control_penalty=lam,
                                    return_info=True)
    assert info["iterations"] <= 5
    assert info["sweeps"] == 0
    assert value == pytest.approx(_joint(B, A, S, C, beta, lam)[0], rel=1e-7)


def test_the_fallback_sweep_carries_the_penalty():
    """On this problem no Newton step improves and one sweep closes the gap;
    the sweep's pull on the treated fit is beta (1 + lam), not beta."""
    from mlsynth.tests.test_marex_exact_weakly_targeted import _fallback_problem
    B, A, S, C = _fallback_problem(12)
    value, w, v, info = coupled_fit(B, A, S, C, 1e-6, control_penalty=0.1,
                                    return_info=True)
    assert info["sweeps"] >= 1
    assert info["iterations"] <= 2
    assert value == pytest.approx(_joint(B, A, S, C, 1e-6, 0.1)[0],
                                  rel=1e-9, abs=1e-12)


def test_the_cutoff_discards_a_candidate_that_cannot_win():
    B, A = _problem(7)
    S, C = _support(B, A, 7)
    value = coupled_fit(B, A, S, C, 1.0, control_penalty=0.3)[0]
    assert coupled_fit(B, A, S, C, 1.0, control_penalty=0.3,
                       cutoff=0.5 * value) is None
    assert coupled_fit(B, A, S, C, 1.0, control_penalty=0.3,
                       cutoff=2.0 * value)[0] == pytest.approx(value)


# -------------------------------------------------------------- the search
@pytest.mark.parametrize("beta,lam", ((1.0, 0.1), (0.5, 1.0), (1e-6, 0.1)))
def test_the_search_matches_brute_force(beta, lam):
    B, A = _problem(8, n=7)
    designs = search_cluster(B, A, sizes=[2], filt=SubsetFilter(), beta=beta,
                             control_penalty=lam)
    value, S = _brute_force(B, A, 2, beta, lam)
    assert designs[0].treated == S
    assert designs[0].value == pytest.approx(value, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("lam", (0.1, 0.6))
def test_exact_matches_the_mip(lam):
    df, T0 = _panel()
    _assert_same_design(*_both(df, T0, m_eq=2, design="weakly_targeted",
                               beta=1.0, control_penalty=lam))


def test_exact_matches_the_mip_with_two_clusters():
    df, T0 = _panel(J=10, seed=5, region_sizes=(5, 5))
    _assert_same_design(*_both(df, T0, m_eq=2, cluster="region",
                               design="weakly_targeted", beta=1.0,
                               control_penalty=0.2))


def test_the_relaxed_path_takes_the_penalty():
    df, T0 = _panel()
    res = _fit(df, T0, "mip", m_eq=2, design="weakly_targeted", beta=1.0,
               control_penalty=0.1, relaxed=True)
    assert res.globres.control_weights_agg.sum() == pytest.approx(1.0)


def test_the_penalty_changes_the_design_somewhere():
    """The parameter reaches the solve: at a large penalty the controls move."""
    moved = False
    for seed in range(6):
        df, T0 = _panel(seed=seed)
        a = _fit(df, T0, "exact", m_eq=2, design="weakly_targeted", beta=1.0)
        b = _fit(df, T0, "exact", m_eq=2, design="weakly_targeted", beta=1.0,
                 control_penalty=0.6)
        if not np.allclose(a.globres.control_weights_agg,
                           b.globres.control_weights_agg, atol=1e-4):
            moved = True
            break
    assert moved


def test_zero_penalty_leaves_every_design_unchanged():
    df, T0 = _panel(seed=2)
    for backend in ("exact", "mip"):
        a = _fit(df, T0, backend, m_eq=2, design="weakly_targeted", beta=1.0)
        b = _fit(df, T0, backend, m_eq=2, design="weakly_targeted", beta=1.0,
                 control_penalty=0.0)
        assert a.globres.objective == b.globres.objective
        np.testing.assert_array_equal(a.globres.control_weights_agg,
                                      b.globres.control_weights_agg)


# ---------------------------------------------------------- the MIQP objective
def test_the_miqp_objective_is_program_one_on_the_simplex():
    """The sum-of-squares form CVXPY receives equals the objective as written
    at feasible points, so the MIQP optimises Vives-i-Bastida's program."""
    rng = np.random.default_rng(9)
    N, T = 7, 6
    X = rng.normal(size=(N, T)) + 3.0
    members = [np.arange(N)]
    Xbar = [X.mean(axis=0)]
    w, v, z = init_cvxpy_variables(N, 1, boolean=False)
    obj = build_objective(X, Xbar, members, w, v, z, "weakly_targeted",
                          beta=0.8, control_penalty=0.4)
    for _ in range(5):
        S = sorted(rng.choice(N, 2, replace=False).tolist())
        C = [j for j in range(N) if j not in S]
        ws, vs = rng.dirichlet(np.ones(2)), rng.dirichlet(np.ones(len(C)))
        wf, vf = np.zeros((N, 1)), np.zeros((N, 1))
        wf[S, 0], vf[C, 0] = ws, vs
        w.value, v.value = wf, vf
        want = _objective(X.T, Xbar[0], S, C, ws, vs, 0.8, 0.4)
        assert obj.value == pytest.approx(want, rel=1e-10)


# --------------------------------------------------------------------- edge
def test_the_bound_itself_is_accepted_and_solved():
    lam = (np.sqrt(5.0) - 1.0) / 2.0          # beta lam (1 + lam) = 1 at beta 1
    MAREXConfig(**_cfg(*_panel(), "exact", m_eq=2, design="weakly_targeted",
                       beta=1.0, control_penalty=lam))
    B, A = _problem(10)
    S, C = _support(B, A, 10)
    value = coupled_fit(B, A, S, C, 1.0, control_penalty=lam)[0]
    assert value == pytest.approx(_joint(B, A, S, C, 1.0, lam)[0], rel=1e-6)


def test_a_single_control_market():
    B, A = _problem(11, n=4)
    value, w, v = coupled_fit(B, A, [0, 1, 2], [3], 1.0, control_penalty=0.5)
    assert v.tolist() == [1.0]
    assert value == pytest.approx(_joint(B, A, [0, 1, 2], [3], 1.0, 0.5)[0],
                                  rel=1e-6)


def test_tiny_beta_is_the_treated_fit():
    B, A = _problem(12)
    S, C = _support(B, A, 12)
    value = coupled_fit(B, A, S, C, 1e-9, control_penalty=0.5)[0]
    assert value == pytest.approx(_joint(B, A, S, C, 0.0, 0.0)[0], rel=1e-6)


# ------------------------------------------------------------------ failure
def test_a_negative_penalty_is_refused():
    with pytest.raises(MlsynthConfigError, match="control_penalty"):
        MAREXConfig(**_cfg(*_panel(), "exact", m_eq=2,
                           design="weakly_targeted", control_penalty=-0.1))


@pytest.mark.parametrize("design", ("standard", "penalized", "unit_penalized"))
def test_the_penalty_belongs_to_the_weakly_targeted_design(design):
    with pytest.raises(MlsynthConfigError, match="weakly_targeted"):
        MAREXConfig(**_cfg(*_panel(), "mip", m_eq=2, design=design,
                           control_penalty=0.1))


@pytest.mark.parametrize("backend", ("exact", "mip"))
def test_a_penalty_past_the_convexity_bound_is_refused(backend):
    with pytest.raises(MlsynthConfigError, match=r"beta \* control_penalty"):
        MAREXConfig(**_cfg(*_panel(), backend, m_eq=2,
                           design="weakly_targeted", beta=1.0,
                           control_penalty=0.7))


def test_the_helpers_refuse_what_the_config_refuses():
    Y = np.zeros((4, 6))
    with pytest.raises(ValueError, match="control_penalty"):
        validate_scm_inputs(Y, 4, 0, "standard", control_penalty=0.1)
    with pytest.raises(ValueError, match="convex"):
        validate_scm_inputs(Y, 4, 0, "weakly_targeted", beta=4.0,
                            control_penalty=1.0)
    with pytest.raises(ValueError, match="must be >= 0"):
        validate_scm_inputs(Y, 4, 0, "weakly_targeted", beta=1.0,
                            control_penalty=-0.1)
    B, A = _problem(13)
    with pytest.raises(MlsynthConfigError, match="convex"):
        coupled_fit(B, A, [0, 1], [2, 3, 4], 4.0, control_penalty=1.0)
    # Between -1 and 0 the bound reads negative and passes; the sign is
    # checked on its own (found by the mutant a-negative-penalty-is-fitted).
    with pytest.raises(MlsynthConfigError, match="control_penalty must be >= 0"):
        coupled_fit(B, A, [0, 1], [2, 3, 4], 1.0, control_penalty=-0.1)
    with pytest.raises(MlsynthConfigError, match="control_penalty must be >= 0"):
        search_cluster(B, A, sizes=[2], filt=SubsetFilter(), beta=1.0,
                       control_penalty=-0.1)
    with pytest.raises(MlsynthConfigError, match="control_penalty"):
        search_cluster(B, A, sizes=[2], filt=SubsetFilter(),
                       control_penalty=0.1)


# ------------------------------------------------------------------ property
_beta = st.floats(1e-4, 4.0)
_frac = st.floats(0.0, 1.0)


def _inside(beta, frac):
    """A penalty at ``frac`` of the way to the bound for this ``beta``."""
    top = (-1.0 + np.sqrt(1.0 + 4.0 / beta)) / 2.0
    return frac * top


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 10_000), _beta, _frac)
def test_the_oracle_is_the_joint_optimum(seed, beta, frac):
    lam = _inside(beta, frac)
    B, A = _problem(seed, n=6, T=8)
    S, C = _support(B, A, seed)
    value, w, v = coupled_fit(B, A, S, C, beta, control_penalty=lam)
    ref = _joint(B, A, S, C, beta, lam)[0]
    assert value == pytest.approx(ref, rel=1e-5, abs=1e-8)
    assert value == pytest.approx(_objective(B, A, S, C, w, v, beta, lam),
                                  rel=1e-9, abs=1e-12)


@settings(max_examples=15, deadline=None)
@given(st.integers(0, 10_000), st.floats(0.05, 2.0))
def test_more_penalty_never_raises_the_control_spread(seed, beta):
    """Comparative statics of a penalised optimum: raising the weight on the
    penalty can only lower the penalty at the optimum (for a fixed treated
    set, both optima are over the same feasible set)."""
    B, A = _problem(seed, n=6, T=8)
    S, C = _support(B, A, seed)
    top = _inside(beta, 1.0)
    spreads = []
    for lam in (0.0, 0.3 * top, 0.9 * top):
        _, w, v = coupled_fit(B, A, S, C, beta, control_penalty=lam)
        t = B[:, S] @ w
        spreads.append(sum(v[j] * float((t - B[:, C[j]]) @ (t - B[:, C[j]]))
                           for j in range(len(C))))
    assert spreads[1] <= spreads[0] * (1 + 1e-6) + 1e-9
    assert spreads[2] <= spreads[1] * (1 + 1e-6) + 1e-9
