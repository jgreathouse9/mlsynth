"""One implementation of the sunny screen, not two.

``bilevel/mscmt.py`` solved for a separating direction; ``solvers/sunny.py`` solves
Eq (9) for ``alpha``. They are the two sides of one linear program: writing ``H``
through its support function gives

    alpha*(j) = sup over c with <c,x_j> > 0 of  min_i <c,x_i> / <c,x_j>,

so ``alpha*(j) = 1`` holds exactly when some ``c`` has ``<c,x_j> > 0`` and
``<c,x_j> = min_i <c,x_i>`` -- the direction the hyperplane form searches for.

The old form is transcribed here as a reference rather than deleted outright, so the
agreement stays pinned after the implementation it checked is gone. Two conventions
have to survive the port and are asserted separately: the sign (the old form centred
treated-minus-donor, the shared module centres donor-minus-treated) and the raw
all-shady verdict that #670's exact-fit branch reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

from mlsynth.utils.bilevel import BilevelProblem, solve_bilevel
from mlsynth.utils.solvers.sunny import sunny_alphas, sunny_donors


def _hyperplane_reference(X1: np.ndarray, X0: np.ndarray, tol: float = 1e-9):
    """The implementation this consolidation removes, kept as a cross-check.

    ``max_a a . (X1 - X0_d)`` subject to ``a . (X0_i - X0_d) <= 0`` and
    ``-1 <= a <= 1``; sunny when the optimum is positive. Returns the raw verdict,
    with no all-shady guard.
    """
    K, J = X0.shape
    mask = np.ones(J, dtype=bool)
    bnds = [(-1.0, 1.0)] * K
    for d in range(J):
        gap = X1 - X0[:, d]
        if float(gap @ gap) <= tol:
            continue
        res = linprog(c=-gap, A_ub=(X0 - X0[:, d:d + 1]).T, b_ub=np.zeros(J),
                      bounds=bnds, method="highs")
        if res.success:
            mask[d] = (-res.fun) > tol
    return mask


def _basque_predictor_design():
    PRED = ["sec.agriculture", "sec.energy", "sec.industry", "sec.construction",
            "sec.services.venta", "sec.services.nonventa", "school.illit",
            "school.prim", "school.med", "school.higher", "popdens", "invest",
            "gdpcap"]
    d = pd.read_csv("basedata/basque_mscmt.csv")
    d = d[(d.regionname != "Spain (Espana)") & (d.year < 1975)]
    X = d.groupby("regionname")[PRED].mean()
    X = X.div(X.std(axis=0), axis=1)
    T = "Basque Country (Pais Vasco)"
    donors = [r for r in X.index if r != T]
    return X.loc[T].to_numpy(float), X.loc[donors].to_numpy(float).T


# --------------------------------------------------------------------------- #
# the two forms agree
# --------------------------------------------------------------------------- #
def test_the_shared_screen_matches_the_hyperplane_form_on_basque():
    X1, X0 = _basque_predictor_design()
    assert sunny_donors(X0, X1).tolist() == _hyperplane_reference(X1, X0).tolist()


def test_the_shared_screen_matches_the_hyperplane_form_across_designs():
    rng = np.random.default_rng(5)
    seen_shady = seen_all_shady = False
    for _ in range(40):
        K, J = int(rng.integers(2, 7)), int(rng.integers(3, 14))
        X0 = rng.normal(size=(K, J)) * 2.0
        X1 = (X0 @ rng.dirichlet(np.ones(J)) if rng.random() < 0.3
              else rng.normal(size=K) * 2.0)
        got, ref = sunny_donors(X0, X1), _hyperplane_reference(X1, X0)
        assert got.tolist() == ref.tolist(), (K, J, got, ref)
        seen_shady |= bool((~got).any())
        seen_all_shady |= not got.any()
    assert seen_shady, "the sweep never produced a shady donor"
    assert seen_all_shady, "the sweep never produced the all-shady case"


def test_the_shared_screen_never_prunes_a_donor_the_reference_calls_sunny():
    """The direction that matters: a wrong "shady" drops a donor that can carry
    weight, a wrong "sunny" only keeps one."""
    rng = np.random.default_rng(11)
    unsafe = 0
    for _ in range(40):
        K, J = int(rng.integers(2, 8)), int(rng.integers(3, 16))
        X0 = rng.normal(size=(K, J)) * 2.0
        X1 = rng.normal(size=K) * 2.0
        unsafe += int((~sunny_donors(X0, X1) & _hyperplane_reference(X1, X0)).sum())
    assert unsafe == 0


# --------------------------------------------------------------------------- #
# the conventions that have to survive the port
# --------------------------------------------------------------------------- #
def test_the_classification_is_invariant_to_the_centring_sign():
    """The old form centred treated-minus-donor, the shared module the other way.
    Negating every column negates the hull, so the condition is preserved -- which is
    what makes the port a rename and not a change of meaning."""
    rng = np.random.default_rng(13)
    for _ in range(15):
        K, J = int(rng.integers(2, 7)), int(rng.integers(3, 12))
        Xt = rng.normal(size=(K, J)) * 2.0
        a = sunny_alphas(Xt, np.zeros(K))          # columns as given
        b = sunny_alphas(-Xt, np.zeros(K))         # every column negated
        np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-9)


def test_swapping_the_arguments_fails_loudly():
    """``_sunny_mask`` took the treated vector first and the shared screen takes the
    donor matrix first, so the port is exactly where an argument swap would hide. It
    must raise rather than return something plausible."""
    X1, X0 = _basque_predictor_design()
    with pytest.raises(ValueError, match="2-D"):
        sunny_donors(X1, X0)


def test_the_raw_all_shady_verdict_still_reaches_the_cascade():
    """#670's exact-fit branch fires on an all-False mask, so the shared screen must
    not soften it."""
    X0 = np.array([[0.0, 2.0, 1.0, 3.0], [0.0, 0.0, 2.0, 2.0]])
    X1 = X0 @ np.full(4, 0.25)
    assert not sunny_donors(X0, X1).any()
    assert not _hyperplane_reference(X1, X0).any()


# --------------------------------------------------------------------------- #
# the backend actually uses the shared screen
# --------------------------------------------------------------------------- #
def test_the_backend_calls_the_shared_screen(monkeypatch):
    """The consolidation itself. Without this the suite only pins that the two forms
    agree, which was already true while the backend kept its own copy.

    The patch target is the name ``mscmt`` looks up, not the one ``sunny`` defines --
    a module-level import binds the callable into ``mscmt``'s namespace, so patching
    ``sunny.sunny_donors`` would intercept nothing. The counter is asserted non-empty
    rather than assumed, for the same reason the cascade tests carry a control."""
    import mlsynth.utils.bilevel.mscmt as mod
    from mlsynth.utils.solvers.sunny import sunny_donors as real

    calls = []

    def _counted(B, A, **kw):
        calls.append((B.shape, A.shape))
        return real(B, A, **kw)

    monkeypatch.setattr(mod, "sunny_donors", _counted, raising=True)
    rng = np.random.default_rng(7)
    J, K, T = 12, 3, 16
    prob = BilevelProblem(y1_pre=rng.normal(size=T), Y0_pre=rng.normal(size=(T, J)),
                          X1=rng.normal(size=K) * 3.0, X0=rng.normal(size=(K, J)))
    solve_bilevel(prob, method="mscmt", seed=0, maxiter=60)
    assert len(calls) >= 1, "the backend did not go through the shared screen"
    # and it passed them the right way round: donor matrix first, treated vector second
    assert calls[0] == ((K, J), (K,)), calls[0]


# --------------------------------------------------------------------------- #
# the backend still behaves
# --------------------------------------------------------------------------- #
def test_the_cascade_branches_still_fire_through_the_shared_screen():
    X0 = np.array([[0.0, 2.0, 1.0, 3.0], [0.0, 0.0, 2.0, 2.0]])
    X1 = X0 @ np.full(4, 0.25)
    rng = np.random.default_rng(5)
    Y0 = rng.normal(size=(14, 4))
    y1 = Y0 @ np.array([0.0, 0.6, 0.4, 0.0]) + 0.02 * rng.normal(size=14)
    sol = solve_bilevel(BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0),
                        method="mscmt", seed=0, maxiter=40)
    assert sol.stage == "mscmt-exact-fit", sol.stage
    assert sol.metadata["n_sunny"] == 0


def test_pruning_still_reports_the_same_counts():
    rng = np.random.default_rng(7)
    J, K, T = 12, 3, 16
    prob = BilevelProblem(y1_pre=rng.normal(size=T), Y0_pre=rng.normal(size=(T, J)),
                          X1=rng.normal(size=K) * 3.0, X0=rng.normal(size=(K, J)))
    sol = solve_bilevel(prob, method="mscmt", seed=0, maxiter=60)
    if sol.stage == "mscmt":
        assert sol.metadata["n_sunny"] + sol.metadata["n_shady_pruned"] == J
        assert sol.metadata["n_sunny"] == int(sunny_donors(prob.X0, prob.X1).sum())
