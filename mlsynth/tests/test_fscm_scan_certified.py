"""The forward scan's winner, carried as a certified solution.

The scan reaches the solver twice over. Every candidate goes through
``solve_simplex_qp``, the primitive, because a candidate that loses needs a
score and nothing else. The winner at each step goes through
``solve_weights``, which returns the same minimiser with the certificate,
the uniqueness verdict and the identification check attached.

The split is a cost decision and it is measured: the layer adds a fixed
0.2 ms per call for the certificate and the face null space, which is 6.1x the
primitive at two donors and 1.0x at thirty-eight. Paying it per candidate would
add about 0.15 s to a 0.57 s Proposition 99 fit for diagnostics no losing
candidate uses. Paying it per step is ``J`` calls against ``J^2/2``.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.solvers.active_set import solve_simplex_qp
from mlsynth.utils.fscm_helpers.estimation import scan_candidates
from mlsynth.utils.weights import WeightSolution, solve_weights


@pytest.fixture(scope="module")
def panel():
    import pandas as pd
    from pathlib import Path
    from mlsynth.utils.datautils import dataprep
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "basedata" / "P99data.csv")
    df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)
    p = dataprep(df, "state", "year", "cigsale", "treat")
    T0 = int(p["pre_periods"])
    return (np.asarray(p["donor_matrix"], float)[:T0],
            np.asarray(p["y"], float).ravel()[:T0])


# --------------------------------------------------------------------------
# The two entry points agree
# --------------------------------------------------------------------------
def test_the_scan_and_the_layer_reach_the_same_minimiser(panel):
    """Selection and reporting must not disagree about the same program."""
    X, y = panel
    j, score, w = scan_candidates(X, y, [], list(range(X.shape[1])))
    sol = solve_weights(X[:, [j]], y)
    assert np.asarray(w).ravel() == pytest.approx(np.array(sol.weights), abs=1e-12)


@pytest.mark.parametrize("step", [0, 1, 2, 3])
def test_they_agree_at_every_depth_the_order_is_determined(panel, step):
    X, y = panel
    sel, rem = [], list(range(X.shape[1]))
    for _ in range(step + 1):
        j, _, w = scan_candidates(X, y, sel, rem)
        sel.append(j); rem.remove(j)
    assert np.asarray(w).ravel() == pytest.approx(
        np.array(solve_weights(X[:, sel], y).weights), abs=1e-10)


# --------------------------------------------------------------------------
# The winner comes back certified
# --------------------------------------------------------------------------
def test_the_winner_carries_its_certificate(panel):
    X, y = panel
    j, score, w, sol = scan_candidates(
        X, y, [], list(range(X.shape[1])), certify=True)
    assert isinstance(sol, WeightSolution)
    assert sol.status == "optimal" and sol.kkt_residual < 1e-9
    assert np.array(sol.weights) == pytest.approx(np.asarray(w).ravel(), abs=1e-12)


def test_certifying_does_not_change_the_winner(panel):
    X, y = panel
    sel, rem = [], list(range(X.shape[1]))
    for _ in range(4):
        plain = scan_candidates(X, y, sel, rem)
        certified = scan_candidates(X, y, sel, rem, certify=True)
        assert plain[0] == certified[0]
        assert plain[1] == pytest.approx(certified[1])
        assert np.asarray(plain[2]).ravel() == pytest.approx(
            np.asarray(certified[2]).ravel(), abs=0.0)
        sel.append(plain[0]); rem.remove(plain[0])


def test_the_default_return_is_unchanged(panel):
    """Three values without `certify`, so existing callers are untouched."""
    X, y = panel
    out = scan_candidates(X, y, [], list(range(X.shape[1])))
    assert len(out) == 3


# --------------------------------------------------------------------------
# It costs one extra solve per step, not one per candidate
# --------------------------------------------------------------------------
def test_certifying_costs_one_solve_for_the_step_not_one_per_candidate(panel, monkeypatch):
    import mlsynth.utils.fscm_helpers.estimation as estimation

    calls = {"primitive": 0, "layer": 0}
    real_p, real_l = estimation.solve_simplex_qp, estimation.solve_weights
    monkeypatch.setattr(estimation, "solve_simplex_qp",
                        lambda *a, **k: (calls.__setitem__("primitive", calls["primitive"] + 1),
                                         real_p(*a, **k))[1])
    monkeypatch.setattr(estimation, "solve_weights",
                        lambda *a, **k: (calls.__setitem__("layer", calls["layer"] + 1),
                                         real_l(*a, **k))[1])
    X, y = panel
    J = X.shape[1]
    scan_candidates(X, y, [], list(range(J)), certify=True)
    assert calls["primitive"] == J          # one per candidate
    assert calls["layer"] == 1              # one for the winner


# --------------------------------------------------------------------------
# What the certificate is for
# --------------------------------------------------------------------------
def test_a_duplicated_donor_shows_up_on_the_winner(panel):
    """Selection can hand back a set whose weights are a continuum. The scan
    scores the fit, which is identical across that continuum, so only the
    certified winner can say so."""
    X, y = panel
    lead = int(np.argmax(solve_weights(X, y).weights))
    Xd = np.column_stack([X, X[:, lead]])
    sel = [lead]
    _, _, _, sol = scan_candidates(
        Xd, y, sel, [Xd.shape[1] - 1], certify=True)
    assert sol.unique is False


def test_the_certificate_is_taken_on_the_winner_and_not_the_first_candidate(panel):
    """A certificate computed on the wrong candidate has to be visible in the
    weights, and at the first step it is not: every candidate set is one donor,
    which on the simplex carries weight 1.0 whichever donor it is. The trap only
    arms once the selected set is non-trivial, so this runs at depth three and
    hands the candidates in an order whose first entry loses."""
    X, y = panel
    J = X.shape[1]
    sel, rem = [], list(range(J))
    for _ in range(3):
        j, _, _ = scan_candidates(X, y, sel, rem)
        sel.append(j); rem.remove(j)

    winner, _, _ = scan_candidates(X, y, sel, rem)
    order = [j for j in rem if j != winner]
    order.append(winner)                                  # a loser goes first
    assert order[0] != winner                             # the trap is armed

    j, _, w, sol = scan_candidates(X, y, sel, order, certify=True)
    assert j == winner
    assert len(sel) + 1 > 1                               # weights can differ
    assert np.array(sol.weights) == pytest.approx(np.asarray(w).ravel(), abs=1e-12)
    # and the wrong set really would have given different weights
    wrong = np.array(solve_weights(X[:, sel + [order[0]]], y).weights)
    assert np.abs(wrong - np.asarray(w).ravel()).max() > 1e-6


def test_past_saturation_the_first_candidate_evaluated_wins(panel):
    """Once the fit saturates every remaining candidate scores identically --
    on Proposition 99, 32 of them to the last digit. The step is then decided
    by the improvement test, and a strict one keeps the first evaluated. A
    non-strict one would hand it to the last, silently reordering the tail of
    `selection_path` while every score stayed optimal."""
    X, y = panel
    sel, rem = [], list(range(X.shape[1]))
    for _ in range(7):                                     # into the tied region
        j, score, _ = scan_candidates(X, y, sel, rem)
        sel.append(j); rem.remove(j)

    scores = {}
    for j in rem:
        idx = sel + [j]
        w = solve_simplex_qp(X[:, idx], y)
        scores[j] = float(np.sum((y - X[:, idx] @ w) ** 2))
    best = min(scores.values())
    tied = sorted(j for j, s in scores.items() if s - best <= 1e-12)
    assert len(tied) > 5                                   # genuinely saturated

    picked, _, _ = scan_candidates(X, y, sel, rem)
    assert picked == tied[0] if rem[0] in tied else picked in tied
    # order the candidates so a non-strict test would pick a different one
    reordered = [tied[0]] + [j for j in rem if j != tied[0]]
    assert scan_candidates(X, y, sel, reordered)[0] == tied[0]
