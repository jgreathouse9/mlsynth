"""Invariants of the FSCM forward scan, searched over generated panels.

The scan's defining invariant needs no reference solver and no tolerance
tuning, because it is a fact about the feasible sets rather than about any
particular numerics: the ``k``-donor simplex is the face of the
``(k+1)``-donor simplex where the new weight is zero, so admitting a donor can
only lower the optimum. A scan whose reported fit ever rises has not solved
something it claimed to solve.

That makes this the right shape for ``hypothesis`` rather than for a
hand-picked panel. The example tests in ``test_fscm_exact_scan`` pin the
Prop 99 incident this was found on; here the panel family is parameterised and
searched, so a solver that is accurate on the panels someone thought to write
down but not in general gets caught on one nobody thought of, and shrinks to
the simplest case that breaks it.

Two ingredients decide whether a panel can expose an inexact solve at all, and
both are drawn rather than fixed:

* **saturation** -- the treated unit is an exact convex combination of a few
  donors, so past that point the true optimum stops moving and no genuine
  improvement is left to mask solver error;
* **collinearity** -- donors pushed toward a common direction, which drives the
  conditioning up with ``k`` and slows a first-order method.

Both are drawn rather than assumed to be necessary, and that turned out to
matter: hunting by hand produced a violation only at ``collinearity = 0.999``,
while the search shrank straight past that to ``N=8, T0=5`` with no
collinearity and no noise at all -- a panel fitted exactly, where the reported
path still rose by 1.4e-08. The hand-built story about conditioning was not the
boundary; saturation alone was enough.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.fscm_helpers.estimation import scan_candidates

SETTINGS = settings(max_examples=60, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


def _panel(N, T0, r, collinearity, noise, seed):
    """A donor pool of tunable conditioning whose fit saturates at ``r`` donors."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T0, r))
    L = np.abs(rng.normal(size=(N, r)))
    X = F @ L.T
    X = collinearity * X.mean(axis=1, keepdims=True) + (1 - collinearity) * X
    X = X + noise * rng.normal(size=(T0, N))
    w = np.zeros(N)
    w[:r] = np.full(r, 1.0 / r)
    return X, X @ w


def _path(X, y, steps):
    sel, rem, path, weights, warm = [], list(range(X.shape[1])), [], [], None
    for _ in range(steps):
        j, rmspe, w = scan_candidates(X, y, sel, rem, warm=warm)
        warm = w
        sel.append(j)
        rem.remove(j)
        path.append(rmspe)
        weights.append(w)
    return sel, np.asarray(path), weights


PANEL = dict(
    N=st.integers(8, 30),
    T0=st.integers(5, 20),
    r=st.integers(1, 4),
    collinearity=st.floats(0.0, 0.999),
    noise=st.sampled_from([0.0, 1e-6, 1e-3, 1e-1]),
    seed=st.integers(0, 2 ** 16),
)


@SETTINGS
@given(**PANEL)
def test_the_reported_fit_never_rises_as_donors_are_admitted(
        N, T0, r, collinearity, noise, seed):
    """The invariant the Prop 99 incident violated, over the whole family."""
    X, y = _panel(N, T0, r, collinearity, noise, seed)
    steps = min(10, N - 1)
    _, path, _ = _path(X, y, steps)
    rise = float(np.diff(path).max()) if steps > 1 else 0.0
    assert rise <= 1e-9, f"path rose by {rise:.3e}: {np.round(path, 10)}"


@SETTINGS
@given(**PANEL)
def test_the_reported_fit_is_what_the_returned_weights_achieve(
        N, T0, r, collinearity, noise, seed):
    """The scan reports its RMSPE from the Gram form; recomputing it from the
    weights and the donor matrix must give the same number. A Gram assembled
    from the wrong rows would satisfy monotonicity and fail here."""
    X, y = _panel(N, T0, r, collinearity, noise, seed)
    cand = list(range(X.shape[1]))
    sel = cand[:min(3, len(cand) - 1)]
    rem = [j for j in cand if j not in sel]
    j, rmspe, w = scan_candidates(X, y, sel, rem)
    direct = float(np.sqrt(np.mean((y - X[:, sel + [j]] @ w) ** 2)))
    assert rmspe == pytest.approx(direct, abs=1e-8)


@SETTINGS
@given(**PANEL)
def test_every_step_returns_weights_on_the_simplex(
        N, T0, r, collinearity, noise, seed):
    X, y = _panel(N, T0, r, collinearity, noise, seed)
    _, _, weights = _path(X, y, min(6, N - 1))
    for w in weights:
        assert (w >= -1e-12).all()
        assert abs(float(w.sum()) - 1.0) <= 1e-9
