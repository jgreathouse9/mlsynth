"""The dual maximisers return valid bounds, reach the lift's value, and stop at a target."""
import numpy as np
import pytest

from .brute import panel
from .dual import smoothed_lbfgs, spectral_bundle
from .lifts import Lift
from .oracle import Oracle


@pytest.fixture(scope="module")
def node():
    y, X = panel(4, p=8, n=40)
    L = Lift(y, X, "full")
    ref = L.piece(0.4, 0.9, 3) / L.scale ** 2
    return Oracle(y, X, 0.4, 0.9, 3), ref


def test_smoke_lbfgs_returns_a_bound_and_its_multipliers(node):
    O, ref = node
    th, yb, info = smoothed_lbfgs(O, np.zeros(O.dim))
    assert np.isfinite(th) and yb.shape == (O.dim,) and info["evals"] > 0


def test_lbfgs_reports_the_bound_its_multipliers_give(node):
    O, ref = node
    th, yb, _ = smoothed_lbfgs(O, np.zeros(O.dim))
    assert O.theta(yb)[0] == th
    assert np.all(yb[O.ineq] >= 0)


def test_lbfgs_reaches_the_lift_from_below(node):
    O, ref = node
    th, _, _ = smoothed_lbfgs(O, np.zeros(O.dim))
    assert th <= ref * (1 + 1e-6)
    assert (ref - th) / abs(ref) < 1e-3


def test_lbfgs_stops_once_the_target_is_cleared(node):
    O, ref = node
    target = ref * 0.9 if ref > 0 else ref * 1.1
    th, _, info = smoothed_lbfgs(O, np.zeros(O.dim), target=target)
    full = smoothed_lbfgs(O, np.zeros(O.dim))[2]
    assert th >= target and info["evals"] < full["evals"]


def test_the_stall_rule_hands_the_node_back(node):
    O, ref = node
    _, _, info = smoothed_lbfgs(O, np.zeros(O.dim), target=ref * 10 if ref > 0 else ref / 10,
                                handoff=lambda e, b, t: e >= 50)
    assert info["stopped"] == "handoff" and info["evals"] <= 51


def test_a_growing_stage_budget_is_honoured(node):
    O, ref = node
    seen = []
    smoothed_lbfgs(O, np.zeros(O.dim), stage_budget=lambda k: seen.append(k) or 20, alpha_min=1e-4)
    assert seen[:3] == [0, 1, 2]


def test_the_spectral_bundle_returns_a_valid_bound_below_the_lift(node):
    """The negative result: valid, and far from the lift within a small budget."""
    O, ref = node
    th, yb, info = spectral_bundle(O, np.zeros(O.dim), iters=15, gs_passes=20)
    assert th <= ref * (1 + 1e-6) and th == O.theta(yb)[0]


def test_the_bundle_stops_on_its_target_and_on_its_clock(node):
    O, ref = node
    low = O.theta(np.zeros(O.dim))[0] - 1.0
    assert spectral_bundle(O, np.zeros(O.dim), target=low)[2]["iters"] == 0
    assert spectral_bundle(O, np.zeros(O.dim), time_limit=0.0)[2]["iters"] == 0
