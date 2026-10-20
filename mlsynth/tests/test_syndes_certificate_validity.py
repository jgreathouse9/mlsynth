"""SYNDES's certificate reports a gap only when it is one (#804).

The certificate is ``lower_bound <= optimum <= incumbent``. It holds only if the
bound comes from a search on the data and penalty the incumbent is scored on,
and a bound above the incumbent is reported as the contradiction it is. Since
#822 the bound is the exact search's, so the designs here come from the exact
backends, and nothing needs SCIP.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.syndes_helpers.certificate import (SYNDESCertificateWarning,
                                                      _gap, design_objective,
                                                      syndes_certificate)
from mlsynth.utils.syndes_helpers.exact import solve_synthetic_design_exact
from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem
from mlsynth.utils.syndes_helpers.one_way_exact import solve_one_way_exact
from mlsynth.utils.syndes_helpers.per_unit_exact import solve_per_unit_exact

_SOLVE = {"global_2way": solve_synthetic_design_exact,
          "global_equal_weights": solve_one_way_exact,
          "per_unit": solve_per_unit_exact}


def solve_synthetic_design(Y, K, mode):
    return _SOLVE[mode](Y, K)


def _panel(N=8, T=14, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, 3)) @ rng.standard_normal((3, N))
            + 0.3 * rng.standard_normal((T, N)))


def _df(Y, n_post=4):
    T, N = Y.shape
    rows = [{"unit": j, "time": t, "y": float(Y[t, j]), "post": int(t >= T - n_post)}
            for j in range(N) for t in range(T)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["global_2way", "global_equal_weights"])
def test_a_certificate_brackets_the_design(mode):
    Y = _panel()
    d = solve_synthetic_design(Y, 2, mode=mode)
    inc = design_objective(Y, mode, d, d.lambda_value)
    cert = syndes_certificate(Y, 2, mode, inc, lam=d.lambda_value)
    assert cert.certified and cert.incumbent == inc
    assert cert.lower_bound <= inc * (1 + 1e-8) and 0.0 <= cert.optimality_gap < 1.0


# --------------------------------------------------------------------------- #
# the incumbent is the design scored on the bound's data
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["global_2way", "global_equal_weights", "per_unit"])
def test_the_design_objective_is_the_mode_objective_at_its_weights(mode):
    Y = _panel(seed=1)
    d = solve_synthetic_design(Y, 2, mode=mode)
    assert design_objective(Y, mode, d, d.lambda_value) == pytest.approx(
        d.objective_value, rel=1e-9)


def test_under_holdout_the_incumbent_is_scored_on_the_whole_pre_period():
    Y = _panel(N=8, T=24, seed=2)
    res = SYNDES({"df": _df(Y), "outcome": "y", "unitid": "unit", "time": "time",
                  "post_col": "post", "K": 2, "mode": "two_way_global",
                  "top_K": 3, "certify": True}).fit()
    d, cert = res.design, res.certificate
    Y_pre = res.inputs.Y_pre
    expected = design_objective(Y_pre, "global_2way", d, d.lambda_value)
    assert cert.incumbent == pytest.approx(expected, rel=1e-12)
    assert cert.lower_bound <= cert.incumbent and cert.optimality_gap >= 0.0
    # The bound uses the design's penalty, not one re-estimated on Y_pre.
    same = syndes_certificate(Y_pre, 2, "global_2way", expected, lam=d.lambda_value)
    assert cert.lower_bound == pytest.approx(same.lower_bound, rel=1e-12)


# --------------------------------------------------------------------------- #
# the gap
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("inc,lb,expected", [
    (2.0, 1.0, 0.5), (1.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 1.0),
    (1.0, 2.0, -0.5),
])
def test_the_gap_is_relative_to_the_larger_of_the_two(inc, lb, expected):
    assert _gap(inc, lb) == pytest.approx(expected)


def test_the_gap_does_not_depend_on_units():
    for s in (1e-12, 1e-6, 1.0, 1e6):
        assert _gap(2.0 * s, 1.5 * s) == pytest.approx(0.25, rel=1e-12)


def test_a_bound_above_the_incumbent_is_a_contradiction_not_a_zero_gap():
    Y = _panel()
    with pytest.warns(SYNDESCertificateWarning, match="exceeds the incumbent"):
        cert = syndes_certificate(Y, 2, "global_2way", 1e-6)
    assert not cert.certified and cert.optimality_gap < 0


def test_a_bound_above_the_incumbent_by_rounding_is_a_zero_gap():
    Y = _panel()
    lb = syndes_certificate(Y, 2, "global_2way", 1e6).lower_bound
    with warnings.catch_warnings():
        warnings.simplefilter("error", SYNDESCertificateWarning)
        cert = syndes_certificate(Y, 2, "global_2way", lb * (1 - 1e-12))
    assert cert.optimality_gap == 0.0 and cert.certified


# --------------------------------------------------------------------------- #
# K=None
# --------------------------------------------------------------------------- #


def test_a_two_way_design_of_free_size_is_bounded_over_every_size():
    """The free-size optimum may sit at any size, so its bound is the smallest
    over sizes, not the best size's."""
    Y = _panel(N=7)
    engine = BoundEngine.from_problem(TwoWayProblem.from_panel(Y, lam=None))
    expected = min(engine.global_lower_bound(k) for k in range(1, 7))
    cert = syndes_certificate(Y, None, "global_2way", 10.0)
    assert cert.lower_bound == pytest.approx(expected, rel=1e-12)
    d = solve_synthetic_design_exact(Y, None)
    assert d.raw_results["lower_bound"] == pytest.approx(expected, rel=1e-12)


def test_the_estimator_certifies_a_free_size_design():
    Y = _panel(N=6, T=18, seed=4)
    res = SYNDES({"df": _df(Y), "outcome": "y", "unitid": "unit", "time": "time",
                  "post_col": "post", "K": None, "mode": "two_way_global",
                  "certify": True}).fit()
    cert = res.certificate
    assert cert.lower_bound <= cert.incumbent and cert.certified


@pytest.mark.parametrize("mode", ["global_equal_weights", "per_unit"])
def test_a_free_size_is_refused_where_the_bound_needs_k(mode):
    with pytest.raises(MlsynthConfigError, match="K"):
        syndes_certificate(_panel(), None, mode, 1.0)


def _fake_sizes(monkeypatch, outcomes):
    """Replace the per-size search with fixed outcomes: size -> (value, lower,
    certified)."""
    import mlsynth.utils.syndes_helpers.exact as ex

    def fake(problem, K, filt, **kw):
        value, lower, certified = outcomes[K]
        N = problem.N
        a = np.zeros(N); a[:K] = 1.0 / K
        b = np.zeros(N); b[K:] = 1.0 / (N - K)
        return ex.SearchOutcome(treated=np.arange(K), value=value,
                                treated_weights=a, control_weights=b,
                                duality_gap=0.0, lower_bound=lower,
                                certified=certified)
    monkeypatch.setattr(ex, "_search", fake)


def test_a_free_size_bound_is_the_smallest_over_sizes_not_the_best_sizes(monkeypatch):
    """Size 2 holds the best design; size 3's bound is lower, so it is the bound."""
    _fake_sizes(monkeypatch, {1: (5.0, 4.0, True), 2: (1.0, 0.9, True),
                              3: (2.0, 0.5, True)})
    d = solve_synthetic_design_exact(_panel(N=4), None)
    assert int(d.assignment.sum()) == 2
    assert d.raw_results["lower_bound"] == 0.5


def test_a_free_size_is_certified_only_when_every_size_is(monkeypatch):
    _fake_sizes(monkeypatch, {1: (5.0, 4.0, True), 2: (1.0, 1.0, True),
                              3: (2.0, 1.5, False)})
    d = solve_synthetic_design_exact(_panel(N=4), None)
    assert d.raw_results["certified"] is False
    assert d.raw_results["status"] == "feasible"
