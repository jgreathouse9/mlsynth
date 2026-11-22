"""PANGEO draws each supergeo pair's treatment by a fair coin.

The supergeo design (Chen, Doudchenko, Jiang, Stein and Ying, "Supergeo design:
generalized matching for geographic experiments", arXiv:2301.12044, section 3)
chooses the pairs and then, independently in each pair, assigns one half to
treatment with probability one half. That draw is what the design-based
inference in ``effects.py`` rests on (the sign-flip test enumerates the 2^K
assignments it could have made). These tests pin that PANGEO makes the draw:

* the coins: one per pair, fair, independent across pairs and arms, reproducible
  from ``assignment_seed`` (``_coin_flips``);
* the design: the pairs do not depend on the seed, each pair's treated half is
  one of its two halves, the flag on the pair says which, and the means, the
  assignment map and the treated set follow the labels;
* the forwarding: the fixed-Q, automatic-Q and fast paths and the analysis run
  (``post_col``) all draw from the seed;
* the configuration: the seed is a non-negative integer, anything else raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import PANGEO
from mlsynth.config_models import PANGEOConfig
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.pangeo_helpers import prepare_pangeo_inputs, run_pangeo
from mlsynth.utils.pangeo_helpers.pipeline import _coin_flips


def _long(Y: np.ndarray, arms=None) -> pd.DataFrame:
    """A long panel from a units x periods matrix; arms per unit (default one arm)."""
    n, T = Y.shape
    arms = ["A"] * n if arms is None else list(arms)
    return pd.DataFrame({
        "unit": np.repeat([f"u{i:02d}" for i in range(n)], T),
        "time": np.tile(np.arange(T), n),
        "sales": Y.ravel(),
        "arm": np.repeat(arms, T),
    })


def _panel(seed: int, n: int, T: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 + rng.normal(0, 5, (n, 1)) + rng.normal(0, 1, (n, T)).cumsum(axis=1)


def _design(Y, seed=0, arms=None, **kw):
    inp = prepare_pangeo_inputs(_long(Y, arms), "sales", "arm", "unit", "time")
    kw.setdefault("max_supergeo_size", 1)
    kw.setdefault("compute_power", False)
    return inp, run_pangeo(inp, assignment_seed=seed, **kw)


def _pairs(res, arm="A"):
    """The design's pairs as unordered sets of units, in a canonical order."""
    return sorted(tuple(sorted(p.treatment + p.control)) for p in res.arm_designs[arm].pairs)


def _labels(res, arm="A"):
    """{pair (canonical): treated half (sorted)}."""
    return {tuple(sorted(p.treatment + p.control)): tuple(sorted(p.treatment))
            for p in res.arm_designs[arm].pairs}


# --- the coins ---------------------------------------------------------------------------------------------------

def test_one_coin_per_pair_per_arm():
    """Unit: _coin_flips returns, per arm, one boolean per pair."""
    out = _coin_flips([3, 0, 5], seed=7)
    assert [len(a) for a in out] == [3, 0, 5]
    assert all(a.dtype == bool for a in out)


def test_the_coins_are_reproducible_from_the_seed():
    """Unit: the same seed gives the same coins; the draw uses nothing else."""
    a, b = _coin_flips([6, 4], seed=11), _coin_flips([6, 4], seed=11)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_the_coins_are_fair_and_independent():
    """Unit, over 2000 seeds: each of six pairs lands heads about half the time, two pairs land heads together
    about a quarter of the time (independent, not one coin for all), and the two arms of a design are not dealt
    the same sequence (one stream across arms, not a fresh one per arm)."""
    heads = np.array([np.concatenate(_coin_flips([6, 6], seed=s)) for s in range(2000)])
    assert np.all(np.abs(heads.mean(axis=0) - 0.5) < 0.05)
    assert abs((heads[:, 0] & heads[:, 1]).mean() - 0.25) < 0.04
    assert (heads[:, :6] == heads[:, 6:]).all(axis=1).mean() < 0.05


@settings(max_examples=200, deadline=None)
@given(sizes=st.lists(st.integers(0, 8), max_size=4), seed=st.integers(0, 2 ** 63 - 1))
def test_the_coins_over_the_domain(sizes, seed):
    """Property, over arm sizes (empty arms included) and seeds up to 2^63: one boolean per pair, the same on a
    second call."""
    out = _coin_flips(sizes, seed=seed)
    assert [len(a) for a in out] == sizes
    assert all(np.array_equal(x, y) for x, y in zip(out, _coin_flips(sizes, seed=seed)))


# --- the design --------------------------------------------------------------------------------------------------

def test_smoke_the_result_records_the_seed_and_each_pairs_coin():
    """Smoke: a default PANGEO fit records assignment_seed 0 and a coin on every pair."""
    res = PANGEO({"df": _long(_panel(0, 8)), "outcome": "sales", "arm": "arm", "unitid": "unit",
                  "time": "time", "max_supergeo_size": 1, "display_graphs": False,
                  "compute_power": False}).fit()
    assert res.assignment_seed == 0
    assert all(isinstance(p.first_half_treated, bool) for p in res.arm_designs["A"].pairs)


def test_the_pairs_do_not_depend_on_the_seed_and_the_labels_do():
    """Unit: on 12 units (6 pairs), seeds 0..9 give the same pairs, and at least two seeds label them differently."""
    Y = _panel(1, 12)
    runs = [_design(Y, seed=s)[1] for s in range(10)]
    assert all(_pairs(r) == _pairs(runs[0]) for r in runs)
    assert len({tuple(sorted(_labels(r).items())) for r in runs}) > 1


def test_the_labels_follow_the_coins_in_canonical_order():
    """Unit: pairs taken in order of their first unit (the lowest index), the k-th coin says whether the half
    holding that unit is treated, and the flag on the pair records it."""
    Y = _panel(2, 10)
    inp, res = _design(Y, seed=5)
    coins = _coin_flips([5], seed=5)[0]
    pos = {u: i for i, u in enumerate(inp.unit_names)}
    pairs = sorted(res.arm_designs["A"].pairs, key=lambda p: min(pos[u] for u in p.treatment + p.control))
    for coin, p in zip(coins, pairs):
        first = min(p.treatment + p.control, key=pos.get)
        assert p.first_half_treated == bool(coin) == (first in p.treatment)


def test_the_means_the_map_and_the_treated_set_follow_the_labels():
    """Unit: each pair's treatment_mean is the mean pre trajectory of its treated units, the assignment map and
    the treated set agree with the pairs, under a seed that flips some pairs."""
    Y = _panel(3, 12)
    inp, res = _design(Y, seed=3)
    pos = {u: i for i, u in enumerate(inp.unit_names)}
    assert any(not p.first_half_treated for p in res.arm_designs["A"].pairs)
    for p in res.arm_designs["A"].pairs:
        np.testing.assert_allclose(p.treatment_mean, inp.Y[[pos[u] for u in p.treatment]].mean(axis=0))
        np.testing.assert_allclose(p.control_mean, inp.Y[[pos[u] for u in p.control]].mean(axis=0))
    treated = {u for p in res.arm_designs["A"].pairs for u in p.treatment}
    assert set(res.arm_designs["A"].treatment_units) == treated
    assert {u for u, a in res.assignment.items() if a == "treatment"} == treated


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(panel=st.integers(0, 10 ** 6), half=st.integers(2, 6), q=st.sampled_from([1, 2]),
       seed=st.integers(0, 2 ** 63 - 1))
def test_the_design_over_the_domain(panel, half, q, seed):
    """Property, over panels of 4 to 12 units, Q in {1, 2} and seeds up to 2^63: the pairs equal seed 0's, each
    pair's halves partition it, and the flag agrees with which half holds the pair's first unit."""
    Y = _panel(panel, 2 * half)
    inp, res = _design(Y, seed=seed, max_supergeo_size=q, min_pairs=1)
    _, ref = _design(Y, seed=0, max_supergeo_size=q, min_pairs=1)
    assert _pairs(res) == _pairs(ref)
    pos = {u: i for i, u in enumerate(inp.unit_names)}
    for p in res.arm_designs["A"].pairs:
        assert not set(p.treatment) & set(p.control) and p.treatment and p.control
        assert p.first_half_treated == (min(p.treatment + p.control, key=pos.get) in p.treatment)


def test_a_single_pair_takes_both_labellings_across_seeds():
    """Edge: two units, one pair; across seeds 0..19 each unit is treated at least once."""
    Y = _panel(4, 2)
    treated = {_design(Y, seed=s)[1].arm_designs["A"].pairs[0].treatment[0] for s in range(20)}
    assert treated == {"u00", "u01"}


def test_two_arms_draw_from_one_stream():
    """Unit: with two arms of three pairs each, the coins are _coin_flips([3, 3], seed), arm by arm in the
    order of the panel's arms."""
    Y = _panel(5, 12)
    inp, res = _design(Y, seed=9, arms=["A"] * 6 + ["B"] * 6)
    want = _coin_flips([3, 3], seed=9)
    pos = {u: i for i, u in enumerate(inp.unit_names)}
    for arm, coins in zip(["A", "B"], want):
        pairs = sorted(res.arm_designs[arm].pairs, key=lambda p: min(pos[u] for u in p.treatment + p.control))
        assert [p.first_half_treated for p in pairs] == [bool(c) for c in coins]


# --- the forwarding ----------------------------------------------------------------------------------------------

def test_the_fast_path_draws_the_coins():
    """Unit: the fast partition's pairs are labelled by the same coins."""
    Y = _panel(6, 12)
    inp, res = _design(Y, seed=4, fast=True, max_supergeo_size=2)
    pos = {u: i for i, u in enumerate(inp.unit_names)}
    pairs = sorted(res.arm_designs["A"].pairs, key=lambda p: min(pos[u] for u in p.treatment + p.control))
    assert [p.first_half_treated for p in pairs] == [bool(c) for c in _coin_flips([len(pairs)], seed=4)[0]]


def test_automatic_q_draws_the_coins_for_the_design_it_returns():
    """Unit: with max_supergeo_size None, the returned design is labelled as the fixed-Q design at the chosen
    Q with the same seed, and the result records the seed."""
    Y = _panel(7, 12, T=16)
    _, auto = _design(Y, seed=13, max_supergeo_size=None, compute_power=True)
    _, fixed = _design(Y, seed=13, max_supergeo_size=auto.max_supergeo_size, compute_power=True)
    assert _labels(auto) == _labels(fixed) and auto.assignment_seed == 13


def test_the_analysis_run_reproduces_the_design_runs_labels():
    """Unit: a fit with post_col (the analysis after the experiment) labels the pairs exactly as the design-only
    fit on the pre rows with the same seed, which is what lets the realized ATT use the labels the experiment ran."""
    Y = _panel(8, 10, T=16)
    df = _long(Y)
    df["post"] = (df["time"] >= 12).astype(int)
    base = dict(outcome="sales", arm="arm", unitid="unit", time="time", max_supergeo_size=1,
                display_graphs=False, compute_power=False, assignment_seed=21)
    design = PANGEO(dict(base, df=df[df.post == 0].drop(columns="post"))).fit()
    analysis = PANGEO(dict(base, df=df, post_col="post")).fit()
    assert _labels(design) == _labels(analysis) and analysis.effects is not None


# --- the configuration -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [-1, 1.5, "0", True, None, 2 ** 64])
def test_a_seed_that_is_not_a_non_negative_integer_is_rejected(bad):
    """Failure: assignment_seed must be an integer from 0 to 2^63 - 1; a negative, fractional, string, boolean,
    missing or too large seed raises MlsynthConfigError naming it."""
    with pytest.raises(MlsynthConfigError, match="assignment_seed"):
        PANGEO({"df": _long(_panel(0, 4)), "outcome": "sales", "arm": "arm", "unitid": "unit", "time": "time",
                "assignment_seed": bad, "display_graphs": False})


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_run_pangeo_rejects_a_bad_seed(bad):
    """Failure: the pipeline checks the seed too, for callers that bypass the config."""
    inp = prepare_pangeo_inputs(_long(_panel(0, 4)), "sales", "arm", "unit", "time")
    with pytest.raises(MlsynthConfigError, match="assignment_seed"):
        run_pangeo(inp, max_supergeo_size=1, compute_power=False, assignment_seed=bad)


def test_the_config_default_is_zero():
    """Unit: the default seed is 0, fixed, so an analysis run without the argument reproduces a design run
    without it."""
    cfg = PANGEOConfig(df=_long(_panel(0, 4)), outcome="sales", arm="arm", unitid="unit", time="time")
    assert cfg.assignment_seed == 0


def test_the_estimator_passes_its_seed_to_the_design():
    """Unit: PANGEO with assignment_seed s labels the pairs as run_pangeo with s, for a seed whose labels differ
    from the default's."""
    Y = _panel(9, 12)
    base = {"df": _long(Y), "outcome": "sales", "arm": "arm", "unitid": "unit", "time": "time",
            "max_supergeo_size": 1, "fast": False, "display_graphs": False, "compute_power": False}
    _, want = _design(Y, seed=17)
    _, default = _design(Y, seed=0)
    assert _labels(want) != _labels(default)
    got = PANGEO(dict(base, assignment_seed=17)).fit()
    assert _labels(got) == _labels(want) and got.assignment_seed == 17


def test_the_q_sweep_designs_use_the_seed():
    """Unit: the diagnostic Q sweep of a fixed-Q fit (compute_q_sweep) designs each Q with the fit's seed, so its
    row for the fitted Q reports the fitted design's own mean program MDE (the MDE depends on the labels through
    the augmented DiD fit of treated on control)."""
    from mlsynth.utils.pangeo_helpers.pipeline import _mean_program_mde
    Y = _panel(10, 12, T=16)
    _, res = _design(Y, seed=29, compute_power=True, compute_q_sweep=True)
    row = next(r for r in res.metadata["q_sweep"] if r["q"] == 1)
    assert row["mean_program_mde_pct"] == pytest.approx(_mean_program_mde(res), rel=1e-12)
    _, other = _design(Y, seed=0, compute_power=True)
    assert _mean_program_mde(other) != pytest.approx(_mean_program_mde(res), rel=1e-12)
