import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings

from conftest import model_and_path
from gppmta import (
    GPPParams, Path, attribute, channel_shares, direct_removal, hellinger, kl_divergence,
    path_shares, removal_probabilities, total_removal, total_removal_backprop,
    total_removal_thinning, touchpoint_scores,
)


def draw_removal(rng, c):
    R = rng.random(c) < 0.4
    if not R.any():
        R[rng.integers(c)] = True
    return R


@settings(max_examples=150, deadline=None)
@given(model_and_path())
def test_forward_solve_equals_algorithm2_backprop(case):
    params, path, rng = case
    c = len(path) - 1
    R = draw_removal(rng, c)
    assert total_removal(params, path, c, R) == pytest.approx(
        total_removal_backprop(params, path, c, R), rel=1e-10, abs=1e-12)


@settings(max_examples=150, deadline=None)
@given(model_and_path())
def test_score_invariants(case):
    """0 <= DRE <= TRE <= 1; DRE additive; TRE subadditive; shares sum to 1."""
    params, path, rng = case
    c = len(path) - 1
    R1 = draw_removal(rng, c)
    R2 = draw_removal(rng, c) & ~R1
    dre1, tre1 = direct_removal(params, path, c, R1), total_removal(params, path, c, R1)
    assert -1e-12 <= dre1 <= tre1 + 1e-12 and tre1 <= 1 + 1e-12
    P = removal_probabilities(params, path, c, R1)
    assert np.all(P >= -1e-12) and np.all(P <= 1 + 1e-12) and np.all(P[R1] == 1)
    if R2.any():
        both = R1 | R2
        assert direct_removal(params, path, c, both) == pytest.approx(
            dre1 + direct_removal(params, path, c, R2), abs=1e-12)
        assert total_removal(params, path, c, both) <= tre1 + total_removal(params, path, c, R2) + 1e-12
    lam, W = path_shares(params, path)
    assert params.mu[0] / lam[c] + W[:c, c].sum() == pytest.approx(1.0)


def test_algorithm1_thinning_converges_to_the_exact_value(rng):
    from conftest import random_params, random_path

    for _ in range(5):
        params = random_params(rng, 5, 3, density=0.9, scale=1.5)
        params = GPPParams(np.maximum(params.mu, 0.05), params.alpha, params.decay)
        path = random_path(rng, 5, 10)
        c = len(path) - 1
        R = draw_removal(rng, c)
        exact = total_removal(params, path, c, R)
        mc = total_removal_thinning(params, path, c, R, n_draws=20_000, seed=1)
        assert mc == pytest.approx(exact, abs=4e-3)


def line_graph(p):
    """Sec. 3.3, Fig. 7: p -> ... -> 3 -> 2 -> conversion, no baselines.
    Index layout: conversion 0, chain types 1..p-2 (customer), p-1 firm."""
    alpha = np.zeros((p, p - 1))
    for k in range(1, p):
        alpha[k, k - 1] = 0.5
    params = GPPParams(mu=np.zeros(p - 1), alpha=alpha, decay=1.0)
    types = np.r_[np.arange(p - 1, 0, -1), 0]
    return params, Path(np.arange(1.0, p + 1), types, p + 1.0)


def test_line_graph_extreme_case():
    params, path = line_graph(6)
    c = len(path) - 1
    scores = touchpoint_scores(params, path, c)
    # DRE: 100% to the last touch, 0 elsewhere. TRE: 100% to every touch.
    np.testing.assert_allclose(scores["dre"], [0, 0, 0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(scores["tre"], 1.0)
    every = np.ones(c, dtype=bool)
    assert direct_removal(params, path, c, every) == pytest.approx(1.0)
    assert total_removal(params, path, c, every) == pytest.approx(1.0)


def test_firm_initiated_events_are_never_thinned():
    """Removing a touch upstream of a firm event leaves the firm event alone."""
    # types: 0 conversion, 1 customer, 2 firm. Edge 1 -> 0 only, plus 1 -> (nothing firm).
    alpha = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])
    params = GPPParams(mu=np.array([0.1, 0.1]), alpha=alpha, decay=1.0)
    path = Path(np.array([1.0, 2.0, 3.0]), np.array([1, 2, 0]), 4.0)
    P = removal_probabilities(params, path, 2, [0])
    assert P[1] == 0  # the firm event survives removal of the customer event


def test_attribute_and_shares():
    params, path = line_graph(5)
    other = Path(np.array([0.5, 1.0]), np.array([1, 0]), 2.0)
    no_conv = Path(np.array([0.5]), np.array([4]), 2.0)
    table = attribute(params, [path, other, no_conv], {"a": [4, 3], "b": [2, 1]})
    assert list(table["path"]) == [0, 1]
    assert {"dre_a", "tre_a", "dre_b", "tre_b", "baseline"} <= set(table.columns)
    s = channel_shares(table, "tre")
    assert s.sum() == pytest.approx(1.0) and set(s.index) == {"a", "b"}
    with pytest.raises(ValueError):
        channel_shares(table, "shapley")
    with pytest.raises(ValueError):
        attribute(params, [path], {"bad": [0]})  # conversion is not a touchpoint


def test_zero_intensity_conversion_is_reported():
    params = GPPParams(mu=np.zeros(2), alpha=np.zeros((3, 2)), decay=1.0)
    path = Path(np.array([1.0, 2.0]), np.array([2, 0]), 3.0)
    with pytest.warns(RuntimeWarning, match="zero fitted intensity"):
        table = attribute(params, [path], {"x": [2]})
    assert np.isnan(table.loc[0, "baseline"])


def test_bad_removal_sets_and_conversions_raise():
    params, path = line_graph(4)
    with pytest.raises(ValueError):
        total_removal(params, path, 0, [0])  # event 0 is not a conversion
    with pytest.raises(ValueError):
        total_removal(params, path, 3, [3])  # the conversion itself
    with pytest.raises(ValueError):
        total_removal(params, path, 3, np.array([True, False]))


def test_divergences():
    p = np.array([0.38, 0.62])
    assert kl_divergence(p, p) == 0 and hellinger(p, p) == 0
    assert kl_divergence(p, [0.35, 0.65]) > 0 and 0 < hellinger(p, [0.35, 0.65]) < 1
