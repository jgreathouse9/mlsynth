"""MAREX's global test against the cyclic shifts of the contrast.

``global_test="subsets"``, the default, is Abadie and Zhao's eq. (17): the mean
absolute post contrast against the ``T_post``-subsets of the blank and post
periods. It is exact when those periods are exchangeable. Serially dependent
contrasts are not, and it then over-rejects (16.5% at 5% with factors
autocorrelated at 0.8, ``benchmarks/studies/marex_global_test_dependence``).

``global_test="cyclic"`` is Chernozhukov, Wuthrich and Zhu's moving-block test
with their Remark 1 statistic: the magnitude of the post block's signed sum
against the same statistic on every cyclic shift of the blank and post
contrast. The shifts keep neighbouring periods together, so the test is valid
under stationary, weakly dependent contrasts.

These tests pin the option's wiring and its invariants: the statistic is the
signed one, the reference holds the blank and post periods and nothing else,
only the global p-value changes, and every contrast MAREX reports (each
cluster, the aggregate, each treated unit and eq. (11)'s aggregate) is tested
the same way.

Levels: smoke, unit invariants, edge, failure.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import MAREX
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.conformal import moving_block_pvalue
from mlsynth.utils.marex_helpers.inference import GLOBAL_TESTS, compute_inference


def _panel(J=10, T=48, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 1, (T, 2))
    lam = rng.normal(0, 1, (J, 2))
    Y = lam @ F.T + 0.4 * rng.standard_normal((J, T))
    return pd.DataFrame([{"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t])}
                         for j in range(J) for t in range(T)])


# Tb + T_post = 26 periods in the reference, so the cyclic p-value can reach
# 1/26 and a 5% test can reject.
_CFG = dict(outcome="y", unitid="unit", time="time", T0=40, m_eq=2,
            inference=True, blank_periods=18, T_post=8, backend="exact")


def _fit(**kw):
    cfg = dict(df=_panel(), **_CFG)
    cfg.update(kw)
    return MAREX(cfg).fit()


def _direct(post, blank, fit=(0.0, 0.0, 0.0), **kw):
    """``compute_inference`` on a contrast given directly, the control at zero."""
    y_t = np.concatenate([np.asarray(fit, float), np.asarray(blank, float),
                          np.asarray(post, float)])
    te, tb = len(fit), len(blank)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_inference(y_t, np.zeros_like(y_t), te + tb, te, tb,
                                 random_state=0, **kw)


def _reference(inf):
    """The cyclic test computed by the shared helper on an inference's window."""
    window = np.concatenate([inf.placebo_effects, inf.treated_effects])
    return moving_block_pvalue(window, block=inf.treated_effects.size,
                               statistic="abs_sum")


_RNG = np.random.default_rng(3)
_BLANK = _RNG.normal(size=20)
_POST = _RNG.normal(size=6)


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #

def test_a_cyclic_fit_reports_a_p_value_in_the_unit_interval():
    p = _fit(global_test="cyclic").globres.inference.global_p_value
    assert 0.0 < p <= 1.0


def test_the_options_are_the_paper_test_and_the_cyclic_one():
    assert GLOBAL_TESTS == ("subsets", "cyclic")


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #

def test_the_default_is_the_paper_test():
    default = _fit().globres.inference
    paper = _fit(global_test="subsets").globres.inference
    assert default.global_p_value == paper.global_p_value
    assert _direct(_POST, _BLANK).global_p_value == \
        _direct(_POST, _BLANK, global_test="subsets").global_p_value


def test_the_cyclic_p_value_is_the_shared_moving_block_test():
    inf = _direct(_POST, _BLANK, global_test="cyclic")
    assert inf.global_p_value == _reference(inf)


def test_the_statistic_is_the_magnitude_of_the_mean_post_contrast():
    assert _direct(_POST, _BLANK, global_test="cyclic").s_obs == \
        pytest.approx(abs(_POST.mean()), rel=1e-12)
    assert _direct(_POST, _BLANK).s_obs == \
        pytest.approx(np.abs(_POST).mean(), rel=1e-12)


def test_signs_that_cancel_are_no_evidence_under_the_signed_statistic():
    # Large post contrasts that alternate in sign: eq. (16) reads their
    # magnitude, the signed statistic reads their sum, which is zero.
    blank = np.tile([0.3, -0.2, 0.1, -0.25], 6)
    post = np.array([5.0, -5.0] * 3)
    assert _direct(post, blank).global_p_value < 0.05
    assert _direct(post, blank, global_test="cyclic").global_p_value == 1.0


def test_a_constant_shift_reaches_the_smallest_p_value():
    inf = _direct(_POST + 50.0, _BLANK, global_test="cyclic")
    assert inf.global_p_value == pytest.approx(1 / (_BLANK.size + _POST.size))


def test_the_fitting_periods_stay_out_of_the_reference():
    quiet = _direct(_POST, _BLANK, fit=np.zeros(12), global_test="cyclic")
    loud = _direct(_POST, _BLANK, fit=np.full(12, 1e3), global_test="cyclic")
    assert quiet.global_p_value == loud.global_p_value


def test_the_cyclic_test_draws_nothing():
    a = _direct(_POST, _BLANK, global_test="cyclic", max_combinations=10)
    y = np.concatenate([np.zeros(3), _BLANK, _POST])
    b = compute_inference(y, np.zeros_like(y), 23, 3, 20, random_state=99,
                          global_test="cyclic")
    assert a.global_p_value == b.global_p_value


def test_only_the_global_p_value_and_its_statistic_change():
    s = _fit().globres.inference
    c = _fit(global_test="cyclic").globres.inference
    np.testing.assert_array_equal(s.per_period_pvals, c.per_period_pvals)
    np.testing.assert_array_equal(s.ci, c.ci)
    np.testing.assert_array_equal(s.treated_effects, c.treated_effects)
    np.testing.assert_array_equal(s.placebo_effects, c.placebo_effects)


def test_every_cluster_and_the_aggregate_take_the_cyclic_test():
    df = _panel()
    df["grp"] = np.where(df["unit"].str[1:].astype(int) < 5, "a", "b")
    res = MAREX(dict(df=df, cluster="grp", **_CFG, global_test="cyclic")).fit()
    infs = [c.inference for c in res.clusters.values()] + [res.globres.inference]
    assert len(infs) == 3
    for inf in infs:
        assert inf.global_p_value == _reference(inf)


def test_every_treated_unit_and_eq11_take_the_cyclic_test():
    level = _fit(global_test="cyclic", unit_inference=True).globres.unit_level
    infs = [u.inference for u in level.units.values()] + [level.eq11]
    assert len(infs) == 3
    for inf in infs:
        assert inf.global_p_value == _reference(inf)


def test_the_joint_test_over_units_is_the_same_under_either_option():
    a = _fit(unit_inference=True).globres.unit_level.joint_p_value
    b = _fit(global_test="cyclic", unit_inference=True).globres.unit_level
    assert a == b.joint_p_value


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #

def test_with_no_blank_period_the_cyclic_p_value_is_one():
    # Every shift's block is then the whole window, so every shift ties with
    # the observed one; summed in another order they can differ by rounding,
    # which would break the tie (it does for this window).
    post = np.array([0.1, 0.2, 0.3])
    assert moving_block_pvalue(post, block=3, statistic="abs_sum") < 1.0
    assert _direct(post, [], global_test="cyclic").global_p_value == 1.0


def test_a_single_post_period_is_ranked_among_the_blank_periods():
    inf = _direct([4.0], np.arange(-9.0, 10.0) / 10, global_test="cyclic")
    assert inf.global_p_value == pytest.approx(1 / 20)


def test_no_post_period_gives_no_p_value():
    y = np.concatenate([np.zeros(3), _BLANK])
    inf = compute_inference(y, np.zeros_like(y), y.size, 3, 20,
                            global_test="cyclic")
    assert np.isnan(inf.global_p_value)


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #

def test_an_unknown_test_is_refused_by_name():
    with pytest.raises(MlsynthConfigError,
                       match="global_test must be.*subsets.*cyclic"):
        _fit(global_test="blocks")


@pytest.mark.parametrize("bad", [None, 1, True, ["cyclic"]])
def test_a_non_string_test_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="global_test must be"):
        _fit(global_test=bad)


def test_compute_inference_refuses_an_unknown_test():
    with pytest.raises(MlsynthConfigError, match="global_test must be"):
        _direct(_POST, _BLANK, global_test="windows")
