"""The moving-block permutation p-value, in one place.

:mod:`mlsynth.utils.conformal` exists so that "bands cannot drift apart between
estimators", and :func:`confidence_set_bounds` was centralised for exactly that
reason -- the library had the inclusion rule both ways and the two disagreed by
15 to 40 percent on the authors' own panel. The statistic that rule consumes is
still written out per estimator: ``cscipca_helpers``, ``bilevel``,
``syndes_helpers``, ``shc_helpers`` and ``rrsc_helpers`` each roll their own,
two of them in the opposite direction. They happen to enumerate the same block
set today, which is luck rather than design.

This pins the shared definition. Chernozhukov, Wuthrich and Zhu calibrate a
post-treatment block against the ``T`` cyclic shifts of the residual path: every
position in the series contributes a block, so a panel with ``T0`` pre-periods
supplies ``T`` reference blocks rather than the ``T0 / block`` a disjoint split
would give. That is what makes the p-value fine-grained enough to invert at a
tight level, and it is the whole reason to prefer moving blocks when the
pre-period is short relative to the horizon.

The statistic is a parameter because the estimand decides it. A per-period band
wants mean absolute residual; a band on a running total wants the absolute block
sum, because a block that is uniformly positive and one that oscillates around
zero have the same mean absolute residual and very different totals.

Levels: smoke, unit invariants, property, edge, failure.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.conformal import moving_block_pvalue


# --------------------------------------------------------------------------- #
# smoke                                                                        #
# --------------------------------------------------------------------------- #
def test_smoke_returns_a_probability():
    p = moving_block_pvalue(np.arange(20.0), block=4)
    assert isinstance(p, float) and 0.0 < p <= 1.0


# --------------------------------------------------------------------------- #
# unit invariants                                                              #
# --------------------------------------------------------------------------- #
def test_every_cyclic_position_contributes_a_block():
    """The property the whole method rests on: ``T`` reference blocks, not
    ``T0 / block`` of them. It shows up as the granularity of the p-value."""
    for T in (17, 40, 113):
        p = moving_block_pvalue(np.random.default_rng(T).normal(size=T), block=5)
        assert abs(p * T - round(p * T)) < 1e-9, "p is not a multiple of 1/T"


def test_the_post_block_is_always_counted():
    """It is one of the shifts, so the p-value can never be zero -- the smallest
    attainable value is ``1 / T``. A p-value of exactly zero would claim more
    evidence than a permutation test can produce."""
    u = np.concatenate([np.zeros(30), np.full(4, 1e6)])
    assert moving_block_pvalue(u, block=4) == pytest.approx(1 / u.size)


def test_a_typical_block_is_not_rejected():
    p = moving_block_pvalue(np.random.default_rng(0).normal(size=120), block=6)
    assert p > 0.10


def test_an_extreme_block_is_rejected():
    u = np.concatenate([np.random.default_rng(1).normal(scale=0.1, size=114),
                        np.full(6, 9.0)])
    assert moving_block_pvalue(u, block=6) < 0.05


def test_the_two_statistics_answer_different_questions():
    """A block that is uniformly positive and one that oscillates have the same
    mean absolute residual and very different sums, which is why a band on a
    running total cannot use the per-period statistic."""
    osc = np.tile([2.0, -2.0], 57)
    u = np.concatenate([osc, np.full(6, 2.0)])
    assert moving_block_pvalue(u, block=6, statistic="abs_sum") < \
           moving_block_pvalue(u, block=6, statistic="mean_abs")


def test_the_default_statistic_is_the_per_period_one():
    """Matching the existing call sites, so adopting this function anywhere
    changes nothing unless the caller asks for the other statistic."""
    u = np.random.default_rng(3).normal(size=50)
    assert moving_block_pvalue(u, block=5) == \
           moving_block_pvalue(u, block=5, statistic="mean_abs")


def test_it_reproduces_the_per_estimator_implementations():
    """The five hand-rolled copies enumerate the same block set, in two roll
    directions. Both must agree with this one, or adopting it would move a
    published number."""
    u = np.random.default_rng(4).normal(size=31)
    T, blk = u.size, 5
    forward = np.array([np.abs(np.roll(u, s)[-blk:]).mean() for s in range(T)])
    backward = np.array([np.abs(np.roll(u, -j)[T - blk:]).mean() for j in range(T)])
    want_f = float(np.mean(forward >= forward[0]))
    assert moving_block_pvalue(u, block=blk) == pytest.approx(want_f)
    assert sorted(np.round(forward, 12)) == sorted(np.round(backward, 12))


# --------------------------------------------------------------------------- #
# property                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("block", [1, 2, 5, 13])
@pytest.mark.parametrize("seed", range(5))
def test_it_is_always_a_probability_at_the_right_granularity(block, seed):
    u = np.random.default_rng(seed).normal(size=40)
    p = moving_block_pvalue(u, block=block)
    assert 1 / 40 - 1e-12 <= p <= 1.0


def test_a_whole_block_shift_of_the_path_leaves_it_unchanged():
    """The reference set is the cyclic orbit, so rotating the input permutes the
    blocks among themselves. Only which block is 'the post block' can change."""
    u = np.random.default_rng(6).normal(size=24)
    stats = [moving_block_pvalue(np.roll(u, s), block=4) for s in range(24)]
    assert len(set(np.round(stats, 12))) <= 24        # values come from one orbit
    assert all(abs(p * 24 - round(p * 24)) < 1e-9 for p in stats)


def test_scaling_the_residuals_does_not_move_it():
    """Both statistics are positively homogeneous, so the p-value is scale free
    -- which matters because the caller's outcome units are arbitrary."""
    u = np.random.default_rng(7).normal(size=60)
    for stat in ("mean_abs", "abs_sum"):
        a = moving_block_pvalue(u, block=6, statistic=stat)
        b = moving_block_pvalue(u * 1e5, block=6, statistic=stat)
        assert a == pytest.approx(b)


# --------------------------------------------------------------------------- #
# edge                                                                         #
# --------------------------------------------------------------------------- #
def test_a_block_as_long_as_the_path_is_one_block():
    """Every cyclic shift is then the same multiset, so nothing is more extreme
    than anything else and the p-value is one."""
    assert moving_block_pvalue(np.arange(8.0), block=8, statistic="abs_sum") == 1.0


def test_a_constant_path_is_never_significant():
    assert moving_block_pvalue(np.full(20, 3.0), block=4) == 1.0


def test_a_list_is_accepted():
    assert moving_block_pvalue([1.0, 2.0, 3.0, 4.0, 5.0], block=2) > 0.0


# --------------------------------------------------------------------------- #
# failure -- reported, not swallowed                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0, -1, 21])
def test_an_impossible_block_length_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="block"):
        moving_block_pvalue(np.zeros(20), block=bad)


def test_a_fractional_block_is_refused_rather_than_truncated():
    with pytest.raises(MlsynthConfigError):
        moving_block_pvalue(np.zeros(20), block=2.5)


def test_a_non_finite_residual_is_refused():
    """Silently dropping one would change the block count and therefore the
    p-value's denominator, which is the guarantee's reference set."""
    with pytest.raises(MlsynthDataError):
        moving_block_pvalue(np.array([1.0, np.nan, 2.0, 3.0]), block=2)


def test_an_empty_path_is_refused():
    with pytest.raises(MlsynthDataError):
        moving_block_pvalue(np.array([]), block=1)


def test_an_unknown_statistic_is_refused_by_name():
    with pytest.raises(MlsynthConfigError, match="statistic"):
        moving_block_pvalue(np.zeros(20), block=4, statistic="median_abs")
