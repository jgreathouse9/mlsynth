"""The block permutation reference, built as one gather instead of T rolls.

``_reference_stats`` supplies the conformal test's reference distribution. Under
the moving-block scheme that is the ``T`` cyclic shifts of the residual path,
each sliced to the post block and scored -- built, until now, by a Python loop
of ``np.roll`` over a length-``T`` array. Since ``np.roll(u, s)[pre:]`` is
``u[(arange(pre, T) - s) % T]``, the whole family is a single ``(T, H)`` fancy
index, and the loop disappears.

Test inversion calls this once per candidate effect, and a grid sweep runs it
tens of times per draw, so the loop was 62 percent of a sweep's cost.

What makes this delicate is the exponent. The statistic is
``(sum|x|^q / sqrt(len))^(1/q)``, and a fractional power evaluated on a
``(T, H)`` block does not always take the same numpy code path as one evaluated
on a length-``H`` row: measured, ``q = 1.5`` differs in the last bit. One bit is
not negligible here. A conformal p-value counts how many reference statistics
reach the observed one, so a value moved by an ULP can cross a tie and change a
rank -- the SPSC port had exactly this and the band came out 13 percent wide.

So the fast path is taken only at ``q == 1``, where the exponent is the identity
and equality is provable, not merely observed. That is the augsynth default and
every call site in the library. Anything else keeps the loop, and the tests pin
both halves: the values agree with the loop everywhere, and the work only drops
where the guard allows it.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest

from mlsynth.utils.bilevel.ridge_inference import _reference_stats, _stat


def _oracle(resids, post_slice, q):
    """The construction as it was written, transcribed as the test's reference.

    Kept here instead of imported so the tests still describe the intended
    behaviour once the implementation stops looking like this.
    """
    n = np.asarray(resids).shape[0]
    return np.array(
        [_stat(np.roll(resids, s)[post_slice], q) for s in range(n)], dtype=float)


def _roll_calls(fn):
    """How many times ``np.roll`` is called inside ``fn``.

    The loop calls it once per shift; the gather never calls it. Counting them
    is a machine-independent way to assert which path ran, where a wall-clock
    comparison on a shared runner is not.
    """
    real = np.roll
    seen = []

    def counting(*args, **kwargs):
        seen.append(1)
        return real(*args, **kwargs)

    np.roll = counting
    try:
        fn()
    finally:
        np.roll = real
    return len(seen)


def _series(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


_SHAPES = [(117, 104), (31, 18), (60, 40), (25, 12), (200, 150), (8, 4), (3, 1)]


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_it_returns_one_statistic_per_cyclic_shift():
    u = _series(31)
    out = _reference_stats(u, slice(18, None), 1.0, conformal_type="block",
                           ns=0, seed=0)
    assert out.shape == (31,)
    assert np.isfinite(out).all()
    assert (out >= 0.0).all()


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,pre", _SHAPES)
def test_it_matches_the_loop_exactly_at_the_default_exponent(n, pre):
    """Bit-identical, not close: a reference statistic moved by an ULP can cross
    a tie with the observed one and change the p-value's rank."""
    u = _series(n, seed=n)
    got = _reference_stats(u, slice(pre, None), 1.0, conformal_type="block",
                           ns=0, seed=0)
    assert np.array_equal(got, _oracle(u, slice(pre, None), 1.0))


@pytest.mark.parametrize("q", [0.5, 1.5, 2.0, 3.0])
def test_it_matches_the_loop_at_every_other_exponent_too(q):
    """The guard is about which path runs, never about what comes out."""
    u = _series(60, seed=7)
    got = _reference_stats(u, slice(40, None), q, conformal_type="block",
                           ns=0, seed=0)
    assert np.array_equal(got, _oracle(u, slice(40, None), q))


def test_the_first_entry_is_the_observed_statistic():
    """Shift zero is the identity, so the reference set always contains the
    observed value. That is why a block p-value can never fall below ``1 / T``."""
    u = _series(31, seed=3)
    out = _reference_stats(u, slice(18, None), 1.0, conformal_type="block",
                           ns=0, seed=0)
    assert out[0] == _stat(u[18:], 1.0)


def test_the_default_exponent_takes_the_gather_and_rolls_nothing():
    """The work assertion. Without it, a change that reverts to the loop leaves
    every value correct and only the speed gone."""
    u = _series(117, seed=1)
    calls = _roll_calls(lambda: _reference_stats(
        u, slice(104, None), 1.0, conformal_type="block", ns=0, seed=0))
    assert calls == 0


@pytest.mark.parametrize("q", [0.5, 1.5, 2.0])
def test_another_exponent_keeps_the_loop(q):
    """The other half of the guard: where equality is only observed and not
    provable, the original path runs and the cost is accepted."""
    u = _series(60, seed=2)
    calls = _roll_calls(lambda: _reference_stats(
        u, slice(40, None), q, conformal_type="block", ns=0, seed=0))
    assert calls == 60


def test_the_iid_scheme_is_untouched():
    """Its permutations come from a seeded generator in a fixed order, so the
    loop there is the draw sequence and not an inefficiency."""
    u = _series(40, seed=5)
    a = _reference_stats(u, slice(25, None), 1.0, conformal_type="iid",
                         ns=50, seed=11)
    b = _reference_stats(u, slice(25, None), 1.0, conformal_type="iid",
                         ns=50, seed=11)
    assert np.array_equal(a, b)
    assert a.shape == (50,)


def test_a_fancy_index_post_block_works_like_a_slice():
    """``post_slice`` is documented as a slice or a fancy index; the gather has
    to honour both, since it turns the slice into indices either way."""
    u = _series(31, seed=9)
    by_slice = _reference_stats(u, slice(18, None), 1.0, conformal_type="block",
                                ns=0, seed=0)
    by_index = _reference_stats(u, np.arange(18, 31), 1.0,
                                conformal_type="block", ns=0, seed=0)
    assert np.array_equal(by_slice, by_index)


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_a_single_period_post_block():
    u = _series(20, seed=4)
    got = _reference_stats(u, slice(19, None), 1.0, conformal_type="block",
                           ns=0, seed=0)
    assert np.array_equal(got, _oracle(u, slice(19, None), 1.0))


def test_a_post_block_covering_the_whole_path():
    """Every shift then scores the same multiset, so all T statistics agree and
    the p-value is exactly one -- the degenerate case a caller reaches by asking
    for a horizon as long as the series."""
    u = _series(12, seed=6)
    got = _reference_stats(u, slice(0, None), 1.0, conformal_type="block",
                           ns=0, seed=0)
    assert np.array_equal(got, _oracle(u, slice(0, None), 1.0))
    assert np.allclose(got, got[0])


def test_an_all_zero_path_gives_zero_everywhere():
    u = np.zeros(15)
    got = _reference_stats(u, slice(10, None), 1.0, conformal_type="block",
                           ns=0, seed=0)
    assert np.array_equal(got, np.zeros(15))


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["moving", "BLOCK", ""])
def test_an_unknown_scheme_is_still_refused(bad):
    u = _series(20)
    with pytest.raises(ValueError, match="conformal_type"):
        _reference_stats(u, slice(10, None), 1.0, conformal_type=bad,
                         ns=10, seed=0)
