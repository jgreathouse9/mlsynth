"""The synthetic stand-in for the calibration study's panel.

The study was measured on a proprietary weekly geo panel that cannot be shipped,
which left every real-data arm unrunnable for anyone without it. The generator
under test replaces that panel with one drawn to behave the same way, so the arms
run anywhere and the real panel becomes an option instead of a prerequisite.

What the tests hold it to is behaviour, not values: the panel has to be dominated
by a single factor, carry mild idiosyncratic persistence, keep a small positive
correlation inside a block, and be reproducible from its seed. Nothing here pins
a number that came from any real panel.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest

from benchmarks.studies.cumulative_calibration import synthetic_geo_panel as sgp


def _decompose(Y):
    """Market-demeaned panel, its leading factor share, and the residual after it."""
    Yc = Y - Y.mean(0)
    U, s, Vt = np.linalg.svd(Yc, full_matrices=False)
    share = s[0] ** 2 / (s ** 2).sum()
    return Yc, share, Yc - np.outer(U[:, 0] * s[0], Vt[0])


# --------------------------------------------------------------------------- smoke
def test_a_panel_comes_back_finite_and_the_right_shape():
    Y, block = sgp.make_panel(np.random.default_rng(0))
    assert Y.shape == (sgp.N_WEEKS, sgp.N_MARKETS)
    assert block.shape == (sgp.N_MARKETS,)
    assert np.isfinite(Y).all()


def test_the_blocks_partition_the_markets():
    _, block = sgp.make_panel(np.random.default_rng(0))
    counts = np.bincount(block, minlength=len(sgp.BLOCK_SIZES))
    assert counts.sum() == sgp.N_MARKETS
    assert (counts > 0).all()


# ------------------------------------------------------------------ unit invariants
def test_the_same_seed_gives_the_same_panel():
    a, _ = sgp.make_panel(np.random.default_rng(7))
    b, _ = sgp.make_panel(np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_different_seeds_give_different_panels():
    a, _ = sgp.make_panel(np.random.default_rng(7))
    b, _ = sgp.make_panel(np.random.default_rng(8))
    assert not np.allclose(a, b)


def test_one_factor_dominates_the_cross_section():
    """A national demand path is what a geo panel is mostly made of."""
    shares = [_decompose(sgp.make_panel(np.random.default_rng(s))[0])[1]
              for s in range(5)]
    assert np.mean(shares) > 0.85


def test_the_idiosyncratic_part_is_mildly_persistent():
    """Persistent enough that block structure matters, far from a unit root."""
    ars = []
    for s in range(5):
        Y, _ = sgp.make_panel(np.random.default_rng(s))
        _, _, E = _decompose(Y)
        ars.append(np.median([np.corrcoef(E[:-1, j], E[1:, j])[0, 1]
                              for j in range(Y.shape[1])]))
    assert 0.08 < np.mean(ars) < 0.32


def test_the_signal_to_noise_lands_where_it_is_set():
    Y, _ = sgp.make_panel(np.random.default_rng(1))
    Yc, _, E = _decompose(Y)
    snr = np.median((Yc.std(0) ** 2 - E.std(0) ** 2) ** 0.5 / E.std(0))
    assert 0.7 * sgp.SNR < snr < 1.4 * sgp.SNR


def test_markets_inside_a_block_move_together_more_than_across():
    Y, block = sgp.make_panel(np.random.default_rng(2))
    _, _, E = _decompose(Y)
    C = np.corrcoef(E.T)
    iu = np.triu_indices(Y.shape[1], 1)
    same = (block[:, None] == block[None, :])[iu]
    assert C[iu][same].mean() > C[iu][~same].mean()


def test_market_levels_are_spread_but_centred():
    """Log-levels carry the size dispersion; their centre is not a real magnitude."""
    Y, _ = sgp.make_panel(np.random.default_rng(3))
    means = Y.mean(0)
    assert 0.6 * sgp.LEVEL_SD < means.std() < 1.6 * sgp.LEVEL_SD
    assert abs(means.mean()) < 0.5


# ------------------------------------------------------------------------- edge
@pytest.mark.parametrize("n_markets,n_weeks", [(30, 60), (500, 300), (10, 20)])
def test_a_requested_size_is_honoured(n_markets, n_weeks):
    Y, block = sgp.make_panel(np.random.default_rng(0), n_markets, n_weeks)
    assert Y.shape == (n_weeks, n_markets)
    assert block.size == n_markets
    assert np.isfinite(Y).all()


def test_fewer_markets_than_blocks_still_partitions():
    Y, block = sgp.make_panel(np.random.default_rng(0), n_markets=4, n_weeks=30)
    assert Y.shape == (30, 4)
    assert np.bincount(block).sum() == 4


# ---------------------------------------------------------------------- failure
@pytest.mark.parametrize("n_markets", [0, -3])
def test_a_non_positive_market_count_is_refused(n_markets):
    with pytest.raises(ValueError, match="n_markets"):
        sgp.make_panel(np.random.default_rng(0), n_markets=n_markets, n_weeks=30)


@pytest.mark.parametrize("n_weeks", [0, 1, -5])
def test_a_horizon_shorter_than_two_weeks_is_refused(n_weeks):
    with pytest.raises(ValueError, match="n_weeks"):
        sgp.make_panel(np.random.default_rng(0), n_markets=20, n_weeks=n_weeks)


# ------------------------------------------------- a real, public size profile
def test_the_top_thirty_profile_is_a_descending_share_vector():
    p = sgp.TOP30_POPULATION_SHARE
    assert len(p) == 30
    assert all(a >= b for a, b in zip(p, p[1:]))
    assert all(x > 0 for x in p)


def test_a_size_profile_sets_the_market_levels():
    """Given a profile, levels are its logs -- not a random draw."""
    Y, _ = sgp.make_panel(np.random.default_rng(0),
                          size_profile=sgp.TOP30_POPULATION_SHARE, n_weeks=60)
    p = np.asarray(sgp.TOP30_POPULATION_SHARE, dtype=float)
    expected = np.log(p / p.mean())
    assert Y.shape == (60, 30)
    assert np.allclose(Y.mean(0), expected, atol=1e-9)


def test_a_size_profile_fixes_the_market_count():
    Y, block = sgp.make_panel(np.random.default_rng(0), size_profile=[3.0, 2.0, 1.0],
                              n_weeks=40)
    assert Y.shape == (40, 3)
    assert block.size == 3


def test_the_profile_ordering_is_preserved_in_the_levels():
    Y, _ = sgp.make_panel(np.random.default_rng(1),
                          size_profile=sgp.TOP30_POPULATION_SHARE, n_weeks=60)
    means = Y.mean(0)
    assert np.all(np.diff(means) < 0)


def test_a_profile_does_not_disturb_the_factor_structure():
    Y, _ = sgp.make_panel(np.random.default_rng(2),
                          size_profile=sgp.TOP30_POPULATION_SHARE, n_weeks=130)
    _, share, _ = _decompose(Y)
    assert share > 0.80


@pytest.mark.parametrize("bad", [[], [1.0, -2.0], [0.0, 1.0]])
def test_a_profile_that_is_empty_or_non_positive_is_refused(bad):
    with pytest.raises(ValueError, match="size_profile"):
        sgp.make_panel(np.random.default_rng(0), size_profile=bad, n_weeks=40)
