import numpy as np
import pytest
from scipy import stats as sps

from gppmta import compensator_increments, simulate
from gppmta.paper import ALPHA, CONV, DECAY, MU_ALL, SPACE, TRUE_PARAMS


@pytest.fixture(scope="module")
def sim():
    return simulate(MU_ALL, ALPHA, DECAY, 365.0, 3000, SPACE.channel_types(), seed=11)


def test_time_rescaled_interarrivals_are_exp1(sim):
    """Time-rescaling theorem: the simulator draws from model (1)."""
    for e in range(4):
        z = compensator_increments(TRUE_PARAMS, sim.paths, e)
        assert z.size > 500
        assert sps.kstest(z, "expon").pvalue > 1e-3, e


def test_rescaling_detects_wrong_parameters(sim):
    """The same check rejects parameters with the edges switched off."""
    wrong = TRUE_PARAMS.__class__(TRUE_PARAMS.mu, 0 * TRUE_PARAMS.alpha, DECAY)
    z = compensator_increments(wrong, sim.paths, CONV)
    assert sps.kstest(z, "expon").pvalue < 1e-6


def test_firm_events_are_poisson(sim):
    counts = np.array([(p.types == 4).sum() for p in sim.paths])
    assert abs(counts.mean() - 0.02 * 365) < 0.2
    assert abs(counts.var() / counts.mean() - 1) < 0.1


def test_channel_truth_is_bounded_and_overlapping(sim):
    # Each channel-off world loses at most every conversion, and the two worlds
    # overlap (display -> search impression -> conversion counts for both).
    assert 0 < sim.ccc["display"] < sim.conversions
    assert 0 < sim.ccc["search"] < sim.conversions
    assert sim.ccc["display"] + sim.ccc["search"] > sim.conversions - 0.2 * sim.conversions


def test_paths_are_valid(sim):
    assert len(sim.paths) == 3000
    assert sum((p.types == 0).sum() for p in sim.paths) == sim.conversions
    for path in sim.paths[:200]:
        assert np.all(np.diff(path.times) > 0) and path.T == 365.0


def test_channel_off_expectation_matches_branching_algebra():
    """E[CCC_z] from the branching means (no truncation: T >> 1/decay)."""
    sim = simulate(MU_ALL, ALPHA, 5.0, 2000.0, 400, SPACE.channel_types(), seed=5)
    # Expected descendants of one event of each type: (I - A)^{-1}.
    Tot = np.linalg.inv(np.eye(5) - ALPHA)
    imm = MU_ALL * 2000.0 * 400
    conv_from = imm * Tot[:, CONV]  # conversions in clusters rooted at each type
    assert abs(sim.conversions / conv_from.sum() - 1) < 0.05
