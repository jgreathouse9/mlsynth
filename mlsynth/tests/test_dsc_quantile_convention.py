"""Which empirical-quantile convention DSC implements, and why it is that one.

DSC matches quantile functions, so the rule for turning a finite cell sample
into a quantile function is not an implementation detail -- it is part of the
estimator. Two conventions are in play and they are not the same choice:

``type 1``
    The inverse of the empirical CDF: a step function returning actual order
    statistics. This is what Gunsilius (2023) §3.1 writes down, and what the
    Stata Journal paper's eq. (1) defines as ``F^-1(q) = inf{x : q <= F(x)}``.

``type 7``
    R's default, linear interpolation between order statistics.

So the papers define type 1. But *both* official implementations use type 7 --
the DiSCos R package passes ``qtype = 7``, and the Stata command's
``disco_quantile_sorted`` interpolates at ``alpha = (N-1)p + 1``. mlsynth
follows the implementations, because agreement with a shipped reference is
checkable and agreement with prose neither author implemented is not. That
choice is what lets ``test_dsc_stata_exact.py`` reproduce the Stata Journal's
published weights; using type 1 instead moves them by ~0.003.

This file pins the convention and, just as importantly, keeps the distinction
visible so it stays a recorded decision rather than folklore.

Gold from a live R 4.3.3 run; see ``benchmarks/reference/dsc_quantile/reference.R``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.dsc_helpers.quantiles import empirical_quantile

_GOLD = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
         / "dsc_quantile" / "gold_quantiles.csv")

#: The sample the gold was generated from. Small, odd-length and irregularly
#: spaced on purpose: on a large smooth sample the two types nearly coincide,
#: which would make every test below pass for the wrong reason.
SAMPLE = np.array([0.0, 0.5, 1.25, 2.0, 3.5, 3.5, 7.0, 11.25, 19.0, 42.0, 100.5])


def _gold():
    import csv
    with _GOLD.open() as fh:
        rows = list(csv.DictReader(fh))
    return (np.array([float(r["q"]) for r in rows]),
            np.array([float(r["type1"]) for r in rows]),
            np.array([float(r["type7"]) for r in rows]))


class TestTheTwoConventionsActuallyDiffer:
    """The premise the rest of the file rests on, asserted rather than assumed."""

    def test_the_gold_sample_separates_type_1_from_type_7(self):
        q, t1, t7 = _gold()
        assert np.abs(t1 - t7).max() > 25.0, (
            "the fixture no longer distinguishes the two quantile types, so the "
            "tests below would pass regardless of which one mlsynth implements")

    def test_they_agree_where_the_quantile_lands_on_an_order_statistic(self):
        """Not everywhere -- which is why a coarse check would miss a swap."""
        q, t1, t7 = _gold()
        assert np.isclose(t1[q == 0.5], t7[q == 0.5]).all()


class TestEmpiricalQuantileIsType7:
    """mlsynth follows the implementations, not the papers' prose."""

    def test_matches_r_type_7_exactly(self):
        q, _, t7 = _gold()
        np.testing.assert_allclose(empirical_quantile(SAMPLE, q), t7,
                                   rtol=0, atol=1e-12)

    def test_does_not_match_r_type_1(self):
        """Stated as behaviour, so a silent revert fails here.

        Type 1 is a defensible reading -- it is what both papers define -- so
        without this a change back would look like a correction rather than a
        break of the Stata cross-validation.
        """
        q, t1, _ = _gold()
        assert np.abs(empirical_quantile(SAMPLE, q) - t1).max() > 1.0

    def test_interpolates_between_order_statistics(self):
        """The defining property of type 7, independent of any gold file.

        Type 1 can only return values the sample contains; type 7 can return
        values between them, and does.
        """
        q = np.linspace(0.001, 0.999, 997)
        out = empirical_quantile(SAMPLE, q)
        assert not set(np.unique(out)) <= set(SAMPLE)
        assert out.min() >= SAMPLE.min() and out.max() <= SAMPLE.max()


class TestTheGridIsClosed:
    """``q = 0`` and ``q = 1`` are evaluated, not rejected.

    Both references include the endpoints -- DiSCos via
    ``seq(0, 1, length.out = G+1)``, Stata via ``disco_prob_grid``'s
    ``(j-1)/(N-1)`` -- so they fit the sample minimum and maximum.

    mlsynth used to raise on them. Restoring them was the single largest step
    toward reproducing the Stata reference: excluding the endpoints moves the
    fitted weights on the tenure panel by ~0.09, an order of magnitude more than
    the quantile type. Worth recording that an earlier test of this against the
    R package measured only ~0.003 and dismissed it -- that comparison was
    against a target carrying ~3 percent Monte Carlo noise, which swamped the
    effect.
    """

    def test_the_endpoints_give_the_sample_min_and_max(self):
        out = empirical_quantile(SAMPLE, np.array([0.0, 1.0]))
        np.testing.assert_allclose(out, np.array([SAMPLE.min(), SAMPLE.max()]),
                                   rtol=0, atol=1e-12)

    def test_the_reference_grid_is_accepted_whole(self):
        """The concrete consequence: the references' own grid now runs."""
        out = empirical_quantile(SAMPLE, np.linspace(0.0, 1.0, 1001))
        assert out.shape == (1001,) and np.all(np.diff(out) >= 0)

    @pytest.mark.parametrize("bad", [-1e-9, 1.0 + 1e-9, np.nan])
    def test_outside_the_closed_interval_still_raises(self, bad):
        """Widening to [0, 1] must not become 'anything goes'."""
        with pytest.raises(MlsynthEstimationError, match=r"\[0, 1\]"):
            empirical_quantile(SAMPLE, np.array([0.5, bad]))
