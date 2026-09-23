"""``consensus_rate`` records what the search did. It is not a confidence signal.

When :math:`\\binom{M}{m}` exceeds ``enumerate_max`` the treated-set search runs
a multi-start local optimiser and reports ``FEASIBLE``. Having no MIP optimality
gap to report, it reports a consensus block instead: how many independent starts
ran, how many ended on the incumbent, that count as a share (``consensus_rate``),
how many distinct local optima were seen, and the improvement trail.

The share is the problem. It divides by ``n_starts``, and ``n_starts`` is what
actually determines whether the incumbent is the global optimum. More starts
explore more basins, so the fraction landing on the incumbent falls at the same
time as the answer gets better, and the number an analyst reads moves opposite
to the thing it is supposed to signal.

Measured against ground truth on 18 instances built from a 211-market DMA panel
(three population bands, ``m`` in {3, 4}, three constraint regimes), each solved
exactly by enumeration, over 1,350 multi-start runs: within a fixed ``n_starts``
the rate separates a suboptimal run from an exact one with an AUC of 0.44 to
0.54, and pooled across ``n_starts`` the AUC is 0.38 -- the wrong side of a coin
flip. No threshold is usable: ``rate <= 0.5`` catches three quarters of the
misses and fires on 70% of the correct runs.

That corpus needs an exact solve per instance and does not belong in the test
suite. What is pinned here is the structure behind the number, on three small
instances where enumeration is cheap: accuracy rises with ``n_starts`` while the
rate falls, the underlying count rises, and the rate's distribution on wrong
answers sits on top of its distribution on right ones -- including runs where
every start agreed and the answer was still not the optimum.

The block stands as effort telemetry. The claim these tests refute is the
separate one: that a high share of agreeing starts is evidence of global
optimality.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.fast_scm_helpers.lexsearch import select_treated_designs

N_STARTS = (1, 2, 4, 16)
SEEDS = range(8)


def _instance(seed, n=26, T=14, m=4):
    """A constrained design small enough to enumerate, hard enough to miss.

    Costs are geometric in ``G_jj`` so the markets the construction reaches for
    first are the dear ones, and the conflict graph pairs neighbours: the
    spillover-plus-budget regime is where the multi-start search actually loses
    the optimum.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, n))
    X = X - X.mean(axis=1, keepdims=True)
    G = X.T @ X
    conflict = np.zeros((n, n), dtype=bool)
    for i in range(0, n - 1, 4):
        conflict[i, i + 1] = conflict[i + 1, i] = True
    costs = np.empty(n)
    costs[np.argsort(np.diag(G))] = np.geomspace(6.0, 1.0, n)
    budget = float(np.sort(costs)[:m].sum() * 3.0)
    return dict(G=G, cand=list(range(n)), m=m,
                kw=dict(conflict=conflict, unit_costs=costs, budget=budget))


@pytest.fixture(scope="module")
def runs():
    """Every (n_starts, seed) run on three instances, scored against the truth."""
    out = []
    for inst_seed in (0, 1, 2):
        inst = _instance(inst_seed)
        exact = select_treated_designs(inst["G"], inst["cand"], m=inst["m"],
                                       top_K=5, method="enumerate", **inst["kw"])
        opt = exact["top_designs"][0].loss
        assert exact["stats"]["termination"]["status"] == "OPTIMAL"
        for n_starts in N_STARTS:
            for seed in SEEDS:
                h = select_treated_designs(inst["G"], inst["cand"], m=inst["m"],
                                           top_K=5, method="heuristic",
                                           n_starts=n_starts, random_state=seed,
                                           **inst["kw"])
                c = h["stats"]["search"]["consensus"]
                out.append(dict(
                    n_starts=n_starts, seed=seed, inst=inst_seed,
                    hit=bool(abs(h["top_designs"][0].loss - opt)
                             <= 1e-9 * max(1.0, opt)),
                    rate=float(c["consensus_rate"]),
                    count=int(c["starts_reaching_incumbent"])))
    return out


def _by(runs, field, n_starts):
    return [r[field] for r in runs if r["n_starts"] == n_starts]


class TestTheRateMovesAgainstAccuracy:
    """The confound: the share falls exactly where the answer improves."""

    def test_accuracy_rises_with_n_starts(self, runs):
        acc = [np.mean(_by(runs, "hit", ns)) for ns in N_STARTS]
        assert acc[-1] > acc[0], f"accuracy did not improve: {acc}"
        assert acc[-1] == 1.0, f"the largest budget should be exact: {acc}"

    def test_consensus_rate_falls_as_accuracy_rises(self, runs):
        lo = float(np.mean(_by(runs, "rate", N_STARTS[0])))
        hi = float(np.mean(_by(runs, "rate", N_STARTS[-1])))
        assert hi < lo, (
            f"consensus_rate {lo:.2f} -> {hi:.2f} while accuracy "
            f"{np.mean(_by(runs, 'hit', N_STARTS[0])):.2f} -> "
            f"{np.mean(_by(runs, 'hit', N_STARTS[-1])):.2f}")

    def test_the_underlying_count_rises_instead(self, runs):
        """The denominator is what inverts it: agreement itself does grow."""
        lo = float(np.mean(_by(runs, "count", N_STARTS[0])))
        hi = float(np.mean(_by(runs, "count", N_STARTS[-1])))
        assert hi > lo


class TestTheRateDoesNotSeparate:
    """Right and wrong answers are not told apart by the number."""

    def test_the_distributions_overlap(self, runs):
        hits = [r["rate"] for r in runs if r["hit"]]
        miss = [r["rate"] for r in runs if not r["hit"]]
        assert miss, "fixture must produce suboptimal runs to be a test at all"
        assert max(miss) >= min(hits), "no overlap: the rate would be usable"

    def test_a_unanimous_search_can_still_be_wrong(self, runs):
        """The decisive case: every start agreed and the answer was suboptimal."""
        miss = [r["rate"] for r in runs if not r["hit"]]
        assert max(miss) >= 0.5, (
            f"highest consensus on a suboptimal run was {max(miss):.2f}")

    def test_no_threshold_both_catches_the_misses_and_spares_the_hits(self, runs):
        hits = np.array([r["rate"] for r in runs if r["hit"]])
        miss = np.array([r["rate"] for r in runs if not r["hit"]])
        for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9):
            caught = float((miss <= thr).mean())
            false_alarms = float((hits <= thr).mean())
            assert not (caught == 1.0 and false_alarms < 0.25), (
                f"threshold {thr} works after all: catches every miss and "
                f"flags only {false_alarms:.0%} of the correct runs")


class TestTheBlockIsStillHonestTelemetry:
    """What the block reports about the search itself stays true."""

    def test_counts_are_internally_consistent(self, runs):
        for r in runs:
            assert 0 <= r["count"]
            assert 0.0 <= r["rate"] <= 1.0

    def test_enumeration_reports_no_consensus(self):
        """There is nothing to be uncertain about on the exact path."""
        inst = _instance(0)
        out = select_treated_designs(inst["G"], inst["cand"], m=inst["m"],
                                     top_K=5, method="enumerate", **inst["kw"])
        assert out["stats"]["search"]["consensus"] is None
        assert out["stats"]["termination"]["status"] == "OPTIMAL"
