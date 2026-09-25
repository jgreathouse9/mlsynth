"""SNN's anchor search on irregular missingness.

SNN needs a fully observed cross around the target entry: anchor rows observed
at the target column, anchor columns observed in the target row, and the block
they span complete. Finding the largest such block is the maximum-biclique
problem, which the reference implementation solves exactly by enumerating
maximal cliques. mlsynth uses a dependency-free greedy search instead: strip
the worst row or column until the block is complete.

The rule for "worst" has to compare a row against a column, and the two counts
live on different denominators -- a row's missing count is out of the number of
columns, a column's out of the number of rows. Comparing the raw counts makes
the longer side always look worse, so on a mask with more rows than columns the
search strips every column and returns nothing while the rows sit untouched.
Measured on scattered MNAR masks, that lost the cross on 55 percent of missing
entries, where the exact search never failed. Comparing shares instead fixes
it: 0 percent lost, and the mean block min-dimension goes from 2.86 to 5.36
against the exact search's 5.94.

Block missingness never reached the bug. There the neighborhood submatrix is
already complete, so the search returns on the first check without stripping
anything, which is why every panel benchmark stayed green while the MNAR case
this estimator exists for did not work.

Layered per agents/agents_tests.md:

* smoke -- a cross is found on an irregular mask.
* unit invariants -- the returned block is always fully observed; feasibility
  does not depend on orientation; the panel fast path is exact.
* edge -- one observed row, one observed column, a 1x1 cross.
* failure -- genuinely impossible crosses report empty, and the regression that
  motivated the rule is pinned directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.snn_helpers.completion import _find_anchors


def _mnar_mask(m, n, r=3, p_lo=0.55, p_hi=0.95, seed=0):
    """Observation probability rising with the entry's value: scattered MNAR."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    z = (A - A.min()) / (A.max() - A.min())
    return (rng.random((m, n)) < p_lo + (p_hi - p_lo) * z).astype(int)


def _is_complete_cross(mask, i, j, AR, AC):
    """The contract: rows observed at ``j``, cols observed at ``i``, block full."""
    if AR.size == 0 or AC.size == 0:
        return True                      # vacuous; emptiness is checked elsewhere
    return (i not in AR and j not in AC
            and mask[AR, j].all() and mask[i, AC].all()
            and mask[np.ix_(AR, AC)].all())


# --------------------------------------------------------------------- smoke
def test_a_cross_is_found_on_an_irregular_mask():
    mask = _mnar_mask(24, 18, seed=0)
    found = 0
    for i, j in np.argwhere(mask == 0):
        AR, AC = _find_anchors(mask, int(i), int(j))
        found += AR.size > 0 and AC.size > 0
    assert found > 0


# ----------------------------------------------------------- unit invariants
def test_every_returned_block_is_fully_observed():
    """The one hard contract: PCR is fed a complete matrix or nothing."""
    for seed in range(4):
        for shape in [(24, 18), (18, 24), (30, 9), (9, 30)]:
            mask = _mnar_mask(*shape, seed=seed)
            for i, j in np.argwhere(mask == 0):
                AR, AC = _find_anchors(mask, int(i), int(j))
                assert _is_complete_cross(mask, int(i), int(j), AR, AC), (seed, shape, i, j)


def test_feasibility_does_not_depend_on_orientation():
    """Whether a cross exists is a property of the mask, not of which side is long.

    The exact block may differ under transpose, because ties break toward
    dropping a row. Whether anything is found may not.
    """
    for seed in range(4):
        for shape in [(20, 14), (14, 20), (30, 8), (8, 30)]:
            mask = _mnar_mask(*shape, seed=seed)
            for i, j in np.argwhere(mask == 0):
                a = _find_anchors(mask, int(i), int(j))[0].size > 0
                b = _find_anchors(mask.T, int(j), int(i))[0].size > 0
                assert a == b, (seed, shape, i, j)


def test_block_missingness_returns_the_whole_neighborhood():
    """The panel fast path: the neighborhood is already complete, so nothing is stripped."""
    mask = np.ones((15, 12), dtype=int)
    mask[0, 8:] = 0                              # one treated unit, post-periods
    AR, AC = _find_anchors(mask, 0, 8)
    assert sorted(AR) == list(range(1, 15))
    assert sorted(AC) == list(range(8))


def test_a_long_thin_mask_still_yields_a_cross():
    """The regression, in its simplest form.

    Many rows, few columns, one hole in the neighborhood. Comparing raw missing
    counts makes every column look worse than every row, so the search strips
    all the columns and returns nothing. Comparing shares keeps the block.
    """
    mask = np.ones((40, 5), dtype=int)
    mask[3, 2] = 0                               # one hole inside the block
    mask[7, 4] = 0                               # and another
    AR, AC = _find_anchors(mask, 0, 1)
    assert AR.size > 0 and AC.size > 0
    assert mask[np.ix_(AR, AC)].all()


def test_no_cross_is_lost_across_a_sweep_of_mnar_masks():
    """Aggregate form of the regression: the search must not strip a side to zero.

    Every one of these targets has at least one observed row and one observed
    column, and a 1x1 cross is always available among them, so a search that
    reports nothing has thrown away a block it held.
    """
    lost = total = 0
    for seed in range(5):
        mask = _mnar_mask(24, 18, seed=seed)
        for i, j in np.argwhere(mask == 0):
            i, j = int(i), int(j)
            rows = [r for r in np.where(mask[:, j] > 0)[0] if r != i]
            cols = [c for c in np.where(mask[i, :] > 0)[0] if c != j]
            if not rows or not cols:
                continue                          # no cross can exist
            total += 1
            lost += _find_anchors(mask, i, j)[0].size == 0
    assert total > 100
    assert lost == 0, f"{lost}/{total} targets lost their anchor cross"


def test_the_block_is_not_degenerate_on_mnar_masks():
    """A 1x1 cross is legal but useless; the search should do much better.

    Against the exact maximum biclique's mean min-dimension of 5.94 on these
    masks, the greedy search reaches 5.36. The floor here is set well below
    that so ordinary drift does not trip it, and well above 1 so a collapse
    back toward degenerate blocks does.
    """
    dims = []
    for seed in range(5):
        mask = _mnar_mask(24, 18, seed=seed)
        for i, j in np.argwhere(mask == 0):
            AR, AC = _find_anchors(mask, int(i), int(j))
            if AR.size:
                dims.append(min(AR.size, AC.size))
    assert np.mean(dims) > 4.0


# ----------------------------------------------------------------------- edge
def test_a_single_observed_row_and_column_give_a_one_by_one_cross():
    mask = np.zeros((6, 6), dtype=int)
    mask[2, 3] = 1                               # the only observed cell
    mask[2, 5] = 1                               # target row 5 observed at col 3
    mask[0, 3] = 1
    AR, AC = _find_anchors(mask, 0, 5)
    assert AR.tolist() == [2] and AC.tolist() == [3]


def test_a_target_column_nobody_observes_reports_empty():
    mask = np.ones((6, 6), dtype=int)
    mask[:, 2] = 0
    AR, AC = _find_anchors(mask, 0, 2)
    assert AR.size == 0 and AC.size == 0


def test_a_target_row_observed_nowhere_reports_empty():
    mask = np.ones((6, 6), dtype=int)
    mask[0, :] = 0
    AR, AC = _find_anchors(mask, 0, 3)
    assert AR.size == 0 and AC.size == 0


# -------------------------------------------------------------------- failure
def test_min_anchor_refuses_a_block_below_the_requested_size():
    """A caller asking for a 3x3 cross gets nothing when only 1x1 exists."""
    mask = np.zeros((6, 6), dtype=int)
    mask[2, 3] = mask[2, 5] = mask[0, 3] = 1
    AR, AC = _find_anchors(mask, 0, 5, min_anchor=3)
    assert AR.size == 0 and AC.size == 0


@pytest.mark.parametrize("shape", [(40, 5), (5, 40), (60, 3)])
def test_the_count_rule_regression_on_every_aspect_ratio(shape):
    """The bug was orientation-specific, so the guard has to cover both."""
    mask = _mnar_mask(*shape, r=2, seed=3)
    rows_with_cross = 0
    eligible = 0
    for i, j in np.argwhere(mask == 0):
        i, j = int(i), int(j)
        if not mask[:, j].any() or not mask[i, :].any():
            continue
        rows = [r for r in np.where(mask[:, j] > 0)[0] if r != i]
        cols = [c for c in np.where(mask[i, :] > 0)[0] if c != j]
        if not rows or not cols:
            continue
        eligible += 1
        rows_with_cross += _find_anchors(mask, i, j)[0].size > 0
    assert eligible > 0
    assert rows_with_cross == eligible
