r"""Path C: what LEXSCM's cumulative paths cover, and what they do not.

LEXSCM's cumulative band (``mlsynth.utils.fast_scm_helpers.post_inference``)
prices a total effect over an ``H``-period horizon from a block-sum pivot:
circular moving blocks of held-out residuals, pooled across the other treated
units and any extra held-out series, rescaled to the series being priced. There
is no paper behind it and no reference implementation, so this case validates it
against the only external target available -- its own nominal level -- under a
simulation whose truth is set here.

Three claims are stated in that module's docstrings and nothing checked any of
them. Each is a cell below.

1. The aggregate path covers the effect on the treated, ``tau^T = w . tau``, at
   about its nominal level, and a thin pool is what costs it. A four-unit design
   reading its null from three peers has three series to build blocks from, so
   the tail quantiles are thin; ``extra_pool`` adds held-out series and the
   aggregate recovers.

2. The per-unit paths stay short of nominal where the aggregate reaches it,
   and a thicker pool narrows that gap without closing it. Each unit's
   synthetic control sits at its own fitted level, that offset runs through the
   post window unchanged, and no resampling of residuals shifts a location
   offset. Measured here: twenty extra series take the aggregate from 0.800 to
   0.890 and the per-unit paths from 0.826 to 0.878, both at a nominal 0.90.
   The residual is the offset, and it is small at this offset size because the
   module's approximability screen now drops the series whose offsets are large
   -- so the regime where this shortfall was severe is the regime the screen
   removes.

3. The population path's representation term is what separates ``tau`` from
   ``tau^T``. The two differ by ``sum_j (w_j - f_j) tau_j``, which has mean zero
   under exchangeable effects but a variance of ``var(tau_j) ||w - f||^2``, and
   over ``h`` periods a per-period offset accumulates as ``h`` against the gap
   term's ``sqrt(h)``. The treated interval, carrying only the gap term, misses
   the population effect badly; the population interval repairs most of it.

The DGP. Each treated unit ``j`` has a constant per-period effect ``tau_j``, a
fitted-level offset drawn once per replication and held across both windows, and
iid Gaussian period noise. The blank window carries the offset and the noise and
no effect, which is what the estimator assumes of it. The offset's spread is
small against the noise (0.15 against 1.0) so the module's approximability
screen keeps the pool: a larger offset is detected and dropped, which is that
screen working and would measure something else. For the population cells the
per-unit effects are drawn for every unit in the panel, treated or not, so
``f . tau`` is a known number; the untreated units' effects are never observed,
which is the assumption the term rests on.

What the numbers are not. This is not the experiment behind the figures in
``unit_level_cumulative``'s and ``population_cumulative``'s docstrings. Those
state a four-unit design at nominal 0.90 and the pool sizes, and no horizon,
blank length, noise scale or weights, so they cannot be reproduced from what is
written; they also predate the pool screen that now runs by default. The
qualitative geometry agrees -- aggregate short at an empty pool and recovering,
per-unit short everywhere, the representation term adding tens of points of
population coverage -- and the levels here belong to this DGP.

Provenance: no external source. The construction is mlsynth's own, documented on
``docs/lexscm.rst`` under "Cumulative effects after the intervention" and
verified here. Companion: ``conformal_window_count``, which isolates the same
family's dependence on the number of calibration windows.
"""
from __future__ import annotations

import warnings

import numpy as np

_H = 8                      # horizon the totals are read at
_TB = 24                    # blank (held-out) pre-periods
_SD_LEVEL = 0.15            # spread of the per-unit fitted-level offset
_SD_NOISE = 1.0             # per-period idiosyncratic noise
_LEVEL = 0.90               # nominal coverage
_N_DRAWS = 400

# Design (cells 1-2): four treated units, unequal weights.
_TAU = np.array([1.0, 0.6, -0.4, 0.2])
_W = np.array([0.4, 0.3, 0.2, 0.1])

# Population design (cell 3): four treated of twenty-four, equal design weights,
# uniform population weights, effects drawn N(1.0, 0.8^2) for every unit.
_N_UNITS = 24
_N_TREATED = 4
_TAU_MEAN, _TAU_SD = 1.0, 0.8


def _gaps(rng, tau, n_extra):
    """One replication's post gaps, blank gaps and extra held-out series."""
    n = tau.size
    level = rng.normal(0.0, _SD_LEVEL, n)
    post = tau[None, :] + level[None, :] + rng.normal(0.0, _SD_NOISE, (_H, n))
    blank = level[None, :] + rng.normal(0.0, _SD_NOISE, (_TB, n))
    extra = [rng.normal(0.0, _SD_LEVEL) + rng.normal(0.0, _SD_NOISE, _TB)
             for _ in range(n_extra)]
    return post, blank, extra


def _treated_coverage(n_extra: int, seed: int):
    """Coverage of the aggregate path and of the per-unit paths at ``_H``."""
    from mlsynth.utils.fast_scm_helpers.post_inference import unit_level_cumulative

    rng = np.random.default_rng(seed)
    truth_aggregate = _H * float(_W @ _TAU)
    hit_aggregate = hit_unit = seen_unit = 0
    for _ in range(_N_DRAWS):
        post, blank, extra = _gaps(rng, _TAU, n_extra)
        with warnings.catch_warnings():       # the screen reports its drops
            warnings.simplefilter("ignore")
            out = unit_level_cumulative(post, blank, _W, level=_LEVEL,
                                        extra_pool=extra or None)
        point = out.aggregate[_H - 1]
        hit_aggregate += point.lower <= truth_aggregate <= point.upper
        for j, path in enumerate(out.per_unit):
            own = path[_H - 1]
            hit_unit += own.lower <= _H * _TAU[j] <= own.upper
            seen_unit += 1
    return hit_aggregate / _N_DRAWS, hit_unit / seen_unit


def _population_coverage(seed: int, n_extra: int = 8):
    """Coverage of ``tau`` by the population path, and by the treated path."""
    from mlsynth.utils.fast_scm_helpers.post_inference import (
        population_cumulative, unit_level_cumulative,
    )

    rng = np.random.default_rng(seed)
    f = np.full(_N_UNITS, 1.0 / _N_UNITS)
    w = np.full(_N_TREATED, 1.0 / _N_TREATED)
    treated_index = list(range(_N_TREATED))
    hit_population = hit_treated = 0
    for _ in range(_N_DRAWS):
        tau = rng.normal(_TAU_MEAN, _TAU_SD, _N_UNITS)
        post, blank, extra = _gaps(rng, tau[:_N_TREATED], n_extra)
        truth = _H * float(f @ tau)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wide = population_cumulative(post, blank, w, f, treated_index,
                                         level=_LEVEL, extra_pool=extra)
            narrow = unit_level_cumulative(post, blank, w, level=_LEVEL,
                                           extra_pool=extra)
        p, q = wide.aggregate[_H - 1], narrow.aggregate[_H - 1]
        hit_population += p.lower <= truth <= p.upper
        hit_treated += q.lower <= truth <= q.upper
    return hit_population / _N_DRAWS, hit_treated / _N_DRAWS


def run() -> dict:
    empty_aggregate, empty_unit = _treated_coverage(0, seed=31)
    four_aggregate, _ = _treated_coverage(4, seed=32)
    many_aggregate, many_unit = _treated_coverage(20, seed=33)
    population, treated_on_population = _population_coverage(seed=101)

    return {
        # 1. the aggregate, and what a thin pool costs it
        "aggregate_pool_empty": empty_aggregate,
        "aggregate_pool_four": four_aggregate,
        "aggregate_pool_gain": four_aggregate - empty_aggregate,
        # 2. the per-unit paths, and that a thicker pool does not repair them
        "unit_pool_empty": empty_unit,
        "unit_pool_twenty": many_unit,
        "unit_pool_gain": many_unit - empty_unit,
        "aggregate_pool_twenty": many_aggregate,
        # 3. the representation term
        "population": population,
        "treated_interval_on_population": treated_on_population,
        "representation_gain": population - treated_on_population,
    }


# Filled from a run of this module. Coverage is a share of 400 draws, so the
# Monte Carlo standard error is at most 0.025 and the tolerance below is about
# three of those: a real change in the pivot moves these by more than 0.075,
# and resampling noise does not. The per-unit shares pool four units per draw
# and are correlated within a draw, so they get the same tolerance and not a
# tighter one. Differences of two shares carry both errors, hence 0.10.
#
# The pair to read together is `aggregate_pool_twenty` and `unit_pool_twenty`:
# the aggregate reaches its nominal level and the per-unit paths do not, on the
# same draws and the same pool. That ordering is the claim. A change that lifted
# the per-unit paths to nominal by thickening the pool alone would trip
# `unit_pool_twenty`, because the offset it would have to remove is not in the
# pool.
EXPECTED = {
    "aggregate_pool_empty": (0.800, 0.075),
    "aggregate_pool_four": (0.878, 0.075),
    "aggregate_pool_gain": (0.078, 0.10),
    "unit_pool_empty": (0.826, 0.075),
    "unit_pool_twenty": (0.878, 0.075),
    "unit_pool_gain": (0.052, 0.10),
    "aggregate_pool_twenty": (0.890, 0.075),
    "population": (0.785, 0.075),
    "treated_interval_on_population": (0.490, 0.075),
    "representation_gain": (0.295, 0.10),
}
