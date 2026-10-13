"""TBR Path B: Kerman section 5.2's coverage and bias, through mlsynth.TBR.

Kerman, Wang and Vaver (2017) section 5 assesses TBR by simulation. Figure 7
reports that the 90% and 50% posterior intervals attain nominal coverage across
every combination of between-geo correlation and noise and every pretest
length; section 5.2 reports the iROAS posterior median as practically unbiased
at a true value of 2.0.

``benchmarks/studies/tbr_geo/simulation.py`` established that against
``studies/tbr_geo/tbr.py``, a port written from sections 3.2 and 9.1 before the
estimator existed. This case runs the same design through ``mlsynth.TBR``,
which is a different body of code: ingestion through ``dataprep``, the fit
through ``fit_pretest``, the posterior through ``cumulative_posterior`` and
``interval``. Cross-validation against ``google/matched_markets``
(``benchmarks/cases/tbr.py``) shows the two implementations agree on one panel.
It cannot show that either attains nominal coverage, because agreement is not
calibration. That is what this case measures.

The generator is ``studies/tbr_geo/dgp.py`` unchanged, so there is one section
5.1 in the repository and not two.

Two assignment schemes
----------------------

The paper does not say how geos are split between the groups. The study draws a
free permutation. The authors' own R package, released in section 6 as
``google/GeoexperimentsResearch``, uses ``GeoStrata(n.groups=2,
group.ratios=c(1, 1))`` with ``Randomize``: sort geos by volume, divide into
strata of size two, one geo of each pair to each group. Both run here, over the
same cells at the same replication count, so the two are comparable by
construction. Stratified pairing balances volume between the groups and so
raises the between-group correlation and narrows the intervals; nominal
coverage under both is a stronger statement than under either alone.

What is asserted, and what is not
---------------------------------

Pooled coverage is asserted against nominal with a tolerance of roughly four
sampling standard deviations, which is a property of the replication count and
not a number read off a previous run.

Per-cell coverage is asserted against the paper's own criterion: the posterior
of the rate under Kerman (2011)'s neutral prior, ``Beta(1/3 + y, 1/3 + n - y)``,
has to contain the nominal rate. The allowance is three cells of thirty-six at
a 99% interval, where a calibrated estimator misses 0.36 cells on average and
four or more is a one-in-sixteen-hundred event. Asserting zero misses would
fail by chance: at a 95% interval a calibrated estimator misses 1.8 cells of
thirty-six on average.

Squared bias as a share of MSE is reported and not asserted. Section 5.2 gives
it as 0.04% and reads it as evidence of unbiasedness, and it cannot carry that
reading: for any unbiased estimator its expectation is ``1 / n``, so it
measures the replication count.
``studies/tbr_geo/results/monte_carlo_floor.txt`` has the sweep. The median
iROAS is asserted instead.

``MLSYNTH_TBR_MC_REPS`` sets the replications per cell. The default keeps the
case to a few minutes; the paper's own 2000 is about forty minutes for the two
arms and is what a deliberate run should use.
"""
from __future__ import annotations

import os

from benchmarks.studies.tbr_geo import mlsynth_coverage as mc

#: Replications per cell. The paper uses 2000; see the module docstring.
REPS = int(os.environ.get("MLSYNTH_TBR_MC_REPS", "150"))

#: Sampling standard deviation of a pooled rate over the whole grid.
_N = len(mc.RHOS) * len(mc.CS) * len(mc.PRES) * REPS


def _sd(p: float) -> float:
    return (p * (1.0 - p) / _N) ** 0.5


def run() -> dict:
    out = {}
    for scheme, tag in (("permutation", "perm"), ("stratified", "strat")):
        rows = mc.run_grid(REPS, scheme=scheme, seed=29)
        s = mc.summarise(rows, level=0.99)
        out.update({
            f"{tag}_cov90": s["cov90_pooled"],
            f"{tag}_cov50": s["cov50_pooled"],
            f"{tag}_cells_off_90": s["cov90_cells_off_nominal"],
            f"{tag}_cells_off_50": s["cov50_cells_off_nominal"],
            f"{tag}_iroas_median_dev": s["iroas_median_pooled_abs_dev"],
            f"{tag}_cells": s["n_cells"],
        })
        # Reported, not asserted: the 1/n floor. See the module docstring.
        print(f"  [{scheme}] squared bias / MSE = "
              f"{s['bias_share_of_mse_pct']:.4f}%   (1/n = {100.0 / REPS:.4f}%)"
              f"   max |median iROAS - 2| over cells = "
              f"{s['iroas_median_max_abs_dev']:.4f}")
    return out


EXPECTED = {
    # Figure 7: nominal coverage, both assignment schemes. The tolerance is
    # four sampling standard deviations of the pooled rate at this replication
    # count, so it tightens when REPS is raised and is not a value read off a
    # previous run.
    "perm_cov90": (0.90, round(4 * _sd(0.90), 4)),
    "perm_cov50": (0.50, round(4 * _sd(0.50), 4)),
    "strat_cov90": (0.90, round(4 * _sd(0.90), 4)),
    "strat_cov50": (0.50, round(4 * _sd(0.50), 4)),
    # Per-cell, against the paper's Beta criterion at 99%. Three of thirty-six
    # allowed; see the module docstring for why zero would be the wrong test.
    "perm_cells_off_90": (0.0, 3.5),
    "perm_cells_off_50": (0.0, 3.5),
    "strat_cells_off_90": (0.0, 3.5),
    "strat_cells_off_50": (0.0, 3.5),
    # Section 5.2's bias claim, as the median and not as the share of MSE.
    "perm_iroas_median_dev": (0.0, 0.05),
    "strat_iroas_median_dev": (0.0, 0.05),
    # The grid the paper specifies: three correlations by three noise levels by
    # four pretest lengths.
    "perm_cells": (36.0, 0.5),
    "strat_cells": (36.0, 0.5),
}
