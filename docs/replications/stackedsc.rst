.. _replication-stackedsc:

STACKEDSC -- Walmart Supercenters and local employment (Wiltshire 2023)
=======================================================================

:Estimator: :doc:`../stackedsc` -- :class:`mlsynth.STACKEDSC`
:Source: Wiltshire, J. C. (2023), *"Walmart Supercenters and Monopsony Power:
   How a Large, Low-Wage Employer Impacts Local Labor Markets,"* Section 4.2.
:Reference implementation: ``allsynth`` (Stata), Wiltshire (2022)
:Replication type: Path A -- the paper's empirical result on the author's data.
:Status: Partial. The event-study shape and the qualitative finding reproduce;
   the point estimates do not match cell for cell, and the reason is recorded
   below rather than tuned away.

The question
------------

Walmart opened Supercenters across the United States over 1990-2005, county by
county, in different years. Did their arrival raise or lower total local
employment? A single-treated-unit synthetic control cannot answer that, and a
regression with two-way fixed effects has the staggered-adoption problems that
motivated much of the recent difference-in-differences literature.

The paper's answer is to build a separate synthetic control for each treated
county from a deliberately constructed donor pool -- counties where Walmart
tried to open a Supercenter and was blocked by local opposition -- and then
average the county-level effects on a common event clock.

Data
----

566 treated counties in six adoption cohorts (1995 through 2000) and 39
never-treated donor counties, over 1990-2005. The donor pool is the paper's
identifying contribution: places where the firm revealed an intention to enter
but did not, rather than places selected on observables.

Effects are reported as percent changes, because county employment in the
sample runs from about 3,900 to 866,000. Aggregation is weighted by 1990
population, which the paper adopts so that large percentage swings in small
counties do not drive the average.

What reproduces
---------------

The event-study shape, which is what the paper's Figure 6 shows and what its
narrative rests on:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - feature
     - paper
     - mlsynth
   * - pre-treatment fit
     - "an excellent pre-treatment fit"
     - within :math:`\pm 0.07` percent
   * - effect at entry
     - "no effect in the year of Supercenter entry"
     - :math:`+0.16`
   * - onset
     - "a downward trend begins in the year following"
     - decline from :math:`e = 2`
   * - direction
     - large negative by :math:`e = 5`
     - :math:`-0.90`

The gap at :math:`e = -1` comes out at :math:`-6 \times 10^{-16}`, which is the
indexing constraint holding to machine precision rather than a fitted result.

The specification, and the part of it that is not expressible
-------------------------------------------------------------

Appendix A.2 matches on ten covariate averages plus four outcome lags. The
covariate averages are over an absolute window -- the five years before any unit
is treated -- which ``covariate_windows`` expresses directly. The outcome lags
are relative to each cohort's own base period, which it does not: a covariate is
one value per unit, and a donor serves all six cohorts with six different base
periods, so no single column can carry "the outcome two years before treatment"
for a donor.

The benchmark therefore matches on the pre-treatment outcome path, which is a
superset of four lags of it, and which is the specification that delivers the
paper's stated pre-treatment fit. The covariate specification as expressible
today is measured alongside it rather than quietly dropped: its pre-treatment
RMSE is 1.354 against 0.049, a factor of 27. That ratio is pinned, so a future
option for base-period-relative covariate windows can be seen to close it.

What does not reproduce, and why
--------------------------------

Table 4 of the paper reports aggregate employment five years after entry at
:math:`-1.77` percent without bias correction and :math:`-3.14` percent with
it. The estimator here does not land on either figure, and the reason is a
component that cannot be recovered from the published materials.

The reference implementation calls Stata's ``synth``, whose default
predictor-weight rule is a regression on the pre-treatment outcomes rather than
the nested optimisation of Abadie, Diamond and Hainmueller. That rule ships as
a compiled Mata library with no source distributed. Its structure is recoverable
from the library's symbol table -- scale by the pooled standard deviation, pool
the treated unit into the regression, set the weights proportional to squared
coefficients -- but two choices are not: whether the regression is run once over
all unit-period observations or once per pre-treatment period, and whether an
intercept is fitted.

Those two readings are not a rounding difference. Across eight of the paper's
reported cells they give total absolute errors of 1.854 and 1.768 -- a tie -- and
the choice of rule moves the five-year estimate by 1.1 to 1.4 percentage points
on an effect of about :math:`-1.8`. Larger, that is, than the quantity being
estimated.

Worse for a clean claim, no single rule reproduces both of the paper's columns.
The regression rule wins the uncorrected column (:math:`-1.777` against a target
of :math:`-1.767`); the nested search wins the corrected one. The split is
systematic rather than noisy, because the bias correction depends on the
residual predictor imbalance and therefore weights predictor fit differently
from the uncorrected estimator.

So this replication claims the shape and the sign, and does not claim the
magnitudes. Closing the gap requires running ``allsynth`` once and capturing the
per-county weights for a single cohort -- a measurement, not an argument.

Two findings from the port
--------------------------

An early version of this port regressed on raw rather than normalised
predictors, inverting the order in which the reference implementation applies
its two steps. That single ordering error moved the five-year estimate from
:math:`-1.43` to :math:`-1.78` on one outcome and flipped the sign on another,
from :math:`-0.655` to :math:`+0.620` against a target of :math:`+1.112`.

The lesson generalises past this paper: when a predictor-weight rule is
involved, the order of normalisation and estimation is load-bearing, and
getting it wrong produces output that looks entirely reasonable.

A second one, about the weights themselves. Each cohort has five to ten
pre-treatment periods against 39 donors, so the design matrix has a null space
of at least 29 dimensions. Moving the weights along it changes the
post-treatment prediction without touching the pre-treatment fit at all: the
per-county weight vectors are not identified, and two solvers that both reach
the optimum can differ by several percentage points on an individual county
while agreeing on the average to within 0.03. Read the aggregate; do not read a
single county's weights as that county's synthetic control.

Inference
---------

The estimator implements the paper's sampled-placebo-average procedure
(``inference="placebo"``), described on :doc:`../stackedsc`. It is not part of
what this page claims: the p-values the paper reports rest on the same
predictor-weight rule the point estimates do, so a replication of them is
blocked behind the same missing measurement. Running it on the paper's panel
costs 566 x 39 simplex solves under the default donor pool.

Reproducing it
--------------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case wiltshire_walmart

The case is `benchmarks/cases/wiltshire_walmart.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/wiltshire_walmart.py>`_.
It runs in about 45 seconds and pins twenty-one quantities: the design's counts,
the three structural identities (the base-period gap at 6e-16, the weights
summing to one, the aggregate being exactly the weighted mean of the parts), the
paper's four prose claims, the two specification contrasts above, and the
forfeited batching under a commuting-zone donor restriction.

One row is a floor rather than a value. ``county_dispersion_at_five`` records
that the per-county five-year gaps have a standard deviation of 14.1 percent
around a weighted mean of -0.90, and it carries a deliberately wide band. Part
of that spread is genuine heterogeneity across counties and part is the
non-identification described above, so the number itself is not reproducible
across solvers -- what would be a real regression is the dispersion collapsing,
which would mean the per-county fits had stopped varying at all.

Inference is not part of this case. The placebo layer costs 22,074 simplex
solves in its default donor pool on this panel, well past the runtime budget,
and the paper's p-values rest on the same unrecoverable predictor-weight rule as
its magnitudes.
