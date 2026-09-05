Gaussian-Process Interrupted Time Series (GPITS)
================================================

.. currentmodule:: mlsynth

Overview
--------

Some interventions reach everybody at once. A Supreme Court ruling binds every
state on the day it is handed down; a national lockdown applies to the whole
country; a platform-wide product change ships to all users. Difference-in-differences
and synthetic control both work by comparing the treated to somebody who stayed
untreated, and in these settings nobody did. The comparison has to come from the
same unit at a different time.

That design is the interrupted time series: learn how the outcome behaved before
the intervention, project that forward, and read the effect off the gap between
what happened and the projection. GPITS is Cho's (2026) version of it. Instead of
committing to a straight line or a polynomial, it puts a Gaussian process over
the possible trends and conditions on the pre-treatment record. What survives is
the set of trajectories consistent with what was observed, and the counterfactual
is their posterior mean.

The reason to carry the whole set, and not one fitted curve, is uncertainty.
Many trends fit a pre-period equally well and disagree about the future, and they
disagree more the further out you look. A segmented regression reports the
uncertainty of its own fitted line, which does not grow with the horizon, so its
interval is the same width one month out and two years out. A Gaussian process
reports how much the trends it still admits disagree at the point you are asking
about, and that grows with distance from the data.

When to use this estimator
--------------------------

Reach for GPITS when there is no untreated comparison unit at all and the
pre-treatment series has structure a kernel can name:

* Universal treatments. A nationwide statute, a court decision, a pandemic — the
  paper's application is the 2008 ruling in *District of Columbia v. Heller*,
  which bound every U.S. jurisdiction but in practice struck down only D.C.'s
  handgun ban.
* Seasonal series with a known cycle. Monthly or daily data where the periodic
  kernel can carry the cycle and the linear component the trend.
* Short post-treatment windows. The identifying assumption weakens as the horizon
  lengthens, so the design is most credible over a handful of periods.

If you have credible untreated donors, use them: :doc:`clustersc`, :doc:`pda` and
:doc:`sbc` will be more precise. If every unit is treated but the series has
recurring local structure without a clean period, :doc:`shc` matches overlapping
historical blocks of the same series instead of imposing a kernel, and infers by
conformal permutation. The two divide as follows: reach for :doc:`shc` when you
cannot name the cycle, and for GPITS when you can, or when you want an interval
that widens with the forecast horizon.

Notation
--------

Let :math:`Y_{it}` be the outcome for unit :math:`i` at time :math:`t`, and
:math:`D_{it} = \mathbf{1}(t \ge t_0)` the treatment indicator, which turns on at
the same :math:`t_0` for every unit. Write :math:`Y_{it}(1)` and :math:`Y_{it}(0)`
for the potential outcomes. Let :math:`X_{it}` collect the observed inputs — the
calendar time itself, plus any covariates — and :math:`U_{it}` the unobservables.

The untreated outcome is governed by

.. math::

   Y_{it}(0) = h_i(X_{it}, U_{it}) + \varepsilon_{it},
   \qquad \mathbb{E}[\varepsilon_{it} \mid X_{it}, U_{it}] = 0 ,

and the object the design targets is the best prediction available from
observables,

.. math::

   g_i(X_{it}) := \mathbb{E}[Y_{it}(0) \mid X_{it}] .

The estimand is the unit-period effect against that conditional mean,

.. math::

   \tau_{it^*} = Y_{it^*}(1) - g_i(X_{it^*}), \qquad t^* \ge t_0 ,

with the average and cumulative versions over the post-treatment window following
from it.

Assumptions
-----------

1. Consistency and no anticipation.
   :math:`Y_{it} = D_{it} Y_{it}(1) + (1 - D_{it}) Y_{it}(0)`, and before
   :math:`t_0` the treated and untreated potential outcomes coincide.

   Remark. The second half is what makes the pre-period usable as training data.
   It rules out anticipatory drift, not stable expectations: a belief held at a
   constant level throughout the pre-period is absorbed into the learned
   relationship. Behaviour that shifts in advance of the intervention is the
   failure mode, and it is what the placebo checks below are built to detect.

2. Mean sufficiency. For every unit and period,
   :math:`\mathbb{E}[Y_{it}(0) \mid X_{it}, U_{it}] = \mathbb{E}[Y_{it}(0) \mid X_{it}]`.

   Remark. This is the assumption that replaces the donor pool, and it is a
   restriction on first moments only, weaker than conditional independence. It
   says that once you know the calendar date and the covariates, knowing the
   unobservables would not change your expectation of the untreated outcome. It
   fails when something other than the intervention starts moving the series
   after :math:`t_0` — a concurrent policy, a structural break, a shock that took
   values in the post-period it never took before. Nothing in the pre-period can
   confirm it, which is why shorter post-treatment windows are more credible and
   why domain knowledge about concurrent events is doing real work.

3. A kernel that can represent the counterfactual. The trend must lie in, or
   close to, the space of functions the kernel treats as plausible, at a
   complexity the pre-period can pin down.

   Remark. This is where the choice between ``kernel="gaussian"`` and
   ``kernel="gaussian_periodic_linear"`` bites, and it is not a cosmetic setting.
   The Gaussian kernel is stationary: far from the training data its posterior
   reverts to the prior mean and its band flattens at a ceiling, so it cannot
   carry a trend forward. The combined kernel adds a periodic component for the
   cycle and a linear component for the trend, and it is the working form for any
   series that is going somewhere. Cho reports the practical size of this: on the
   *Heller* series a unit-variance trend costs a complexity budget of 0.96 under
   the combined kernel against 3.52 under the Gaussian alone, and the reported
   interval is short by that factor when the budget exceeds one.

Inference and diagnostics
-------------------------

The counterfactual band is the Gaussian-process posterior variance. Cho's
Proposition 2 gives it a frequentist reading: it bounds the worst-case
disagreement, at the period you are asking about, among the functions in the
kernel's unit ball that agree with everything observed before the intervention.
The interval is therefore calibrated to the worst case the model class admits,
not to the typical case.

Two consequences follow, and both matter in practice. The intervals cover at or
above their nominal rate, which is what you want from an extrapolation. And they
are wide: in replication of the paper's own simulation the coverage sits at 1.000
in most cells, with intervals a median of 2.3 times wider than a segmented
regression's. Coverage is bought with power, so an effect that is small relative
to the pre-period noise will sit inside the band. The *Heller* effect survives
because it is roughly 20 times D.C.'s pre-period standard deviation.

Set ``placebo_periods`` to run the temporal placebo check. It refits on
everything before each of the last few pre-treatment periods and predicts one
step ahead, where the true effect is zero. The check is one-sided by
construction: a confounder inside the training window is absorbed into the fit
and leaves the placebo clean even when the same disturbance breaks mean
sufficiency after :math:`t_0`. A clean placebo reports the absence of detected
instability, not the presence of identification.

``result.inference.ci_lower`` / ``ci_upper`` give the ATT interval, and
``result.cumulative_ci`` the interval on each running cumulative total. Both use
the full post-period posterior covariance, not its diagonal, because
successive counterfactual errors covary and summing variances alone would
understate a running total.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import GPITS

   # Monthly handgun background checks in D.C., 2002-07 to 2008-10;
   # the Heller decision lands 2008-07.
   df = pd.read_csv("basedata/dc_handgun_heller.csv", parse_dates=["date"])

   res = GPITS({
       "df": df,
       "outcome": "handgun_rate",
       "treat": "treated",
       "unitid": "unit",
       "time": "date",
       "covariates": ["month"],
       "categorical_covariates": ["month"],
       "kernel": "gaussian_periodic_linear",
       "period": 12,
       "placebo_periods": 4,
   }).fit()

   print(res.effects.att)                  # mean monthly effect
   print(res.cumulative_effect[-1])         # 15.13 checks per 100k over 4 months
   print(res.cumulative_ci[-1])             # (12.97, 17.30)
   print(res.placebo.all_cover)             # True

The plot helpers return figures and leave showing and saving to you.
``plot_gpits`` draws the fit panel, which is what ``display_graphs`` shows:

.. code-block:: python

   from mlsynth.utils.gpits_helpers import plot_gpits, plot_gpits_panels

   fig = plot_gpits(res)
   fig.savefig("heller_fit.png", dpi=150)

The fit panel draws the pre-period band in grey and the post-period band in the
counterfactual colour, because the two are different quantities: before the
intervention the band is a fit's uncertainty, after it the band is an
extrapolation's.

``plot_gpits_panels`` returns the four panels of the paper's own plotting code
(``plot.gp_its`` in the replication repository), keyed by its names:

.. code-block:: python

   figs = plot_gpits_panels(res)
   figs["fit"]          # observed points, fitted trend, counterfactual
   figs["pointwise"]    # per-period effects, placebo window shaded separately
   figs["cumulative"]   # running total with its interval
   figs["average"]      # running average, ending at the ATT

The pointwise panel is the one that carries the paper's argument: the placebo
periods sit flat near zero immediately before an intervention the observed
series jumps away from. Time on the effect panels is measured from the
intervention, so period 0 is the first treated period.

Verification
------------

GPITS reproduces the paper's empirical result exactly and is cross-validated
against the author's own R implementation. See :doc:`replications/gpits` for the
numbers, and
`benchmarks/cases/gpits.py <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/gpits.py>`_
for the durable case.

Core API
--------

.. autoclass:: GPITS
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.utils.gpits_helpers.config.GPITSConfig
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.gpits_helpers.structures.GPITSResults
   :members:
   :undoc-members:

References
----------

Cho, S. (2026). "Let Time Tell: Identification and Gaussian Process Estimation
for Interrupted Time Series." arXiv:2608.20610.

Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine
Learning*. MIT Press.
