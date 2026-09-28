TBR: Time-Based Regression
==========================

When to use it
--------------

You ran a geo experiment: some geographic markets got the advertising change,
others did not, and you want to know what the change did to sales. The obvious
approach is to compare the two groups across markets, which is what geo-based
regression does, and it needs many markets to work, because each market is one
observation. In a small country, a subregion, or a matched-market test with a
single test city and a single control city, there are not enough.

TBR turns the problem on its side. It adds up the response in the treated
markets to get one series through time, adds up the untreated markets to get
another, and learns how the two moved together before the change. That learned
relation predicts what the treated group would have done had nothing happened.
Observations are now time periods, not markets, so a handful of geos is enough.

Reach for it when the treatment groups are already fixed by an experimental
assignment and the unit count is small. Reach for :doc:`vanillasc` or
:doc:`sdid` instead when you are picking donor weights from a large pool of
untreated units and no experiment was run. :doc:`cmbsts` produces a similarly
shaped answer, a cumulative effect with a credible interval, from a full
state-space model; TBR is one regression on two series, which is why it
survives a panel with five geos in it.

Notation
--------

Index geos by :math:`i` and periods by :math:`t`. The experiment splits the
geos into a treatment group, a control group, and geos assigned to neither. Let

.. math::

   y_t = \sum_{i \in \text{treatment}} m_{i,t},
   \qquad
   x_t = \sum_{i \in \text{control}} m_{i,t}

be the group totals of the response metric :math:`m`. Periods before the
intervention form the pretest window, of length :math:`n`; periods from the
intervention onward form the test window.

Over the pretest window TBR fits

.. math::

   y_t = \alpha + \beta x_t + \epsilon_t,
   \qquad \epsilon_t \sim N(0, \sigma^2),

and for each test period takes the counterfactual to be
:math:`y^{*}_t = \alpha + \beta x_t + \epsilon^{*}_t`. The per-period effect is
:math:`\phi_t = y_t - y^{*}_t` and the estimand is its running total,

.. math::

   \Delta(T) = \sum_{t=1}^{T} \phi_t .

Assumptions
-----------

1. The two group totals are linearly related, and that relation is stable
   through the test window.

   Remark. This is the assumption the method rests on, and only half of it is
   checkable. The pretest half is visible in the residuals; the test half is
   not, because the treated group's untreated path stops existing the moment the
   intervention starts. A design that hands TBR two groups whose relation drifts
   will still produce an interval, and the interval will be wrong. Au (2018)
   exists to choose groups for which the relation is plausible, and that is a
   separate step from estimating with them.

2. The pretest residuals are independent and identically distributed normal.

   Remark. Serial correlation in the residuals is the common failure, and it
   makes the interval too narrow because the effective sample size is smaller
   than the period count. A Durbin-Watson or Breusch-Godfrey test on the pretest
   residuals detects it, and both are part of how Au scores a candidate design.

3. The control geos are unaffected by the intervention.

   Remark. If advertising in the treated markets moves demand in the control
   markets, :math:`x_t` carries part of the effect and the counterfactual
   absorbs it, which shrinks the estimate toward zero. Geographic separation is
   the usual defence. Geos assigned to neither group are the mechanism for
   removing a contaminated market without deleting it from the panel.

4. Aggregation is over a fixed set of geos.

   Remark. A geo that enters or leaves mid-experiment changes the meaning of the
   totals. An absent geo-period is summed as a zero, which is correct when the
   cell is absent because nothing happened and wrong when it is absent because
   the data are missing. The estimator reports how many cells it filled so the
   distinction is visible.

Inference
---------

Under a prior uniform on :math:`(\alpha, \beta, \log \sigma)`, the posterior of
:math:`\Delta(T)` is a shifted and scaled t-distribution on :math:`n - 2`
degrees of freedom, with median

.. math::

   \Delta(T) = T \left( \bar{y}_T - \alpha - \bar{x}_T \beta \right)

and scale

.. math::

   T s \left( v_\alpha + 2 \bar{x}_T v_{\alpha\beta}
              + v_\beta \bar{x}_T^2 + 1/T \right)^{1/2},

where :math:`\bar{y}_T` and :math:`\bar{x}_T` are test-window averages,
:math:`s` is the residual standard deviation, and :math:`v_\alpha`,
:math:`v_\beta`, :math:`v_{\alpha\beta}` are entries of the unscaled
:math:`(X'X)^{-1}`. No resampling is involved.

The first three terms inside the root carry uncertainty about the fitted
relation and the :math:`1/T` carries the unobserved errors of the test window
itself, so parameter uncertainty grows as :math:`T^2` while observation noise
grows only as :math:`T`. A longer experiment buys precision about the
per-period effect and spends it on the total.

The scale is reported and not the standard deviation, which does not exist for
four or fewer pretest periods.

Return on ad spend
------------------

Supplying a spend column adds a second estimand, the incremental return on ad
spend,

.. math::

   \mathrm{iROAS}(T) = \Delta_{\text{resp}}(T) / \Delta_{\text{cost}}(T),

the cumulative effect on the response over the cumulative effect on cost. The
ratio of two t-distributed quantities has no closed form, so it is simulated
from the two posteriors and summarised by its median.

One case avoids the simulation. When the campaign is new, there is no spend
anywhere in the pretest window and none in the control group, so the cost
counterfactual is zero with certainty and the denominator is a known constant.
The ratio is then a rescaled t, and the estimator reports which route it took
on ``iroas.fixed_cost``.

Cooldown
--------

An intervention's effect does not always stop when the intervention does.
``cooldown_col`` flags the periods after the campaign ends, as a 0/1 indicator
that is a property of the period: constant across geos within a period, and
once it turns 1 it stays 1. The cumulative posterior covers the whole
post-treatment window either way; the flag says where inside it the campaign
stopped, so ``effect_at_intervention_end`` and ``effect_at_cooldown_end`` can be
read against each other. Effects that have stopped accumulating by the end of
the campaign mean the cooldown added uncertainty and nothing else.

Designing against estimating
----------------------------

TBR runs in two modes, because scoring a candidate experiment and estimating
from a finished one are different jobs.

Estimation mode takes ``treat``, the ordinary indicator naming which units were
treated and when.

Design mode takes ``post_col`` and ``treatment_col`` instead, with no ``treat``
at all, and so runs on a panel in which nothing was treated. This is what
scoring a hypothetical group split requires, and what an A/A test requires: take
the control geos of a finished experiment, split them in two, and TBR should
report no effect. Au's Example 3 measures exactly that, and it is how a design's
error rate is checked before an experiment is run. :doc:`lexscm` and
:doc:`syndes` separate the two phases the same way.

Both modes compute identically once the window and the groups are named.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import TBR
   from mlsynth.config_models import TBRConfig

   # long panel: one row per geo per day
   res = TBR(TBRConfig(
       df=panel,
       unitid="geo", time="date", outcome="sales", treat="D",
       control_col="is_control",      # which untreated geos are controls
       cost_col="cost",               # optional: adds iROAS
       level=0.9,
       display_graphs=False,
   )).fit()

   res.cumulative.estimate[-1]        # cumulative effect at the end
   res.cumulative.lower[-1], res.cumulative.upper[-1]
   res.iroas.estimate                 # return on ad spend
   res.tbr_fit.beta                   # the fitted pretest relation
   res.filled_cells                   # absent geo-days summed as zero

Design mode, with no treatment in the panel:

.. code-block:: python

   res = TBR(TBRConfig(
       df=history,
       unitid="geo", time="date", outcome="sales",
       control_col="is_control", treatment_col="is_treatment",
       post_col="post",
       display_graphs=False,
   )).fit()

The three-panel figure of the paper's Section 3.3, the two series, the per-period
difference, and the cumulative band, comes from the estimator's own plotter,
which returns the figure for the caller to display or save:

.. code-block:: python

   from mlsynth.utils.tbr_helpers.plotter import plot_tbr
   fig = plot_tbr(res)

Verification
------------

TBR is cross-validated against Google's reference implementation,
`google/matched_markets <https://github.com/google/matched_markets>`_, on the
reference's own panel, and against the paper's Section 5.2 simulation. The study
is `benchmarks/studies/tbr_geo
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/studies/tbr_geo>`_.

Every reported quantity agrees with the reference: the pretest coefficients and
residual variance to the digit, the cumulative response effect to 6.4e-10, the
cumulative cost effect to 7.3e-12, and the iROAS point estimate to 7.1e-15 with
its interval to 2.4e-12. On the paper's own simulation design, the 90% and 50%
posterior intervals attain 0.8999 and 0.4981 coverage over 36 cells at 2000
replications each.

The study also records what Section 5.2's squared-bias-over-MSE figure measures.
For any unbiased estimator that statistic has expectation :math:`1/n`, and it
tracks that floor across a sixteenfold range of replication counts, landing on
the published 0.04% at the paper's own 2000. The estimator is consistent with
being unbiased; the figure reports the replication count.

Core API
--------

.. autoclass:: mlsynth.TBR
   :members:

.. autoclass:: mlsynth.config_models.TBRConfig
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.tbr_helpers.structures.TBRResults
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.tbr_helpers.structures.TBRFit
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.tbr_helpers.structures.CumulativeEffect
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.tbr_helpers.structures.IROASResult
   :members:
   :undoc-members:

References
----------

.. [Kerman2017] Kerman, J., Wang, P., & Vaver, J. (2017). *Estimating Ad
   Effectiveness using Geo Experiments in a Time-Based Regression Framework.*
   Google.

.. [Au2018] Au, T. C. (2018). *A Time-Based Regression Matched Markets Approach
   for Designing Geo Experiments.* Google LLC.

.. [VaverKoehler2011] Vaver, J., & Koehler, J. (2011). *Measuring Ad
   Effectiveness Using Geo Experiments.* Google.
