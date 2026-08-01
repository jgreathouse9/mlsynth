Distribution-Regression Synthetic Control (DRSC)
================================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Most synthetic control methods return one number per period: the treated unit's
outcome minus its synthetic counterpart. That is the right summary when the
policy shifts everyone in the same direction. It is the wrong summary when the
policy reshapes the outcome distribution -- a minimum wage that lifts the bottom
tail and leaves the median alone, a transfer that compresses inequality without
moving the mean.

:doc:`dsc` already answers part of that: it matches whole distributions instead
of means, and returns quantile treatment effects. But it works on the
*unconditional* distribution, the one you get by pooling everybody. If the
outcome depends on individual characteristics -- and wages depend heavily on
education and experience -- then pooling can hide the effect entirely. An
increase that raises wages for young low-education workers and nobody else
barely moves the overall wage distribution, because that group is a small slice
of it.

DRSC conditions. It estimates the counterfactual distribution of the outcome
*given* covariates, so you can ask what the policy did to a 25-year-old with a
high-school education specifically, rather than to the workforce on average.

Reach for it when all of these hold: you have micro-data (many individuals per
unit-period, not one aggregate number), one treated unit, and a reason to think
the effect varies across people in a way you can describe with covariates.

Notation
--------

There are :math:`J + 1` groups observed over :math:`T` periods. Group
:math:`i = 1` is treated from period :math:`T_0 + 1` onward; groups
:math:`i = 2, \ldots, J+1` are the donor pool. Within each group-period cell
:math:`(i, t)` we observe :math:`n_{it}` individuals, each with an outcome
:math:`Y` and a covariate vector :math:`x \in \mathbb{R}^{p}`.

The conditional distribution is modelled semiparametrically as

.. math::

   F_{it}(y \mid x) \;=\; \Lambda\bigl(x^{\top}\theta_{it}(y)\bigr),

where :math:`\Lambda` is a known link (the normal CDF by default) and
:math:`\theta_{it} : \mathcal{Y} \to \mathbb{R}^{p}` is an unknown parameter
function. This is the distribution regression of Foresi and Peracchi (1995).
For a fixed :math:`y` it is just a binary regression of
:math:`\mathbf{1}\{Y \le y\}` on :math:`x`, so estimation reduces to running one
probit per outcome grid point per cell.

The object of interest is the pointwise difference between the observed and
counterfactual conditional distributions,

.. math::

   \Delta_t(y \mid x) \;=\; F_{1,t}(y \mid x) - F^{0}_{1,t}(y \mid x),

and its integrated square over a region :math:`\mathcal{Y}_0`,

.. math::

   f_t(x, \mathcal{Y}_0) \;=\; \int_{\mathcal{Y}_0} \Delta_t(y \mid x)^2 \, dy .

Setting :math:`\mathcal{Y}_0` to the whole support gives the total effect;
restricting it to a policy-relevant window -- the corridor between the old and
new minimum wage -- concentrates the test where the policy actually operates.

Assumptions
-----------

Assumption 1 (Parallel trends in parameters). There exist weights
:math:`w_2, \ldots, w_{J+1}` summing to one such that, for all
:math:`y \in \mathcal{Y}_0` and all post-treatment :math:`t`,

.. math::

   \theta^{0}_{1,t}(y) - \theta_{1,T_0}(y)
   \;=\; \sum_{i=2}^{J+1} w_i \bigl[\theta_{i,t}(y) - \theta_{i,T_0}(y)\bigr].

Remark. This is the familiar parallel-trends idea moved into the parameter
space of the model rather than imposed on the outcome or the CDF. That choice
does real work. The parameter function is linear in the common factors whereas
the CDF is not, so the weights come out constant across the whole distribution,
and the counterfactual is guaranteed to still be a valid distribution in the
model class. Economically it says the treated unit's exposure to aggregate
shocks -- skill-biased technical change, the business cycle -- can be written as
a weighted blend of the donors' exposures.

Assumption 2 (Sampling). Within each cell the individuals are i.i.d., and cells
are independent, with :math:`n_{it}/n \to r_{it} \in (0, \infty)`.

Remark. The asymptotics run in :math:`n`, the number of individuals, with the
donor count and the number of periods held fixed. This is unlike every other
estimator in mlsynth, where precision comes from a long panel. Here one
pre-period is enough for consistency, and extra pre-periods buy precision in the
weights rather than in the distribution regressions. The practical consequence:
worry about thin cells, not short panels.

Assumption 3 (Pre-treatment balance). The treated unit's pre-treatment
parameter function lies in the affine span of the donors'.

Remark. The functional analogue of the usual requirement that the treated unit
sit inside the donor hull. It is only affine, not convex -- see the note on
negative weights below -- and it is testable, which is what the pre-trend tests
in the next section do.

Estimation
----------

The weights minimise the pre-treatment discrepancy, averaged over pre-periods
and integrated over the outcome grid, subject only to adding up to one:

.. math::

   \widehat w \;=\; \operatorname*{arg\,min}_{\mathbf{1}'w = 1}
   \frac{1}{T_0 m} \sum_{t=1}^{T_0} \sum_{l=1}^{m}
   \bigl\| \widehat\theta_{1t}(y_l) - \sum_{i=2}^{J+1} w_i \widehat\theta_{it}(y_l)
   \bigr\|^{2} .

Because the objective is quadratic and the single constraint is linear, this has
a closed form -- no solver, no iteration:

.. math::

   \widehat w \;=\; \widehat G^{-1}\widehat c
   \;-\; \widehat G^{-1}\mathbf{1}\,
   \frac{\mathbf{1}'\widehat G^{-1}\widehat c - 1}
        {\mathbf{1}'\widehat G^{-1}\mathbf{1}} .

Two consequences deserve stating plainly, because both look like faults and
neither is.

Negative weights are normal. Non-negativity is deliberately dropped
(Doudchenko and Imbens), which lets the synthetic unit sit outside the donor
hull and is what makes a good pre-treatment fit attainable when the treated unit
is unusual. In the New Jersey application 20 of 42 donors receive negative
weight.

The Gram matrix is ill-conditioned, and that requires float64. Donors' parameter
functions are similar to each other -- which is precisely why a weighted
combination can track the treated unit -- so :math:`\widehat G` is nearly
singular. On the New Jersey panel its condition number is
:math:`9.6 \times 10^{4}`, and the solve amplifies relative input error by
roughly :math:`10^{6}`. Passing float32 data (relative error
:math:`\sim 6 \times 10^{-8}`) moves Florida's weight from 0.515 to 0.505, and
the largest donor weight by about 0.04; random perturbation of the same
magnitude, which does not partially cancel the way rounding does, moves weights
by 0.12 to 0.17.
The *estimand* is far more stable than the weights, so the reported effects
barely move, but the weights themselves are what people quote. Keep the inputs
in float64. Setting ``ridge`` to a small positive value stabilises the solve at
the cost of shrinking the weights toward :math:`1/J`.

Inference and diagnostics
-------------------------

The null of no effect, :math:`H_0 : \Delta_t(\cdot \mid x) = 0` on
:math:`\mathcal{Y}_0`, is tested with the supremum statistic

.. math::

   T_n(x, t, \mathcal{Y}_0) \;=\; \sqrt{n} \,
   \sup_{y_l \in \mathcal{Y}_0} \bigl| \widehat\Delta_t(y_l \mid x) \bigr| ,

whose limit under the null is the supremum of a mean-zero Gaussian process.
Critical values come from simulating that process using a plug-in estimate of
its covariance kernel. The kernel carries two sources of uncertainty at the same
:math:`\sqrt{n}` rate: the distribution-regression error in the post period, and
the weight-estimation error inherited from the pre-periods.

When the test rejects, a one-sided lower confidence bound for
:math:`f_t(x)` is reported. It is only meaningful conditional on rejection:
under the null the delta-method derivative vanishes, the variance degenerates,
and the bound collapses to zero.

With two or more pre-periods, parallel trends is testable directly by treating
an earlier period as a pseudo-post period and asking whether the same machinery
finds an effect where none can exist.

One reproducibility caveat, worth knowing before comparing numbers across
machines. The simulated critical values require a matrix square root of the
estimated kernel, obtained from an eigendecomposition. Eigenvector *signs* are
implementation-defined, so a different LAPACK build produces a different factor
and hence a different realisation from identical Gaussian draws, even with the
seed fixed. The effect is a few tenths of a percent on the critical value. The
test statistic itself involves no eigendecomposition and is exact.

Example
-------

.. code-block:: python

   import numpy as np, pandas as pd
   from mlsynth import DRSC

   # micro-panel: one row per (state, policy year, worker)
   df = pd.read_parquet("benchmarks/reference/wied_nj_minwage/"
                        "nj_estimation_sample.parquet")
   df["treat"] = ((df.STATEFIP == 34) & (df.t == 4)).astype(int)
   df["x_std_sq"] = df["x_std"] ** 2

   res = DRSC({
       "df": df, "outcome": "logwage", "treat": "treat",
       "unitid": "STATEFIP", "time": "t",
       "covariates": ["e_std", "x_std", "x_std_sq"],
       "evaluation_points": {
           "low_skill_young": {"e_std": -0.98, "x_std": -1.17,
                               "x_std_sq": 1.37},
           "high_skill":      {"e_std":  1.64, "x_std":  1.44,
                               "x_std_sq": 2.08},
       },
       "focus_region": (np.log(4.25), np.log(5.10)),   # the MW corridor
       "display_graphs": False,
   }).fit()

   e = res.conditional_effects["low_skill_young"]
   e.f_hat, e.p_value, e.p_value_focused     # 1.71e-3, 0.054, 0.012
   res.gram_condition_number                  # 9.6e4 -- see the float64 note
   res.n_negative_weights                     # 20 of 42, expected

Verification
------------

Reproduced against Wied (2026), Tables 1 and 2, on the author's own CPS
estimation sample: all five donor weights and all twenty cells of Table 2 match
to the paper's printed precision, along with the active grid size
(:math:`m = 32`), the Gram condition number and the negative-weight count. See
:doc:`replications/drsc` and
`benchmarks/reference/wied_nj_minwage/
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/reference/wied_nj_minwage>`_.

Not to be confused with
-----------------------

:doc:`drosc` is Distributionally *Robust* synthetic control -- robustness of a
conventional estimate to distributional shift, a different question entirely.
:doc:`dsc` is the unconditional distributional estimator DRSC extends.

Core API
--------

.. autoclass:: DRSC
   :members: fit

.. autoclass:: mlsynth.utils.drsc_helpers.config.DRSCConfig
   :members:

.. autoclass:: mlsynth.utils.drsc_helpers.structures.DRSCResults
   :members:

.. autoclass:: mlsynth.utils.drsc_helpers.structures.ConditionalEffect
   :members:
