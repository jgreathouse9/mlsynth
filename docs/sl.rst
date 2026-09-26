Synthetic Learner (SL)
======================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Every estimator in this library predicts what would have happened to one treated
unit had the treatment not arrived, and each does it by committing to an
assumption. :class:`VanillaSC` assumes a convex combination of control units
tracks the treated one. :class:`FDID` assumes parallel trends on a selected
subset. :class:`GSYNTH` assumes a low-dimensional factor structure. If the
assumption holds the answer is good, and if it does not the answer is wrong
without saying so.

SL, from Viviano and Bradic [VB2023]_, declines the commitment. It keeps a
library of predictors, called experts, fits each on part of the pre-treatment
window, and then scores them against the part they did not see. The prediction is
a weighted combination, with weight going to whichever experts forecast the
held-out stretch well. Nothing in the procedure requires any single expert to be
correctly specified.

The default library is the authors' own four: a cross-validated lasso of the
treated outcome on the donors, a one-factor model of the donor pool loaded onto
the treated unit, a random forest that also reads any covariates, and
difference-in-differences over the whole donor pool. The last of these is the
conventional estimate :class:`FDID` reports beside its forward one, and SL calls
that fit instead of keeping a second copy of the formula.

The forest is the only member with an information set of its own, so covariates
reach it and nothing else. ``covariates`` names panel columns and gives the
forest every unit's path for each; ``external_covariates`` takes a frame of
series for units that are not in the panel at all, aligned by period label. The
second exists because the first cannot express the authors' own application,
which hands the forest employment for 50 states against a donor pool of six.

The inference is where SL differs most from its neighbours. It supplies a
hypothesis test of the no-effect null, with a critical value from a moving-block
bootstrap and a size guarantee that does not lean on the experts being right. It
does not supply a standard error or a confidence interval, and neither does the
paper. See :ref:`sl-inference`.

Reach for it when

- one unit is treated and the rest are not;
- you can name several plausible ways to build the counterfactual and cannot
  defend a choice among them;
- what you want is a defensible test of whether anything happened, more than a
  precise interval around how much;
- the pre-treatment window is long enough to split in two and still leave both
  halves usable. Roughly twenty periods on each side is the regime the paper
  works in; much less and neither the experts nor the weights have anything to
  learn from.

Reach for something else when the pre-window is short, since the split costs
half of it, or when you need an interval, not a test.

Notation
--------

Let :math:`i = 1, \dots, N` index units and :math:`t = 1, \dots, T` index
periods. Unit 1 is treated from :math:`T_0 + 1` onward and units
:math:`2, \dots, N` never are. Write :math:`y_{t}` for the treated unit's
outcome, :math:`\mathbf{x}_t = (y_{2t}, \dots, y_{Nt})'` for the control
outcomes, and :math:`y^0_t` for the treated unit's untreated outcome, observed
up to :math:`T_0` and to be predicted after.

The pre-treatment window splits at :math:`T_1 < T_0`. Periods
:math:`1, \dots, T_1` are the training window :math:`\mathcal{T}`; periods
:math:`T_1 + 1, \dots, T_0` are the weighting window :math:`\mathcal{W}`.

An expert is a map :math:`k \mapsto \hat{m}_k`, fit on :math:`\mathcal{T}` alone,
that returns a predicted path :math:`\hat{m}_k(t)` for every :math:`t`. With
:math:`K` experts, collect them as :math:`\hat{\mathbf{m}}(t) \in \mathbb{R}^K`.
The ensemble carries weights :math:`\mathbf{w}` on the simplex, and its
prediction is :math:`\hat{y}^0_t = \mathbf{w}' \hat{\mathbf{m}}(t)`.

Assumptions
-----------

1. One treated unit, and the controls are never treated.

   Remark. SL reads the treated unit and the control pool from the panel through
   :func:`~mlsynth.utils.datautils.dataprep`. A control contaminated by the
   treatment enters the experts' training data as though it were clean, and every
   expert inherits the contamination, so the ensemble cannot detect it.

2. The pre-treatment window splits into two usable halves, and no expert sees the
   weighting window while fitting.

   Remark. This is the whole basis of the weighting. Equation 12 scores an expert
   by its loss on :math:`\mathcal{W}`, which is informative only if that loss is
   out of sample. An expert that chose a hyperparameter by cross-validating over
   the entire pre-window has already seen :math:`\mathcal{W}`, and its score
   flatters it. Every hyperparameter in this implementation is pinned by the
   configuration or derived from :math:`\mathcal{T}` alone.

3. No expert has to be correctly specified.

   Remark. This is what the method buys. The regret bound of Lemma C.1 measures
   the ensemble against the best single expert in the library, not against the
   truth, so the guarantee survives every expert being misspecified. What the
   bound does not promise is that the best expert is any good. If the library
   spans nothing close to the counterfactual, SL returns a confident weighted
   average of wrong answers.

4. Under the null of no effect, the periods outside the training window are
   exchangeable in blocks.

   Remark. Algorithm 2 builds the null distribution by resampling blocks of those
   periods, so this is the assumption the size guarantee of Theorem 3.1 rests on.
   A block length of 3, the paper's choice, absorbs serial correlation up to
   roughly that horizon and no further. A panel with a trend that survives the
   split violates it, and the test will over-reject.

5. The effect is constant enough over the measured window that its mean is worth
   reporting.

   Remark. Equation 10's estimate is an average over the post window, so an
   effect that reverses sign inside it averages toward zero while the test
   statistic, a sum of squares, still fires. Divergence between a small
   ``att`` and a significant ``p_value`` is that case, and ``post_skip`` is how
   to measure away from the switch-on.

.. _sl-inference:

Inference and Diagnostics
-------------------------

The test statistic of Equations 7 and 8 is

.. math::

   S = \frac{1}{\sqrt{|\mathcal{P}|}}
       \sum_{t \in \mathcal{P}} \bigl(\hat{y}^0_t - y_t\bigr)^2 ,

over the measured post window :math:`\mathcal{P}`. Algorithm 2 resamples blocks
of the periods outside the training window, refits the weights inside each
replicate, and recomputes :math:`S`; the upper quantiles of those draws are the
critical values, and the share of draws at or above the observed :math:`S` is the
p-value.

:math:`S` is non-negative and quadratic, so its null quantiles are critical
values and not interval endpoints. The result therefore reports
``inference.p_value`` and ``fit.critical_values``, and leaves
``effects.att_std_err``, ``inference.standard_error`` and the interval fields
empty. That is not an omission in the implementation: the method has no standard
error. An interval for the effect would come from inverting this test over a grid
of candidate constant effects, which is separate work and is not offered here.

Three diagnostics on the result are not in the paper, and each answers a question
the point estimate cannot.

``effective_k`` is the perplexity of the weights, between 1 and :math:`K`. At 1
the weighting picked a single expert; at :math:`K` it is the simple average. The
learning rate decides where a fit lands, and the paper's own
:math:`\eta = 1 / (\sqrt{T}\,\widehat{\operatorname{Var}}(y))` lands near
:math:`K`: on its own application ``effective_k`` is 3.69 of 4, and
concentrating on the best expert there needs :math:`\eta` between 17 and 60 times
larger. An SL fit reported without this number does not say whether the
ensembling did anything.

``error_participation_ratio`` is the effective number of independent directions
in the experts' errors on :math:`\mathcal{W}`, also between 1 and :math:`K`.
Averaging cancels error when the members err independently. Measured on two
panels this ratio is 1.19 and 1.03 of 4, so they do not: the members miss in the
same direction at the same times, and an equal-weight ensemble of four
donor-projection experts comes out at 1.17 and 2.92 times the best single
member's in-window error. A ratio near 1 says the library is one expert wearing
several hats, and adding more of the same kind will not help.

``degenerate_experts`` names a member that is constant on :math:`\mathcal{W}`, so
contributes only a level, or one fitting an order of magnitude worse than the
best while still carrying weight. The second case is real: on Prop 99 an l2
expert fits 51 times worse than the best and keeps 14 percent of the weight,
because :math:`\eta` is too small to discriminate, and that is what drags the
ensemble's estimate away from every member that fits.

``dropped_experts`` names a member that could not be built at all, with the
reason. A single-donor panel drops the factor expert, for instance. A dropped
expert changes the library and therefore the estimate, so it is recorded, not
silently absent.

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import SL

   rng = np.random.default_rng(0)
   T, T0, N = 60, 40, 7
   f = rng.standard_normal((T, 2))
   lam = rng.standard_normal((N, 2))
   Y = 10.0 + f @ lam.T + 0.5 * rng.standard_normal((T, N))
   Y[T0:, 0] -= 3.0                       # the effect to be recovered

   units = np.repeat(np.arange(N), T)
   times = np.tile(np.arange(T), N)
   panel = pd.DataFrame({
       "unit": units, "time": times, "y": Y.T.ravel(),
       "D": ((units == 0) & (times >= T0)).astype(int),
   })

   res = SL({
       "df": panel, "outcome": "y", "treat": "D",
       "unitid": "unit", "time": "time",
       "n_boot": 2000, "display_graphs": False,
   }).fit()

   print(res.effects.att)                       # about -3
   print(res.fit.p_value, res.fit.critical_values[0.05])
   print(res.fit.effective_k, res.fit.error_participation_ratio)
   print(res.fit.weights)
   print(res.fit.degenerate_experts or "library healthy")

The plotter returns its figure and does not display it:

.. code-block:: python

   from mlsynth.utils.sl_helpers import plot_sl

   fig = plot_sl(res)
   fig.savefig("sl.png", dpi=150)

Verification
------------

Two pinned cases, and the full account is on :doc:`replications/sl`.

`benchmarks/cases/sl.py <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/sl.py>`_
is Path B: the size and power of the paper's own test on the paper's own Monte
Carlo design.

`benchmarks/cases/sl_tennessee.py <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/sl_tennessee.py>`_
is a cross-validation against an R implementation of the authors' own expert
library on their empirical panel. On the three experts that are algorithmically
determined the two implementations agree to 2.3e-12, 4.9e-11 and 7.0e-07 path by
path, and the effect to 3.9e-06 at every horizon. That last figure is a floor set
by the lasso solution's non-uniqueness at the penalty their grid selects, not a
target.

Two corrections to the authors' replication code are part of what is validated:
their bootstrap refits the ensemble with a different learning rate than the point
estimate, and their lasso expert's penalty is drawn from the random number
generator instead of being computed from the data. Two defects in this port are
also pinned there, both found by the cross-validation and by nothing else -- the
paper's two penalty grids, which it did not carry, and glmnet's default
standardization of the design, which it did not do.

Core API
--------

.. autoclass:: SL
   :members:

.. autoclass:: mlsynth.config_models.SLConfig
   :members:

.. autoclass:: mlsynth.utils.sl_helpers.structures.SLResults
   :members:

.. autoclass:: mlsynth.utils.sl_helpers.structures.SLFit
   :members:

.. autofunction:: mlsynth.utils.sl_helpers.weights.exponential_weights

.. autofunction:: mlsynth.utils.sl_helpers.weights.effective_k

.. autofunction:: mlsynth.utils.sl_helpers.experts.build_experts

.. autofunction:: mlsynth.utils.sl_helpers.inference.block_bootstrap_test

.. autofunction:: mlsynth.utils.sl_helpers.diagnostics.participation_ratio

.. autofunction:: mlsynth.utils.sl_helpers.plotter.plot_sl

References
----------

.. [VB2023] Viviano, D., & Bradic, J. (2023). Synthetic learner: model
   evaluation with limited overlap. *Journal of Econometrics*, 234(2), 691-713.
   https://doi.org/10.1016/j.jeconom.2022.07.005
