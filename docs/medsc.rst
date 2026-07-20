Mediation Analysis Synthetic Control (MEDSC)
============================================

.. currentmodule:: mlsynth

When to Use This Method
-----------------------

You have run a synthetic control and measured the total effect of an
intervention on a treated unit, and now you want to know why the effect
happened — how much of it ran through one particular channel you can observe,
and how much did not. Concretely, you observe a mediator: a second series that
the intervention plausibly moved, and that in turn moves the outcome. For
California's Proposition 99, the outcome is per-capita cigarette sales, the
intervention is the 1989 tobacco-control program, and the natural mediator is
the retail price of cigarettes: the program raised the price (through its
excise tax), and a higher price lowers sales. Part of the total drop in smoking
is this price channel; the rest is everything else the program did — media
campaigns, clean-air laws, changed norms. A total effect alone cannot separate
the two.

MEDSC, following Mellace and Pasquini ([MellacePasquini2022]_), splits the
synthetic-control effect into a direct effect and an indirect effect that runs
through the mediator, using the counterfactual machinery of synthetic control
rather than the linear structural equations of classical mediation analysis.
The key device is a cross-world synthetic control: alongside the ordinary
synthetic control that reproduces the treated unit's pre-treatment outcome
path, MEDSC builds a second control that also matches the treated unit's
post-treatment mediator path. This second control answers a counterfactual
question — what would the outcome have been under no treatment but with the
mediator forced to its treated value — and the gap between the two controls
isolates the channel.

Reach for MEDSC when
^^^^^^^^^^^^^^^^^^^^^

* you have a single treated unit, a donor pool, and a clean pre/post
  intervention, exactly the setting of an ordinary synthetic control;
* you also observe a mediator — a channel variable the intervention moves and
  that moves the outcome — over the same panel;
* the question is mechanism, not just size: how much of the effect flows
  through that channel;
* you can assemble a donor pool wide enough that some donors reach the treated
  unit's post-treatment mediator values (see Assumption 4).

Do not use MEDSC when
^^^^^^^^^^^^^^^^^^^^^^^

* you only care about the total effect — an ordinary synthetic control
  (:doc:`vanillasc`) is simpler and makes fewer assumptions;
* no donor comes close to the treated unit's post-treatment mediator path, so
  the cross-world control has to extrapolate far outside the donor range;
* the mediator is measured with substantial error or is realised after the
  outcome, so the ordering "treatment moves mediator moves outcome" is not
  credible.

Notation
--------

Let :math:`i = 1` denote the treated unit and :math:`i = 2, \dots, n` the
donors, observed for :math:`t = 1, \dots, T` with the intervention at
:math:`T_0 + 1`. Write :math:`Y_{it}` for the outcome and :math:`M_{it}` for the
mediator. Under the treatment the treated unit realises the outcome
:math:`Y_{1t}` and the mediator :math:`M_{1t}(1)`; the estimands are three
counterfactual contrasts against its untreated potential outcomes.

The total effect compares the observed outcome to the no-treatment
counterfactual :math:`\widehat Y^{0,M0}_{1t}`, the outcome the treated unit
would have had with no treatment and its own untreated mediator:

.. math::

   \widehat\tau^{\text{tot}}_t = Y_{1t} - \widehat Y^{0,M0}_{1t}.

The direct effect compares the observed outcome to a cross-world
counterfactual :math:`\widehat Y^{0,M1}_{1t}`, the outcome under no treatment
but holding the mediator at its treated post-treatment value
:math:`M_{1t}(1)`:

.. math::

   \widehat\tau^{\text{dir}}_t = Y_{1t} - \widehat Y^{0,M1}_{1t}.

Because the mediator is fixed at its treated path, whatever the direct
contrast leaves is the effect not routed through the mediator. The indirect
effect is the remainder,

.. math::

   \widehat\tau^{\text{ind}}_t
     = \widehat\tau^{\text{tot}}_t - \widehat\tau^{\text{dir}}_t
     = \widehat Y^{0,M1}_{1t} - \widehat Y^{0,M0}_{1t},

the difference between the two synthetic controls, which is the movement in the
counterfactual outcome attributable purely to forcing the mediator to its
treated value. The reported ATTs are the post-period means of each path.

Estimation
----------

The total effect is an ordinary synthetic control: donor weights
:math:`\mathbf{w}^{\text{tot}}` on the simplex reproduce the treated unit's
pre-treatment outcome path, and
:math:`\widehat Y^{0,M0}_{1t} = \sum_{i} w^{\text{tot}}_i Y_{it}`.

The direct effect is a cross-world synthetic control, re-estimated separately
for each post period :math:`t' > T_0`. Its donor weights match two blocks at
once: the treated unit's pre-treatment outcome path (as for the total effect)
and the treated unit's post-treatment mediator path up to :math:`t'`,
:math:`\{M_{1s}(1)\}_{T_0 < s \le t'}`. The predictor weighting places
``pre_weight`` (default :math:`3/4`) on the pre-treatment block and the
remaining :math:`1/4` equally across the post-treatment mediator constraints,
following the paper. The counterfactual
:math:`\widehat Y^{0,M1}_{1t'} = \sum_i w^{\text{dir}}_i(t') Y_{it'}` is then
read off at :math:`t'`. Re-estimating per period lets the mediator match tighten
as the post-treatment mediator path lengthens.

With no covariates (the default) the pre-treatment block is the full
pre-treatment outcome path and the weights are a simplex least-squares fit; this
is the specification under which the paper's Proposition 99 decomposition
reproduces. With covariates the predictor weights are cross-validated by the
bilevel global search (the ``mscmt`` backend, shared with :doc:`vanillasc`),
which chooses how to trade off outcome lags, pre-treatment mediator lags and
covariates.

Assumptions
-----------

1. Sequential ignorability of the mediator. Conditional on the donor pool
   spanning the untreated outcome, the mediator is as good as randomly assigned
   with respect to the outcome — the cross-world outcome
   :math:`Y^{0,M1}` is identified by matching the mediator path.

   Remark. This is the mediation-analysis analogue of the no-unmeasured-
   confounder condition, transposed to the synthetic-control setting: there is
   no third variable that moves both the mediator and the outcome and is left
   unmatched. It is the strong assumption of any mediation decomposition and
   cannot be tested from the data alone.

2. Synthetic-control validity. The treated unit's untreated outcome path lies
   in (or near) the convex hull of the donors, so both the total and the
   cross-world controls reproduce it without extrapolating.

   Remark. This is the standard Abadie condition, applied twice — once for the
   total control and once for the mediator-matched control. A poorly fitting
   pre-treatment path invalidates the decomposition just as it would a plain
   synthetic control.

3. Correct ordering. The treatment moves the mediator, which moves the outcome;
   the mediator is realised before the outcome in each period.

   Remark. If the outcome feeds back into the mediator within a period, the
   "through the mediator" language is not well defined and the split is not
   interpretable.

4. A mediator-spanning donor pool. Some donors attain mediator values close to
   the treated unit's post-treatment mediator path, so the cross-world control
   can match it on the simplex.

   Remark. This is why MEDSC exposes two donor pools. The treated unit's
   mediator often jumps at the intervention to a value the ordinary donor pool
   never reaches (California's post-1989 price exceeded every classic donor
   state's), so the direct pool adds units — the high-mediator states the total
   pool excludes — to bracket it. Without such donors the mediator match must
   extrapolate and the direct effect is unreliable.

Inference and diagnostics
-------------------------

Inference is Abadie's in-space placebo test on the total effect: each donor is
treated as a pseudo-treated unit, its own synthetic control is refit on the
remaining donors, and the treated unit's post-to-pre RMSPE ratio is ranked
against the placebo ratios. Placebo units whose pre-treatment fit is poor —
RMSPE above ``placebo_cutoff`` times the treated unit's (default five) — are
dropped, and the p-value is the treated unit's rank among the survivors. The
placebo test speaks to the total effect, the identified estimand; the direct
and indirect channels are reported as point paths, and their credibility rests
on the mediation assumptions above rather than on a sampling distribution.

The pre-treatment RMSE of the total control (``pre_rmse_total``) is the primary
fit diagnostic — the decomposition is only as trustworthy as the synthetic
control it rests on.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import MEDSC

   # Proposition 99: does the tobacco-control effect run through price?
   df = pd.read_csv("basedata/prop99_mediation.csv")
   program = ["Massachusetts", "Arizona", "Oregon", "Florida",
              "District of Columbia"]                 # other tobacco programs
   tax = ["Alaska", "Hawaii", "Maryland", "Michigan", "New Jersey",
          "New York", "Washington"]                   # high-tax states
   allstates = sorted(df["state"].unique())
   direct_pool = [s for s in allstates if s not in ["California"] + program]
   total_pool = [s for s in direct_pool if s not in tax]
   df = df[df["state"].isin(["California"] + direct_pool)].copy()
   df["treated"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)

   res = MEDSC({
       "df": df, "outcome": "cigsale", "mediator": "price", "treat": "treated",
       "unitid": "state", "time": "year",
       "total_donors": total_pool,       # 38 classic Abadie controls
       "direct_donors": direct_pool,     # + 7 high-price states to span the mediator
       "display_graphs": False,
   }).fit()

   res.att                               # total effect (packs per capita)
   res.att_direct                        # direct channel
   res.att_indirect                      # indirect (price) channel
   res.decomposition.direct              # per-period direct path
   res.inference.p_value                 # placebo p-value on the total effect

The headline ``att`` is the total effect; ``att_direct`` and ``att_indirect``
are the two channels, and ``decomposition`` carries the per-period paths and
both counterfactuals. The total-effect donor weights are on ``weights``; the
direct fit's final-period weights are in its summary statistics.

Verification
------------

MEDSC reproduces the Mellace-Pasquini Proposition 99 decomposition on the CDC /
Orzechowski-Walker Tax Burden on Tobacco data (``basedata/prop99_mediation.csv``).
The novel cross-world direct effect matches the paper's Table 1 nearly cell for
cell — :math:`-16.8` in 1995 (paper :math:`-16.77`) and :math:`-18.0` in 2000
(paper :math:`-17.28`) — and the indirect price channel reproduces its
qualitative signature, opening at roughly zero in 1989 and growing negative
thereafter (durable case ``benchmarks/cases/medsc_prop99.py``). See
:doc:`replications/medsc` for the full comparison, including why the indirect
channel's magnitude tracks the outcome-path total rather than the paper's
predictor-tuned total.

Core API
--------

.. autoclass:: MEDSC
   :members: fit

.. autoclass:: mlsynth.config_models.MEDSCConfig
   :members:

References
----------

.. [MellacePasquini2022] Mellace, G., & Pasquini, A. (2022). "Identification and
   estimation of mediation effects with a synthetic control." Bank of Italy
   Working Paper (and arXiv:1912.12073).
