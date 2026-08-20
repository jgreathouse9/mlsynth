Time Varying Parameter Bayesian Lasso (BLTVP)
=============================================

.. currentmodule:: mlsynth

When to use BLTVP -- and when not to
------------------------------------

Synthetic control builds a counterfactual for one treated unit as a weighted
combination of untreated units, and it fixes those weights once, for the whole
window. That is a cross-sectional answer to a time-series question: each
pre-treatment period is treated as an interchangeable observation, so shuffling
the years would give the same weights.

Often the relationship between the treated unit and its donors does not hold
still. An oil shock changes which economies resemble each other. A competitor
enters one market and not another. A donor's industrial mix drifts over thirty
years. When that happens, a single fixed weight cannot fit the pre-period well,
and a counterfactual built on a poor pre-period fit is not trustworthy.

BLTVP lets each donor's weight move over time, but does not force it to. Every
coefficient is split into two halves --- a level that stays put and a
random-walk component that drifts --- and each half is shrunk separately. The
model then decides, donor by donor, which of four descriptions fits: the
relationship is drifting, it is static and non-zero, it drifts around zero, or
the donor is irrelevant.

Reach for BLTVP when the pre-treatment fit of an ordinary synthetic control is
poor and you suspect the donor relationships are not stable, when the panel is
long enough for drift to be visible (monthly or long annual series), and when
you want the model to tell you which donors moved.

Do not reach for it when the pre-period is short. A random-walk coefficient
needs transitions to be informed by the data, and with a handful of
pre-treatment periods the shrinkage will collapse the dynamics to nothing while
you pay for the extra parameters in interval width. Do not reach for it when
you need weights that are non-negative and sum to one: BLTVP places no simplex
constraint, so its weights can be negative and its counterfactual can
extrapolate outside the donor hull. If interpretable convex weights matter more
than pre-period fit, use :doc:`vanillasc` or :doc:`masc`.

The alternative BLTVP is arguing against
-----------------------------------------

Making every coefficient dynamic is already possible --- ``CausalImpact``
(:doc:`cmbsts` in mlsynth) offers it. The paper's objection is that forcing all
coefficients to be dynamic when only some are misspecifies the model, and the
misspecification shows up as credibility intervals so wide that almost no
effect can be rejected. In the paper's own simulation that alternative produces
intervals 222 percent wider than BLTVP's.

The fix is the split. Because the drifting half of each coefficient can be
shrunk to zero independently of the level, the model can describe a mostly
static donor pool with one or two moving relationships, which is what applied
panels tend to look like.

Notation and the causal estimand
---------------------------------

Index units by :math:`j` and time by :math:`t`. Unit :math:`1` is treated from
:math:`T_0 + 1` onward; units :math:`j \in \{2, \dots, J+1\}` are the donors,
untreated throughout. Write :math:`x_{j,t}` for donor :math:`j`'s outcome at
time :math:`t` and :math:`y_t` for the treated unit's.

Let :math:`y_t(0)` be the outcome the treated unit would have had without the
intervention. The per-period effect and the average effect are

.. math::

   \tau_t = y_t - y_t(0),
   \qquad
   \Delta_\tau = \frac{1}{T - T_0} \sum_{t > T_0} \tau_t .

Only :math:`y_t` is observed after treatment, so estimating :math:`\tau_t`
means imputing :math:`y_t(0)`.

The model
---------

BLTVP writes the untreated potential outcome as a regression on the donors
whose coefficients evolve:

.. math::

   y_t(0) = \sum_{j} \beta_{j,t}\, x_{j,t} + \epsilon_t,
   \qquad \epsilon_t \sim N(0, \sigma^2),

with each coefficient decomposed into a fixed part and a scaled random walk,

.. math::

   \beta_{j,t} = \beta_j + \sqrt{\theta_j}\, \tilde\beta_{j,t},
   \qquad
   \tilde\beta_{j,t} = \tilde\beta_{j,t-1} + \tilde\eta_{j,t},
   \qquad \tilde\eta_{j,t} \sim N(0,1).

Here :math:`\beta_j` is the time-invariant level of donor :math:`j`'s weight
and :math:`\sqrt{\theta_j}` scales how far that weight wanders. This is the
noncentered parameterisation: the drift is standardised to unit variance and
all the scale information sits in :math:`\sqrt{\theta_j}`.

The step that makes the method work is writing that scale as a signed square
root, so :math:`\sqrt{\theta_j}` ranges over the whole real line. Zero is then
an interior point of its prior and can carry mass. Placing an inverse gamma on
:math:`\theta_j` directly --- the conventional choice --- puts zero at the edge
of the support, where no prior mass can sit, and every coefficient is forced to
retain some drift. That is the difference between a model that can switch the
dynamics off per donor and one that cannot.

Independent Bayesian Lasso priors are placed on the two blocks,

.. math::

   \beta_j \mid \alpha_j^2 \sim N(0, \alpha_j^2), \quad
   \alpha_j^2 \mid \lambda^2 \sim \mathrm{Exp}(\lambda^2/2), \quad
   \lambda^2 \sim \mathrm{Gamma}(z_1, z_2),

and the same hierarchy, with its own rate, on :math:`\sqrt{\theta_j}`. Each
donor then falls into one of four categories, read off the pair
:math:`(\beta_j, \sqrt{\theta_j})`: both non-zero (an unrestricted drifting
weight), only :math:`\beta_j` non-zero (a static weight), only
:math:`\sqrt{\theta_j}` non-zero (a weight drifting around zero), or both
shrunk (an irrelevant donor).

Two nestings follow directly. Setting :math:`\sqrt{\theta_j} = 0` for all
:math:`j` gives a Bayesian Lasso synthetic control with static weights, which
``time_varying=False`` fits. Setting :math:`\sqrt{\theta_j} = 0` and swapping
the Lasso for a horseshoe prior gives :doc:`bscm`.

Assumptions
-----------

1. Conditional independence. The pre-treatment outcomes of the treated unit
   and the donors carry enough information about unobserved confounders to
   impute :math:`y_t(0)` after treatment.

   Remark. This is what licenses reading a close pre-treatment fit as evidence
   that the data-generating process has been captured. It fails if a confounder
   appears only after treatment.

2. No spillovers. The donors' outcomes are unaffected by the treated unit's
   treatment.

   Remark. If donors absorb some of the effect, the counterfactual is
   contaminated and the estimate is biased in a direction the model cannot
   detect. Use :doc:`spillsynth` when interference is the concern.

3. Random-walk coefficients. The drift in each donor relationship follows a
   random walk.

   Remark. Klinenberg is explicit (sec. 2.2) that this choice is made for its
   forecasting performance and not from theory. A random walk can track a level
   shift or a slow trend; it will not capture a seasonal or cyclical
   relationship, and it lets the coefficient wander without bound over long
   horizons.

4. Gaussian errors and correctly specified priors. The observation noise is
   normal and the shrinkage priors describe the coefficients.

   Remark. There is no safeguard against a misspecified prior. The paper's own
   recommendation (sec. 7) is to refit under a different shrinkage prior --- a
   horseshoe or double gamma --- and check that the answer holds.

Inference and diagnostics
-------------------------

Uncertainty comes from the posterior. The sampler draws from the posterior
predictive distribution of :math:`y_t(0)` at every period, so the reported
bands include observation noise as well as parameter uncertainty. The ATT
credible interval is the corresponding quantile of the per-draw average gap.

The bands widen after :math:`T_0`, and they should: the coefficient path is
unobserved once treatment begins, so the random walk diffuses and the
counterfactual becomes less certain the further out it is carried. A BLTVP
interval that did not widen would mean the dynamics had been shrunk away.

The pre-treatment RMSE is not by itself evidence of a good model. With many
donors and a short pre-period, the model carries two free coefficients per
donor against few observations, and a near-zero pre-period residual is what an
overparameterised regression does. Read it together with the interval width and
the sensitivity check in assumption 4.

The drift scales, reported per donor in ``dynamic_scales``, are the posterior
mean of :math:`|\sqrt{\theta_j}|`. They say which relationships the model
judged to be moving. Because the Bayesian Lasso shrinks continuously and never
sets a coefficient exactly to zero (paper, p. 1068), these are magnitudes and
not inclusion probabilities, and mlsynth reports them as such. For per-donor
inclusion probabilities, :doc:`bscm` with ``prior="spike_slab"`` is the
estimator that supplies them.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import BLTVP

   # Proposition 99: California treated in 1989, 38 donor states,
   # 19 pre-treatment years (1970-1988).
   df = pd.read_csv("basedata/smoking_data.csv")
   df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)

   res = BLTVP({
       "df": df, "outcome": "cigsale", "unitid": "state",
       "time": "year", "treat": "treat",
       "n_iter": 10000, "burn_in": 5000, "seed": 2019,
       "display_graphs": False,
   }).fit()

   lo, hi = res.att_ci
   print(f"pre-period RMSE : {res.pre_rmse:.3f}")
   print(f"ATT             : {res.att:+.2f}  [{lo:+.2f}, {hi:+.2f}]")

   drift = res.dynamic_scales
   for k in sorted(drift, key=drift.get, reverse=True)[:3]:
       print(f"  drifts most : {k:<14} scale={drift[k]:.3f}")

which prints

.. code-block:: text

   pre-period RMSE : 0.040
   ATT             : -19.18  [-51.17, +11.65]
     drifts most : Utah           scale=0.011
     drifts most : New Mexico     scale=0.009
     drifts most : Nebraska       scale=0.009

The estimated reduction is about 19 packs per capita, and the credible interval
covers zero, so on this evidence the effect is not statistically distinguishable
from none. That is the paper's own reading of Proposition 99 (sec. 6), and it
agrees with the synthetic difference-in-differences estimate of Arkhangelsky et
al. (2021) while disagreeing with Abadie, Diamond and Hainmueller (2010). The
proposed explanation is structural breaks in the control group, which violate
the linear-factor model the original estimate rests on.

The pre-period RMSE of 0.040 against a series averaging 116 packs is the
overparameterisation described above --- 38 donors, 76 coefficients, 19
pre-treatment observations --- and is why the interval, not the fit, carries the
information here.

Verification
------------

BLTVP reproduces Klinenberg (2023) Table 2 on Proposition 99. Over four chains
of 10,000 draws at the author's own settings the average reduction comes to
17.78 against the published 17.7, with credible bounds of -16.65 and 52.28
against -16.1 and 51.7; all three agree within Monte Carlo error
(:math:`|z| \le 1.22`). See the replication page :doc:`replications/bltvp` and
the durable case ``benchmarks/cases/bltvp_prop99.py``. The sampler, setup,
inference, plotter and result contract are unit-tested
(``mlsynth/tests/test_bltvp.py``, full coverage).

Core API
--------

.. automodule:: mlsynth.estimators.bltvp
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.BLTVPConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``BLTVP.fit()`` returns a
:class:`~mlsynth.utils.bltvp_helpers.structures.BLTVPResults` --- an
``EffectResult`` whose standardized sub-models carry the ATT, counterfactual,
gap and pre-RMSE, with the posterior draws, per-draw ATT samples,
counterfactual credible bands, the time-invariant weights and the per-donor
drift scales on the typed fields. The prepared NumPy panel is exposed as a
:class:`~mlsynth.utils.bltvp_helpers.structures.BLTVPInputs`.
