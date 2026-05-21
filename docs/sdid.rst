Synthetic Difference-in-Differences (SDID)
==========================================

.. currentmodule:: mlsynth

Overview
--------

Synthetic Difference-in-Differences (SDID) combines the unit-weighting
of classical synthetic control with the *time*-weighting of difference-in-
differences. Originally proposed by Arkhangelsky, Athey, Hirshberg,
Imbens, and Wager (2021, *AER*), the estimator fits two sets of weights
on the pre-treatment panel — donor-unit weights :math:`\omega_i` that
match the treated unit's pre-treatment level *trajectory*, and time
weights :math:`\lambda_t` that match the post-treatment donor *mean*
— and then computes a doubly-weighted DiD-style contrast on the
post-treatment window.

The :mod:`mlsynth` implementation exposes two distinct but linked
estimators in a single ``SDID.fit()`` call:

* The **overall ATT** of Arkhangelsky et al. (2021), aggregated across
  cohorts and post-treatment periods (Clarke, Pailanir, Athey, & Imbens,
  2023).
* The **event-study extension** of Ciccia (2024,
  `arXiv:2407.09565 <https://arxiv.org/abs/2407.09565>`_), which
  disaggregates the overall ATT into cohort-specific dynamic effects
  :math:`\hat\tau_{a, \ell}^{\,sdid}` and pools them into a single
  treated-unit-weighted event-study path :math:`\hat\tau_\ell^{\,sdid}`.

Both single-treated-unit designs (the canonical Proposition 99 setup)
and staggered-adoption designs are handled by the same orchestrator;
:func:`mlsynth.utils.datautils.dataprep` decides between the two
representations automatically.

Mathematical Formulation
------------------------

Setup
^^^^^

Let :math:`Y_{i, t}` denote the outcome of unit :math:`i` in period
:math:`t`, with :math:`i \in \{1, \dots, N\}` and
:math:`t \in \{1, \dots, T\}`. There are :math:`N_{co}` never-treated
control units, and the treated units are partitioned into cohorts by
their adoption period: cohort :math:`a` is the set
:math:`I^a \subseteq \{N_{co} + 1, \dots, N\}` of units that first
receive treatment in period :math:`a`. Let
:math:`A = \{a_1, \dots, a_K\}` denote the set of distinct adoption
periods, :math:`N_{tr}^a = |I^a|` the cohort size, and
:math:`T_{tr}^a = T - a + 1` the number of post-treatment periods in
cohort :math:`a`. Aggregate post-treatment exposure (Clarke et al.,
2023) is :math:`T_{post} = \sum_{a \in A} N_{tr}^a T_{tr}^a`.

The classical Arkhangelsky et al. (2021) SDID estimator targets a
single cohort. The mlsynth implementation runs that estimator
*per cohort*, accumulates the cohort-specific effects, and then
aggregates them in two complementary ways (Ciccia, 2024).

Cohort-Specific SDID (Equation 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a single cohort :math:`a`, SDID fits weights :math:`\omega_i` over
:math:`N_{co}` donor units and :math:`\lambda_t` over the cohort's
pre-treatment window :math:`t < a` by solving two convex programs:

.. math::

   \omega
   \;=\;
   \arg\min_{\sum \omega_i = 1,\ \omega_i \geq 0}
     \sum_{t = 1}^{a - 1}
       \left(
         \bar Y_{I^a, t}
         -
         \omega_0 - \sum_{i = 1}^{N_{co}} \omega_i Y_{i, t}
       \right)^{\!2}
     + T_0 \zeta^2 \|\omega\|_2^2,

.. math::

   \lambda
   \;=\;
   \arg\min_{\sum \lambda_t = 1,\ \lambda_t \geq 0}
     \sum_{i = 1}^{N_{co}}
       \left(
         \bar Y_{i, [a, T]}
         -
         \lambda_0 - \sum_{t = 1}^{a - 1} \lambda_t Y_{i, t}
       \right)^{\!2},

where :math:`\bar Y_{I^a, t}` is the treated-unit mean at time
:math:`t`, :math:`\bar Y_{i, [a, T]}` is donor :math:`i`'s mean over
the post-treatment window, and :math:`\zeta` is a regularization
parameter scaled by the standard deviation of first-differenced donor
outcomes. The cohort-specific SDID estimator is then

.. math::

   \hat\tau_a^{\,sdid}
   \;=\;
   \frac{1}{T_{tr}^a} \sum_{t = a}^{T}
     \left(
       \frac{1}{N_{tr}^a} \sum_{i \in I^a} Y_{i, t}
       -
       \sum_{i = 1}^{N_{co}} \omega_i Y_{i, t}
     \right)
   -
   \sum_{t = 1}^{a - 1} \lambda_t
     \left(
       \frac{1}{N_{tr}^a} \sum_{i \in I^a} Y_{i, t}
       -
       \sum_{i = 1}^{N_{co}} \omega_i Y_{i, t}
     \right).

This is Equation 2 of Ciccia (2024). Each cohort is fit independently
inside
:func:`mlsynth.utils.sdid_helpers.cohort.estimate_cohort_sdid_effects`.

Cohort-Specific Event Study (Equation 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cohort ATT is the average of a sequence of *dynamic* effects, one
per post-treatment offset :math:`\ell \in \{1, \dots, T_{tr}^a\}`:

.. math::

   \hat\tau_{a, \ell}^{\,sdid}
   \;=\;
   \frac{1}{N_{tr}^a} \sum_{i \in I^a} Y_{i, a - 1 + \ell}
   \;-\;
   \sum_{i = 1}^{N_{co}} \omega_i Y_{i, a - 1 + \ell}
   \;-\;
   \sum_{t = 1}^{a - 1} \lambda_t
     \left(
       \frac{1}{N_{tr}^a} \sum_{i \in I^a} Y_{i, t}
       -
       \sum_{i = 1}^{N_{co}} \omega_i Y_{i, t}
     \right).

The first two terms are the *post-treatment gap* between the treated
cohort and its synthetic control at offset :math:`\ell`; the third
term is the time-weighted *pre-treatment baseline*. By construction,

.. math::

   \hat\tau_a^{\,sdid}
   \;=\;
   \frac{1}{T_{tr}^a} \sum_{\ell = 1}^{T_{tr}^a} \hat\tau_{a, \ell}^{\,sdid},

i.e. the cohort ATT is the sample mean of its dynamic effects
(Equation 4 of Ciccia 2024). These effects are exposed on the result
object as :py:attr:`SDIDCohort.event_effects`.

Pooled Event Study (Equation 6)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`A_\ell = \{a \in A : a - 1 + \ell \le T\}` be the set of
cohorts for which the :math:`\ell`-th dynamic effect is computable,
and :math:`N_{tr}^\ell = \sum_{a \in A_\ell} N_{tr}^a` the
corresponding treated-unit count. The pooled event-study estimator is

.. math::

   \hat\tau_\ell^{\,sdid}
   \;=\;
   \sum_{a \in A_\ell}
     \frac{N_{tr}^a}{N_{tr}^\ell}
     \hat\tau_{a, \ell}^{\,sdid},

a treated-unit-weighted average of the cohort-specific dynamic effects.
This is the central quantity Ciccia (2024) recommends researchers
report. In the :mod:`mlsynth` API it is :py:attr:`SDIDEventStudy.tau`,
indexed by the corresponding event time on
:py:attr:`SDIDEventStudy.event_times`.

Overall ATT (Equation 7)
^^^^^^^^^^^^^^^^^^^^^^^^

Define :math:`T_{tr} = \max_{a \in A} T_{tr}^a`, the post-treatment
length of the earliest cohort. The overall ATT of Clarke et al. (2023)
admits the equivalent disaggregated form

.. math::

   \widehat{ATT}
   \;=\;
   \frac{1}{T_{post}} \sum_{\ell = 1}^{T_{tr}} N_{tr}^\ell \,
     \hat\tau_\ell^{\,sdid},

i.e. the average of the pooled event-study effects weighted by the
number of treated units contributing to each offset. This is
:py:attr:`SDIDInference.att`, with a placebo-based standard error and
confidence interval at :py:attr:`SDIDInference.se` /
:py:attr:`SDIDInference.ci`.

Placebo Inference
^^^^^^^^^^^^^^^^^

Variance estimation follows the placebo procedure of Arkhangelsky
et al. (2021), generalized to cohort and event-time effects by Clarke
et al. (2023). For each of :math:`B` iterations
(:py:attr:`SDIDConfig.B`), the donor pool is sampled to replace the
true treated units with pseudo-treated controls, the full SDID
pipeline is rerun, and the sample variance of the resulting effects
is taken as the variance of the actual estimator. The implementation
lives in
:func:`mlsynth.utils.sdid_helpers.inference.estimate_placebo_variance`.

The two-sided placebo p-value reported on
:py:attr:`SDIDInference.p_value` uses the canonical
:math:`((k + 1) / (B + 1))` correction, where :math:`k` is the count
of placebo iterations whose :math:`|\hat\tau^{\,*}_{att}|` is at least
as large as the observed :math:`|\widehat{ATT}|`.

Two-DataFrame and Single-Cohort Convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the panel has a single treated unit (e.g., California in the
Proposition 99 study), :func:`mlsynth.utils.datautils.dataprep` returns
a single-treated payload rather than a cohorts dict. The
:func:`mlsynth.utils.sdid_helpers.setup.prepare_sdid_inputs` helper
unifies both shapes into a single ``cohorts_dict`` keyed by adoption
period *index* (1-based), which is what the cohort estimator's
``\ell = t - (a - 1)`` math requires. In the single-cohort case, the
cohort ATT and the overall ATT are numerically identical by
construction.

Core API
--------

.. automodule:: mlsynth.estimators.sdid
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SDIDConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.sdid_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdid_helpers.weights
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdid_helpers.cohort
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdid_helpers.event_study
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdid_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdid_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdid_helpers.plotter
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdid_helpers.structures
   :members:
   :undoc-members:

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import SDID

   df = pd.read_csv(
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/smoking_data.csv"
   )
   df["Proposition 99"] = df["Proposition 99"].astype(int)

   results = SDID({
       "df":       df,
       "outcome":  "cigsale",
       "treat":    "Proposition 99",
       "unitid":   "state",
       "time":     "year",
       "B":        500,        # placebo iterations
       "display_graphs": True,
   }).fit()

   # Overall ATT (Ciccia 2024 Eq. 7) and placebo inference.
   print(results.inference.att)        # -14.486 (matches Arkhangelsky et al. 2021)
   print(results.inference.se)
   print(results.inference.ci)
   print(results.inference.p_value)

   # Pooled event-study trajectory (Ciccia 2024 Eq. 6).
   es = results.event_study
   for ell, tau, se in zip(es.event_times, es.tau, es.se):
       print(f"ell={int(ell):>3}  tau={tau:+.3f}  se={se:.3f}")

   # Per-cohort decomposition (Ciccia 2024 Eqs. 2 and 3).
   for adoption_period, cohort in results.cohorts.items():
       print(adoption_period, cohort.n_treated, cohort.att)
       print(cohort.event_effects[1])  # the first-period dynamic effect

References
----------

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager,
S. (2021). "Synthetic Difference-in-Differences." *American Economic
Review* 111(12):4088-4118.

Ciccia, D. (2024). "A Short Note on Event-Study Synthetic
Difference-in-Differences Estimators." `arXiv:2407.09565
<https://arxiv.org/abs/2407.09565>`_.

Clarke, D., Pailanir, D., Athey, S., & Imbens, G. (2023). "Synthetic
difference in differences estimation." arXiv preprint.
