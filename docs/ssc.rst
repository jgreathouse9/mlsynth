Staggered Synthetic Control (SSC)
=================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

``SSC`` implements the staggered-adoption synthetic-control estimator of Cao, Lu
and Wu [SSC]_. It is for the setting where **many units adopt a policy at
different times** and you have a **long pre-treatment history** relative to the
number of units and post-periods (large :math:`T`, moderate :math:`N`, small
:math:`S` -- e.g. monthly or weekly outcomes for a few dozen jurisdictions).
Two features distinguish it from the alternatives.

First, it **uses every unit -- including not-yet-treated units -- as a donor.**
Each unit's untreated outcome is modelled as an intercept plus a *simplex*
synthetic control on all the other units. It therefore does **not** require a
pool of never-treated units (existing staggered SC methods lean heavily on
them, and degrade when treated units are the majority), and it does **not** rely
on parallel trends (unlike staggered difference-in-differences).

Second, it delivers **valid inference for policy-relevant aggregates.** All
individual unit-by-time effects are estimated jointly; the target is any linear
functional :math:`\gamma = L\tau` -- the event-time ATT, the overall ATT, or a
contrast between two policies. Inference is Andrews' (2003) end-of-sample
stability test, whose reference distribution is built from pre-treatment
residual windows, and which can test both *sharp* and *non-sharp* nulls.

Reach for ``SSC`` when adoption is **staggered**, the **pre-period is long**,
never-treated units are **scarce or absent**, and you want an **event-study**
of dynamic effects with confidence bands. It is well suited to high-frequency
aggregate outcomes (crime rates, prices, bond yields) for a moderate number of
units.

Do not use SSC when
~~~~~~~~~~~~~~~~~~~

* **The pre-period is short.** SSC's guarantees and its end-of-sample inference
  are large-:math:`T` (they need :math:`T_0 > S` clean pre-periods, and more in
  practice). With a short pre-period use :doc:`sdid`, :doc:`mcnnm`, or
  :doc:`ppscm`.
* **There is a single treated unit or a single adoption date.** SSC's leverage
  comes from pooling many staggered adopters. For one treated unit start at
  :doc:`fdid`/:doc:`tssc`; for a block of simultaneous adopters use
  :doc:`msqrt` or :doc:`sdid`.
* **No unit is well approximated by a convex combination of the others** (the
  treated units sit outside the donors' convex hull). The simplex fit will be
  poor; consider :doc:`mcnnm` (which regularises a latent factor model instead).
* **Anticipation is a concern.** SSC puts not-yet-treated units in the donor
  pool; if units change behaviour *before* adoption this can bias the fit
  (plot the pre-trends to check).
* **Spillovers across units violate SUTVA** -- use :doc:`spsydid` or
  :doc:`spillsynth`.

Notation
--------

A balanced panel of :math:`N` units over :math:`T_0 + S` periods, where
:math:`T_0` is the number of **clean** pre-treatment periods (before *any* unit
adopts) and :math:`S` is the number of post periods. Adoption times
:math:`(t_1, \ldots, t_N)` are observed (:math:`t_i = \infty` for never-treated
units); treatment is absorbing. The observed outcome is the never-treated
potential outcome before adoption and the treated one after. The individual
effect is :math:`\tau_{i,t} = y_{i,t}(t_i) - y_{i,t}(\infty)`, and the target is
a linear functional :math:`\gamma = L\tau` of the stacked effect vector
:math:`\tau \in \mathbb{R}^K` (:math:`K` = number of treated cells).

The estimator
~~~~~~~~~~~~~

*Step 1 -- synthetic-control weights.* For each unit :math:`i`, fit a demeaned
simplex synthetic control on **all other units** over the clean pre-period
(paper eq. 2.1):

.. math::

   (\widehat a_i, \widehat b_i)
     = \operatorname*{argmin}_{a,\,b \in W_i}
       \sum_{t=1}^{T_0}\bigl(y_{i,t} - a - Y_t' b\bigr)^2,
   \qquad W_i = \{b \ge 0,\ \textstyle\sum_j b_j = 1,\ b_i = 0\}.

Collect the intercepts :math:`\widehat a` and the weight matrix
:math:`\widehat B` (row :math:`i` is :math:`\widehat b_i`), and let
:math:`\widehat M = (I - \widehat B)'(I - \widehat B)`. The prediction error is
:math:`u_{i,t} = y_{i,t}(\infty) - (\widehat a_i + Y_t(\infty)'\widehat b_i)`.

*Step 2 -- joint effect estimation.* With selector matrices :math:`A_s` mapping
:math:`\tau` to the period-:math:`(T_0+s)` effect vector, the GLS estimator
(paper eq. 2.4) is

.. math::

   \widehat\tau = \Bigl(\sum_{s=1}^{S} A_s'\widehat M A_s\Bigr)^{-1}
     \sum_{s=1}^{S} A_s'(I - \widehat B)'
       \bigl((I - \widehat B) Y_{T_0+s} - \widehat a\bigr).

The invertibility of :math:`\sum_s A_s' M A_s` (Assumption 2.1) is the key
identifying condition; its smallest eigenvalue is a useful diagnostic. The
event-time ATT at horizon :math:`s` is the average of :math:`\widehat\tau` over
cells with event time :math:`s`, and the overall ATT is the grand mean.

Inference
~~~~~~~~~

SSC tests :math:`H_0: C\tau = d` (e.g. event-time ATT :math:`= 0`, or two
policies equal) with **Andrews' (2003) end-of-sample stability test**. The test
statistic is :math:`\widehat P = (C\widehat\tau - d)'(C\widehat\tau - d)`; its
critical value comes from sliding a length-:math:`S` window across the
:math:`T_0` pre-treatment residuals to form :math:`T_0 - S` placebo realisations
of the estimator under the null. Under a stationarity/ergodicity assumption on
the prediction error the test has asymptotically correct size as
:math:`T \to \infty` -- crucially **without** point-identifying :math:`\tau`.
``mlsynth`` reports, for the overall ATT and each event-time ATT, a band (the
point estimate plus the placebo distribution's quantiles) and a two-sided
p-value, on :class:`~mlsynth.utils.ssc_helpers.structures.SSCBand`.

Assumptions and econometric theory
-----------------------------------

SSC is a **large-:math:`T`, fixed-:math:`N`-and-:math:`S`** method. The
individual effects :math:`\tau_{i,t}` are **not point-identified** (there are
more unknowns than the data can pin down); the payoff is that any aggregate
:math:`\gamma = L\tau` is *asymptotically unbiased* and admits valid inference
as the pre-period lengthens.

*Setup (SUTVA, no anticipation).* Potential outcomes follow a Rubin model in
which (i) a unit stays treated once treated (absorbing), (ii) a unit's outcome
depends only on its own treatment status and timing -- **no interference /
spillovers across units** -- and (iii) pre-adoption outcomes equal the
never-treated potential outcome (**no anticipation**).

*Assumption 2.1 (invertibility).* :math:`\sum_{s=1}^{S} A_s' M A_s` is
invertible, with :math:`M = (I-B)'(I-B)`. *Remark.* This is the key identifying
condition: it makes the linear map from the post-treatment prediction errors to
:math:`\tau` full rank, so the estimator (eq. 2.4) is well defined. It fails
only in degenerate cases -- a "disconnected treated cohort" whose units lie in
one another's convex hull -- and staggered timing typically *bridges* cohorts
and restores it. The smallest eigenvalue of the sample
:math:`\sum_s A_s'\widehat M A_s` is a practical diagnostic (the paper's
Table 1); ``mlsynth`` reports it as ``results.metadata["gram_min_eigenvalue"]``.

*Assumption 2.2 (stationary prediction error; consistent weights).* The
prediction error :math:`u_{i,t} = y_{i,t}(\infty) - (a_i + Y_t(\infty)'b_i)` is
strictly stationary with mean zero, and the synthetic-control weights converge
(:math:`\widehat a \to a`, :math:`\widehat B \to B`). *Remark.* The authors show
this holds when the untreated outcomes share **stationary or cointegrated**
common factors -- the cointegrating relationship is exactly what lets a *stable*
cross-sectional synthetic control exist with a stationary remainder, which is
why a long, well-behaved pre-period matters.

*Assumption 2.3 (ergodicity; regularity for inference).* :math:`\{u_t\}` is
ergodic with finite second moment, a normalising sequence controls the
regressors, the weight estimates converge uniformly across the placebo windows,
and the test statistic's distribution is continuous and increasing at its
:math:`(1-\alpha)` quantile. *Remark.* These are the conditions under which the
pre-treatment placebo windows are a valid stand-in for the post-treatment
sampling distribution of the estimator.

**Theorem 2.1 (asymptotic unbiasedness).** Under Assumptions 2.1--2.2, as
:math:`T \to \infty`,

.. math::

   \widehat\gamma - (\gamma + L V_T) \xrightarrow{p} 0,
   \qquad \mathbb{E}[L V_T] = 0,

so :math:`\widehat\gamma` -- and, by Corollary 2.1, the event-time ATT
:math:`\widehat{\mathrm{ATT}}^e_s = l_s'\widehat\tau` -- is an asymptotically
unbiased estimator of its target *without* point-identifying the individual
effects. (The remaining :math:`L V_T` term is mean-zero estimation noise that
the inference procedure quantifies.)

**Theorem 2.2 (valid end-of-sample inference).** Under Assumptions 2.1--2.3 and
the null :math:`H_0: C\tau = d`, the Andrews test has asymptotically correct
size,

.. math::

   \Pr\!\bigl(\widehat P > \widehat q_{1-\alpha}\bigr) \to \alpha
   \quad\text{as } T \to \infty,

and confidence regions are obtained by inverting the test. The result holds for
both *sharp* nulls (e.g. a single :math:`\mathrm{ATT}^e_s = 0`) and *non-sharp*
nulls (restrictions on aggregates), which is what makes it suited to
policy-relevant hypotheses under staggered adoption.

*Why large-:math:`T`.* The leverage comes entirely from the long pre-period: it
identifies the synthetic-control weights and supplies the placebo windows that
calibrate inference. This is why SSC fits high-frequency aggregate outcomes
(monthly, weekly) with a moderate number of units -- and why it is **not** for
short panels.

Example
-------

A staggered panel of twenty units (four never treated) following a three-factor
model, adopting across a six-period window, with a dynamic effect that grows
with event time (:math:`\tau = 1 + e`). ``SSC`` recovers the event-study path
with end-of-sample bands and reports the overall ATT.

.. code-block:: python

   from mlsynth import SSC
   from mlsynth.utils.ssc_helpers.simulation import simulate_ssc_panel

   df = simulate_ssc_panel(
       n_units=20, n_never=4, T0=50, S=6, base_effect=1.0, seed=1,
   )

   res = SSC({
       "df": df, "outcome": "Y", "treat": "treated",
       "unitid": "unit", "time": "time",
       "inference": True,         # Andrews end-of-sample bands + p-values
       "display_graphs": True,    # event-study plot
   }).fit()

   print(f"overall ATT = {res.att:+.3f}  (p = {res.att_band.p_value:.3f})")
   for e in sorted(res.event_att):
       b = res.event_bands[e]
       print(f"  event time {e}: {b.point:+.3f}  [{b.lower:+.3f}, {b.upper:+.3f}]"
             f"  (true {1.0 + e:.0f})")

Empirical replication (Guanajuato police reform)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package ships the paper's Section 4 data (Alcocer 2024, Harvard Dataverse)
and the authors' reference estimates in ``basedata/``. The block below is
**copy-paste runnable after a fresh install** -- it pulls the panels straight
from the ``basedata/`` raw URL, fits ``SSC`` through the public API, and checks
every estimate against the authors' published table:

.. code-block:: python

   import pandas as pd
   from mlsynth import SSC

   BASE = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/basedata/"

   # --- One outcome, directly through the public API ----------------------
   # Homicide rate: monthly panel, the paper's sample window (time < 253).
   crime = pd.read_csv(BASE + "guanajuato_crime_ssc.csv").query("time < 253")
   res = SSC({"df": crime[["idunico", "time", "Policial", "hom_all_rate"]],
              "outcome": "hom_all_rate", "treat": "Policial",
              "unitid": "idunico", "time": "time",
              "inference": True, "alpha": 0.05, "display_graphs": False}).fit()
   print("homicide ATT^e_1 =", round(res.event_att[0], 4), " (paper: 0.0743)")

   # --- All seven outcomes vs the authors' reference table ----------------
   from mlsynth.utils.ssc_helpers import replicate_guanajuato
   est = replicate_guanajuato(verbose=False)        # downloads both panels from basedata/
   ref = pd.read_csv(BASE + "guanajuato_ssc_reference.csv").rename(
       columns={"event time": "event_time", "att estimate": "ref_att"})
   m = est.merge(ref[["outcome", "event_time", "ref_att"]],
                 on=["outcome", "event_time"])
   m["abs_diff"] = (m["att"] - m["ref_att"]).abs()
   print("\nmax |mlsynth - paper| ATT, per outcome (", len(m), "cells):")
   print(m.groupby("outcome")["abs_diff"].max().round(6).to_string())

prints::

   homicide ATT^e_1 = 0.0743  (paper: 0.0743)

   max |mlsynth - paper| ATT, per outcome ( 357 cells):
   outcome
   co_num                   0.001015
   hom_all_rate             0.000187
   hom_ym_rate              0.000097
   presence_strength        0.000046
   theft_nonviolent_rate    0.000016
   theft_violent_rate       0.000149
   war                      0.000081

Every one of the 357 reference cells (seven outcomes x their event-time paths)
is reproduced: the homicide and theft rates match the authors' table to about
:math:`10^{-4}`, and the short annual cartel outcomes to :math:`10^{-3}` (the
residual is the simplex-weight solver -- cvxpy here vs. the reference's
``fmincon``). The confidence bands match where the reference has them (present
for homicide and the cartel outcomes; ``NaN`` for theft, where :math:`T_0 < S`
leaves no pre-treatment placebo window). The reference table itself is shipped
at `basedata/guanajuato_ssc_reference.csv <https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/basedata/guanajuato_ssc_reference.csv>`_,
and the two panels at
`guanajuato_crime_ssc.csv <https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/basedata/guanajuato_crime_ssc.csv>`_
and
`guanajuato_cartel_ssc.csv <https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/basedata/guanajuato_cartel_ssc.csv>`_.

Simulation study (Path B)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The paper's Section 3 Monte Carlo is reproduced through the same public API.
``run_ssc_simulation`` simulates the staggered factor DGP and returns SSC's
event-time RMSE per ``(r, T0)`` cell (the paper's Figure 1):

.. code-block:: python

   from mlsynth.utils.ssc_helpers.replication import (
       run_ssc_simulation, SSCSimConfig, PAPER,
   )

   # fast, reduced-count preset (use PAPER for the exact N=33, 1000-rep study)
   rmse = run_ssc_simulation(SSCSimConfig(n_units=20, n_never=4, S=6,
                                          n_factors=2, T0_grid=[42], n_reps=20))
   for cell, by_event in rmse.items():
       print(cell, {e: round(v, 3) for e, v in sorted(by_event.items())})

prints (Monte-Carlo values vary by seed/preset, but the *pattern* -- event-time
RMSE rising with the horizon, as in the paper's Figure 1 -- is stable)::

   (2, 42) {0: 0.37, 1: 0.416, 2: 0.547, 3: 0.552, 4: 0.907, 5: 0.991}

Verification
------------

.. note::

   **Path B replication of the paper's simulation study (Section 3).**
   :mod:`mlsynth.utils.ssc_helpers.replication` reproduces the authors'
   *synthetic* Monte-Carlo study -- a Path B replication, since we replicate
   their simulation-section results rather than an empirical data set -- through
   the public :meth:`mlsynth.SSC.fit` API. The DGP is the paper's factor model
   (:func:`~mlsynth.utils.ssc_helpers.simulation.simulate_ssc_panel`):
   ``N = 33`` units (``30`` treated, staggered over an ``S = 7`` window),
   ``r in {3, 6}`` AR(1) factors, ``T in {15, 42, 157}`` pre-periods, and a
   dynamic effect :math:`\tau = 1 + e`. The reported quantity is the
   **event-time RMSE** of the ATT estimates (the paper's Figure 1). SSC
   recovers the increasing effect path, and its event-time RMSE is lowest in the
   early post-periods -- below GSC (Xu 2017) and partially-pooled SC
   (Ben-Michael et al. 2022) there -- because it builds the synthetic controls
   from *all* units rather than only the scarce never-treated ones, which
   inflate those methods' variance. The ``PAPER`` preset runs the authors' exact
   1,000-replication configuration; the ``DEMO`` preset is a faster,
   reduced-count version that reproduces the qualitative pattern.

   **Path A replication of the empirical application (Section 4).** Running
   ``SSC`` on the paper's Guanajuato police-reform data (Alcocer 2024;
   :math:`N = 33` municipalities, :math:`10` staggered adopters) reproduces the
   authors' reference event-time ATT estimates for all seven outcomes -- the
   long-pre-period homicide rates (:math:`T_0 = 174`, :math:`S = 78`) and theft
   rates (:math:`T_0 = 42`) to about :math:`10^{-4}`, and the short annual cartel
   outcomes (:math:`T_0 = 15`) to about :math:`10^{-3}` (the residual is the
   simplex-weight solver, cvxpy here vs. the reference's ``fmincon``). The bands
   are reported exactly where the reference has them: present for homicide and
   the cartel outcomes, and ``NaN`` for theft, where :math:`T_0 < S` leaves no
   pre-treatment placebo window.

   **Inference.** The end-of-sample band is calibrated on pre-treatment
   residual windows, so coverage does not require point-identification of the
   individual effects -- only stationarity of the prediction error.

Core API
--------

.. automodule:: mlsynth.estimators.ssc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SSCConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``SSC.fit()`` returns a
:class:`~mlsynth.utils.ssc_helpers.structures.SSCResults`: the per-cell effects
``tau`` with their ``index`` (post-period, unit, event time), the overall
``att`` and its :class:`~mlsynth.utils.ssc_helpers.structures.SSCBand`, the
``event_att`` path and per-event ``event_bands``, the per-cell ``effects`` grid,
the synthetic-control intercepts ``a_hat`` and weight matrix ``B_hat`` (plus a
:class:`~mlsynth.config_models.WeightsResults`), the pre-treatment
``residuals``, and the
:class:`~mlsynth.utils.ssc_helpers.structures.SSCInference` summary.

.. automodule:: mlsynth.utils.ssc_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Staggered-panel ingestion: pivots the long panel, locates the clean pre-period,
and checks the absorbing-treatment and pre-period conditions.

.. automodule:: mlsynth.utils.ssc_helpers.setup
   :members:
   :undoc-members:

Per-unit simplex synthetic-control weights (each unit on all others).

.. automodule:: mlsynth.utils.ssc_helpers.weights
   :members:
   :undoc-members:

The selector tensor, the GLS effect estimator, linear aggregation, and the
Andrews end-of-sample inference.

.. automodule:: mlsynth.utils.ssc_helpers.estimation
   :members:
   :undoc-members:

Run loop: weights, effect estimation, event-time / overall aggregation, and the
optional end-of-sample bands.

.. automodule:: mlsynth.utils.ssc_helpers.pipeline
   :members:
   :undoc-members:

Staggered-adoption factor-model DGP for examples and tests.

.. automodule:: mlsynth.utils.ssc_helpers.simulation
   :members:
   :undoc-members:

Path-B replication of the paper's Section 3 Monte-Carlo study (event-time RMSE)
through the public ``SSC.fit`` API, with the ``PAPER`` / ``DEMO`` presets.

.. automodule:: mlsynth.utils.ssc_helpers.replication
   :members:
   :undoc-members:
