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
