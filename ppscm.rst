Partially Pooled SCM (PPSCM)
============================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

``PPSCM`` is a faithful port of ``augsynth::multisynth`` -- the partially
pooled synthetic control of Ben-Michael, Feller and Rothstein [PPSCM]_ for
**staggered adoption**. Use it when several units are treated but at *different*
times, with a pool of never-treated (or late-treated) comparison units, and you
want a single estimate of the average treatment effect on the treated (ATT) over
relative time (time-since-treatment), pooling information across cohorts.

The central idea is a pooling dial :math:`\nu`. Fitting a *separate* synthetic
control for each treated unit gives the best per-unit pre-treatment fit but high
variance; a *fully pooled* control (one synthetic match for the average treated
unit) is stable but may fit any individual unit poorly. PPSCM interpolates
between the two, choosing :math:`\nu` to balance overall and unit-level
imbalance. ``time_cohort=True`` collapses units sharing an adoption time into a
single fully-pooled cohort (one synthetic control per cohort).

The problem PPSCM solves is that the two reflexive extensions of SCM to
staggered adoption are each flawed. **Separate SCM** (fit a synthetic
control per treated unit, then average -- common practice) requires a good
synthetic control for *every* treated unit, which often fails, and its
strong per-unit fits can still leave the *average* poorly matched, biasing
the ATT. **Pooled SCM** (match the average treated unit) nails the average
fit but can fit individual units badly, biasing unit-level effects and the
average when the data-generating process drifts over time. Ben-Michael,
Feller and Rothstein bound the estimation error by *both* the average
imbalance and the per-unit imbalances, and partially pooled SCM minimises a
weighted combination of the two -- the regime where neither extreme is
trustworthy.

Reach for PPSCM when
^^^^^^^^^^^^^^^^^^^^^

* Several units are treated at **different adoption times**, with a pool of
  never-treated (or not-yet-treated) comparison units.
* You want an **ATT over relative time** (an event-study path), pooling
  information across cohorts rather than trusting each cohort's own fit.
* **No single donor mix matches every treated unit**, so separate SCM
  leaves you with unreliable per-unit fits -- the partial-pooling dial lets
  the average fit borrow strength without abandoning unit-level fit.
* You want an estimator that **nests** the familiar special cases (separate
  and fully pooled SCM) and a principled way to choose between them.

Do not use PPSCM when
^^^^^^^^^^^^^^^^^^^^^^

* **All treated units adopt at the same time** (a single cohort). The
  staggered machinery is unnecessary; use classic SC (:doc:`tssc`,
  :doc:`fdid`) or, for many treated units at one time, :doc:`sdid`.
* **You are willing to assume parallel trends after weighting** and want
  the DiD-flavoured double weighting / time weights. :doc:`sdid` (and, for
  efficiency under interactive fixed effects, :doc:`seq_sdid`) is the more
  natural home; PPSCM is a *synthetic-control* estimator, not a
  difference-in-differences one.
* **Spillovers violate SUTVA** across the donor pool -- use :doc:`spsydid`.
* **The treated paths lie outside the donor convex hull / the donor pool is
  large and noisy.** Partial pooling cannot manufacture a hull that does
  not contain the treated units; a factor-model (:doc:`fma`) or low-rank
  (:doc:`clustersc`, :doc:`mcnnm`) approach is better.
* **Distributional** effects (quantiles, tails) -- use :doc:`dsc`.

Notation
--------

Units :math:`i = 1, \ldots, n` are observed over periods
:math:`t = 1, \ldots, T`. Treated unit (or cohort) :math:`j` adopts at period
:math:`T_j`; never-treated units have :math:`T_j = \infty` and form the donor
pool. The panel is split at the **last** adoption time into a pre-period of
length :math:`d` and the post-period. For cohort :math:`j`, donor weights
:math:`\boldsymbol{\omega}_j` live on the simplex; the synthetic control matches
the cohort's pre-treatment residuals.

Method
------

PPSCM follows ``multisynth`` in three stages.

**1. Two-way fixed effects (``fixedeff=True``, the default).** A time effect is
the never-treated units' per-period mean; a unit effect is each unit's mean over
its own pre-adoption window. Both are removed and the synthetic control balances
the **residuals** -- the "intercept-shifted" estimator of the paper.

**2. Partially pooled QP.** With per-cohort pre-treatment imbalance
:math:`\mathbf{q}_j = \mathbf{x}_j - \mathbf{X}_{0,j}\boldsymbol{\omega}_j`
(residuals; the pooled imbalance aligned by **relative time**), the weights
solve

.. math::

   \min_{\{\boldsymbol{\omega}_j \in \Delta\}} \;
     \frac{\nu}{\text{norm}_{\text{pool}}\,J^2}
       \Bigl\|\textstyle\sum_j \mathbf{q}_j\Bigr\|^2
     + \frac{1-\nu}{\text{norm}_{\text{sep}}\,J}
       \sum_j \frac{\|\mathbf{q}_j\|^2}{\text{ndim}_j}
     + \lambda \sum_j \|\boldsymbol{\omega}_j\|^2 ,

where :math:`\text{norm}_{\text{pool}}` and :math:`\text{norm}_{\text{sep}}` are
the separate-fit (``nu=0``) global and individual imbalance norms. Small
:math:`\nu` approaches a separate SCM per cohort; large :math:`\nu` a fully
pooled SCM.

**3. Choosing :math:`\nu`.** With ``nu="auto"`` (default) PPSCM uses augsynth's
triangle-inequality ratio :math:`\nu = \text{global\_l2}\cdot\sqrt{d}/\text{avg\_l2}`
from the separate fit; a float fixes it.

**Assumptions / Remarks.**

*Assumption 1 (no anticipation, parallel residual trends).* After removing the
two-way fixed effects, the treated cohorts' residual paths would have matched a
convex combination of donor residual paths absent treatment. *Remark.* This is
the staggered-adoption analogue of the SCM identifying assumption; the fixed
effects absorb level and common-time shifts so the weights only need to match
the residual dynamics.

*Assumption 2 (overlap / donor availability).* Each cohort has eligible donors
-- never-treated units, or units treated more than ``n_leads`` periods later.
*Remark.* Late-treated units can serve as "clean" controls for earlier cohorts
until they themselves are treated, which the donor-eligibility rule enforces.

*Remark (pooling).* :math:`\nu` is a bias--variance dial, not an identification
parameter: the estimand (the wATET over the treated cohorts) is the same;
:math:`\nu` only trades per-cohort fit against stability of the pooled average.

Inference
---------

``PPSCM`` reports the paper's **delete-one jackknife**: drop each unit, refit the
full estimator (holding :math:`\nu` fixed), and form
:math:`\widehat{\text{se}}^2 = \tfrac{n-1}{n}\sum_i(\hat\theta_i - \bar\theta)^2`
for the overall ATT and each relative-time horizon, with Wald intervals.

Empirical Illustration: mandatory collective bargaining
-------------------------------------------------------

The ``multisynth`` vignette studies the effect of state mandatory
collective-bargaining laws on log per-pupil education expenditure
(Paglayan 2018), a staggered design. ``basedata/Teachingaugsynth.scv`` ships the
panel; the analysis restricts to 1959--1997, drops DC and Wisconsin, and treats
a state from the year it required bargaining.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import PPSCM

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/Teachingaugsynth.scv"
   df = pd.read_csv(url)
   df = df[~df["State"].isin(["DC", "WI"])]
   df = df[(df["year"] >= 1959) & (df["year"] <= 1997)].copy()
   df["cbr"] = (df["year"] >= df["YearCBrequired"].fillna(np.inf)).astype(int)

   res = PPSCM({"df": df, "outcome": "lnppexpend", "treat": "cbr",
                "unitid": "State", "time": "year", "display_graphs": True}).fit()

   print(f"nu (auto)   : {res.design.nu_used:.4f}")
   print(f"Average ATT : {res.att:.3f}  (SE {res.inference.se:.3f})")

This prints::

   nu (auto)   : 0.2607
   Average ATT : -0.011  (SE 0.020)

reproducing the augsynth vignette (``nu = 0.2607``, Average ATT ``-0.011``).
Setting ``time_cohort=True`` collapses to adoption-time cohorts and gives
``nu = 0.3939``, Average ATT ``-0.017`` (augsynth: ``-0.018``).

Verification
------------

.. note::

   **Exact replication of augsynth.** On the Paglayan data PPSCM matches
   ``augsynth::multisynth`` to high precision: the auto-:math:`\nu` agrees to
   four decimals (0.2607 default, 0.3939 time-cohort), the Average ATT matches
   (:math:`-0.011` default; :math:`-0.017` vs :math:`-0.018` time-cohort), and
   the raw global/individual L2 imbalances agree (0.003 / 0.028). The full
   relative-time event study matches the vignette's per-horizon averages to
   3--4 decimals. The decisive fidelity detail is aligning the **pooled**
   imbalance by relative time on top of two-way fixed effects. The jackknife SE
   (0.020) is close to augsynth's default wild-bootstrap SE (0.022); they differ
   only by inference procedure. This is locked in by
   ``test_matches_augsynth_vignette`` in ``mlsynth/tests/test_ppscm.py``.

Core API
--------

.. automodule:: mlsynth.estimators.ppscm
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.PPSCMConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``PPSCM.fit()`` returns a
:class:`~mlsynth.utils.ppscm_helpers.structures.PPSCMResults`: the
:class:`~mlsynth.utils.ppscm_helpers.structures.PPSCMDesign` (pooling level and
balance diagnostics), the relative-time
:class:`~mlsynth.utils.ppscm_helpers.structures.PPSCMEventStudy`, the overall
:class:`~mlsynth.utils.ppscm_helpers.structures.PPSCMInference`, and the
per-cohort donor weights.

.. automodule:: mlsynth.utils.ppscm_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Staggered long-to-wide formatting (the only DataFrame touchpoint): derive
adoption times, split pre/post at the last adoption.

.. automodule:: mlsynth.utils.ppscm_helpers.setup
   :members:
   :undoc-members:

The engine: two-way fixed effects (``fit_feff``), the partially-pooled QP,
auto-:math:`\nu`, and the relative-time event study / ATT.

.. automodule:: mlsynth.utils.ppscm_helpers.engine
   :members:
   :undoc-members:

The paper's delete-one jackknife inference.

.. automodule:: mlsynth.utils.ppscm_helpers.inference
   :members:
   :undoc-members:
