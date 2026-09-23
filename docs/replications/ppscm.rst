PPSCM — augsynth ``multisynth`` (Paglayan collective bargaining)
================================================================

.. currentmodule:: mlsynth

Cross-validation against the reference implementation. ``PPSCM`` is mlsynth's
port of partially-pooled SCM (Ben-Michael, Feller & Rothstein 2021), whose
canonical implementation is ``augsynth::multisynth`` in R. This page reproduces
the package's own `multisynth vignette
<https://github.com/ebenmichael/augsynth/blob/master/vignettes/multisynth-vignette.md>`_
cell-for-cell — point estimates, the event study, **and** standard errors, the
last cross-checked against a live R run of augsynth for *both* of its inference
procedures.

Data
----

The Paglayan (2018) public-sector collective-bargaining panel shipped at
``basedata/Teachingaugsynth.scv``: log per-pupil expenditure (``lnppexpend``) by
``State`` and ``year``, treatment ``cbr`` derived from ``YearCBrequired``.
Restricted exactly as the vignette does — drop DC and WI, keep 1959–1997 —
leaving **32 staggered-treated and 17 never-treated** states.

Point estimates
---------------

================================  =========================  ===========================
Quantity                          PPSCM (mlsynth)            ``augsynth::multisynth``
================================  =========================  ===========================
Partial-pooling :math:`\nu`       0.2607                     0.2607
Average ATT                       −0.011                     −0.011
Global L2 imbalance               0.0026                     0.003
:math:`\nu` (``time_cohort``)     0.3939                     0.3939
Average ATT (``time_cohort``)     −0.017                     −0.018
Event-study path                  match to ``< 5e-4``        (reference)
================================  =========================  ===========================

The OSQP solver (the same one augsynth uses) and the heuristic :math:`\nu`
reproduce the reference to display precision; the per-horizon point estimates
match to ``< 5e-4`` (unit cohorts) and ``< 2.2e-3`` (time cohorts).

Inference — both of augsynth's procedures
-----------------------------------------

augsynth offers two inference types; ``PPSCM`` reproduces **each**, method for
method, and exposes them via ``inference_method``:

* ``inference_method="jackknife"`` — the delete-one jackknife
  (``inf_type="jackknife"``). mlsynth's per-horizon SEs match augsynth's to
  ``< 1.5e-3``.
* ``inference_method="bootstrap"`` — the Mammen wild/multiplier bootstrap
  (``inf_type="bootstrap"``), which is augsynth's **default** and the SE the
  vignette prints. The ported bootstrap reproduces the overall ATT SE
  (``0.022``) and the per-horizon path to ``< 4e-3`` (the residual is Monte-Carlo
  noise — R's RNG vs numpy's at ``n_boot``).

================================  ====================  ====================
Per-horizon SE (rel. time)        jackknife             bootstrap (default)
================================  ====================  ====================
augsynth                          0.0186 … 0.0350       0.0225 … 0.0325
PPSCM                             0.0185 … 0.0354       0.0224 … 0.0325
================================  ====================  ====================

.. note::

   The two procedures legitimately differ by ~10% (the bootstrap is wider early
   on). An earlier apparent "SE gap" was simply comparing mlsynth's *jackknife*
   to augsynth's *bootstrap* default — different methods. Matched
   method-for-method (verified against augsynth's R source and a live run), they
   agree.

Path B — the authors' own simulation designs
---------------------------------------------

The cross-validation above matches ``augsynth`` on real data, which pins the port but
cannot say whether the intervals cover: the truth is unknown on a real panel. The
paper's Section 6 supplies designs where it is known, under a sharp null, so every
true effect is exactly zero and an interval that excludes zero has missed.

``benchmarks/cases/ppscm_bfr_mc.py`` runs all three designs — two-way fixed effects, a
linear factor model, and a heterogeneous AR(3) — and scores coverage of that zero. The
DGPs live in :mod:`mlsynth.utils.ppscm_helpers.simulation`.

.. list-table:: Coverage of the overall ATT, nominal 95%
   :header-rows: 1
   :widths: 22 20 14 20 14

   * - Design
     - mlsynth bootstrap
     - BFR
     - mlsynth jackknife
     - BFR
   * - Two-way fixed effects
     - 0.940
     - 0.937
     - 0.980
     - 0.973
   * - Linear factor model
     - 1.000
     - 0.959
     - 0.840
     - 0.885
   * - Autoregressive
     - 0.960
     - 0.972
     - 0.880
     - 0.893

The two-way fixed effects cells land on the paper almost exactly. The factor cells
differ most, with mlsynth's bootstrap more conservative and its jackknife covering
less. That gap is expected and is recorded rather than tuned away: the paper's DGPs are
calibrated to the Paglayan panel with fitted parameters it does not report, there is no
replication archive, and the number of Monte Carlo replications is never stated, so
exact cells are not recoverable. The case therefore asserts the geometry the paper
argues for, not its numbers:

- PPSCM is unbiased under the sharp null, and the autoregressive design is where
  inexact fit bites (bias 0.153 here, about 0.294 in the paper);
- the wild bootstrap is near nominal when there is no bias from inexact fit, and turns
  conservative once factor structure or serial dependence is present;
- the jackknife runs the other way — above nominal under two-way fixed effects, below
  it on the other two — so the bootstrap covers at least as much on those designs.

That last contrast is the same mechanism behind the calibrated cumulative band: the
bootstrap over-states the per-period standard error as shared structure strengthens.
Reaching it from the authors' own designs is independent confirmation.

The cumulative band
~~~~~~~~~~~~~~~~~~~

The same case scores the per-unit cumulative conformal band (``conformal_horizon``) in
two calibration regimes, because split conformal guarantees coverage at or above the
nominal level for any number of windows but only approaches it as that number grows:

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Calibration windows
     - Coverage (nominal 0.90)
     - Reading
   * - 39
     - 0.891
     - at nominal — the order statistic is interior
   * - 11
     - 0.978
     - valid, but the half-width *is* the largest score

So the pre-period requirement documented on :doc:`../ppscm` is the condition for the
band to be *finite*; comfortably exceeding it is the condition for it to be *tight*. On
a short panel a "90% band" is silently much more conservative than 90%.

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py ppscm_paglayan
   python benchmarks/run_benchmarks.py ppscm_bfr_mc

The durable case is ``benchmarks/cases/ppscm_paglayan.py`` (it cross-checks the
point estimates, the event study, and both SE methods); the unit-level
regressions are pinned in ``mlsynth/tests/test_ppscm.py``
(``test_matches_augsynth_vignette``, ``test_jackknife_se_matches_augsynth_vignette``,
``test_bootstrap_se_matches_augsynth_vignette``). All run on the in-repo data,
so no R or network access is required.
