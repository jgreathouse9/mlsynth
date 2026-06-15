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

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py ppscm_paglayan

The durable case is ``benchmarks/cases/ppscm_paglayan.py`` (it cross-checks the
point estimates, the event study, and both SE methods); the unit-level
regressions are pinned in ``mlsynth/tests/test_ppscm.py``
(``test_matches_augsynth_vignette``, ``test_jackknife_se_matches_augsynth_vignette``,
``test_bootstrap_se_matches_augsynth_vignette``). All run on the in-repo data,
so no R or network access is required.
