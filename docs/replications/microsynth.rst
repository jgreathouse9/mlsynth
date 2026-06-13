MicroSynth — panel method vs the R ``microsynth`` (Seattle DMI)
===============================================================

.. currentmodule:: mlsynth

Cross-validation of MicroSynth's panel method
(``weight_method="panel"``) against the R ``microsynth`` package
(Robbins, Saunders & Kilmer 2017, JASA; Robbins & Davenport 2021, JSS v97i02)
on the package's canonical **Seattle Drug Market Intervention** example.

What the panel method does
--------------------------

Reading ``microsynth/R/weights.r``: when ``match.out`` (lagged outcomes) is
supplied, the package's ``my.qp`` (a ``LowRankQP`` solve) chooses control
weights by a **non-negative quadratic program** — exactly balance the
covariate **totals** (an intercept makes the weights sum to the treated count)
and **least-squares**-fit the lagged outcomes, with :math:`w \\ge 0`. Raking
(``survey::calibrate``) is only the covariate-only initialization/fallback, not
the panel weights themselves.

That objective is rank-deficient over a large control pool: it constrains the
weights only through a handful of totals (here 10 covariate + 12 lagged-outcome
constraints across ~9600 controls), so the counterfactual is **not identified
by the constraints alone**. On this data the feasible period-13 effect ranges
over roughly :math:`[-392, +153]`; ``LowRankQP`` simply returns its
interior-point iterate. mlsynth adds a strictly-convex ridge
(``panel_ridge``) that selects the unique **minimum-norm / maximum-ESS**
optimum — the most diffuse synthetic control consistent with exact covariate
balance and the best lagged-outcome fit. Because ``LowRankQP``'s interior-point
solution is itself near that point, the two coincide to 3–4 significant
figures, making this a genuine cross-validation rather than a comparison of
solver artifacts.

Data
----

``basedata/seattledmi.parquet`` — the R ``microsynth`` package's ``seattledmi``
dataset (``data(seattledmi)``), trimmed to the columns this case uses
(ID/time/Intervention + the four Table 2 outcomes + the 9 census covariates).
Full panel: 9642 census blocks × 16 periods, 39 treated blocks,
``Intervention`` on from ``time >= 13``.

Configuration matches mlsynth's one-outcome MicroSynth exactly:
``match.out = c("any_crime")``, ``match.covar`` = the 9 census covariates,
``start.pre = 1``, ``end.pre = 12``, ``end.post = 16``.

Result
------

Per-period total treatment effect on ``any_crime`` (Treated − synthetic
Control):

==========  ===========  ================
Period      mlsynth      R ``microsynth``
==========  ===========  ================
13          −33.06       −33.06
14          −74.43       −74.35
15          −45.35       −45.45
16          −64.86       −64.89
**ATT**     **−54.43**   **−54.44**
==========  ===========  ================

Per-period effects agree to ~0.1 crimes and the ATT to ~0.01. The identified
quantities both packages pin agree exactly: weights sum to the treated count
(39) and covariate + lagged-outcome balance is exact (max \|SMD\| ≈ 1e-10).
The ridge-selected optimum has effective sample size ≈ 378 (the most diffuse
control consistent with the fit).

The placebo-permutation test (mlsynth ``run_inference=True``,
``permutation_test="lower"``) finds the crime reduction significant: the
observed ATT lies well below every placebo group (≈ 7 placebo SDs out), so the
one-sided p-value sits at the ``1/(1 + n_permutations)`` floor — matching the
R package's reported significant reductions on this example.

JSS Table 2 (multi-outcome joint match)
---------------------------------------

The package's headline analysis (``sea1`` in the JSS vignette) matches **all
four outcomes jointly** — one synthetic control balancing every outcome's
pre-period trajectory plus the covariates — and reports a cumulative effect per
outcome (Table 2, top panel). mlsynth reproduces it with ``match_outcomes`` set
to the four outcomes (run once per primary ``outcome``; the shared weight vector
is identical across runs). Cumulative ``Pct.Chng`` over the post window 13–16:

==============  =========  ================
Outcome         mlsynth    R ``microsynth``
==============  =========  ================
i_felony        −32.6%     −32.6%
i_misdemea      −37.3%     −37.3%
i_drugs         −15.9%     −15.8%
any_crime       −20.1%     −20.1%
==============  =========  ================

All four match to <0.5 percentage points (``Trt``/``Con`` levels agree to ~0.1
crimes). As in the paper, the permutation test flags felonies, misdemeanors and
total crime as significant reductions but **not** drug crimes — the JSS
``Perm.pVal`` column (felony 0.020, misdemea 0.008, drugs 0.304, any_crime
0.020) under ``test="lower"``.

.. note::

   **What is and isn't reproduced.** The top panel's point estimates
   (``Trt``/``Con``/``Pct.Chng``) are reproduced exactly. The top panel's
   *Linear* p-values/CIs are a survey Taylor-linearization variance estimate
   that mlsynth does not implement; mlsynth's inference is the placebo
   permutation, so it is the package's ``Perm`` column that is the comparison
   target.

.. note::

   **R reference is baked in, not run in CI.** ``microsynth`` does not install
   from CRAN in the CI/sandbox network (CRAN-over-HTTPS is firewalled). The
   reference numbers above come from ``benchmarks/R/microsynth_seattle.R``
   (R ``microsynth`` 2.0.51), which documents the apt + GitHub-mirror install
   route and regenerates them. The benchmark asserts mlsynth's output matches
   the baked reference within tolerance.

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case microsynth_seattle
   # regenerate the R references (needs R microsynth installed):
   Rscript benchmarks/R/microsynth_seattle.R    # single-outcome any_crime
   Rscript benchmarks/R/microsynth_table2.R     # JSS Table 2 joint match
