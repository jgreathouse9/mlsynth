.. _replication-ferman-pinto-mc:

Ferman & Pinto (2021) Monte Carlo -- SC, demeaned SC and DID under imperfect fit
================================================================================

:Estimator: :doc:`../vanillasc` -- :class:`mlsynth.VanillaSC` (original SC) and
   :doc:`../tssc` -- :class:`mlsynth.TSSC` MSCa (demeaned SC)
:Source: Ferman, B. and Pinto, C. (2021), *"Synthetic controls with imperfect
   pretreatment fit,"* Quantitative Economics 12:1197-1221,
   `doi:10.3982/QE1596 <https://doi.org/10.3982/QE1596>`_ -- the Monte Carlo of
   Section 4 (Table 1).
:Replication type: Path B (the paper's Monte Carlo) together with a live,
   value-for-value cross-validation against the authors' own R code.
:Status: verified -- mlsynth's SC and demeaned SC reproduce the authors' two
   quadratic programs value-for-value on identical simulated panels, reproduce
   the Panel A/B bias close to the published numbers, and exhibit the paper's
   qualitative findings across all four panels.
:Benchmark: ``benchmarks/cases/ferman_pinto_mc.py``
   (`source <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/ferman_pinto_mc.py>`__).

Why this case exists
--------------------

The companion page :doc:`ferman` validates the demeaned SC estimator on a single
empirical panel (the Basque study). This case validates the same estimators on
the paper's Monte Carlo, where the true effect is known to be zero, so the
estimated effect *is* the bias -- and where the paper's whole argument lives.

Ferman and Pinto (2021) study three estimators when the pre-treatment fit is
imperfect and outcomes follow a linear factor model:

* the original synthetic control, with donor weights on the simplex
  (non-negative, summing to one) -- mlsynth's :class:`~mlsynth.VanillaSC`;
* their proposed demeaned synthetic control, which adds a free intercept to that
  simplex -- exactly the ``MSCa`` variant of :class:`~mlsynth.TSSC`;
* difference-in-differences, i.e. equal donor weights with an intercept -- the
  foil, computed inline.

Their Table 1 makes two points. First, with no structural break and a treated
unit at the extreme of the fixed-effect distribution, the original SC is biased
even as the number of pre-periods grows -- it cannot reconstruct the treated
unit's *level* -- while the demeaned SC and DID are approximately unbiased, and
the demeaned SC carries standard errors 40-50% below DID's. Second, with a break
in the first common factor (selection on unobservables), all three are biased,
but the SC and demeaned-SC biases are far below DID's and shrink as the panel
grows. This case reproduces both points with mlsynth's estimators.

The setup
---------

The data-generating process is calibrated once from the authors' panel of monthly
US state employment rates (50 states and DC, 1982-2019). A linear factor model --
unit fixed effects :math:`c_j`, time fixed effects, and four common factors
:math:`\boldsymbol{\lambda}_t \boldsymbol{\mu}_j` -- is fit by interactive fixed
effects, and the idiosyncratic shocks and factors are given autoregressive
dynamics. Each replication draws fresh factors and shocks, fixes the loadings, and
computes the three estimators' one-year effect. The four panels vary the treated
unit and whether a factor break is present, exactly as in the authors'
``Table and Figures - MC.do``:

.. list-table::
   :header-rows: 1
   :widths: 12 46 22

   * - Panel
     - Treated unit
     - Post-treatment
   * - A
     - second-largest fixed effect :math:`c_j`
     - no break
   * - B
     - second-smallest fixed effect :math:`c_j`
     - no break
   * - C
     - second-largest factor-1 loading :math:`\mu_{1j}`
     - break in factor 1
   * - D
     - second-smallest factor-1 loading :math:`\mu_{1j}`
     - break in factor 1

The calibrated environment is baked into
``benchmarks/reference/ferman_pinto_mc/fixed_env.npz`` (produced once by
``calibrate.py`` there, from the shipped CPS panel).

Same estimators as the paper
----------------------------

The reference is not transcribed numbers: it is the authors' ``_aux.R ::
synth_control_est`` and ``synth_control_est_demean`` (two ``quadprog`` quadratic
programs), reproduced verbatim in
``benchmarks/reference/ferman_pinto_mc/reference.R`` and run live via ``Rscript``.
The Python case simulates a handful of panels from the calibrated DGP, writes
them out, and has R solve both programs on the identical panels. mlsynth then runs
its real estimators on the same panels:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Quantity
     - mlsynth
     - authors' R (live)
   * - original SC donor weights
     - --
     - agree to :math:`\approx 3\times10^{-10}`
   * - demeaned SC weights + intercept
     - --
     - agree to :math:`\approx 3\times10^{-4}`
   * - effect (both estimators)
     - --
     - agree to :math:`\approx 3\times10^{-4}`

The original SC uses the same active-set quadratic program as R's ``solve.QP``, so
the weights coincide to machine precision; the demeaned SC uses a different solver
(``cvxpy``/CLARABEL) and coincides at the solvers' tolerance. mlsynth is not
approximating the paper's estimators; it *is* the paper's estimators. The case
``BenchmarkSkipped``s when ``Rscript`` / ``quadprog`` / ``jsonlite`` is absent, so
a missing R toolchain never turns the suite red.

Reproducing Table 1
-------------------

Running mlsynth's estimators over the four panels (300 replications, seeded)
reproduces the bias of the original SC in Panels A and B close to the published
values -- these turn on the unit fixed effect, which is invariant to how the
factor model is rotated:

============================  ==========  ===============
Quantity (:math:`T_0`)        mlsynth     Ferman & Pinto
============================  ==========  ===============
Panel A SC bias (120)         0.234       0.243
Panel A SC bias (480)         0.169       0.196
Panel B SC bias (480)         -0.498      -0.596
============================  ==========  ===============

The no-break demeaned SC and DID are both approximately unbiased (Panel A bias
:math:`\approx 0.001` and :math:`0.020`), and the demeaned SC is markedly more
efficient: its Monte-Carlo standard error is about 0.65 of DID's in Panel A and
0.50 in Panel B, matching the paper's "40-50% smaller." In the break panels (C and
D) all three estimators are biased, the SC and demeaned-SC biases sit far below
DID's, and the SC bias shrinks as :math:`T_0` grows -- the paper's Propositions 1
to 3 in the simulation.

A bug this case caught
----------------------

The demeaned SC carries a *free* intercept: its sign is unconstrained, which is
the entire point of "demeaning." Building this case surfaced a latent error in
mlsynth's ``MSCa`` (and ``MSCc``) fit, which had constrained the whole coefficient
vector -- intercept included -- to be non-negative. Whenever the treated unit sits
below its donors the optimal intercept is negative, and the estimator was silently
clamping it to zero and landing on a worse fit (Panels B and D have negative
optimal intercepts). The empirical Basque panel on the :doc:`ferman` page happens
to have a positive intercept, so it never exposed the problem. The fix frees the
intercept; the regression is pinned in
``mlsynth/tests/test_tssc.py::TestFreeIntercept``.

The calibration
---------------

The value-for-value tie above is exact and independent of any calibration detail.
The Table 1 *magnitudes*, though, depend on the calibrated factor model. The
authors fit it with ``gsynth::interFE`` and ``auto.arima``; this case uses a NumPy
port of the interactive-fixed-effects estimator and AR(1) dynamics (``gsynth`` is
not installable in the firewalled CI). Panels A and B reproduce the published bias
because they turn on the unit fixed effect, which is rotation-invariant. Panels C
and D turn on the first factor's loadings, whose rotation the port cannot
bit-match, so those panels reproduce Table 1's sign and ordering (all biased; SC
and demeaned SC far below DID; shrinking in :math:`T_0`) but not the exact
magnitudes. The benchmark pins the reproducible cells and the qualitative
structure; the side-by-side in
``benchmarks/reference/ferman_pinto_mc/comparison.csv`` shows all four panels.

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case ferman_pinto_mc

The fast offline unit test (a short Monte Carlo, no R) is
``mlsynth/tests/test_ferman_pinto_mc.py``. The case appears on the
:doc:`../validation` dashboard under VanillaSC / TSSC.
