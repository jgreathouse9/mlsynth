.. _replication-ferman:

Demeaned SC (Ferman & Pinto 2021) -- via TSSC's MSCa variant
============================================================

:Estimator: :doc:`../tssc` -- :class:`mlsynth.TSSC` (the ``MSCa`` variant)
:Source: Ferman, B. and Pinto, C. (2021), *"Synthetic controls with imperfect
   pretreatment fit,"* Quantitative Economics 12:1197-1221.
:Replication type: golden cross-validation against the authors' own R code,
   executed live via ``Rscript`` on every run.
:Status: verified -- mlsynth's ``MSCa`` reproduces the authors' demeaned-SC
   quadratic program value-for-value on the identified Basque panel.

What the paper proposes
-----------------------

Ferman and Pinto (2021) study synthetic control when the pre-treatment fit is
imperfect. They show that SC and related estimators are generally biased in that
regime, and propose a *demeaned* synthetic control -- donor weights on the
simplex (non-negative, summing to one) together with a free intercept -- as a
bias-and-variance improvement over difference-in-differences, along with a
specification test for it.

Why this uses TSSC (and why that is incidental)
-----------------------------------------------

mlsynth has no estimator called "demeaned SC," and it does not need one: the
demeaned SC is exactly the ``MSCa`` variant of :doc:`../tssc` -- the Two-Step
SC's simplex-plus-intercept model. This case therefore validates the *estimator*,
not TSSC's variant-selection machinery; it reaches into the ``MSCa`` fit and
ignores the Step-1 test. Adding a separate "demeaned SC" estimator or bolting an
intercept onto ``VanillaSC`` would be redundant -- ``MSCa`` already is it.

The reference is the authors' own code
--------------------------------------

The reference is not transcribed numbers: it is the authors' ``_aux.R ::
synth_control_est_demean`` (a ``quadprog`` QP), reproduced verbatim in
``benchmarks/reference/ferman_demeaned_basque/reference.R`` and **run live via
Rscript on the same shipped panel every time the benchmark runs**. The case
``BenchmarkSkipped``s when ``Rscript`` or ``quadprog`` is absent, so a missing R
toolchain never turns the suite red.

Cross-validation -- the identified case (1975)
----------------------------------------------

The panel is the Basque Country / ETA terrorism study (Abadie & Gardeazabal
2003), ``basedata/basque_data.csv``, with treatment in 1975. That year is chosen
deliberately: it is the *identified* regime -- twenty pre-treatment periods
(1955-1974) against sixteen donors, so :math:`C < n` and the demeaned-SC weights
are unique. mlsynth's ``MSCa`` QP and the authors' ``quadprog`` QP then agree
value-for-value:

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * - Quantity
     - MSCa (mlsynth)
     - demeaned SC (R, live)
   * - Cataluña weight
     - 0.5605
     - 0.5605
   * - Rioja weight
     - 0.2792
     - 0.2792
   * - Asturias weight
     - 0.1064
     - 0.1064
   * - Madrid weight
     - 0.0539
     - 0.0539
   * - free intercept
     - 0.5922
     - 0.5922
   * - ATT (GDP per capita)
     - :math:`-0.797`
     - :math:`-0.797`

Donor-weight L1 distance :math:`\approx 10^{-4}`, the intercept agrees to
:math:`\approx 10^{-5}`, and the ATT to :math:`\approx 4\times10^{-4}` -- the
residual is just the two QP solvers' tolerances.

A note on identification (why 1975, not 1970)
---------------------------------------------

The paper is about *imperfect* pre-treatment fit, so it is worth being explicit
about when the demeaned-SC estimate is even well-defined. With a 1970 cutoff the
Basque panel is rank-deficient -- sixteen donors against fifteen pre-treatment
years, :math:`C > n`. The demeaned-SC weights are then *not unique*: many weight
vectors fit the pre-period equally well, and mlsynth's QP and the authors' own
``constrOptim`` fallback legitimately land on different minimisers (a Rioja /
Cataluña corner versus a Murcia-heavy corner), giving different ATTs. That is not
a disagreement between implementations -- it is non-identification of the
estimand. Moving the cutoff to 1975 makes :math:`C < n`, the solution unique, and
the two codes coincide. The benchmark pins the identified case; the
non-identified contrast is documented here as the cautionary half of the story.

Verification
------------

The check is committed as the ``ferman_demeaned_basque`` benchmark case
(`benchmarks/cases/ferman_demeaned_basque.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/ferman_demeaned_basque.py>`__),
which runs :class:`~mlsynth.TSSC` and the authors' R demeaned SC live, side by
side, and asserts the donor weights, the free intercept, and the ATT agree to
solver tolerance. It appears on the :doc:`../validation` dashboard under TSSC.
The fast offline unit test is ``mlsynth/tests/test_ferman_replication.py``.
