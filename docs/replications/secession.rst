.. _replication-secession:

VanillaSC -- lost-autonomy triggers and secessionism
====================================================

:Estimator: :doc:`../vanillasc` -- :class:`mlsynth.VanillaSC`
:Source: Schulte, F., Scantamburlo, M. and Ackren, M. (2026), *"Lost autonomy
   triggers and the rise of secessionism,"* European Political Science Review
   18:175-193, DOI 10.1017/S175577392510026X.
:Replication type: Path A -- reproduce the paper's empirical finding on the
   authors' data, and track the authors' published synthetic-control series.
:Status: verified -- mlsynth recovers the paper's finding (a large post-trigger
   secessionist surge in both cases) and tracks the authors' synthetic closely;
   the agreement with their specific package is close, not value-for-value.

What the paper does
-------------------

Secessionist movements often surge suddenly. Schulte, Scantamburlo and Ackren
(2026) argue that a "lost-autonomy trigger" -- a dramatic event symbolising a
loss of self-rule -- can set off such a surge, and test this with synthetic
control. They study two cases: the 2010 Spanish Constitutional Court decision
that pared back Catalonia's autonomy statute, and the 1994 economic shock (a
major bank collapse and Denmark-imposed constraints) in the Faroe Islands. For
each treated region they build a synthetic control from the other European
autonomous regions and read the post-trigger gap in secessionist sentiment
(``av_sec1_all``), with an in-space placebo (RMSPE-ratio) test for inference.

Their replication uses the ``SyntheticControlMethods`` Python package
(``Synth``, ``pen="auto"``, ``n_optim=100``) with covariate predictors -- a
penalized (Abadie-L'Hour) synthetic control with a V-optimized predictor match.

Method mapping
--------------

mlsynth's canonical counterpart is :class:`~mlsynth.VanillaSC` with
``backend="outcome-only"`` -- the well-posed convex problem, donor weights on the
simplex minimising the pre-treatment outcome fit -- and ``inference="placebo"``,
the same in-space RMSPE-ratio test. The panel is the authors'
``basedata/secession_autonomy.csv`` (13 autonomous regions, 1975-2021); each
treated region is compared with the other twelve as donors, and treatment turns
on at the trigger year (Catalonia 2010, Faroe Islands 1994).

The finding reproduces
----------------------

Both methods deliver the paper's headline -- a large post-trigger surge in
secessionist sentiment in both regions:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Quantity
     - VanillaSC (outcome-only)
     - Authors (SyntheticControlMethods)
     - correlation
   * - Catalonia -- post-2010 gap (ATT)
     - :math:`+28.0`
     - :math:`+23.9`
     - 0.92
   * - Faroe Islands -- post-1994 gap (ATT)
     - :math:`+23.9`
     - :math:`+26.8`
     - 0.75
   * - Catalonia -- pre-2010 fit RMSE
     - 2.66
     - 2.85
     -
   * - Faroe Islands -- pre-1994 fit RMSE
     - 4.92
     - 5.05
     -

Under the in-space placebo, Catalonia is the extreme case (RMSPE-ratio
:math:`p \approx 0.077`, the smallest attainable with thirteen regions), while
the Faroe result is weaker -- consistent with the paper treating Catalonia as the
cleaner test. The Catalonia synthetic leans on South Tyrol.

A note on tightness
-------------------

This is a *close* reproduction, not a value-for-value one. The reference is a
different implementation -- ``SyntheticControlMethods`` optimises predictor
(V) weights, auto-selects an Abadie-L'Hour penalty, and uses one hundred random
restarts -- so mlsynth's deterministic outcome-only fit tracks the authors'
synthetic (correlation 0.75-0.92) and lands within about fifteen percent on the
ATT, rather than matching digit for digit. mlsynth's pre-treatment fit is in fact
a touch tighter in both cases, which is expected: outcome-only minimises exactly
the pre-period outcome error, whereas the authors' penalized, covariate-matched
fit trades a little of that for predictor balance and regularisation. When
mlsynth is put on the same regularised footing (``backend="penalized"`` or
``"mscmt"`` with the paper's covariates) it lands right around the authors' fit.

Because the cross-package agreement is approximate, this case is a Path-A finding
replication rather than a tight cross-validation, and it is not listed on the
value-for-value :doc:`../validation` dashboard.

Verification
------------

The check is committed as the ``secession_scm`` benchmark case
(`benchmarks/cases/secession_scm.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/secession_scm.py>`__),
which runs :class:`~mlsynth.VanillaSC` on the shipped panel and asserts the
deterministic ATT, pre-treatment RMSE, in-space placebo p-value, and the
correlation with the authors' published synthetic (captured under
``benchmarks/reference/secession_scm/``). The fast offline unit test is
``mlsynth/tests/test_secession_replication.py``.
