FSC — Okano and Kurisu (2026)
=============================

Okano, R. and Kurisu, D. (2026). *Functional Synthetic Control Methods for
Metric Space-Valued Outcomes.* arXiv:2601.07539. Replication package:
`RyoOkano21/FSC <https://github.com/RyoOkano21/FSC>`_.

Path A, on the authors' own data, for all three of the paper's empirical
applications. Two artefacts back it, and they answer different questions.

`benchmarks/cases/fsc_okano.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/fsc_okano.py>`_
pins a faithful port of the authors' R code and reproduces every number they
publish, exactly. It answers: is the method as implemented by its authors
correctly understood here?

`benchmarks/cases/fsc_estimator.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/fsc_estimator.py>`_
runs :class:`mlsynth.FSC` itself. It answers a different question: does the
estimator mlsynth ships land on the paper's results? It does on one application
exactly, and diverges on the other two for reasons that are measured and stated
below.

What the reference port reproduces
----------------------------------

Every published figure, at the four decimals the paper prints.

.. list-table::
   :header-rows: 1
   :widths: 34 14 14 14 14

   * - Application
     - FSC
     - paper
     - augmented
     - paper
   * - 6.1 fertility, ASFR curves in :math:`L^2`
     - 0.1259
     - 0.1259
     - 0.0687
     - 0.0687
   * - 6.2 mortality, age-at-death distributions
     - 0.2092
     - 0.2092
     - 0.0634
     - 0.0634
   * - 6.3 service trade, covariance matrices
     - 39.3429
     - 39.3429
     - 20.0639
     - 20.0639

All twenty augmented donor weights of Table 1, all seventeen of Table 2, and the
three nonzero FSC weights of Table 3 match to their printed three decimals, with
a maximum deviation of 0.000. The cross-validated penalties of Remark 5
reproduce too: mortality lands on 5.889182 to eight digits and service on
0.001864 against a printed 0.00186.

Four details of the reference code are load-bearing, and none is visible from
the paper alone.

The rounding. ``FSCM`` returns ``round(weight_scm, 4)``, and the rounding is
carried into every downstream number. On the service application it is the
difference between the published 39.3429 and the exact quadratic program's
39.3423.

The norm is not consistent across the reported figures. The pre-treatment fit is
computed as ``sum(diffs[-1])`` — dropping the first grid point — for the plain
estimator in all three applications, but the augmented leg is coded
``sum(diffs)``. The published augmented figures come out under drop-first for
fertility and mortality and under all-points for service, so the three reported
pairs are not on one common norm.

The rearrangement is not a sort. ``Rearrangement::rearrangement`` rescales the
grid to the unit interval, interpolates onto 1001 equispaced points, and returns
their type-7 empirical quantiles at the rescaled grid. Substituting a plain sort
lands the mortality figure at 0.0640 where the published value is 0.0634.

The covariance outcome is a plain half-vectorisation, with no :math:`\sqrt 2` on
the off-diagonals. The Frobenius metric of the paper's Example 3 counts each
off-diagonal entry twice, so that map is not an isometry and 39.3429 is a vech
norm rather than a Frobenius one. The same weights scored under Example 3's own
metric give 51.9613.

Table 1 also contains an arithmetic slip: the Switzerland entry of 0.089 makes the FSC column sum to 1.089, which the simplex
constraint forbids. Evaluating the objective settles it — the reported fit of
0.1259 is attained at Austria 0.396, Bulgaria 0.416, Czechia 0.188 and nothing
else, while the Table 1 vector gives 0.2324 and its renormalisation 0.1482.

What the shipped estimator reproduces
-------------------------------------

:class:`mlsynth.FSC` matches the fertility application exactly — 0.1259 before
augmentation and 0.0687 after, from the standard configuration with the penalty
cross-validated rather than supplied. That is the paper's flagship application
and the one its Example 1 is built around.

The other two diverge.

Mortality reproduces on the plain estimator, 0.2092, which does not involve the
basis. The augmented figure comes out at 0.0630 against a published 0.0634, a gap
of 0.6 percent. The cause is the basis inner product: mlsynth integrates over the
whole argument grid, the reference code drops the first point. For fertility that
choice is invisible — the fertility rate at age 12 is essentially zero for every
country, so the dropped coordinate carries no information and the weights agree
to 7e-7. For mortality it is not, because the quantile function at
:math:`p = 0.01` is a real number and dropping it discards real information;
holding everything else fixed, the augmented weights move by up to 0.109. Using
every point the data provides is the defensible reading of an :math:`L^2` inner
product, so that is what mlsynth does.

Service trade is not a divergence in the estimate so much as in the yardstick.
The estimator applies the :math:`\sqrt 2` off-diagonal scaling that makes the
half-vectorisation a genuine Frobenius isometry, which the reference code does
not, so its fit is a Frobenius norm and the published 39.3429 is a vech norm.
The two are not comparable and should not be compared. The like-for-like
comparator is 51.9613 — the authors' own weights scored under Frobenius — and
mlsynth attains 51.7665, which is what re-optimising under the correct metric
should do.

Both are corrections rather than discrepancies, and both are pinned in the
estimator benchmark so a future change to either surfaces as a failure rather
than drifting quietly.

The penalty needs one more note, because getting it wrong caused a real bug
here. The cross-validation objective of Remark 5 is nearly flat near its
minimum: on the fertility data it varies by 0.03 percent across
:math:`\lambda \in [5, 7]` while the pre-treatment fit moves in its fourth
decimal. The penalty is weakly identified, so its selected value is not a
quantity to interpret. Worse, its natural *scale* is set by the Gram matrix of
the basis coefficients, which moves with the square of the outcome — so a fixed
absolute search interval means something different on every dataset. The
authors' scripts search :math:`(0, 10)`, which suits their fertility panel and
is four orders of magnitude too coarse for their mortality panel once the
coefficients are a genuine :math:`L^2` inner product; searching it there puts
the optimum outside the interval and lands the augmented fit at 0.1688 instead
of 0.0630. mlsynth therefore searches the penalty as a multiple of the design's
own Gram scale and reports both the relative and the absolute value.

Data
----

``basedata/okano_fsc_fertility.csv``, ``basedata/okano_fsc_mortality.csv`` and
``basedata/okano_fsc_service.csv``, converted from the authors' ``asfr.RData``,
``aad.RData`` and ``service.RData`` with the ``rdata`` package — nested R lists,
which ``pyreadr`` cannot read, so no R installation is needed to rebuild them.
`benchmarks/reference/fsc_okano_data.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/reference/fsc_okano_data.py>`_
regenerates them. Underlying sources are the Human Fertility Database, the Human
Mortality Database, and UN Trade and Development.
