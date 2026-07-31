.. _replication-brabander-brexit:

Seven estimators on Brexit, and a placebo test between them (de Brabander et al. 2025)
======================================================================================

:Estimators: :doc:`../sdid` — :class:`mlsynth.SDID`; :doc:`../tssc` —
   :class:`mlsynth.TSSC`; :doc:`../masc` — :class:`mlsynth.MASC`;
   :doc:`../vanillasc` — :class:`mlsynth.VanillaSC`
:Source: de Brabander, E., Juodis, A., & Miyazato Szini, G. (2025), *"On the
   Use of Synthetic Difference-in-Differences Approach with(-out) Covariates:
   The Case Study of Brexit Referendum,"* Econometric Reviews.
:Replication type: Path A — the paper's empirical results on the authors' own
   data, both the headline table and the placebo table that judges it.
:Status: All fourteen cells of Table 1 and all twenty-one of Table 7
   reproduce, to the precision the paper prints.
:Cases: `benchmarks/cases/brabander_brexit_table1.py
   <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/brabander_brexit_table1.py>`_,
   `benchmarks/cases/brabander_brexit_insample.py
   <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/brabander_brexit_insample.py>`_

What the paper does
-------------------

Born, Müller, Schularick and Sedláček (2019) put the cost of the Brexit
referendum at about 2.4 percent of UK GDP by the end of 2018, estimated with a
synthetic control built from other OECD economies. This paper asks a question
that sits underneath that one: how much of the answer came from the method, and
how much from choices that are usually left unstated?

To find out it runs seven estimators on the same panel — quarterly log real GDP
for the UK and twenty-three donor countries — and tabulates what each says the
UK lost by the end of 2018 and of 2019. The seven are classic synthetic control;
demeaned synthetic control, which lets the donors sit at a different level than
the treated unit; synthetic difference-in-differences under three different
conventions about which quarters the panel carries; MASC, which blends synthetic
control with nearest-neighbour matching; and the augmented synthetic control
method, which corrects the fit with a ridge regression.

The three SDID rows are the paper's real subject. They are the same estimator on
the same data; they differ only in bookkeeping:

(i)
   the panel stops one quarter after the referendum, and the effect is reported
   for a later quarter — so the number being read lies outside the window the
   weights were fit on;
(ii)
   the panel runs through to the quarter being reported;
(iii)
   the panel is the pre-referendum quarters plus the reported quarter alone.

Nothing in the method says which to pick, and the paper's point is that the
choice is not innocuous.

Table 1: the headline estimates
-------------------------------

Treatment at 2016:Q3, no covariates in the matching. Losses in percent of GDP.

.. list-table:: Table 1, 2016:Q3 block — published against mlsynth
   :header-rows: 1
   :widths: 20 16 16 16 16

   * - Method
     - 2018:Q4 paper
     - 2018:Q4 mlsynth
     - 2019:Q4 paper
     - 2019:Q4 mlsynth
   * - SC
     - 3.06
     - 3.04
     - 4.20
     - 4.17
   * - DSC
     - 2.98
     - 2.99
     - 4.12
     - 4.12
   * - SDID (i)
     - 2.76
     - 2.77
     - 3.89
     - 3.91
   * - SDID (ii)
     - 2.79
     - 2.80
     - 3.92
     - 3.94
   * - SDID (iii)
     - 2.79
     - 2.80
     - 3.92
     - 3.94
   * - MASC
     - 2.73
     - 2.73
     - 3.83
     - 3.83
   * - ASCM
     - 3.04
     - 3.04
     - 4.19
     - 4.19

The largest deviation is 0.028, on SC at 2019:Q4; eleven of the fourteen cells
agree to better than 0.02. The paper prints two decimals, so this reproduces at
display precision throughout.

Cases (ii) and (iii) come out identical, in mlsynth and in the published table
alike. That is not rounding. With the ridge switched off, the unit weights are a
function of the pre-treatment outcomes only, and the time-weight problem lands on
the vertex that puts all of its mass on the final pre-treatment quarter — so the
two panels hand the weight programs the same information. Case (i) does differ,
because its time weights spread a little mass onto two earlier quarters.

Table 7: which of them to believe
---------------------------------

Table 1 says what each estimator claims; it cannot say which claim to trust. The
paper's answer is a placebo exercise. Move the treatment date back to each of
twenty pre-referendum quarters (2010:Q1 through 2014:Q4), where nothing happened
and the true effect is zero, and score each method by how far its estimate
strays from zero. Smaller is better.

.. list-table:: Table 7 — published against mlsynth
   :header-rows: 1
   :widths: 18 14 14 14 14 14 14

   * - Method
     - RMSE paper
     - RMSE mlsynth
     - MAB paper
     - MAB mlsynth
     - MedAB paper
     - MedAB mlsynth
   * - SC
     - 0.0089
     - 0.0086
     - 0.0072
     - 0.0068
     - 0.0055
     - 0.0051
   * - DSC
     - 0.0087
     - 0.0086
     - 0.0070
     - 0.0068
     - 0.0052
     - 0.0051
   * - SDID (i)
     - 0.0067
     - 0.0066
     - 0.0037
     - 0.0037
     - 0.0016
     - 0.0016
   * - SDID (ii)
     - 0.0134
     - 0.0133
     - 0.0111
     - 0.0109
     - 0.0103
     - 0.0102
   * - SDID (iii)
     - 0.0134
     - 0.0133
     - 0.0111
     - 0.0110
     - 0.0107
     - 0.0102
   * - MASC
     - 0.0080
     - 0.0080
     - 0.0062
     - 0.0062
     - 0.0045
     - 0.0045
   * - ASCM
     - 0.0086
     - 0.0086
     - 0.0068
     - 0.0068
     - 0.0051
     - 0.0051

MAB is the mean absolute bias and MedAB the median absolute bias, across the
twenty placebo dates. All twenty-one cells reproduce; the largest deviation is
0.00052 and eighteen of the twenty-one land inside 0.00022.

The ranking survives, which is what the table is for. SDID case (i) is the most
accurate of the seven by a wide margin, and cases (ii) and (iii) are the least
accurate — roughly twice the error of plain SC. Since the three cases are the
same estimator, that spread is a statement about bookkeeping.

One caveat the paper itself makes, and worth repeating here: cases (ii) and
(iii) are evaluated four quarters past the placebo date while the other five are
evaluated one quarter past, so part of their larger error is the longer horizon
rather than the convention. Case (i) against case (ii) is the clean comparison,
since both fit a panel that extends past the treatment date, and there the gap
is still a factor of two.

What it took to match
---------------------

Two settings decide every SDID number on this page, and both are the paper's
rather than mlsynth's defaults.

The first is the penalty. The authors pass ``zeta.omega = 0`` throughout,
switching off the ridge on the unit weights that Arkhangelsky et al. (2021)
calibrate from the data. mlsynth expresses this with
:py:attr:`SDIDConfig.zeta`; with the default ridge instead, case (ii) reads
2.53 and 3.60 against the published 2.79 and 3.92.

The second is which counterfactual to read. The paper's estimator subtracts the
time-weighted pre-treatment gap,

.. math::

   \hat\tau \;=\; y_{1,T} - y_{0,T}'\boldsymbol{\omega}
   \;-\; \boldsymbol{\lambda}'\bigl(Y_{1,pre}
   - Y_{0,pre}\boldsymbol{\omega}\bigr),

which is mlsynth's ``intercept_adjust=True`` series. Reading the default
un-adjusted counterfactual compares a different quantity to the paper's number
and costs about 0.05 percentage points on Table 1 — small enough to look like a
minor disagreement rather than the wrong estimand, which is exactly what makes
it worth stating.

One discrepancy in the authors' own package is worth recording. Both scripts set
MASC's cross-validation with a variable named ``min_value_five``, computed as
``T_pre - 5``, which selects five folds. The published cells do not come from
that setting; they come from the fold counts stated in the table notes — eighty
folds for Table 1, ``T_pre - 5`` folds for Table 7 — which correspond to a
smallest fold of six and five respectively. Under the notes' reading, MASC
reproduces exactly (2.73 / 3.83 in Table 1; 0.0080 / 0.0062 / 0.0045 in
Table 7); under the scripts' reading, Table 7's MASC row comes out at 0.0060 /
0.0041 / 0.0023. The notes are followed here, and the variable name appears to
be the slip.

Provenance
----------

The panel is ``benchmarks/reference/brabander_sdid_match/brexit_panel.csv``,
built by ``reference.R`` in the same directory from the authors'
``brexit_data_raw_eo_nov2018.mat``: log real GDP for twenty-four countries over
one hundred quarters, with the twelve countries the authors drop for missing
data already removed, and the UK marked as treated from 2016:Q3.

Window indices, the three SDID panel constructions, the placebo loop and the
evaluation offsets are all read from the authors' replication scripts —
``Treatment effects/Treatment effects - 2016Q3 no covariates - no penalty.R``
and ``In-sample across periods/In sample across periods - no covariates - no
penalty - {SC DSC SDID, MASC AUGSYNTH}.R``.

Both cases are deterministic: no resampling, and MASC's cross-validation grid is
exhaustive rather than sampled. Together they run in under ten seconds.

The Monte Carlo: why the demeaned estimators win
------------------------------------------------

Tables 1 and 7 say what the estimators claim and which of them tracks a
known-zero effect best. Neither can say why, because on real data the truth is
unobserved. Section 5 answers that by simulating.

The design calibrates the factor model of Abadie, Diamond and Hainmueller
(2010) to this same panel,

.. math::

   y_{jt} \;=\; \delta_t \;+\; \boldsymbol{\theta}_t' \mathbf{c}_j
   \;+\; \gamma_t \mu_j \;+\; \varepsilon_{jt},

where :math:`\mathbf{c}_j` holds unit :math:`j`'s six covariate means,
:math:`\delta_t` and :math:`\boldsymbol{\theta}_t` are fit to the real data, and
:math:`\gamma_t \mu_j` is an interactive fixed effect whose strength
:math:`\gamma` the study sweeps. The true effect is zero, so an estimate at the
final period is the estimation error, and the paper splits that error into the
part traceable to covariates, the part traceable to the common factor, and
idiosyncratic noise.

The result is about the middle term. As :math:`\gamma` rises, SC, MASC and ASCM
pick up a bias that grows with it; DSC and SDID barely move, because demeaning
absorbs the factor. Measuring the bias at :math:`\gamma = 0` and
:math:`\gamma = 1.5` and taking the difference:

.. list-table:: Bias attributable to the common factor
   :header-rows: 1
   :widths: 24 24 24 28

   * - Method
     - paper
     - mlsynth
     - ratio to SC (paper / mlsynth)
   * - SC
     - +0.362
     - +0.425
     - 1.0 / 1.0
   * - DSC
     - +0.073
     - +0.090
     - 5.0 / 4.7
   * - SDID
     - +0.068
     - +0.092
     - 5.3 / 4.6

The ratio is what the case asserts, not the level and not the slope. A bias
level carries a Monte Carlo standard error of roughly 0.08 at the case's
:math:`M = 100` -- larger than the levels themselves -- and even the slopes move
by about 0.05 between :math:`M = 100` and :math:`M = 200`. The ratio is stable
across both, because numerator and denominator share the same draws, and it is
what the paper's recommendation actually rests on: demeaning cuts the
factor-driven bias by a factor of roughly five.

Two checks, not one
~~~~~~~~~~~~~~~~~~~

A Monte Carlo aggregate on its own cannot separate a wrong DGP from a wrong
estimator; both show up as cells that miss. So the case carries two independent
checks.

The first is per-replication cross-validation. ``reference.R`` draws 48 panels
from the authors' DGP in R and records ``synthdid``'s own error on each.
mlsynth re-runs those same matrices and reproduces every error to within
:math:`5\times 10^{-3}` -- the documented exact-solve versus ``min.decrease``
gap, an order of magnitude below the biases being measured. That pins the
estimators without reference to the DGP.

The second is the aggregate table above, run on mlsynth's own port of the DGP
(:mod:`mlsynth.utils.sdid_helpers.simulate`). That pins the port without
reference to the estimators.

Where the authors' code and their prose disagree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three places, and the code is followed in each, because the code is what
produced the printed tables. All three are asserted in
``mlsynth/tests/test_sdid_simulation.py`` so the choices cannot be quietly
reverted.

The common factor is :math:`\gamma(1 - 1/t)` in the script, where the paper
writes :math:`t\gamma/T_0`. The noise is drawn with standard deviation
:math:`\sigma^2`, where the paper describes variance :math:`\sigma^2` -- and
this one is checkable rather than a judgment call, since the published
:math:`\sigma = 0.25` RMSEs are :math:`0.0625` times the :math:`\sigma = 1`
ones, which a standard deviation of :math:`0.25` cannot produce. Finally the
treated unit's loading :math:`\mu_1` is the second largest of the draws rather
than an independent uniform; that one is a deliberate design choice rather than
a slip, since it places the treated unit inside the donors' convex hull, so the
simplex weight problem is feasible without extrapolation.

What the Monte Carlo case leaves out
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``SC(B) all`` column, which needs the nested predictor-weight search per
replication and is the one expensive piece of this design. MASC and ASCM appear
in the aggregate check but not the cross-validation: the authors' ``masc``
package requires Gurobi, and ``augsynth`` 0.2.0 warns that its ridge tuning is
unstable when the unit and pre-period counts are close -- 24 and 24 here -- and
then errors out, so there is no reference to capture. mlsynth's ridge ASCM is
cross-validated against a live augsynth run in :doc:`ascm_kansas`, on a panel
where augsynth does run.
