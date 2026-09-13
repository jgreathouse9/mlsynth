.. _replication-scmo:

SCMO — Synthetic Control with Multiple Outcomes (Tian et al. 2026; Sun et al. 2025)
===================================================================================

:Estimator: :doc:`../scmo` — :class:`mlsynth.SCMO`
:Source: Tian, W., Lee, S., & Panchenko, V. (2026), *"Synthetic controls with
   multiple outcomes,"* Econometrics Journal (the **concatenated** variant); and
   Sun, L., Ben-Michael, E., & Feller, A. (2025), *"Using Multiple Outcomes to
   Improve the Synthetic Control Method,"* Review of Economics and Statistics
   (the **averaged** variant).
:Replication type: **Path A** — Tian et al.'s German-reunification balance
   table and their Sweden NPI application, both reproduced cell by cell — **and
   Path B** — the concatenated simulation (Tian Table 1, also the Sun et al.
   ``Simulation1.R`` output), the demeaned simulation (Tian Table B.1) and the
   averaged regime contrast (Sun et al. Appendix D).
:Status: **Verified** — both applications and all three simulations reproduced.

Validation strategy
-------------------

The two SCMO papers share the German-reunification illustration and tell the
same story from two angles: matching the synthetic control on *several related
outcomes* — not a single long outcome trajectory — sharpens the
identification of the latent factors and reduces post-treatment bias. The
``concatenated`` scheme (Tian-Lee-Panchenko, following the same stacking as Sun
et al.) stacks the standardized pre-period outcomes; the ``averaged`` scheme
(Sun-Ben-Michael-Feller) matches their per-period average, which cancels
idiosyncratic noise when the outcomes share a common factor.

Path A — German reunification balance (Tian et al. Table 2)
-----------------------------------------------------------

On ``basedata/germany_augmented.csv`` (West Germany + 16 OECD donors), SCMO
matches West Germany to the donors on nine economic indicators in the single
year 1989. The fit reproduces all 36 cells of Tian et al.'s printed 1989 balance
table — the treated unit, the synthetic West Germany built on the nine outcomes,
the one built on 30 years of GDP per capita, and the comparison-group simple
average:

.. list-table::
   :header-rows: 1
   :widths: 30 16 18 18 14

   * - Outcome (1989)
     - West Germany
     - Synthetic (multiple)
     - Synthetic (single)
     - Sample mean
   * - Private social expenditure
     - 3.4
     - 3.5
     - 3.7
     - 2.0
   * - Energy supply per GDP
     - 0.2
     - 0.1
     - 0.1
     - 0.1
   * - Electricity generation
     - 9.0
     - 8.7
     - 10.1
     - 7.6
   * - Triadic patent families
     - 0.1
     - 0.1
     - 0.0
     - 0.0
   * - Real GDP growth
     - 3.9
     - 4.1
     - 3.5
     - 3.5
   * - CPI
     - 2.8
     - 3.1
     - 4.0
     - 5.5
   * - Trade openness
     - 57.7
     - 59.1
     - 59.3
     - 60.4
   * - Total tax revenue
     - 36.2
     - 34.1
     - 32.9
     - 33.7
   * - GDP per capita
     - 18994.0
     - 19029.8
     - 19075.9
     - 16493.8

The reference side is a live captured run of the authors' own ``Germany.R``
(their ``fn_W`` :math:`\texttt{solve.QP}` program), kept under
``benchmarks/reference/scmo_germany/`` with its provenance pinned. mlsynth's two
synthetic-control columns, reconstructed from ``res.donor_weights``, agree with
that run to :math:`1.8 \times 10^{-4}` relative at the worst cell; the two data
columns — West Germany's own values and the donor average — agree to floating
point, which ties mlsynth's spec transforms (the per-capita normalizations, the
GDP and trade joins) to the authors' read of ``all.xlsx``.

The paper reads the table as both synthetic controls sitting much closer to West
Germany than the simple average does. Counted over the nine outcomes, the
multiple-outcomes synthetic is closer on all nine and the single-outcome
synthetic on eight, missing on total tax revenue; and the multiple-outcomes
synthetic is closer than the single-outcome one on all nine. Matching on the
nine 1989 outcomes balances every one of them better than matching on 30 years
of the GDP path.

The concatenated SC — fit on one year's nine indicators, never shown the GDP
path — tracks West Germany's pre-1990 GDP to a root-mean-squared error of
:math:`110` (vs. :math:`74` for the conventional SC fit directly to 30 years of
GDP). The post-1990 effect is reported only graphically in the paper (Tian et
al. Figure 1), so no ATT number is asserted against the paper; mlsynth's
deterministic ATTs (concatenated :math:`-1463`, averaged :math:`-1720`) ride
along as regression guards. Durable case: ``scmo_germany``.

Path A — Sweden's light-touch NPIs (Tian et al. Appendix B.3)
--------------------------------------------------------------

The paper's second application, and the one the method was built for. Sweden
did not impose the strict non-pharmaceutical interventions its neighbours
adopted in March 2020, so a synthetic Sweden built from countries that did
estimates what those interventions would have done. There is no long
pre-treatment series to match on — the pandemic is weeks old — so the synthetic
control is matched on several outcomes at once, in three domains estimated
separately: public health (COVID-19 cases, COVID-19 deaths, deaths from all
causes), the labour market (employment, absence from work, hours worked), and
the economy (GDP, imports, exports, industrial production, retail sales, CPI).

The application exercises three things the appendix adds to the main text, all
of which SCMO now carries: outcomes observed at four frequencies share one
panel (daily cases matched alongside quarterly GDP), each outcome is matched
after centering on its own pre-treatment mean (``demean=True``), and each
outcome carries the same total weight in the objective however often it is
observed (``metric_weighting="outcome"``). Inference is the permutation test on
the post-to-pre-treatment RMSPE ratio (``inference="placebo"``), one-sided, with
the guard :math:`\eta = 0.01\sigma_k`.

mlsynth reproduces all 78 cells of the paper's Table B.3 — the synthetic
control weights of 26 donors in each of the three domains — to within
:math:`0.005`, against a table printed to two decimals. Sweden's public-health
synthetic is the Netherlands :math:`0.31`, Denmark :math:`0.26`, Finland
:math:`0.20`, Poland :math:`0.09`, Norway :math:`0.07`, France and Greece
:math:`0.03`, Italy :math:`0.02`; its labour-market and economic synthetics
reproduce cell for cell in the same way.

The effect magnitudes the appendix reports in text come back with them:

.. list-table::
   :header-rows: 1
   :widths: 44 28 28

   * - Quantity
     - Tian et al.
     - mlsynth
   * - Cumulative COVID-19 cases by July, per million
     - −5,300 (−70%)
     - −5,347 (−70.1%)
   * - Cumulative COVID-19 deaths by July, per million
     - −390 (−68%)
     - −389 (−68.2%)
   * - Cumulative all-cause deaths since April, per million
     - −364 (−11%)
     - −368 (−11.6%)
   * - Weekly all-cause deaths at the peak
     - −20%
     - −20.6%
   * - Absence from work, 2020 Q2
     - +76%
     - +75.9%
   * - Hours worked, 2020 Q2
     - −12%
     - −12.2%
   * - Employment, 2020 Q2 and Q3
     - no visible effect
     - +0.3%, +0.7%
   * - Retail sales, March to May
     - −5% to −13%
     - −6.6%, −13.4%, −5.0%
   * - GDP, imports, exports, industry, CPI
     - close to zero
     - at most 6.1% in absolute value

The significance pattern of Figure B.6 reproduces as well, at the paper's own
threshold :math:`\alpha = 3/(J+1)` (the treated unit among the three largest
RMSPE ratios): cases and deaths significant from May, deaths from all causes
from April to June, absence from work and hours worked in the second quarter,
employment never, retail sales in March alone, and no other economic outcome at
any point. One divergence: COVID-19 deaths reach the threshold here in April
too, a month before the paper's figure reads, on a rank-three tie. The
aggregate treatment effects and aggregate p-values (Figures B.5 and B.7) carry
their numbers inside the plots, so nothing is asserted against them. Durable
case: ``scmo_covid_sweden``.

Path B — concatenated simulation (Tian et al. Table 1)
------------------------------------------------------

Tian et al.'s Section-3 factor model — identical to the Sun et al. replication
package's ``Simulation1.R`` — draws :math:`N = 30` units whose outcomes share the
unit predictors. As the number of related outcomes :math:`K` grows, the
post-treatment bias falls while the pre-treatment fit rises toward the true
noise floor (not overfitting to near-zero). mlsynth reproduces all 36 cells of
Table 1 — the pre-treatment fit, the average absolute bias, and the standard
deviation of the post-period gap, for each of the four estimators at
:math:`T_0 \in \{1, 5, 10\}`. At :math:`T_0 = 5`, across the conventional SC,
the five- and ten-outcome SC and the augmented SC, the bias is
:math:`1.21 / 1.04 / 1.00 / 0.97`, the pre-fit :math:`0.46 / 0.95 / 1.02 / 0.95`
and the gap's standard deviation :math:`1.53 / 1.31 / 1.26 / 1.22`. The run uses
:math:`M = 250` draws (the paper uses 5,000).

The fourth estimator is the paper's augmented SC: the same ten-outcome design
under a ridge-augmented fit (``augment="ridge"``, the penalty chosen by
cross-validation), which the paper reports as cutting the bias further where the
pre-treatment fit is imperfect. mlsynth lands every cell of that column within
:math:`0.07` of the printed value, the same distance as the three plain columns,
though the two implementations differ in one step: mlsynth standardizes the
matching columns by their cross-unit SD and ``augsynth``, which the authors
call, works on the raw stacked series. On identical draws the augmentation
lowers the ten-outcome SC's bias at all three :math:`T_0` (by
:math:`0.014 / 0.023 / 0.030`, the size of the gain the paper prints) and leaves
the pre-treatment fit above the single-outcome floor. The DGP lives in
:func:`mlsynth.utils.scmo_helpers.simulation.simulate_tian`. Durable case:
``scmo_concatenated_mc``.

Path B — demeaned simulation (Tian et al. Table B.1)
----------------------------------------------------

The Online Appendix repeats the Monte Carlo under a DGP where matching on
levels fails: each outcome carries a large mean of its own, and a parameter
:math:`d` places the treated unit, which at :math:`d = 1` is as likely as a
donor to take an extreme predictor value and so to fall outside the donors'
convex hull. Four estimators — one outcome in levels, one demeaned, and three
and ten demeaned outcomes — are compared over :math:`d \in \{1, 0.5, 0\}` and
:math:`T_0 \in \{5, 10, 20\}`, on four statistics: pre-treatment fit, average
absolute bias, the standard deviation of the gap, and the rejection rate of the
10% permutation test. Under the null DGP that last column is the test's size,
and anything above :math:`0.10` is size distortion.

mlsynth reproduces all 144 cells at :math:`M = 100` draws (the paper uses
5,000), and the appendix's three readings of them hold exactly:

1. demeaning improves the pre-treatment fit in all nine settings, and lowers the
   bias at :math:`d = 1`, where the treated unit is as extreme as the donors;
2. the test holds its nominal size at :math:`d = 1`, and distorts as :math:`d`
   falls — the treated unit fits its donors better, its pre-treatment RMSPE
   shrinks and its ratio grows — while demeaning and more outcomes pull the size
   back toward 10%;
3. more pre-treatment periods reduce the distortion too.

The DGP lives in
:func:`mlsynth.utils.scmo_helpers.simulation.simulate_tian_demeaned` and the
test in :func:`mlsynth.utils.scmo_helpers.inference.permutation_inference`.
Durable case: ``scmo_demeaned_mc``.

Path B — averaged regime contrast (Sun et al. Appendix D)
---------------------------------------------------------

Sun et al. report their Monte Carlo as box plots (Figures D.1, D.2), so the
benchmark matches the **published geometry**, not numeric cells. Under a
common factor shared across outcomes (``rho = 1``), the multi-outcome schemes
beat the separate single-outcome SC and **averaging reduces bias**; under purely
idiosyncratic factors (``rho = 0``), the outcomes share no signal and
**averaging hurts** (the separate SC is best). mlsynth reproduces this
common-vs-idiosyncratic adaptivity (common :math:`T_0 = 10, K = 10`: averaged
bias :math:`0.91` < separate :math:`1.03`; idiosyncratic: averaged :math:`1.23`
> separate :math:`1.00`). The DGP lives in
:func:`mlsynth.utils.scmo_helpers.simulation.simulate_sun`. Durable case:
``scmo_averaged_mc``.
