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
   table reproduced cell by cell — **and Path B** — the concatenated simulation
   (Tian Table 1, also the Sun et al. ``Simulation1.R`` output) and the averaged
   regime contrast (Sun et al. Appendix D).
:Status: **Verified** — empirical balance and both simulations reproduced.

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
