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
outcomes* — rather than a single long outcome trajectory — sharpens the
identification of the latent factors and reduces post-treatment bias. The
``concatenated`` scheme (Tian-Lee-Panchenko, following the same stacking as Sun
et al.) stacks the standardized pre-period outcomes; the ``averaged`` scheme
(Sun-Ben-Michael-Feller) matches their per-period average, which cancels
idiosyncratic noise when the outcomes share a common factor.

Path A — German reunification balance (Tian et al. Table 2)
-----------------------------------------------------------

On ``basedata/germany_augmented.csv`` (West Germany + 16 OECD donors), SCMO
matches West Germany to the donors on **nine economic indicators in the single
year 1989**. The fitted synthetic West Germany reproduces Tian et al.'s printed
1989 balance table cell by cell — for the concatenated (multiple-outcomes)
synthetic, reconstructed from ``res.donor_weights``:

.. list-table::
   :header-rows: 1
   :widths: 30 18 22 14

   * - Outcome (1989)
     - West Germany
     - Synthetic (multiple)
     - mlsynth
   * - GDP per capita
     - 18994.0
     - 19029.8
     - 19029.8
   * - CPI
     - 2.8
     - 3.1
     - 3.06
   * - Trade openness
     - 57.7
     - 59.1
     - 59.06
   * - Total tax revenue
     - 36.2
     - 34.1
     - 34.07
   * - Real GDP growth
     - 3.9
     - 4.1
     - 4.12

The single-outcome synthetic reproduces the "Synthetic (single outcome)" column
(1989 GDP per capita :math:`19075.9`). The concatenated SC — fit on one year's
nine indicators, never shown the GDP path — tracks West Germany's pre-1990 GDP to
a root-mean-squared error of :math:`110` (vs. :math:`74` for the conventional SC
fit directly to 30 years of GDP). The post-1990 effect is reported only
graphically in the paper (Tian et al. Figure 1), so no ATT number is asserted
against the paper; mlsynth's deterministic ATTs (concatenated :math:`-1463`,
averaged :math:`-1720`) ride along as regression guards. Durable case:
``scmo_germany``.

Path B — concatenated simulation (Tian et al. Table 1)
------------------------------------------------------

Tian et al.'s Section-3 factor model — identical to the Sun et al. replication
package's ``Simulation1.R`` — draws :math:`N = 30` units whose outcomes share the
unit predictors. As the number of related outcomes :math:`K` grows, the
post-treatment **bias falls** while the **pre-treatment fit rises** toward the
true noise floor (rather than overfitting to near-zero). mlsynth reproduces all
nine cells of Table 1 across :math:`T_0 \in \{1, 5, 10\}` and
:math:`K \in \{1, 5, 10\}` (e.g. at :math:`T_0 = 5` the bias is
:math:`1.21 / 1.04 / 1.00`, pre-fit :math:`0.46 / 0.95 / 1.02`), at
:math:`M = 250` draws (the paper uses 5,000). The DGP lives in
:func:`mlsynth.utils.scmo_helpers.simulation.simulate_tian`. Durable case:
``scmo_concatenated_mc``.

Path B — averaged regime contrast (Sun et al. Appendix D)
---------------------------------------------------------

Sun et al. report their Monte Carlo as box plots (Figures D.1, D.2), so the
benchmark matches the **published geometry** rather than numeric cells. Under a
common factor shared across outcomes (``rho = 1``), the multi-outcome schemes
beat the separate single-outcome SC and **averaging reduces bias**; under purely
idiosyncratic factors (``rho = 0``), the outcomes share no signal and
**averaging hurts** (the separate SC is best). mlsynth reproduces this
common-vs-idiosyncratic adaptivity (common :math:`T_0 = 10, K = 10`: averaged
bias :math:`0.91` < separate :math:`1.03`; idiosyncratic: averaged :math:`1.23`
> separate :math:`1.00`). The DGP lives in
:func:`mlsynth.utils.scmo_helpers.simulation.simulate_sun`. Durable case:
``scmo_averaged_mc``.
