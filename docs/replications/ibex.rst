.. _replication-ibex:

VanillaSC -- the Iberian exception on the day-ahead electricity price
=====================================================================

:Estimator: :doc:`../vanillasc` -- :class:`mlsynth.VanillaSC`
:Source: Haro Ruiz, M., Schult, C. and Wunder, C. (2024), *"The effects of the
   Iberian exception mechanism on wholesale electricity prices and consumer
   inflation: a synthetic-controls approach,"* Applied Economics Letters
   33(9):1310-1316, DOI 10.1080/13504851.2024.2425834.
:Replication type: cross-validation against the authors' own public replication
   code (``github.com/mharoruiz/ibex``) on the day-ahead-price outcome.
:Status: verified -- the donor weights and the per-period gap reproduce the
   ibex synthetic control value-for-value (weight L1 distance :math:`\approx 0`,
   gap agreement to ~1e-8).

What the paper does
-------------------

In June 2022 Spain and Portugal introduced the Iberian exception mechanism
(IbEx), a cap on the gas price that electricity generators may bid into the
day-ahead market. Haro Ruiz, Schult and Wunder (2024) estimate its effect with a
standard synthetic control: each treated country is compared with a synthetic
counterpart built from a weighted average of untreated European countries that
tracks it before the intervention.

The authors follow Ferman, Pinto and Possebom (2020) and match on all
pre-treatment outcome lags rather than a hand-picked set of covariates. Their
replication code (``01_functions/sc.R``) solves the canonical
Abadie-Diamond-Hainmueller program: choose donor weights
:math:`\mathbf{w}` on the simplex (non-negative, summing to one) that minimise
the pre-treatment fit to the treated series, using ``limSolve::lsei``. That is
exactly what mlsynth's :class:`~mlsynth.VanillaSC` computes with
``backend="outcome-only"`` -- the well-posed convex problem, no predictor
weights.

Validation strategy
-------------------

Spain (ES) and Portugal (PT) are both treated in June 2022, and the paper
excludes each treated country from the other's donor pool ("we exclude Portugal
... and vice versa"). mlsynth's multi-treated design drops every treated unit
from the shared donor pool, so for these two mutually-excluded units a single
:class:`~mlsynth.VanillaSC` fit reproduces the paper's two per-country loops
against the identical 20-country pool. We fit once, on the paper's own day-ahead
data (``basedata/ibex_day_ahead_price.csv``, monthly, 2015--2023, treatment in
June 2022), and compare to the ibex outputs
(``03_results/sc_optimized_weights.csv`` and ``03_results/sc_series_001.csv``,
captured at commit ``c5371314``).

Cross-validation -- weights and gap
-----------------------------------

Because both implementations solve the same simplex program on the same data,
the donor weights coincide to solver tolerance, and with identical weights and
donor data the per-period gap coincides pointwise:

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * - Quantity
     - VanillaSC (mlsynth)
     - ibex (lsei / scinference)
   * - Spain -- Slovenia (SI) weight
     - 0.526
     - 0.526
   * - Portugal -- Slovenia (SI) weight
     - 0.516
     - 0.516
   * - Spain -- donor-weight L1 distance
     - :math:`\approx 0`
     - 0
   * - Portugal -- donor-weight L1 distance
     - :math:`\approx 0`
     - 0
   * - Spain -- mean post ATT (Euros/MWh)
     - :math:`-46.4`
     - :math:`-46.4`
   * - Portugal -- mean post ATT (Euros/MWh)
     - :math:`-45.5`
     - :math:`-45.5`

Both countries show a large cut in the day-ahead price -- about 40% below the
synthetic counterfactual over the first year -- which is the paper's headline
finding. The two treated series load on the same donors (Slovenia, Belgium,
Greece, the Netherlands, Italy, Latvia, Sweden), with Slovenia dominant.

Scope
-----

The paper reports four outcomes: the day-ahead price and three consumer-price
indices (energy, non-energy and all-items HICP at constant taxes). The three CPI
series are pulled live from Eurostat (``prc_hicp_cind``) at run time and are not
redistributed with the ibex repository, so this replication pins the offline,
fully reproducible day-ahead-price result. The CPI outcomes use the same
estimator and the same specification; only the input data differs.

A note on the snapshot: the ibex ``sc_series`` output extends the day-ahead
series to March 2024, while the committed input snapshot ends December 2023. The
donor weights depend only on the shared January-2015 to May-2022 pre-treatment
window, so they match exactly; the average post-treatment gap reported above is
taken over the common window.

Verification
------------

The check is committed as the ``ibex_dap`` benchmark case
(`benchmarks/cases/ibex_dap.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/ibex_dap.py>`__),
which runs :class:`~mlsynth.VanillaSC` on the shipped day-ahead panel and
asserts the weights, the dominant-donor weight and the post-treatment ATT
against the captured ibex reference under
``benchmarks/reference/ibex_dap/``. It appears on the :doc:`../validation`
dashboard under VanillaSC.
