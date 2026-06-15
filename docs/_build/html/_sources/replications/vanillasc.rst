.. _replication-vanillasc:

VanillaSC — Standard Synthetic Control (ADH 2010/2015; Abadie-Gardeazabal 2003)
===============================================================================

:Estimator: :doc:`../vanillasc` — :class:`mlsynth.VanillaSC`
:Source: Abadie, Diamond & Hainmueller (2010) [ABADIE2010]_; Abadie, Diamond &
   Hainmueller (2015) [Abadie2015]_; Abadie & Gardeazabal (2003) [ABADIE2003]_;
   leave-two-out placebo of Lei & Sudijono (2025).
:Replication type: **Path A** — the three canonical synthetic-control studies on
   their original datasets, plus the Lei-Sudijono (2025) Table-1 placebo
   relations.
:Status: **Fully verified** — donor pools, ATTs and (for LTO) the paper's
   p-value relations reproduce. Locked as regression tests in
   ``mlsynth/tests/test_vanillasc_replications.py``.

Three canonical SCM studies
---------------------------

Each is trained on its **full pre-treatment period**, from the datasets shipped
under ``basedata/``.

California / Proposition 99 (ADH 2010)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Treatment in 1989; pre-period 1970-1988. Covariates averaged over 1980-1988
(beer 1984-1988) plus three lagged cigarette-sales predictors (1975, 1980,
1988). With ``mscmt`` this reproduces ADH Table 2 almost exactly — Utah 0.335,
Nevada 0.236, Montana 0.202, Colorado 0.160, Connecticut 0.068 (ADH: 0.334 /
0.234 / 0.199 / 0.164 / 0.069) — and an ATT of about :math:`-19` packs.

.. code-block:: python

   import pandas as pd
   from mlsynth import VanillaSC

   d = pd.read_csv("basedata/augmented_cali_long.csv")
   for yr, col in [(1975, "cig_1975"), (1980, "cig_1980"), (1988, "cig_1988")]:
       d[col] = d.state.map(d[d.year == yr].set_index("state").cigsale)
   d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)

   res = VanillaSC({
       "df": d, "outcome": "cigsale", "treat": "treated",
       "unitid": "state", "time": "year",
       "backend": "mscmt", "canonical_v": "min.loss.w", "seed": 1,
       "covariates": ["p_cig", "pct15-24", "loginc", "pc_beer",
                      "cig_1975", "cig_1980", "cig_1988"],
       "covariate_windows": {"p_cig": (1980, 1988), "pct15-24": (1980, 1988),
                             "loginc": (1980, 1988), "pc_beer": (1984, 1988)},
       "display_graphs": False,
   }).fit()
   print(res.effects.att)                 # ~ -19
   print(res.weights.donor_weights)       # Utah/Nevada/Montana/Colorado/Connecticut

(In ``augmented_cali_long.csv`` the columns are labelled such that ``p_cig`` is
log GDP per capita and ``loginc`` is the retail price — the predictor means
reproduce ADH's Table 1 "Real California" column.)

German reunification (ADH 2015)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Treatment (reunification) in 1990; pre-period 1960-1990. GDP, trade, inflation
and industry share averaged over 1981-1990; investment rate and schooling over
1980-1985. With ``mscmt`` the synthetic West Germany is Austria-dominant with the
USA, Switzerland, Japan and the Netherlands — the ADH 2015 set — and a negative
ATT (reunification lowered per-capita GDP relative to the synthetic).

.. code-block:: python

   import pandas as pd
   from mlsynth import VanillaSC

   d = pd.read_stata("basedata/repgermany.dta")
   d["treated"] = ((d.country == "West Germany") & (d.year >= 1990)).astype(int)

   res = VanillaSC({
       "df": d, "outcome": "gdp", "treat": "treated",
       "unitid": "country", "time": "year",
       "backend": "mscmt", "seed": 1,
       "covariates": ["gdp", "trade", "infrate", "industry", "invest80", "schooling"],
       "covariate_windows": {"gdp": (1981, 1990), "trade": (1981, 1990),
                             "infrate": (1981, 1990), "industry": (1981, 1990),
                             "invest80": (1980, 1980), "schooling": (1980, 1985)},
       "display_graphs": False,
   }).fit()
   print(res.weights.donor_weights)       # Austria/USA/Switzerland/Japan/Netherlands

Basque terrorism (Abadie-Gardeazabal 2003)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The treatment indicator (terrorism) first turns on in **1975**, so the model
trains on the full **1955-1974** pre-period. On this long pre-period the problem
is well-conditioned and the synthetic Basque is **Cataluna :math:`\approx 0.8`,
Madrid :math:`\approx 0.2`** — the published Abadie-Gardeazabal result — with an
ATT of about :math:`-0.68` (the roughly 10% per-capita GDP gap). Outcome-only
already recovers this; ``mscmt`` with the special-predictor covariates confirms
it.

.. note::

   This is instructive: on the *short* 1960-1969 window used by some later papers
   the Basque donor weights are fragile (they drift to Baleares/Madrid), but on
   the full 1955-1974 pre-period the long outcome path pins :math:`W` to the
   Cataluna/Madrid solution. The training window matters; ``VanillaSC`` uses the
   full pre-period defined by the treatment indicator.

.. code-block:: python

   import pandas as pd
   from mlsynth import VanillaSC

   b = pd.read_csv("basedata/basque_data.csv")
   b = b[b.regionno != 1]                                  # drop Spain
   b["treated"] = ((b.regionno == 17) & (b.year >= 1975)).astype(int)

   res = VanillaSC({
       "df": b, "outcome": "gdpcap", "treat": "treated",
       "unitid": "regionno", "time": "year",
       "backend": "outcome-only", "display_graphs": False,
   }).fit()
   print(res.effects.att)                 # ~ -0.68
   print(res.weights.donor_weights)       # region 10 (Cataluna) ~0.8, 14 (Madrid) ~0.2

Leave-two-out placebo: Lei & Sudijono (2025) Table 1
----------------------------------------------------

With ``inference="lto"`` VanillaSC runs the Lei-Sudijono (2025) refined placebo.
Their Table 1 (covariate-matched Synth, :math:`\alpha = 0.05`) lays out how the
methods relate across the three canonical datasets, which mlsynth reproduces:

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14

   * - quantity
     - Prop 99
     - Basque
     - German
   * - :math:`N`
     - 39
     - 17
     - 17
   * - :math:`p_{\text{app-placebo}}`
     - 0.00
     - 0.35
     - 0.00
   * - :math:`p_{\text{exact-placebo}}`
     - 0.026
     - 0.41
     - 0.059
   * - :math:`p_{\mathrm{naive\text{-}LTO}}`
     - 0.024
     - 0.67
     - 0.042
   * - :math:`p_{\mathrm{powered\text{-}LTO}}(\alpha)`
     - 0.022
     - 0.66
     - 0.03
   * - :math:`\Gamma_{\mathrm{LTO}}`
     - 1.4
     - NA
     - 1.1

Three relations are worth internalising:

* **LTO can change the conclusion (German).** The exact placebo p-value of 0.059
  does *not* reject at 0.05, but the naive LTO (0.042) and powered LTO (0.03)
  both do. With only 16 donors the placebo grid is too coarse to resolve a
  borderline effect; LTO's finer grid does. The small :math:`\Gamma = 1.1`,
  though, warns that this significance is fragile to mild departures from uniform
  assignment.
* **LTO is not mechanically smaller (Basque).** Here LTO (0.67) is *larger* than
  the placebo (0.41); nothing is significant by any method. The refinement
  changes granularity, not direction — it does not manufacture significance.
* **LTO ≈ placebo when both already reject (Prop 99).** The two p-values (0.024
  vs 0.026) nearly coincide; the powered version (0.022) buys a little extra
  margin, and :math:`\Gamma = 1.4` says the conclusion survives moderate
  confounding.

The Lei-Sudijono helper constants reproduce the paper's reported values exactly
(``c(39, 0.05) = 0.002``, ``c(17, 0.05) = 0.0125``; see
``test_lto_helpers_match_paper``), and the covariate-matched ordinary placebo
reproduces California's exact-placebo p-value (rank 1 of 39, :math:`p = 0.0256`
vs the paper's 0.026). The p-value tracks the chosen specification: the
covariate-matched Synth concentrates the effect on California
(:math:`p_{\mathrm{naive\text{-}LTO}} \approx 0.024`), whereas the
``outcome-only`` fit — where California is only rank 3 of 39 — gives
:math:`\approx 0.10`; both are internally consistent with their respective
ordinary placebo p-values. Choose the specification before reading the test.
