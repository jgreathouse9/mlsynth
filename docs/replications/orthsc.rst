.. _replication-orthsc:

ORTHSC -- Fry's Orthogonalized Synthetic Control (carbon tax + Monte Carlo)
===========================================================================

:Estimator: :doc:`../orthsc` -- :class:`mlsynth.ORTHSC`
:Source: Fry, J. (2026). "Orthogonalized Synthetic Controls"
   (`arXiv:2510.22828 <https://arxiv.org/abs/2510.22828>`_), with the author's
   R reference (``JosephPatrickFry/OrthogonalizedSyntheticControl``).
:Replication type: Path A (the paper's empirical result on Andersson's data)
   and Path B (the paper's size/power simulation), plus a live cross-check
   against the author's R code.
:Status: Done -- the empirical ATT, p-value, smoothing parameter, and CI
   reproduce the live R reference to the digit; the simulation reproduces the
   size-control / power pattern of Tables 1-2.
:Durable check: ``benchmarks/cases/orthsc_carbontax.py`` (``orthsc_carbontax``)
   and ``benchmarks/cases/orthsc_size_power.py`` (``orthsc_size_power``); plus
   ``mlsynth/tests/test_orthsc.py``.

The empirical target
--------------------

Fry's working example applies the Orthogonalized SC to Andersson (2019)'s
Swedish carbon-tax panel. The point estimate matches Andersson's synthetic
control -- an average reduction of :math:`0.29` metric tons of transport
CO\ :sub:`2` per capita over 1990-2005 -- but where Andersson's placebo test
(and, the paper shows, conformal and cross-fitting inference) fails to clear
conventional significance, the orthogonalized t-test returns ``p = 0.00018``.

The setup that makes this work is the donor split, and it took the Andersson
paper to pin it down. Andersson's donor pool is 14 OECD countries; he excluded
Finland, Germany, Ireland, Italy, Netherlands, Norway, and the United Kingdom
(carbon or large fuel-tax changes) and Austria, Luxembourg, Turkey (fuel
tourism / outliers). Fry's method uses the *excluded* carbon/fuel-tax countries
as instruments -- so the ORTHSC control pool is Andersson's 14, and the
instruments are his 7 excluded tax countries. With that split the public
estimator reproduces the headline:

.. code-block:: python

   import pandas as pd
   from mlsynth import ORTHSC

   url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/"
          "basedata/carbontax_fullsample_data.dta.txt")
   df = pd.read_stata(url).rename(columns={"CO2_transport_capita": "Y"})
   df["treat"] = ((df.country == "Sweden") & (df.year >= 1990)).astype(int)

   res = ORTHSC({
       "df": df, "outcome": "Y", "treat": "treat", "unitid": "country",
       "time": "year",
       "controls": ["Australia", "Belgium", "Canada", "Denmark", "France",
                    "Greece", "Iceland", "Japan", "New Zealand", "Poland",
                    "Portugal", "Spain", "Switzerland", "United States"],
       "instruments": ["Finland", "Germany", "Ireland", "Italy", "Netherlands",
                       "Norway", "United Kingdom"],
       "display_graphs": False,
   }).fit()
   print(res.att, res.inference.p_value,
         res.method_details.parameters_used["smoothing_K"])
   # -0.29013   0.000183   4

Live cross-check vs the author's R
----------------------------------

The author's R reference (``OrthoganilzedSCE`` over ``RegularizedEstimate.R`` /
``SeriesHAC.R``) was run on the same panel. It installs with no CRAN access --
``limSolve`` / ``corpcor`` / ``comprehenr`` compile from the GitHub CRAN
mirrors against the system R -- and returns

.. list-table::
   :header-rows: 1
   :widths: 30 24 24

   * - Quantity
     - mlsynth (NumPy/cvxpy)
     - R reference
   * - ATT :math:`\widehat{\tau}`
     - :math:`-0.29013`
     - :math:`-0.29013`
   * - p-value
     - :math:`0.000183`
     - :math:`0.000183`
   * - smoothing :math:`K`
     - :math:`4`
     - :math:`4`
   * - 95% CI
     - :math:`[-0.4757,\,-0.1045]`
     - :math:`[-0.476,\,-0.105]`

The control weights differ slightly between the two (``limSolve``'s ``ldei``
and cvxpy's ``CLARABEL`` land on different, equally valid elements of the
identified set), but the ATT and its p-value match to the digit. That is the
paper's central claim made operational: the Neyman-orthogonalized
:math:`\widehat{\tau}` is first-order insensitive to the nuisance weights, so a
faithful port does not need to bit-match the reference's weight solver.

The simulation (Tables 1-2)
---------------------------

Fry's simulation study shows the orthogonalized t-test controls size where
naive IV-SC, cross-fitting, and ArCo over-reject, while keeping high power.
``orthsc_size_power`` reproduces that behaviour on a linear-factor DGP (the
treated unit a convex mix of controls plus idiosyncratic noise; instruments
sharing the factors but independent of the treated unit's shocks). At the
:math:`5\%` level over 200 replications:

.. list-table::
   :header-rows: 1
   :widths: 12 12 22 22

   * - :math:`T_0`
     - :math:`T_2`
     - size (effect = 0)
     - power (effect = :math:`-0.25`)
   * - 30
     - 16
     - 0.070
     - 0.655
   * - 30
     - 32
     - 0.035
     - 0.880
   * - 60
     - 32
     - 0.060
     - 0.960

Size sits at or below the nominal level (up to Monte Carlo noise) and power
rises with the post-period length -- the qualitative content of the paper's
Tables 1 and 2. The runnable grid is in :doc:`../orthsc` (Verification).

Lessons carried into the port
-----------------------------

* The whole replication hinged on the donor split, not the numerics: the wrong
  pool gave ATTs of :math:`-0.17` / :math:`-0.48`; Andersson's 14-control /
  7-instrument split gave :math:`-0.29013` exactly.
* The reference's ``SeriesHAC.R`` demeans only the last moment row (an
  off-by-one in the loop bound); because the sample moments are zero by
  construction it is harmless, and the port demeans all rows and still matches.
* The t-statistic carries a :math:`\sqrt{n}` factor while the CI uses
  :math:`\sqrt{\widehat{V}}` directly; both are reproduced as written so the
  p-value and the interval agree with the reference.
