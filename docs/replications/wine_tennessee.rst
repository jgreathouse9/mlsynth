.. _replication-wine-tennessee:

Tennessee's wine reform (Sun et al. 2025)
==========================================

:Estimators: :doc:`../vanillasc` — :class:`mlsynth.VanillaSC`; :doc:`../sdid` —
   :class:`mlsynth.SDID`
:Source: Sun, Jiayu, Vincenzina Caputo, Trey Malone & Christos Vassilopoulos
   (2025), *"When Wine Walked into Grocery Stores: Examining the (Un)intended
   Economic Consequences of Wine Sale Reform in Tennessee,"* American Journal
   of Agricultural Economics. `doi:10.1111/ajae.70033
   <https://doi.org/10.1111/ajae.70033>`_.
:Replication type: Path A — the authors' empirical results on the authors' own
   data (`OSF rbt4n <https://osf.io/rbt4n>`_).
:Status: Two of three studies reproduced; the third cannot be, and the reason is
   the data, not the method.
:Case: `benchmarks/cases/wine_tennessee.py
   <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/wine_tennessee.py>`_

What the paper asks
-------------------

From July 2016 Tennessee allowed grocery stores to sell wine, which until then
only liquor stores could do. The intended consequence was more wine sold and so
more excise tax revenue. The possible unintended consequence was damage to the
liquor stores that had held the monopoly. The paper measures both, and the
interest for a synthetic control library is that it estimates each with two
different estimators, not one, so a replication exercises two code paths
against the same underlying question.

Tennessee is a single treated unit with a clean adoption date, which is the
canonical synthetic control setting: build a weighted combination of untreated
states that tracks Tennessee before the reform, and read the effect off the gap
afterwards.

What can be replicated
----------------------

.. list-table::
   :header-rows: 1
   :widths: 8 34 30 28

   * - study
     - outcome
     - panel
     - status
   * - 1
     - liquor store establishments
     - proprietary NielsenIQ
     - not replicable
   * - 2
     - liquor store employment (QCEW)
     - 11 states x 40 quarters
     - reproduced
   * - 3
     - wine excise tax revenue
     - 6 states x 6 fiscal years
     - reproduced

Study 1's outcome series is licensed NielsenIQ Retail Measurement data, which
the authors are not free to redistribute; the replication package says so
explicitly. That also rules out the first column of the paper's Table 6. It is
recorded here, not passed over, because "two of three" is the honest
description of this replication.

The other two panels ship with the package and are vendored under
``benchmarks/reference/wine_tennessee/`` with their provenance.

How the checks are layered
--------------------------

Three layers, and the layering carries the argument.

The oracle layer takes the paper's own published donor weights, feeds them
through mlsynth's counterfactual arithmetic, and asks for the paper's own
per-period effects. It reproduces Table 4 quarter by quarter and Table 5 year by
year to within 0.01. Because the weights are held fixed, nothing in the
predictor-weight search can disturb this layer, so it isolates the data handling
and the arithmetic and pins them on their own. If a fitted check below ever
drifts while this still holds, the drift is in the weight search and nowhere
else.

The fitted layer runs the estimators for real, with tolerances that are stated
and defended.

The recorded layer states what the data do not determine, instead of pinning it.

Path A — the numbers
--------------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - quantity
     - published
     - mlsynth
     - difference
   * - Table 4, SCM employment, average
     - -42.500
     - -40.500
     - 2.000
   * - Table 5, SCM tax revenue, average
     - 0.589
     - 0.589
     - 0.001
   * - Table 6 B, SDID employment, average
     - -53.154
     - -53.170
     - 0.016
   * - Table 6 C, SDID tax revenue, average
     - 0.610
     - 0.610
     - 0.0004

The SDID rows use Stata's default covariate method. The paper's call is
``sdid ..., covariates(...)`` with no type named, and that default is
``optimized``, not ``projected`` — a distinction that decides which estimator
runs. See :doc:`../sdid` for the two.

Why one row is looser than the others
-------------------------------------

Study 2's SCM lands at -40.500 against a published -42.500, and that gap is
recorded.

The donor set is the published one, and the weights agree to about a percentage
point:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - donor
     - published
     - mlsynth
     - difference
   * - KY
     - 0.231
     - 0.221
     - 0.010
   * - NY
     - 0.551
     - 0.558
     - 0.007
   * - UT
     - 0.218
     - 0.221
     - 0.003

Small weight differences move a ten-quarter counterfactual by more than they
look like they should, which is where the two units go. The question is whether
mlsynth's weights are worse, and they are not: they achieve a pre-treatment RMSE
of 14.52 against the published weights' 14.74. Stata's ``synth ... nested
allopt`` and mlsynth's differential-evolution predictor-weight search are two
optimisers on a non-convex nested problem, and mlsynth's lands on a
better-fitting local optimum. Pinning the published figure tightly would amount
to asserting that Stata's optimum is the correct one.

So the case pins the RMSE comparison as well as the effect. The loose tolerance
has a justification, and the justification has to keep being true or the case
fails.

The fitted search is deterministic here: five different seeds give the same
weights and the same effect to three decimals, so the gap is systematic, not a
draw.

What the data do not determine
------------------------------

Study 3's donor weights are not pinned, and the reason is a property of the
panel, not of the estimator.

It has three pre-treatment fiscal years and five donors. Searching every donor
subset under the simplex constraint, the best attainable pre-treatment RMSE is
0.0413; the published weights give 0.0439. The gap is under a tenth of one
percent of the outcome's own standard deviation. Several weight vectors fit
about equally well, the published one is not the best of them, and with three
periods to match there is not enough information to choose. Demanding the
published weights would be demanding one arbitrary point from a flat region, and
such a test would fail on any harmless change to the search while proving
nothing.

The effect is a different matter: it is identified, it reproduces to 0.001, and
it is pinned. What is asserted instead of the weights is the non-identification
itself — that the published weights are *not* the best pre-treatment fit — so
that if the situation ever changes, the case fails and this page gets revisited.

A reader of the paper cannot see this from
the published figures. A synthetic control with a near-perfect pre-treatment fit
on three periods looks convincing, and a weight vector reported to three decimals
looks determined. Neither impression survives the check above.

A defect this replication found
-------------------------------

Reproducing Table 6 was blocked on the ``optimized`` covariate method, which was
added for this purpose. When it was first applied to the employment panel it
returned :math:`1.7 \times 10^{11}`.

The cause was that ``incomepc`` has roughly seventy times the outcome's
within-unit dispersion. The reference fits the covariate coefficients by
gradient descent on a fixed :math:`1/t` step schedule with no curvature scaling,
which diverges in a direction that steep. The exact minimiser of that objective
is scale-equivariant, so the estimator *looks* like it should not care about
units; the early-stopped descent that defines it in practice does. Covariates
are now scaled before the descent, which reproduces the published figure on this
panel and on the two others available, and moves the previously pinned quota
panel closer to Stata as well.

The employment panel is kept in the case as the regression guard for it. The
episode is the argument for doing replications on real applied data: the
estimator passed a full test suite, complete coverage and a mutation sweep on a
panel whose single covariate happened to be mild enough to hide the problem.

Reproducing it
--------------

.. code-block:: python

   import pandas as pd
   from mlsynth import SDID

   df = pd.read_csv("benchmarks/reference/wine_tennessee/study2_employment.csv")
   covariates = ["popdense2010", "pop21per", "incomepc", "wineper",
                 "college", "unemploy", "nonwhite"]
   res = SDID({"df": df, "outcome": "employper", "treat": "treat",
               "unitid": "state", "time": "time", "vce": "noinference",
               "display_graphs": False,
               "covariates": {"optimized": covariates}}).fit()
   res.effects.att        # -53.17, against the paper's -53.154

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case wine_tennessee
