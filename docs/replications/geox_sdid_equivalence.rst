.. _replication-geox-sdid-equivalence:

GEOX — does the harness change the estimator it wraps?
======================================================

:Estimator: :doc:`../geox` — :class:`mlsynth.GEOX`
:Reference: mlsynth's own :doc:`../sdid` — the engine GEOX runs under
   ``engine="sdid"``.
:Replication type: differential cross-validation, mlsynth against mlsynth.
   Not Path A or B: no paper pairs GeoLift's market-selection loop with
   synthetic difference-in-differences, so the composition has no published
   number. What can be checked is that composing does not change the
   component.
:Status: verified — agreement is exact on this tree, over six treated units
   and seven design-knob settings.

What is and is not in question
------------------------------

GEOX is a design harness around a scoring engine. Under ``engine="sdid"`` the
engine reaches straight into ``mlsynth.utils.sdid_helpers.weights``, so the
two estimators are not independent implementations and their agreement on the
weight programs is not news.

What surrounds the fit is unverified, and it is a lot: GEOX aggregates a
candidate region into a treated series, builds a donor pool from the
complement, indexes a pre/post split, applies a pre-period fit across the
whole panel, and packages the readout into the standardized result models.
Each of those can shift the estimate without touching a line of the
estimator. An off-by-one on the split, a donor pool that keeps a treated
market, an aggregation on the wrong axis — all produce a number, and all
produce the wrong one.

So this study fixes the design question and asks the harness to disappear.
Force one market as the treated region, hand GEOX the real post-period, and
compare its realized readout to ``SDID(...).fit()`` on the identical panel.

The panel is the Abadie–Diamond–Hainmueller Proposition 99 smoking panel
(``basedata/smoking_data.csv``: 39 states over 31 years, 1970–2000,
California treated from 1989) — the same one :doc:`sdid` cross-validates
against the authors' ``synthdid`` R package.

.. code-block:: python

   import pandas as pd
   from mlsynth import GEOX, SDID
   from mlsynth.config_models import GEOXConfig

   df = pd.read_csv("basedata/smoking_data.csv")
   df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)
   df["post"] = (df["year"] >= 1989).astype(int)

   sdid = SDID({"df": df[["state", "year", "cigsale", "treat"]],
                "outcome": "cigsale", "treat": "treat", "unitid": "state",
                "time": "year", "display_graphs": False}).fit()

   design = GEOX(GEOXConfig(
       df=df[["state", "year", "cigsale", "post"]], unitid="state",
       time="year", outcome="cigsale", post_col="post",
       treatment_size=1, to_be_treated=["California"],
       durations=[6], effect_sizes=[-0.6, -0.3, 0.0, 0.3, 0.6],
       n_backtests=2, n_draws=10, seed=0, n_validation_backtests=0)).fit()

   sdid.att, design.report.att      # -15.605399261018, -15.605399261018

What it establishes
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 18 42

   * - Quantity
     - Value
     - Reading
   * - ATT, GEOX vs SDID
     - 0
     - Exact, not close.
   * - Donor weights, max difference
     - 0
     - All 38 states. The same fit, not two fits agreeing on a mean.
   * - Six treated units and splits
     - 0
     - Not a property of California in 1989.
   * - Seven design-knob settings
     - 0
     - The search does not leak into the readout.
   * - GEOX vs the authors' ``synthdid`` R
     - 0.00157
     - Inherited whole from :doc:`sdid`.
   * - ``engine="augsynth"`` gap
     - 0.714
     - A guard: the engines must disagree.

The last row is what keeps the rest from being vacuous. If the harness
collapsed every engine onto one number, every zero above would still be zero
and the case would be testing one estimator six times. Under
``engine="augsynth"`` the same call returns −14.892 against SDID's −15.605, so
the equality is the engine's doing.

The transitivity row is the practical payoff. mlsynth's SDID sits 1.6e-3 packs
from the authors' ``synthdid`` R estimate of −15.603829, and GEOX sits 0 from
mlsynth's SDID, so GEOX reaches the authors' number at the same distance. The
external validation is inherited, and measured instead of asserted.

Where the two reports differ
----------------------------

Two quantities do not agree, and they are not the same object.

SDID's reported ``counterfactual`` is :math:`\mathbf{Y}_{0}\omega`, which
carries no level term, so it sits a constant below GEOX's — on this panel
25.260 packs — and its ``pre_rmse`` is computed against that level-free path.
The case pins the constant instead of the paths: the standard deviation of the
offset across all 31 periods is a reported quantity, and it comes back at
7.1e-15, so the two differ by a level and nothing else.

GEOX's path is the one for which ``mean(observed - counterfactual)`` over the
post window reproduces the ATT, at −15.605399261018. Taking the same mean
against SDID's own reported path gives −40.866, which is that path's level
offset showing through. Both estimators report the same ATT; only one reports
a path consistent with it.

The case
--------

`benchmarks/cases/geox_sdid_equivalence.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/geox_sdid_equivalence.py>`_,
seeded and deterministic — repeat runs give identical values — and it completes
in about seven seconds.

The four agreement rows carry a tolerance of 1e-9. They compare two calls into
the same weight programs on the same arrays, so they are exactly zero on this
tree; the tolerance records that the claim is bit-level agreement while leaving
room for a BLAS that reorders a reduction. It is four orders tighter than any
regression in aggregation, donor-pool construction or pre/post indexing could
hide beneath, since those move the ATT by whole packs.
