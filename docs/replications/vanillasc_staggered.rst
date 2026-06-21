.. _replication-vanillasc-staggered:

VanillaSC — Staggered-Adoption Prediction Intervals (Cattaneo et al. 2025)
=========================================================================

:Estimator: :doc:`../vanillasc` — :class:`mlsynth.VanillaSC`
:Source: Cattaneo, Matias D., Yingjie Feng, Filippo Palomba and Rocío
   Titiunik (2025), *"Uncertainty Quantification in Synthetic Controls with
   Staggered Treatment Adoption,"* and the companion ``scpi`` package.
:Replication type: Cross-validation — matched against the ``scpi`` Python
   package on the authors' own Germany reunification illustration.
:Status: Verified — point estimates and the cross-unit (event-time)
   prediction intervals reproduced to solver tolerance.

What is reproduced
------------------

When more than one unit adopts treatment at possibly different times,
``VanillaSC`` fits one synthetic control per treated unit on the never-treated
donor pool and aggregates the per-unit effects into the causal predictands of
Cattaneo, Feng, Palomba and Titiunik (2025): the per-unit time average (TAUS),
the unit-by-period effect (TSUS) and — the focus of this page — the event-time
average treatment effect on the treated (TSUA), averaged across the treated
units at each event time. Each predictand carries an SCPI prediction interval:
an in-sample term from the conic simulation of the weight-estimation error and
an out-of-sample term from a sub-Gaussian location-scale model.

The cross-unit intervals are produced by a self-contained, clean-room engine
(:mod:`mlsynth.utils.vanillasc_helpers.staggered_engine`) that reimplements the
data preparation, simplex weight estimation, the in-sample conic program and
the out-of-sample model from the published methodology. It does not import the
GPL ``scpi`` package; it is validated numerically against it.

The illustration
----------------

The canonical panel (``basedata/scpi_germany.csv`` — 17 countries, 1960–2003)
has West Germany adopting in 1991 and Italy, the package's own placebo unit, in
1992, with the 15 never-treated countries as donors. Outcome-only, simplex
weights, run through the public ``fit()``:

.. code-block:: python

   import pandas as pd
   from mlsynth import VanillaSC

   df = pd.read_csv("basedata/scpi_germany.csv")
   df["status"] = 0
   df.loc[(df.country == "West Germany") & (df.year >= 1991), "status"] = 1
   df.loc[(df.country == "Italy") & (df.year >= 1992), "status"] = 1

   res = VanillaSC({"df": df, "outcome": "gdp", "treat": "status",
                    "unitid": "country", "time": "year",
                    "inference": "scpi", "scpi_compat": True,
                    "display_graphs": False}).fit()
   res.additional_outputs["event_study_intervals"]   # {event_time: bands}

In ``scpi``-compatibility mode the event-time prediction intervals reproduce
``scpi``'s ``scpi(scdataMulti(effect="time"))`` band (its ``CI_all_gaussian``)
to solver tolerance — the largest relative width difference across the twelve
balanced event times is about 0.05 percent, and the per-unit and overall point
estimates match the published digits (West Germany −1.85, Italy −1.12, overall
−1.50).

A scaling discrepancy in the published in-sample band
-----------------------------------------------------

Reproducing the event-time band surfaced a discrepancy worth recording. For the
time predictand the average effect over :math:`\iota` treated units has, under
independent per-unit weight-estimation errors, an in-sample interval that scales
as :math:`1/\iota`. The ``scpi`` package, however, divides the predictand matrix
:math:`P` by :math:`\iota` once in ``scdataMulti`` (correct — this forms the
average) and then divides the simulated in-sample draws by :math:`\iota` a second
time in ``scpi_in_diag``, so its published time-aggregated in-sample interval
scales as :math:`1/\iota^{2}`. The point estimate and the out-of-sample term use
the correct single :math:`1/\iota`.

This was isolated against ``scpi`` at machine precision: removing only the
second division, with the random draws otherwise held fixed, rescales ``scpi``'s
in-sample width by exactly :math:`\iota` at every event time (here :math:`\iota
= 2`, so exactly two-fold).

.. list-table:: In-sample event-time width, ``scpi`` vs. removing the extra division
   :header-rows: 1
   :widths: 16 28 28 14

   * - event time
     - ``scpi`` (:math:`1/\iota^{2}`)
     - corrected (:math:`1/\iota`)
     - ratio
   * - 1
     - 0.3124
     - 0.6248
     - 2.000
   * - 2
     - 0.3031
     - 0.6062
     - 2.000
   * - 3
     - 0.3397
     - 0.6795
     - 2.000
   * - …
     - …
     - …
     - 2.000
   * - 12
     - 1.4909
     - 2.9818
     - 2.000

``mlsynth`` therefore defaults to the statistically correct :math:`1/\iota`
scaling (``scpi_compat=False``) and exposes ``scpi_compat=True`` to reproduce
``scpi``'s published numbers bit-for-bit. The default and compatibility bands
differ only in this in-sample term, by exactly the factor :math:`\iota`.

Verification
------------

The durable benchmark ``benchmarks/cases/scpi_staggered_pi.py`` drives the
comparison entirely through ``VanillaSC.fit()``: it checks that the
``scpi``-compatible event-time band matches ``scpi``'s ``CI_all_gaussian`` to
within five percent (it agrees to about 0.05 percent) and that the correct
default in-sample band is exactly :math:`\iota` times the compatibility band.
The point-estimate benchmark lives in ``benchmarks/cases/scpi_staggered.py``.
Both skip themselves when ``scpi_pkg`` is not installed.
