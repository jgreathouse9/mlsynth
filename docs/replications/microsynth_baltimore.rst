.. _replication-microsynth-baltimore:

MicroSynth — Baltimore crime information centers (Lawrence et al. 2026)
======================================================================

:Estimator: :doc:`../microsynth` — :class:`mlsynth.MicroSynth`
   (``weight_method="panel"``)
:Source: Lawrence, Daniel S., Bryce E. Peterson & Madison March (2026),
   *"The relationship between crime information centers and crime: A
   micro-synthetic evaluation of a district-level policing strategy,"*
   Journal of Criminal Justice 102:102572,
   `doi:10.1016/j.jcrimjus.2025.102572
   <https://doi.org/10.1016/j.jcrimjus.2025.102572>`_.
:Replication type: cross-validation against the authors' reference tool
   (R ``microsynth``), input scenario 3 (full runnable replication package).
:Status: identified quantities reproduced exactly; the under-identified
   counterfactual quantified.

Why this case exists
--------------------

The Seattle Drug Market Intervention cross-check (:doc:`microsynth`) shows
mlsynth's panel method agreeing with R ``microsynth`` to three or four
significant figures. That agreement is real but, it turns out, data-specific.
The Baltimore study — a much larger application of the same R package, with a
~7,000-block donor pool per district and outcomes as sparse as a few dozen
shootings — is where the panel method's identification boundary becomes visible,
and it is the more honest advertisement of what the estimator does and does not
pin down.

The study evaluates four Baltimore City Intelligence Centers (BCICs), opened in
different districts on different dates, on crime recorded in each district-block
over 30-day periods. Each of the four treated districts (Central, Eastern,
Southwestern, Western) is analysed against a synthetic comparison drawn from the
five districts that never received a BCIC, for eight headline outcomes — person,
property, shooting and all-crime, each in a total and an outdoor-only panel.
That is 32 micro-synthetic control models; the authors ran every one in R
``microsynth`` and released the code and panels on `OSF
<https://osf.io/gpzye/>`_.

What the constraints identify — and what they do not
----------------------------------------------------

Run under an identical constraint set — ``match.out`` set to the full
pre-period outcome trajectory and ``match.covar`` to the 21 census and
parcel covariates — the two implementations agree exactly on everything the
data pins down:

- the treated ("BCIC") totals, which are pure data;
- the control weights summing to the treated block count;
- covariate balance (``max |SMD|`` on the order of :math:`10^{-10}`);
- the pre-period outcome fit, which both drive to an exact zero residual.

The post-period counterfactual is a different matter. Over a ~7,000-block donor
pool, the exact pre-period fit and exact covariate balance are satisfied by a
whole face of non-negative weight vectors — a handful of totals cannot pin
thousands of weights — so the post-period prediction is not identified by the
constraints. R ``microsynth`` (its ``LowRankQP`` solve) returns whichever
interior-point iterate it lands on; mlsynth adds a strictly-convex ridge that
selects the unique maximum-effective-sample-size point, the most diffuse
synthetic control consistent with the fit. Both are valid; they are different
points in the same feasible set.

So this case cross-validates what the data identifies and measures the rest:

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * - Quantity
     - mlsynth vs R ``microsynth``
     - Interpretation
   * - Treated totals (all 32 models)
     - exact
     - data; also matches Appendix A1–A4 (see below)
   * - Control weight sum
     - exact (= treated count)
     - identified
   * - Covariate balance, pre-period fit
     - exact
     - identified
   * - Cumulative counterfactual, dense outcomes
     - agree to ``~1–2%``
     - well-identified in aggregate
   * - Cumulative counterfactual, shootings
     - differ ``~10–15%``
     - sparse; largest under-identification
   * - Per-period counterfactual
     - differ most
     - the split of a fixed cumulative is unpinned

The gap is not a bug in either implementation: it is the panel method telling
you that with this many controls and this few binding totals, the counterfactual
level is aggregate-identified but its period-by-period shape, and its level for
rare outcomes, are not. mlsynth's ridge makes the reported estimate a
well-defined function of the data; R's is whatever the solver returns.

A dividend: a typo in the published appendix
--------------------------------------------

Because the treated totals are pure data, they must match the study's Appendix
Tables A1–A4 to the integer, and they do for 119 of 120 reported cells. The
120th — Western District, property crime, 12-period total — is printed as 1467;
the panel sums to 1741, and the 24-period total of 3337, which the appendix
prints and which matches, is consistent with 1741, not 1467. The exact-total
check caught a transcription error in the published table.

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case microsynth_baltimore

The mlsynth side reads ``basedata/bcic_baltimore/`` (the OSF panels trimmed to
the columns these models use and stored as parquet). The R reference is baked
into ``benchmarks/reference/microsynth_baltimore/`` from ``microsynth`` 2.0.51;
regenerate it with

.. code-block:: bash

   # install route (CRAN is firewalled; git clone the CRAN mirror):
   #   git clone --depth 1 https://github.com/cran/LowRankQP   && R CMD INSTALL LowRankQP
   #   git clone --depth 1 https://github.com/cran/microsynth  && R CMD INSTALL microsynth
   #   git clone --depth 1 https://github.com/cran/nanoparquet && R CMD INSTALL nanoparquet
   Rscript benchmarks/R/microsynth_baltimore.R basedata/bcic_baltimore \
       benchmarks/reference/microsynth_baltimore/ref_configA.csv A
   python benchmarks/reference/microsynth_baltimore/build_reference_json.py
