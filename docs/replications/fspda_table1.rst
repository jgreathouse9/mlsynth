.. _replication-fspda-table1:

fsPDA — Shi & Huang (2023) Table 1
===================================

:Estimator: :doc:`../pda` — :class:`mlsynth.PDA` (``methods=["fs", "LASSO"]``)
:Source: Shi, Zhentao and Huang, Jingyi (2023), *"Forward-selected panel data
   approach for program evaluation,"* Journal of Econometrics 234(2), 512-535,
   Table 1 (p. 521). Replication package: ``zhentaoshi/fsPDA`` at ``5f4542c``.
:Replication type: Path B — the paper's Monte Carlo table, all 108 cells,
   against the published values and against the authors' own estimators.
:Status: Reproduced. The twelve selection-count cells match exactly; RMSPE
   matches to 0.011 in eleven of twelve rows; the rejection rates reproduce the
   paper's size and power geometry.
:Durable case: ``benchmarks/cases/fspda_table1.py``; bundle
   ``benchmarks/reference/fspda_table1/``.

What Table 1 reports
--------------------

Shi & Huang's simulation puts forward selection and a modified-BIC LASSO on a
four-factor panel: one treated unit, 100 donors of which four are relevant, and
:math:`T_1 = T_2 \in \{50, 100, 200\}`, under two factor structures (i.i.d.
factors, and factors with AR / MA / ARMA dynamics). Each row reports the median
number of donors the rule selects, the out-of-sample RMSPE of the counterfactual,
and the rejection rate of the post-selection t-test at the 5% level under seven
treatment processes :math:`D1`-:math:`D7`. The null holds under :math:`D1`-
:math:`D3`, so those three columns are size; it is false under :math:`D4`-
:math:`D7`, so those four are power.

Twelve rows of nine columns. The case compares all of them.

Reconstructing the design
-------------------------

The data-generating process is not taken from the paper's prose. It is traced to
``simulation/nonsparse/FS.simulation.dense.R`` and lives in
:func:`mlsynth.utils.pda_helpers.simulation.simulate_pda_panel`; the factor
loadings are the ``loading.RData`` their driver loads, carried in the bundle, so
the design matrix is theirs and not a redrawing of it.

That distinction decides the answer, because Section 4.1 and the released code
disagree in three places:

.. list-table::
   :header-rows: 1
   :widths: 22 30 30

   * - Parameter
     - Section 4.1
     - The code Table 1 was run on
   * - i.i.d. factors
     - ``N(0, I^2)``
     - ``rnorm(Tn, sd = k * sigma.u)``: factor :math:`\ell` has standard
       deviation :math:`\ell`
   * - idiosyncratic shocks
     - ``N(0, 0.5^2)``
     - ``sigma.eta = 0.5``, a standard deviation
   * - irrelevant loadings
     - ``U(-0.1, 0.1)``
     - ``U(-0.5, 0.5)``

The first is typography: the ``I`` is the running index :math:`\ell` set in an
upright font, and the code confirms it. The third is the one that changes the
study. With the irrelevant loadings five times wider, the 96 irrelevant donors
carry real signal, the regression is dense, and no method can select its way to
a sparse truth — which is the paper's subject. Reading ``U(-0.1, 0.1)`` off the
page produces a nearly sparse design and a different table.

One fit per replication
-----------------------

The seven :math:`D` columns of a row come from a single fit. Both rules select
and estimate on pre-treatment data only, so the post-period prediction error
:math:`d` does not depend on the treatment effect, and their
``FS.simulation.dense`` forms the effect under DGP :math:`j` as
:math:`\Delta_j + d` and tests that. The case does the same, then checks the
step it rests on: on twelve replications the whole panel is rebuilt with
:math:`\Delta_j` added to the treated unit and put through ``PDA(...).fit()``
end to end, and the ATE, standard error and rejection come back identical
(``shortcut_matches_full_api``, pinned at 1.0 with no tolerance).

Results
-------

At 400 replications per cell:

Selection counts match outright. The median number of donors selected is the
published integer in all twelve rows — 6, 7, 9 and 6, 7, 8 for forward
selection, 9, 11, 14 and 6, 8, 13 for the LASSO — for both rules, both factor
structures and all three lengths.

RMSPE matches to 0.011 in eleven rows. The twelfth is the LASSO at
:math:`T_1 = 50` under i.i.d. factors: 0.930 against their 0.968. The direction
is the one ``fspda_dense_mc`` documents at the panel level. ``lasso.BIC`` calls
``glmnet`` at its default ``thresh = 1e-7``; this design is :math:`p = 100`
against :math:`n = 50`, where that default stops short of the optimum, and
mlsynth attains the lower LASSO objective. A better fit is a smaller prediction
error, so mlsynth's RMSPE is below theirs and not above it.

The rejection rates reproduce the paper's geometry: forward selection's size is
near nominal at the longest panel under both factor structures, the LASSO takes
in more donors than forward selection, the LASSO's size inflates under dynamic
factors while forward selection's holds, and both rules are near-fully powered
by :math:`T_1 = 200` against every alternative.

Two references, not one
-----------------------

The cells are compared against the table as printed and against the same table
as their own ``FS.R`` and ``lasso.BIC.R`` produce it, run at 2000 replications
on the same design and recorded in the bundle. The second comparison is the
tighter one, because only one of the two sides then carries appreciable Monte
Carlo noise, and it separates a disagreement in mlsynth from the gap between
this benchmark's replication count and the paper's.

What this replication found
---------------------------

Reproducing the table surfaced a defect in the LASSO's inference. mlsynth's
``lasso`` reported Li & Bell's (2017) two-component variance whatever the caller
asked for: ``lrvar_lag`` was accepted by the configuration, read by ``fs`` and
``hcw``, and dropped on the LASSO branch. On this design Li & Bell's first-stage
term is 44-46% of the total variance and the automatic truncation lag is 3 or 4
where the paper's design specifies 1 or 2, so the standard error ran about a
third large and the null rejection rate came out at 0.005-0.125 against the
paper's 0.056-0.244. Supplying the lag now selects Shi & Huang's t-test, which
matches their ``lasso.BIC`` to :math:`7 \times 10^{-6}` on the panels in the
``fspda_dense_mc`` bundle.

Reproducing it
--------------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case fspda_table1
   Rscript benchmarks/reference/fspda_table1/reference.R   # regenerate the bundle

The case runs 200 replications on each of six cells and takes about twenty
minutes. The replication count is set by the selection-count cells, which are
the tight ones: subsampled from a 400-replication run, all twelve are exact at
:math:`M = 400`, and one of 24 half-samples misses by a single donor at
:math:`M = 200`.

Related
-------

- ``fspda_dense_mc`` — the same estimators on their own panels, cell by cell.
- ``fspda_sparse_mc`` — their second Monte Carlo, the sparse one.
- ``pda_table1`` — the same design at mlsynth's defaults, where the cost of
  cross-validating the penalty is measured.
