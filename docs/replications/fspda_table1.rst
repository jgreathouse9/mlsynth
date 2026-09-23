.. _replication-fspda-table1:

fsPDA — Shi & Huang (2023) Table 1
===================================

:Estimator: :doc:`../pda` — :class:`mlsynth.PDA` (``methods=["fs", "LASSO"]``)
:Source: Shi, Zhentao and Huang, Jingyi (2023), *"Forward-selected panel data
   approach for program evaluation,"* Journal of Econometrics 234(2), 512-535,
   Table 1 (p. 521). Replication package: ``zhentaoshi/fsPDA`` at ``5f4542c``.
:Replication type: Path B — the paper's Monte Carlo table, all 108 cells,
   against the published values and against the authors' own estimators.
:Status: Reproduced. Eleven of the twelve selection-count cells match the
   published integer exactly and the twelfth is one donor low; RMSPE matches to
   0.005 for forward selection and to 0.007 for the LASSO outside one row; every
   rejection deviation is within 0.033.
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

Selection counts match on eleven of the twelve rows. The median number of
donors selected is the published integer for the LASSO in all six — 9, 11, 14
under i.i.d. factors and 6, 8, 13 under dynamic ones — and for forward selection
in five, at 6, 7 and 6, 7, 8.

The exception is forward selection under i.i.d. factors at :math:`T_1 = 200`,
which comes out at 8 donors against their 9. That row is a coin flip and not a
disagreement: 55% of its 400 draws select 8 donors or fewer, and the counts run
from 5 to 13, so the median sits on the boundary and lands either side of it
depending on the draws. The case pins the deviation in donors — no median more
than one donor from the published integer — because an exact match is not
something a discrete median can be held to.

RMSPE matches to 0.005 for forward selection in all six rows and to 0.007 for
the LASSO in five of six. The sixth is the LASSO at :math:`T_1 = 50` under
i.i.d. factors: 0.930 against their 0.968. The direction is the one
``fspda_dense_mc`` documents at the panel level. ``lasso.BIC`` calls ``glmnet``
at its default ``thresh = 1e-7``; this design is :math:`p = 100` against
:math:`n = 50`, where that default stops short of the optimum, and mlsynth
attains the lower LASSO objective. A better fit is a smaller prediction error,
so mlsynth's RMSPE is below theirs and not above it.

Every one of the 84 rejection deviations is within 0.033, with a root-mean-square
of 0.011 for forward selection and 0.013 for the LASSO. The paper's qualitative
claims all hold: the LASSO takes in more donors than forward selection, forward
selection predicts better out of sample in every row, the LASSO's size inflates
under dynamic factors while forward selection stays the better-sized of the two
at every length, and both rules are near-fully powered by :math:`T_1 = 200`
against every alternative.

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

The case runs 400 replications on each of six cells, spread over up to four
processes: about fifteen minutes on four cores, three quarters of an hour on
one. Each replication is addressed by its own seed and touches nothing shared,
so the numbers do not depend on the worker count. The replication count is set
by the selection-count cells, which are the tight ones: halving it to 200 costs
three of the LASSO's six exact matches, because the median of a discrete count
needs the draws to settle.

Related
-------

- ``fspda_dense_mc`` — the same estimators on their own panels, cell by cell.
- ``fspda_sparse_mc`` — their second Monte Carlo, the sparse one.
- ``pda_table1`` — the same design at mlsynth's defaults, where the cost of
  cross-validating the penalty is measured.
