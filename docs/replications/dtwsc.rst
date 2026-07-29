DTWSC -- Cao and Chadefaux (2025) on the Basque panel
=====================================================

:doc:`../dtwsc` is cross-validated against the authors' own R package,
`conflictlab/dsc <https://github.com/conflictlab/dsc>`_ (MIT), pinned at commit
``b1cd241``. The reference is not vendored; the generator that produces the
comparison dump lives in ``benchmarks/reference/dtwsc_basque/`` and the
dependency chain it needs is scripted in ``benchmarks/R/install_dtwsc.sh``.

The setup is the reference package's own README example: Synth's ``basque``
panel, 1955--1997, treatment in 1970, the national aggregate dropped so 16
donors remain, the full 14-predictor Abadie specification, ``k = 4``,
``filter.width = 5``, ``buffer = 0``, ``n.burn = 3``, ``symmetricP1`` for the
first alignment and ``asymmetricP2`` for the second.

Headline numbers
----------------

.. list-table::
   :header-rows: 1

   * - method
     - pre-RMSE
     - ATT
   * - standard synthetic control
     - 0.0886
     - -0.6027
   * - DSC, R reference
     - 0.0705
     - -0.5579
   * - DTWSC, mlsynth's warp through R's ``Synth``
     - 0.0705
     - -0.5592

The paper's claim reproduces: warping tightens the pre-treatment fit by about
20 percent, and moves the ATT modestly toward zero.

What matches, seam by seam
--------------------------

The comparison holds the input fixed at R's own filtered panel, so only the
warping is being tested.

.. list-table::
   :header-rows: 1

   * - quantity
     - agreement
   * - ``cutoff``
     - 16/16 donors exact
   * - ``weight.a`` (first-phase speeds)
     - 16/16 exact, worst 2.2e-16 (one ULP)
   * - second-phase window search (``j_opt``, ``margin``, candidate count)
     - 28/28 windows on the worst-fitting donor
   * - outlier-filter decisions
     - 13848/13888 cells (99.71 percent)
   * - pre-RMSE
     - matches to four decimals
   * - ATT
     - within 0.0013, or 0.23 percent

Two findings worth keeping
--------------------------

The alignment kernel is exact. mlsynth implements the two Sakoe--Chiba step
patterns directly rather than taking a DTW dependency, and over 268 random
query/reference pairs spanning lengths 2--25 it agrees with R ``dtw`` 1.23-3 on
the accumulated cost to 1e-6, on the full ``index1``/``index2`` warping paths
exactly, and on which pairs are inadmissible.

The residual is a tie, not a bug. Forty of the 13888 outlier-filter decisions
differ, split near-evenly in both directions (21 R-only, 19 mlsynth-only). The
warping weights are small-denominator rationals -- 2/3, 1, 4/3 -- so they land
within a few units in the last place of the :math:`Q_1 \pm 3\,\mathrm{IQR}`
bound routinely, and which side they fall on turns on floating-point summation
order rather than on the data. One contributor was identified and fixed: R's
``quantile(type = 7)`` evaluates :math:`(1-h)x_{lo} + h\,x_{hi}` where numpy
takes a different interpolation branch, a two-ULP difference on a bound that
can sit three ULP from the data. A symmetric tolerance does not close the rest,
because the disagreements run both ways.

The practical consequence is worth stating plainly: the last digit of a DSC ATT
is not reproducible across languages, and should not be reported as if it were.
Any cross-check should use a tolerance no tighter than about 0.005.

A second artefact is inherited from the method rather than the port. Three
donors' warped series end one period short of 1997, so the reference's own
counterfactual is ``NA`` there and its ATT is really taken over 1971--1996.
mlsynth reproduces that rather than extrapolating over it, and reports the
number of dropped periods in ``res.metadata["n_post_periods_undefined"]``.

Placebo inference
-----------------

``inference="placebo"`` implements the paper's two inferential procedures. Run
on the Basque panel through the public API -- outcome-only synthetic control,
fixed warping hyperparameters, 17 pools and 256 placebo runs:

.. list-table::
   :header-rows: 1

   * - quantity
     - mlsynth
     - paper
   * - efficiency test ``t``
     - -7.00
     - -7.91
   * - efficiency test ``p``
     - 2.7e-11
     - < 0.0001
   * - mean ``log(MSE_DSC / MSE_SC)``
     - -0.233
     - -0.18 (from the reported log MSEs)
   * - implied MSE reduction
     - 21 percent
     - 16 percent

The direction and rough magnitude reproduce. The gap is expected: the paper
fits each placebo run at its own grid-optimal ``filter.width`` / ``k`` / step
pattern, and uses a 14-predictor Abadie specification for the synthetic control
where this run is outcome-only.

The band does not reproduce, and that is worth stating plainly. On this
specification the warped band comes out about 1 percent *wider* than the
unwarped one, where the paper reports it narrower. The two results are not in
conflict: the efficiency test is a within-run paired comparison of MSE, while
the band is the cross-run spread of gaps, and lowering each run's error need not
shrink the dispersion across runs -- the band is a tail quantity. The most
likely source of the difference is the per-run hyperparameter optimisation,
which would tighten the tails specifically, and which mlsynth cannot reproduce
because the selection rule is not in the replication package. Only the sweep
and the resulting ``gridOpt`` tables are shipped; nothing in the released code
builds them.

Treat the efficiency test as the reproducible claim and the band as
descriptive.

Cost. The placebo procedure runs one full two-phase warp per donor per pool.
Basque with ``placebo_pairs=0`` is 256 warped fits and takes about 12 minutes;
the paper's ``placebo_pairs=100`` construction is roughly 2000 fits.
