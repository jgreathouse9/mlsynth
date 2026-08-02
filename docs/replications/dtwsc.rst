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
   * - DTWSC, mlsynth end to end
     - 0.0663
     - -0.5310
   * - standard SC, mlsynth end to end
     - 0.0845
     - -0.6232

The paper's claim reproduces: warping tightens the pre-treatment fit by about
20 percent, and moves the ATT modestly toward zero.

Read the units before comparing any of these. ``dsc(rescale = TRUE)`` is the
reference's default, and it maps every unit onto a common pre-treatment range
before fitting anything, so every number R reports is in rescaled units rather
than in GDP per capita. The mlsynth rows above are run with
``rescale_units=True`` so the whole table sits on one scale; without it mlsynth
reports in the outcome's own units, where R's standard-SC ATT is -0.6405 and
not -0.6027.

That distinction is easy to lose, and this page previously lost it: an mlsynth
ATT of -0.6026 in GDP per capita was compared against R's -0.6027 in rescaled
units and reported as a four-decimal reproduction. It was a coincidence of two
different quantities landing on the same digits. On one scale the two arms
agree to about 0.004 on pre-RMSE and 0.03 on the ATT, which is the honest
figure and is what ``benchmarks/cases/dtwsc_basque.py`` now pins.

Do not read that ATT agreement as agreement between the counterfactual paths,
either. Comparing the two pointwise rather than in the mean, the pre-treatment
halves sit almost on top of each other -- worst single-period gap 0.027 warped
and 0.036 unwarped, against a series spanning 2.18 rescaled units -- while the
post-treatment halves reach 0.19 and 0.16. The paths cross rather than running
parallel, so most of that error cancels in the average and a 0.19 divergence
presents as a 0.03 ATT gap.

That is the expected signature of the Savitzky--Golay difference described
below: it is held down where the pre-treatment fit constrains it and free to
grow where nothing does. The pointwise maximum is pinned in the benchmark
alongside the ATT, because a regression that tilted the counterfactual while
leaving its mean intact would pass an ATT check untouched.

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
     - 16/16 bit-identical
   * - ``avg.weight`` (second-phase speeds)
     - 16/16, worst 2.2e-16 (one ULP)
   * - warped donor series
     - 16/16, worst 7.1e-15
   * - second-phase window search (``j_opt``, ``margin``, candidate count)
     - 28/28 windows on the worst-fitting donor
   * - outlier-filter decisions
     - 13888/13888 cells

These rows are pinned in ``benchmarks/cases/dtwsc_basque.py`` rather than left
as a one-off measurement, so a regression in the warping engine fails the
benchmark.

Two findings
------------

The alignment kernel is exact, across every step pattern the method uses.
mlsynth implements them directly rather than taking a DTW dependency, so a
mistyped recursion coefficient is the kind of error that would produce a
plausible warp and a wrong answer. Six are supported -- ``symmetricP1``,
``symmetricP2``, ``asymmetricP1``, ``asymmetricP2``, ``typeIc`` and
``mori2006`` -- and each is checked against R ``dtw`` 1.23-3 on committed gold
in ``benchmarks/reference/dtwsc_dtw/``: 214 admissible alignments agree on the
accumulated cost, the normalised cost, and the full ``index1``/``index2``
warping paths exactly, while 138 inadmissible cases raise where R errors.
``warp`` and ``ref_too_short``, which consume those alignments, are checked
under all six as well.

Six rather than two because of how the method is meant to be used. The authors
do not fix the step pattern; they select it per run by grid search over seven,
and the choices shipped in their replication archive span six. An
implementation carrying only the two their published example happens to use
could reproduce 39 percent of their Basque design, 31 percent of the
California one and 50 percent of the German one -- so the omission was not a
matter of completeness but of whether the paper's own procedure is reachable at
all.

Normalisation is not uniform across the patterns: the symmetric ones divide the
accumulated cost by the query plus reference length, the asymmetric ones by the
query's, and ``mori2006`` by the reference's. Treating that as a two-way choice
leaves every path correct and every normalised distance wrong, which changes
nothing visible until an open-ended alignment picks its endpoint -- and that
endpoint decides which windows ``ref_too_short`` rules out of the second-phase
search.

One bit was load-bearing. An earlier version of this page recorded a residual
disagreement in 40 of the 13888 outlier-filter decisions and attributed it to
floating-point ties that no implementation could be expected to reproduce. That
was wrong, and the way it was wrong is instructive.

``second_dtw`` discards outlying speeds with a type-7 :math:`Q_3 + 3\,
\mathrm{IQR}` fence. The speeds are small-denominator rationals, so a column
routinely looks like :math:`(1, 1, 7/6, 1)` -- and on such a column the fence
evaluates to exactly :math:`7/6`, landing on the very value it is judging.
Which side of that comparison the value falls on is then decided by the last
bit of arithmetic, and dropping the cell rather than keeping it moves the
column mean by :math:`1/24`.

The last bit was ours to get right. R's ``stats::filter`` smooths the speeds by
accumulating :math:`x_k \cdot (1/w)` across the window; taking the window's
mean instead computes :math:`(\sum x_k)/w`, which rounds 7/6 to
``1.1666666666666667`` where R gets ``1.1666666666666665``. Everything
downstream inherited that one bit. Matching R's summation order makes the
first-phase speeds bit-identical and the entire warping engine exact, which is
what the table above now records. A tolerance on the fence does not work, and
was tried: because the disagreements run in both directions, widening the fence
fixes three donors and breaks four.

The general lesson is about the method rather than the port. A statistic whose
value turns on which side of an exactly-coincident bound a number falls is
fragile by construction, and DSC has one in its inner loop. mlsynth reproduces
the reference's choice here, but a cross-language check of a DSC ATT should not
be read as validating the last digit.

A second artefact is inherited from the method rather than the port. Three
donors' warped series end one period short of 1997, so the reference's own
counterfactual is ``NA`` there and its ATT is really taken over 1971--1996.
mlsynth reproduces that rather than extrapolating over it, and reports the
number of dropped periods in ``res.metadata["n_post_periods_undefined"]``.

The one remaining difference
----------------------------

Every seam above is measured with R's own filtered panel substituted for
mlsynth's, which is the honest way to test the warping engine on its own. Run
end to end on mlsynth's own preprocessing instead, 5 of the 16 donors pick up a
different alignment and the warped series diverge. That difference is the
Savitzky--Golay stage, and it is the only one left.

Before differentiating, the reference extends each series at both ends with a
two-period ``forecast::auto.arima`` forecast, filters the extended series, and
trims back. mlsynth edge-pads instead. The filters agree to 1e-16 across the
interior of every series and disagree only over the two points at each end --
but a second derivative is small, and at those four points the disagreement
reaches 0.083 against a series whose own scale is 0.090. Two of them sit at the
start of the pre-treatment window the first alignment learns from.

This is recorded rather than fixed, and deliberately. Closing it would take a
bit-exact ``auto.arima``, not a good one. Because the outlier fence described
above lands exactly on the values it is testing, an approximate forecast would
fall on an arbitrary side of it and would not track the reference any more
reliably than edge-padding does -- while adding a heavyweight dependency and a
stepwise order search to a preprocessing step. The gap is bounded, measured,
and pinned in the benchmark case as ``dtwsc_speed_max_abs_diff_vs_r``.

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
     - -6.79
     - -7.91
   * - efficiency test ``p``
     - 9.0e-11
     - < 0.0001
   * - mean ``log(MSE_DSC / MSE_SC)``
     - -0.226
     - -0.18 (from the reported log MSEs)
   * - implied MSE reduction
     - 20 percent
     - 16 percent

The direction and rough magnitude reproduce, but read the table as a
comparison of two different procedures rather than a reproduction attempt. The
authors' replication archive (Dataverse ``10.7910/DVN/DIUPUA``) shows their
placebo design differs from mlsynth's built-in one in five ways: they keep
Spain in the donor pool where the estimator page's example drops it; they use
118 perturbed datasets and 1789 runs against mlsynth's 17 pools and 256; they
give every run its own grid-optimal ``filter.width`` / ``k`` / step pattern
where mlsynth holds one configuration fixed; they fit the full 14-predictor
specification on every placebo where mlsynth's example is outcome-only; and
they map each gap back to the outcome's own units before squaring it, so their
reported log MSEs are not on the scale mlsynth reports.

An earlier version of this page said the rule that picked those per-run
hyperparameters was absent from the replication package. That was wrong, and
in the useful direction: the archive ships the resulting choices themselves,
per run and per panel, in ``Figure_5_*_gridOpt.Rds``. The selection rule is
indeed not shown, but it is not needed -- the choices can be read directly.

So the table above is a comparison of two procedures that happen to measure
the same thing, not a replication of the reported statistic. Reproducing the
published ``t`` means adopting their design wholesale -- pool, datasets,
per-run hyperparameters, predictors, and units -- rather than tightening
anything in mlsynth's own.

The band does not reproduce. On this
specification the warped band comes out marginally wider than the unwarped one
-- 2.436 against 2.296, or 0.3 percent -- where the paper reports it narrower.

The two results are not in conflict. The efficiency test is a within-run paired
comparison of MSE; the band is the cross-run spread of gaps. Lowering each
run's error need not shrink the dispersion across runs, because the band is a
tail quantity.

Two candidate explanations have been tested and ruled out. The first was the
synthetic-control specification: the paper fits ``Synth`` over 14 predictors
where the default here matches on the outcome alone, so ``sc_backend="malo"``
with the paper's predictors ought to have tightened the band. It does the
opposite -- the band widens to 2.9 percent and the efficiency test collapses
(:math:`t = +0.56`, MSE reduction :math:`-2` percent). The second was fidelity
of the warp itself, which is now bit-exact against the reference; fixing it
moved the band from 1.3 percent wider to 0.3 percent wider, so it was a real
contributor but not the explanation.

What remains untested is the per-run hyperparameter optimisation, and the pool
construction: the paper draws its quantiles over roughly 2000 runs where the
figures here use 256, and a 2.5 percent tail quantile is not well determined at
that count. Neither has been measured, so neither is claimed.

Treat the efficiency test as the reproducible claim and the band as
descriptive.

Cost. The placebo procedure runs one full two-phase warp per donor per pool.
Basque with ``placebo_pairs=0`` is 256 warped fits and takes about 12 minutes;
the paper's ``placebo_pairs=100`` construction is roughly 2000 fits.
