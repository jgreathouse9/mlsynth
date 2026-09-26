SL: Viviano and Bradic (2023)
=============================

.. currentmodule:: mlsynth

What is validated, and how
--------------------------

Two cases, both pinned.

`benchmarks/cases/sl.py <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/sl.py>`_
is Path B, the size and power of the paper's own test on the paper's own Monte
Carlo design.

`benchmarks/cases/sl_tennessee.py <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/sl_tennessee.py>`_
is the cross-validation: mlsynth's SL against an R implementation of the authors'
own expert library, on their Medicaid-expansion panel. The reference bundle is
``benchmarks/reference/sl_tennessee``, whose ``reference.R`` transcribes
``generate_experts`` and ``Exp_algorithm`` from their ``libraries/library.R`` and
marks every line where it departs from them. The panel is
``basedata/sl_tennessee_medcost.csv``.

That case earned its place immediately: it found two defects in this port that
nothing else had. See :ref:`sl-what-the-crossval-caught`.

Path B: size and power on their Factor_model design
---------------------------------------------------

The design is theirs. ``sim_process_factor_model`` in ``libraries/library.R``,
under ``DGP = 1``, draws a common factor :math:`f_t` and a common :math:`\theta_t`
shared by the treated unit and every donor, gives donor :math:`j` its own AR(1)
error at :math:`\rho = 0.6` with loading and mean :math:`\lambda_j = \mu_j = j/p`,
and sets

.. math::

   x_{jt} = \lambda_j f_t + \mu_j + \varepsilon_{jt} + \theta_t ,
   \qquad
   y_t = 0.5 + 0.5 f_t + \theta_t + u_t .

``simulate_data_helper`` then adds a constant effect to the treated unit's
post-treatment periods. Their Table 1 sweeps that effect over
:math:`\{0, 0.1, \dots, 1.5\}` and reports rejection rates.

Measured on 80 periods with 70 pre-treatment, 10 donors, 500 bootstrap
replicates, 120 simulations per cell:

.. list-table::
   :header-rows: 1

   * - Effect
     - Reject at 5%
     - Reject at 10%
     - Mean estimate
     - Mean absolute error
   * - 0.00
     - 0.067
     - 0.117
     - 0.033
     - 0.212
   * - 0.20
     - 0.108
     - 0.183
     - 0.234
     - 0.212
   * - 0.35
     - 0.175
     - 0.275
     - 0.384
     - 0.212
   * - 0.50
     - 0.308
     - 0.417
     - 0.534
     - 0.213
   * - 1.00
     - 0.808
     - 0.892
     - 1.036
     - 0.214
   * - 1.50
     - 0.992
     - 0.992
     - 1.537
     - 0.215

Power rises monotonically across the whole grid and reaches 0.992 at the top of
it. The point estimate tracks the planted effect at every cell, and the mean
absolute error is flat in the effect size at about 0.21, which is what a correctly
centred estimator does.

Size is 0.067 here. A second 120-simulation run on a different seed range gives
0.092, pooling to 0.079 over 240 draws against a nominal 0.05, with a Monte Carlo
standard error of 0.014. That is 2.1 standard errors above nominal, so the test
over-rejects mildly at this panel size and the excess is larger than sampling
noise. Theorem 3.1's size control is asymptotic and this design is small for it:
the split leaves 28 weighting periods, and Algorithm 2's resampling pool is 38
periods long at a block length of 3. The benchmark pins the size measured on the
case's own seed range and tolerates 0.05 either way, which catches a blowout while
surviving a boundary p-value flipping on a different BLAS. A user who needs the
nominal level on a short panel should treat the reported p-value as optimistic.

The mean absolute error of 0.21 does not shrink as the effect grows, because it is
dominated by the ensemble's own prediction error on the post window and not by
anything the effect changes.

The AR(1) draws are not bit-identical to R's ``arima.sim``, which uses its own
burn-in convention, so this reproduces the design and not the authors' exact
sample paths. The quantities pinned are rejection rates and errors, which are
properties of the design.

Two corrections to the authors' code
------------------------------------

Both were found by running their replication package and both move published
numbers, so both are pinned.

The bootstrap's learning rate
"""""""""""""""""""""""""""""

``function4boot_TE`` refits the ensemble with ``eta = 1`` hard-coded at
``library.R:214``, while the observed statistic uses
``1/(sqrt(88) * var(med_ts))``, which evaluates to 51.43 on their panel. At
:math:`\eta = 1` the weights are close to uniform, so the null distribution
describes a near-equal-weighted ensemble and the statistic it gates describes an
exponentially weighted one.

Measured on their own two raw blocks:

.. list-table::
   :header-rows: 1

   * - Block
     - Observed statistic
     - Critical value at 10%, ``eta = 1``
     - Critical value at 10%, ``eta = 51.43``
   * - raw, train 20-51, weights 1-20
     - 1.3707
     - 1.3559
     - 1.1355
   * - raw, train 1-30, weights 31-50
     - 0.6886
     - 1.2655
     - 0.7881

The critical values inflate by 19 and 61 percent, so the shipped test is
conservative. Neither verdict flips here, but the second block goes from a
comfortable non-rejection (0.689 against 1.266) to sitting 3 percent inside the 20
percent critical value (0.689 against 0.710). SL refits the bootstrap with the
same :math:`\eta` as the estimate, and ``critical_values_move_with_eta`` pins the
wiring.

The lasso expert's penalty
""""""""""""""""""""""""""

``generate_experts`` selects the lasso's penalty with ``cv.glmnet(nfolds = 5)``
and no ``foldid``, so the fold assignment comes off the random number generator.
On their 30-period window against 6 donors that choice has three attractors:

.. list-table::
   :header-rows: 1

   * - ``lambda.min``
     - Donors kept
     - In-window SSR :math:`\times 10^4`
     - Share of 8 seeds
   * - 0.00005
     - 6
     - 119.35
     - 4
   * - 0.00476
     - 4
     - 311.14
     - 2
   * - about 0.36
     - 0
     - 450.18
     - 2

The third keeps no donors at all, collapsing the expert to a constant. The
downstream effect is a bimodal estimate: over eight seeds the reported effect
lands at 3.74 four times and 5.23 four times, with nothing in between, and the
published value is 5.223 -- the mode on which that expert degenerated. Their
``set.seed(123)`` does not pin it, because each block's position in the random
stream depends on what ran before it, and their ``.Rhistory`` shows the blocks run
interactively.

SL's folds are contiguous and unshuffled, so the penalty is a function of the
data. ``lasso_expert_is_deterministic`` pins that the expert does not move with
the seed at all. Contiguous folds also respect time order, which a random split of
a time series does not.

.. _sl-what-the-crossval-caught:

What the cross-validation caught
--------------------------------

Neither defect below was visible from the paper, from the estimator's 104 unit
tests, or from the simulation benchmark. Both changed published numbers.

The penalty grid
~~~~~~~~~~~~~~~~

``generate_experts`` hands ``cv.glmnet`` two explicit grids, and they are not the
same one: ``seq(exp(-10), exp(-1), 79)`` for the lasso expert and
``seq(exp(-10), exp(2), 79)`` for the factor expert, the second regressing a
well-conditioned factor instead of the outcome. The port carried neither and let
scikit-learn derive its own grid.

On this window that decides the answer. The mean cross-validated error varies by a
factor of only 1.064 across the whole of the paper's grid, so the curve barely
separates the null model from the six-donor one:

.. list-table::
   :header-rows: 1

   * - Grid searched
     - Penalty chosen
     - Donors kept
     - In-window SSR
   * - the paper's, ``exp(-10)`` to ``exp(-1)``
     - 0.367879
     - 0
     - 0.045018
   * - scikit-learn's default
     - 0.0000597
     - 5
     - 0.015990

Two different experts, and the first is the one the authors' own published column
came from. With the grid restored the lasso expert's path agrees with the
reference to 2.3e-12.

Standardization
~~~~~~~~~~~~~~~

glmnet standardizes the design by default and the authors do not turn it off. The
port fitted unstandardized, which applies a different effective penalty to every
donor, since the lasso is not scale invariant. The divisor matters too: glmnet
uses the population standard deviation.

.. list-table::
   :header-rows: 1

   * - Fit
     - Factor expert's path, max abs difference from the reference
   * - unstandardized
     - 2.1e-02
   * - standardized, population SD (glmnet's)
     - 7.0e-07
   * - standardized, sample SD
     - 6.8e-06

How tight it is now
-------------------

Both sides deterministic, on the three experts that are algorithmically
determined:

.. list-table::
   :header-rows: 1

   * - Quantity
     - mlsynth vs reference
   * - lasso expert, path over 100 quarters
     - 2.3e-12 absolute
   * - did expert, path over 100 quarters
     - 4.9e-11 absolute
   * - factor expert, path over 100 quarters
     - 7.0e-07 absolute
   * - lasso penalty, and donors kept
     - same grid point, same count
   * - factor penalty
     - same grid point
   * - learning rate
     - 8.2e-13 relative
   * - ensemble weights
     - 1.8e-06 absolute
   * - each expert's in-window SSR
     - 6.8e-06 relative
   * - test statistic
     - 7.4e-06 relative
   * - the effect, worst of four horizons
     - 3.9e-06 relative

Two of the path rows are bounded by the capture and not by the port. The bundle
prints at ten decimals, so a path comparison cannot resolve below 5e-11, and the
did expert's 4.9e-11 is that limit. That expert calls ``did_from_mean``, the
difference-in-differences :class:`~mlsynth.FDID` reports beside its forward fit,
and against the authors' own line evaluated at full precision the two agree to
5.6e-17. The lasso row is the same limit met once instead of a hundred times: at
the penalty their grid selects that expert keeps no donors, so its path is one
constant and there is a single rounded value to disagree with.

The last few rows are a floor and not a target. The factor expert's penalty lands
at its grid's minimum, where the fit is nearly unregularized on a 30-by-6 design
of highly correlated state series, and the two solvers converge to different
points of a near-flat optimum. Tightening scikit-learn's tolerance from 1e-04 to
1e-13 moves the disagreement from 7.0e-07 to 9.2e-07 and no further, so this is
the lasso solution's own non-uniqueness and not an error either side can remove.
It is the same shape of problem as the non-identified weight vector of #30.

The random forest is left out of the compared library on purpose. randomForest and
scikit-learn's ``RandomForestRegressor`` are different implementations, so its path
cannot agree cell for cell in any language pair, and including it would bound the
measured accuracy of the port by that gap instead of by the port.

Against their published table
-----------------------------

With the forest back in, against their Table 4 raw block:

.. list-table::
   :header-rows: 1

   * - ``m``
     - SL statistic
     - Theirs
     - SL effect
     - Theirs
   * - 0
     - 0.7047
     - 0.6910
     - 4.5228
     - 5.2227
   * - 1yr
     - 0.6688
     - 0.6225
     - 4.8413
     - 5.3624
   * - 2yr
     - 0.6031
     - 0.6247
     - 4.7981
     - 5.5167
   * - 3yr
     - 0.5853
     - 0.6108
     - 4.7714
     - 5.5870

The statistic agrees to between 2.0 and 7.4 percent and the verdict agrees at
every horizon: no rejection, p between 0.19 and 0.26 against their non-rejection
at both the 10 and 20 percent levels. The lasso expert now reaches their own mode,
``alpha`` 0.367879 keeping no donors, in-window SSR 450.2.

The effect is 9.7 to 14.6 percent below theirs, and the forest is what is left.
Excluding it the two implementations agree to 3.9e-06, so the entire residual is
the one member that cannot be matched across languages, carrying 20.8 percent of
the weight. Before the two fixes above the same gap was 17 to 19 percent.

That the panel itself is right was established against the authors' own output.
``employment_BFRSS.txt``'s 51st column, ``employ_ts``, is a Tennessee series they
did ship, appended at ``do_file.R:312``. Rebuilding it from the BRFSS microdata
with ``do_file.R`` lines 256-277 matches all 300 cells to 4.44e-16, correlation 1,
with the same ``unique(IYEAR)[-2]`` year ordering. The outcome series itself is not
in their package -- all four of its state matrices have 50 columns and none is 47,
since the ``[-43]`` at ``do_file.R:82`` drops exactly Tennessee -- so it is
reconstructed and shipped as ``basedata/sl_tennessee_medcost.csv``.

What the library was doing
--------------------------

The diagnostics on the fit, on the paper's own panel:

.. list-table::
   :header-rows: 1

   * - Expert
     - Weight
     - In-window SSR :math:`\times 10^4`
   * - lasso
     - 0.337
     - 159.9
   * - factor
     - 0.250
     - 222.2
   * - forest
     - 0.155
     - 321.1
   * - did
     - 0.258
     - 215.3

``eta`` resolves to 48.25, ``effective_k`` to 3.863 of 4, and
``error_participation_ratio`` to 1.19 of 4, with pairwise error correlations
between 0.53 and 0.94. No expert is dropped and none is flagged degenerate.

Read together those two numbers say the ensembling did little here. At
``effective_k`` 3.86 the weighting is close to a simple average, and at a
participation ratio of 1.19 the four members err in roughly one direction, so
averaging them cannot cancel much. The same pair reproduces on the paper's own
simulation design at 3.70 and 1.37, so it is a property of this library and not of
this panel. Both are pinned.

What is pinned
--------------

.. list-table::
   :header-rows: 1

   * - Quantity
     - What it protects
   * - ``size_at_5pct``
     - the test does not over-reject under the null
   * - ``power_at_effect_1p5``
     - it detects an effect at the top of their grid
   * - ``power_is_monotone_in_the_effect``
     - rejection rises with the effect
   * - ``recovery_mean_abs_error_at_1p5``
     - the point estimate is centred on the planted effect
   * - ``effective_k_at_paper_eta``
     - the averaging regime is reported, not hidden
   * - ``error_participation_ratio``
     - the errors' collinearity is reported
   * - ``critical_values_move_with_eta``
     - the bootstrap uses the estimate's learning rate
   * - ``lasso_expert_is_deterministic``
     - the penalty is not drawn from the generator
   * - ``eta_zero_is_the_simple_average``
     - the weighting's lower limit is exact
   * - ``eta_large_selects_one_expert``
     - and its upper limit

And from ``sl_tennessee``:

.. list-table::
   :header-rows: 1

   * - Quantity
     - What it protects
   * - ``lasso_path_max_abs_diff``
     - the lasso expert, against losing the paper's penalty grid
   * - ``factor_path_max_abs_diff``
     - the factor expert, against losing the standardization or its own grid
   * - ``did_path_max_abs_diff``
     - the closed-form expert
   * - ``lasso_lambda_abs_diff``, ``factor_lambda_abs_diff``
     - both penalties are the same grid point, not merely close
   * - ``lasso_n_selected_diff``
     - and select the same donors
   * - ``eta_rel_diff``
     - the learning rate
   * - ``weights_max_abs_diff``, ``expert_ssr_max_rel_diff``
     - the weighting
   * - ``statistic_rel_diff``, ``att_max_rel_diff_over_horizons``
     - what a reader is shown, at every horizon


Open
----

There is no standard error and no confidence interval, here or in the paper. An
interval would come from inverting the test of Equations 7 and 8 over a grid of
candidate constant effects. ``conformal_att_interval`` does not substitute for it:
that function refits a ridge on a donor design, so its interval belongs to a
different estimator's point estimate. This was checked, not assumed.

The random forest expert is cross-validated by nothing. Pinning it would need a
forest whose trees match across languages, which randomForest and scikit-learn do
not provide, so an R reference for it would measure the two implementations
against each other and not the port.

The library is the paper's own four experts. Measured on two panels, those four
span roughly one error direction, so a wider or better-chosen library is the
obvious next question and is separate work.
