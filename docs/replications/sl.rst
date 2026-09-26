SL: Viviano and Bradic (2023)
=============================

.. currentmodule:: mlsynth

What is validated, and how
--------------------------

SL is validated on Path B, the paper's own Monte Carlo design, and cross-checked
against its empirical application. The Path B measurement is pinned in
`benchmarks/cases/sl.py <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/sl.py>`_.
The empirical comparison is reported below as evidence and is not pinned, for a
reason given in its own section.

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

The empirical application, and why it is not pinned
---------------------------------------------------

Their replication package ships no Tennessee outcome series. All four of its state
matrices have 50 columns and none is 47; the ``[-43]`` at ``do_file.R:82`` drops
exactly Tennessee, and ``data1_SC.txt``, the BRFSS extract ``do_file.R:7`` reads,
is not included. Reproducing the application therefore needs microdata that is not
vendored in this repository, so this comparison is evidence and not a check. If
the series is ever vendored, this becomes a Path A case.

That the extraction is right was established against the authors' own output.
``employment_BFRSS.txt``'s 51st column, ``employ_ts``, is a Tennessee series they
did ship, appended at ``do_file.R:312``. Rebuilding it from the microdata with
``do_file.R`` lines 256-277 matches all 300 cells to 4.44e-16, correlation 1, with
the same ``unique(IYEAR)[-2]`` year ordering.

Running SL on that panel -- six southern states that did not expand Medicaid,
experts trained on quarters 1-30, weights on 31-50, treatment at 51, measured from
52 with the paper's ``m`` rows skipping 0, 4, 8 and 12 quarters:

.. list-table::
   :header-rows: 1

   * - ``m``
     - SL statistic
     - Their statistic
     - SL effect
     - Their effect
   * - 0
     - 0.7038
     - 0.6910
     - 4.3163
     - 5.2227
   * - 1yr
     - 0.6754
     - 0.6225
     - 4.6188
     - 5.3624
   * - 2yr
     - 0.6013
     - 0.6247
     - 4.5698
     - 5.5167
   * - 3yr
     - 0.5819
     - 0.6108
     - 4.5452
     - 5.5870

The statistic agrees to between 2 and 9 percent, and the verdict agrees: no
rejection at any horizon, p between 0.19 and 0.25 against their non-rejection at
both the 10 and 20 percent levels.

The effect lands 17 to 19 percent below theirs, and the lasso finding above is
why. SL's penalty is computed from the data and lands at ``alpha`` 5.97e-5 keeping
5 donors, in-window SSR 159.9 -- the well-fitting attractor. Their published
column came from the draw where that expert kept no donors. The two modes of their
own estimator bracket 3.74 and 5.22; SL's 4.32 sits between them, nearer the mode
whose expert fits. A deterministic penalty cannot land on the degenerate draw, so
this gap is the correction working, not a porting error.

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

Open
----

There is no standard error and no confidence interval, here or in the paper. An
interval would come from inverting the test of Equations 7 and 8 over a grid of
candidate constant effects. ``conformal_att_interval`` does not substitute for it:
that function refits a ridge on a donor design, so its interval belongs to a
different estimator's point estimate. This was checked, not assumed.

The library is the paper's own four experts. Measured on two panels, those four
span roughly one error direction, so a wider or better-chosen library is the
obvious next question and is separate work.
