# Changelog

All notable changes to **mlsynth** are recorded here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); per the result-migration
Definition of Done (`agents/agents_results.md`, gate F), every estimator
migrated onto the two-family result contract gets an entry describing what it
now returns and the back-compat guarantee.

## [Unreleased]

### Added
- `conformal_horizon` on `PPSCMConfig`: a conformal band on each treated unit's
  CUMULATIVE effect, reported on `PPSCMUnitFit` as `cumulative_effect`,
  `cumulative_lower`, `cumulative_upper` and `cumulative_windows`. PPSCM reported the
  cumulative effect as a point estimate with no interval calibrated for it; its
  per-unit bands carry only the CFPT out-of-sample term, and the in-sample bound
  mlsynth ships assumes unconstrained weights where PPSCM's live on a simplex.
  Calibration slides an origin across the pre-period and treats every unit as adopting
  there: the partially-pooled fit produces all of them in one solve, so a pass costs
  one solve per origin rather than one per unit per origin, and each unit's summed
  out-of-sample error is one conformity score for it. The half-width is the shared
  `conformal.cumulative_conformal_interval`, so the order statistic keeps a single
  definition across estimators. The band is additional rather than a mode
  (`inference_method` still selects the bootstrap or jackknife behind the ATT) and is
  off unless the field is set. Too few non-overlapping windows for the requested level
  gives an infinite band rather than one that does not cover.

### Added
- `inference="conformal_cumulative"` on `VanillaSC`: a prediction interval for the
  cumulative (total) treatment effect over `conformal_horizon` post-periods,
  defaulting to the whole post-period. mlsynth already reported the cumulative
  effect as a point estimate, and the one existing "total" band rescaled a
  per-period interval by the horizon; this calibrates a band for the sum itself.
  The half-width is the split-conformal order statistic of *summed* out-of-sample
  errors, collected by refitting at sliding non-overlapping origins across the
  pre-period, so neither in-sample optimism nor an assumption about how
  period-to-period errors accumulate enters. The figure lands in
  `res.inference.details` (`cumulative_effect`, `cumulative_lower`,
  `cumulative_upper`, `conformal_q`, `n_calibration_windows`);
  `ci_lower`/`ci_upper` carry the per-period equivalent only when the horizon
  spans the whole post-period, since a shorter window is not the ATT. Too few
  non-overlapping windows for the requested level warns and returns an infinite
  band rather than one that does not cover.
- `mlsynth/utils/conformal/`, collecting the conformal machinery into one package
  (following `utils/bilevel/`): `quantile.split_conformal_quantile` (moved from
  `utils/inferutils.py`, which re-exports it, so existing imports are unaffected),
  `scores.rolling_origin_block_sums`, `cumulative.cumulative_conformal_interval`
  and `cumulative_conformal_from_refit`, and `structure.CumulativeConformalBand`.
  The pure combiner takes precomputed scores, so an estimator whose refit produces
  several treated units at once can build its own scores in a single pass and
  reuse the same calibration.

### Changed
- The FISTA warm start that seeds the exact simplex active set is computed in
  `utils/bilevel/active_set.py::solve_simplex_qp` instead of in
  `ridge_augment.simplex_qp`. Being accelerated was a property of one entry
  point, and of the thirteen call sites in the library twelve never supplied a
  warm start: MEDSC, SCD, COMPSC, StackedSC, mlSC, the proximal over-identified
  weights, the two `minnorm` fallbacks and SDID's two simplex programs all
  started from the uniform point. From there the active set sheds one donor per
  pivot, so its work tracked the donor pool and not the support it ends on --
  0.62 to 0.87 pivots per donor from J = 20 to J = 320, against a support that
  grows 7 to 43. Seeded, the same problems take 0 or 1 pivot. That is why SDID
  took 117 inner least-squares solves on a 101x120 panel where `VanillaSC`, which
  entered through the accelerated door, took 31. Speed only: the exact active set
  still determines the weights, and on SDID the two paths agree to 0.0e+00. Pass
  `accelerate=False` for the cold path.
- `fista_warm_start` stops once the seed's support has held for
  `SUPPORT_PATIENCE` consecutive iterations (25), read at the `SUPPORT_TOL` the
  active set itself pins variables at. Its `tol=1e-7` rule tests how far the
  iterate moved, which is the wrong question for a seed whose job is to name the
  support: on the two SDID programs of a 101x120 panel the support was final by
  iteration ~150 and the loop ran to 400. Pass `support_patience=None` for the
  old behaviour.

### Changed
- `VanillaSC`'s per-fold refit closure is defined once and shared by
  `inference="ttest"` and `inference="conformal_cumulative"`, and takes period
  indices rather than sliced arrays so a covariate-aware refit can subset its
  covariates by the same periods.

### Removed
- `mode="two_way_global_annealed"` and its five `utils/syndes_helpers/relaxed_*.py`
  modules (932 lines), the `relaxed_max_iter` / `relaxed_decay` fields, the
  `RelaxedSolverResults` container, `permutation_test_relaxed_global` and
  `plot_relaxed_design`. The annealed relaxation existed to dodge the two-way
  MIP's cost on problems SCIP could not finish; the treated-set search finishes
  those directly and exactly, so the slower approximate path has no remaining
  use. Breaking: configurations naming that mode now raise `MlsynthConfigError`.
- `utils/syndes_helpers/accelerate.py` and the `accelerate`, `accel_min_tuples`,
  `accel_safety_margin` and `certify_sdp_n_max` fields. The warm start and SDP
  objective cut only ever applied to the two-way branch-and-bound, which is no
  longer the default route, and the certificate that consumed the same lift now
  uses the closed-form Rayleigh bound. `_sdp_moment_bound_two_way` goes with
  them. `utils/miqp_accel.py` stays: `solve_synthetic_design` still routes to it
  when a caller supplies `warm_start_D` or `objective_lower_bound`, though
  nothing in the library reaches it by default any more.
- Donor-side restrictions under `mode="two_way_global"`: `donor_exclusion`,
  `donor_region_col` and `exclude_bordering_donors` now raise for that mode, and
  `donor_constraints` refuses `global_2way` directly. A donor rule reads
  `w[j] - q[j] <= 1 - D[i]`, which ties the control weights to which units are
  treated, so a design stops being scoreable from its treated set alone -- the
  one restriction the search cannot express. `one_way_global` and `per_unit`
  keep them unchanged. Every other restriction (forced, forbidden, cluster and
  adjacency conflicts, stratum quotas, size eligibility, costs with a budget) is
  a predicate on the treated set and is applied during the search.

- The GeoLift market-selection estimators (`GEOLIFT`, `MULTICELLGEOLIFT`) and
  their `utils/geolift_helpers/` package are removed from the public library.
  The two size/attribute eligibility primitives that SYNDES borrowed
  (`eligible_by_size`, `unit_attribute_map`) now live in
  `mlsynth.utils.syndes_helpers.eligibility`; SYNDES, LEXSCM, and MAREX are
  unaffected. `compare_methods` / the design-comparison utility drop the
  `"GEOLIFT"` method and the `from_geolift` adapter (SYNDES / LEXSCM / MAREX
  remain). The remaining design estimators cover market selection under a
  budget.


### Added
- `PDAConfig.lasso_criterion` and `PDAConfig.lasso_mbic_const`: the L1 variant
  can now select its penalty by Shi & Huang's modified BIC,
  `log(sigma^2) + H log(log N) log(T1)/T1 k`, minimised over `fsPDA`'s grid
  `seq(0.01, 1, by = 0.01)`. `lasso_criterion="mbic"` reproduces their
  `lasso.BIC`, which means the fit conventions move with the criterion: no
  intercept and `glmnet`'s column scaling, since the penalty is chosen by
  scoring that fit. `fit_lasso` gains a `standardize` flag for it, and the
  prediction-interval bootstrap refits under the same conventions at the same
  fixed penalty. Checked against `glmnet` 4.1.8: the scaled path agrees to
  5e-09, and on a 20-donor panel both implementations return a penalty of 0.32,
  the same single donor, and a coefficient of 0.51838043. The default is
  unchanged: `lasso_criterion="cv"` is the 5-fold cross-validated rule the
  estimator has always used, and the point estimate it returns is bit-identical
  to before.
- `utils/syndes_helpers/enumeration.py`: the exact two-way backend now builds
  candidate treated sets inside the structural restrictions instead of generating
  every `C(N, K)` subset and testing each one. Forced units, forbidden units
  (including size-ineligible ones) and stratum quotas decide which candidates
  exist, and `design_restrictions` keys each unit to one stratum, so the
  admissible designs are a product of per-group choices that `SearchSpace.size`
  counts exactly without walking. `candidate_limit` is compared against that
  count, so an instance whose unrestricted `C(N, K)` is past the limit is now
  solved exactly whenever the restrictions leave few enough designs -- where
  before it raised `MlsynthConfigError`. Conflict pairs, `costs` with a `budget`
  and the pool's no-good sets do not decompose over a stratum, so they remain
  tests on finished candidates and do not lower the count; an instance carrying
  only those is still refused past the limit. With no restrictions the walk
  reproduces `combinations(range(N), K)` term for term, including order, so no
  existing result moves.

- `mlsynth.save_spec` / `mlsynth.load_spec`: serialize an analysis specification
  to a portable JSON or YAML file and load it back into a ready-to-fit estimator.
  Because a configuration is plain, validated data, everything but the
  `DataFrame` (and any runtime arrays such as adjacency or spatial matrices) can
  be written to a text record, version-controlled, and reloaded -- separating the
  durable record of *what* analysis was run from the data it ran on. The
  DataFrame and matrix payloads are dropped on save and re-attached at load time
  (`load_spec(path, df=..., adjacency=...)`). Adds `PyYAML` as a dependency.
  Covered by `tests/test_spec.py`, including a parametrized check that `load_spec`
  resolves and behaves gracefully for every estimator the package ships.

- `SYNDESConfig.backend` selects how `mode="two_way_global"` is solved (see
  Changed for the default it now takes). `"exact"` searches
  treated sets directly: naming the set removes the `q = w * D` coupling, so the
  outer problem is a choice of treated set and the inner problem is convex. The
  search scores every candidate with the closed form in
  `utils/syndes_helpers/gram`, settles the candidates whose sign conditions are
  slack, prunes the rest against the incumbent, and reaches the projected-
  gradient solver in `utils/syndes_helpers/partition` only for the survivors --
  nought or one design out of thousands on the panels used to develop it. A
  candidate costs one row of a matrix product instead of a branch-and-bound
  node, and `T` leaves the per-candidate cost once the Gram matrix is formed.
  Above a candidate count the search falls back to a swap search reported
  against the Rayleigh bound, and says so instead of claiming a certificate.

  Comparing the two backends needs one caveat: `gap_limit` defaults to `0.05`,
  so the MIP path may return a design 5 percent above the optimum, and does on
  some panels. The two agree on the treated set once the MIP is asked to prove
  optimality (`gap_limit=0.0`); otherwise the exact backend's design is the one
  that is at least as good.

  `backend="exact"` is rejected for `one_way_global` and `per_unit`, where the
  reformulation does not apply. Every other restriction is a predicate on
  the treated set and is applied during the search. `select_by_holdout` takes a
  `pool_fn` so holdout and IC selection reach the new backend. Covered by
  `tests/test_syndes_exact.py`, `tests/test_syndes_partition.py`,
  `tests/test_syndes_backend.py`, `tests/test_syndes_exact_properties.py`, and
  nine semantic mutants in `tools/mutation/targets.toml`.


### Fixed
- `pda_helpers/inference.hac_lrv` divided the lag-`l` autocovariance by its own
  product count `n - l`; every standard HAC estimator, R's
  `acf(type = "covariance")` among them, divides by `n`. The lagged terms were
  inflated by `n / (n - l)`, which cost the autocovariance sequence its positive
  semi-definiteness and broke the claim that `fs` and `hcw`'s `lrvar_lag` branch
  reproduces Shi & Huang's released `fsPDA` package. On their dense simulation at
  `T2 = 50, h = 1` the forward-selection t-statistic was wrong in its third
  decimal; under the correct convention it agrees with their `FS()` to 2e-11.
  The error grows with the truncation lag. Effects, weights and selected donor
  sets are untouched -- only standard errors, t-statistics, p-values and
  confidence intervals move, and only where a fixed lag is in play: `fs` and
  `hcw` at their default `lrvar_lag=None` use the prewhitened Newey-West path and
  are unaffected. The `l2` HAC t-statistic moves by about 1% (Hong Kong
  7.799 -> 7.825, PPI 4.482 -> 4.547), inside the tolerances those benchmark
  cases already carry.

- `utils/miqp_accel.solve_warm_cut` wrote each warm-start bit to the wrong SCIP
  variable on problems above roughly eleven units. cvxpy returns the boolean
  columns as a `set`, and the accelerator iterated it directly, so `warm_bits[j]`
  was paired with whichever column the hash table happened to yield, not with
  entry `j` of the boolean variable. The two orders agree for short index
  runs, which is why small problems behaved correctly and larger ones did not:
  SCIP was started from a design the caller never named. At N=200, K=6 under a
  60 s limit the start left the solver worse off than no start at all
  (objective 0.2151 against 0.2070); it now returns the started design, 0.1812.
  The positions are read in the variable's own order, which for one contiguous
  boolean block is ascending column. No shipped estimator was affected: nothing
  in the library passes `warm_start_D` or `objective_lower_bound` since the
  SYNDES accelerator was removed, so this was a latent defect on a reachable but
  unused path.
- `AccelInfo.warm_applied` is set from what `addSol` returned instead of from
  having called it, and its documentation no longer says the start was
  "accepted". A stored partial start is one SCIP will try to complete over the
  continuous variables, not one it has already adopted.

### Changed
- The fixed-penalty LASSO fit in `pda_helpers/lasso/estimation.py` runs to
  `tol = 1e-12` instead of scikit-learn's default `1e-4`, which is what the
  modified-BIC path needs to reach `glmnet`'s coefficients. The only existing
  caller is the LASSO prediction-interval bootstrap; on Hong Kong its
  counterfactual moves by 6e-06. The `LassoCV` point estimate is unaffected,
  which leaves that bootstrap refitting the cross-validated fit's problem more
  exactly than `LassoCV` solved it -- a gap of the same 6e-06, pinned in
  `test_pda_lasso_mbic.py` and left alone, since closing it would move every
  cross-validated LASSO result in the library.
- `SYNDESConfig.backend` defaults to `"exact"`, so `mode="two_way_global"` is
  solved by searching treated sets instead of by branch-and-bound. `"mip"`
  remains selectable and `one_way_global` / `per_unit` are untouched -- the
  default resolves to `"mip"` for them, since one-way pins the treated weights
  and per-unit carries an `(N, N)` weight matrix, so neither reduces to a search
  over treated sets. Asking for `backend="exact"` there is an error.

  This changes what a two-way fit returns on some panels, and in one direction:
  `gap_limit` defaults to `0.05`, so the MIP could stop at a design 5 percent
  above the optimum, and on the panels in `tests/test_syndes_backend.py` it does.
  The search has no early exit below its candidate limit, so where the two differ
  the new default is the better design. Set `backend="mip"` to reproduce a prior
  result exactly.
- Version bumped to 2.0.0 for the removed public API.
- The SYNDES two-way optimality certificate (`certify=True` with
  `mode="two_way_global"`) now reports a closed-form bound on the Gram matrix
  instead of the SDP / moment lift, and `result.certificate.method` reads
  `"rayleigh"` in place of `"sdp_moment"`. Naming the treated set removes the
  `q = w * D` coupling, leaving a convex program in `G = Y'Y / T`; the two weight
  normalisations are then a pair of linear equalities, and dropping the sign
  conditions gives `lb(S) = 4 alpha / (sigma' R sigma)` with
  `R = alpha H - p p'`, `H = (G + lam I)^-1`, `p = H1`, `alpha = 1'H1`. Since
  `R1 = 0`, Rayleigh's inequality bounds every size-`K` design at once. On the
  panels used to develop it the bound reached 88.8 / 90.6 / 92.0 percent of the
  true optimum at `K = 3 / 5 / 7`, against the lift's 83.2 / 84.9 / 86.2, in
  about 0.1 ms against 0.1--0.23 s. Two consequences for callers: the two-way
  certificate no longer consults `certify_sdp_n_max`, since there is no size at
  which it needs to fall back to the loose continuous bound, and it no longer
  goes absent because a conic solve hit its iteration cap. It does report
  `lower_bound=None` when `G + lam I` is near-singular, which happens as `lam`
  approaches zero on a panel with fewer pre-periods than units; the note names
  the cause. New `mlsynth.utils.syndes_helpers.gram`, covered by
  `tests/test_syndes_gram.py` and `tests/test_syndes_gram_properties.py`.
- SDID solves its placebo draws' weights as one family. Placebo inference
  (Arkhangelsky et al. 2021, Algorithm 4) refits both weight programs once per
  draw and `B` defaults to 500, which is where an SDID fit spends its time: 84
  percent of `sdid_prop99`'s wall clock was inside the simplex solver. The draws
  differ only in which donors are in the design, what the target is, and how
  large the ridge is, and none of that needs its own factorisation -- centring is
  per column so it survives subsetting, and the ridge augmentation carries no
  target rows so with the weights summing to one it enters the Gram as
  `+ ridge I`. `estimate_placebo_variance` now draws every assignment first,
  solves the family through the new
  `mlsynth.utils.sdid_helpers.weights.solve_intercept_simplex_many`, then
  replays; the draws are built in the order the old loop used them, so the RNG
  stream is untouched and the same controls are cast as pseudo-treated. On
  Prop 99 at `B = 500` a fit runs in 0.75s against 1.77s, with the ATT unchanged
  bit for bit and the placebo standard error moving in its eleventh significant
  figure. `sdid_prop99`, `sdid_ddd_hpv` and `seq_sdid_mc` all pass. Available to
  SDID because its weight designs are overdetermined, so
  `gram_reduction_is_safe` passes and the batched and one-at-a-time solvers
  return the same weights and not merely the same fit; it is checked per problem
  regardless. Pinned by `tests/test_sdid_weights_batch.py`.

- mlSC scores its penalty grid in one pass under `lambda_est="cross-validation"`.
  Folding a penalty into the design as a `sqrt(lambda sigma_y^2) R` augmentation
  adds rows carrying no target, so with the weights summing to one the augmented
  Gram is affine in the penalty, `G(p) = (X - Y 1')'(X - Y 1') + p R'R`, verified
  against the assembled design to 1e-9 across the grid. The two matrices are
  formed once, each grid point is a broadcast off them, and the batched active
  set certifies the whole grid together: on the Bottmer et al. panel (108
  training periods, 90 disaggregate controls, 56 grid points) 0.27s against
  4.49s, the same penalty selected, and `mlsc_bottmer`'s agreement with the
  author's `mlSC_estimator` unchanged at `path_a_cv_lambda_rel = 0`.

  The reduction is guarded. Forming the Gram squares the design's condition
  number, which is free only where the design has full column rank -- and this
  grid runs the penalty to zero, where the augmentation is a `1e-8` uniqueness
  ridge. On a rank-deficient training design that ridge is the only thing
  separating the columns and squaring puts it below what float64 resolves: on a
  9-period, 12-disaggregate panel the Gram form then finished 225 percent above
  the optimum at `lambda = 1e-8` and selected a different penalty. The new
  `mlsynth.utils.bilevel.minnorm.gram_reduction_is_safe` decides this from the
  design before anything is solved, and a rank-deficient one keeps the
  one-penalty-at-a-time solve. It states both failure modes the reduction has --
  the other being a design whose minimiser is a face and not a point, where
  both solvers are optimal but land in different places. Pinned by
  `tests/test_mlsc_crossval_batch.py`.

- VanillaSC solves its in-space placebo as one leave-one-out family when there
  are no covariates. The refits fit each column of the donor matrix from the
  others, so the family falls out of a single `Y0' Y0` -- deleting a donor
  deletes a row and a column of it, and each target is itself a column -- and
  `mlsynth.utils.bilevel.minnorm.solve_simplex_loo_exact` assembles them with no
  product with the data per refit. The default no-covariate call is where this
  lands: a Proposition 99 fit is 0.007s with inference off and 0.205s with it on,
  94 percent of that in the solver.

      38 donors x 19 pre     0.209s -> 0.025s     8.5x
      48 donors x 89 pre     0.394s -> 0.038s    10.2x
      119 donors x 30 pre    5.617s -> 0.137s    41.0x

  The p-value is a rank statistic over these fits, so each member is verified
  with `simplex_optimum_is_unique` and any whose minimiser is a face is
  re-solved with the single-problem active set the loop used -- a different
  exact solver has an equal claim there, and the published ranks came from that
  one. p-value, rank, RMSPE ratio and ATT are identical on every panel tried.
  Refits that are not a plain simplex fit keep the loop (covariates, ridge
  augmentation, the penalized backend), which `tests/test_vanillasc_placebo_batch.py`
  asserts by disabling the family solver and requiring the answer not to move.

- STACKEDSC solves each cohort's weight programs as one shared-design family.
  Every treated unit in a donor pool faces the same design and differs only in
  its own pre-treatment target, so with `c_j = A' b_j` and `s_j = b_j' b_j` the
  `j`-th Gram is `A'A - c_j 1' - 1 c_j' + s_j 1 1'`: one Gram and one cross
  product carry the group, and
  `mlsynth.utils.bilevel.minnorm.solve_simplex_shared_design` runs it in
  lockstep. A donor predicate that binds gives one batch per distinct pool, down
  to a batch of one per unit. On a Wiltshire cohort of 89 units against 39
  donors the outcome-only design goes 396ms -> 43ms, 9.3x. The covariate design
  does not batch (see below) and is unchanged.

- STACKEDSC's placebo layer solves each pool as a leave-one-out family. Casting
  every donor as treated in turn and refitting against the rest is one matrix's
  columns fitted against one another, which `solve_simplex_loo_exact` assembles
  from a single `M' M`. Under `donors-only` that matrix is the cohort's design
  and the family is shared by the cohort; under `permutation` -- the default,
  following the reference implementation -- the treated unit's column is
  appended, a donor to every placebo and a target to none. On the Walmart panel
  that is 22,074 programs in 21s where the loop takes 101s, 4.8x, with the
  RMSPE-ranked p-values bit-identical and every other reported statistic
  unchanged to 1e-11.

- `mlsynth.utils.bilevel.minnorm.simplex_point_is_optimal` certifies a simplex
  weight vector against the design it claims to solve, from the KKT conditions
  on `B'(Bw - A)`. The batched solvers now require it as well as uniqueness
  before standing in for the one-at-a-time solve. Uniqueness alone was not
  enough, and the gap is not academic: the Gram reduction squares the condition
  number, so a design merely awkward at `cond(B) ~ 1e7` -- covariates measured
  in different units -- gives a Gram at the edge of float64, and the batched
  active set then converges on that Gram to a point that does not solve the
  program it came from. On the covariate specification of the Wiltshire panel
  that is 62 of 76 members of a cohort, with a KKT residual of 7e-3 where the
  design-form solver leaves 6e-10; nothing computed from the Gram reveals it.
  Those members now fall back and the reported effects are unchanged to 1e-9.

- `solve_simplex_loo_exact` and `solve_simplex_shared_design` take a `fallback`
  solver, since which one-at-a-time solve a batch stands in for is part of the
  contract. STACKEDSC calls the primal active set directly; VanillaSC's engine
  calls it through a wrapper that escalates to CVXPY when the active set reports
  failure on itself, and on a design pathological enough to trip that hatch the
  two disagree. Each call site now names its own.

- `mlsynth.utils.bilevel.minnorm.simplex_optimum_is_unique` settles, after
  solving, whether a simplex least-squares minimiser is the only one -- the
  question that decides whether the batched and one-at-a-time solvers can stand
  in for each other. `gram_reduction_is_safe` answers it from the design's shape
  and is sound but far from tight: rank deficiency is only a precondition for a
  face, since the objective is flat along a direction only where that direction
  is also feasible at the solution, which needs a support large relative to the
  design's rank. Synthetic-control solutions are sparse, so the ordinary
  geometry -- more donors than pre-treatment periods -- usually has a unique
  minimiser after all. On the Proposition 99 placebo family all 38 solves do, at
  37 donors against 19 pre-periods, and the shape test admits none of them; on
  STACKEDSC's Walmart data 85.5 percent of 2264 real solves do, against the
  blanket "not available" an earlier synthetic probe suggested. The predicate
  checks the design restricted to the weakly-active set, costs nothing
  measurable next to the solve, and is asserted against what the two solvers
  actually return in `tests/test_simplex_uniqueness.py`.

- VanillaSC's `mscmt` backend (the default when covariates are supplied) solves
  its inner donor-weight program exactly, and for a whole outer-search
  generation at once. Because the weights sum to one the design matrix drops out
  of the inner objective: `X1 - X0 w = R w` for `R = X1 1' - X0`, so the
  V-weighted predictor loss is the quadratic form `w' G(V) w` with
  `G(V) = sum_p v_p r_p r_p'`, and the donor weights are the minimum-norm point
  in the convex hull of the donors' predictor discrepancies (Wolfe 1976), which
  an active set over the donors solves exactly and finitely. `G` is linear in
  `V`, so the `P` rank-one pieces are formed once, a differential-evolution
  generation of Grams is one matrix product against them, and the active set
  certifies the generation in a handful of batched linear solves -- the data
  never enters the search loop. This replaces a per-candidate Lawson-Hanson NNLS
  whose sum-to-one constraint was a big-M penalty row, so the equality is now
  exact and the inner solution exactly scale-free in `V`, as the outer objective
  assumes. On the Abadie-Gardeazabal Basque specification the bilevel fit runs
  in 1.0s against 1.7s and the default call (in-space placebo, 17 refits) in
  22s against 26s, with the MSCMT reference weights unchanged
  (`benchmarks/cases/mscmt_basque.py`). The new solver is
  `mlsynth.utils.bilevel.minnorm` (`simplex_gram`, `solve_simplex_minnorm`,
  `solve_simplex_minnorm_batch`), covered by `tests/test_simplex_minnorm.py` and
  `tests/test_simplex_minnorm_perf.py`. `solve_mscmt` gains an `inner_max_iter`
  cap and reports `metadata["inner_unconverged"]`, warning once when an inner
  solve scored a candidate without certifying. MEDSC and `determine_v`, which
  share the `_inner_weights` primitive, inherit the exact solve; their pinned
  replications are unchanged.

  Each generation is solved cold. Seeding each candidate's active set from the
  previous generation's weights cuts the inner work by about a third, and it was
  measured and rejected: where the inner optimum is a face and not a point, the
  member returned would then depend on the search's history, and members of that
  face tie on predictor fit while differing on outcome fit, so the outer
  objective would stop being a function of `V`. On the Lamba et al. tiger
  reserves that showed as a seed spread of 5e-2 ha on a 2825 ha effect, against
  2e-6 ha cold (`tests/test_lamba_tigers.py`, which is the guard).

- The `mscmt` outer search stops on a tolerance calibrated to the estimate, and
  that tolerance is reachable: `VanillaSCConfig` gains `mscmt_tol`, and its
  default (and `solve_mscmt`'s) moves from `1e-10` to `1e-6`. scipy ends
  differential evolution when the population's spread in pre-fit MSPE falls
  below `atol + tol * |mean|`, and with `atol = 0` that is purely relative. At
  `1e-10`, on the Abadie-Gardeazabal Basque specification whose mean energy is a
  pre-fit MSPE of 0.0043, the rule asked 195 candidate predictor weightings to
  agree to 4.3e-13 -- thirteen significant figures. Tracing the search shows the
  donor weights reach 1e-5 of their final position by generation 93 and move by
  1e-8 over the 120 generations after that; many panels never reach the
  threshold at all and simply exhaust `maxiter`. The new default stops around
  generation 100, leaving the weights and the ATT within 5e-6 of where the old
  one left them -- three orders finer than the four decimals the MSCMT
  replication compares to. On Basque the default call runs in 12.7s against
  22.5s (and 26.5s before both changes), the bilevel fit in 0.57s against 1.04s.
  Agreement with the captured MSCMT R run is unchanged, marginally closer on
  three of its four pinned quantities. MASC and MEDSC share `solve_mscmt` and
  inherit the default; their replications are unchanged. Pinned by
  `tests/test_mscmt_search_budget.py`.

## [1.0.0] - 2026-06-20

First stable release, published to PyPI (``pip install mlsynth``).

### Packaging
- Distribution publishes to PyPI via OIDC Trusted Publishing
  (`.github/workflows/release.yml`) on each GitHub Release -- no stored API
  token. `python -m build` + `twine check` gate the artifacts first.
- License metadata modernised to the SPDX form (`license = "MIT"` +
  `license-files = ["LICENSE.md"]`), dropping the deprecated
  `License :: OSI Approved :: MIT License` classifier; build backend bumped to
  `setuptools>=77`.
- The supported Python range (3.10-3.13) is now exercised in CI: the full suite
  runs on 3.10, 3.12 and 3.13 (`pyversions` matrix) plus 3.11 (`build`), so the
  `requires-python` floor and the Python classifiers are test-backed, not
  asserted. Development status promoted to Production/Stable.
- MSCMT no longer depends on scipy's `nnls` being a working release: the inner
  non-negative least squares selects scipy's compiled solver where it is the
  fixed, fast version (>= 1.15) and an in-house pure-NumPy Lawson-Hanson solver
  otherwise (e.g. the regressed `nnls` in scipy 1.13). Same optimum either way.

### Added
- **PDA gains the original HCW best-subset method (`method="hcw"`).**
  Hsiao-Ching-Wan (2012): the treated unit's counterfactual is unrestricted OLS
  on the AICc/AIC/BIC best subset of controls. The combinatorial search is exact
  and fast -- a Furnival-Wilson sweep-operator branch-and-bound, a
  Bertsimas-King-Mazumder discrete first-order warm start, and a node budget that
  returns the incumbent with a certified optimality gap instead of refusing large
  pools -- with an optional SCIP mixed-integer backend (`hcw_backend="scip"`,
  behind the new `scip` extra) for exact certification past the branch-and-bound's
  reach. Jiang et al. (2025) prediction intervals carry over. 100% covered;
  reproduces HCW Table XVI value-for-value, cross-validated against the `pampe`
  R package.
- **MAREX geographic design restrictions.** MAREX gains the SYNDES/GEOLIFT
  restriction vocabulary, on top of what it already had natively (region
  clustering, `m_min`/`m_max` stratum quotas, cost/budget, same-region donors):
  `to_be_treated` / `not_to_be_treated`, `adjacency` + `spillover_threshold`
  (no two treated markets border each other), `size_col` + `min/max_size`, and
  `exclude_bordering_donors` (drop a treated market's neighbours from its
  within-cluster control pool). Enforced as constraints on the MIP's `z`/`v`,
  reusing the shared estimator-agnostic `DesignRestrictions` / `build_restrictions`
  (new `marex_helpers/restrictions.apply_restrictions_marex` for the applier).
  MIQP-only (rejected with `relaxed=True`); infeasible combinations raise a
  translated `MlsynthEstimationError`. Docs gallery on real DMA geography.

### Changed
- **MAREX routes unit/time identity through `IndexSet` + `geoex_dataprep`.**
  `prepare_marex_panel` now ingests via the canonical `geoex_dataprep` (which
  enforces a strongly balanced panel) and carries `unit_index` / `time_index`
  IndexSets as the single source of truth, threaded through the optimizer and
  orchestrator instead of being re-derived from the frame. Unit order is
  preserved, so numerics are unchanged; the only behavioural change is that an
  unbalanced panel now raises a translated `MlsynthDataError`.

### Fixed
- **SYNDES no longer leaks an `IndexError` when restrictions make the candidate
  pool empty.** Over-constrained `top_K > 1` designs (in-sample, holdout, or ic
  selection) whose every MIP solve is infeasible now raise a translated
  `MlsynthEstimationError` naming the restrictions, instead of a bare "list
  index out of range". Surfaced while building the docs gallery.

### Added
- **GEOLIFT docs gallery** consolidating every geographic design constraint
  (cardinality, force in/out, spillover non-interference via `cluster_col` /
  `adjacency` with the matching donor spillover exclusion, coverage
  `min`/`max_per_stratum`, size bands, budget planning, and the full shortlist)
  into one unified showcase on the bundled real US DMA contiguity map + Census
  divisions, with a grouped linear factor sales model -- mirroring the SYNDES
  gallery so the two design estimators document the same surface in parallel.
- **SYNDES docs gallery** showcasing every design-customisation knob
  (cardinality, force in/out, no-two-treated conflict, stratum quotas, size
  bands, region-matched and non-bordering donor pools, per-unit multi-region
  designs, and the restriction-aware `top_K` menu) as runnable MWEs on the
  bundled real US DMA geography + Census regions, with a region-grouped linear
  factor sales model.
- **SYNDES donor-side restrictions (region-matched / non-bordering donors).**
  Beyond constraining *who is treated*, SYNDES can now constrain *who may serve
  as a treated unit's donor*. The primitive is a donor-exclusion relation
  `B[i,j]` ("if `i` is treated, `j` may not be its donor"), enforced by coupling
  the assignment `D` to the mode's control weights, so it works in every mode
  (one-way global `c[j] ≤ 1−D[i]`, two-way global `w[j]−q[j] ≤ 1−D[i]`, per-unit
  `w[i,j] = 0`). Filled by `donor_region_col` (a donor must share the treated
  unit's region), `exclude_bordering_donors` (drop a treated unit's spillover
  neighbours from its donor pool — the Vives-i-Bastida exclusion restriction,
  reusing the conflict graph), or an explicit `donor_exclusion` matrix (escape
  hatch), combined by union. In the global modes a region rule forces the treated
  set into one region; `per_unit` supports a multi-region design (each treated
  unit draws its own same-region donors). Validated on the bundled DMA borders +
  CDC Census regions. Helpers `build_restrictions` / `donor_constraints` in
  `utils/syndes_helpers/restrictions.py`.
- **SYNDES design restrictions (geography / clustering / size / forcing).**
  SYNDES now accepts the same restriction vocabulary as GEOLIFT and LEXSCM,
  enforced exactly as linear constraints on the MIP assignment vector `D`:
  `to_be_treated` (`D_i = 1`) / `not_to_be_treated` (`D_i = 0`, stays a donor);
  `cluster_col` and/or `adjacency` + `spillover_threshold` → no two interfering
  units both treated (`D_i + D_j ≤ 1`, via the shared conflict-graph helper
  LEXSCM uses); `stratum_col` + `min_per_stratum` / `max_per_stratum` → coverage
  quotas; `size_col` + `min_size` / `max_size` → a treated-unit size band.
  Restrictions compose with `costs`/`budget` and flow through every selection
  rule (in-sample, holdout, ic). Not supported with the annealed mode or an
  `arm` column. Over-constrained designs return translated errors, never a
  leaked solver status: config-detectable cases (unknown/overlapping forced
  units, forcing more than `K`, too few treatable units) raise
  `MlsynthConfigError`, and a solve-time infeasibility (e.g. asking for more
  mutually non-adjacent treated markets than the conflict graph allows) raises
  `MlsynthEstimationError` with a message naming the restrictions as the cause.
  Validated against the bundled real DMA contiguity matrix
  (`basedata/markets/`), restricting to Florida + Georgia. New helper module
  `utils/syndes_helpers/restrictions.py` (`build_restrictions`,
  `apply_restrictions`, `DesignRestrictions`).
- **SYNDES information-criterion (IC) design selection.** A new `selection`
  config field unifies the design-selection rule into `{"in_sample", "holdout",
  "ic"}` (default `None` infers `"holdout"` when `holdout_frac` is set, else
  `"in_sample"`, so existing configs are unchanged). `selection="ic"` ranks the
  `top_K` pool (solved on the whole pre-period — no data split) by an
  information criterion `IC = SSR_pre + 2·sigma^2·df`, with `df = active control
  donors − 1` (Pouliot-Xie-Liu's `df = |A| − 1` for the unpenalised SCM) and a
  Mallows-Cp noise estimate, penalising designs that buy fit by activating more
  donors. Preferable to holdout when the pre-period is short. Each `results.pool`
  entry gains `ic` and `df`; requires `top_K >= 2` and a MIP mode. New helper
  module `utils/syndes_helpers/infocriterion.py` (`design_df`, `select_by_ic`).
  In `compare_methods`, pass `syndes_options={"selection": "ic"}` to use it
  (overrides the default holdout).
- **SYNDES holdout (train/validate) design selection.** A new optional
  `holdout_frac` config field switches SYNDES from in-sample MIP selection to
  out-of-sample selection: the `top_K` candidate pool is learned on the leading
  `1 - holdout_frac` of the pre-period and the winning design is the one whose
  *held-out* contrast error on the trailing `holdout_frac` is smallest (e.g.
  `0.3` for a 70/30 split) — a guard against overfitting transient pre-period
  co-movement. The returned `results.pool` is ranked by OOS error and each entry
  carries an `oos_rmse`. Requires `top_K >= 2` and a MIP mode; power and
  inference are unchanged. `holdout_frac=None` (default) preserves the
  Doudchenko et al. (2021) in-sample behaviour exactly. `compare_methods` now
  defaults to holdout selection for SYNDES (`syndes_holdout_frac=0.3`), exposes
  an `oos_rmse` column, and ranks the SYNDES rows by it; pass
  `syndes_holdout_frac=None` to revert. New helper module
  `utils/syndes_helpers/holdout.py` (`split_pre`, `oos_contrast_rmse`,
  `select_by_holdout`).

### Changed
- **SpSyDiD migrated onto the two-family result contract** (final estimator of
  the 9-estimator migration). `SpSyDiD.fit()` now returns `SpSyDiDResults` as a
  frozen pydantic `EffectResult`. SpSyDiD is a **spillover decomposition**, so
  the standardized surface describes the **direct** effect: `att` (= the WLS
  `tau`), `counterfactual`, `gap`, `pre_rmse` resolve via the inherited
  accessors against the directly-treated group's observed mean vs the
  pure-control SDID synthetic (the same reconstruction the plotter draws). The
  **indirect** (`aite`) and **total** (`ate`) effects — which have no single
  counterfactual path — are kept as typed fields and mirrored into
  `effects.additional_effects`. The pure-control SDID unit weights live in the
  standardized `weights` slot (time weights in `summary_stats`); `inference` is
  `None`. **Breaking surface change:** the flat `att` field is now an inherited
  accessor (the `tau` / `tau_s` aliases still resolve), and the standalone
  `weights` field is replaced by the standardized slot. `inputs` / `aite` /
  `ate` / `unit_weights` / `time_weights` / `zeta` / `metadata` remain typed
  fields. Plotting routes through `result.plot()`. Conformance is pinned in
  `test_spsydid.py::test_two_family_result_contract` (SpSyDiD needs a spatial
  matrix, so it can't join the single-df loop in `test_result_contract.py`).
  `docs/spsydid.rst` notation rewritten to the `agents_docs.md` canon
  (calligraphic sets, bold spatial matrix `\mathbf{W}`, hatted estimates,
  `\coloneqq`, the `T_0` time split).
- **TASC migrated onto the two-family result contract.** `TASC.fit()` now
  returns `TASCResults` as a frozen pydantic `EffectResult` with the
  standardized sub-models built from the observed target vs the smoother-based
  counterfactual; `att` / `counterfactual` / `gap` / `pre_rmse` resolve via the
  inherited accessors. TASC is a state-space / EM estimator with **no donor
  weights**, so the `weights` slot records the method, not per-donor
  weights. **Breaking surface change:** the raw inference object (counterfactual
  + per-period posterior bands: `.counterfactual` / `.ci_lower` / `.ci_upper` /
  `.posterior_variance` / `.alpha`) moved from `res.inference` to
  `res.inference_detail`; the `inference` slot now holds the standardized
  `InferenceResults` (with the raw object in `.details`). The flat `att` /
  `pre_rmse` fields are now inherited accessors; `design` / `inference_detail`
  remain typed fields. Mutating the frozen result raises pydantic
  `ValidationError`. TASC plots via `result.plot()` and is pinned in
  `tests/test_result_contract.py`.
- **SparseSC migrated onto the two-family result contract.** `SparseSC.fit()`
  now returns `SparseSCResults` as a frozen pydantic `EffectResult` with the
  standardized sub-models built from the treated series via
  `build_effect_submodels`; `att` / `counterfactual` / `gap` / `att_ci` /
  `pre_rmse` / `donor_weights` resolve via the inherited accessors, and the
  donor weights live in the `weights` slot (predictor weights in
  `summary_stats`). **Breaking surface change:** the raw placebo/conformal
  inference object moved from `res.inference` to `res.inference_detail` (still
  `.method` / `.p_value` / `.placebo_atts` / `.pointwise_*` / ...); the
  `inference` slot now holds the standardized `InferenceResults` built from it
  (so `res.att_ci` resolves), and is `None` when `method="none"`. The flat
  `att` / `pre_rmse` / `donor_weights` fields are now inherited accessors;
  `design` / `predictor_weights` / `inference_detail` remain typed fields.
  Mutating the frozen result raises pydantic `ValidationError`. SparseSC plots
  via `result.plot()` and is pinned in `tests/test_result_contract.py`.
- **MLSC migrated onto the two-family result contract.** `MLSC.fit()` now
  returns `MLSCResults` as a frozen pydantic `EffectResult` with the
  standardized sub-models built from the aggregate treated series
  (`observed = counterfactual + gap`, `T0` the adoption reference); `att` /
  `counterfactual` / `gap` / `pre_rmse` / `donor_weights` resolve via the
  inherited accessors, and the disaggregate donor weights live in the `weights`
  slot (`aggregate_donor_weights` in `summary_stats`). mlSC has no statistical
  inference, so the `inference` slot is `None`. **Breaking surface change:** the
  old `res.inference` field (which carried the fitted *paths*, not statistical
  inference — it clashed with the contract's `inference` slot) is renamed to
  `res.paths` (still `.counterfactual` / `.gap`; the same series are exposed
  flat as `res.counterfactual` / `res.gap`). The flat `att` / `pre_rmse` /
  `donor_weights` fields are now inherited accessors. `design` and
  `aggregate_donor_weights` remain typed fields. Plotting routes through
  `result.plot()` (the `PlotConfig` is built from MLSCConfig's legacy color
  fields, since `MLSCConfig` is a plain `BaseModel`). Conformance is pinned in
  `test_mlsc.py::test_two_family_result_contract` (MLSC's two-level panel can't
  join the single-df loop in `test_result_contract.py`).
- **MCNNM migrated onto the two-family result contract.** `MCNNM.fit()` now
  returns `MCNNMResults` as a frozen pydantic `EffectResult` with the
  standardized sub-models built from the cross-treated-unit observed / imputed
  paths (T0 the common adoption reference); `att` / `counterfactual` / `gap` /
  `att_ci` / `pre_rmse` resolve via the inherited accessors, and the *implied*
  (non-unique) donor weights stay in the `weights` slot. **Breaking surface
  change (matrix-completion convention):** `res.counterfactual` is now the
  **1-D treated counterfactual path** (was the full `(N, T)` fitted matrix);
  the matrix moved to `res.counterfactual_matrix`, and the per-cell `effects`
  matrix to `res.effects_matrix` (the `effects` slot now holds
  `EffectsResults`). The raw jackknife object moved from `res.inference` to
  `res.inference_jackknife`; the `inference` slot now holds the standardized
  `InferenceResults` (so `res.att_ci` resolves). The staggered-adoption extras
  (`cohort_att`, `event_study`) and the factor diagnostics (`L`, `gamma`,
  `delta`, `unit_factors`, `time_factors`, `singular_values`, `rank`) remain
  typed fields. Mutating the frozen result raises pydantic `ValidationError`.
  MCNNM plots via `result.plot()` and is pinned in
  `tests/test_result_contract.py`.
- **MSQRT migrated onto the two-family result contract.** `MSQRT.fit()` now
  returns `MSQRTResults` as a frozen pydantic `EffectResult` with the
  standardized sub-models built from the cross-treated-unit observed / synthetic
  paths; `att` / `counterfactual` / `gap` / `att_ci` / `pre_rmse` resolve via
  the inherited accessors, and the per-treated-unit PCR donor weights stay in
  the `weights` slot. **Breaking surface changes:** `res.counterfactual` /
  `res.gap` are now the **1-D treated paths**; the full `(T, m)` synthetic / gap
  matrices moved to `res.counterfactual_matrix` / `res.gap_matrix`. The raw
  SCPI prediction-interval object moved from `res.inference` to
  `res.inference_intervals`; the `inference` slot now holds the standardized
  `InferenceResults` (so `res.att_ci` resolves). Mutating the frozen result
  raises pydantic `ValidationError`. MSQRT plots via `result.plot()` and is
  pinned in `tests/test_result_contract.py`.
- **SNN migrated onto the two-family result contract.** `SNN.fit()` now
  returns `SNNResults` as a frozen pydantic `EffectResult` with the
  standardized sub-models built from the cross-treated-unit observed / imputed
  paths; `att` / `counterfactual` / `gap` / `att_ci` / `pre_rmse` resolve via
  the inherited accessors, and the PCR donor weights stay in the `weights`
  slot. **Breaking surface change (matrix-completion convention):**
  `res.counterfactual` is now the **1-D treated counterfactual path** (was the
  full `(N, T)` imputed matrix); the matrix moved to
  `res.counterfactual_matrix`, and the per-cell `effects` matrix to
  `res.effects_matrix` (the `effects` slot now holds `EffectsResults`). The raw
  jackknife object moved from `res.inference` to `res.inference_jackknife`; the
  `inference` slot now holds the standardized `InferenceResults` (so
  `res.att_ci` resolves). Mutating the frozen result raises pydantic
  `ValidationError`. SNN plots via `result.plot()` and is pinned in
  `tests/test_result_contract.py`.
- **RMSI migrated onto the two-family result contract.** `RMSI.fit()` now
  returns `RMSIResults` as a frozen pydantic `EffectResult` with the
  standardized sub-models populated from the treated aggregate paths
  (`treated_mean` / `synthetic_mean`); `att` / `counterfactual` / `gap` /
  `pre_rmse` resolve via the inherited accessors. **Breaking surface change
  (matrix-completion convention):** `res.counterfactual` is now the **1-D
  treated counterfactual path** (was the full `(N, T)` imputed matrix); the
  matrix moved to `res.counterfactual_matrix`, and the per-cell `effects`
  matrix moved to `res.effects_matrix` (the `effects` slot now holds the
  standardized `EffectsResults`). RMSI is a matrix-completion method with no
  donor weights, so `weights` records the method/rank. Mutating the frozen
  result raises pydantic `ValidationError`. RMSI plots via `result.plot()` and
  is pinned in `tests/test_result_contract.py`.
- **SPOTSYNTH migrated onto the two-family result contract.** `SPOTSYNTH.fit()`
  now returns `SpotSynthResults` as a frozen pydantic `EffectResult`: it
  populates the standardized sub-models (`effects`, `time_series`, `weights`,
  `inference`, `fit_diagnostics`, `method_details`) and exposes the flat
  accessors (`att`, `att_ci`, `counterfactual`, `gap`, `donor_weights`,
  `pre_rmse`). All previously public attributes still resolve, and `att_ci` now
  reads from `inference`. **One rename:** the result's former `inference` field
  (the `"bayes"`/`"frequentist"` label) is now `inference_method`, because
  `inference` is the standardized `InferenceResults` slot; the config field
  `SPOTSYNTHConfig.inference` is unchanged. Mutating the frozen result now raises
  pydantic `ValidationError` (not `dataclasses.FrozenInstanceError`). SPOTSYNTH
  plots via the standardized `result.plot()` and is pinned in
  `tests/test_result_contract.py`.

### Added
- **Standardized plotting foundation.** A nested `PlotConfig` on
  `BaseEstimatorConfig` (`plot=...`) centralizes plot cosmetics — observed/
  counterfactual color, linewidth, linestyle; intervention reference line;
  axis-label and title overrides; a user-suppliable `theme` — with sensible
  defaults and full back-compat (legacy `treated_color`/`counterfactual_color`/
  `display_graphs`/`save` fold in via `config.resolved_plot()`). The shared
  `utils.plotting.Plotter` gains `gap` and `event_study` archetypes alongside
  `observed_vs_counterfactual`, all config-driven and rendered in the house
  style. `EffectResult.plot(kind=...)` is the single entry point, driven by the
  standardized `time_series` sub-model (+ `intervention_time`) and the
  `PlotConfig` captured at fit time. FDID is migrated as the reference;
  the event-study archetype is validated on SDID output.
- **FDID validation (Path A + B), officially complete.** New
  `benchmarks/cases/fdid_hongkong.py` reproduces Li (2024)'s public Hong Kong
  GDP companion replication cell by cell (FDID ATT 0.0254 / 53.84% / pre-R² 0.843
  / 9 of 24 controls; DID 0.0317 / 77.62% / 0.505), guarded in CI by
  `tests/test_fdid_replication.py`; the Table 5 simulation (`fdid_table5.py`)
  remains the Path B check. Replication page updated to document both.
- **Vectorized metric primitives.** `utils/effectutils.py` (treatment effects)
  and `utils/fitutils.py` (goodness-of-fit / loss) split the former
  `effects.calculate` blob into bite-sized pure functions; `effects.calculate`
  and the new `results_helpers.build_effect_submodels` compose them so every
  estimator computes ATT/%ATT/gap/RMSE/R² from one consistent source.
- **Two-family result contract.** A common `MlsynthResult` base with two faces:
  `EffectResult` (alias of `BaseEstimatorResults`, the observational report)
  and `DesignResult` (the research design, whose `report` is an
  `EffectResult`). Exposed via the new `mlsynth/results.py`. Flat convenience
  accessors (`att`, `att_ci`, `counterfactual`, `gap`, `donor_weights`,
  `pre_rmse`) on `BaseEstimatorResults`.
- `mlsynth/tests/test_result_contract.py` pinning the contract on the
  reference estimators.
- `agents/agents_results.md`: the result-object contract and per-estimator
  migration Definition of Done.

### Changed
- `WeightsResults` is now the single weights container library-wide, exposing
  `donor_weights` / `time_weights` / `unit_weights` (was `donor_weights`
  only). Purely additive.
- **VanillaSC, FDID, TSSC** migrated onto the contract:
  - `FDID.fit()` and `TSSC.fit()` now return frozen Pydantic `EffectResult`
    subclasses (were frozen dataclasses). All previously public attributes and
    methods are preserved; the standardized sub-models (`effects`,
    `time_series`, `weights`, `inference`, `fit_diagnostics`,
    `method_details`) are now populated, so `res.effects.att`,
    `res.weights.donor_weights`, etc. read uniformly across the three.
  - `VanillaSC.fit()` is unchanged in shape (already returned
    `BaseEstimatorResults`); it gains the flat accessors via the base.
- Per-estimator configs relocated next to their helpers:
  `VanillaSCConfig`, `FDIDConfig`, `TSSCConfig` now live in
  `mlsynth/utils/<name>_helpers/config.py`. Re-exported from
  `mlsynth.config_models` via a lazy `__getattr__`, so existing imports
  (`from mlsynth.config_models import FDIDConfig`) keep working unchanged.

### Backward compatibility
- No public estimator API removed. Config imports from
  `mlsynth.config_models` still resolve to the same classes. The only
  intentional surface change: assigning to a frozen FDID/TSSC result now
  raises pydantic's `ValidationError` instead of the dataclass
  `FrozenInstanceError`.
