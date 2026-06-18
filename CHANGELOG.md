# Changelog

All notable changes to **mlsynth** are recorded here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); per the result-migration
Definition of Done (`agents/agents_results.md`, gate F), every estimator
migrated onto the two-family result contract gets an entry describing what it
now returns and the back-compat guarantee.

## [Unreleased]

### Fixed
- **SYNDES no longer leaks an `IndexError` when restrictions make the candidate
  pool empty.** Over-constrained `top_K > 1` designs (in-sample, holdout, or ic
  selection) whose every MIP solve is infeasible now raise a translated
  `MlsynthEstimationError` naming the restrictions, instead of a bare "list
  index out of range". Surfaced while building the docs gallery.

### Added
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
  weights**, so the `weights` slot records the method rather than per-donor
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
