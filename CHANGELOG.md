# Changelog

All notable changes to **mlsynth** are recorded here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); per the result-migration
Definition of Done (`agents/agents_results.md`, gate F), every estimator
migrated onto the two-family result contract gets an entry describing what it
now returns and the back-compat guarantee.

## [Unreleased]

### Changed
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
