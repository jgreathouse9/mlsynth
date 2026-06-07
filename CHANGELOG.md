# Changelog

All notable changes to **mlsynth** are recorded here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); per the result-migration
Definition of Done (`agents/agents_results.md`, gate F), every estimator
migrated onto the two-family result contract gets an entry describing what it
now returns and the back-compat guarantee.

## [Unreleased]

### Added
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
