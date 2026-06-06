---
description: Scaffold a new mlsynth estimator to the repository contract
argument-hint: "<ESTIMATOR_NAME> [--paper <ref>] [--dispatcher <existing>]"
---

# New Estimator

Create a new estimator that matches mlsynth's conventions exactly. Prefer
continuation over reinvention — copy the nearest existing estimator's shape.

## Checklist

1. **Config** — `config_models.py`: `<NAME>Config(BaseEstimatorConfig)`, typed
   `Field(...)` with descriptions, `extra="forbid"`, validators raising
   `MlsynthConfigError`/`MlsynthDataError`.
2. **Helpers** — `utils/<name>_helpers/`: `setup.py` (long df → inputs, via
   `dataprep`), `pipeline.py` (the method), `structures.py` (frozen results),
   `plotter.py`, `__init__.py`.
3. **Estimator** — `estimators/<name>.py`: thin class taking the config, a
   `.fit()` returning a `BaseEstimatorResults` (or a mirroring structure).
4. **Export** — `mlsynth/__init__.py` (import + `__all__`).
5. **Tests** — `tests/test_<name>.py`: reproduction/recovery, config-validation
   error paths, plotting smoke; aim for high coverage (`coverage run --timid`).
6. **Docs** — `docs/<name>.rst` (When-to-use → Notation → Assumptions+Remarks →
   Inference → runnable Example → Verification pointer → autodoc) and a
   `docs/replications/<name>.rst` page; add both to the toctrees.
7. **Benchmark** — `benchmarks/cases/<name>.py` capturing the validation.
8. **Verify** — full `pytest`, `tools/gen_llms_txt.py`, RST underline check.
