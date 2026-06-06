# CLAUDE.md

Operational guidance for AI agents (Claude Code) working in **mlsynth**. This
complements the design philosophy in `agents/*.md` — read those for *why*; read
this for *what to run and what invariants to preserve*.

## What mlsynth is

A strongly-typed Python library of synthetic-control / DiD estimators for causal
inference on panel data. ~45 estimators, each: a Pydantic **config**, a thin
**estimator class** with `.fit()`, a `utils/<name>_helpers/` package doing the
work, and a standardized **results object**. Most carry a **replication** that
matches the source paper or a reference implementation.

## Common commands

```bash
pip install -e .                      # editable install
pytest mlsynth/tests/                 # full test suite
pytest mlsynth/tests/test_fdid.py -q  # one file
python -m pytest mlsynth/tests/test_<x>.py -q -p no:cacheprovider   # avoid cache noise

# coverage of one estimator (pure-Python tracer avoids the numpy C-tracer clash)
coverage run --timid -m pytest mlsynth/tests/test_<x>.py && coverage report

python tools/gen_llms_txt.py          # regenerate the agent-facing llms.txt index
python benchmarks/run_benchmarks.py --all     # durable paper/reference validation
```

## Architecture invariants (do not break these)

1. **Every estimator has a dedicated Pydantic config** inheriting
   `BaseEstimatorConfig` (or `BaseMAREXConfig`), with `extra="forbid"`,
   `Field(...)` descriptions, and validators that fail early with
   `MlsynthConfigError` / `MlsynthDataError`. No free-form kwargs.
2. **Data ingestion goes through `mlsynth.utils.datautils.dataprep`** (or a
   `<name>_helpers/setup.py` that wraps it). Do not hand-pivot pandas in an
   estimator — `dataprep` returns the canonical `Ywide` / `y` / `donor_matrix`
   / `pre_periods` / `post_periods` contract.
3. **Results use the standardized models** in `config_models.py`
   (`BaseEstimatorResults`, `EffectsResults`, `FitDiagnosticsResults`,
   `TimeSeriesResults`, `WeightsResults`, `InferenceResults`,
   `MethodDetailsResults`) — or a frozen `<name>_helpers/structures.py`
   dataclass that mirrors them. No ad-hoc dicts as the public return.
4. **One estimator = one package**: `estimators/<name>.py` (thin) +
   `utils/<name>_helpers/{setup,pipeline,structures,plotter,...}.py`. Dispatcher
   estimators (e.g. `SPILLSYNTH`) add a method subpackage, not a new top-level
   estimator.
5. **Export** the class in `mlsynth/__init__.py` (import + `__all__`).
6. **Match the nearest existing estimator** before inventing a pattern.
   Canonical references: `MAREXConfig`, `LEXSCMConfig`, `RESCMConfig`, the
   `BaseEstimatorResults` hierarchy, and `mcnnm`/`vanillasc` for layout.

## The replication contract

Every estimator is validated by one of (see `docs/replications.rst`):
- **Path A** — the paper's empirical result on the authors' data;
- **Path B** — the paper's Monte Carlo / simulation table;
- **Cross-validation** — match an authoritative reference implementation.

Make validation **durable**: add a `benchmarks/cases/<name>.py` (and an R script
under `benchmarks/R/` if it needs a reference), not a throwaway script. Each
replication gets a **dedicated docs page** under `docs/replications/<name>.rst`,
linked from the estimator page's short "Verification" pointer (see
`docs/replications/fdid.rst` for the template).

## Docs conventions

- One `docs/<name>.rst` per estimator: When-to-use → Notation → Assumptions
  (numbered, each with a Remark) → Inference/diagnostics → runnable Example →
  Verification pointer → Core API autodoc. Follow `agents/agents_docs.md`
  (Shi–Huang notation canon).
- Section underlines must be **≥** the title length (RST requirement).

## Git

- Develop on the assigned feature branch; commit with clear messages.
- Commit author/committer email: `noreply@anthropic.com`.
- Don't create a PR unless asked.

## AI workflow (slash commands)

Reusable, codified workflows live in `.claude/commands/`:
- `/paper-review <pdf|url>` — assess a candidate paper for mlsynth (new method?
  implementable? replication path? build cost? recommendation).
- `/replicate <paper>` — the demonstrate-first replication loop (dataprep →
  port → validate vs reference → decide build).
- `/new-estimator <name>` — scaffold a new estimator to the contract above.
- `/ai-review` — cross-model review of the working diff before a PR.

Optional plan-gate: `.claude/hooks/check-plan-review.sh` (wire via
`.claude/settings.json`) blocks plan approval until a plan review exists.
