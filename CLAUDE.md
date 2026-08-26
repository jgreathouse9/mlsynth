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
   `MlsynthConfigError` / `MlsynthDataError`. No free-form kwargs. The shared
   bases live in `config_models.py`; per-estimator configs are being relocated
   next to their helpers (`utils/<name>_helpers/config.py`) and re-exported
   from `config_models.py` for backward compatibility.
2. **Data ingestion goes through `mlsynth.utils.datautils.dataprep`** (or a
   `<name>_helpers/setup.py` that wraps it). Do not hand-pivot pandas in an
   estimator — `dataprep` returns the canonical `Ywide` / `y` / `donor_matrix`
   / `pre_periods` / `post_periods` contract.
3. **Results use the two-family contract** (see `agents/agents_results.md`):
   every `fit()` returns an `EffectResult` (observational; alias of
   `BaseEstimatorResults`) or a `DesignResult` (experimental design, whose
   `report` is an `EffectResult`) — both subclass `MlsynthResult`. Result
   containers are **Pydantic models** (frozen where practical) that populate
   the standardized sub-models (`EffectsResults`, `FitDiagnosticsResults`,
   `TimeSeriesResults`,
   `WeightsResults`, `InferenceResults`, `MethodDetailsResults`) and keep
   estimator-specific outputs as typed fields. No ad-hoc dicts as the public
   return. Conformance is pinned in `tests/test_result_contract.py`.
4. **One estimator = one package**: `estimators/<name>.py` (thin) +
   `utils/<name>_helpers/{setup,pipeline,structures,plotter,...}.py`. Dispatcher
   estimators (e.g. `SPILLSYNTH`) add a method subpackage, not a new top-level
   estimator.
5. **Export** the class in `mlsynth/__init__.py` (import + `__all__`).
6. **Match the nearest existing estimator** before inventing a pattern.
   Canonical references: `MAREXConfig`, `LEXSCMConfig`, `RESCMConfig`, the
   `BaseEstimatorResults` hierarchy, and `mcnnm`/`vanillasc` for layout.
7. **Computation and presentation are separate.** Estimators and helpers compute
   and return; displaying, saving, formatting and printing are the caller's.
   A `plot_*` helper returns its `Figure` and does not call `plt.show()`; library
   code does not `print`. A diagnostic the caller might act on becomes a typed
   field on the result (usually `MethodDetailsResults`) or a `warnings.warn` —
   never stdout, and never discarded.

## Design doctrine (the Unix rules)

`agents/agents_unix.md` settles which of the Unix design rules bind here, which
are re-implemented in a typed medium, and which are refused — with the citations,
the measured backlog, and the checks. Read it before arguing that a structure is
or is not idiomatic. Three results from it are already invariants above:
invariant 3 is the Rule of Composition (the result contract is this library's
universal interface, and `mlsynth/spec.py` is its text boundary), invariant 4 is
the Rule of Modularity, and invariant 7 is the Rule of Separation plus the Rule
of Silence.

One rule is refused on purpose: Postel's "be liberal in what you accept". A
lenient validator turns a malformed panel into a number that looks like an
estimate, so `extra="forbid"` and fail-early validation stand.

Sweeping the two code rules (documentation-level checks, not gates — the
baseline counts and the AST versions are in `agents/agents_unix.md`):

```bash
grep -rn "^\s*print(" mlsynth/ --include=*.py | grep -v /tests/   # Rule of Silence
grep -rn "plt.show()" mlsynth/ --include=*.py | grep -v /tests/    # Rule of Separation
```

## Testing & TDD (test-first is mandatory)

Write tests **before** the code. Any new feature, helper, function, estimator
branch, or config option lands test-first — write the tests, watch them fail for
the right reason, then implement to green. Each new unit of behavior ships with,
at minimum: a **smoke** test (end-to-end on minimal input, right type / finite
output), **unit** tests of its invariants (assert invariants, not brittle
floats), **edge-case** tests (empty / degenerate inputs: no donors, single
donor, no pre-periods, near-singular / collinear, treatment at `t=0`), and
**failure** tests (invalid input raises the correct translated `Mlsynth*Error`
and a test asserts the failure is *reported*, not swallowed). A change is not
done until it is red→green across these levels and the new code is fully
covered — defensive / unreachable branches get `# pragma: no cover` with a
stated reason, never an untested gap. The layered architecture, patterns,
exception contract, and the instrument-selection contract — which of
`coverage` / `pytest` / `hypothesis` / `cosmic-ray` answers which question, and
why two of them are complements — live in `agents/agents_tests.md`, along with
the Unix rules applied to tests (a hard-to-test function is a design report;
generous fixtures and strict assertions; the failure names the invariant).

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

The **definitions of done** for benchmarking — by input scenario (paper only /
code excerpt / full repo) and path (A / B) — live in
`agents/agents_benchmarking.md`. Benchmark authoring is a **separate workstream**
from estimator/result-contract work: don't bundle new benchmark cases into a
migration or refactor PR.

## Docs conventions

- One `docs/<name>.rst` per estimator: When-to-use → Notation → Assumptions
  (numbered, each with a Remark) → Inference/diagnostics → runnable Example →
  Verification pointer → Core API autodoc. Follow `agents/agents_docs.md`
  — the binding math-notation canon (symbols derived from the `qdocs`
  blog; Shi–Huang for expository structure).
- No bold in prose. Never use RST bold (`**...**`) on doc pages or in README
  prose — the author does not write in bold. Bold is reserved for mathematics
  (`\mathbf{}` in math). For emphasis use wording, not weight; for terms-of-art
  use ``literal`` or *italic* sparingly. (Table content and section headings are
  fine; the rule is about emphasis in prose.)
- No self-referential framing. Never tell the reader that something is worth
  stating, worth knowing, worth noting, deserves mention, needs to be precise
  about, or is being recorded/flagged/emphasized. If it did not merit inclusion
  it would not be on the page, so the announcement is pure filler and reads as
  padding. Delete the frame and assert the thing: "The other two diverge, and by
  amounts worth stating." becomes "The other two diverge."; "One detail is worth
  recording: X." becomes "X."; "It is worth being precise about the norm here."
  becomes the precise statement of the norm. The same goes for "importantly",
  "note that", "crucially", and "interestingly" — cut them. This applies to docs
  pages, module and function docstrings, benchmark-case prose, and commit
  messages alike.
  - The frame hides in more than the obvious openers. Also cut "worth" plus a
    gerund in any form ("worth keeping", "worth recording", "worth having",
    "worth flagging", "worth restating", "worth repeating", "worth surfacing",
    "worth remembering", "worth getting right"), "deserves emphasis", "deserves
    a caution", "bears repeating", and "is easy to get wrong / overlook / miss".
    A section titled "A note on X" is the same move as a heading: name the thing
    ("A note on tightness" is "Tightness"; "Two findings worth keeping" is "Two
    findings"). Ordinary senses of the word stay — "no covariates worth
    balancing on", "post weeks are worth far fewer than pre weeks".
- No "rather", ever. The word is banned in docs prose, docstrings, benchmark
  prose and commit messages. It is an absolute rule so it can be checked with a
  grep, not argued about case by case. Rewrite:
  - `X rather than Y` -> `X, not Y` when Y is a noun phrase ("a face, not a
    point"; "in seconds, not minutes"), or `X instead of Y` when Y is a gerund.
  - `Rather than X, Y` -> `Instead of X, Y`.
  - `would rather X` -> `would prefer to X`.
  - The only exemption is quoted material: the SYNDES authors wrote "rather
    strong assumptions", and changing words inside quotation marks misquotes
    the source.
  This subsumes the conduct-foil problem it replaced. Constructions like
  "documented rather than tuned away", "reported as such rather than papered
  over", "surfaces this as a diagnostic rather than hiding it" and "enforced at
  ingestion rather than assumed" all named an alternative nobody would have
  chosen, so they praised the choice instead of stating the fact. With the word
  gone, the positive form is forced: "documented"; "the estimator checks this at
  ingestion and raises when it fails".
- No "quietly" or "loudly". Whether a check raises or returns silently is an
  implementation concern; these pages are about econometrics. "the fit fails
  loudly instead of dropping a criterion" is "the fit raises instead of dropping
  a criterion"; "a choice that quietly changes the answer" is "a choice that
  changes the answer".
- No "load-bearing". Say what the sentence claims: "the relaxation the result
  rests on", "three details decide the answer", "the step that decides the
  answer".
- Sweeping for all of the above:

  ```
  \brather\b|\bquietly\b|\bloudly\b|load.bearing
  \bworth\b|\bdeserves?\b|^A note on
  \b(crucially|importantly|interestingly)\b|\bnote that\b
  ```

  The first line should return only quoted material. The other two need
  reading: ordinary senses of "worth" stay, and "Zheng (2025) note that ..."
  reports what an author wrote. When rewriting at scale, substitute in place so
  only the affected lines change -- reflowing whole paragraphs produces a diff
  nobody can review -- and re-check that bullet continuations kept their
  hanging indent, which a naive line-join silently destroys.
- Write for a non-expert reader: assume the reader wants to learn what the
  method is and does, not that they already know synthetic control. Define
  jargon on first use.
- Link every estimator page to its verification: the benchmark case
  (`benchmarks/cases/<name>.py`, link the source on GitHub) and/or its
  replication page, so readers can see it is validated and inspect the outputs.
  Keep the benchmarks index page current as cases are added.
- Keep the decision tree (`docs/choose.rst`) current as estimators are added or
  change regime — it is the primary "which estimator do I use" navigation.
- Section underlines must be ≥ the title length (RST requirement).

## Git

- Develop on the assigned feature branch; commit with clear messages.
- Commit author/committer email: `noreply@anthropic.com`.
- Don't create a PR unless asked.

### One estimator, one branch, one scope

Every new estimator gets its **own branch** and its **own scope**. Do not add a
new estimator on a branch that is already carrying other work, and do not carry
unrelated work onto an estimator's branch. Concretely:

- Branch per estimator, named for it (e.g. `claude/compsc`), cut fresh from
  `main` — not stacked on another feature branch.
- The branch contains only that estimator: config, estimator class,
  `utils/<name>_helpers/`, its tests, its docs page, the `__init__.py` export,
  and the `docs/choose.rst` entry. Nothing else.
- Adjacent work goes on its own branch: benchmark cases (see the replication
  contract above), refactors of shared helpers, doc-wide edits, and changes to
  this file. If an estimator genuinely needs a shared helper changed, land that
  change first on its own branch and rebase.
- A paper review or replication spike is its own scope too — it produces a
  recommendation, not estimator code.

### Merging PRs (standing authorization, with guardrails)

Claude Code may merge a pull request **without re-asking** only when **all** of
these hold:

1. **Trigger** — the user explicitly says to merge it (e.g. "merge it",
   "go ahead and merge"), **or** has given standing authorization for
   `claude/*` PRs (see below). Absent either, ask first.
2. **Preconditions** — required CI is **green**, there are **no unresolved
   review threads**, and the PR has **no merge conflicts** with its base.
3. **Scope** — the PR originates from a `claude/*` branch produced in this
   session, targeting `main`. Anything else: ask first.
4. **Method** — **squash-merge**, and **delete the source branch** after a
   successful merge.

**Hard stops (never, even if asked):** do not force-merge or override a failing
required check; do not merge if the diff has drifted from what was discussed;
do not merge into anything other than the agreed base. When in doubt, ask.

#### Standing authorization for `claude/*` PRs

The author has granted blanket approval: Claude Code may open **and merge** a
pull request from a `claude/*` branch into `main` without asking each time, as
long as conditions 2–4 above all hold — CI green, no unresolved review threads,
no conflicts, squash-merge, delete the branch. This is a default, not an
obligation: still ask when the change is larger or more contentious than the
work that was discussed, when it touches the result contract or another shared
invariant, or when you are not confident the diff is what the author expected.
The hard stops above are unaffected.

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
