# agents_benchmarking.md

Operational guide for adding **durable, re-runnable** validation to an mlsynth
estimator. This is the *process* doc (the "why" and the definitions of done);
`benchmarks/README.md` is the *mechanics* doc (the "what to run"), and
`docs/replications.rst` is the public catalogue. Read `.claude/commands/`'s
`/replicate` for the demonstrate-first loop that *precedes* the work described
here.

A benchmark is not finished when it prints a number that looks right once. It is
finished when it is **registered, seeded, tolerance-justified, documented, and
re-runnable in CI** — so the cross-check that would otherwise be done by hand
and thrown away becomes a permanent regression guard.

## The two axes

Every benchmark is classified by **what you're matching** (the path) and **what
you have to match against** (the input scenario). The definition of done is the
intersection of the two.

### Paths — the estimand of the match

- **Path A** — reproduce the paper's *empirical* result on the authors' own
  data (ATT, %ATT, pre-fit, donor counts, ...).
- **Path B** — reproduce the paper's *Monte Carlo / simulation* table
  (PMSE grids, coverage, bias/variance geometry).
- **Cross-validation** is not a separate path here: it is the *strongest form of
  evidence* available under scenarios 2–3 (matching a reference implementation
  cell-by-cell), and it is folded into the DoD for those scenarios below.

### Input scenarios — what the source gives you (weakest → strongest)

1. **Paper only** — prose, tables, figures. No code; data must be obtained or
   reconstructed.
2. **Code excerpt** — R / MATLAB / Stata snippets or a partial script, but not a
   runnable end-to-end project.
3. **Full repo** — the authors' complete, runnable replication package (code
   **and** data).

## Universal DoD (every benchmark, every cell)

Regardless of path or scenario, a benchmark is *done* only when **all** of:

- [ ] `benchmarks/cases/<name>.py` exposes `run() -> dict[str, float]` and
      `EXPECTED: dict[str, tuple[float, float]]` (value, absolute tolerance).
- [ ] Registered in `benchmarks/registry.py` (and added to `NEEDS_REFERENCE`
      if the case reads an R/MATLAB/Stata dump).
- [ ] `python benchmarks/run_benchmarks.py --case <name>` passes — every
      reported number is within its declared tolerance.
- [ ] **Deterministic**: seeded; re-running yields identical numbers. Stochastic
      cases pin `seed=` and document `M` (the number of draws).
- [ ] **Tolerance is justified** in a comment — tight enough to catch a real
      regression, loose enough to absorb the *declared* Monte-Carlo noise (a
      smaller `M` than the paper) or display rounding. State the reasoning, not
      just the number.
- [ ] **Values read through the standardized result accessors** (`res.att`,
      `res.att_ci`, `res.counterfactual`, `res.gap`, `res.pre_rmse`,
      `res.donor_weights`) — never estimator-private fields — so a future
      result-contract change cannot silently break the check.
- [ ] A dedicated `docs/replications/<name>.rst` (template:
      `docs/replications/fdid.rst`) states the **headline number(s)** and how
      the match was established; it is linked from `docs/replications.rst` and
      from the estimator page's short "Verification" pointer.
- [ ] **Provenance** recorded in the case docstring: the exact paper
      table / figure / page (Path A/B), and for scenarios 2–3 the upstream file
      plus commit / version of the reference.

## DoD by scenario × path

### Path A — empirical on the authors' data

| Scenario | Increment over the Universal DoD |
|---|---|
| **1. Paper only** | Source the data: the authors' public dataset, or a **faithfully reconstructed panel**. A reconstructed panel **counts as a full Path A pass provided the reconstruction is documented** — source series, transformations, donor pool, and treatment timing recorded in enough detail that a reader could rebuild it independently. Match the paper's **printed** ATT / %ATT / pre-fit / donor-count to **display precision** (the decimal places the paper shows). |
| **2. Code excerpt** | Use the excerpt to pin down ambiguous preprocessing the prose leaves underspecified (normalization, donor pool, predictor set, timing). DoD = scenario 1 **plus** every modeling choice traced to a line in the excerpt and cited in the case docstring. |
| **3. Full repo** | **Cross-validation is mandatory.** Run the authors' code on the same panel, dump its output, and match **cell-by-cell to numerical precision** (not merely display precision). The reference dump is regenerable — a script under `benchmarks/R/` (or `benchmarks/<lang>/`) — and the case reads that output, skipping gracefully when the toolchain is absent. |

### Path B — Monte Carlo / simulation table

| Scenario | Increment over the Universal DoD |
|---|---|
| **1. Paper only** | Re-implement the DGP **from the paper's description** into a reusable `simulate_*` helper under `utils/<name>_helpers/` (never inline in the case). Match the **qualitative geometry plus the headline cells** of the target table (e.g. all MSE-ratio cells `< 1`; the specific cells the paper highlights). Use a modest `M` for runtime; tolerance absorbs the gap to the paper's `M`. Document every DGP detail you had to infer. |
| **2. Code excerpt** | The DGP re-implementation is **validated against the excerpt** — same noise structure, factor loadings, treatment assignment. DoD = scenario 1 **plus** the simulator's parameterization traced to the excerpt; tolerances may tighten because the DGP is no longer inferred. |
| **3. Full repo** | **Cross-validation is mandatory.** Port (or call) the authors' exact DGP and seed-match so that at the paper's `M` you converge to the published cells within tight tolerance. The gold standard is cross-validating the *simulator itself* — your draws vs theirs on a fixed seed — with the seed correspondence recorded. |

## Tolerances & runtime

- **Deterministic (Path A, no resampling):** tolerance covers only display
  rounding — typically `<= 5e-3`, or one unit in the last printed place.
- **Stochastic (Path B; placebo / bootstrap inference):** tolerance scales with
  `1/sqrt(M)`. State the paper's `M`, the benchmark's `M`, and why the chosen
  tolerance brackets the published value.
- **Runtime budget:** keep a single case well under a minute where possible
  (`fdid_table5` uses `M=400` vs the paper's 10,000). Heavy reference runs live
  behind `--with-reference` and are skipped when R/MATLAB is unavailable.

## Scope guardrails

- A benchmark validates **numbers**, not the result-object shape. This is why
  values are read through the standardized accessors.
- Prefer **pure-Python, no-reference** cases when the paper gives enough to match
  Path A/B directly; reserve reference dumps for scenario 3 (where they are
  mandatory) or when the paper alone is insufficient to disambiguate the method.
- Benchmark authoring is a **workstream of its own**, independent of the result
  contract and of the estimator's `fit()` implementation. A migration or refactor
  PR should not bundle new benchmark cases, and vice versa.

## Adding a benchmark — checklist

1. Run the `/replicate` demonstrate-first loop until you have a number match in a
   scratch script.
2. Promote it: `benchmarks/cases/<name>.py` with `run()` + `EXPECTED`.
3. Register in `benchmarks/registry.py` (+ `NEEDS_REFERENCE` if applicable).
4. If scenario 3, add the reference script under `benchmarks/R/` (or
   `benchmarks/<lang>/`) and have the case read its dumped output.
5. Write `docs/replications/<name>.rst`; link it from `docs/replications.rst`
   and the estimator page's Verification pointer.
6. Confirm `python benchmarks/run_benchmarks.py --case <name>` passes, then
   `--all` to confirm no regression.
