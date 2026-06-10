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

## Field notes (hard-won)

Patterns and traps from building the SEQ_SDID, SPARSE_SC, NSC, VanillaSC, MASC,
TASC, FSCM and PDA cases. These complement the DoD above; they are the things
that actually cost time.

### Tolerances & determinism

- **Center `EXPECTED` on the *measured* deterministic value, not the paper's
  printed one.** Demonstrate-first, capture the exact output, then set the
  center to what the code produced and the tolerance to the regression band you
  want. Re-centering after the first registered run is normal (did it for
  `seq_sdid_mc`, `nsc_mc`, `tasc_mc`). A deterministic case's tolerance is a
  *regression guard*, not a noise budget — it does not need to be wide.
- **Pin every RNG seed and say which.** Structural seed, per-draw shock seed,
  and bootstrap seed are independent — fix all three (`seq_sdid_mc` uses
  structural `2024`, shock `8000+m`, bootstrap `7`). Global-search optimisers
  count too: MSCMT's `differential_evolution` is seeded, so VanillaSC/MASC
  covariate fits are deterministic.
- **Read the actual got-values off a failing run.** Registering with rough
  `EXPECTED` then reading the harness's `got` column is the fastest way to pin
  exact deterministic cells (and to catch a key-name typo — a `MISSING` row
  means the `run()` key doesn't match an `EXPECTED` key, e.g. `LASSO` vs
  `lasso`).

### Monte-Carlo design (Path B)

- **Fix the structure; redraw only the shocks.** Re-drawing the whole DGP per
  replication injects variance a within-panel bootstrap cannot see, so coverage
  comes out wrong. The paper's "treat estimated components as fixed, resample
  the idiosyncratic shocks" device is what makes the bootstrap a valid coverage
  measure. This was *the* fix for `seq_sdid_mc`; `simulate_*` helpers that back a
  coverage MC should separate a `calibrate_*` (structure, once) from a
  `simulate_replication` (shocks, per draw). Same shape in `tasc_mc`.
- **Don't assert a signed bias at benchmark `M`.** Per-period signed bias is
  often ~0.01 with SE ≈ SD/√M; at `M`≈40 it is pure noise. Assert the *robust*
  quantities the paper highlights — coverage near nominal, error-shrinks-with-J,
  RMSE ratios `< 1` — and merely *report* bias (`nsc_mc`). MC tolerance scale:
  coverage SE ≈ √(p(1−p)/(M·n_periods)).
- **Geometry vs cells, stated honestly.** Under scenario-1 reconstruction you
  usually can't hit exact cells. Assert the qualitative geometry (all ratios
  `<1`; DiD coverage collapses while the method stays nominal) with headline
  cells in a generous band, and *say in prose it is geometry, not a cell match*.
- **Watch for design-specific feasibility traps.** SEQ_SDID needs ≥2
  later/never-treated donor cohorts per estimated cohort or its effect collapses
  to an unbalanced DiD; the MC must spread adoption and cap `a_max` so every
  estimated cohort is donor-balanced. Surface such traps as estimator
  diagnostics, not just benchmark knobs.

### Cross-validation against the authors' code

- **Stochastic hyperparameter selection does not port.** If the reference's CV
  draws a random held-in unit each fold (NSC's `fn_cv`, MASC's analytic CV), you
  cannot reproduce its *selection* in Python. Fix the reference's **reported
  selected** hyperparameter (NSC `a*=0.3, b*=0.7`) and cross-validate the
  deterministic downstream estimate. Document the substitution.
- **Keep the reference script for provenance even when you can't run it.** No
  R/Gurobi in CI is fine: store the script under `benchmarks/R/` (e.g.
  `nsc_tian2023_reference.R`) and embed the published table values as the
  reference. A future R-equipped run regenerates the dump.
- **The reference *code* is the source of truth for algorithmic choices the
  paper glosses.** The MASC paper text did not make clear that its Basque
  application matches on covariates and uses `synth`'s V-optimiser; only
  `SC_application.R` / `Estimator_Code.R` did. Read the estimator, CV, and
  application scripts, not just the prose.

### Isolating a discrepancy

- **When an estimate drifts from a reference, bisect the pipeline.** MASC on
  Basque gave −$816 vs KMPT −$580. Running *VanillaSC* (which has the ADH/MSCMT
  V-optimiser) on the same data reproduced −$585 to within $5 — proving the SC
  weights and pre-period were right and localising the bug to MASC's *solver
  choice* (it delegated to the Malo bilevel solver) and its *matching basis*
  (outcomes, not covariates). Reproduce each component with a known-good
  estimator before concluding "the method differs".
- **Verify treatment timing and the pre-period before blaming the method.**
  Confirm the `treat` column turns on at the paper's date and the pre-period
  spans the paper's years (Basque: `terrorism=1` at 1970, pre 1955–1969). A
  wrong window mimics a method discrepancy.
- **Predictor-weight `V` is non-identified.** Different V-optimisers (MSCMT/synth
  vs Malo bilevel) give different `W` on over-parameterised predictor sets, and
  the difference cascades (here, a CV over-blending matching). Match the
  reference's optimiser; expose the alternative as a toggle and default to the
  reference-faithful one (`sc_backend="mscmt"`, `match_on="covariates"`).

### Data, scope, and drift

- **The shipped panel must be in the paper's transformed form.** PDA reproduced
  the Hong Kong CEPA effect (2.48% vs the paper's 2.65%) because `HongKong.csv`
  *is* the HCW panel, but did **not** reproduce the luxury-watch −27.92% because
  the shipped China series is not the paper's YoY-growth / `T1=24` form. Check
  the outcome transformation and pre-period length, not just column names.
- **Benchmarks pin reality; prose drifts — re-measure when graduating.** This
  round found a stale worked example (`docs/masc.rst` claimed MASC ATT −$641; the
  code gave −$816) and an *unbacked* catalogue claim (TASC "4.5× margin"; the
  durable measurement was ~1.1–1.2×). Graduating a prose/test claim means
  *running it and correcting the docs to what the code does*.
- **"Graduate the test" usually means "run on the real dataset and capture".**
  Estimator tests mostly assert *structural* properties on synthetic panels
  (weights sum to 1, ATT≈planted), not the paper's numbers — so the durable case
  is a fresh empirical/MC run, not a promoted assertion.

### Runtime

- **Profile one fit before sizing `M`; the cost is usually solve *count*, not the
  solver.** NSC/MASC inner QPs already use Clarabel with cached skeletons — the
  9 s/fit at `J=50` was the CV grid × leave-one-out × iterations. Reduce solves
  (fix hyperparameters, the paper's CV-once-per-structure trick), don't rewrite
  the solver. Bayesian/MCMC cases are the real runtime wall: BVSS's 3000-iter
  MCMC on 87 donors exceeded 9 min and is **not** a practical durable case
  without a fast-MCMC config; conformal inference is heavy too (`sparse_sc_prop99`
  ≈ 208 s — acceptable, but know it going in).

### Triage before committing to a case

Split the un-benchmarked estimators into **quick graduations** (shipped data +
a crisp printed number → a durable case in ~an hour: FSCM, PDA-HK, VanillaSC)
and **reconstruction projects** (need a reference run, data re-derivation, or a
DGP rebuild: MLSC's reference dump, the PDA size/power simulation, BVSS
fast-MCMC). Do the quick ones first; scope the reconstructions as their own
workstreams.

### Stacked benchmark PRs

`docs/replications.rst` is the one file every benchmark PR edits, so parallel
benchmark branches conflict there. Either keep one PR per coherent unit, or
stack on the prior branch and, after the base squash-merges, rebase with
`git rebase --onto <new-main> <old-base> <branch>` to drop the squashed commits
— the per-case tolerances are content-stable, so it replays clean.
