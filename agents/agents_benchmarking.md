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

## Provisioning and running the full reference stack

The cross-validation cases skip themselves when their reference toolchain is
absent, so a bare `--all` run is **green-but-incomplete** — it silently skips
every R and external-Python reference. To actually exercise the cross-checks
(the strongest evidence we have) you must stand up the whole reference stack.
This section is the recipe, learned by doing it end-to-end in a sandboxed cloud
environment. It complements `benchmarks/README.md` (which lists the cases) with
the *how to provision the toolchain* that neither doc previously captured.

### Know your network profile first — it dictates everything

The install strategy is entirely determined by what the environment can reach.
**Probe before you install** (`curl -sS -m 12 -o /dev/null -w "%{http_code}"`):

| Endpoint | Cloud sandbox (observed) | Consequence |
|---|---|---|
| CRAN (`cloud.r-project.org`) | **403 blocked** | `install.packages()` / `Rscript requirements.R` **do not work**. You cannot pull R packages the normal way. |
| GitHub codeload (`codeload.github.com`) | **200 open** | This is the *only* way in for R packages: compile from the CRAN GitHub mirror (`cran/<pkg>`) and from upstream repos at pinned commits. Reference-repo clones also work. |
| PyPI (`pypi.org`) | **200 open** | Python reference deps install normally; one (`scmrelax`) comes from a git URL. |
| apt (Ubuntu main) | reachable | Prebuilt `r-cran-*` packages (the bulk of R dependencies) install fast — *if* you fix the broken PPAs first. |

This is exactly why `benchmarks/R/install_*.sh` are written the way they are:
**apt for the prebuilt majority, `codeload.github.com/cran/<pkg>/tar.gz/<ref>` +
`R CMD INSTALL` for the leaves, never a CRAN call.** Trust those scripts; they
encode the pinned commits/tags that keep the cross-check byte-stable.

### R: the install order that works from a CRAN-blocked box

1. **Disable broken third-party PPAs, then `apt-get update`.** The sandbox ships
   `deadsnakes` and `ondrej/php` PPAs that are unsigned/403 and make
   `apt-get update` error out (which then aborts installs). Move them aside:
   `mv /etc/apt/sources.list.d/*deadsnakes* …disabled` (same for `ondrej`),
   then `apt-get update`.
2. **apt-install the prebuilt R deps.** `r-base r-base-dev build-essential cmake
   gfortran` plus the `r-cran-*` dependency closure (dplyr, tidyr, magrittr,
   rlang, purrr, fnn, rcpp, rcpparmadillo, rcppeigen, bh, glmnet, mass, matrix,
   kernlab, optimx, rgenoud, survey, …). **A single bad package name aborts the
   whole `apt-get install` — it installs nothing.** Verify names with
   `apt-cache policy <pkg>` first: `r-cran-optmatch`, `r-cran-microsynth`,
   `r-cran-synth`, `r-cran-synthdid`, `r-cran-did` are **not** in apt and must
   come from the GitHub mirror.
3. **Compile the non-apt leaves from the GitHub CRAN mirror.** Helper:
   `curl -sL https://codeload.github.com/$REPO/tar.gz/$REF -o p.tgz && tar xzf
   p.tgz && R CMD INSTALL --no-docs --no-help <dir>`. `Synth` (needs kernlab,
   optimx, rgenoud — all apt) and `microsynth` (needs survey — apt) install from
   `cran/Synth` / `cran/microsynth`.
4. **Run the pinned install scripts in dependency order, one at a time:**
   - `benchmarks/R/install_augsynth.sh` → `augsynth` + `osqp` + `S7` +
     `LiblineaR` (for `geolift_augsynth_ref`, `ascm_kansas`).
   - `benchmarks/R/install_pensynth.sh` → `LowRankQP` (for `pensynth_prop99`;
     the authors' `wsoll1.R`/`TZero.R` are *cloned* separately by
     `clone_pensynth.py`, not installed).
   - `benchmarks/R/install_geolift.sh` → re-runs install_augsynth.sh, then the
     heavy `Boom → BoomSpikeSlab → bsts → CausalImpact → MarketMatching → gsynth
     → GeoLift` chain (for `geolift_marketselection_ref`). **`Boom` is a
     ~15–25 min single C++ compile** — it dominates the whole provisioning time.
     The chain is frozen to the last R-4.3-compatible release set (newer `Boom`
     needs R ≥ 4.5).
   - `Synth`/`synthdid`/`did` in `requirements.R` go via CRAN, which is blocked
     — pull `Synth` from the mirror (above); `synthdid`/`did` are not actually
     `library()`-d by any current case (the `sdid`/`did` cross-checks use the
     Python `causaltensor` reference), so skip them unless a new case needs them.

   **Do not run two apt-using scripts concurrently** — they collide on the dpkg
   lock. Serialize them (each `install_*.sh` calls `apt-get`).

### External Python references: install individually, mind two apt-managed traps

The cross-val cases that use a Python reference need these — but **install them
one at a time**, because a batched `pip install a b c …` aborts wholesale on the
first failure and leaves none installed:

| Package | Source | Cases |
|---|---|---|
| `causaltensor` | PyPI | `sdid_prop99`, `mcnnm_prop99` |
| `cvxopt` | PyPI | `linf_crossval_ref` |
| `cvxpy` | PyPI | `rescm_relax_ref` |
| `libpysal` | PyPI | `spsydid_state_mc` (reads `.gal` weights) |
| `kneed` | PyPI | `clustersc_subgroups_ref` (authors' `syclib`) |
| `toolz` | PyPI | `rsc_shen_coverage` (authors' `var.py`) |
| `scmrelax` | **git**: `git+https://github.com/PanJi-0/scmrelax.git` (not on PyPI) | `rescm_relax_ref` |

Two environment traps that masquerade as install failures:

- **`beautifulsoup4` is apt-managed with no `RECORD` file**, so when `libpysal`'s
  deps try to upgrade it pip dies with `Cannot uninstall beautifulsoup4 … RECORD
  file not found` and aborts the transaction. Pre-empt with
  `pip install --ignore-installed beautifulsoup4` once, then install `libpysal`.
- **`numpy`'s apt-installed dist-info has invalid metadata**, so pip prints
  `Skipping …numpy-…dist-info due to invalid metadata` warnings. Harmless on its
  own, but combined with a batched install it can confuse the resolver — another
  reason to install one package per `pip` call.
- **`causaltensor` pins `numpy<2.0`** but imports and runs fine under the
  installed `numpy 2.4.x`. Do **not** downgrade numpy globally to satisfy it —
  that would break mlsynth itself. The pin is advisory; the cases pass.

Reference *repos* (not packages) clone lazily on first case run into
`benchmarks/reference/.cache/` (git-ignored) via the `clone_*.py` pins. GitHub
is open, so they just work; no pre-step needed (running `clone_*.py` directly
needs the repo root on `PYTHONPATH`, e.g. `python -m benchmarks.reference.clone_x`).

### Running it, and reading the result honestly

- **`--with-reference` is currently a forward-compat no-op:**
  `registry.NEEDS_REFERENCE` is empty, so every one of the ~86 cases already runs
  under `--all` and each *self-skips* (`[SKIP]`, never a fail) when its reference
  is missing. The flag exists for cases that would be *gated out* of `--all`;
  pass it anyway for intent, but know `--all` already attempts every reference.
- **Verify references individually before the full sweep.** A full
  `--all --with-reference` run is 30–60 min; iterate with
  `--case <name> --with-reference` per reference case to confirm each passes
  *live* (fast feedback, isolates a broken install to one line).
- **Do the installs first, then run the suite — never overlap them.** Running
  `--all` *while* R packages compile starves the Python process of CPU; a slow
  Monte-Carlo case (e.g. `msqrt_sim`) then looks hung when it is merely waiting.
  Kill any sweep you started before provisioning finished and re-run clean.
- **The clean end state:** with the stack up, all reference cases flip
  `[SKIP] → [PASS]`. In the reference run this means the 7 external-Python cases
  (causaltensor ×2, libpysal, kneed, toolz, scmrelax/cvxopt, cvxpy), the 9 R
  cross-checks (`masc_basque`, `microsynth_seattle`, `nsc_prop99`,
  `pensynth_prop99`, `pda_luxurywatch`, `scmo_concatenated_mc`, `siv_syria_mc`,
  `cwz_ttest`, `cwz_mc`), and the GeoLift cases (`geolift_augsynth_ref`,
  `geolift_marketselection`, `geolift_marketselection_ref` against *live*
  `GeoLiftMarketSelection`) all pass alongside the ~67 pure-Python cases.

### If you had to do it again

Wrap the whole thing in a provisioning script (the obvious next step is the
deferred `benchmarks/Dockerfile` → GHCR image baking exactly these `apt` +
`codeload` + `pip` steps, so the Boom compile happens once). Until then: probe
the network, disable the bad PPAs, apt the prebuilt R closure, run the three
`install_*.sh` in order (budget for Boom), `pip install` the Python refs one at
a time with the bs4 pre-empt, then verify case-by-case before the full sweep.

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
