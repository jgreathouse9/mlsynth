# Future Integrations

> **Purpose.** A living roadmap of cross-cutting capabilities we want to
> bring into mlsynth, with the learnings that motivate them. Items start in
> **Planned / In progress** and move to **Done** (bottom) as they land in the
> library. Keep the *learnings* even after an item is done — they are the
> rationale future agents will need.

---

## 1. Degrees of Freedom & Information Criteria for SCM

**Status: Planned (exploration done; no library code yet).**

### Source

> Pouliot, G. A., Xie, Z., & Liu, Z. (2026). "Degrees of Freedom and
> Information Criteria for the Synthetic Control Method." arXiv:2207.02943v2.

Related, for the forward-selection / factor-PDA branch:

> Shi, Z., & Huang, J. (2023). "Forward-selected panel data approach for
> program evaluation." *Journal of Econometrics*, 234(2), 512–535.

### The idea in one line

Give SCM an *estimable degrees-of-freedom* statistic, then use it to build a
SURE information criterion that selects tuning parameters (or model variants)
**without cross-validation**:

```
df(SCM)            = |A| - 1            # |A| = number of donors with nonzero weight
df(PSCM, lam)      = (1 + lam) * (|A| - 1)
IC(theta)          = ||Y_pre - Yhat_pre(theta)||^2 + 2 * sigma2 * df(theta)
```

Select `theta` (penalty `lam`, weighting matrix `V`, constraint set, factor
count, ...) by minimizing `IC`. The `-1` is the sum-to-one constraint; an
intercept adds `+1`; ridge / elastic-net give SVD-based forms; covariates give
`rank(X_tilde_A) - n_cov - 1`.

### Why this is attractive

- **No data splitting.** Cross-validation is data-hungry; in short panels it
  trains on a fraction of an already-short pre-period and its validation curve
  is flat and noisy. The IC uses all pre-treatment data and prices complexity
  analytically.
- **`df` is a robust overfitting diagnostic on its own.** Even where the
  *selection* is soft, reporting `df` flags overfitting cheaply (e.g.
  "df = 14 out of 36 pre-periods" is a red flag).
- **The paper's headline:** SCM's df equals the active-donor count minus one —
  the implicit donor selection is "free" in df terms. SCM does NOT overfit in
  the classic low-dimensional applications; it DOES overfit when donors are
  many relative to pre-periods, which is exactly when CV is weakest.

### Learnings from our own experiments (keep these)

We prototyped the IC against the Abadie–L'Hour PSCM
(`min ||Y-Xb||^2 + lam * sum_j b_j ||Y-X_j||^2 s.t. 1'b=1, b>=0`) with
`sigma2` from a held-out pre-period tail (fit first 2/3, residual variance on
last 1/3). Throwaway scripts, not committed.

1. **Low-dimensional (Basque: 16 donors, 20 pre-periods).** SCM has df 1–2; it
   is not overfitting, so the IC barely penalizes — it only trimmed a
   negligible 0.005 donor (df 2 → 1) at zero fit cost. ATT essentially
   unchanged (−0.69 → −0.70), synthetic = Cataluña + Madrid (the canonical
   Abadie answer). **Takeaway:** in low dimensions the IC ≈ no-op, as theory
   predicts.

2. **High-dimensional (China import: 87 donors, 36 pre-periods).** Unpenalized
   SCM used 15 donors (df = 14) — textbook overfit. The IC and CV **diverged
   sharply in complexity**: IC → 1 donor (1-NN, df 0); CV → 9 donors (df ≈ 9).
   But all selectors agreed ATT ≈ 0. **Takeaway:** the value here is the
   overfitting *diagnosis* and the IC/CV divergence, not a treatment-effect
   flip.

3. **The IC's "aggressiveness" is the `sigma2` estimate talking — not an
   inherent property.** Sensitivity check on the China case:
   - held-out `sigma2 = 0.046` (≈ the entire treated variance) → collapse to
     1 donor;
   - `RSS/(n - phat) sigma2 = 0.019` → a sensible interior model, 7 donors.

   ATT stable (+0.004 to +0.006) either way. The held-out estimate is
   *conservative* (the short-window fit forecasts poorly, inflating residuals);
   `RSS/(n-phat)` is *anti-conservative* (overfit in-sample residuals are too
   small). The truth is between. **Takeaway:** treat `sigma2` as a deliberate
   modeling choice; report the IC *curve* and the assumed `sigma2`; the 1-NN
   collapse is fragile, the "regularize vs not" verdict is robust.

4. **Structural quirk to remember.** For PSCM, `df = (1+lam)(|A|-1)`; once the
   active set hits a single donor, `df = 0` and the penalty vanishes, so 1-NN
   matching is a *privileged, penalty-free* endpoint. Do not over-read a
   collapse exactly to 1-NN.

### Design: how to integrate across the library

Two layers — only the first is universal:

- **Universal: the IC machinery.** Any estimator producing pre-period fitted
  values `Yhat_pre` and exposing *some* complexity knob can be selected by
  `IC = RSS_pre + 2*sigma2*df`. σ² estimation, grid search, and argmin are
  estimator-agnostic.
- **Per-estimator: the `df` formula.** Each optimization structure needs its
  own df. Use **closed forms where the paper provides them** (plain SCM, PSCM,
  constrained-ridge, elastic-net, covariate-SCM) and a **generic Monte-Carlo
  divergence fallback** otherwise: `df = (1/sigma2) * sum_i Cov(Y_i, Yhat_i)`,
  estimated by perturbing each pre-period `y_i` by a small Gaussian `eps`,
  re-fitting, and measuring `d Yhat_i / d Y_i` (Efron 2004 / SURE-by-sim).
  Costs K refits; works for anything refittable.

**Proposed home:** `mlsynth/utils/infocrit.py`, exposing
- closed-form `df` helpers (simplex, +intercept, +sum-to-one drop, ridge/EN SVD,
  covariates),
- a generic `mc_divergence_df(fit_closure, Y_pre, sigma2, ...)`,
- `sigma2` estimators (`held_out`, `rss_over_n_minus_p`),
- a thin `select_by_ic(...)` driver.

**Keep it a selector layer**, switched on per estimator (e.g. `tuning="ic"`),
never baked into an estimator's core.

**What an estimator must expose to opt in:** (1) pre-period fitted values as a
function of `Y_pre`; (2) a closed-form `df` *or* a re-fit hook; (3) a knob to
select over (λ, factor count, or a discrete variant menu).

### Per-estimator targets

| Estimator | Knob the IC would select | df source | Notes / priority |
|---|---|---|---|
| **TSSC** | one of {SIMPLEX, MSCa, MSCb, MSCc} | closed form (see below) | **Best first slice** — discrete menu, all df trivial, no λ-grid or σ² drama. Alternative/complement to current `step2` recommendation. |
| NSC | penalty `(a*, b*)` | MC divergence (nonneg + L1/L2, no clean closed form) | Replaces / augments its CV grid. |
| SparseSC | predictor-importance `lambda` (V step) | MC divergence | Replaces validation-MSE selection. |
| FMA / PDA | factor count / # forward-selected donors | closed form `|A|-1` per step | Matches the fsPDA modified-BIC stopping rule directly. |
| Plain SCM (FSCM etc.) | none | `|A|-1` | IC only *reports* df / overfitting; nothing to select. |

**TSSC variant df's** (the `-1` is sum-to-one, `+1` is intercept):

| Variant | Constraints | df |
|---|---|---|
| SIMPLEX | `w>=0, sum w = 1` | `|A| - 1` |
| MSCa | intercept + `w>=0, sum w = 1` | `|A|` |
| MSCb | `w>=0` only | `|A|` |
| MSCc | intercept + `w>=0` | `|A| + 1` |

### Out of scope / caveats

- **Bayesian members are exempt.** BVSS and BASC already quantify complexity
  via posterior inclusion probabilities; an IC on top is redundant.
- **Closed-form df validity is structural** — wrong structure ⇒ wrong penalty.
  Prefer the MC fallback when unsure.
- **Assumptions.** SURE rests on (roughly) Gaussian + homoskedastic errors; the
  paper shows robustness and provides a heteroskedasticity-robust IC (eq. 21)
  and a HAR variant for serial dependence — implement those before trusting the
  IC on strongly heteroskedastic / serially correlated panels.
- **σ² fragility** propagates to every consumer (see learning #3). Always
  surface the σ² choice and, ideally, the IC curve.

### Suggested implementation order

1. `infocrit.py` skeleton + closed-form df's + `sigma2` estimators.
2. Wire into **TSSC** variant selection (`tuning="ic"`); validate the
   recommendation against `step2` on a known dataset.
3. Add the generic MC-divergence df; wire into NSC / SparseSC.
4. FMA / PDA forward-selection stopping by IC.
5. Heteroskedasticity-robust + HAR IC variants.

---

## 2. C-Lasso (Su-Shi-Phillips) latent-group classifier SCM

**Status: Planned (speculative -- original methodology, not a replication).**

Reference repo to clone downstream (pinned, never vendored):
**https://github.com/zhan-gao/classo** (Su, Shi & Phillips 2016, *Identifying
Latent Structures in Panel Data*, Econometrica).

### CRITICAL terminology warning (do not conflate)

"Group lasso" names **two different methods** here:

* In Liao-Shi-Zheng's relaxed-SCM Monte Carlo (arXiv:2508.01793v2, §5), the
  "Group Lasso" is the **standard Yuan-Lin (2006)** group lasso used as a
  competitor baseline, **fed the true group membership as an oracle**. It does
  no classification. (We did *not* build it: ``rescm_relax_mc`` pins the L2
  relaxation vs SC, which is the head-to-head that matters.)
* **Su-Shi-Phillips's C-Lasso** is a *different* estimator -- a mixed
  additive-multiplicative ``sum_i prod_k ||beta_i - alpha_k||`` penalty that
  **classifies units' regression-slope vectors** into latent groups. **C-Lasso
  appears nowhere in the relaxed-SCM paper.**

### The idea, and why it is harder than it looks

Classify donors into latent groups with C-Lasso from their pre-treatment
behaviour, then fit SCM per group, vs ``RELAX_L2``. From reading both papers,
the obstacles (this would be original research, not a replication):

* C-Lasso classifies regression **slope vectors and needs covariates**
  ``x_it`` -- neither paper runs it on bare outcome series. Repurposing it for
  outcome-only SCM (e.g. classify factor loadings, factors-as-regressors) is not
  done or claimed in either paper.
* It needs ``T -> infinity`` and large groups (classification ~80-89% correct at
  ``T = 15``); SCM pre-periods are short and donor pools split into K groups
  leave few units per group.
* It fights the relaxed-SCM thesis: Liao-Shi-Zheng (Remark 2) show you **do not
  need to recover membership** -- L2-relaxation exploits the groups implicitly
  and *beats* the oracle-informed group lasso. "Classify-then-SCM-per-group" is
  the harder two-stage route they argue is unnecessary.
* C-Lasso itself is heavy: non-convex product-of-distances penalty, iterative
  joint estimation, init/local-optima sensitivity, an IC step for K, a
  post-Lasso refit.

**Verdict:** pursue only with appetite for original methodology; pin the
loadings-as-slopes specification against ``zhan-gao/classo`` first.

---

## 3. DSC -- Dynamic Synthetic Control (DTW speed-warping)

**Status: Parked (demonstrate-first DONE; blocked on a `dtw-python` vs R-`dtw`
`open.end` library difference -- not worth finishing until that is resolved).**

### Source

> Cao, J. & Chadefaux, T. (2025). "Dynamic Synthetic Controls: Accounting for
> Varying Speeds in Comparative Case Studies." *Political Analysis* 33, 18-31.

Reference package (cloned, read in full; pin downstream, never vendor):
**https://github.com/conflictlab/dsc** -- `R/{TFDTW,dsc,misc,synth}.R`.

### The idea in one line

Standard SCM assumes every unit reacts to common shocks at the **same speed**;
when donors adapt at different rates the target depends on donor *lags* (Eq. 1),
so omitting them biases the effect and inflates SEs. DSC uses **Dynamic Time
Warping** to learn each donor's pre-period speed warp vs. the treated unit,
**warps the donor outcome series** to align speeds, then runs **ordinary `Synth`
on the warped donors**. It removes *inherent* speed differences while preserving
treatment-induced ones (so the post-period effect survives).

### Naming (do not collide)

`DSC` is taken (**Distributional** SC) and `DSCAR` is taken (**dynamic SC for
AR** processes, Zheng-Chen). The Cao-Chadefaux method needs a third name -- e.g.
**`DTWSC`**. Three "dynamic-ish" SCs will share mindspace; disambiguate in docs.

### Demonstrate-first findings (the reason it is parked) -- KEEP THESE

The novelty is entirely the warping; the SC half is plain `Synth`, which mlsynth
already replicates (VanillaSC). The warping ports through **`dtw-python`** (Toni
Giorgino's official Python port of R `dtw`, identical step patterns
`symmetricP1`/`asymmetricP2`). Validated against the R package, cell by cell:

* **Bit-exact (synthetic AND real Basque):** `first.dtw`, `warp2weight`,
  `warpWITHweight`, the cutoff, and the pre-period speed weights `weight.a`. On
  Basque the **pre-period warped donors match R to ~1e-15** (confirmed on the
  worst donor: `weight.a` and the pre-warp `0, 0.2036, ..., 2.1799` identical).
* **NOT bit-reproducible: `second.dtw`** (the post-period double-sliding-window).
  The *entire* residual is one library-level disagreement: **`dtw-python`'s
  `open.end=TRUE` alignment covers the full query, while R `dtw` lets the
  open-ended alignment stop short.** This changes `RefTooShort`'s window-include
  decision, so the post-period `avg.weight` (and hence the warped post series)
  diverges. The 0-vs-1-index `+1` patch in `RefTooShort` fixes the overlap
  window but then *over-includes* the short boundary window everywhere (it wins
  the cost search with a spuriously cheap open-ended match) -- one fix trades for
  another, because the libraries genuinely terminate open ends differently.

**Reference numbers (R `dsc`, gold, full Abadie 14-predictor spec, Synth's
`basque`, Spain dropped -> 16 donors):**

| Method | pre-RMSE | post-ATT |
|---|---|---|
| Standard SC | 0.0888 | -0.585 |
| DSC (R) | **0.0728** | **-0.537** |

This **reproduces the paper**: DSC tightens the fit (-18% RMSE) and lands
ATT ~ SC (the Basque section's "similar ATT, better counterfactual"). Because
the pre-warp is bit-exact, a Python port already matches the **pre-RMSE 0.0728
exactly**; only the ATT rides on the un-portable `second.dtw` (the warp gap moved
an outcome-only ATT from R's -0.480 to Python's -0.573, so it is *not* negligible
-- bitwise matters here).

### Build path if resumed

1. **Resolve the `open.end` blocker** -- the only real obstacle. Hand-roll the
   DTW open-end recursion to replicate R's exact truncation/tie-breaking, or
   vendor R's step-pattern open-end logic. Uncertain payoff; reverse-engineering
   R `dtw`'s open-end termination point by point. *Everything else is done.*
2. Port `preprocessing` (S-G filter `scipy.signal.savgol_filter` 2nd-deriv --
   matches up to a constant that DTW re-normalizes away; the `auto.arima`
   edge-buffer is the one inherently-fuzzy piece, ~2 edge points, approximate it).
3. SC on warped donors -> **reuse mlsynth's `Synth` replication** (predictors).
4. Scaffold to the estimator contract; cross-validate the warped donors + ATT
   against the R package as a `benchmarks/` case.

### R reference install (figured out; reuse it)

`Synth` chain on Ubuntu + R 4.3 (CRAN blocked, apt+GitHub open): `apt` for
`libnlopt-dev r-cran-nloptr r-cran-pracma r-cran-quadprog ...`; compile from the
GitHub `cran` mirror in order `pracma, nloptr, optimx -> kernlab, rgenoud ->
Synth`; plus `signal, forecast` (forecast pulls `tseries/quantmod/urca`) and
`dtw`. `dtw-python` is a plain `pip install dtw-python`. Run `dsc()` with
`parallel=FALSE` to avoid `furrr`. (augsynth's install recipe in
`benchmarks/R/install_augsynth.sh` is the template.)

### Verdict

The method is worth having (genuine gap, top venue, recent, reproduces cleanly)
and is ~90% ported -- but **do not finish it until the `open.end` library
difference is solved**, because without it the post-period is not bit-faithful to
R and the ATT is only *close*. Revisit when there is appetite to hand-roll the
open-end DTW.

---

## 4. Design constraints for GEOLIFT (geography / coverage / size)

> **Status: implemented** (steps 1-4 below are done; whole GeoLift package at
> 100% coverage). Constraint primitives live in
> ``geolift_helpers/marketselect/helpers/constraints.py``; wired through
> ``config.py`` / ``orchestration.py`` / ``shaping.py`` / ``design.py`` /
> ``batch.py``; documented in ``docs/geolift.rst`` ("Design constraints"). The
> **only** remaining item is the optional step 5 -- cross-family consolidation
> into a shared ``utils/design_constraints/`` module. LEXSCM was **not**
> modified. Learnings below are kept for that future consolidation.

### Source / motivation

Meta's GeoLift exposes only **hard market lists** on the treated side
(`include_markets` / `exclude_markets`, mirrored today by `to_be_treated` /
`not_to_be_treated`). LEXSCM already carries a richer, *rule-based* constraint
vocabulary -- spillover/cluster non-interference, coverage quotas, treated-unit
size bands (see `docs/lexscm.rst`, "Spillover / interference exclusions" and
"Coverage quotas and size bands"). Bringing that vocabulary to GEOLIFT makes
mlsynth's market selection strictly **more expressive than the upstream R
package**, and is the thing that makes our GeoLift genuinely *ours* rather than a
faithful port.

### The idea in one line

Every design restriction reduces to **where the optimizer is allowed to look**:
LEXSCM's constraints never enter the inner weight solve -- they are *filters on
admissible treated supports* ("treatment criteria") and *pins in the control
program* ("control criteria"). GEOLIFT has the same two seams, so the same
constraint layer drops straight in.

### Why this is attractive

- **Two clean seams already exist.** Stage 1 (candidate nomination,
  `marketselect/helpers/candidates.py` + `similarity.py`) builds each region
  `S = {anchor} ∪ top-(k-1) correlated`, deduped, honoring the forced in/out
  sets -- exactly where *treatment* criteria belong. Stage 2 (the SCM, donor pool
  `N \ S`, in `fit.py` / `shaping.py`) is exactly where *control* criteria
  belong.
- **The geometry already justifies one of them.** A treated market far larger
  than the donors cannot be reproduced by a convex combination of them -- which
  is *precisely* what GeoLift's scaled-L2 imbalance `κ` measures post-hoc. So a
  `max_size` ceiling is an **a-priori** version of `κ → 1`. The method's own
  diagnostic motivates the constraint.
- **Purely subtractive.** With none of the new fields supplied, behaviour is
  byte-identical to today's GEOLIFT. Constraints only ever *shrink* the candidate
  set / donor pool.

### The mapping (treatment vs control criteria)

| Constraint | GEOLIFT seam | Effect |
|---|---|---|
| **`cluster_col` no-interference** -- the treated set must be an *independent set* of the cluster conflict graph | Stage 1: when extending an anchor with correlated neighbours, skip any sharing a cluster with an already-chosen member | "at most one treated geo per DMA / state"; still `O(N)` candidates, just a shrunk neighbour set |
| **`adjacency` / spillover exclusion** -- the donor pool drops `S ∪ A(S)` | Stage 2: pin spillover neighbours out of the donor matrix for that candidate | a treated geo's neighbours cannot contaminate its own synthetic control |
| **`stratum_col` + `min_per_stratum` / `max_per_stratum`** (coverage quotas) | Stage 1: admit only regions meeting the per-stratum counts | "test in every region"; "≤ 2 from the Northeast" |
| **`size_col` + `min_size` / `max_size`** (treated size bands) | Stage 1: restrict the *eligible nomination pool* (markets stay available as donors) | floor = power / operational minimum; ceiling = synthesizability (the `κ` argument above) |

The conflict graph is the same object in both rows 1-2: a symmetric
`A ∈ {0,1}^{N×N}` from `cluster_col` (`A_ij = 1` iff same cluster) **or** an
`adjacency` matrix thresholded at `spillover_threshold`, combined by logical OR
-- identical to LEXSCM's construction.

### Config surface (additive to `GeoLiftConfig`)

All optional; each defaults to `None` / off:

```
cluster_col, adjacency, spillover_threshold      # no-interference + donor exclusion
stratum_col, min_per_stratum, max_per_stratum    # coverage quotas
size_col, min_size, max_size                     # treated size bands
```

### Where it plugs in

- `candidates.py` -- after correlation nomination, filter nominees to satisfy
  independent-set + stratum quota + size band (all *treatment* criteria; act on
  the support, never the weight program).
- `fit.py` / `shaping.py` -- when assembling a candidate's donor matrix, drop
  `A(S)` (the lone *control* criterion).

### Failure modes (mirror LEXSCM exactly)

Raise `MlsynthConfigError`, never return a degenerate design, when: `k` exceeds
the number of clusters (no conflict-free `k`-tuple); `min_per_stratum` is imposed
over more strata than `k`; fewer than `k` markets lie inside the size band; or the
spillover exclusions empty a candidate's donor pool.

### LEXSCM is reference-only -- do not modify it

LEXSCM already implements all of these primitives (conflict graph, independent-set
filter, stratum-quota validator, size-band mask, aligned to its IndexSet). **Read
it for the logic; do not touch its code, tests, or behaviour.** It is
replication-pinned and there is no reason to disturb a working estimator. GEOLIFT
gets its **own self-contained** constraint primitives inside `geolift_helpers/`
(informed by LEXSCM, copying nothing structural). A little duplication is the
accepted price for zero risk to LEXSCM. Consolidating the two implementations into
a shared `utils/design_constraints/` algebra across the MAREX family (LEXSCM,
GEOLIFT, later SYNDES / MAREX / PANGEO) is a **deliberate later step**, taken only
once both are stable and with LEXSCM's replication tests as the guard.

### Test plan (test-first, per the contract)

Per constraint: a **smoke** test (constraint on, runs, design respects it), an
**invariant** test (independent-set holds / quota satisfied / sizes in band /
`A(S)` absent from donor weights), an **edge** test (constraint that admits
exactly one feasible region), a **failure** test (each infeasibility above raises
the translated error and the failure is *reported*), and a **no-op** test
(constraint columns present but trivially satisfied → identical shortlist to the
unconstrained run).

### Suggested implementation order

1. Build GEOLIFT's **own** constraint primitives in `geolift_helpers/`
   (conflict graph, independent-set filter, stratum-quota validator, size-band
   mask), test-first. **LEXSCM is not modified.**
2. Wire the **cluster/spillover pair** into GEOLIFT (highest-value geographic
   constraint), with docs + tests.
3. Add **coverage quotas**, then **size bands**.
4. Docs: extend `docs/geolift.rst` with a "Design constraints" section paralleling
   the LEXSCM treatment, each example a full MRE.
5. *(Later, optional)* consolidate GEOLIFT's and LEXSCM's primitives into a shared
   `utils/design_constraints/` module once both are stable.

### Out of scope / caveats

GEOLIFT's nomination is a cheap correlation heuristic, not LEXSCM's QP search, so
constraints are applied as *post-nomination filters* -- a heavily-constrained
problem could shrink the shortlist sharply (acceptable, and surfaced via the
diagnostics). ROI / cost interactions with `cpic` + `budget` already exist and are
orthogonal to these support constraints.

---

## Done

*(empty -- move completed items here, preserving their Learnings subsection.)*
