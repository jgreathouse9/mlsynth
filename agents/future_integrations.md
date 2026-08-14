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

**Status: UNBLOCKED -- build. The `open.end` blocker was misdiagnosed and is
resolved; see "Blocker resolved" below. Everything else was already ported.**

### Source

> Cao, J. & Chadefaux, T. (2025). "Dynamic Synthetic Controls: Accounting for
> Varying Speeds in Comparative Case Studies." *Political Analysis* 33, 18-31.

Reference package (cloned, read in full; pin downstream, never vendor):
**https://github.com/conflictlab/dsc** -- `R/{TFDTW,dsc,misc,synth}.R`. MIT
(`LICENSE`; the README's "GPL-3" line contradicts `DESCRIPTION` and the actual
file -- treat MIT as governing, and clean-room the port either way).

There is now also a JOSS software paper in the repo (`paper.md`, `paper.pdf`,
`.github/workflows/draft-pdf.yml`, Sep 2025). It restates the method; it adds no
new algorithm. `git log` confirms `R/TFDTW.R`, `R/misc.R` and `R/synth.R` are
untouched since 2023-09-14, and the only 2025 change to `R/dsc.R` is a ggplot
block appended to `dsc()`. The reference implementation this entry was written
against is byte-identical to the current `main`.

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
* **NOT bit-reproducible at the time: `second.dtw`** (the post-period
  double-sliding-window). The symptom was real -- `RefTooShort`'s window-include
  decision differed, so the post-period `avg.weight` (and hence the warped post
  series) diverged, and the 0-vs-1-index `+1` patch fixed the overlap window
  while over-including the short boundary window. The *diagnosis* was wrong: it
  was attributed to `open.end` termination differing between the libraries. It
  does not. See below.

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

### Blocker resolved -- it was `warp()`, not `open.end` (KEEP THIS)

Re-tested head-to-head, R `dtw` 1.23-3 vs `dtw-python` 1.7.5, both step patterns
used by the package (`symmetricP2` for `first.dtw`, `asymmetricP2` for
`second.dtw`), 268 usable random (query, reference) pairs spanning lengths 2-25
(white noise, random walks, and noisy sinusoids), all with `open.end=TRUE`:

| quantity | agreement |
|---|---|
| `distance` | 268/268 exact (1e-6) |
| full `index1`/`index2` warping paths | 268/268 identical |
| `RefTooShort` boolean, after the fix below | 268/268 identical |

**The open ends terminate identically.** `dtw-python` does *not* force coverage
of the full query; it stops short exactly where R does, with the same `jmin`.

The entire divergence is in **`warp()`'s handling of the many-to-one map**. R's
`dtw::warp` averages the tied indices; `dtw-python`'s does not. Worked case
(`length(Q) = length(R) = 8`, identical alignment in both):

```
index1 = 1 2 3 3 4 5 6 7 8      index2 = 1 2 3 4 5 6 7 8 8
R   warp(index.reference=FALSE) -> 1 2 3 3 4 5 6 7.5     <- mean(7, 8)
py  warp(index_reference=False) -> 1 2 3 3 4 5 6 7       (after +1)
```

`RefTooShort` then tests `round(max(wq)) < length(query)`: R gets
`round(7.5) = 8`, not `< 8`, so FALSE (window kept); Python gets `7 < 8`, TRUE
(window dropped). That single tie-break is the whole "short boundary window"
over-inclusion described above. The fix is one function:

```python
def warp_R(a, index_reference=False):
    src, dst = (a.index1, a.index2) if index_reference else (a.index2, a.index1)
    return np.array([dst[src == j].mean() for j in range(int(src.max()) + 1)])
```

With `warp_R` substituted for `dtw.warp`, `RefTooShort` matches R on 120/120
`second.dtw`-shaped pairs and 268/268 in the wide sweep, both step patterns. Use
numpy's `round` (banker's rounding, same as R's) for the comparison.

### Build path

1. ~~Resolve the `open.end` blocker~~ **Done -- see above.** Reimplement `warp`
   with R's tie-averaging semantics; do not call `dtw.warp` directly. Both call
   sites matter: `RefTooShort` and `first.dtw`'s `wr`/`cutoff`.
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

### Gap vs overlap (re-checked)

`grep -rilE "dynamic time warp|dtw|warp" mlsynth --include=*.py` hits only
`CMBSTS`, whose `control_selection='dtw'` uses DTW *distance* to screen the donor
pool (optional `fastdtw`). That is DTW as a similarity metric for donor
selection; nothing in the library *warps a donor's outcome path*. Genuine gap.
Closest siblings by intent: `MASC` (reshapes the donor set rather than the donor
series), `CMBSTS` (DTW screening), `DSCAR` (a different "dynamic"). Naming
guidance above still stands -- `DSC` and `DSCAR` are taken, use `DTWSC`.

### Verdict

**Build.** Genuine gap, top venue (*Political Analysis*), MIT reference package,
now also a JOSS submission, and it reproduces cleanly. The one obstacle that
parked it was a misdiagnosis: the libraries agree bit-for-bit on open-ended
alignment, and the residual is a five-line `warp` tie-averaging fix, verified
268/268. The pre-warp was already bit-exact, so the remaining work is mechanical
-- swap in `warp_R`, re-run `second.dtw`, and confirm the Basque ATT lands on R's
outcome-only -0.480 (the earlier Python run gave -0.573; that gap *is* the warp
bug and is the acceptance test). Then `/replicate` against the R package for the
full 14-predictor spec (pre-RMSE 0.0728, ATT -0.537) and `/new-estimator` as
`DTWSC`.

### Shipped, and one thing worth carrying forward

`DTWSC` landed. The warping engine is bit-exact against the R package (see
`docs/replications/dtwsc.rst`); the sole remaining difference is the
Savitzky-Golay edge treatment, where the reference pads with an `auto.arima`
forecast and mlsynth edge-pads.

Carry forward the reason that took three wrong diagnoses to find, because it
generalizes past this estimator. `second.dtw` filters speeds with a type-7
`Q3 + 3*IQR` fence, and the speeds are small-denominator rationals -- so on a
column like `(1, 1, 7/6, 1)` the fence evaluates to exactly `7/6` and sits on
the value it is judging. One bit decides whether the cell is kept, and that
moves the column mean by `1/24`. The bit came from smoothing: R's
`stats::filter` accumulates `x[k] * (1/w)` where a plain window mean computes
`(sum x[k]) / w`, and the two disagree in the last place on 7/6.

Two rules follow for any future port. Match the reference's summation order in
any step feeding a discrete decision, rather than assuming an algebraically
equal formula is numerically equal. And when a port is close but not exact,
substitute the reference's own intermediate outputs stage by stage to localize
the divergence -- inferring from the end result gave the wrong answer three
times (blamed hyperparameter selection, then the SC backend, then the ARIMA
buffer; each was refuted by measurement).

---

## 4. Rolling-transformation DiD (ROLLDID) -- scope boundary with `diff-diff`

> **Status: ROLLDID shipped; broader DiD surface deliberately OUT OF SCOPE
> (deferred to the sibling package `diff-diff`).** The small-:math:`N` exact
> rolling-transformation DiD landed as the ``ROLLDID`` estimator (Lee &
> Wooldridge 2026), validated Path A on California Prop 99 (Table 3) and the
> castle laws (§7.2), 100% covered, durable case ``rolldid_lw``. This section
> records the **decision not to grow a general DiD family** inside mlsynth and
> the learnings behind it.

### Source

> Lee, S. J., & Wooldridge, J. M. (2026). "Simple Approaches to Inference with
> Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes."

Sibling package (the home for the broader DiD ecosystem):
**https://github.com/igerber/diff-diff** (Isaac Gerber, MIT) -- Callaway-Sant'Anna,
Sun-Abraham, Borusyak et al., Gardner, **Wooldridge ETWFE**, stacked DiD,
de Chaisemartin-D'Haultfoeuille, honest-DiD sensitivity, and the large-:math:`N`
staggered estimators, validated against R ``did`` / ``synthdid`` / ``fixest``.

### Why ROLLDID belongs in an SCM package (keep this)

The boundary is **the regime, not "DiD vs SCM."** ROLLDID's value is the
**small-:math:`N` exact** case -- one treated region against a handful of donors,
exact :math:`t_{N-2}` inference (valid at :math:`N_1 = 1`) -- which is exactly the
synthetic-control use case. It collapses the panel to one cross-sectional
observation per unit and reads the ATT off a cross-sectional regression, so it
estimates **no donor weights at all**. That is precisely why it stays stable
where the SC family gets fragile.

**Demonstrate-first learning (castle, keep this).** On the short, donor-starved
castle panel (T=11; cohorts of size {2005:1, 2006:13, 2007:4, 2008:2, 2009:1}),
``SequentialSDID``'s per-cohort weight optimisation blew up -- cohort-2009
:math:`\tau = -17.5` (:math:`\omega` going negative, :math:`\lambda \sim \pm200`),
cohort-2008 :math:`\tau = +64.8` (:math:`\lambda \sim \pm400`) -- poisoning the
pooled estimate to **5.47**, while ROLLDID returned **0.092** (demean) / **0.067**
(detrend), next to the paper's own SDID of 0.099. This is the concrete
illustration of *why* the weight-free estimator earns its place: it is the more
trustworthy tool exactly when units/periods are scarce.

### What is deliberately NOT being built (closed out)

These rolling-DiD follow-ups were considered and **declined as out-of-scope** --
they are the *large-:math:`N`, full-covariate* DiD surface, which is precisely
what `diff-diff` already does well (it ships Wooldridge ETWFE, the large-:math:`N`
sibling of ROLLDID's method):

* **IPW / IPWRA doubly-robust estimation with covariates** (propensity scores,
  weighted least squares). `diff-diff` covers covariate-conditioned DiD.
* **Seasonal transforms** (``demeanq`` / ``detrendq`` / ``demeanm`` / ``detrendm``).
* **Large-:math:`N` influence-function multiplier bootstrap** (the Lee-Wooldridge
  2026a appendix). Needs that companion paper and only pays off at large
  :math:`N` -- where `diff-diff`'s machinery is the better home.

**Principle.** mlsynth is an SCM package whose mission is *access*, not method
turf wars: if a simple DiD is the better tool for a problem, ship the minimal
version that complements synthetic control and **point users to the cousin
package for the rest**. The ``ROLLDID`` docs page carries a "See also: diff-diff"
note to that effect.

---

## 5. MIQP warm-start for the design estimators (MAREX / SYNDES / SPCD)

### Source

Internal -- surfaced while building the MAREX Walmart benchmark
(`benchmarks/cases/marex_walmart.py`). The exact MIQP design is intractable on
the full 45-store panel via the free SCIP backend, so the benchmark subsets to 10
stores. The authors' R (`jinglongzhao2/SCDesign`) uses Gurobi.

### The idea in one line

Solve the continuous relaxation first, round it to a feasible integer design, and
hand that to the exact MIQP solver as its initial **incumbent** (MIP warm start),
so branch-and-bound starts with a real upper bound and prunes aggressively.

### Why this is attractive

* mlsynth already produces the warm-start candidate for free:
  `marex_helpers.optimization.solve_design_relaxed` -> `post_hoc_discretize`
  returns a feasible K-unit design. It is currently a *standalone heuristic*; it
  should instead *seed* the exact solve.
* Could make the **full-panel** exact design tractable on the free SCIP backend
  (no Gurobi, no subsetting) -- raising the MAREX/SYNDES benchmarks from a subset
  to the paper's full instance.
* Same trick applies to SYNDES (per-unit / two-way MIQP) and any future MIQP
  design.

### Why it is parked (the interface catch)

* MAREX/SYNDES solve via `cp.Problem.solve(solver=cp.SCIP)`. **cvxpy does not
  expose MIP warm-starts** -- its `warm_start` flag is continuous-only; an integer
  incumbent is not passed through. So this needs a **PySCIPOpt-direct** path
  (build the model, `model.addSol(rounded_design)` before `optimize()`),
  bypassing cvxpy for the MIQP -- a new solver backend, not a flag.
* The relaxed optimum is **degenerate** for the `standard` design (the rounded
  design is weak -- see the 62%-RMSE 45-store result), so it is a *weak*
  incumbent: it still bounds the search but prunes less. A smarter warm start
  (greedy / local search on the relaxation) would help more.
* Warm-starting speeds finding/proving the optimum; it does not change
  NP-hardness.

### Cheaper interim alternative

A **`time_limit` / `gap_limit` passthrough** (SYNDES already exposes this via
`scip_params`) lets the full-panel MIQP return a near-optimal design in bounded
time with ~10 lines and no new backend. Tradeoff: gap-limited solutions can drift
across solver versions, so they are less suitable for a *deterministically
pinned* benchmark (hence the 10-store exact subset for `marex_walmart`).

### Suggested order if resumed

1. Add the `time_limit`/`gap_limit` passthrough to `MAREXConfig` (mirror SYNDES);
   cheap, immediately useful for applied full-panel runs.
2. Add a PySCIPOpt-direct MIQP path that seeds `post_hoc_discretize`'s design as
   the incumbent; benchmark the speedup and (optionally) lift `marex_walmart` /
   `syndes_bls` to the full panel.

### Verdict

Worthwhile **performance** work (not correctness): the exact MIQP is already
faithful; this only makes it scale. Keep the benchmarks on the deterministic
exact subset until the warm-start path lands.

---

## 6. rfPDA -- Random-Forest control selection for the Panel Data Approach

**Status: Parked (built test-first; demonstrate-first DONE; the defining
random-forest selection step does NOT port faithfully via scikit-learn -- it is
implementation- and seed-unstable. OLS + inference ports are exact.)**

The faithful port (TDD, 100% covered) is preserved, unmerged, on branch
**``claude/pda-rf``** -- not on ``main``. Take it up from there.

### Source

> Liu, G., Long, W., & Luo, X. (2025). "A Random Forests-Based Panel Data
> Approach for Program Evaluation." *Journal of Applied Econometrics*, 40(5),
> 591-607.

Reference package (R; read in full): the JAE replication archive
(``RF.R`` = the method, ``test.R`` = the MA(1)-West HAC inference,
``Application_2`` = the China anti-corruption luxury-watch example with
``china_import.rda`` = ``basedata/china_watches_long.csv``).

### The idea in one line

A fourth PDA control-selection rule (alongside ``l2`` / ``lasso`` / ``fs``):
rank the candidate controls by random-forest permutation importance on a 70/30
split of the pre-period, forward-select the prefix of the ranked list that
minimizes held-out forest MSE, then fit ordinary OLS PDA on the selected
controls and test the ATE with the West (1997) MA(1) long-run variance.

### What ported cleanly (validated cell-for-cell)

Running the reference ``RF.R`` natively in R reproduces the paper exactly: watch
ATT = **-0.0266**, R^2 = 0.777, **7** selected commodities, p = **0.0634**.

- **OLS PDA counterfactual + MA(1)-West HAC inference are exact.** Fed the
  reference's exact 7 selected controls, mlsynth's ``_ols_pda`` +
  ``rfpda_ate_inference`` returns ATT = -0.0266, p = 0.0636 -- bit-for-bit the
  reference. The inference HAC formula is pinned against ``test.R`` in a unit
  test.
- **The importance ranking is reasonable.** 5 of the reference's 7 controls land
  in mlsynth's top 8 (ranks 2,4,5,7,8; the other two at 20,26).

### Why it is parked -- the RF selection does not reproduce (KEEP THIS)

Through the public API (``PDA(method="rf")``) the selection is **seed-unstable
and does not match R**:

| seed | ATT | p | # selected |
|---|---|---|---|
| 0 | -7.06% | 0.125 | 41 |
| 42 | -9.04% | 0.214 | 34 |
| 372236 (the reference's own seed) | -5.74% | 0.240 | 21 |
| 7 | -3.57% | 0.020 | 11 |
| 2024 | -3.18% | 0.049 | 6 |
| **reference (R)** | **-2.66%** | **0.063** | **7** |
| ``fs`` (cross-check) | -3.09% | 0.012 | 3 |

The estimate swings from -3.2% to -9% (p 0.02-0.24) with the seed alone.

**Root cause -- two compounding issues:**

1. **scikit-learn's ``RandomForestRegressor`` != R's ``randomForest``.** Different
   bootstrap RNG, CART split selection, and importance routine (R's ``type=1``
   is OOB ``%IncMSE``; sklearn's ``permutation_importance`` permutes a supplied
   set). On the tiny pre-period (T0 = 35 -> ~24 train / 11 test) these diverge
   materially -- they are simply different forests.
2. **The forward-sweep criterion is raw ``argmin`` of an 11-point held-out MSE,
   which is intrinsically unstable.** Diagnosed on the watch panel: the MSE curve
   drops to ~0.027 by 8 controls then stays *flat and noisy* (MSE 0.0313 at i=7
   vs a global min 0.0264 that wanders to i=15-41 by seed). With so few test
   points the ``argmin`` over-selects and jumps around. This fragility is
   **inherent to the method on short panels**, not specific to sklearn: R's own
   result is the value its single pinned seed (372236) happens to land on.

So the method's *defining* step is the part that does not port. The OLS +
inference half (the part that is implementation-independent) is exact.

### The main issue, distilled

rfPDA's contribution is the RF selection, and that selection is governed by an
``argmin`` over a noisy held-out forest MSE on a short pre-period. It is doubly
non-deterministic across (a) RF implementation and (b) seed, so there is no
value-for-value Python reproduction of the paper's headline, and -- worse -- no
*stable* point estimate to ship. We declined to merge it or pin a benchmark.

### Build path if resumed (and: "would a custom RF be the solution?")

Yes -- a **custom, pure-NumPy random forest that mirrors R's ``randomForest``**
(same bootstrap draw + RNG, CART splitting, ``mtry = floor(p/3)``,
``nodesize = 5``, OOB ``%IncMSE`` importance) is the route to a *faithful*
replication: it would let mlsynth bit-reproduce the reference for a given seed.
But weigh two things first:

- It is a large undertaking -- effectively re-implementing Breiman's algorithm's
  numerics in Python (the "torture yourself" route). sklearn cannot be coerced
  into bit-equality with R here.
- **It would reproduce the paper's chosen-seed answer, not cure the fragility.**
  Even R's rfPDA is seed-dependent on T0 = 35 (the flat/noisy MSE curve is the
  method's, not the implementation's). A faithful custom RF makes
  *replication* exact; it does **not** make rfPDA a *stable* estimator on short
  panels. If the goal is a trustworthy tool rather than a literal port, the more
  productive direction is a **stabilized selection rule** (seed-ensembled MSE
  curve + a 1-SE "smallest prefix within one SE of the min" rule, which on the
  watch data should land near the ~7-control / ~-3% region) -- but that is a
  deliberate *deviation* from the reference, i.e. new methodology, not a port.

What is already done and reusable on ``claude/pda-rf``: the ``pda_helpers/rf/``
subpackage (``rank_controls_rf``, ``forward_select_rf``, ``rfpda_ate_inference``
with the exact MA(1)-West HAC), ``method="rf"`` wired through ``PDAConfig`` /
orchestration (with ``n_jobs`` / ``sweep_n_estimators`` / ``max_support`` /
``patience`` efficiency knobs), and the full TDD suite. Only the RF *selection*
needs replacing (custom RF) or *stabilizing* (ensemble + 1-SE) to resume.

### Verdict

Genuine gap, in-lane, top venue -- but the contribution does not port faithfully
through scikit-learn and is not seed-stable on the short panels PDA targets.
Parked: keep ``claude/pda-rf`` for a future custom-RF or stabilized-selection
effort; do not merge or benchmark the sklearn version.

---

## 7. AFS -- Adaptive Forward Stepwise (shrinkage-bridged FS <-> LASSO selection)

**Status: Planned (paper read in full; no code). Natural successor to ``fs`` in
the PDA family, parallel to the parked ``rfPDA`` (#7).**

### Source

> Zhang, I., & Tibshirani, R. (2026). "Adaptive Forward Stepwise: A Method for
> High Sparsity Regression." *Journal of Machine Learning Research*, 27, 1-24.
> (A Python + R package is "forthcoming" per the paper's discussion.)

Sits on the same FS <-> best-subset <-> LASSO axis as the work already in the
library; it cites, and is motivated by, two papers we already lean on:

> Hastie, Tibshirani & Tibshirani (2020) ``[HTT2020]`` -- FS underperforms the
> LASSO in low SNR (the gap AFS exists to close).
> Bertsimas, King & Mazumder (2016) ``[BKM2016]`` -- the best-subset reference
> behind the HCW ``fw`` / ``scip`` engines.

Parent method for the PDA framing:

> Shi & Huang (2023) ``[fsPDA]`` -- forward-selected PDA, mlsynth's ``fs``. AFS
> is forward selection with shrinkage, so an AFS-PDA generalizes ``fs`` directly.

### The idea in one line

Forward stepwise jumps straight to the active-set OLS fit (no shrinkage -> high
variance, weak in low SNR); the LASSO shrinks but over-selects in medium/high
SNR. AFS adds a single step-size ``rho in (0,1]`` and updates by a *shrunken
convex combination* toward the active-set OLS instead of jumping:

```
j*_m   = argmax_j | x_j' (y - X beta_{m-1}) |     # same criterion as FS / LAR
nu_m   = OLS(y ~ X_{A_m})                          # refit on the active set
beta_m = (1 - rho) * beta_{m-1} + rho * nu_m       # fractional, shrunken step
```

``rho = 1`` *is* Forward Stepwise; ``rho -> 0`` traces LAR (hence ~the LASSO,
their Theorem 2). Intermediate ``rho`` is a genuine, CV-tuned middle ground. A
variable can be re-selected across steps, so its coefficient builds up
geometrically toward OLS while staying shrunk -- under orthogonal design this is
an approximate soft-thresholding estimator (Theorem 3) and a cousin of
L2Boosting. Stop at ``m = M`` or when ``||beta||_1`` reaches the LASSO path's max
ell-1 norm; tune ``rho`` and the step count by CV (df is intractable here, so no
Cp/AIC/BIC).

### Why this is attractive (and where it fits)

- **Closes the exact gap our own results just exposed.** On the China luxury-
  watch panel (87 controls, T0=35, low SNR, n<p) ``fs`` gave a significant
  -3.09%, while the uncertified HCW best-subset gave an insignificant -1.44%
  (gap=inf -- it could not certify). That low-SNR, n<<p regime is precisely where
  HTT2020 say unshrunk FS and best-subset struggle and where AFS's shrinkage is
  designed to help. AFS is the principled "shrunk forward path" for this case.
- **Drops into the PDA control-selection slot.** Same shape as ``rfPDA`` (#7): a
  new ``method="afs"`` selector that produces a support + counterfactual, then
  reuses the existing ``fsPDA`` post-selection HAC t-test (``fs``'s inference is
  valid for any pre-period-only selection rule under the sample-splitting
  argument). The AFS coefficients are shrunk, but the ATE inference rides on the
  post-period gap, so the existing HAC machinery applies.
- **Sparser than LASSO, with shrinkage FS lacks** -- the paper's headline: lowest
  or near-lowest MSE across SNR x correlation x dimension while staying much
  sparser than the LASSO (which carries ~0.15 FPR). Computationally cheap (rank-
  one inverse updates on the active set), comparable to our other PDA solves.
- **GLM-ready (their Algorithm 2):** swap the OLS refit + squared-error residual
  for a GLM fit + score residual. Not needed for PDA (continuous outcomes) but
  noted for any future classification use.

### Two integration framings

1. **Standalone sparse-regression estimator / utility.** A general
   ``utils/afs.py`` (or an estimator) implementing Algorithm 1 with CV over
   ``(rho, steps)`` -- usable anywhere mlsynth needs a sparse linear fit. Cleanest
   if we want AFS reusable beyond PDA.
2. **A PDA variant ``method="afs"`` (recommended first slice).** Forward path
   with ``rho``-shrinkage as the control selector; OLS-PDA counterfactual on the
   selected support; ``fsPDA`` HAC inference. Mirrors the ``fs`` plumbing exactly
   (``pda_helpers/afs/{estimation,inference}.py``, wired through ``PDAConfig`` /
   orchestration), with ``afs_rho`` (or CV) and ``afs_max_steps`` knobs. Lets us
   benchmark AFS directly against ``fs`` / ``lasso`` / ``l2`` / ``hcw`` on Hong
   Kong and the watch panel.

### Caveats / open questions (decide before building)

- **Tuning cost.** AFS needs CV over ``rho`` *and* step count; on short PDA
  pre-periods CV is data-hungry and noisy (the same fragility that sank the
  ``rfPDA`` ``argmin``-over-held-out-MSE selection -- see #7). Consider the
  Pouliot et al. SURE information criterion (#1) as a CV-free selector instead,
  or a fixed/small ``rho`` grid with a 1-SE rule. This interaction with #1 and #7
  is the main design question.
- **Weak spot is our regime.** The paper reports AFS is beaten by the relaxed
  LASSO in the high-dimensional, high-correlation ``n << p`` setting -- which
  describes the watch panel (87 correlated commodity-import series). So benchmark
  AFS *and* RLASSO there; AFS may not dominate exactly where we'd most want it.
- **Inference validity.** ``fsPDA``'s post-selection t-test is justified for
  forward selection's support; AFS's shrunk coefficients change the *fit* but the
  selection is still pre-period-only, so the sample-splitting argument should
  carry -- but confirm the HAC test's size on a quick Monte Carlo before trusting
  it (AFS is not in the fsPDA theory as written).
- **No reference implementation yet.** The paper's package is "forthcoming", so
  there is nothing to pin against value-for-value today; validate against the
  paper's simulation *geometry* (Figs 1-3, 6-7: AFS path between FS and LASSO,
  lower FPR than LASSO) rather than a cell-for-cell port. Revisit once the
  authors' package ships.

### Verdict

Genuinely in-lane and well-motivated -- it is the shrinkage upgrade to ``fs``
that HTT2020 (and our own watch-panel result) say is missing, and it slots into
the existing PDA selector plumbing. Build it test-first as ``method="afs"`` when
there is appetite, leading with the ``fs`` parallel; settle the tuning question
(CV vs the #1 SURE-IC) first, and benchmark against RLASSO in the ``n << p``
regime where the paper itself cedes ground.

---

## 8. BASC -- Bayesian Donor Set Selection in Synthetic Controls

**Status: Parked on cost (demonstrate-first DONE; faithful NumPy port validated
against the authors' R within its own MCMC chain envelope). Capability overlaps
the existing ``BVSS``; the sampler needs a very large number of MCMC iterations
to converge. Revisit only if it shows a clear performance edge at reasonable
sample sizes.**

### Source

> Lee, S., Lim, J., Kim, J., & Wang, X. (2025). "Bayesian Donor Set Selection in
> Synthetic Controls." Manuscript CSDA-D-25-01439, *Computational Statistics &
> Data Analysis* (under review at time of writing; JAG was a referee).

Reference repo (R; read in full, run): **https://github.com/sll-lee/paper-BASC**
-- ``BASC_realdata.R`` (a single-script hand-written Gibbs/MH sampler; deps
``tmvtnorm`` / ``MCMCpack`` / ``fields`` / ``coda``). Data: ``repgermany.dta``
(already in ``basedata/``). Builds on, and is pitched as an improvement over:

> Martinez & Vives-i-Bastida (2024), Bayesian SCM (``bsynth``) -- the
> Dirichlet-prior Bayesian SC. NOT in mlsynth.

### The idea in one line

A Bayesian hierarchical SCM that does donor-set selection JOINTLY with weight
estimation: replace B-MV's Dirichlet weight prior with a Gamma x Bernoulli
construction ``w_j = gamma_j u_j / sum_k gamma_k u_k`` (Bernoulli
``gamma_j ~ Ber(eta)`` selects active donors, Gamma ``u_j`` sets relative
influence), add a Gaussian-process temporal term ``f_t`` (squared-exponential
kernel) and a basis-expanded post-treatment effect ``sum_m alpha_m D_mt``, fit by
MCMC. Model: ``y_1t = sum_j w_j y_jt + f_t + sum_m alpha_m D_mt * 1(t>T0) + e_t``.

### Closest existing estimator -- capability overlap with BVSS

``BVSS`` ("Bayesian SC with a Soft Simplex", BVS-SS) already does Bayesian donor
selection via Bernoulli inclusion indicators (posterior inclusion probabilities,
spike-and-slab) -- the same CAPABILITY. BASC differs by the prior construction
(Gamma x Bernoulli vs soft-simplex spike-and-slab) plus the GP temporal term and
flexible effect, which BVSS lacks. So not a duplicate, but the marginal value
over BVSS is narrow (see verdict).

### Demonstrate-first findings (KEEP THESE)

Ported the full Gibbs/MH sampler to NumPy/SciPy (oracle: ``BASC_realdata.R``):
conjugate draws for ``f`` (GP), ``sigma^2``, ``tau^2``, ``eta`` (Beta), ``alpha``
(truncated normal <= 0); RW-MH for the length-scale ``kappa`` (log scale);
Bernoulli-Gibbs for ``gamma_j``; log-``u`` RW-MH (x5/iter) for the Gamma weights.
Hyperparameters from the R: ``a_sig=10, b_sig=5000, a_tau=3, b_tau=2e4, a_l=3,
b_l=1000, a_u=2.5``; ``sig.a ~ IG(3, 2e6)``. The three R deps (``rinvgamma``,
``rtmvnorm``, ``rdist``) reduce to one-liners (q=1 => the truncated MVN is a
univariate TN). Cholesky updates need jitter (``safe_chol``) -- the GP covariance
goes near-singular when ``kappa`` is large; the R jitters too.

1. **The port is FAITHFUL -- it lands inside the authors' own MCMC chain
   envelope.** On West Germany at N=4000 the R's three chains (seeds 100/200/300)
   themselves disagree: ATT -238 / -599 / -365, 3rd donor flipping
   Italy<->Portugal. The NumPy port (ATT -310; Switzerland .48, Japan .36,
   Portugal .11) sits squarely inside that envelope -- identical top-2 donors,
   ATT in the middle of the R's range, 3rd donor matching R's chain 3.
   Statistically indistinguishable from a fourth R chain => port validated within
   MC error (the ``CMBSTS`` standard).

2. **West Germany comparison (raw per-capita GDP, the repo's actual scale):**

   | Method | pre-RMSE | ATT |
   |---|---|---|
   | VanillaSC (classic) | 60.8 | -1297 |
   | Bayani RPCA-SC (``CLUSTERSC method="rpca"``) | 88.6 | -1501 |
   | BVSS | 385.0 | -313 |
   | BASC (R / port) | ~172 | ~-600 |

   The two Bayesian methods fit the pre-period looser (regularization) and give
   attenuated effects; BASC's GP absorbs post-period trend, roughly halving the
   canonical reunification cost.

3. **Code/paper mismatch on normalization (the referee's Major #2).** The paper
   Section 5 text says each series is centered on its own pre-mean and scaled by
   *West Germany's* pre-SD; the RELEASED ``BASC_realdata.R`` does NOT -- it uses
   raw scale (``y <- raw.y; x <- t(raw.x); # Final manuscript version: original
   per-capita GDP scale``). The raw-scale ATT (~-599) matches the referee's
   reported "~-635". For an mlsynth build, raw ``dataprep`` ingestion is the
   natural choice and sidesteps the disputed normalization.

4. **Attribution (the referee's Major #1).** The RPCA-SC comparator (functional
   PCA + Robust PCA + NNLS) is **Bayani (2021, arXiv:2108.12542; 2022
   dissertation)**, NOT Greathouse 2023 (the paper miscredits it). mlsynth
   already cites Bayani correctly (``clustersc_rpca_germany``). Do NOT call the
   CLUSTERSC-rpca method "fPCA-SYNTH (Greathouse)".

### Why it is parked -- the cost (KEEP THIS)

The authors run **N=500000 + nburn=500000 (one million iterations)**. At N=4000
(3 chains, ~20 min) the chains do NOT agree (ATT -238 to -599), i.e. it mixes
slowly; the convergence point is somewhere between 4k and 500k and was not
bisected, but even a fraction of 1M is hours per fit -- the wrong cost class for
mlsynth, whose value is interactive estimator-swapping. (Note: the chain spread
at N=4000 is partly a reduced-N artifact -- the full run almost certainly mixes
tighter, as the referee credited "proper convergence diagnostics" -- so the
honest framing is "slow-mixing / expensive," not "weakly identified".) Combined
with: capability overlap with ``BVSS`` (Bayesian donor selection in seconds), and
a narrow benefit (per the referee, BASC's gains concentrate in sparse /
heterogeneous donor pools; comparable to B-MV in full-donor settings)
=> narrow benefit x high cost = poor ROI.

### Build path if resumed

The NumPy sampler port is validated and ~complete (faithful to the R, reproduces
within the chain envelope). To resume: (1) bisect a REASONABLE iteration budget
between 4k and 500k -- if it needs >> ~50k to stabilize, do not build; (2) wrap
as a top-level ``BASC`` estimator (``utils/basc_helpers/{sampler,gp,selection,
pipeline,structures}.py``), riding ``dataprep`` + ``BaseEstimatorResults``,
returning counterfactual + credible bands + posterior inclusion probabilities
(mirror ``BVSS``); (3) MUST ship multi-chain R-hat diagnostics and report the ATT
as an interval (the authors' own chains span -238 to -599); (4) durable
validation = West Germany cross-val vs ``BASC_realdata.R`` posterior summaries
within MC error (raw scale), a captured-reference case under ``benchmarks/``.
Catalogue home: next to ``BVSS`` / ``CMBSTS``.

### The bar to clear (owner's call, recorded verbatim)

> "Maybe it'll be added someday, but I'd need to see much better performance in
> reasonable samples."

Park until BASC demonstrates clearly better predictive performance at reasonable
MCMC sample sizes than the cheap alternatives already in the library (``BVSS``).
A method that needs ~1M iterations to stabilize is not worth the build cost
unless the sparse-pool edge is large and shows up well before that.

### Verdict

In-lane, sound, faithfully portable, and the referee / demonstrate-first
cross-checks agree -- but the capability overlaps ``BVSS`` and the convergence
cost is prohibitive for an interactive library. Parked on cost, not correctness.

---

## 9. TWP -- Tangential Wasserstein Projections (multivariate distributional SC)

**Status: Planned (paper reviewed in full; reference code read; demonstrate-first
partial -- the 1D reduction to ``DSC`` is verified. No library code yet). Verdict:
build via prototype-first -- the strongest in-lane candidate of the recent review
batch (the multivariate sibling of an estimator we already ship).**

### Source

> Gunsilius, F., Hsieh, M. H., & Lee, M. J. (2024). "Tangential Wasserstein
> Projections." *Journal of Machine Learning Research* 25, 1-41.

Reference implementation (Python; read ``twp_utils.py``):
**https://github.com/menghsuanhsieh/tangential-wasserstein-projection**
(``Python Code/twp_utils.py`` = the method: ``baryc_proj``, ``tan_wass_proj``,
``tan_wass_proj2``; deps POT ``ot`` + ``cvxpy`` + ``multiprocess``).

The multivariate generalization of, and explicitly pitched as complementing:

> Gunsilius, F. (2023). "Distributional Synthetic Controls." *Econometrica*
> 91(3):1105-1117 -- mlsynth's existing ``DSC`` (univariate).

### The idea in one line

Distributional SC for MULTIVARIATE outcomes: treat each unit as a probability
measure over R^d and project the treated unit's outcome distribution onto the
donor distributions in the 2-Wasserstein space. Because W2 is positively curved
for d>1 (no closed-form barycenter), TWP lifts the problem to the tangent cone at
the treated measure and reduces it to a simplex-constrained linear regression:
(i) OT plans ``gamma_0j`` from treated ``P0`` to each donor ``Pj`` (POT emd /
Sinkhorn); (ii) barycentric-project each to a tangent map ``b_0j``; (iii) solve
``min_{lambda in Delta} ||sum_j lambda_j (b_0j - Id)||^2_{L2(P0)}`` for the donor
weights; counterfactual is ``exp_{P0}(sum_j lambda_j b_0j - Id)``.

### Use case (when to reach for it)

Distributional SC where the per-``(unit,time)`` outcome is a JOINT distribution of
several outcomes and you want the counterfactual joint distribution plus treatment
heterogeneity across dimensions -- where ``DSC`` (one outcome's quantile function)
cannot go. Paper's application: a Medicaid expansion in Montana with a
d=28-dimensional, non-regular outcome measure. Same micro-level panel ingestion as
``DSC`` (one row per unit x time x individual observation), but the observations
are d-vectors rather than scalars.

### Relationship to ``DSC``, and the "should we just replace DSC?" decision (KEEP THIS)

TWP is the d>=1 generalization of ``DSC``; in 1D they coincide. Verified
demonstrate-first this session:

- **Derivation.** In 1D the OT plan is the monotone (sorted) coupling, so ``b_0j``
  is the sorted donor and TWP's tangent regression
  ``min_lambda ||sum_j lambda_j (b_0j - Id)||^2_{L2(P0)}`` becomes *exactly* DSC's
  quantile-function regression ``min_lambda ||sum_j lambda_j Q_j - Q_0||^2``.
- **Numeric check** (pure numpy, no POT; 4 donors + a Gaussian target): DSC
  weights ``[0.034, 0.493, 0, 0.474]`` vs TWP-1D ``[0.037, 0.494, 0, 0.469]`` --
  agree to 4e-3. The residual is ``np.quantile`` interpolation vs order
  statistics; in the real multivariate TWP it becomes EMD LP tolerance / Sinkhorn
  ``reg=0.005`` bias.

**Decision: do NOT replace ``DSC`` with TWP.** The 1D agreement is only to solver
tolerance, and ``DSC``'s 1D engine is the exact, closed-form (sort + simplex QP),
fast, theory-pinned (Zhang et al. 2026 Algorithm 1, benchmarked) path. Routing 1D
through an OT solver trades an exact engine for an approximate, slower,
POT-dependent one for the *same* answer, and breaks the cell-exact ``DSC``
replication. Instead: build TWP as the d>1 engine and use "TWP-in-1D reproduces
``DSC`` to tolerance" as the port's acceptance test (``DSC`` becomes the oracle).
If parsimony is wanted, unify under one API that dispatches on outcome dimension
(d=1 -> exact quantile engine; d>1 -> OT-tangent engine), sharing ``dsc_helpers``
ingestion + the simplex solver -- one estimator, no duplication, exact 1D path
preserved.

### Implementability & cost

- Deps: ``cvxpy`` (have) for the simplex regression; POT (``pot``/``ot``) is a NEW
  optional dep for the OT plans -- lightweight, gate it like ``numpyro``
  (``ot = ["pot"]``, lazy import). Exact EMD could fall back to
  ``scipy.optimize.linprog``; Sinkhorn is ~20 lines of numpy if avoiding the dep.
- Reference to validate against: the authors' Python repo -- same-language
  cross-validation, the easiest kind.
- Hard parts: (1) matching the barycentric-projection + tangent regression weights
  numerically (expect match-to-tolerance, not bit-exact -- OT solvers differ);
  (2) the results container -- a multivariate distributional effect is not a scalar
  or a single QTE curve; surface the counterfactual measure + weights, and likely
  per-dimension marginal QTEs and/or a Wasserstein-distance effect summary (a real
  ``BaseEstimatorResults`` extension, and the likely bottleneck).
- Estimate: ~3-5 days incl. results design + tests.

### Architecture

Sibling of ``DSC`` reusing ``dsc_helpers`` micro-level ingestion + simplex-weight
machinery: ``estimators/twp.py`` + ``utils/twp_helpers/{setup, ot_plans,
barycentric, weights, structures}.py``, riding ``dataprep`` (micro-level, d-dim
outcome cube) + a distributional ``BaseEstimatorResults``. Naming: ``TWP`` (or
``MVDSC``) -- ``DSC`` / ``DSCAR`` are taken, and ``DTWSC`` is reserved for the
parked Cao-Chadefaux dynamic method (#3); keep the distributional/dynamic SCs
disambiguated in docs.

### Replication path

Cross-validate against ``twp_utils.py`` on the repo's synthetic experiments --
match ``lambda`` and the reconstructed projection to tolerance; then the Montana
Medicaid d=28 counterfactual if the data ships in the repo. First validation gate:
TWP-1D reproduces ``DSC`` to tolerance.

### Caveats

- OT solver tolerance => "matches to tolerance," not bit-exact; pin the solver +
  Sinkhorn regularization to the reference.
- Inference is thin in the paper (consistency of weights/projection, no ready CI);
  lean on placebo / permutation as ``DSC`` does.
- The multivariate-effect representation is the real design work -- settle it
  before building.

### Verdict

Build (prototype-first). Genuine multivariate gap next to ``DSC``, JMLR-published,
reproducible with same-language Python reference code, and the new dependency (POT)
is a reasonable optional add. Run ``/replicate`` against ``twp_utils.py`` (acceptance
gate: reproduces ``DSC`` in 1D) -> ``/new-estimator`` as a ``DSC`` sibling.

---

## 10. DRDIDSC -- doubly robust DiD-meets-SC identification (Sun, Xie & Zhang 2025)

**Status: Prototype-first (paper reviewed; no library code yet).**

### Source

> Sun, Y., Xie, H., & Zhang, Y. (2025). "Difference-in-Differences Meets
> Synthetic Control: Doubly Robust Identification and Estimation."
> arXiv:2503.11375v2 (25 Sep 2025). No replication package, no reference
> implementation.

Empirical application: Alaska's 2003 minimum-wage increase on equivalized family
income, CPS 1998--2003, following Gunsilius (2023) / Dube (2019).

### The idea in one line

One moment function that identifies the ATT if *either* conditional parallel
trends *or* a group-level synthetic-control condition holds -- neither implying
the other -- so the researcher no longer has to pick between DiD and SC:

```
phi(S) = (1/pi_1) * [ G_1 - sum_g w_g(X) * (p_1(X)/p_g(X)) * G_g ] * (dY - m_D(X))
```

with `dY = Y_T - Y_{T-1}`, `m_D(X) = E[dY | G != 1, X]`, `p_g(X) = P(G=g|X)` and
covariate-indexed SC weights `w_g(X)` summing to one. Read the bracket first and
it is SC-with-a-DiD-adjustment; read the difference first and it is
DiD-augmented-by-SC. Under PT the moment is Neyman orthogonal (weights are
irrelevant -- Theorem 1(i)); under SC it is *not*, so the asymptotics carry an
extra first-stage correction (Theorem 3). A multiplier bootstrap (iid weights,
mean 1, variance 1) is valid under either assumption, which is the practical
point: you never have to declare which assumption you are relying on.

Estimation is a three-nuisance cross-fitted plug-in: local-polynomial `m_{g,t}(x)`
and `m_D(x)`; the propensity *ratio* `r_{1,g}(x) = p_1/p_g` fit directly by local
polynomial on `rho(r,G) = r^2 G_g - 2 r G_1`; and weights from the
exactly-solvable linear system `w_0(x) = (M'M)^{-1} M' m_1` built from the
pre-period conditional means. Weights are *not* simplex-constrained (footnote 3).
Extensions cover repeated cross-sections (Section 4 -- required for the CPS
application) and staggered adoption (Section 5).

### Scope gate: passes, but through the micro-panel door

Panel, units x time, pre/post -- but the asymptotics are `n -> infinity` with `T`
and `N_G` *fixed*, and the unit of observation is the individual inside a group,
not the group. So it does not ride `datautils.dataprep`; it needs its own
`utils/<name>_helpers/setup.py` over long micro data, exactly as `DSC` and `SCD`
already do. That precedent is the reason this is in-lane rather than a new family.

Required inputs beyond the usual four columns: a group column (states) and a
covariate list -- one continuous covariate (the theory in footnote 8 is written
for scalar `X`) plus discrete covariates handled by partitioning the sample and
estimating cell by cell, which is what the paper's Table 1 does (nine cells).

### Gap vs overlap

Overlap-grep (`doubly robust|propensity|parallel trends|Sant'?Anna|multiplier
bootstrap|cross-fitting`) returns `PROXIMAL`'s DR module, `PPSCM`, `MICROSYNTH`,
`SSC`, `BEAST`, `SEQ_SDID`. Adjudicated:

- **Genuine gap.** No estimator in `mlsynth` does *identification-level* double
  robustness (PT-or-SC), and nothing implements a Sant'Anna--Zhao-style
  covariate-conditional DR DiD at all (`grep "Sant'Anna"` over non-test source is
  empty). `PROXIMAL`'s DR is a different double robustness (outcome bridge vs
  treatment bridge, Qiu et al. 2024) on aggregate proximal panels.
- **Closest sibling: `SCD`** (Rincon & Song 2026, Synthetic Control with
  Differencing). Same data shape (grouped microdata / repeated cross-sections),
  same fixed-`T`, `sqrt(n)`-in-individuals regime, and the same
  differencing-plus-SC combination. `SCD` has *no covariates*, uses simplex
  weights on the differenced pre-period, and inverts influence functions rather
  than bootstrapping. In the no-covariate limit the two nearly coincide -- which
  is both the strongest cross-check available and the clearest overlap risk.
- Also adjacent: `DSC` (Gunsilius 2023 -- the paper's own empirical benchmark,
  already ported, already benchmarked on this data), `MICROSYNTH` (individual-level
  balancing, selection-on-observables), `SDID` / `MASC` (DiD+SC hybrids, aggregate).
- **Naming.** `DSC`, `SCD`, `DPSC` and `DROSC` are all taken; `DRSC` would read as
  a typo for `DROSC`. Prefer `DRDIDSC` (or another unambiguous acronym) and say
  in the docstring that it is unrelated to `DROSC`.

### Implementability & cost

Pure NumPy/SciPy. No solver: the SC weights are an OLS solve, not a QP, because
nonnegativity is deliberately dropped. Local linear regression on a scalar
covariate, `L`-fold cross-fitting, and a 500-replication multiplier bootstrap that
refits every nuisance under the weighted empirical operator. The only piece with
no Python reference is the MSE-optimal bandwidth the paper takes from `nprobust`
(Calonico--Cattaneo--Farrell); a Fan--Gijbels rule-of-thumb times the paper's
`n^{1/5 - 1/3.5}` undersmoothing factor is a defensible substitute, and the paper
states bandwidth sensitivity is mild -- expose `bandwidth` in the config either way.

Cost: roughly 3--4 days for the panel + repeated-cross-section cases with the
bootstrap and full test layering; staggered adoption (Section 5) is another day
and should be deferred to a follow-up. The genuinely fiddly parts are (a) the
bootstrap having to re-run the *nuisance* fits, not just reweight the moment,
(b) cross-fitting interacting with the per-cell partition on discrete covariates,
and (c) deciding the public estimand -- the paper reports nine per-cell ATTs and
never aggregates them.

### Replication path -- weak, and this is the crux

- **Path A is not reachable from repo data.** `basedata/dube_minwage.parquet`
  (already here, used by `benchmarks/cases/dsc_dube.py`) is the `DiSCo` package's
  `dube` extract: `(time_col, id_col, y_col)` only, 34 states x 1998--2004,
  subsampled to 250 draws per state-year. Table 1 needs household age, education
  count and child count, which that file does not carry -- so reproducing the nine
  cells means reconstructing a CPS/IPUMS extract to Dube's household definitions.
  Legitimate under scenario-1 rules (documented reconstruction counts as a pass),
  but matching estimates like `-0.636 (-1.949, 1.287)` to display precision with
  no author code is not realistic. The only durable claim is the qualitative one:
  all nine intervals cover zero.
- **Path B is not self-contained either.** DGP1--3 are calibrated to the authors'
  imputed CPS panel (factor loadings from a matrix-completion fill of their own
  data), which is not published. The *structure* is fully specified, though --
  factor model, `w_g(x) = 0.2 + 0.8 * softmax((g - N_G - 1)x)`, treatment effect
  scaled off `sin(2 pi x)` -- so it re-implements cleanly on a substitute panel.
  Target the geometry rather than the cells: negligible bias, SD falling at
  `n^{-1/2}` across `n = 1000/2000/3000`, bootstrap coverage near 0.95 under all
  three DGPs, and -- the claim that matters -- a plain DR-DiD biased under DGP1
  while this estimator is not, and a pure SC estimator biased under DGP2.
- **The strong anchor is cross-validation against `pedrohcgs/DRDID`.** Set
  `N_G = 1` and `w = 1` and the estimator collapses to Sant'Anna--Zhao DR DiD. With
  a *discrete* covariate the local polynomial degenerates to cell means and a
  saturated DRDID does too, so the two should agree to machine precision. `DRDID`
  is a GitHub install, which is the route that works in this sandbox
  (`agents_r_environment.md`). Make this the acceptance gate for any build.
- Second anchor, no reference needed: with no covariates the estimator must equal
  equation (4), `dYbar_1 - sum_g w_g dYbar_g`, computable in closed form.

### Findings from the feasibility probe (keep these)

Ran the no-covariate special case (eq. 4) on `basedata/dube_minwage.parquet` with
the paper's own six donors (VA, NH, MD, UT, MI, OH), AK treated, 1998--2003:

```
T = 6, N_G = 6  ->  M is 5x5, square: the weight system is EXACTLY identified
weights: VA 1.154, NH 0.708, MD -1.059, UT 0.438, MI 1.650, OH -1.892   (sum 1)
cond(A) = 10.95 ;  eq.(4) ATT = 0.622
```

The weights interpolate five noisy pre-period differences with zero degrees of
freedom and land far outside the simplex. Under Assumption PT this is harmless --
Theorem 1(i) says the weights do not matter. Under Assumption SC, which is the
whole selling point, it is the estimator, and it is the non-orthogonal branch with
the harder variance. The paper's own application sits exactly at `T = N_G`, its
footnote 4 concedes the `T < N_G` case needs a penalty and defers it to future
work, and its nine intervals are correspondingly wide (one is
`(-5.667, 2.471)`). Any build should ship a conditioning/extrapolation diagnostic
and should probably offer an optional ridge or simplex restriction on the weight
solve, flagged as a deviation from the paper.

### Caveats

- The six donors are chosen by reading off Gunsilius's estimated weights -- a
  data-driven donor selection performed outside the method, whose inference does
  not account for it. The `T >= N_G` requirement is what forces the pruning.
- No code, no data, recent preprint, single empirical application whose headline
  result is a null. Nothing here is *wrong*, but there is no published number this
  library could be pinned to.
- Genuine value is the identification-robustness argument, not a demonstrated
  accuracy edge: the paper never shows the estimator beating a competent DiD or SC
  baseline on real data, only that it stays valid in simulations where each
  baseline in turn is invalid.
- Practitioner reach: state- or region-level policy evaluated on survey microdata
  (CPS, ACS, PSID, NLSY) or firm-level digital-trace panels, where `T` is short,
  groups are few, individuals are many, and the analyst has real covariates and no
  confidence in parallel trends. That regime is genuinely under-served in the
  library -- `SCD` is the only current answer and it ignores covariates.

### Verdict

Prototype-first, not build-now. The gap is real and the method fits the library's
micro-panel door, but validation is the binding constraint: no reference code, a
Path-A table that needs data the repo does not have, and a Path-B DGP calibrated
to an unpublished panel. Run `/replicate` with the `DRDID` collapse as the
acceptance gate (`N_G = 1`, discrete `X`, machine-precision agreement) plus the
eq.(4) no-covariate identity; only if both hold, proceed to `/new-estimator` as an
`SCD`/`DSC` sibling, with the weight-conditioning diagnostic in scope from day one.

---

## 11. Panel-shape coverage inventory for the benchmark suite

**Status: Planned (motivated by three defects found on 2026-07-30; no code yet).**

### The idea in one line

The suite catalogues its ~152 cases by *what* each validates and never by the
*shape* of the panel it validates on, so blind spots are invisible until
something stumbles into one.

### Why this is worth doing

Three defects landed on 2026-07-30 that the suite could not have caught, and all
three surfaced only because one newly added panel had an unusual shape.

The ridge-ASCM cross-validation defects (#297) were invisible on augsynth's own
canonical Kansas example: its selected penalty came out identical to fifteen
significant figures whether the fold count was right or wrong, because with
`T0 = 89` against `J = 49` the CV curve is flat near its optimum. The Song panel
detected both immediately -- `T0 = 25` against `J = 37`, with the final
pre-period on a seasonal ramp carrying nine times the average held-out error.
Same code, same test, one panel blind and one panel diagnostic.

The covariate-aggregation defect (#299) is sharper still. Kansas *does* have
sparse covariates and could have caught it, but two errors were partly
cancelling; it took fixing the CV first to expose the second. And when that fix
landed, an audit found **no other case in the suite has sparse covariates at
all** -- so that fix has exactly one witness.

There is a selection effect at work. Canonical benchmark panels are chosen by
paper authors partly because the method works cleanly on them, which is the
wrong sampling distribution for finding bugs: it tests each method where it is
best behaved.

### What to build

A short inventory -- a script or a generated table -- recording per case:
`T0` vs `J`, cells per unit and their spread, covariate sparsity, donor-matrix
conditioning, and structural oddities (a donor-pool aggregate used as its own
treated unit, as in Song's `"Southern control"`, is a degeneracy no Prop 99 /
Basque / Kansas panel contains).

Cheap to produce, and it turns "we have 152 cases" into a statement about what
they collectively cover.

### Caveat

Out-of-distribution benchmarks also cost more. Roughly half the effort on Song
went into establishing that a disagreement was the *reference's* fault rather
than ours -- it needed all 1024 cells plus a live augsynth run to attribute.
More coverage finds more, and each finding takes longer to adjudicate.

---

## 12. `q_min` / `q_max` for DSC

**Status: Planned (surfaced while closing #304; small and well-specified).**

### The idea in one line

Both official DSC implementations let the user restrict the matched quantile
range; `DSCConfig` has no equivalent.

### Detail

DiSCos exposes `q_min` / `q_max`, and the Stata command exposes `qmin` / `qmax`,
both defaulting to the full `[0, 1]`. The Stata Journal paper (§2.2) gives the
motivation: "researchers may sometimes wish to match or conduct inference on
specific parts of the distribution."

This is not cosmetic. The DiSCo vignette's own headline run uses `q_max = 0.9`,
and at that setting the two implementations select **entirely different donor
sets** from the same data than they do at `q_max = 1`. A user following the
vignette cannot currently reproduce it in mlsynth.

Scope: a config field on an existing estimator, threaded into
`sample_quantile_grid` (which after #307 builds `linspace(q_min, q_max, M)`,
already the right shape) plus validation that `q_min < q_max` within `[0, 1]`.
Its own branch and tests per the repo contract.

### Learnings

Worth pinning `q_max = 0.9` against the Stata implementation when this lands --
it is a second published configuration on a panel already vendored
(`basedata/disco_tenure.parquet`), so it costs one R run and buys another
external check.

---

## 13. Why the R and Stata DSC implementations disagree at small `M`

**Status: Open question (documented, not explained; no action required).**

### The observation

After #307, mlsynth reproduces the Stata `disco` command's published tenure
weights to 5e-05. The DiSCos R package, on the *same* vendored panel with the
same donor pool, sits 0.034 from those values at `M = 10,000` and does not
converge to them as `M` grows (0.0442 at M=100, 0.0431 at M=1,000, 0.0341 at
M=10,000).

### What is already known

Most of the gap is resolution, not method. The published run uses `m(100)`,
where the answer is far from converged -- the `amazon` weight runs
0.2203 (M=100) -> 0.1860 (1,000) -> 0.1827 (5,000) -> 0.1821 (20,000). Both grid
rules converge to about 0.182, and R's random rule reaches 0.1826 by
`M = 10,000`. Replicating R's `runif` rule in Python reproduces R's behaviour,
so there is no further bug in the R package beyond the seed noise recorded in
#304.

Ruled out: different data (both read the same parquet), different donor pools
(identical 31), `simplex`, `q_max`, and the aggregation convention -- Stata
loops `t8 = 1..T0-1` dividing by `T0-1` while R loops `1:T0` dividing by `T0`,
but their `T0` differ by exactly one, so both average the same two pre-periods.

### What remains

A residual systematic difference at matched large `M`, small enough to be
quadrature-rule bias in the argmin (Monte Carlo quadrature makes the *argmin* a
biased estimate even where the objective is unbiased, since the argmin is a
nonlinear functional of the sampled objective) but not demonstrated to be that.

Not worth chasing unless it grows. Recorded because the next person to compare
the two implementations will otherwise rediscover it from scratch, and because
the honest position is that mlsynth matches one reference exactly and the two
references do not fully agree with each other.

---

## 14. Kranz (2022) two-step SDID (`xsynthdid`)

**Status: DONE.** Shipped as `SDIDConfig.covariates` (issue #308,
branch `claude/sdid-covariates`): `mlsynth/utils/sdid_helpers/covariates.py`,
tests in `mlsynth/tests/test_sdid_covariates.py`, gold from
`benchmarks/reference/sdid_kranz/reference.R`. The notes below are kept as the
record of what was planned; two things came out differently in the build and are
worth carrying forward:

- The description below (taken from Kubo et al.) says "residualise on nuisance
  fixed effects and run SDID on the residuals". That is not what
  `adjust.outcome.for.x` does. It subtracts only `X @ beta` and leaves the unit
  and time effects in the outcome, because SDID constructs its own unit and time
  weights and handles them itself. Removing the fixed effects too would be a
  different estimator.
- Cross-validating the endpoint turned up a `synthdid` fact worth knowing
  generally: `synthdid_estimate` solves its weight programs by projected
  gradient and stops at `min.decrease = 1e-5 * noise.level`, while mlsynth
  solves them exactly. Usually the two agree to ~1e-7, but on a ridge-dominated
  panel (adjusting away a strong covariate leaves a residual small next to
  `zeta.omega`, so `omega` sits near the uniform point on a nearly flat surface)
  the gap reaches 5e-3. mlsynth attains the strictly lower objective. Any future
  SDID cross-check against R should compare at a tightened `min.decrease`, not
  at the default -- see `TestSynthdidsEarlyStop`.

### The idea in one line

Residualise the outcome on nuisance fixed effects first, then run SDID on the
residuals -- so seasonal and other controls can enter an SDID design that
otherwise only admits unit and time effects.

### Detail

> Kranz, S. (2022). "Synthetic Difference-in-Differences with Time-Varying
> Covariates." Package: `skranz/xsynthdid`.

The procedure, as Kubo et al. (2025) describe it:

1. regress `Y_{i,y,q}` on quarter dummies, unit fixed effects and year fixed
   effects;
2. take the residuals `Ytilde`;
3. run ordinary SDID on `Ytilde`.

Kranz shows this outperforms passing the controls to `synthdid` directly when
there are controls other than individual and time fixed effects -- seasonal
dummies being the motivating case.

### Gap vs overlap

Genuine gap, and small. `grep -rn "Kranz\|xsynthdid\|residuali" mlsynth/utils/sdid_helpers/
mlsynth/estimators/sdid.py` returns nothing, and `SDIDConfig` has no
residualisation or covariate field. Scope is a config option plus a
pre-processing step on an existing estimator -- comparable to items 12
(`q_min`/`q_max` for DSC) -- not a new estimator.

`SDID` itself is already cross-validated against the authors' `synthdid` R
package (`benchmarks/cases/sdid_prop99.py`, agreement pinned to 0.02 packs), so
only the residualisation step needs new validation.

### Replication path

Cross-validation against `skranz/xsynthdid` directly. Do **not** validate it
against Kubo et al. (2025) -- see item 15 for why that paper cannot serve as a
target.

---

## 15. Kubo et al. (2025) wildlife-trade spillover -- assessed, NOT worth building

**Status: Closed as not-worth-doing. Recorded so the assessment is not repeated.**

### Source

> Kubo, T., Mieno, T., Uryu, S., Terada, S., & Verissimo, D. (2025). "Banning
> Wildlife Trade Can Boost the Unregulated Trade of Threatened Species."
> *Conservation Letters* 18:e13077. Replication package:
> `nies-consplan/wt_policy_spillover` (data included, 58 MB of
> transaction-level CSVs).

### Why it looked attractive

A rare combination: an SDID application with published effects and confidence
intervals for three taxa, the full transaction-level data shipped in the repo,
and the estimation script included. On paper a clean Path A.

### Why it is not worth building

Two independent reasons, either sufficient.

**1. The authors' code does not reproduce the authors' paper.** Running their
`scm.R` unmodified (R 4.3.3, `synthdid` at HEAD, locale set as their script
does):

    taxon             scm.R                        published paper
    water bug         +16.70 [ 12.87, 20.54]       +17.54 [14.03, 21.06]
    salamander        +12.22 [  1.94, 22.51]       +10.06 [ 2.73, 17.39]
    freshwater fish    +1.36 [ -5.90,  8.62]        +6.19 [ 0.12, 12.25]

The freshwater-fish row changes the conclusion, not just the digits: the paper
reports a statistically significant positive spillover, and the shipped code
gives a point estimate 4.5x smaller with a CI spanning zero.

The likely cause is that the paper describes the Kranz two-step (item 14) --
residualising on quarter, unit and year fixed effects, citing `xsynthdid` -- and
`scm.R` calls `synthdid_estimate` on raw panel matrices, with no residualisation
and no `xsynthdid` anywhere in the repository. So the published numbers appear to
come from an estimator the replication package does not contain. Until that is
resolved by the authors there is no stable Path-A target.

**2. Pinning against their code instead would be redundant.** That comparison is
"mlsynth's SDID equals synthdid's SDID", which `sdid_prop99` already pins on
Prop 99. A second case asserting the same fact on wildlife data adds coverage of
panel *shape* (40 quarters, count outcomes, 14-44 donors) but not of behaviour --
and item 11 is the cheaper way to address shape coverage.

### Findings worth keeping

mlsynth's `SDID` reproduces the authors' code to 3 decimal places on identical
panels -- 0.0033, 0.0005 and 0.0265 across the three taxa. So the estimator was
never in question here; only the target was.

Getting there took the one-file-two-readers discipline. Rebuilding the panel in
Python from the raw CSVs gave a salamander estimate of -17.07 against the
reference's +12.22 -- a sign flip caused entirely by harness error (a 20-entry
species-renaming table in `R/case.R`, fiscal-quarter arithmetic with
`fiscal_start = 2` and a 9-day shift, and a taxon-specific relabelling of the
exotic *Kirkaldyia*). Exporting the panel matrices from R and feeding mlsynth the
identical file collapsed the disagreement to 5e-4. Rebuild the *inputs* in the
reference, never in the port.

One environment note: the data is Japanese-language and the species matching is
by string. Without `Sys.setlocale(category = "LC_ALL", locale = "C.UTF-8")` --
which their `scm.R` sets and which is easy to drop when porting -- every species
silently classifies as "Control" and `panel.matrices` fails with "no variation in
treatment status".

### If this is ever revisited

The finding in (1) is worth reporting to the authors; it is their defect, not
mlsynth's, and the exact reproduction is recorded above. Should they publish a
corrected replication package that reproduces the paper, the case becomes
attractive again -- but only after item 14 lands, since the paper's stated method
needs it.

---

## 16. Identification-bound diagnostic for synthetic control

**Status: Planned (small, self-contained). Surfaced twice while assessing
candidate papers; both assessments turned on it.**

### The idea in one line

Report the range of post-treatment effects attainable by *any* synthetic control
that fits the pre-period nearly as well as the optimum, so a reader can see
whether an estimate is pinned by the data or merely one of many equally good
answers.

### Why

Synthetic control reports a point estimate from one weight vector. When the
donor pool is large relative to the pre-period -- which is the common case in
applied work, not an edge case -- many weight vectors fit the pre-period almost
identically and extrapolate to very different post-treatment effects. Nothing in
the standard output distinguishes "the data pin this number" from "this is one
draw from a wide set", and placebo inference does not answer it either: placebo
tests ask whether the gap is large relative to donors, not whether the gap is
determined.

This came up independently in two paper assessments and decided both:

- Kennedy-Shaffer (2025), MLB shift ban. 24 predictors, 20 donors, 5 pre-periods
  per player. Three implementations (the paper, its own shipped code, mlsynth)
  gave OBP effects of 0.075, 0.085 and 0.080 for Corey Seager. All three lie
  inside the set of controls fitting the pre-period within 2 percent of optimal,
  which spans [0.074, 0.085]. There is no tolerance at which one is right, so
  the paper was declined as a benchmark case.
- Lamba et al. (2023), tiger reserves. Averted forest loss on the headline
  reserve is 2645 ha published, 1999 ha under the current `tidysynth`, 2825 ha
  under mlsynth. The 2-percent band is [2768, 2922], the 10-percent band
  [2594, 3026]; the current `tidysynth` value falls outside even a 50-percent
  band, which is what identified it as a poor fit rather than a rival answer.

In both cases the bound turned an argument about whose implementation was right
into a measurement. That is worth having as a first-class output.

### What it is

For a fixed tolerance `eps`, with `w` on the simplex:

    minimise / maximise   tau(w) = y_post - Y0_post @ w
    subject to            ||y_pre - Y0_pre @ w||^2 <= (1 + eps) * SSR*

where `SSR*` is the minimum attainable pre-period sum of squares. The feasible
set is convex (a norm-ball intersected with the simplex) and the objective is
linear, so both ends are exact convex programs -- no sampling, no heuristic. It
is roughly fifteen lines of cvxpy and runs in milliseconds on the panels above.

### Gap vs overlap

Genuine gap. `grep -rniE "identification|partial.?ident|bound" mlsynth/utils
--include=*.py` returns nothing of this kind, and no `*Config` carries a
tolerance-band option. Existing inference modes answer a different question:
`placebo` and `lto` rank the treated unit against donors, `scpi` / `conformal` /
`eiv` build prediction intervals under a sampling model, `ttest` debiases the
ATT. None of them varies the weights subject to a fit constraint.

The nearest relative in spirit is the sensitivity literature rather than
anything currently implemented.

### Where it should live

Most likely `inference="idbound"` on `VanillaSC`, or a standalone utility under
`mlsynth/utils/vanillasc_helpers/` callable on any fitted result, since it needs
only `Y0_pre`, `Y0_post`, `y` and the pre/post split -- not the estimator that
produced the weights. A standalone utility is probably the better first cut: it
then applies to ASCM, penalised SC and the rest without touching their configs.

Open design questions, all small:

- which tolerances to report by default (2 / 10 / 25 percent were informative);
- whether to bound the per-period path as well as the aggregate ATT;
- whether to express the tolerance in SSR, RMSE, or as a fraction of the
  treated unit's pre-period variance, which would be scale-free and comparable
  across panels;
- what to do when `SSR* == 0` (perfect pre-fit), where a relative tolerance
  degenerates and an absolute one is needed.

### Validation

No external reference exists, so this is neither Path A nor cross-validation.
Validate by construction instead, which is stronger here: the bound has exact
properties that can be asserted rather than compared. The optimum must lie
inside every band; bands must be nested in `eps`; width must be monotone in
`eps`; at `eps = 0` the band must collapse onto the argmin's effect whenever the
argmin is unique; and on a constructed panel with a known-unique optimum the
width must go to zero. Add a Prop 99 and a Basque case so the reported widths on
familiar panels are on the record.

### Cost

Small -- a day or so including tests and a docs section. The maths is settled;
the work is in the API choice and in writing the docs so a non-expert reads the
output correctly. The failure mode to avoid is a reader treating the band as a
confidence interval. It is not one: it carries no sampling model and no
coverage guarantee. It is a statement about what the data can and cannot
distinguish, and the docs need to say so plainly.

---

## 17. Shen (2026) Two-Way Synthetic Forecasting (TWSF) -- assessed, PARKED pending two answers from the author

**Status: Parked. Not a rejection -- the method is novel, the gap is real, and
an implementation exists in scratch. It is blocked because neither replication
path can be completed from the paper as published. Recorded so the two spikes
are not repeated.**

### Source

> Shen, D. (2026). "Causal Forecasting in Panel Data: A Two-Way Synthetic
> Forecasting Approach." arXiv:2606.18512v1. Single author, no code release.

Data for the case study (NFL stadium openings) is available: the NYT county
series plus the stadium-county mapping are both vendored in
`Joshuashou/Synthetic-Control-Paper-Model` (a *different* paper -- "Synthetic
Control Method with Many Outcomes", pending at JMLR -- which reuses the same
setting). That repo is **not** the PNAS replication package, and its
opening-date column must not be used: Kansas City is typed `9/10/2021`, New
Orleans and Washington are marked "Closed to fans" though the paper has them
opening 10/25 and 11/8, and Pittsburgh and Tennessee are blank. Take dates from
the TWSF paper's Table 1 instead.

### The idea in one line

Forecast a *treated* potential outcome for a unit that has never been treated,
at a time *beyond the end of the panel* -- by combining a Synthetic
Interventions unit-side regression with an mSSA Page-matrix time-side
forecaster, bilinearly.

### Why it is worth wanting

It is a genuinely different estimand from everything in the library. Every
mlsynth estimator imputes a counterfactual *inside* the observed window; TWSF
extrapolates past it. The applied regime is prospective rollout: six metros
already have the intervention, what happens to Denver next month if we switch
it on? Today that question can only be answered by abusing `SI`.

### Gap vs overlap

Unusually clean. Grepping the library:

* HSVT / PCR -> `mlsynth/utils/pcr/core.py`, `si.py`, `clustersc.py`:
  **capability overlap**, already built and validated (`si_prop99`,
  `pcr_rsc_ref`).
* Synthetic Interventions -> `estimators/si.py`: **capability overlap**, and
  `SIConfig.inters` is already the exact interface for the treated donor pool.
* Page matrix / Hankel / mSSA -> **zero hits. Genuine gap.**
* Forecasting beyond the panel -> zero (`sbc_helpers/trend_forecast` is a
  within-panel detrend). **Genuine gap.**

Closest existing estimator: `SI` (same author lineage, same PCR kernel).

### Cost -- low, and this was verified rather than estimated

Both the one-step and the recursive multi-step orthogonalized estimators were
implemented during the spikes on top of `hsvt` and `pcr_weights` **without
modifying them**. Pure NumPy: SVD, pseudo-inverse, a companion matrix and its
Jacobian. No solver, no compiled dependency. The "half of it already exists"
claim held up completely.

### Why it is parked: two replication paths, two unreported specifications

**Path B (the paper's simulation) -- geometry not reproducible.**

What checked out:

* the estimator algebra is exact -- with `sigma = 0` the forecast error is 0 to
  machine precision at every `d`, confirming the Page-block layout, the
  identification, and every step of the orthogonalization;
* the plug-in variance is essentially exact -- empirical SD / plug-in SE =
  **0.92-1.08** across every configuration tried. This is a strong positive
  result for the paper's inference theory.

What did not:

* coverage. Nominal 90%, obtained 0.39-0.87, driven entirely by a
  latent-draw-specific **bias** that does not shrink in `d` (at `d = 150`, bias
  0.195 against an SE of 0.084).

The cause is a calibration the paper states but does not report: "the loading
matrices and scaling are calibrated so that the population design and
forecasting blocks have the intended ranks and *well-separated nonzero
spectra*". Neither `A_0`, `A_1`, the harmonic periods, nor the scaling is
given. A reconstruction from the prose produced a Page spectrum whose nonzero
singular values span **500,000x**, with the smallest signal direction at ~1e-4
against a noise floor of 3.7 -- so PCR at the oracle rank inverts four
directions of pure noise. Rebuilding the DGP for separation moved coverage to
0.86-0.87 at large `d`, but no hand-tuned variant reached nominal.

Worth flagging to the author: **small average bias and broken coverage are
compatible here.** Averaged over latent draws the bias is -0.02 to +0.03,
matching the paper's reported figure, but the bias is draw-specific and does
not cancel *within* a replication. A bias-vs-`d` figure can look clean while
coverage is 0.39.

**Path A (the NFL validation study) -- containment not reproducible.**

The matchable claim is crisp: of 11 cities opening after the donor pool,
exactly three (Carolina, Cincinnati, Pittsburgh) fall outside the pointwise 90%
band. The hyperparameters are *given* (Table 2, CV-selected per opening date),
so unlike Path B there is nothing to guess about the method.

What checked out:

* the data pipeline, verified against the paper's own figure -- Carolina reads
  29,454 at `tau` and 31,688 at `tau+14`; the figure shows ~29,500 and ~31,700;
* the point forecasts -- RMSE 66-1,669 on levels of 13k-150k, i.e. 0.2-4% over
  a 14-day horizon from six donors;
* the relative ranking -- Carolina, Cincinnati and Pittsburgh are outside the
  band under *both* outcome readings below, exactly the three the paper names.

What did not: absolute containment, because the scale the band is computed on
is ambiguous by a factor of ~15.

  ==============================  ==================  =============  =====
  outcome scale                   half-width @ h=14   outside band   paper
  ==============================  ==================  =============  =====
  estimate on cumulative          5,928               7 / 11         3
  estimate on daily, show cumul.  97                  10 / 11        3
  ==============================  ==================  =============  =====

The figure's y-axis reads "Accumulated case count", but bands that tight
require ``sigma_hat`` on a *daily* scale. On cumulative counts ``sigma_hat`` =
732 -- which is rank-4 PCR curve-shape misfit, not idiosyncratic noise, and 6x
a typical daily increment.

The paper's secondary claim did not reproduce either: it reports that longer
treated-donor windows forecast better, but Washington (`T1 = 49`) and Baltimore
(`T1 = 42`) were among the worst targets.

### What would unblock this

Two one-line answers from the author:

1. `A_0`, `A_1` and the harmonic periods used in the simulation DGP (or the
   simulation script).
2. Whether the case-study estimation runs on cumulative or daily counts.

Either likely unblocks its path; both would make this a build.

### Architecture, if it is ever built

New top-level estimator `TWSF`, not a `method=` on `SI` -- different estimand,
different result semantics, four extra hyperparameters (`L`, `k_y`, `k_z`,
`k_w`) plus `horizon` and `multistep: Literal["direct","recursive"]`. It should
import `mlsynth.utils.pcr.core` rather than reimplement HSVT/PCR, and copy
`SIConfig.inters` for the treated-donor pool. It rides the result contract:
`observed_outcome` = the realized control path, `counterfactual_outcome` = the
forecast treated path, `estimated_gap` = the contrast, `counterfactual_lower/
upper` = the pointwise band -- with the sign convention inverted relative to
classic SC, the same inversion `SI` already handles.

Feasibility traps to surface as diagnostics, not crashes: `B = T_1 / L` must
give at least two blocks, and the *direct* multi-step estimator is infeasible
at short horizons -- the paper's own case study cannot use it, which is why it
uses the recursive one.

### Learnings

* **A "green" that is really an absence of measurement.** The Path B spike
  looked like an estimator failure and was not; the variance formula was exact
  throughout. Separating "algebra wrong" from "variance wrong" from "design
  wrong" needed three targeted diagnostics -- `sigma = 0` recovery, empirical
  SD vs plug-in SE, and the signal spectrum. Run those three before concluding
  anything about a factor-model estimator.
* **Read the figure axes.** The outcome transform for Path A was not in the
  prose; it was the y-axis label of a PNG in the arXiv source tarball. One
  `Read` of the image settled a question that guessing had got wrong.
* **A paper can report a calibration without reporting the calibration.**
  "Calibrated so that the spectra are well separated" is a description of an
  intent, not a specification. When a Monte Carlo's behaviour is governed by
  spectral separation and the loadings are unreported, Path B is not
  reproducible in principle, however complete the algorithm is.
* **Grep substrings lie.** Searching the repo for NFL data matched
  `fast_scm_helpers/conflict.py` -- "co-nfl-ict". The data was genuinely absent
  from `basedata/` and from all of git history.

---

## 18. SURVSC -- Synthetic Survival Control (censored time-to-event outcomes)

**Status: Parked, build-ready. Paper reviewed in full; Path B replication DONE
and reproduces (`benchmarks/reference/survsc_mc/`). No estimator code. Nothing
blocks a build -- this is parked on sequencing, not on doubt.**

### Source

> Han, J. X., & Shah, D. (2025). "Synthetic Survival Control: Extending
> Synthetic Controls for 'When-If' Decision." arXiv:2511.14133v1 (MIT).

No code release, no data release. The clinical application uses proprietary
retrospective T-cell-lymphoma records from 13 institutions across 10 countries,
so the paper's Section 4 Monte Carlo is the only external check that exists.

### The idea in one line

Synthetic control where each unit-period outcome is an entire survival curve:
Kaplan-Meier per unit-period absorbs the censoring, the curves are subsampled
onto a shared grid, PCR learns donor weights from the pre-period curves, and
those weights are applied to the donors' post-period curves.

### Why it fills a real gap

`grep -rilE "survival|hazard|kaplan|meier|time-to-event|censoring" mlsynth
--include=*.py` returns zero files across all 99 exports. Nothing in the
library touches censored time-to-event data.

Closest existing estimators, by two different measures. By estimand shape,
`DSC` -- distributional SC also has a functional outcome built from grouped
microdata, and a survival function is `1 - F`, so absent censoring the target
is the object DSC already models. They differ in aggregation geometry: DSC
averages quantile functions (2-Wasserstein barycenter), SURVSC averages
survival functions pointwise in probability space. By machinery, `SI` -- same
author lineage, same latent-factor plus PCR construction.

### Naming (do not collide)

`SSC` is taken: `mlsynth.SSC` is Staggered Synthetic Control (Cao, Lu & Wu).
Use `SURVSC`.

### What the Path B spike established (KEEP THESE)

All six Table 1 cells land within 1.5x of the published sup-norm error, two of
them on top of it, and the Figure 4 claim that error falls in `K` holds in both
the Cox and Aalen designs. Step 3 needs no new code: the paper's closed form is
`mlsynth.utils.pcr.pcr_weights` to 8e-17.

Three design details the paper leaves unstated, two of them decisive:

* The rank `r0`. Section 3.4.2 says "a gap rule, elbow, or cross-validation"
  and picks none. Of the rules mlsynth ships, only USVT reproduces the paper;
  `cumvar` and `spectral` report error that *grows* as data is added.
* Whether the latent design is redrawn per replication. Independent draws per
  `K` break the monotone decrease, because the latent draw moves the evaluation
  horizon over two orders of magnitude (pooled 90th percentile: median 93,
  range [20, 5743] on Cox) and swamps the effect of `K`. Common random numbers
  restore it.
* Whether `tau_tilde` pools every cell or uses the treated unit. Not decisive;
  both readings give the same picture.

### Two build decisions, already settled by the spike

The paper's PCR weights are unconstrained, so the counterfactual
`sum_m w_m S_m(t)` need not be monotone or stay inside [0, 1] -- it is a valid
survival function in a minority of replications (monotone in 0-55 percent,
inside [0, 1] in 15-45 percent; at Aalen `K = 100`, zero of twenty). Abadie's
convex-hull condition fixes it for free, since a convex combination of monotone
[0, 1] curves is monotone in [0, 1]. The simplex fit already in
`mlsynth.utils.inferutils._outcome_only_simplex` gives 100 percent valid output
at every cell and costs nothing on average. Ship both weight schemes; simplex
is the defensible default, PCR reproduces the paper.

The estimand is a curve, not a scalar, so `EffectsResults` needs an RMST
difference over `[0, tau_tilde]` as the scalar ATT, with the gap curve in
`TimeSeriesResults`.

### Build path

`estimators/survsc.py` (thin) plus `utils/survsc_helpers/{config, setup, km,
pipeline, inference, plotter, structures}.py`. Ingestion is patient-level
`(unit, period, T, Delta)`, which `dataprep` cannot take -- but Kaplan-Meier
collapses each cohort to a curve on a shared grid, and at that point the data
is a standard panel of units x `2 * T0` points with `pre_periods = T0`, so the
contract reappears after step 1. `MicroSynth` is the precedent for
estimator-owned patient-level ingestion. Inference is the paper's donor-pool
bootstrap (Section 5.2, 500 resamples). Pure NumPy/SciPy; no new dependency.

### Caveats to carry into the build

Theorem 2's rate contains no `K` at all -- `K` enters only as a threshold for
the PCR stability argument -- while the simulation's entire finding is error
falling in `K`. The spike explains the discrepancy: the reported error is the
same order as a single Kaplan-Meier curve's error (Cox 0.0658 / 0.0376 / 0.0235
against Table 1's 0.1177 / 0.0652 / 0.0542), and the KM error falls at
`1/sqrt(K)`. The convergence Figure 4 displays lives in step 1, which the
theorem treats as a precondition. The docs page should say this instead of
restating the theorem as if it covered the regime.

The threshold itself is unmet by the paper's own experiments: at `T0 = 100`,
`N0 = 19` it is about `755c`, so `K = 100` fails it. In the application
`N0 = 9` makes the `N0^{-1/2}` term 0.33, a vacuous bound on a quantity in
[0, 1].

Scope the authors concede: `P = 2` periods only, non-informative censoring
assumed (IPCW left to future work), observed covariates unused in the
latent-confounding case.

### Verdict

Build when the queue allows. New method, real gap, most of the machinery
already exists, and the validation target is reproduced and staged.

### Learnings

* **A replication can fail on seeding, not on the port.** The first grid showed
  no monotone decrease in `K` and looked like a failed reproduction. The port
  was correct; the seeds were not. Where a DGP's nuisance draw has orders of
  magnitude more spread than the parameter being varied, common random numbers
  are not a variance-reduction nicety -- they decide whether the paper's claim
  is visible at all.
* **Check what the error metric is actually measuring.** Comparing the reported
  sup-norm error against the error of one Kaplan-Meier curve showed the headline
  convergence belongs to step 1, not to the synthetic-control step. That
  reconciled the simulation with a theorem whose rate has no `K` in it. Measure
  the floor before crediting an estimator with the decline above it.
* **Unconstrained SC weights do not preserve the object's shape.** Whenever the
  unit-level outcome is a function with structure -- a survival curve, a CDF, a
  density -- a linear-span condition lets the estimate leave the space the
  object lives in. The convex-hull constraint is what buys closure, and here it
  cost nothing.
* **An unguarded pseudo-inverse hides two failure modes, and the quiet one is
  worse.** `pcr_weights` divided by `s_r` with no cutoff: an exactly-zero
  singular value gives 0/0 and NaN with a `RuntimeWarning`, while a near-zero
  one gives weights of order 1e14 and no warning at all. Only the first is
  caught by an `isfinite` assertion. Fixed in `mlsynth/utils/pcr/core.py` with
  `mlsynth/tests/test_pcr.py`.
* **Mutation testing would not have found that bug, and would have said the
  line was fine.** Three mutants on the offending line (`/` to `*`, perturbed
  denominator, division deleted) were all killed by the existing suite, so it
  scored 100 percent while carrying the defect. The fault was an omission -- a
  missing guard -- and there is no line to mutate into one. Mutation testing
  perturbs code and reuses the test inputs; the defect lived in the input
  domain. Property-based testing is the technique matched to that fault, and
  `agents_tests.md` already required the edge cases that would have exposed it.

---

## 19. LPCA -- Local Principal Component Analysis (nonlinear factor structure)

**Status: Parked, build-ready. Paper reviewed in full; Path A replication DONE
and reproduces exactly (`benchmarks/reference/lpca_kansas/`). No estimator code.
Build the estimator against the paper's Monte Carlo, not against its empirical
comparison -- see Learnings.**

### Source

> Feng, Y. (2023). "Optimal Estimation of Large-Dimensional Nonlinear Factor
> Models." arXiv:2311.07243v1.

Replication code released: `yingjieum/Replication_NonlinearFactorModel_2023`
(R; ~60 lines for the estimator). The panel for the empirical application is
already in the repository as `basedata/kansas_taxcut.csv`.

Do not confuse it with Feng (2020), "Causal Inference in Possibly Nonlinear
Factor Models" (arXiv:2008.13651), which applies the same local-PCA building
block to cross-sectional treatment effects with mismeasured confounders. That
one is out of lane; this one has the panel application.

### The idea in one line

Split the time index in two; match each unit to its `K` nearest neighbours on
the first block under a pseudo-max distance; take a truncated SVD of the
neighbour submatrix on the second block. The nonlinear surface is approximated
by its local tangent plane, so the outcome matrix has to be low-rank only
locally.

### Why it fills a real gap

Across the 74 exports, every factor-structure estimator -- `MCNNM`, `RMSI`,
`CFM`, `FMA`, `GSYNTH`, `CSCIPCA`, `DMLFM`, `SNN` -- assumes the untreated
outcome matrix is globally low-rank in a linear factor structure. That is the
assumption this drops.

Closest existing estimator: `SNN` (Agarwal et al. 2021), also nearest
neighbours plus a low-rank imputation, but its neighbours come from the
sparsity pattern (anchor rows and columns) and it regresses; here they come
from a distance on a held-out time block and the step is a local SVD.
`NSC` is a name collision only -- "nonlinear" there refers to the outcome, and
the method is a penalized donor-weight scheme. `LPCA` is free as a name.

### Cost

Low. Pure NumPy/SciPy: a Gram matrix, `argpartition`, one truncated SVD of a
`K x p` block. No solver, no compiled dependency. Every tuning constant is in
the reference script.

### What the replication established

The empirical arm reproduces exactly: the LPCA counterfactual sits 0.5306
points above observed Kansas growth against the paper's 0.53, and the observed
series is below the LPCA path in 9 of 16 post-treatment quarters, as reported.
Full detail in `benchmarks/reference/lpca_kansas/README.md`.

### Learnings (keep these)

* **The paper's synthetic-control comparison is a v1 defect, since fixed.**
  Section 6.1 reports SC predicting growth 0.19 points *below* observed Kansas
  against LPCA's 0.53 *above* -- opposite signs, which is the whole rhetorical
  contrast and the basis for calling the SC answer implausible. The v1
  application script omitted `+ col.mean` on the SC line while carrying it on
  the LPCA line, so the SC path was compared against an observed series it had
  never been re-centred onto. The 2024 upload of the same script adds the term.
  Reproducing the defect gives +0.1948, matching the paper; correcting it gives
  −0.3340. Corrected, both estimators agree the tax cut cost growth and differ
  by 0.20 points of magnitude. Any docs page must not repeat the published
  comparison.
* **Validate a build on Table 1, not on Kansas.** The Monte Carlo is where
  LPCA is shown to beat global PCA, and it is unaffected by the above. The
  Kansas application is a demonstration that the estimator runs on a real
  panel, not evidence of superiority.
* **The pre-fit argument runs the other way.** On the window where both arms
  predict, LPCA's pre-treatment RMSE is 0.866 pp against the synthetic
  control's 0.624 pp. Both are in-sample there and SC is fitted directly
  against Kansas, so the ordering is unsurprising -- but it means "poor
  pre-treatment fit" cannot be the reason to prefer LPCA.
* **Two of the three tuning constants move the answer by 40 percent, and the
  paper sits at the extreme of both.** The sign is stable across everything
  tried. `K` (paper: `round(n^(2/3))` = 14) spans −0.369 to −0.531 over
  {7, 10, 14, 20, 25, 30}, and the component cap spans −0.413 to −0.531 over
  {2, 3, 4, 5}. The matching-block split is tame: 40 through 60 quarters give
  the same answer. Remark 4.3 defers rank selection to future research, so a
  build should surface the chosen rank and the neighbourhood as diagnostics
  and not present the point estimate as settled.
* **The paper reports no inference of any kind** -- no standard errors,
  confidence intervals or p-values, and Figure 3 has no bands. Theorem 6.1 is
  a uniform max-norm rate. Anything mlsynth attaches (the moving-block
  conformal machinery is already in the library) is the library's addition and
  the docs have to say so.
* **`p - p0` is assumed fixed.** Zeroing the treated post-treatment cells is
  defensible only when the post-period is short relative to the panel -- 16 of
  104 quarters here. The config needs a guard, not a footnote.
* **A defect can reproduce a published number exactly and still be a defect.**
  The check that caught this was not re-reading the port; it was pinning the
  solver against an independently validated reference (`ascm_kansas`'s
  classic-SCM rung, itself cross-validated against live augsynth) to 1.3e-07,
  then showing that the published number is unreachable: holding the
  pre-period fit at its optimum, the achievable post-treatment mean is the
  single point +0.5026, and reaching the published value costs 156 percent in
  sum of squares. Only then did the reference repository's history come into
  it. Cross-validate the machinery before doubting the paper, and read the
  reference's git history before concluding the paper is simply wrong.

---

## Done

*(empty -- move completed items here, preserving their Learnings subsection.)*
