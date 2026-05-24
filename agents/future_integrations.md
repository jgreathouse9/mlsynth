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

## Done

*(empty — move completed items here, preserving their Learnings subsection.)*
