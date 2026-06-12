# Learnings: the GeoLift Port (GEOLIFT)

Cross-cutting lessons from porting Meta's GeoLift market-selection routine onto
the `mlsynth` machinery. This is a **case study for porting an external
experimental-design tool** — read it before the next port (CausalImpact, the
GeoLift power module's siblings, etc.).

---

## 1. Faithful-first, then earn the divergences

Replicate the source **exactly** first — names, defaults, even the quirks — so you
have a validation anchor. *Then* add improvements, each behind a flag and a test.

- We kept GeoLift's surprising `dense_rank(power)` **ascending** rank and the
  scaled-L2-is-reported-but-not-ranked choice, with `.. note::` blocks in the
  docs explaining them. Don't "fix" a quirk silently — replicate it, document it,
  make the fix one line.
- Our divergences (mean aggregation, per-anchor RNG, the CV-once optimization)
  are **opt-in and tested against the faithful path**.

> If you find yourself disagreeing with the source's method while porting,
> write the faithful version anyway and add the corrected one as an option.

## 2. Design tools need treatment-*agnostic* data prep

`dataprep` resolves treated/donor/pre/post from a treatment indicator — but a
*design* runs **before** treatment exists. We added `geoex_dataprep`: a balanced
`time × unit` wide panel with **no treatment argument**, sharing `Ywide` /
`time_labels` conventions so it's interchangeable with `dataprep`.

**The `post_col` invariance** (mirrors MAREX/lexscm): once a real experiment has
run, the business wants to *rerun the design* on historical data and get the same
answer. Make this a guarantee:

```
geoex_dataprep(full_panel, post_col="post")["Ywide"]  ==  geoex_dataprep(pre_only_panel)["Ywide"]
```

The design slices to the pre-period; the result is identical whether you hand over
the full post-treatment panel or just the history. Pin it with a test.

## 3. Decompose into *dumb* helpers — one trivial thing each

The scoring layer is the hard part. It became ~12 leaves, each pure, each 100%
covered, each TDD'd before its caller: window arithmetic, treated aggregation,
donor selection, the single fit, scaled-L2, effect injection, the conformal
p-value, the per-placement simulation, the batch driver, power, MDE, rank.

"Dumb in the sense that they are super easy to reason about, debug, and test." A
helper that needs a paragraph to explain is two helpers.

## 4. Prove your optimizations exact — don't just claim them

GeoLift re-cross-validates the model for **every effect size** (the "psychotic
inefficiency"). The injection touches only the *post* block, so the pre-period the
CV sees is identical across effect sizes ⇒ the CV-selected `λ` is the same ⇒
cross-validate **once** and reuse it.

We **proved** this with an equivalence test
(`test_simulate_lookback_cv_once_equals_per_es_refit`): re-CV per effect size and
assert the conformal p-value is bit-identical to the CV-once path. An optimization
that changes a number is a bug; an optimization with an equivalence test is a
free win.

> Be honest about what is *not* exact. augsynth's conformal procedure refits on
> **all** periods; a fit-once pre-period-gap permutation is a *different, weaker*
> test, not the same one. We use the faithful conformal refit (fixed-`λ`) and say
> so.

## 5. The two-family contract + a single `fit()`

A `DesignResult` **resolves to** an `EffectResult` once outcomes exist. Don't
expose that as user ceremony.

- **The estimator exposes only `fit()`** (the library convention —
  VanillaSC/MAREX). Realization and plotting happen *under the hood*, driven by
  the data and config, not by manual method calls:
  - a `post_col` leaving a post window ⇒ `fit()` realizes the winner and populates
    `result.report`;
  - `display_graphs` ⇒ `fit()` plots (design phase, or realized post phase).
- Keep `realize_design` / `plot_design` as **standalone helpers** (called under
  the hood, available to power users) — never as estimator methods.

If a method's behavior is fully determined by "what data did I get," it shouldn't
be a separate method.

## 6. Separate the *estimand* from the estimator's *target geometry*

GeoLift **sums** the treated markets. For the business estimand (total lift,
total spend) that's correct. But as the SCM *target*, a sum of `k` markets sits at
`k×` donor scale — outside the donor convex hull — and a convex/ridge combination
can't reach it, inflating the scaled-L2 imbalance toward 1 and destroying its
power to discriminate.

The fix is to **separate the two**: fit on the **mean** (in-hull, well-posed),
then rescale to the total for reporting. Watch for this whenever an aggregate is
both the reported quantity and the fit target.

> **This is not just a geometry nicety — it is required for parity.** Sum vs mean
> is *not* a global scale for the conformal p-value: fitting the summed target
> (outside the hull) gives p ≈ 0.68 where the mean gives p ≈ 0.01 on the GeoLift
> walkthrough. With `fixed_effects=True` (now the default, §10) `GEOLIFT` fits the
> per-unit mean and rescales the *reported* paths by `k` for `how="sum"` (the
> p-value, a ratio of norms, is invariant to that reporting scale).

## 7. Store every candidate cleanly in the standard models

Each candidate design carries `WeightsResults` (donor weights) + `intercept`
(**a sibling field, not buried in `summary_stats`**) + `TimeSeriesResults` +
`FitDiagnosticsResults`, grouped like LEXSCM's `SEDCandidate` inside a
`MarketSelectSearch(shortlist, candidates, winner)`. Don't return ad-hoc dicts;
populate the standardized sub-models so every design is inspectable the same way.

## 8. Plot through the shared house style

Use `mlsynth.utils.plotting.mlsynth_style()` (the `MLSYNTH_RC` theme) as the
**default**, with a `theme` override — never ad-hoc rcParams per estimator. One
look across the library. Colour convention: black observed, red counterfactual.

## 9. Validate end-to-end on the source's own data

The market-*selection* routine has no published table, so we validated it on
**GeoLift's own example panel** (`basedata/geolift_market_data.csv`): a sensible
recommended design, and a correct **null** (joint conformal p ≈ 0.83) realized
over a no-effect post window. When there's no replication target, a faithful port
+ an end-to-end null/sanity run is the verification.

But the *realized effect report* — the ATT and conformal p — **does** have a
published anchor: GeoLift's `GeoLift_Walkthrough` (the `GeoLift_Test` panel,
`basedata/geolift_test_data.csv`). Look harder for a value-for-value target
before settling for a null-only check: the inference half of a "design" tool is
often the half the reference *does* publish. See §10.

## 10. Cracking augsynth/GeoLift parity — the four ingredients (and two traps)

The walkthrough reports per-unit ATT 155.6, lift 5.4%, incremental 4667,
conformal p 0.01. Reaching it (`mlsynth` hits 156.8 / 5.47% / 0.011) needed four
ingredients, **each a default we had not mirrored or a piece of augsynth's exact
algebra** — pinned in `test_geolift_walkthrough.py`, written up in
`docs/replications/geolift.rst`:

1. **Unit fixed effects** (augsynth `fixed_effects=TRUE`, GeoLift's default; now
   our default). `demean_data` subtracts each unit's *own* pre-period mean from
   all its periods and fits on the residuals — matching **shapes, not levels**.
   This is the mechanism that stops the donor pool from **absorbing a level
   shift**: level-matched donors can chase a post jump; demeaned ones can't.
   Without it the ATT is wrong (≈209 vs 311) *and* the conformal absorbs (p ≈ 0.56).
2. **Fit the mean of the treated units** (augsynth `colMeans`), not the sum — see
   §6; not scale-invariant for the p-value.
3. **The all-period conformal refit** (augsynth `cbind(X, y)`), with the
   full-path demean under fixed effects. This — not a pre-period fit-once — is
   what makes the residuals exchangeable.
4. **augsynth's ridge ASCM**: simplex base + a **period-space** ridge correction,
   `λ` by leave-one-period-out CV under the **1-SE rule** (`min_1se=TRUE`, the
   augsynth default). `ridge_augment_weights` reproduces it to corr 1.0000.

Two cross-codebase-consistency traps, worth carrying to the next port:

- **A calibrated test can *look* anti-powered.** The symptom ("our p=0.57 vs
  their 0.01, the conformal must be broken") was wrong. A 40-market **placebo
  study** showed the all-period refit is well-calibrated (≈10% rejection at
  α=0.10); the tempting "fix" (fit-once on the pre-period, permute the gap path)
  is the *broken* one (~50% false positives — pre residuals are in-sample, post
  out-of-sample, so not exchangeable). The low p was the **fit** (missing fixed
  effects estimating a smaller, absorbed effect), not the test. **Diagnose the
  estimand before blaming the inference; verify calibration with placebos before
  "fixing" a p-value.**
- **Match defaults before mechanisms.** Three of the four ingredients are just
  unmatched augsynth/GeoLift *defaults*. When two codebases disagree, enumerate
  the reference's defaults first — and **reproduce the reference end-to-end from
  scratch** (here ~40 lines of NumPy hitting ATT 312 / p 0.011) to localize which
  default matters, far faster than reading either codebase line by line.
