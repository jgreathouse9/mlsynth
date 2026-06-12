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
then rescale to the total for reporting. We default to faithful `sum` but expose
`how="mean"`. Watch for this whenever an aggregate is both the reported quantity
and the fit target.

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

No formal benchmark/replication exists for GeoLift's selection routine, so we
validated on **GeoLift's own example panel** (`basedata/geolift_market_data.csv`):
a sensible recommended design, and a correct **null** (joint conformal p ≈ 0.83)
realized over a no-effect post window. When there's no replication target, a
faithful port + an end-to-end null/sanity run on the source data is the
verification — state that plainly in the docs' Verification section.
