---
description: Assess a candidate paper for inclusion in mlsynth (new method? implementable? worth it?)
argument-hint: "<path-to-pdf | arxiv-url | doi> [--depth quick|deep]"
---

# Paper Review (mlsynth candidacy)

Assess whether a paper is a good addition to mlsynth, and how to build it if so.
Read the actual paper before judging — never assess from the title/abstract alone.

## Inputs

`$ARGUMENTS`: a PDF path, arXiv/DOI/URL, and optional `--depth` (default `deep`).
If a replication package or reference implementation is linked, fetch it too
(it is the ground truth for any later port).

## Procedure

1. **Read enough to judge.** Intro + method/identification + inference +
   simulation + empirical sections. For a reference repo, read the core
   function file.

2. **Scope gate (do this early — it can end the review).** mlsynth is a
   **panel** synthetic-control / DiD library: ingestion is time-indexed
   (`dataprep` needs a time column and `pre_periods`) and estimators return a
   time-series counterfactual via `BaseEstimatorResults`. Ask:
   - Is the method **panel** (units × time, pre/post), or **cross-sectional**
     (covariates, one outcome per unit — matching / propensity / program-eval)?
   - If cross-sectional, it usually **cannot ride `dataprep`/`BaseEstimatorResults`**
     and would mean standing up a *new family*, i.e. a deliberate scope
     expansion — flag it as such and lean "park" unless that expansion is
     wanted. (Lesson: a "synthetic control"-titled paper can still be
     out-of-lane — e.g. single-index/MAVE cross-sectional ATE.)

3. **Classify the contribution.** With evidence (page/section):
   - **new estimator**, **inference/uncertainty** method, **identification /
     partial-ID** result, a **diagnostic/robustness** check, or a
     **preprocessing/donor-screening** step?
   - **Theory-only vs usable.** Does it ship an algorithm + simulation +
     (ideally) a replication package? "Incomplete draft / no repo / illustrative
     numerics only" → treat as theory, park it.

4. **Gap vs overlap — GROUND IT, don't assert it.**
   - **Overlap-grep:** search the repo for the method's concepts/keywords, e.g.
     `grep -rilE "<keyword1>|<keyword2>|<author>" mlsynth --include=*.py`.
   - **Adjudicate every hit with the docstring/config**, don't trust the grep
     alone:
     `python -c "import mlsynth,inspect; print(inspect.getdoc(mlsynth.<X>)[:400])"`
     and the matching `*Config`. Distinguish three cases:
       * **capability overlap** — mlsynth already does this (lean pass; e.g. a
         second Bayesian-simplex SC when `BVSS` exists);
       * **name/acronym collision** — same abbreviation, different method (e.g.
         `MASC` = Kellogg Matching-and-SC, *not* Mediation-Analysis-SC) → it's a
         *gap*, but flag the naming conflict for any build;
       * **genuine gap** — nothing comparable (grep empty, or hits are
         unrelated).
   - Name the **closest existing estimator** from `mlsynth.__all__` either way.

5. **Implementability & cost.** Algorithm specified? Dependencies (pure
   NumPy/SciPy vs cvxpy vs a solver vs compiled; any piece with **no Python
   reference**, e.g. MAVE, raises cost)? Is there a reference to validate
   against? Estimate build cost (hours/days) and the hard parts.

6. **Replication path.** Path A (empirical on authors' data) / Path B (the
   paper's Monte Carlo) / cross-validation (match a reference implementation) —
   which is feasible with available/`basedata/` data? What's the headline number
   to match?

7. **Architecture.** New top-level estimator, a `method=` on an existing
   dispatcher (e.g. SPILLSYNTH), an `inference=`/`robustness=` mode, or a
   standalone utility? Does it reuse `dataprep` + `BaseEstimatorResults`? If it
   needs an extra input (mediator, spatial weights, covariate cube), say so.

8. **When-to-use (applied lens).** The real business/marketing/policy regime
   where a practitioner reaches for it over the alternatives.

## Output (structured, opinionated)

- **One-line verdict + priority call** — one of: *build-now* / *cheap-add* /
  *prototype-first* / *park* / *pass* — with the reason.
- **Scope gate** result (panel vs cross-sectional; rides the contract?).
- **Classification** (type; theory-vs-usable).
- **Gap vs overlap** — the grep result + docstring adjudication (overlap /
  name-collision / gap) + closest existing estimator.
- **Implementability & cost** (deps, reference, hard parts, estimate).
- **Replication plan** (path, headline number, data).
- **Architecture recommendation**.
- **Honest caveats** — weak baselines, irreproducibility risk, maturity, scope
  creep. Be willing to recommend *against* a build; "novel but out-of-lane" and
  "the problem is real but the edge over a competent baseline is unproven" are
  valid conclusions.

Do **not** start building. This command produces a recommendation; building
happens via `/replicate` then `/new-estimator` after a go-ahead.
