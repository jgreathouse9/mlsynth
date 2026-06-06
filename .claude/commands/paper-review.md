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
2. **Classify the contribution.** Answer each, with evidence (page/section):
   - Is it a **new estimator**, an **inference/uncertainty** method, an
     **identification/partial-ID** result, or a **diagnostic**?
   - Is it **purely theory**, or does it ship an algorithm + simulation +
     (ideally) a replication package? "Incomplete draft / no repo / illustrative
     numerics only" → treat as theory, park it.
   - Does it **fill a gap** in mlsynth, or **overlap** existing estimators?
     Name the closest existing mlsynth estimator(s).
3. **Implementability.** Algorithm specified? Dependencies (pure NumPy/SciPy vs
   cvxpy vs a solver vs compiled)? Is there a reference to validate against?
   Estimate build cost (hours/days) and the hard parts.
4. **Replication path.** Which of Path A / Path B / cross-validation is feasible
   with available data? What's the headline number to match?
5. **Architecture.** New top-level estimator, or a `method=` on an existing
   dispatcher (e.g. SPILLSYNTH), or an `inference=` mode? Reuse `dataprep` +
   `BaseEstimatorResults`.
6. **When-to-use (applied lens).** State the real business/marketing/policy
   regime where a practitioner would reach for it over the alternatives.

## Output (structured, opinionated)

- **One-line verdict** + a priority call (build now / cheap-add / park / pass),
  with the *reason*.
- **Classification** (type; theory-vs-usable; gap-vs-overlap).
- **Implementability & cost** (deps, reference, hard parts, estimate).
- **Replication plan** (path, headline number, data).
- **Architecture recommendation**.
- **Honest caveats** — weak baselines, irreproducibility risks, maturity. Be
  willing to recommend *against* a build; "the problem is worth solving but this
  method's edge over a competent baseline is unproven" is a valid conclusion.

Do **not** start building. This command produces a recommendation; building
happens via `/replicate` then `/new-estimator` after a go-ahead.
