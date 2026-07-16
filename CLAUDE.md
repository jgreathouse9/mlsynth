# CLAUDE.md — the `book` branch

Operational guidance for AI agents (and humans) working on the **`book`** branch
of `jgreathouse9/mlsynth`. This branch is **not the library** — it houses a book,
and it plays by different rules from `main`. Read this before touching anything
here.

## What this branch is

`book` is an **orphan branch** of `jgreathouse9/mlsynth`: it shares **no history**
with `main`, and never will. It exists to hold one thing —

> **Foundations of the Synthetic Control Method** — a Quarto book by Jared
> Greathouse.

The source was imported from the standalone `SCMBook` repository and seeded onto
this orphan branch in July 2026. The library (`mlsynth`) lives on `main`; the
book lives here. They are companions, not the same tree.

## Merge policy (hard rules — do not violate)

- Feature branches **may** be PR'd and merged **into `book`**.
- `book` **may NEVER** be merged into `main`.
- **Any** book-line branch **may NEVER** be merged into `main`.

The book history is a sealed line. Anything book-related targets `book`; nothing
book-related ever targets `main`. When you open a PR from a book feature branch,
its base is `book` — check the base twice before merging.

## What the book is (and is not)

Audience: **master's students and practitioners** who want to *use* synthetic
control, taught from first principles. It is the book the author wished he'd had.

This is the **master's volume**. It is deliberately bounded — the field is large
enough to fill 10,000 pages, and a book that tries to cover every estimator is
read by no one. Advanced material is not omitted; it is **reserved for a future
PhD / Volume 2**.

### Scope (agreed 2026-07)

Spine: **the ladder, plus a short "what's beyond" coda.** The book builds the
counterfactual one rung at a time, each rung motivated by the previous one's
failure:

Foundations (intro · the math you'll actually use · averages · getting started
with mlsynth) → potential outcomes & comparative case studies → control-group
mean / DiD → least squares → penalized SC → convex SC → validation & inference →
donor-pool selection → one end-to-end application → a short survey **coda** that
*names* the advanced methods without treating them.

**Reserved for Vol. 2 (out of scope here):** proximal & negative controls;
matrix completion & factor-model asymptotics (the bias-bound proofs); staggered
adoption & spillovers; the advanced inference zoo; Bayesian SC. The coda points
at these; it does not develop them.

**Inference depth:** kept deliberately light (final shape TBD). The only
candidates for inclusion beyond a basic placebo are the **VanillaSC t-test** and
the **Hirschberg-style subsampling from TSSC**; everything heavier (conformal,
debiased-t, block permutation, leave-two-out, timewise subsampling) is Vol. 2.

Sizing heuristic: keep the master's book to a semester (~180–200 pp). When in
doubt whether something belongs — if justifying it needs measure theory,
empirical-process theory, or a new identification theory, it's Vol. 2.

## The mlsynth-code caveat (read this)

The book was drafted many months ago. **The mlsynth code in the chapters targets
an older mlsynth API and will be deprecated.** Consequences for anyone working
here:

- The chapters may **not render** against the current `mlsynth`. That is
  expected — it is not a regression.
- Do **not** treat a failed render as breakage introduced by a structural or
  editorial change.
- Rewriting the chapter code against the current `mlsynth` API is a **separate
  workstream** from organizing or writing the book. Keep the two in separate PRs.

## Build & CI

- Quarto book. `_quarto.yml` (currently `_quarto.yaml`) defines the chapter set
  and their order — that file, **not** filename prefixes, controls sequencing, so
  do not rename chapters with numeric prefixes.
- `quarto render` builds to `_book/`, which is **gitignored** (regenerable build
  output — keep it out of source history).
- `.github/workflows/render.yml` triggers on push to **`book`** and renders as a
  build check. For a private book it does **not** publish; wiring it to GitHub
  Pages or an artifact upload is an open option.
- Planned: enable `execute: freeze: auto` and commit `_freeze/` once the chapter
  code is rebuilt, so renders are fast and reproducible.

## Code in this branch

Two helper modules are load-bearing; the rest are dead weight for the book:

- **Live:** `style.py` (`set_book_style` — the plot theme, used by almost every
  chapter) and `helpers.py` (validation, donor preprocessing, cross-validated
  fitting — used by the validation, donorselect, and regols chapters).
- **Utility:** `mexdataget.py` regenerates `MexicanHomicideData.csv`; it is a
  one-off data-prep script, not a render dependency.
- **Dead for the book (→ drafts):** `multiplesynth.py`, `multiplesynthhelpers.py`,
  `spotify_proximal_helpers.py`, `sparsedense.py` — nothing in the book imports
  them; they back draft/advanced material.

## Voice & conventions

- First person, warm, from first principles. Define jargon on first use; assume
  the reader wants to *learn* SC, not that they already know it.
- Teach by ladder: motivate each method by the previous rung's limitation.
- Code is interleaved (numpy / pandas / mlsynth) and honest about what is
  pedagogical versus what you'd actually write in a project.
- Sections end with problem sets.
- Citations via `book.bib` + `apa.csl`.

## Planned reorganization (not yet executed)

A structural cleanup is planned: chapters → `chapters/`, assets → `assets/`,
drafts → `_drafts/`, all data consolidated in `data/`, loose helpers folded into
a package, and `freeze` enabled. It is **structural only** — it will not touch a
single mlsynth API call — and it lands as a PR **into `book`**. Until it merges,
the layout is the flat imported tree.
