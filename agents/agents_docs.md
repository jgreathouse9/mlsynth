# AGENTS.md — Documentation Notation & Style (Future Direction)

> **Status: aspirational / future flag.** This file records the intended
> direction for `docs/*.rst` mathematical writing. It does **not** mandate
> an immediate rewrite of existing pages. Apply incrementally: when you
> touch a docs page for another reason, migrate it toward the conventions
> below. New estimator docs should follow them from the start.

## Motivation

mlsynth's docs have accreted **mixed mathematical notation** across pages
(different symbols for the treated unit, the donor pool, pre/post periods,
expectations, etc.). The goal is a **single, unified notation and writing
style** across every estimator page, so a reader who learns the conventions
once can read any page without re-learning symbols.

## Gold-standard reference

Adopt the notation and expository style of:

> Shi, Z., & Huang, J. (2023). "Forward-selected panel data approach for
> program evaluation." *Journal of Econometrics*, 234(2), 512–535.

Its strengths, which we want to emulate:

1. A dedicated **Notations** paragraph that fixes every symbol convention
   up front, before any model is introduced.
2. **Numbered Assumptions** (Assumption 1, 2, …), each stated formally with
   labelled sub-conditions (a)/(b)/(c), and **each paired with a Remark**
   that explains the intuition — *why* the assumption is reasonable and
   what it buys you.
3. Motivating **Examples** placed early that frame *when* the method is the
   right tool (e.g. "nearly 200 countries... how to deal with a much larger
   pool of control units?").

## Three directives for future docs work

### 1. One unified notation across all pages

Fix a canonical symbol set and use it everywhere. Proposed canon, mirroring
Shi & Huang:

- **Typography.** Plain letter `x` = scalar; bold lowercase `x` = column
  vector; bold uppercase `X` = matrix. `I` identity; `1{·}` indicator;
  `Φ(·)` standard-normal CDF; `⌈·⌉`/`⌊·⌋` ceiling/floor; `(·)⁻`
  Moore–Penrose pseudoinverse; `φ_min`/`φ_max` min/max eigenvalue;
  `‖·‖₁`, `‖·‖₂` the L1/L2 norms; `x_U := (x_j)_{j∈U}` a subvector.
- **Units.** `N+1` units indexed by `𝒩₀ := {0, 1, …, N}`; `j = 0` is the
  treated unit; `𝒩 := {1, …, N}` indexes the `N` control/donor units.
- **Potential outcomes.** `y⁰_{jt}`, `y¹_{jt}`; observed
  `y_{jt} = y⁰_{jt}(1 − d_{jt}) + y¹_{jt} d_{jt}`, with treatment dummy
  `d_{jt}`. Treatment effect `Δ_t`.
- **Time.** Series on `{−T₁, …, −1, 0, 1, …, T₂}`; intervention at `t = 0`;
  pre-period `𝒯₁ := {−T₁, …, −1}`, post-period `𝒯₂ := {1, …, T₂}`,
  `𝒯 := 𝒯₁ ∪ 𝒯₂`.
- **Expectations.** `E[x_t]` population; time-averaged population means
  `ℰ₍₁₎[x_t] := T₁⁻¹ Σ_{t∈𝒯₁} E[x_t]` (pre) and `ℰ₍₂₎` (post); sample
  means `𝔼₍₁₎[x_t] := T₁⁻¹ Σ_{t∈𝒯₁} x_t`, `𝔼₍₂₎` (post).

When a paper an estimator implements uses different symbols, translate into
this canon in the docs (a short "notation bridge" note is fine), rather than
importing yet another symbol set.

### 2. Notations block + Assumptions-with-Remarks

Every estimator docs page should carry:

- A short **Notation** subsection (or reuse a shared one) before the model.
- Identification / inference requirements written as **numbered, formal
  Assumptions**, each immediately followed by a plain-language **Remark**
  giving the intuition. Do not bury assumptions inside prose.

### 3. A "When to use this estimator" section

Each page should include, near the top, the **author's argument for when
the method is the right choice** — the regime it targets and what problem it
solves better than the alternatives — framed like Shi & Huang's motivating
Examples. State the *use case*, not just the mechanics, with reference to real life examples from business or marketing.. (This complements
the existing one-draw Monte Carlo Example requirement in
`agents_intro.md §5`.)

## Scope reminder

This is a style/notation unification effort, not a mathematical revision.
Do not change the *content* of any derivation when migrating notation. When
in doubt about the canonical symbol for a concept, prefer the Shi & Huang
choice above.
