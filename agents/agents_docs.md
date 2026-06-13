# AGENTS.md — Documentation Math Notation (canonical & binding)

> **Status: binding.** This is the single source of truth for mathematical
> notation in `docs/*.rst`. It is **mandatory** for every new estimator page
> and for any page you touch. Existing pages migrate **incrementally** — when
> you edit a page for any reason, convert its symbols to this canon (symbols
> only; never alter the content of a derivation). An agent should be able to
> read any page without re-learning symbols.

## Why this exists

mlsynth's docs accreted **mixed notation** across pages — different symbols for
the treated unit, the donor pool, pre/post periods, the treatment effect — so
each page made the reader start over. The fix is one symbol set, used
everywhere. The canon below is **derived from the author's own writing** in the
`qdocs` blog (<https://github.com/jgreathouse9/jgreathouse9.github.io>, e.g.
`fscm.qmd`, `shc.qmd`, `scexp.qmd`), so the docs speak in the same voice as the
blog posts that motivate them.

Two references, two roles:

* **Symbol canon → the `qdocs` blog** (the table below). When in doubt about a
  symbol, match the blog.
* **Expository structure → Shi & Huang (2023),** *"Forward-selected panel data
  approach for program evaluation,"* J. Econometrics 234(2): a dedicated
  Notation block up front, numbered Assumptions each paired with a Remark, and
  motivating examples that frame *when* the method is the right tool.

## The notation canon

### Typography

| Object | Convention | Example |
| --- | --- | --- |
| scalar | plain lowercase | `t`, `N`, `T_0`, `\lambda` |
| vector | **bold lowercase** | `\mathbf{y}`, `\mathbf{w}` |
| matrix | **bold uppercase** | `\mathbf{Y}_0`, `\mathbf{X}` |
| set | calligraphic uppercase | `\mathcal{N}`, `\mathcal{T}` |
| estimate / fitted value | hat | `\widehat{y}_{1t}`, `\widehat{\tau}` |
| optimiser | star superscript | `\mathbf{w}^\ast` |
| transformed series (smoothed, de-meaned, …) | tilde | `\widetilde{\mathbf{y}}` |
| "is defined as" | `\coloneqq` (`:=`) | `\mathcal{T} \coloneqq \{1,\dots,T\}` |

Operators and symbols: norms `\|\cdot\|_1`, `\|\cdot\|_2`, squared
`\|\cdot\|_2^2`; non-negative reals `\mathbb{R}_{\ge 0}`; unit simplex
`\Delta^{N_0} \coloneqq \{\mathbf{w}\in\mathbb{R}_{\ge 0}^{N_0} : \|\mathbf{w}\|_1 = 1\}`;
`\operatorname*{argmin}` / `\operatorname*{argmax}`; indicator
`\mathbf{1}\{\cdot\}`; standard-normal CDF `\Phi(\cdot)`; expectation
`\mathbb{E}[\cdot]`; Moore–Penrose pseudoinverse `(\cdot)^{+}`.

### Prose emphasis — no bold

**Bold is for mathematics only.** Outside of math (`\mathbf{}` vectors and
matrices), never use RST bold (`**...**`) in the prose of a doc page or in
README prose — the author does not write in bold, so a page full of `**…**`
reads as not-them. This is binding for every page you touch: when migrating a
page, strip non-math bold along with the symbol conversion.

* For emphasis, choose the *word*, not the weight. If a sentence needs bold to
  land, rewrite the sentence.
* For a term of art or an API name use a ``literal`` (double-backtick) role;
  use *italic* (single asterisk) only sparingly for genuine emphasis on first
  definition.
* Section headings, and the cells/headers of `list-table` / grid tables, are
  fine — the rule is specifically about `**bold**` emphasis runs in sentences.

(The heading above is itself bold only because this is an agent instruction
file, not a rendered doc page; do not copy that style into `docs/`.)

### Units

* `\mathcal{N} \coloneqq \{1,\dots,N\}` — all `N` units.
* **The treated unit is `j = 1`.** (State it once: "Let `j = 1` denote the
  treated unit.")
* Donor pool `\mathcal{N}_0 \coloneqq \mathcal{N}\setminus\{1\}`, with
  cardinality `N_0`. Generic unit index `j`.

### Time

* `t \in \mathcal{T} \coloneqq \{1,\dots,T\}` — **1-indexed**; the intervention
  takes effect **after** period `T_0`.
* Pre-period `\mathcal{T}_1 \coloneqq \{t\in\mathcal{T} : t \le T_0\}`,
  so `|\mathcal{T}_1| = T_0`.
* Post-period `\mathcal{T}_2 \coloneqq \{t\in\mathcal{T} : t > T_0\}`,
  so `|\mathcal{T}_2| = T - T_0`.

  (Do **not** use a `t = 0`-centered axis. `T_0` is the canonical split point.)

### Outcomes  (subscript order: **unit, then time** — `y_{jt}`)

* Observed scalar `y_{jt}`; treated series
  `\mathbf{y}_1 = (y_{11},\dots,y_{1T})^\top \in \mathbb{R}^T`; donor series
  `\mathbf{y}_j \in \mathbb{R}^T`.
* Donor matrix `\mathbf{Y}_0 \coloneqq [\mathbf{y}_j]_{j\in\mathcal{N}_0}
  \in \mathbb{R}^{T\times N_0}` (one column per donor).
* Potential outcomes (Abadie superscripts): `y_{jt}^N` without the
  intervention, `y_{jt}^I` under it; the observed outcome is
  `y_{jt} = y_{jt}^N + \bigl(y_{jt}^I - y_{jt}^N\bigr)\,d_{jt}` with treatment
  dummy `d_{jt}`.

### Weights, counterfactual, and effects

* Donor weights `\mathbf{w} \in \mathbb{R}^{N_0}` (on the simplex `\Delta^{N_0}`
  when constrained); optimiser `\mathbf{w}^\ast`.
* Counterfactual / synthetic estimate
  `\widehat{\mathbf{y}}_1 \coloneqq \mathbf{Y}_0\,\mathbf{w}^\ast`, with entries
  `\widehat{y}_{1t}`.
* Per-period treatment effect
  `\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}` (it estimates
  `y_{1t}^I - y_{1t}^N`).
* **ATT** `\widehat{\tau} \coloneqq |\mathcal{T}_2|^{-1}\sum_{t\in\mathcal{T}_2}\tau_t`.

> **Docs ↔ code share one vocabulary.** `\tau_t` is exactly the `gap` and
> `\widehat{\tau}` the `att` computed in `utils/effectutils.py` /
> `results_helpers.build_effect_submodels`; pre/post fit are `rmse_pre` /
> `rmse_post` (`utils/fitutils.py`). Write the math so the symbols name the
> quantities the result object returns.

### Weight constraints and penalties (the SC family)

Most mlsynth estimators pick `\mathbf{w}` by a constrained / penalized program.
Write it in this common shape so every weighting page reads alike:

```
\mathbf{w}^\ast \in \operatorname*{argmin}_{\mathbf{w}\in\mathcal{C}}
   \; \mathcal{L}(\mathbf{Y}_0,\mathbf{y}_1,\mathbf{w}) + \mathcal{P}(\mathbf{w})
   \quad\text{s.t.}\quad \mathcal{B}(\mathbf{Y}_0,\mathbf{y}_1,\mathbf{w}) \le \boldsymbol{\delta}
```

* fit loss `\mathcal{L}`, penalty `\mathcal{P}`, balance map `\mathcal{B}`;
* feasible set `\mathcal{C}` with descriptive subscripts —
  `\mathcal{C}_{\text{simplex}}`, `\mathcal{C}_{\text{nonneg}}`,
  `\mathcal{C}_{\text{affine}}`, `\mathcal{C}_{\text{unit}}`,
  `\mathcal{C}_{\text{unconstrained}}`;
* regularization strength `\lambda \ge 0`, elastic-net mix `\alpha \in [0,1]`,
  norm order `q \in \{2,\infty\}` in `\|\mathbf{w}\|_q`;
* optional unpenalized intercept / level shift `b_0`, balance slack `\gamma`.

> **Reserve `\tau` for the treatment effect.** `\tau_t` and `\widehat{\tau}`
> always denote the effect. For a relaxation / balance tolerance use
> `\boldsymbol{\delta}` (or `\varepsilon`), never `\tau` — this is the one clash
> that recurs across the source posts, so fix it here.

### The bridge rule

When the implemented paper uses other symbols — Shi & Huang's `j = 0` treated
unit and `t = 0`-centered time; Li (2024)'s `y_{tr,t}`; Abadie's `\mathbf{Y}_1`
for the treated matrix — **translate into the canon** and, if helpful, add a
one-line *notation bridge* ("Li's `y_{tr,t}` is our `y_{1t}`"). Never carry a
second symbol set into the page.

## Page requirements

Every estimator page carries, in this order near the top:

1. **"When to use this estimator"** — the author's argument for the regime it
   targets and what it solves better than the alternatives, with a concrete
   business/marketing example (cf. the one-draw Monte Carlo Example in
   `agents_intro.md §5`).
2. A **Notation** block fixing the page's symbols (reuse the canon; only define
   what is genuinely page-specific) before any model is introduced.
3. **Numbered Assumptions** (Assumption 1, 2, …) with labelled sub-conditions
   (a)/(b)/(c), **each immediately followed by a Remark** giving the intuition.
   Do not bury assumptions in prose.

## Scope reminder

This is a notation/style contract, not a license to revise mathematics. When
migrating a page, change *symbols only* — leave every derivation's content
intact. When a symbol is genuinely ambiguous, match the `qdocs` blog.
