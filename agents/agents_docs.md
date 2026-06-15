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
* A set and its length are different objects. `\mathcal{T}_2` is the post-period
  *set*; `T_2 \coloneqq |\mathcal{T}_2|` is its *length* (cardinality). Likewise
  the pre-period length is `T_0 = |\mathcal{T}_1|`. Use the cardinality bars, or
  the scalar `T_0` / `T_2`, for counts; never overload a set symbol for a length.

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
* Per-period treatment effect (a scalar)
  `\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}` (it estimates
  `y_{1t}^I - y_{1t}^N`).
* The treatment effect is, by nature, a **vector** — the effect path
  `\boldsymbol{\tau} \coloneqq (\tau_t)_{t\in\mathcal{T}_2} \in \mathbb{R}^{T_2}`,
  bold because it is a vector. Name each object as what it is: a path is
  `\boldsymbol{\tau}`, a per-period effect is the scalar `\tau_t`.
* **ATT** (a scalar)
  `\widehat{\tau} \coloneqq |\mathcal{T}_2|^{-1}\sum_{t\in\mathcal{T}_2}\tau_t`.

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

### The linear factor model (the latent-variable DGP)

Most mlsynth estimators are justified by a *linear factor model* — the linear
special case of a latent-variable model — for the no-intervention outcome. Write
it following Liao, Shi & Zheng (2025, *A Relaxation Approach to Synthetic
Control*, arXiv:2508.01793) and Abadie, Diamond & Hainmueller (2010):

```
y_{jt}^N = \boldsymbol{\lambda}_j^\top \mathbf{f}_t + u_{jt},
   \qquad j \in \mathcal{N},\ t \in \mathcal{T}.
```

* `\mathbf{f}_t \in \mathbb{R}^{r}` — latent common factors at time `t` (random);
  `\boldsymbol{\lambda}_j \in \mathbb{R}^{r}` — unit `j`'s factor loadings
  (deterministic); `r` — the number of factors; `u_{jt}` — idiosyncratic error.
* Matrix form over the panel: `\mathbf{Y}^N = \mathbf{F}\boldsymbol{\Lambda}^\top
  + \mathbf{U}`, with factor matrix `\mathbf{F}\in\mathbb{R}^{T\times r}` (rows
  `\mathbf{f}_t^\top`) and loading matrix
  `\boldsymbol{\Lambda}\in\mathbb{R}^{N\times r}` (rows
  `\boldsymbol{\lambda}_j^\top`). The low-rank term is
  `\mathbf{L} \coloneqq \mathbf{F}\boldsymbol{\Lambda}^\top`.
* Two-way fixed effects are the *nested* special case: setting
  `\boldsymbol{\lambda}_j = (1, \alpha_j, \bar{\boldsymbol{\lambda}}_j^\top)^\top`
  and `\mathbf{f}_t = (\gamma_t, 1, \bar{\mathbf{f}}_t^\top)^\top` gives
  `y_{jt}^N = \alpha_j + \gamma_t + \bar{\boldsymbol{\lambda}}_j^\top
  \bar{\mathbf{f}}_t + u_{jt}` — unit effect `\alpha_j`, time effect `\gamma_t`,
  interactive remainder. "Additive FE" and "interactive FE" are one model at
  different ranks; say so rather than treating them as separate worlds.
* Scalars `r` (factors) and `K` (latent groups in the loadings, when a page uses
  them); vectors `\boldsymbol{\lambda}_j`, `\mathbf{f}_t`; matrices `\mathbf{F}`,
  `\boldsymbol{\Lambda}`, `\mathbf{L}`, `\mathbf{U}`. The factor-model error is
  `u_{jt}`; keep `\varepsilon` for a tolerance only when the page has no
  factor-model noise already named.

### Staggered adoption (multiple treated units / cohorts)

When treatment turns on at different times, prefer the compact matrix / cohort
notation of Porreca (2022, *Synthetic difference-in-differences estimation with
staggered treatment timing*) and Arkhangelsky et al. (2021) — matrix algebra for
the stacked objects, set notation for the index sets:

* Outcome matrix `\mathbf{Y} = \mathbf{L} + \mathbf{W}\circ\boldsymbol{\tau}
  + \mathbf{E}` — Hadamard product `\circ`; `\mathbf{W}` the binary
  treatment-assignment matrix (entries `w_{jt}\in\{0,1\}`); `\mathbf{L}` the
  factor term above; `\mathbf{E}` the error matrix.
* Cohorts: index a cohort by `\ell \in \mathcal{L}`, with `\ell = 0` the
  never-treated; cohort `\ell`'s effect `\widehat{\tau}_\ell` and aggregation
  weight `\mu_\ell` (e.g. the treated share
  `\mu_\ell = N_\ell / \sum_{\ell>0} N_\ell`); the aggregate
  `\widehat{\tau} = \sum_{\ell>0}\mu_\ell\,\widehat{\tau}_\ell`.
* No single `j = 1` here: index treated units by `i` over a treated set
  `\mathcal{N}_1` (`N_1` treated, `N_0` controls). Keep SDID's unit weights
  `\mathbf{w}` / `\boldsymbol{\omega}` and time weights `\boldsymbol{\lambda}`
  bold.

### Edge-case rulings (recurring gray areas)

The single-treated canon above is binding. These rulings settle the cases it
does not pin down, so pages agree *across*, not just within:

1. No universal grammar — clarity in context wins. A symbol need not have one
   global meaning; it must be unambiguous within the estimator's page.
2. `\alpha` is not reserved. Use it where standard and clear — a significance
   level, an elastic-net mix, an intercept, a cohort effect `\alpha_j`. Only
   `\tau` is hard-reserved (the treatment effect). `\delta`, `\lambda` likewise
   carry their paper-standard meaning; rename only on a genuine `\tau` collision.
3. `\coloneqq` marks the first appearance of an idea on the page. The defining
   occurrence uses `\coloneqq` (`c \coloneqq 5`); every later use is an ordinary
   equality (`7 = c + 2`). Model equations, solved/closed forms, and identities
   downstream stay `=`.
4. Optimiser vs. estimate. `\mathbf{w}^\ast` is the program / population
   optimiser; `\widehat{\mathbf{w}}` is the realized fit from data. A quantity
   that is both is written `\widehat{\mathbf{w}}` once it is the data estimate.
5. Name objects as what they are. The treatment effect is a vector
   (`\boldsymbol{\tau}`, the path); a per-period effect is the scalar `\tau_t`;
   the ATT is the scalar `\widehat{\tau}`. Bold the vector, not the scalars.
6. Potential outcomes are always Abadie `y^N` / `y^I` — even on a
   matrix-completion / Rubin-flavored page (`y_{it}(0)` / `y_{it}(1)` becomes
   `y_{it}^N` / `y_{it}^I`).
7. A set and its cardinality are different symbols: `\mathcal{T}_2` vs.
   `T_2 = |\mathcal{T}_2|`; `\mathcal{N}_1` vs. `N_1`; `\mathcal{N}_0` vs. `N_0`.

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

### Code blocks are runnable MWEs

Every ``.. code-block:: python`` on a doc page is a minimal worked example: a
reader must be able to copy it, paste it, and run it as-is. That means each
block, on its own, must

* **obtain its data** — either simulate a panel (a few lines of numpy/pandas,
  seeded with ``np.random.default_rng``) or import a real one (the
  ``raw.githubusercontent.com/jgreathouse9/mlsynth/.../basedata/<file>.csv``
  convention other pages use), never reference an undefined ``df``;
* **call the public API** with real, existing column names and arguments —
  no ``SYNDES({...})`` placeholders, no ``outcome="y"`` when the frame's column
  is ``"Y"``;
* **run and show the feature** — fit, then print/inspect the specific outputs
  the surrounding prose is describing, so the block demonstrates the thing it
  documents.

Snippets that merely list a result object's attribute surface still count: give
them a one-line data load + ``fit()`` so the variable they reference exists. If
a block cannot be made to run, it does not belong on the page. Prefer a small,
fast dataset; when an example is unavoidably slow (e.g. several sequential MIP
solves), say so in a comment rather than shrinking it into something that no
longer runs.

## Scope reminder

This is a notation/style contract, not a license to revise mathematics. When
migrating a page, change *symbols only* — leave every derivation's content
intact. When a symbol is genuinely ambiguous, match the `qdocs` blog.
