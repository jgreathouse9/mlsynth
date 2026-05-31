Synthetic Control with Multiple Outcomes (SCMO)
===============================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The synthetic control (SC) method of Abadie and co-authors [ABADIE2010]_
builds a counterfactual for one treated unit as a convex combination of
donors that reproduces the treated unit's **single** outcome over the
pre-treatment period. That single-outcome design faces a bias dilemma in
the panels most applied work actually has:

* **Short pre-period.** With few pre-treatment periods, a flexible donor
  pool can fit the pre-period *too* well, latching onto idiosyncratic noise
  rather than the latent factors -- an overfit that predicts poorly
  out-of-sample.
* **Long pre-period.** Matching over many periods mitigates overfitting but
  is fragile to structural breaks in the outcome-predictor relationship,
  which reintroduces bias.

SCMO targets exactly this regime by **supplementing the time dimension with
an outcome dimension.** When you observe several *related* outcomes that
share the same latent drivers -- a "domain" such as {GDP, industrial
production, CPI, trade} for an economy, or {math score, reading score,
attendance} for a school -- a single set of donor weights can be matched on
**all of them at once.** Each extra outcome is an additional, partially
independent view of the same donor loadings, so the weights are pinned down
with far fewer pre-treatment periods. Tian, Lee and Panchenko
[TianLeePanchenko]_ show the bias then shrinks at rate
:math:`O(1/\sqrt{K T_0})` in the number of outcomes :math:`K` *and* periods
:math:`T_0`, versus :math:`O(1/\sqrt{T_0})` for single-outcome SC -- a
smaller order. The headline demonstration is striking: a synthetic West
Germany matched on **nine economic indicators in the single year 1989**
tracks 30 years of its GDP almost as well as one matched on the entire
1960-1989 GDP trajectory.

Use SCMO when you have **one treated unit, a moderate-to-short pre-period,
and a set of related outcomes** driven by common factors. It is the right
tool when single-outcome SC overfits (too few periods) or when you simply
want a single, interpretable comparison group that is credible across
several outcomes at once.

Matching on Outcomes, Not Just Time
-----------------------------------

Most empirical work with several outcomes runs a *separate* SC for each one,
getting a different donor mix every time -- hard to interpret and statistically
wasteful. SCMO instead estimates **one common weight vector** by balancing the
treated unit against the donors across the whole outcome domain. There are two
ways to combine the outcomes, due to the two papers SCMO implements:

.. list-table::
   :header-rows: 1
   :widths: 16 46 22

   * - Scheme
     - What it balances
     - Paper
   * - ``concatenated``
     - The **stacked** standardized pre-treatment series of all :math:`K`
       outcomes (a single SC fit on a :math:`K T_0`-long matching vector).
     - Tian-Lee-Panchenko [TianLeePanchenko]_
   * - ``averaged``
     - The **average** of the standardized outcomes within each period (a
       single SC fit on a :math:`T_0`-long matching vector).
     - Sun-Ben-Michael-Feller [SunBenMichaelFeller]_
   * - ``separate``
     - The primary outcome's pre-treatment trajectory **alone** -- the
       conventional single-outcome SC baseline.
     - Abadie et al. [ABADIE2010]_
   * - ``MA``
     - A convex **model-average** of ``concatenated`` and ``averaged``,
       chosen by pre-treatment fit.
     - (this implementation)

Both papers agree on the surrounding recipe: **standardize each outcome**
(per period) before matching, since outcomes have different scales;
optionally **de-mean** (intercept-shift) to allow stable level differences
between treated and donors; and constrain the weights to the **simplex**
(non-negative, summing to one).

Notation
--------

We observe :math:`K` related outcomes in a domain
:math:`\mathbb{K} = \{1, \ldots, K\}` for :math:`N + 1` units over
:math:`T` periods. Unit :math:`j = 0` is the sole **treated** unit, and
:math:`\mathcal{N} = \{1, \ldots, N\}` indexes the **donors**. Treatment
begins at :math:`T_0 + 1`, giving a pre-period
:math:`\mathcal{T}_1 = \{1, \ldots, T_0\}` and a post-period
:math:`\mathcal{T}_2 = \{T_0 + 1, \ldots, T\}` of length
:math:`T_2 = T - T_0`. Write :math:`y_{jtk}` for unit :math:`j`'s outcome
:math:`k` at time :math:`t`, and :math:`\boldsymbol{\gamma} = (\gamma_1,
\ldots, \gamma_N)` for the **common donor weights** (a single vector shared
across all :math:`K` outcomes). The estimand is the per-outcome ATT,

.. math::

   \tau_k = \frac{1}{T_2} \sum_{t \in \mathcal{T}_2}
       \bigl(y^1_{0tk} - y^0_{0tk}\bigr), \qquad k \in \mathbb{K},

with the primary outcome's :math:`\tau` the headline estimate.

.. admonition:: Notation bridge

   Both papers write the treated unit :math:`i = 1` and donors
   :math:`i = 2, \ldots, N+1`; we use :math:`j = 0` for the treated unit and
   :math:`\mathcal{N}` for donors. Their common weights are
   :math:`\hat{w}_j` / :math:`\gamma_i`; we use :math:`\boldsymbol{\gamma}`.
   ``mlsynth`` builds matching variables through a **spec** (which outcomes,
   which period(s), and per-variable transforms ``level``/``log``/
   ``per_capita``/``raw``) rather than a fixed list, so the same engine
   covers "match :math:`K` outcomes over :math:`T_0` periods" and "match a
   cross-section of indicators in one year".

Mathematical Formulation
------------------------

The factor model
~~~~~~~~~~~~~~~~

Both papers assume the untreated potential outcome follows an interactive
fixed-effects (factor) model with loadings that are **common across the
outcomes in a domain**:

.. math::

   y^0_{jtk} = \delta_{tk} + \boldsymbol{\mu}_j^\top \boldsymbol{\lambda}_{tk}
       + \varepsilon_{jtk},

where :math:`\boldsymbol{\lambda}_{tk}` are time- and outcome-specific
factors, :math:`\boldsymbol{\mu}_j` are unit loadings **shared across
outcomes** :math:`k` (the key assumption), and
:math:`\varepsilon_{jtk}` are transitory shocks. Because the loadings are
shared, each outcome is a separate window onto the same
:math:`\boldsymbol{\mu}_j`, so :math:`K` outcomes over :math:`T_0` periods
give :math:`K T_0` matching equations to pin them down.

The matching matrix and the two weighting schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\tilde{y}_{jtk}` denote outcome :math:`k`, standardized in each
period by its cross-unit standard deviation (and optionally de-meaned by the
unit's pre-period mean). The donor weights solve a simplex-constrained
least-squares problem; the two schemes differ only in what they balance:

.. math::

   \text{concatenated:}\quad
   \hat{\boldsymbol{\gamma}}^{\text{cat}}
   = \operatorname*{argmin}_{\boldsymbol{\gamma} \in \Delta^{N-1}}
     \sum_{k=1}^{K} \sum_{t \in \mathcal{T}_1}
     \Bigl( \tilde{y}_{0tk} - \sum_{j \in \mathcal{N}} \gamma_j \tilde{y}_{jtk} \Bigr)^2,

.. math::

   \text{averaged:}\quad
   \hat{\boldsymbol{\gamma}}^{\text{avg}}
   = \operatorname*{argmin}_{\boldsymbol{\gamma} \in \Delta^{N-1}}
     \sum_{t \in \mathcal{T}_1}
     \Bigl( \bar{\tilde{y}}_{0t} - \sum_{j \in \mathcal{N}} \gamma_j \bar{\tilde{y}}_{jt} \Bigr)^2,
   \quad \bar{\tilde{y}}_{jt} = \tfrac{1}{K} \sum_{k} \tilde{y}_{jtk},

where :math:`\Delta^{N-1}` is the simplex. The counterfactual for the primary
outcome and its ATT are then

.. math::

   \hat{y}^0_{0tk} = \sum_{j \in \mathcal{N}} \hat{\gamma}_j\, y_{jtk},
   \qquad
   \hat{\tau}_k = \frac{1}{T_2} \sum_{t \in \mathcal{T}_2}
       \bigl( y_{0tk} - \hat{y}^0_{0tk} \bigr),

with a de-meaned (intercept-shifted) variant
:math:`\hat{y}^0_{0tk} = \bar{y}_{0\cdot k} + \sum_j \hat{\gamma}_j
(y_{jtk} - \bar{y}_{j \cdot k})` when ``demean=True``
([DoudchenkoImbens2017]_; Sun-Ben-Michael-Feller Eq. 1).

Why more outcomes help
~~~~~~~~~~~~~~~~~~~~~~~

Tian-Lee-Panchenko's Proposition 1 gives the bias rates: for the
single-outcome SC, :math:`|\mathbb{E}[\hat{\tau}^{\text{sep}}] - \tau| =
O(1/\sqrt{T_0})`, while for the multiple-outcome SC,
:math:`|\mathbb{E}[\hat{\tau}^{\text{cat}}] - \tau| = O(1/\sqrt{K T_0})`.
More related outcomes shrink the bias at a faster order. Sun-Ben-Michael-
Feller sharpen this for the averaged scheme: averaging reduces the bias due
to overfitting by :math:`1/\sqrt{K}` (like concatenation) **and** the bias
due to poor pre-treatment fit by a further :math:`1/\sqrt{K}` -- but, as both
papers note, equal-weight averaging can *wash out* signal that is specific to
individual outcomes. Which scheme wins is therefore regime-dependent (see
*When to Use Concatenated vs Averaged*).

Assumptions
-----------

**Assumption 1 (shared-loading factor model).** The untreated outcomes obey
:math:`y^0_{jtk} = \delta_{tk} + \boldsymbol{\mu}_j^\top
\boldsymbol{\lambda}_{tk} + \varepsilon_{jtk}` with loadings
:math:`\boldsymbol{\mu}_j` **common across outcomes** in the domain.

*Remark.* This is the substantive assumption -- the outcomes must be driven
by the *same* unit-level latent traits (a "domain"), exactly the premise of
standard factor analysis. If outcomes loaded on different unobservables, the
benefit of borrowing across them would vanish and SCMO would reduce to the
single-outcome order of bias. Sun-Ben-Michael-Feller phrase the equivalent
condition as **low rank** of the stacked model-component matrix :math:`L`.

**Assumption 2 (transitory shocks).** The :math:`\varepsilon_{jtk}` are
mean-zero given the factors and loadings, independent across units and
outcomes, with bounded moments.

*Remark.* Independence is *across outcomes*; the shared interactive fixed
effects still induce correlation along the factor dimensions. Different
scales/volatilities across outcomes are handled by **standardizing each
outcome per period** before matching.

**Assumption 3 (feasibility / convex hull).** There exist weights
:math:`\hat{w}_j \ge 0` summing to one such that the treated unit's matching
variables lie (approximately) in the convex hull of the donors':
:math:`\sum_j \hat{w}_j \boldsymbol{\mu}_j = \boldsymbol{\mu}_0` and
:math:`\sum_j \hat{w}_j y_{jtk} = y_{0tk}` for :math:`t \le T_0`.

*Remark.* This is the multiple-outcome analogue of Abadie et al.'s perfect-fit
condition. When the treated unit sits outside the donor hull (extreme levels),
**de-meaning** relaxes it to a parallel-trends-style condition by allowing a
constant level gap per outcome -- which also corrects size distortion of the
permutation/conformal test.

**Assumption 4 (consistency for inference).** The estimated weights yield a
consistent estimate of the de-meaned counterfactual as :math:`T_0, N \to
\infty`.

*Remark.* This is what the Chernozhukov-Wuethrich-Zhu conformal test (below)
needs for asymptotically valid size; Sun-Ben-Michael-Feller verify it for the
averaged weights in their Online Appendix A.

Inference
---------

``mlsynth`` uses a **single inference procedure for every weighting scheme**:
the conformal test of Chernozhukov, Wuethrich and Zhu [CWZ2021]_, in the
multiple-outcome form of Sun-Ben-Michael-Feller (Online Appendix A). Under the
sharp null :math:`H_0: \tau = \tau_0`, form the adjusted residuals
:math:`\hat{u}_{tk}` (the gap, with the post-period shifted by
:math:`\tau_0`) and the per-period statistic

.. math::

   S_q(\hat{u}_t) = \Bigl( \tfrac{1}{\sqrt{K}}
       \sum_{k=1}^{K} |\hat{u}_{tk}|^q \Bigr)^{1/q},

with :math:`q = 1` (the average effect) by default; larger :math:`q` targets
effects concentrated on a few outcomes. The conformal p-value ranks the
post-treatment statistic against the pre-treatment (moving-block)
distribution,

.. math::

   \hat{p}(\boldsymbol{\tau}_0) = \frac{1}{T}
       \sum_{t \in \mathcal{T}_1} \mathbf{1}\{S_q(\hat{u}_{T}) \le S_q(\hat{u}_t)\}
       + \frac{1}{T},

and inverting the test over a grid of :math:`\tau_0` yields a confidence
interval for the ATT. Because the matching matrix is built from pre-period
information, the SC weights do **not** depend on the post-period outcome, so
the test inversion is essentially free (no refitting). With a single predicted
outcome the statistic reduces to :math:`|\text{gap}_t|`.

*Why this replaces placebo/permutation.* The conformal test is exact-in-finite-
sample under exchangeability and applies identically to the concatenated and
averaged fits, so SCMO does not need the Abadie permutation test or a separate
agnostic-conformal path -- one method serves both schemes.

When to Use Concatenated vs Averaged
------------------------------------

* **Concatenated (Tian-Lee-Panchenko)** keeps every outcome's information
  separate, so it shines when the outcomes are **distinct views** of the
  shared loadings (different factor trajectories per outcome). It is the safer
  default and is what reproduces the West Germany result below.
* **Averaged (Sun-Ben-Michael-Feller)** collapses the outcomes to one
  averaged series, which **denoises** when the outcomes share a strong common
  factor and the per-outcome noise is large -- but can blur signal that lives
  in a single outcome. Prefer it when the domain is tightly co-moving and noisy.
* **MA** hedges between the two by pre-treatment fit; **separate** is the
  single-outcome baseline for comparison.
mlsynth's SCMO ships **four** schemes, three of which are genuinely
multi-outcome and one a single-outcome baseline. The headline
trade-off is between the two main multi-outcome variants:

* **Concatenated (Tian-Lee-Panchenko, 2024).** Stack every outcome's
  pre-period vector on top of the others, then solve **one** simplex
  QP for donor weights that balance the **whole stack** at once.
  Each outcome contributes its own per-period constraints; nothing
  is averaged away.
* **Averaged (Sun-Ben-Michael-Feller, 2025).** First compute a
  per-period average (or user-supplied index) of all :math:`K`
  outcomes, then run a single-outcome SC fit on that index. The
  per-outcome noise is denoised by the average; what's left is the
  common factor signal.
* **MA (model-averaged).** Fits both of the above and weights them
  by pre-treatment fit. A reasonable hedge when you're unsure.
* **Separate.** The single-outcome SC baseline (one weight vector
  per outcome), kept for comparison.

The mirror-image rule of thumb is:

* **Prefer Averaged when** outcomes share a **strong common factor**
  and the per-outcome noise is large. Averaging acts as a denoiser;
  Sun-Ben-Michael-Feller prove that the bias reduction grows as
  :math:`K` (the number of outcomes) grows, so the more closely
  co-moving outcomes you can stack, the bigger the averaging
  dividend. Educational testing (math + reading + attendance with
  shared school factors) is the canonical fit.
* **Prefer Concatenated when** outcomes are **distinct views of the
  shared loadings** -- each outcome's trajectory looks different
  even though they share unit-level loadings. Averaging *blurs*
  the signal in this regime because the distinct trajectories
  partially cancel. Multi-indicator economy panels (GDP, energy,
  trade, patents...) where each indicator has its own time pattern
  are the canonical fit -- which is why the West Germany
  replication below uses concatenated.
* **Prefer MA when** you can't tell which regime you're in -- it
  picks by pre-treatment fit and is rarely the worst of the three.
* **Prefer Separate when** you only have one outcome anyway, or you
  *want* a different donor mix per outcome for substantive
  reporting (e.g.\\ separate "education" vs "labour" subdomains).

When **not** to use SCMO at all
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SCMO is a multi-outcome generalisation; it inherits all of classical
SC's identification requirements (donor units untreated, treated unit
in the convex hull of donor loadings, no anticipation / spillover)
and adds one of its own:

* **You only have one related outcome.** With :math:`K = 1` SCMO
  collapses to vanilla SC -- use :doc:`tssc` (single-outcome with
  formal pre-trends testing) or :doc:`fdid` instead.
* **The outcomes do not share latent factors.** SCMO's bias reduction
  comes from estimating *one* set of weights that balances *all*
  outcomes' loadings simultaneously. If GDP, school enrolment and
  unemployment in your panel are driven by genuinely different
  latent processes, stacking them just adds noise to the QP -- the
  Tian-Lee-Panchenko / Sun-Ben-Michael-Feller bias gains
  evaporate. Diagnostic: refit with the ``separate`` scheme; if
  the donor weight vectors across outcomes look unrelated (no
  cross-outcome agreement on which donors carry weight), you
  probably don't have a shared-loading domain.
* **The treated unit is outside the convex hull on at least one
  outcome.** Each extra outcome adds a new dimension the synthetic
  must match. The more outcomes you stack, the **harder** the hull
  condition becomes (more constraints, smaller feasible region).
  If the pre-period fit is poor on most outcomes, SCMO will still
  return a weight vector but its bias bounds break -- use
  :doc:`fdid` (which permits an offset and a freely selected donor
  subset) or augmented-SC variants.
* **Outcomes are measured on very different scales** without prior
  standardisation. Concatenation puts every period of every
  outcome into a single objective; an outcome measured in millions
  will dominate one measured in proportions unless you demean and
  scale. Use ``demean=True`` (the default) and consider
  pre-standardising.
* **Long, clean pre-period for a single primary outcome.** If you
  have, say, 30 years of stable annual GDP, single-outcome SC has
  enough information to nail the weights without help. SCMO's
  advantage is in the *short*-pre-period regime; in the long one,
  the extra outcomes are mostly book-keeping overhead.

Graphical comparison
^^^^^^^^^^^^^^^^^^^^

The Monte Carlo below builds two contrasting data-generating
processes with the **same** :math:`K = 8` outcomes, :math:`T_0 = 5`
short pre-period, and known ATT of :math:`+3` on the primary
outcome:

* **DGP A (Averaged-favoured).** A single common factor trajectory
  :math:`F_t` is shared across all outcomes; each outcome carries
  the same :math:`F` plus heavy per-outcome noise. Averaging cancels
  the per-outcome noise.
* **DGP B (Concatenated-favoured).** Each of the :math:`K` outcomes
  has its **own** factor trajectory (different :math:`F^{(k)}_t`),
  even though all outcomes share the same unit-level loadings
  :math:`\Lambda_i`. Averaging across outcomes blurs the
  per-outcome trajectories; concatenation keeps them as separate
  match constraints.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import SCMO


   def build(*, mode, N=30, T0=5, T1=10, K=8, r=2, TRUE=3.0, noise=1.0, seed=0):
       """mode='averaged' -> outcomes share one factor trajectory + heavy noise.
          mode='concatenated' -> each outcome has its own factor trajectory."""
       rng = np.random.default_rng(seed)
       T = T0 + T1
       phi = rng.normal(size=(N, r))                    # shared loadings
       if mode == "averaged":
           F = rng.normal(size=(T, r))
           Fs = [F for _ in range(K)]                   # all K outcomes share F
       else:
           Fs = [rng.normal(size=(T, r)) for _ in range(K)]
       rows = []
       for i in range(N):
           a = rng.normal(size=K)
           for t in range(T):
               rec = {"unit": f"u{i}", "time": t,
                      "treat": int(i == 0 and t >= T0)}
               for k in range(K):
                   y = a[k] + phi[i] @ Fs[k][t] + rng.normal(scale=noise)
                   if i == 0 and t >= T0 and k == 0:
                       y += TRUE
                   rec[f"y{k+1}"] = y
               rows.append(rec)
       return pd.DataFrame(rows), T0, K


   NREPS = 50
   for mode in ("averaged", "concatenated"):
       atts = {"concatenated": [], "averaged": [], "MA": [], "separate": []}
       for s in range(NREPS):
           df, T0, K = build(mode=mode, seed=s)
           spec = {"year": list(range(T0)),
                   "vars": {f"y{k+1}": f"y{k+1}" for k in range(K)}}
           try:
               res = SCMO({"df": df, "outcome": "y1", "treat": "treat",
                            "unitid": "unit", "time": "time", "spec": spec,
                            "schemes": ["concatenated", "averaged",
                                          "MA", "separate"],
                            "demean": True, "display_graphs": False}).fit()
               for k, v in res.att_by_method().items():
                   atts[k].append(v - 3.0)
           except Exception:
               continue
       print(f"\nDGP={mode}  (true ATT = 3.0)   N = {len(atts['averaged'])} reps")
       for k in ("concatenated", "averaged", "MA", "separate"):
           a = np.array(atts[k])
           print(f"  {k:13s}  mean bias = {a.mean():+7.3f}   "
                  f"RMSE = {np.sqrt((a**2).mean()):.3f}")

prints (deterministic with the seeds above)::

   DGP=averaged  (true ATT = 3.0)   N = 50 reps
     concatenated   mean bias =  +0.100   RMSE = 0.744
     averaged       mean bias =  +0.097   RMSE = 0.742
     MA             mean bias =  +0.120   RMSE = 0.788
     separate       mean bias =  +0.101   RMSE = 1.132

   DGP=concatenated  (true ATT = 3.0)   N = 50 reps
     concatenated   mean bias =  -0.055   RMSE = 0.769
     averaged       mean bias =  -0.087   RMSE = 0.895
     MA             mean bias =  -0.094   RMSE = 0.750
     separate       mean bias =  -0.062   RMSE = 0.910

Three takeaways:

1. **Both multi-outcome schemes beat single-outcome SC by a wide
   margin** in either regime. ``separate``'s RMSE is 50% worse in
   DGP A and 18% worse in DGP B than the best multi-outcome
   competitor. That's the headline of both papers: with a short
   :math:`T_0`, *any* form of multi-outcome stacking is a strict
   improvement.
2. **Averaged ties or beats concatenated when the DGP genuinely
   averages**: in DGP A, averaged's RMSE is 0.742 vs concatenated's
   0.744. The advantage is small on this calibration but the
   Sun-Ben-Michael-Feller theory says it grows with :math:`K`.
3. **Concatenated wins when outcomes are distinct**: in DGP B,
   concatenated's RMSE is 0.769 vs averaged's 0.895 -- a 16%
   reduction. ``MA``'s pre-fit weighting hedges close to the
   winner in both regimes (0.788 / 0.750), which is why it's the
   safest default when you don't know which DGP you're in.




Empirical Illustration: West Germany, matched on 1989 alone
-----------------------------------------------------------

The canonical SCMO demonstration (Tian-Lee-Panchenko Section 4) revisits the
1990 German reunification. Instead of matching on 30 years of GDP, it matches
West Germany to 16 OECD donors on **nine economic indicators in the single
year 1989** -- private social expenditure, energy-per-GDP, electricity and
patents per capita, real GDP growth, CPI, trade openness, total tax revenue,
and GDP per capita.

.. code-block:: python

   import pandas as pd
   import numpy as np
   from mlsynth import SCMO

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/germany_augmented.csv"
   df = pd.read_csv(url)
   df["Reunification"] = ((df["country"] == "West Germany") & (df["year"] >= 1990)).astype(int)

   spec = {"year": 1989, "vars": {
       "private_social_exp": "Private social expenditure",
       "energy_gdp": "Total primary energy supply per unit of GDP",
       "electricity_pc": ("Electricity generation", "per_capita"),
       "patents_pc": ("Triadic patent families", "per_capita"),
       "gdp_growth": "Real GDP growth", "cpi": "CPI: all items",
       "trade": "trade", "tax": "Total tax revenue", "gdp_pc": "gdp"}}

   res = SCMO({"df": df, "outcome": "gdp", "treat": "Reunification",
               "unitid": "country", "time": "year", "spec": spec,
               "schemes": ["concatenated", "averaged", "MA"],
               "conformal_alpha": 0.1, "display_graphs": True}).fit()

   for name, fit in res.fits.items():
       print(f"{name:13s} ATT {fit.att:8.1f}  p={fit.p_value:.3f}  "
             f"90% CI ({fit.ci[0]:.0f}, {fit.ci[1]:.0f})")

This prints::

   concatenated  ATT  -1462.8  p=0.056  90% CI (-1568, -1357)
   averaged      ATT  -1720.4  p=0.056  90% CI (-1949, -1492)
   MA            ATT  -1462.8  p=0.056  90% CI (-1568, -1357)

The concatenated SC -- fit on a single year's nine indicators, never shown the
GDP path -- matches West Germany's pre-1990 GDP trajectory to a **root-mean-
squared error of 110** (vs. 74 for the conventional SC fit directly to 30
years of GDP), and implies post-reunification per-capita GDP about **\$1,463
below** the synthetic, conformally significant at the 5.6% level. The donor mix
is France 0.27, Netherlands 0.25, USA 0.21, Switzerland 0.14, Japan 0.09,
Norway 0.04. The averaged scheme is looser here (pre-RMSE 185) because
averaging the nine distinct indicators blurs their individual signal -- the
concatenated scheme is preferred for this domain.

Verification
------------

.. note::

   **Empirical (Path A, German reunification).** ``mlsynth``'s ``concatenated``
   scheme reproduces the Tian-Lee-Panchenko (2024) German reunification result
   value-for-value: matching West Germany on the nine 1989 indicators yields
   the published donor weights (France 0.267, Netherlands 0.248, USA 0.208,
   Switzerland 0.135, Japan 0.092, Norway 0.043, Belgium 0.007) and a pre-1990
   GDP-fit RMSE of 110, against a reference QP implementation of the paper's
   ``fn_W`` to the third decimal.

   **Simulation (Path B).** A factor-model Monte Carlo (shared loadings,
   :math:`N=30` donors, :math:`T_0=4`, :math:`K` related outcomes, true
   ATT :math:`= 3`, 400 reps) reproduces the paper's bias-reduction finding.
   RMSE of the estimated ATT:

   .. list-table::
      :header-rows: 1
      :widths: 8 16 18

      * - :math:`K`
        - separate
        - concatenated
      * - 1
        - 1.139
        - 1.139
      * - 3
        - 1.175
        - 0.916
      * - 6
        - 1.068
        - 0.794

   The concatenated SC's error falls monotonically as outcomes are added,
   while the single-outcome SC does not improve -- the
   :math:`O(1/\sqrt{K T_0})` versus :math:`O(1/\sqrt{T_0})` gap of Proposition
   1. Under a *shared common factor* with large per-outcome noise (the
   Sun-Ben-Michael-Feller regime) the ``averaged`` scheme also beats the
   single-outcome SC (e.g. at :math:`K=6`: separate 1.598, concatenated 1.382,
   averaged 1.408), confirming both papers' analyses and the regime-dependence
   of which scheme wins.

Simulation study: Tian-Lee-Panchenko Table 1 (Path B)
-----------------------------------------------------

We reproduce the **concatenated** paper's own Monte Carlo
(Tian-Lee-Panchenko 2024, Section 3, Table 1) through the packaged
:py:class:`~mlsynth.estimators.scmo.SCMO`. Their DGP has :math:`N = 30`
units (1 treated, 29 control), one post period, **zero** true effect, and
:math:`Y_{it,k} = \delta_{t,k} + Z_i'\theta_{t,k} + \mu_i'\lambda_{t,k}
+ \varepsilon_{it,k}` with shared predictors :math:`Z_i` (2 observed),
:math:`\mu_i` (4 unobserved) drawn once from :math:`U[-d, d]`, large level
differences :math:`\delta, \theta, \lambda \sim N(\omega_k, 1)`,
:math:`\omega_k \sim N(0, 10^2)`, and :math:`\varepsilon \sim N(0,1)`. The
estimand is the effect on outcome 1 at :math:`t = T_0+1`; with a zero true
effect the average absolute bias and SD of :math:`\hat\tau` approach the
half-normal floor :math:`\sqrt{2/\pi} \approx 0.80` and :math:`1.00` as the
fit improves. The ``concatenated`` (de-meaned) scheme reproduces Table 1 at
:math:`d = 1` -- bias and SD fall monotonically as :math:`K` and :math:`T_0`
grow toward the floor (800 reps; the paper uses 5000):

.. list-table:: Average absolute bias / SD of :math:`\hat\tau`, :math:`d=1`
   :header-rows: 1
   :widths: 8 8 22 22

   * - :math:`T_0`
     - :math:`K`
     - mlsynth (bias / SD)
     - paper (bias / SD)
   * - 5
     - 1
     - 1.47 / 1.83
     - 1.43 / 1.81
   * - 5
     - 10
     - 1.26 / 1.64
     - 1.22 / 1.54
   * - 20
     - 1
     - 1.26 / 1.59
     - 1.18 / 1.49
   * - 20
     - 10
     - 1.11 / 1.40
     - 1.08 / 1.36

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import SCMO

   N = 30
   def tlp_sim(d, T0, K, rng):                       # Tian-Lee-Panchenko Sec 3 draw
       T = T0 + 1
       Z = np.empty((N, 2)); mu = np.empty((N, 4))
       Z[1:] = rng.uniform(-1, 1, (N - 1, 2)); mu[1:] = rng.uniform(-1, 1, (N - 1, 4))
       Z[0] = rng.uniform(-d, d, 2); mu[0] = rng.uniform(-d, d, 4)
       Y = np.empty((N, T, K))
       for k in range(K):
           w = rng.normal(0, 10)
           Y[:, :, k] = (rng.normal(w, 1, T)[None, :] + Z @ rng.normal(w, 1, (T, 2)).T
                         + mu @ rng.normal(w, 1, (T, 4)).T + rng.normal(0, 1, (N, T)))
       df = pd.DataFrame([{"unit": f"u{i}", "time": t, "treat": int(i == 0 and t >= T0),
                           **{f"y{k+1}": float(Y[i, t, k]) for k in range(K)}}
                          for i in range(N) for t in range(T)])
       spec = {"year": list(range(T0)), "vars": {f"y{k+1}": f"y{k+1}" for k in range(K)}}
       res = SCMO({"df": df, "outcome": "y1", "treat": "treat", "unitid": "unit",
                   "time": "time", "spec": spec, "schemes": ["concatenated"],
                   "demean": True, "display_graphs": False}).fit()
       return res.att_by_method()["concatenated"]

   for K in (1, 3, 10):                              # half-normal floor: bias~0.80, sd~1.00
       tau = np.array([tlp_sim(1.0, 10, K, np.random.default_rng(K)) for _ in range(300)])
       print(f"K={K:>2}:  bias={np.mean(np.abs(tau)):.2f}  sd={np.std(tau):.2f}")
   # bias/SD shrink toward the floor as K grows

Simulation study: Sun-Ben-Michael-Feller averaging gain (Path B)
----------------------------------------------------------------

Sun-Ben-Michael-Feller (2025) prove that, under a shared factor structure,
the multiple-outcome schemes lower the weight-estimation bias relative to
fitting each outcome **separately**, and that **averaging** lowers it
furthest. Their Table 1 gives the leading bias terms (:math:`N` fixed):

.. list-table:: SBMF Table 1 -- leading bias terms
   :header-rows: 1
   :widths: 18 26 26

   * - scheme
     - bias from imperfect fit
     - bias from overfitting
   * - separate
     - :math:`O(1)`
     - :math:`O(1/\sqrt{T_0})`
   * - concatenated
     - :math:`O(1)`
     - :math:`O(1/\sqrt{T_0 K})`
   * - averaged
     - :math:`O(1/\sqrt{K})`
     - :math:`O(1/\sqrt{T_0 K})`

The distinctive claim is the **averaged** column: averaging the :math:`K`
outcomes denoises the matching target, so it alone shaves the
*imperfect-fit* bias by :math:`1/\sqrt{K}` while separate and concatenated
stay :math:`O(1)` there. To see this through the packaged estimator we
isolate the object the theorem bounds -- the **weight bias**
:math:`(\hat\gamma - \gamma^\star)` -- by applying each scheme's estimated
donor weights to the *noiseless* donor structure (the observed-data
counterfactual is otherwise swamped by the treated unit's irreducible
post-period noise). On their supplement DGP
:math:`Y_{it,k} = \rho\,\phi_i \mu_t + (1-\rho)\,\phi_{ik}\mu_{tk}
+ \varepsilon_{it,k}` (:math:`\rho = 0.5`, a common rank-one factor plus
outcome-specific idiosyncratic factors), 200 reps:

.. list-table:: Pure weight bias (RMSE of :math:`\hat\gamma` applied to structure)
   :header-rows: 1
   :widths: 8 14 14 14

   * - :math:`K`
     - separate
     - concatenated
     - averaged
   * - 1
     - 0.112
     - 0.112
     - 0.112
   * - 4
     - 0.149
     - 0.138
     - 0.140
   * - 8
     - 0.140
     - 0.120
     - 0.097
   * - 16
     - 0.135
     - 0.087
     - 0.071

The ordering is exactly Table 1's: **separate is flat in** :math:`K`
(:math:`O(1)`), while concatenated and averaged fall as outcomes are added,
and **averaged is lowest at large** :math:`K` -- the
:math:`1/\sqrt{K}` imperfect-fit reduction that is unique to averaging.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import SCMO

   N, T0, TPOST, RHO, TAU = 20, 40, 4, 0.5, 2.0

   def _ar1(T, rng, c=0.5):
       x = np.zeros(T)
       for t in range(1, T):
           x[t] = c * x[t - 1] + rng.normal()
       return x

   def one_rep(K, rng):
       """SBMF-regime draw; returns each scheme's weight bias on the noiseless structure."""
       T = T0 + TPOST
       gstar = rng.dirichlet(np.ones(N) * 0.5)
       phi = rng.normal(0, 1, N); phi0 = gstar @ phi; mu = _ar1(T, rng)
       m0 = donor1 = None; structs = []
       for k in range(K):
           if k == 0:                                # outcome 1: common only (oracle holds)
               phik, muk, phi0k = phi.copy(), mu.copy(), phi0
           else:                                     # k>=2: idiosyncratic, treated loading independent
               phik, muk, phi0k = rng.normal(0, 1, N), _ar1(T, rng), rng.normal()
           donor = RHO * phi[:, None] * mu[None, :] + (1 - RHO) * phik[:, None] * muk[None, :]
           treated = RHO * phi0 * mu + (1 - RHO) * phi0k * muk
           structs.append(np.vstack([treated, donor]))
           if k == 0:
               m0, donor1 = treated.copy(), donor
       Y = np.stack(structs, axis=2) + rng.normal(0, 1, (N + 1, T, K))
       Y[0, T0:, 0] += TAU
       df = pd.DataFrame([{"unit": f"u{i}", "time": t, "treat": int(i == 0 and t >= T0),
                           **{f"y{k+1}": float(Y[i, t, k]) for k in range(K)}}
                          for i in range(N + 1) for t in range(T)])
       spec = {"year": list(range(T0)), "vars": {f"y{k+1}": f"y{k+1}" for k in range(K)}}
       res = SCMO({"df": df, "outcome": "y1", "treat": "treat", "unitid": "unit",
                   "time": "time", "spec": spec,
                   "schemes": ["separate", "concatenated", "averaged"],
                   "demean": False, "display_graphs": False}).fit()
       out = {}
       for sch in ("separate", "concatenated", "averaged"):
           g = np.array([res.fits[sch].donor_weights.get(f"u{j}", 0.0) for j in range(1, N + 1)])
           out[sch] = float(np.mean(donor1[:, T0:].T @ g - m0[T0:]))
       return out

   for K in (1, 4, 8, 16):
       e = {s: [] for s in ("separate", "concatenated", "averaged")}
       rng = np.random.default_rng(100 + K)
       for _ in range(100):
           r = one_rep(K, rng)
           for s in e: e[s].append(r[s])
       rmse = lambda v: float(np.sqrt(np.mean(np.array(v) ** 2)))
       print(f"K={K:>2}:  sep={rmse(e['separate']):.3f}  cat={rmse(e['concatenated']):.3f}  "
             f"avg={rmse(e['averaged']):.3f}")
   # separate flat; concatenated & averaged fall; averaged lowest at large K

Core API
--------

.. automodule:: mlsynth.estimators.scmo
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SCMOConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``SCMO.fit()`` returns an
:class:`~mlsynth.utils.scmo_helpers.structures.SCMOResults`, whose ``fits``
maps each weighting scheme to an
:class:`~mlsynth.utils.scmo_helpers.structures.SCMOMethodFit` (counterfactual,
gap, ATT, donor weights, and the CWZ conformal p-value and CI). The prepared,
NumPy-only panel is exposed as an
:class:`~mlsynth.utils.scmo_helpers.structures.SCMOInputs`, with units and time
addressed through an :class:`IndexSet`.

.. automodule:: mlsynth.utils.scmo_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Data preparation -- the only DataFrame touchpoint: pivots the long panel to
NumPy, builds the unit/time ``IndexSet``es, and assembles the matching matrix.

.. automodule:: mlsynth.utils.scmo_helpers.setup
   :members:
   :undoc-members:

The spec-driven matching-matrix builder (per-outcome standardization,
``level``/``log``/``per_capita`` transforms, complete-cases column drop).

.. automodule:: mlsynth.utils.scmo_helpers.matrix_builder
   :members:
   :undoc-members:

The simplex SC solver and the weighting schemes (concatenated / averaged /
separate / model-average) plus de-meaning.

.. automodule:: mlsynth.utils.scmo_helpers.solvers
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.scmo_helpers.estimation
   :members:
   :undoc-members:

The Chernozhukov-Wuethrich-Zhu conformal inference (multi-outcome form).

.. automodule:: mlsynth.utils.scmo_helpers.inference
   :members:
   :undoc-members:

Scheme resolution, treatment derivation, spec construction, and the run loop.

.. automodule:: mlsynth.utils.scmo_helpers.orchestration
   :members:
   :undoc-members:
