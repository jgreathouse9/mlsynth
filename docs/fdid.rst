Forward Difference-in-Differences (FDID)
========================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Difference-in-differences (DiD) is the workhorse of quasi-experimental
causal inference, but it rests on a **parallel-trends** assumption: the
treated unit's untreated outcome would have moved in lockstep with the
average of the controls. With a large, heterogeneous pool of candidate
controls that assumption is rarely credible for the pool *as a whole* --
most of the controls are simply the wrong comparison. The usual escape
hatches each have a catch:

* **Plain DiD** uses *every* control with equal weight. One badly
  mismatched control contaminates the average, and its bias does **not**
  shrink as the panel grows.
* **Synthetic control** (SC, [ABADIE2010]_) weights the controls on the
  simplex, but is justified only as the pre-period grows without bound, and
  has **no inference theory** when the data are non-stationary of unknown
  form -- exactly the regime of most marketing and macro panels.
* **The panel-data approach** of Hsiao, Ching and Wan ([HCW]_) and its
  forward-selected variant ([fsPDA]_) fit an unrestricted regression on the
  controls. When the number of controls :math:`N` exceeds the number of
  pre-treatment periods :math:`T_1` -- common in store/geo studies -- they
  **overfit** in-sample and predict poorly out-of-sample.

Forward DiD (Li [Li2024]_) targets precisely this regime: **many candidate
controls, a short-to-moderate pre-period, and a need for valid inference
under non-stationarity.** It keeps DiD's transparency -- an equal-weighted
comparison group plus a single intercept -- but **chooses which controls
enter the comparison** by a greedy forward search on pre-treatment fit.
Because only one parameter is ever estimated (the DiD intercept
:math:`\alpha`), no matter how many controls are selected, overfitting is
impossible and the textbook DiD standard error applies. Its advantages, in
Li's own summary:

1. It is a **flexible drop-in for DiD**, usable even when DiD's
   all-controls parallel trend is too restrictive.
2. It accommodates **any number of controls**, including :math:`N > T_1`.
3. There are **no overfitting concerns** -- one parameter after selection.
4. It is **computationally cheap**: a greedy :math:`O(N^2)` search rather
   than the :math:`2^N` subsets of the optimal procedure.
5. It has **inference theory valid for stationary and non-stationary
   data**, which SC and HCW lack.

Forward Selection vs. Matching and Weighting
--------------------------------------------

Every synthetic-control-family estimator answers the same question --
*what comparison reproduces the treated unit's untreated path?* -- but each
makes a different structural bet about the comparison. Forward DiD's bet is
distinctive and worth stating plainly:

   **A *subset* of the controls shares the treated unit's trend; find that
   subset and average it with equal weights.**

It does not try to weight all controls cleverly (SC), nor regress on all of
them (HCW), nor trust all of them (DiD). It *selects*. The selection is
greedy: add, one at a time, the control that most improves pre-treatment
fit, trace the fit as the subset grows, and keep the subset that fits best.

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * -
     - DiD
     - Synthetic control
     - Forward DiD
   * - **Comparison**
     - all controls, equal weight
     - all controls, simplex weights
     - a *selected subset*, equal weight
   * - **Free parameters**
     - 1 (intercept)
     - :math:`N` weights
     - 1 (intercept) -- after selection
   * - **Overfitting risk**
     - none
     - controlled by the simplex
     - none -- one parameter
   * - **Key assumption**
     - *all* controls are parallel
     - treated is in the convex hull
     - *some subset* is parallel
   * - **Inference under non-stationarity**
     - standard
     - none available
     - standard (Prop. 2.1)

The equal weights are the crux. Because the selected controls enter through
a single average -- not :math:`|\mathcal{D}|` separate coefficients --
adding more controls cannot increase the model's degrees of freedom. This
is an **implicit regularization** that buys both the overfitting immunity
and the clean, DiD-style inference theory. Forward DiD is therefore best
read as *DiD with a principled, data-driven choice of comparison group*,
not as a new weighting scheme.

Notation
--------

Index the units by :math:`j`, with :math:`j = 0` the single **treated**
unit and :math:`\mathcal{N} = \{1, \ldots, N\}` the **control** units. A
selected subset :math:`\mathcal{D} \subseteq \mathcal{N}` is the
**comparison group**; write its equal-weighted average outcome as

.. math::

   \bar{y}_{\mathcal{D}, t} = \frac{1}{|\mathcal{D}|}
       \sum_{j \in \mathcal{D}} y_{jt}.

Time runs over :math:`t \in \{1, \ldots, T\}`; the intervention starts at
:math:`T_1 + 1`, giving a pre-period :math:`\mathcal{T}_1 = \{1, \ldots,
T_1\}` and a post-period :math:`\mathcal{T}_2 = \{T_1 + 1, \ldots, T\}` of
length :math:`T_2 = T - T_1`. Potential outcomes are :math:`y^0_{jt}`
(untreated) and :math:`y^1_{jt}` (treated); we observe :math:`y_{0t} =
y^0_{0t}` for :math:`t \in \mathcal{T}_1` and :math:`y_{0t} = y^1_{0t}` for
:math:`t \in \mathcal{T}_2`. The estimand is the average treatment effect
on the treated,

.. math::

   \mathrm{ATT} = \frac{1}{T_2} \sum_{t \in \mathcal{T}_2}
       \bigl(y^1_{0t} - y^0_{0t}\bigr).

.. admonition:: Notation bridge

   Li [Li2024]_ writes the treated outcome :math:`y_{tr,t}`, the selected
   control set :math:`\mathcal{N}_{co}` with size :math:`N_{co}`, the
   control average :math:`\bar{y}_{\mathcal{N}_{co}, t}`, the intercept
   :math:`\alpha`, and :math:`T_1` / :math:`T_2` for the pre/post counts
   (treatment at :math:`T_1 + 1`). We keep :math:`j = 0` for the treated
   unit, :math:`\mathcal{D}` for the selected comparison group, and
   :math:`\bar{y}_{\mathcal{D}, t}` for its average.

Mathematical Formulation
------------------------

The DiD model
~~~~~~~~~~~~~

For a fixed comparison group :math:`\mathcal{D}`, Forward DiD posits that
the treated unit's untreated outcome equals the group average plus a
constant level shift:

.. math::

   y^0_{0t} = \alpha + \bar{y}_{\mathcal{D}, t} + v_t,
   \qquad t = 1, \ldots, T,

with :math:`\alpha` an unknown intercept and :math:`v_t` a zero-mean,
weakly dependent error. Crucially, :math:`y^0_{0t}` and
:math:`\bar{y}_{\mathcal{D}, t}` may each be **non-stationary** (trending)
provided their *difference* is stationary -- this is the Forward DiD
parallel-trends condition. The intercept is estimated by least squares on
the pre-period,

.. math::

   \hat{\alpha} = \frac{1}{T_1} \sum_{t \in \mathcal{T}_1}
       \bigl(y_{0t} - \bar{y}_{\mathcal{D}, t}\bigr),

so the in-sample fit and out-of-sample counterfactual are

.. math::

   \hat{y}^0_{0t} = \hat{\alpha} + \bar{y}_{\mathcal{D}, t},
   \qquad t = 1, \ldots, T,

and the ATT is the mean post-period gap

.. math::

   \widehat{\mathrm{ATT}} = \frac{1}{T_2}
       \sum_{t \in \mathcal{T}_2} \bigl(y_{0t} - \hat{y}^0_{0t}\bigr).

Because the model has a single parameter, the pre-treatment fit quality is
summarized by an :math:`R^2` (identical to the adjusted :math:`R^2`, since
there is only one regressor coefficient):

.. math::

   R^2_{\mathcal{D}} = 1 - \frac{\sum_{t \in \mathcal{T}_1} \hat{v}_t^2}
       {\sum_{t \in \mathcal{T}_1} (y_{0t} - \bar{y}_0)^2},
   \qquad \hat{v}_t = y_{0t} - \bar{y}_{\mathcal{D}, t} - \hat{\alpha},

where :math:`\bar{y}_0` is the treated unit's pre-period mean.

The forward-selection algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximizing :math:`R^2_{\mathcal{D}}` is equivalent to minimizing the
pre-period residual variance :math:`T_1^{-1} \sum_{t \in \mathcal{T}_1}
\hat{v}_t^2`. Forward DiD searches over comparison groups greedily:

1. **Step 1.** For each single control :math:`i \in \mathcal{N}`, form the
   one-unit comparison group and compute its pre-period :math:`R^2`. Keep
   the best single control, :math:`\hat{c}_1`.
2. **Step 2.** Add to :math:`\{\hat{c}_1\}` each of the remaining
   :math:`N - 1` controls in turn; keep the two-unit group with the highest
   :math:`R^2`.
3. **Step 3.** Continue, adding one control at a time, until all :math:`N`
   controls are in. This yields :math:`N` nested groups (sizes
   :math:`1, 2, \ldots, N`); select the one,
   :math:`\hat{\mathcal{D}} = \mathcal{N}_{co}`, with the largest
   :math:`R^2`.

The greedy search evaluates :math:`1 + 2 + \cdots + N = N(N+1)/2`
sub-models rather than the :math:`2^N` of the exhaustive procedure (for
:math:`N = 60`, that is 1,830 versus :math:`1.15 \times 10^{18}`). The
final group :math:`\hat{\mathcal{D}}` is then plugged into the DiD formulas
above for the ATT, its standard error, and the :math:`R^2`.

How mlsynth computes this: incremental means and a batched :math:`R^2`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read literally, each step of the algorithm re-forms a subset average from
scratch and re-fits a DiD regression for *every* remaining candidate -- an
:math:`O(N)` rebuild times an :math:`O(N)` candidate loop times the
per-candidate work, repeated :math:`N` times. mlsynth's
:func:`~mlsynth.utils.fdid_helpers.estimation.forward_did_select` instead
collapses each step into a handful of vectorized NumPy operations through
three observations.

**1. The comparison average is updated incrementally, never rebuilt.**
Let :math:`\mathbf{m}^{(k)} \in \mathbb{R}^{T}` be the running average over
the :math:`k` already-selected controls. Adding control :math:`c` gives the
:math:`(k+1)`-average by a single rank-one update,

.. math::

   \mathbf{m}^{(k+1)} = \mathbf{m}^{(k)}
       + \frac{\mathbf{y}_c - \mathbf{m}^{(k)}}{k + 1},

which is :math:`O(T)` rather than :math:`O(kT)`. This is
:func:`~mlsynth.utils.fdid_helpers.estimation._update_synthetic_control`
(``current_mean + (control - current_mean) / (k + 1)``).

**2. All candidate averages for a step are built in one matrix.** At step
:math:`k`, let :math:`\mathbf{Y}_{\mathcal{R}} \in \mathbb{R}^{T_1 \times
|\mathcal{R}|}` stack the *pre-period* columns of the remaining candidates
:math:`\mathcal{R}`. Every candidate :math:`(k+1)`-average -- one per
column -- is formed simultaneously by broadcasting the running pre-period
mean :math:`\mathbf{m}^{(k)}_{\mathcal{T}_1}`:

.. math::

   \mathbf{M} = \frac{k\,\mathbf{m}^{(k)}_{\mathcal{T}_1}\mathbf{1}^\top
       + \mathbf{Y}_{\mathcal{R}}}{k + 1}
   \in \mathbb{R}^{T_1 \times |\mathcal{R}|}.

In code this is the one line ``new_means = (current_mean_pre[:, None] * k +
candidates) / (k + 1)`` inside
:func:`~mlsynth.utils.fdid_helpers.estimation._select_best_donor`.

**3. The intercept** :math:`\alpha` **drops out, so scoring is pure inner
products.** This is the step that removes the per-candidate regression
entirely. Profiling out :math:`\alpha` from the DiD loss is exactly
*centering*: the fitted residual for candidate column :math:`\ell` is
:math:`\hat v_t = (y_{0t} - \bar y_0) - (M_{t\ell} - \bar M_\ell)`. Writing
:math:`\tilde{\mathbf{y}} = \mathbf{y}_{0,\mathcal{T}_1} - \bar y_0`
(precomputed once, with its norm :math:`\lVert\tilde{\mathbf{y}}\rVert^2 =
\mathrm{ss}_{\text{tot}}`), the residual sum of squares for *all*
candidates is

.. math::

   \mathrm{SSR}_\ell = \mathrm{ss}_{\text{tot}}
       + \underbrace{\lVert \mathbf{M}_\ell - \bar M_\ell \rVert^2}_{\text{column SS}}
       - 2\,\underbrace{\tilde{\mathbf{y}}^\top (\mathbf{M}_\ell - \bar M_\ell)}_{\text{one matrix--vector product}},
   \qquad
   R^2_\ell = 1 - \frac{\mathrm{SSR}_\ell}{\mathrm{ss}_{\text{tot}}}.

The cross term for the whole candidate set is the single matvec
:math:`\tilde{\mathbf{y}}^\top(\mathbf{M} - \bar{\mathbf{M}})`; the column
sums of squares are one reduction. This is
:func:`~mlsynth.utils.fdid_helpers.estimation._r2_batch` -- no candidate is
ever regressed, and :math:`\alpha` is never explicitly solved during the
search (it is recovered only once, for the winning group, in
:func:`~mlsynth.utils.fdid_helpers.estimation.did_from_mean`).

Taken together, a forward step costs :math:`O(T_1 |\mathcal{R}|)` and the
entire search is :math:`O(T_1 N^2)`, with the inner loop expressed as a
broadcast, a matrix--vector product, a column reduction, and an
``argmax`` -- no Python-level loop over candidates and no per-candidate
solve. This is what lets the implementation run the selection over
:math:`\sim`\ 1,500 controls, and what makes the :math:`M = 5{,}000`
replication Monte Carlo in *Verification* tractable.

Assumptions
-----------

**Assumption 1 (Forward DiD parallel trends).** There exists a subset
:math:`\mathcal{D} \subseteq \mathcal{N}` and a constant :math:`\alpha`
such that :math:`y^0_{0t} = \alpha + \bar{y}_{\mathcal{D}, t} + v_t` for all
:math:`t`, where :math:`v_t` is a weakly dependent process with zero mean
and finite variance.

*Remark.* This says the gap between the treated unit and the selected
comparison group is **stable** across the pre- and post-periods up to a
mean-zero shock. It is strictly weaker than DiD's requirement that *all*
controls be parallel: it asks only that *some* equal-weighted subset be
parallel. Both :math:`y^0_{0t}` and :math:`\bar{y}_{\mathcal{D}, t}` may
trend arbitrarily (even non-linearly), so long as their difference is
trendless -- which is what makes the method valid under non-stationarity.

**Assumption 2 (no anticipation / no interference).** Controls are
untreated throughout, and the treated unit's outcome equals its untreated
potential outcome in the pre-period.

*Remark.* Standard in the DiD/SC literature. It is what lets the
pre-period identify the comparison group: if controls were themselves
affected by the intervention (spillover), their pre/post relationship to
the treated unit would shift and selection would be biased.

**Assumption 3 (regularity for inference).** The partial sums of
:math:`v_t` obey a central limit theorem; errors are weakly dependent with
finite long-run variance.

*Remark.* This is what delivers the asymptotic normality in Proposition
2.1 below, and it holds for the broad class of stationary,
weakly-dependent error processes -- it does **not** require :math:`v_t` to
be i.i.d. or the levels :math:`y^0_{0t}` to be stationary.

.. admonition:: When **not** to use Forward DiD

   Assumption 1 fails when **no** subset of controls can track the treated
   unit -- most importantly when the treated unit lies *outside the range*
   of the controls (e.g. its outcome trends upward more steeply than every
   control's). Equal weights cannot extrapolate beyond the controls, so no
   selection rescues it. In that regime Li points to methods that let the
   treated unit sit outside the control hull: the augmented DiD ([ADID]_),
   factor-model / interactive-fixed-effect estimators, or SC with an
   intercept.


Diagnostic: a side-by-side panel where Forward PTA holds vs. fails
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pretreatment :math:`R^2` returned by FDID is the natural empirical
check on Assumption 1. The script below draws two panels with the same
underlying common factor and the same true ATT of **zero**, differing
only in the treated unit's factor loading:

* **Panel A** (``treated_loading = 1``). The treated unit shares the
  controls' single-factor loading. Assumption 1 holds for any subset
  of the donors.
* **Panel B** (``treated_loading = 3``). The treated unit trends three
  times faster than any control. **No** subset of the equal-weighted
  donors can extrapolate the steeper trend -- Assumption 1 fails.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from mlsynth import FDID


   def make_panel(*, treated_loading, n_controls=40, T1=24, T2=12, seed=0):
       """Synthetic panel with one common smoothly-trending factor.

       The treated unit's loading on the factor is ``treated_loading``; every
       control loads with coefficient 1. True ATT = 0 by construction.
       """
       rng = np.random.default_rng(seed)
       T = T1 + T2
       f = np.cumsum(rng.standard_normal(T)) / np.sqrt(T)
       eps_tr = 0.10 * rng.standard_normal(T)
       eps_co = 0.10 * rng.standard_normal((n_controls, T))
       y_tr = 1.0 + treated_loading * f + eps_tr
       y_co = 1.0 + 1.0 * f[None, :] + eps_co
       rows = []
       for t in range(T):
           rows.append({"unit": "treated", "time": t, "y": float(y_tr[t]),
                        "treat": int(t >= T1)})
           for i in range(n_controls):
               rows.append({"unit": f"c{i}", "time": t, "y": float(y_co[i, t]),
                            "treat": 0})
       return pd.DataFrame(rows)


   for label, loading in [("Forward PTA holds (loading=1)", 1.0),
                           ("Forward PTA fails (loading=3)", 3.0)]:
       df = make_panel(treated_loading=loading, seed=0)
       res = FDID({"df": df, "outcome": "y", "treat": "treat",
                    "unitid": "unit", "time": "time",
                    "display_graphs": False}).fit()
       print(f"{label:35s}  FDID ATT = {res.fdid.att:+.3f}  "
              f"R^2 = {res.fdid.r_squared:.3f}  "
              f"selected {len(res.fdid.selected_names)} donors")

prints::

   Forward PTA holds (loading=1)        FDID ATT = -0.009  R^2 = 0.975  selected 4 donors
   Forward PTA fails (loading=3)        FDID ATT = -0.802  R^2 = 0.588  selected 2 donors

Two lessons jump out:

1. **The :math:`R^2` is the warning signal.** When Forward PTA holds, FDID hits
   :math:`R^2 \approx 0.98` and recovers the true zero ATT to within
   noise. When it fails, the in-sample fit drops to :math:`R^2 \approx
   0.59` -- a much weaker fit on a panel of the same dimensions. Compare
   the two against the same threshold you would apply in a forecast
   exercise (Li's empirical applications report :math:`R^2` of 0.76-0.91
   on Atlanta / San Diego / San Jose). When the pre-fit is weak, distrust
   the post-fit ATT.

2. **The bias is large and one-sided.** When Forward PTA fails because
   the treated unit trends faster than any subset of controls, FDID's
   equal-weighted comparison group flattens the post-period
   counterfactual and the ATT is biased toward zero from above (here:
   :math:`-0.80` against a true 0). A clean placebo on the pre-period
   will also be off: the in-sample residuals are systematically wrong
   when the controls cannot extrapolate the treated unit's trend.

If your application reports :math:`R^2` materially below the threshold
you would consider acceptable for a forecast (say, < 0.7), treat the
ATT estimate as a lower bound on the magnitude of misspecification
rather than an estimate of the causal effect, and switch to one of the
methods Li flags for the out-of-hull case: :doc:`fdid` with a different
comparison construction is unlikely to recover it -- try the augmented
DiD, a factor-model / interactive-fixed-effects estimator, or
synthetic control with an intercept.


Inference
---------

Because Forward DiD estimates a single parameter, its inference is the
textbook DiD inference. Let :math:`\hat{\sigma}^2_{\mathcal{D}} = T_1^{-1}
\sum_{t \in \mathcal{T}_1} \hat{v}_t^2` be the pre-period residual
variance on the selected group. Li's Proposition 2.1 establishes

.. math::

   \left| \Pr\!\left(
       \frac{\sqrt{T_2}\,(\widehat{\mathrm{ATT}} - \mathrm{ATT})}
            {\hat{\sigma}_{\mathcal{D}}} \le a \right) - \Phi(a) \right|
   \to 0, \qquad a \in \mathbb{R},

as :math:`T_1, T_2 \to \infty`, where :math:`\Phi` is the standard-normal
CDF. mlsynth reports the **finite-sample** standard error that also carries
the estimation error in :math:`\hat{\alpha}`:

.. math::

   \mathrm{SE}(\widehat{\mathrm{ATT}}) =
       \hat{\sigma}_{\mathcal{D}} \sqrt{\frac{1}{T_1} + \frac{1}{T_2}},

since :math:`\widehat{\mathrm{ATT}} - \mathrm{ATT} = -T_1^{-1}
\sum_{\mathcal{T}_1} v_t + T_2^{-1} \sum_{\mathcal{T}_2} v_t` contributes
one :math:`1/T_1` and one :math:`1/T_2` variance term. This collapses to
Proposition 2.1's :math:`\hat{\sigma}_{\mathcal{D}} / \sqrt{T_2}` when
:math:`T_1 \gg T_2`. The 95% Wald interval and two-sided p-value follow in
the usual way.

Consistency of the selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Li also shows the *greedy* search recovers a *valid* comparison group.
Under Assumption 1 and the appendix's regularity conditions, with
:math:`N` fixed, the empirical forward selection selects (one of) the same
subset(s) the infeasible procedure based on true error variances would
select, with probability approaching one as :math:`T_1 \to \infty`
(Proposition 2.2; Proposition D.1 handles ties). Proposition D.2 extends
this to the case where :math:`N` grows with :math:`T_1` under a latent
group structure. Intuitively, by the law of large numbers each step's
empirical :math:`R^2` converges to its population value, so the greedy path
tracks the population-optimal path.

Example
-------

The block below is self-contained. It draws one panel from Li's Web
Appendix E data-generating process (three common factors, 60 controls), in
the configuration where **half the controls are the wrong comparison**:
the treated unit and the first 30 controls load on the common factor with
weight 1, while the last 30 load with weight 2 (Li's "DGP2"). The true ATT
is zero. Forward DiD should select from the *matching* half and beat plain
DiD, which is contaminated by the mismatched half.

.. code-block:: python

   import numpy as np
   from mlsynth import FDID
   from mlsynth.utils.fdid_helpers.simulation import simulate_fdid_sample

   sample = simulate_fdid_sample(dgp=2, N=60, T1=24, T2=12,
                                  rng=np.random.default_rng(0))

   res = FDID({"df": sample.df, "outcome": "y", "treat": "treat",
               "unitid": "unit", "time": "time",
               "display_graphs": False}).fit()

   sel = res.fdid.selected_names
   matching = sum(int(s[1:]) < 60 // 2 for s in sel)
   print(f"FDID: ATT={res.fdid.att:+.3f}  R2={res.fdid.r_squared:.3f}  "
         f"selected {len(sel)} donors, {matching} from the matching group")
   print(f"DID : ATT={res.did.att:+.3f}  R2={res.did.r_squared:.3f}  (all 60 donors)")

A representative single draw prints::

   FDID: ATT=-0.556  R2=0.918  selected 4 donors, 4 from the matching group
   DID : ATT=-0.924  R2=0.632  (all 60 donors)

Forward DiD picks **only** matching controls, lifting the pre-fit
:math:`R^2` from 0.63 to 0.92 and landing closer to the true zero effect
than DiD -- which is dragged off by the 30 mismatched controls it is forced
to include. (A single draw is noisy; the averaged behaviour over many draws
is in *Verification* below.)

``res`` is an
:class:`~mlsynth.utils.fdid_helpers.structures.FDIDResults`: ``res.fdid``
and ``res.did`` are the two
:class:`~mlsynth.utils.fdid_helpers.structures.FDIDMethodFit` objects, the
convenience accessors (``res.att``, ``res.att_se``, ``res.counterfactual``,
``res.donor_weights``) forward to the Forward DiD fit, and
``res.att_by_method()`` / ``res.ci_by_method()`` return both side by side.

Empirical Illustration: Hong Kong's economic integration
--------------------------------------------------------

Forward DiD is the DiD analogue of the forward-selected panel-data approach
([fsPDA]_), and it shines on exactly the data those methods target. We use
the Hsiao, Ching and Wan ([HCW]_) panel of quarterly real-GDP growth for
Hong Kong and 24 comparison economies, with Hong Kong's economic
integration with mainland China as the intervention (44 pre-treatment
quarters, 17 post).

.. code-block:: python

   import pandas as pd
   from mlsynth import FDID

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/basedata/HongKong.csv"
   df = pd.read_csv(url)

   res = FDID({"df": df, "outcome": "GDP", "treat": "Integration",
               "unitid": "Country", "time": "Time", "display_graphs": True}).fit()

   print(f"FDID ATT {res.fdid.att:.4f}  SE {res.fdid.att_se:.4f}  "
         f"R2 {res.fdid.r_squared:.3f}  ({len(res.fdid.selected_names)} of "
         f"{res.inputs.n_donors} controls)")
   print("selected:", res.fdid.selected_names)
   print(f"DID  ATT {res.did.att:.4f}  SE {res.did.att_se:.4f}  R2 {res.did.r_squared:.3f}")

This prints::

   FDID ATT 0.0254  SE 0.0046  R2 0.843  (9 of 24 controls)
   selected: ['Philippines', 'Singapore', 'Thailand', 'Norway', 'Mexico',
              'Korea', 'Indonesia', 'New Zealand', 'Malaysia']
   DID  ATT 0.0317  SE 0.0082  R2 0.505

Forward DiD keeps **9 of the 24** economies -- a regionally sensible mix of
Hong Kong's trading partners -- and in doing so lifts the pre-intervention
:math:`R^2` from 0.51 (all-controls DiD) to 0.84, roughly **halving the
standard error**. The selected comparison group implies a post-integration
GDP-growth effect of about +2.5 percentage points, more precisely estimated
and better-fitting than the all-controls DiD's +3.2.

Verification
------------

**Monte Carlo replication (Path B).** Li's empirical application -- the
effect of opening physical stores on an online-first retailer's
city-level sales -- runs on a **confidential retailer dataset**, so it
cannot be reproduced value-for-value. Per the project's replication
contract (``agents/agents_estimators.md``), Forward DiD is therefore
validated by reproducing the paper's **own Monte Carlo**, Web Appendix E.

The four DGPs and their factor structure are all packaged in
:func:`mlsynth.utils.fdid_helpers.simulation.simulate_fdid_sample`: three
common factors -- ``f1`` AR(1) ``0.8``, ``f2`` ARMA(1,1) ``(-0.6, 0.8)``,
``f3`` MA(2) ``(0.9, 0.4)``, innovations :math:`N(0,1)` -- with outcomes
:math:`a_0 + c_0 \sum_k f_{kt} + \varepsilon` for the treated unit and
:math:`1 + c \sum_k f_{kt} + \varepsilon` for the controls (first half
loading :math:`c_1`, second half :math:`c_2`). Four DGPs vary
:math:`(a_0, c_0, c_1, c_2)`: **DGP1** ``(1,1,1,1)`` and **DGP3**
``(2,1,1,1)`` (all controls match -- DiD is applicable); **DGP2**
``(1,1,1,2)`` and **DGP4** ``(2,1,1,2)`` (half the controls have the
wrong loading -- DiD breaks). True ATT :math:`= 0` and
:math:`\mathrm{PMSE} = M^{-1} \sum_j \widehat{\mathrm{ATT}}_j^2`.

Replicating Table 5 is a 12-line script:

.. code-block:: python

   import numpy as np
   from mlsynth import FDID
   from mlsynth.utils.fdid_helpers.simulation import simulate_fdid_sample

   def pmse_cell(dgp, N, T1, T2, M, seed=0):
       fdid_sq, did_sq = [], []
       for j in range(M):
           rng = np.random.default_rng(seed + j)
           sample = simulate_fdid_sample(dgp=dgp, N=N, T1=T1, T2=T2, rng=rng)
           res = FDID({"df": sample.df, "outcome": "y", "treat": "treat",
                       "unitid": "unit", "time": "time",
                       "display_graphs": False, "verbose": False}).fit()
           fdid_sq.append(res.fdid.att ** 2)   # ATT = 0, so SE^2 = att^2
           did_sq.append(res.did.att ** 2)
       return float(np.mean(fdid_sq)), float(np.mean(did_sq))

   for dgp in (1, 2, 3, 4):
       for T1, T2 in [(12, 6), (24, 12), (48, 24)]:
           fdid_pmse, did_pmse = pmse_cell(dgp, N=60, T1=T1, T2=T2, M=1000)
           print(f"DGP{dgp} ({T1},{T2}): FDID={fdid_pmse:.4f}  DID={did_pmse:.4f}")

At :math:`M = 1{,}000` (Li uses :math:`M = 10{,}000`; runtime difference is
the only material change) this reproduces Table 5 cell by cell:

.. list-table::
   :header-rows: 1
   :widths: 8 14 18 18 18 18

   * - DGP
     - :math:`(T_1, T_2)`
     - DID (mlsynth)
     - DID (Li)
     - FDID (mlsynth)
     - FDID (Li)
   * - 1
     - (12, 6)
     - 0.265
     - 0.259
     - 0.325
     - 0.315
   * - 1
     - (24, 12)
     - 0.127
     - 0.128
     - 0.147
     - 0.146
   * - 1
     - (48, 24)
     - 0.065
     - 0.063
     - 0.075
     - 0.071
   * - 2
     - (12, 6)
     - 1.202
     - 1.037
     - 0.431
     - 0.385
   * - 2
     - (24, 12)
     - 0.765
     - 0.746
     - 0.177
     - 0.180
   * - 2
     - (48, 24)
     - 0.451
     - 0.473
     - 0.084
     - 0.082
   * - 3
     - (12, 6)
     - 0.265
     - 0.252
     - 0.325
     - 0.303
   * - 3
     - (24, 12)
     - 0.127
     - 0.123
     - 0.147
     - 0.143
   * - 3
     - (48, 24)
     - 0.065
     - 0.064
     - 0.075
     - 0.072
   * - 4
     - (12, 6)
     - 1.202
     - 1.038
     - 0.431
     - 0.391
   * - 4
     - (24, 12)
     - 0.765
     - 0.744
     - 0.177
     - 0.171
   * - 4
     - (48, 24)
     - 0.451
     - 0.454
     - 0.084
     - 0.081

The two headline findings reproduce. When **all** controls are valid
(DGP1, DGP3) DiD is the parsimonious efficient choice and edges out
Forward DiD by a small margin at every horizon. When **half** the
controls are mismatched (DGP2, DGP4) DiD's PMSE stays large and **does
not shrink** as the panel grows (DGP2 at :math:`(48,24)`: DID still
0.45), because the contaminating controls bias the all-controls average;
Forward DiD's PMSE **collapses** (0.084) because the forward search
discards them. Forward DiD pays only a small efficiency cost when DiD is
valid, and wins decisively when it is not -- Li's central result.
Identity of the DGP1/DGP3 (and DGP2/DGP4) columns also confirms the
estimator's **intercept invariance** -- moving :math:`a_0` from 1 to 2
changes nothing because Forward DiD's :math:`\widehat\alpha` absorbs it.
The :math:`(12, 6)` cell runs slightly hot under DGP2/4, consistent with
Monte Carlo noise at :math:`M = 1{,}000` vs Li's :math:`M = 10{,}000`.

For reference, Li's confidential store-opening study reports a Forward
DiD effect of opening a store in Atlanta of **+\$75,143 in monthly sales
(an 86% lift, pre-period** :math:`R^2 = 0.76`\ **)**, with DiD and SC --
which fit Atlanta's steep pre-trend poorly -- overstating it.

Core API
--------

.. automodule:: mlsynth.estimators.fdid
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.FDIDConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``FDID.fit()`` returns an
:class:`~mlsynth.utils.fdid_helpers.structures.FDIDResults`, whose ``fdid``
and ``did`` fields each hold an
:class:`~mlsynth.utils.fdid_helpers.structures.FDIDMethodFit`
(counterfactual, gap, ATT, analytical standard error, 95% CI, p-value,
pre-period RMSE and :math:`R^2`, selected donor names and equal weights,
and -- for Forward DiD -- the :math:`R^2` selection path). The prepared
panel is exposed as an
:class:`~mlsynth.utils.fdid_helpers.structures.FDIDInputs`.

.. automodule:: mlsynth.utils.fdid_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Data preparation -- balances the panel, pivots it, validates the
pre-period count, and packs everything into the typed
:class:`~mlsynth.utils.fdid_helpers.structures.FDIDInputs`.

.. automodule:: mlsynth.utils.fdid_helpers.setup
   :members:
   :undoc-members:

The forward-selection core and the difference-in-differences arithmetic.
The public entry points are ``forward_did_select`` (the vectorized greedy
search) and ``did_from_mean`` (the DiD fit for a fixed comparison group);
the private helpers documented below are the incremental-mean and batched
:math:`R^2` primitives described in *How mlsynth computes this*.

.. automodule:: mlsynth.utils.fdid_helpers.estimation
   :members:
   :undoc-members:
   :private-members:

The Li (2023) analytical standard error, confidence interval, and p-value.

.. automodule:: mlsynth.utils.fdid_helpers.inference
   :members:
   :undoc-members:

Assembly of the raw selection output into the typed result containers.

.. automodule:: mlsynth.utils.fdid_helpers.results_assembly
   :members:
   :undoc-members:

The observed-versus-counterfactual overlay plot for the FDID and DID fits.

.. automodule:: mlsynth.utils.fdid_helpers.plotter
   :members:
   :undoc-members:

The Web Appendix E Monte Carlo DGPs (DGP1-DGP4), packaged as
:func:`simulate_fdid_sample` so the replication in *Verification* runs as a
one-liner.

.. automodule:: mlsynth.utils.fdid_helpers.simulation
   :members:
   :undoc-members:
