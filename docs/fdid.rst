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
  controls. When the number of donors :math:`N_0` exceeds the number of
  pre-treatment periods :math:`T_0` -- common in store/geo studies -- they
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
2. It accommodates **any number of controls**, including :math:`N_0 > T_0`.
3. There are **no overfitting concerns** -- one parameter after selection.
4. It is **computationally cheap**: a greedy :math:`O(N_0^2)` search rather
   than the :math:`2^{N_0}` subsets of the optimal procedure.
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

Index the units by :math:`j`. Let :math:`j = 1` be the single **treated**
unit and :math:`\mathcal{N} \coloneqq \{1, \ldots, N\}` all units; the
**donor pool** is :math:`\mathcal{N}_0 \coloneqq \mathcal{N} \setminus \{1\}`,
with :math:`|\mathcal{N}_0| = N_0`. A selected subset
:math:`\mathcal{D} \subseteq \mathcal{N}_0` is the **comparison group**;
write its equal-weighted average outcome as

.. math::

   \bar{y}_{\mathcal{D}, t} = \frac{1}{|\mathcal{D}|}
       \sum_{j \in \mathcal{D}} y_{jt}.

Time runs over :math:`t \in \mathcal{T} \coloneqq \{1, \ldots, T\}`; the
intervention takes effect after :math:`T_0`, giving a pre-period
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` (so
:math:`|\mathcal{T}_1| = T_0`) and a post-period
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}` (so
:math:`|\mathcal{T}_2| = T - T_0`). Potential outcomes are :math:`y_{jt}^N`
(without the intervention) and :math:`y_{jt}^I` (under it); for the treated
unit we observe :math:`y_{1t} = y_{1t}^N` on :math:`\mathcal{T}_1` and
:math:`y_{1t} = y_{1t}^I` on :math:`\mathcal{T}_2`. The estimand is the
average treatment effect on the treated,

.. math::

   \mathrm{ATT} = \frac{1}{|\mathcal{T}_2|} \sum_{t \in \mathcal{T}_2}
       \bigl(y_{1t}^I - y_{1t}^N\bigr).

.. admonition:: Notation bridge

   Li [Li2024]_ writes the treated outcome :math:`y_{tr,t}`, the selected
   control set :math:`\mathcal{N}_{co}` of size :math:`N_{co}`, its average
   :math:`\bar{y}_{\mathcal{N}_{co}, t}`, the intercept :math:`\alpha`, and
   uses :math:`T_1` / :math:`T_2` for the pre/post sample sizes (treatment
   at :math:`T_1 + 1`). In the mlsynth canon these become the treated unit
   :math:`j = 1` (hence :math:`y_{1t}`), the comparison group
   :math:`\mathcal{D} \subseteq \mathcal{N}_0` with average
   :math:`\bar{y}_{\mathcal{D}, t}`, and the single split point :math:`T_0`.

Mathematical Formulation
------------------------

The DiD model
~~~~~~~~~~~~~

For a fixed comparison group :math:`\mathcal{D}`, Forward DiD posits that
the treated unit's untreated outcome equals the group average plus a
constant level shift:

.. math::

   y_{1t}^N = \alpha + \bar{y}_{\mathcal{D}, t} + v_t,
   \qquad t = 1, \ldots, T,

with :math:`\alpha` an unknown intercept and :math:`v_t` a zero-mean,
weakly dependent error. Crucially, :math:`y_{1t}^N` and
:math:`\bar{y}_{\mathcal{D}, t}` may each be **non-stationary** (trending)
provided their *difference* is stationary -- this is the Forward DiD
parallel-trends condition. The intercept is estimated by least squares on
the pre-period,

.. math::

   \widehat{\alpha} = \frac{1}{T_0} \sum_{t \in \mathcal{T}_1}
       \bigl(y_{1t} - \bar{y}_{\mathcal{D}, t}\bigr),

so the in-sample fit and out-of-sample counterfactual are

.. math::

   \widehat{y}_{1t} = \widehat{\alpha} + \bar{y}_{\mathcal{D}, t},
   \qquad t = 1, \ldots, T,

and the per-period effect is :math:`\tau_t = y_{1t} - \widehat{y}_{1t}`,
whose post-period average is the ATT,

.. math::

   \widehat{\tau} = \frac{1}{|\mathcal{T}_2|}
       \sum_{t \in \mathcal{T}_2} \bigl(y_{1t} - \widehat{y}_{1t}\bigr).

Because the model has a single parameter, the pre-treatment fit quality is
summarized by an :math:`R^2` (identical to the adjusted :math:`R^2`, since
there is only one regressor coefficient):

.. math::

   R^2_{\mathcal{D}} = 1 - \frac{\sum_{t \in \mathcal{T}_1} \widehat{v}_t^2}
       {\sum_{t \in \mathcal{T}_1} (y_{1t} - \bar{y}_1)^2},
   \qquad \widehat{v}_t = y_{1t} - \bar{y}_{\mathcal{D}, t} - \widehat{\alpha},

where :math:`\bar{y}_1` is the treated unit's pre-period mean.

.. note::

   :math:`\tau_t` and :math:`\widehat{\tau}` are exactly the ``gap`` and
   ``att`` the result object returns (computed in
   :mod:`mlsynth.utils.effectutils`), and the pre/post fit are ``rmse_pre`` /
   ``rmse_post`` (:mod:`mlsynth.utils.fitutils`) -- the math here names the
   quantities :meth:`mlsynth.FDID.fit` reports.

The forward-selection algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maximizing :math:`R^2_{\mathcal{D}}` is equivalent to minimizing the
pre-period residual variance :math:`T_0^{-1} \sum_{t \in \mathcal{T}_1}
\widehat{v}_t^2`. Forward DiD searches over comparison groups greedily:

1. **Step 1.** For each single donor :math:`i \in \mathcal{N}_0`, form the
   one-unit comparison group and compute its pre-period :math:`R^2`. Keep
   the best single donor, :math:`\widehat{c}_1`.
2. **Step 2.** Add to :math:`\{\widehat{c}_1\}` each of the remaining
   :math:`N_0 - 1` donors in turn; keep the two-unit group with the highest
   :math:`R^2`.
3. **Step 3.** Continue, adding one donor at a time, until all
   :math:`N_0` donors are in. This yields :math:`N_0` nested groups (sizes
   :math:`1, 2, \ldots, N_0`); select the one,
   :math:`\widehat{\mathcal{D}} = \mathcal{N}_{co}`, with the largest
   :math:`R^2`.

The greedy search evaluates :math:`1 + 2 + \cdots + N_0 = N_0(N_0+1)/2`
sub-models rather than the :math:`2^{N_0}` of the exhaustive procedure (for
:math:`N_0 = 60`, that is 1,830 versus :math:`1.15 \times 10^{18}`). The
final group :math:`\widehat{\mathcal{D}}` is then plugged into the DiD formulas
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
:math:`k`, let :math:`\mathbf{Y}_{\mathcal{R}} \in \mathbb{R}^{T_0 \times
|\mathcal{R}|}` stack the *pre-period* columns of the remaining candidates
:math:`\mathcal{R}`. Every candidate :math:`(k+1)`-average -- one per
column -- is formed simultaneously by broadcasting the running pre-period
mean :math:`\mathbf{m}^{(k)}_{\mathcal{T}_1}`:

.. math::

   \mathbf{M} = \frac{k\,\mathbf{m}^{(k)}_{\mathcal{T}_1}\mathbf{1}^\top
       + \mathbf{Y}_{\mathcal{R}}}{k + 1}
   \in \mathbb{R}^{T_0 \times |\mathcal{R}|}.

In code this is the one line ``new_means = (current_mean_pre[:, None] * k +
candidates) / (k + 1)`` inside
:func:`~mlsynth.utils.fdid_helpers.estimation._select_best_donor`.

**3. The intercept** :math:`\alpha` **drops out, so scoring is pure inner
products.** This is the step that removes the per-candidate regression
entirely. Profiling out :math:`\alpha` from the DiD loss is exactly
*centering*: the fitted residual for candidate column :math:`\ell` is
:math:`\widehat{v}_t = (y_{1t} - \bar y_1) - (M_{t\ell} - \bar M_\ell)`. Writing
:math:`\widetilde{\mathbf{y}} = \mathbf{y}_{1,\mathcal{T}_1} - \bar y_1`
(precomputed once, with its norm :math:`\lVert\widetilde{\mathbf{y}}\rVert^2 =
\mathrm{ss}_{\text{tot}}`), the residual sum of squares for *all*
candidates is

.. math::

   \mathrm{SSR}_\ell = \mathrm{ss}_{\text{tot}}
       + \underbrace{\lVert \mathbf{M}_\ell - \bar M_\ell \rVert^2}_{\text{column SS}}
       - 2\,\underbrace{\widetilde{\mathbf{y}}^\top (\mathbf{M}_\ell - \bar M_\ell)}_{\text{one matrix--vector product}},
   \qquad
   R^2_\ell = 1 - \frac{\mathrm{SSR}_\ell}{\mathrm{ss}_{\text{tot}}}.

The cross term for the whole candidate set is the single matvec
:math:`\widetilde{\mathbf{y}}^\top(\mathbf{M} - \bar{\mathbf{M}})`; the column
sums of squares are one reduction. This is
:func:`~mlsynth.utils.fdid_helpers.estimation._r2_batch` -- no candidate is
ever regressed, and :math:`\alpha` is never explicitly solved during the
search (it is recovered only once, for the winning group, in
:func:`~mlsynth.utils.fdid_helpers.estimation.did_from_mean`).

Taken together, a forward step costs :math:`O(T_0 |\mathcal{R}|)` and the
entire search is :math:`O(T_0 N_0^2)`, with the inner loop expressed as a
broadcast, a matrix--vector product, a column reduction, and an
``argmax`` -- no Python-level loop over candidates and no per-candidate
solve. This is what lets the implementation run the selection over
:math:`\sim`\ 1,500 controls, and what makes the :math:`M = 5{,}000`
replication Monte Carlo in *Verification* tractable.

Assumptions
-----------

**Assumption 1 (Forward DiD parallel trends).** There exists a subset
:math:`\mathcal{D} \subseteq \mathcal{N}_0` and a constant :math:`\alpha`
such that :math:`y_{1t}^N = \alpha + \bar{y}_{\mathcal{D}, t} + v_t` for all
:math:`t`, where :math:`v_t` is a weakly dependent process with zero mean
and finite variance.

*Remark.* This says the gap between the treated unit and the selected
comparison group is **stable** across the pre- and post-periods up to a
mean-zero shock. It is strictly weaker than DiD's requirement that *all*
controls be parallel: it asks only that *some* equal-weighted subset be
parallel. Both :math:`y_{1t}^N` and :math:`\bar{y}_{\mathcal{D}, t}` may
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
be i.i.d. or the levels :math:`y_{1t}^N` to be stationary.

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
textbook DiD inference. Let :math:`\widehat{\sigma}^2_{\mathcal{D}} = T_0^{-1}
\sum_{t \in \mathcal{T}_1} \widehat{v}_t^2` be the pre-period residual
variance on the selected group. Li's Proposition 2.1 establishes

.. math::

   \left| \Pr\!\left(
       \frac{\sqrt{|\mathcal{T}_2|}\,(\widehat{\tau} - \mathrm{ATT})}
            {\widehat{\sigma}_{\mathcal{D}}} \le a \right) - \Phi(a) \right|
   \to 0, \qquad a \in \mathbb{R},

as :math:`T_0, |\mathcal{T}_2| \to \infty`, where :math:`\Phi` is the
standard-normal CDF. mlsynth reports the **finite-sample** standard error
that also carries the estimation error in :math:`\widehat{\alpha}`:

.. math::

   \mathrm{SE}(\widehat{\tau}) =
       \widehat{\sigma}_{\mathcal{D}} \sqrt{\frac{1}{T_0} + \frac{1}{|\mathcal{T}_2|}},

since :math:`\widehat{\tau} - \mathrm{ATT} = -T_0^{-1}
\sum_{\mathcal{T}_1} v_t + |\mathcal{T}_2|^{-1} \sum_{\mathcal{T}_2} v_t`
contributes one :math:`1/T_0` and one :math:`1/|\mathcal{T}_2|` variance
term. This collapses to Proposition 2.1's
:math:`\widehat{\sigma}_{\mathcal{D}} / \sqrt{|\mathcal{T}_2|}` when
:math:`T_0 \gg |\mathcal{T}_2|`. The 95% Wald interval and two-sided
p-value follow in the usual way.

Consistency of the selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Li also shows the *greedy* search recovers a *valid* comparison group.
Under Assumption 1 and the appendix's regularity conditions, with
:math:`N_0` fixed, the empirical forward selection selects (one of) the same
subset(s) the infeasible procedure based on true error variances would
select, with probability approaching one as :math:`T_0 \to \infty`
(Proposition 2.2; Proposition D.1 handles ties). Proposition D.2 extends
this to the case where :math:`N_0` grows with :math:`T_0` under a latent
group structure. Intuitively, by the law of large numbers each step's
empirical :math:`R^2` converges to its population value, so the greedy path
tracks the population-optimal path.

Example
-------

The block below fits Forward DiD on the Hsiao, Ching and Wan ([HCW]_) panel of
quarterly real-GDP growth for Hong Kong and 24 comparison economies -- the
canonical setting for the forward-selected panel-data approach ([fsPDA]_) that
Forward DiD descends from -- with Hong Kong's economic integration with
mainland China as the intervention (44 pre-treatment quarters, 17 post).

.. code-block:: python

   import pandas as pd
   from mlsynth import FDID

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/basedata/HongKong.csv"
   df = pd.read_csv(url)

   res = FDID({"df": df, "outcome": "GDP", "treat": "Integration",
               "unitid": "Country", "time": "Time", "display_graphs": True}).fit()

   # Forward DiD fit and the all-controls DiD benchmark, side by side.
   print(res.fdid.att, res.fdid.r_squared, res.fdid.selected_names)
   print(res.did.att, res.did.r_squared)

Forward DiD keeps a small, regionally sensible subset of Hong Kong's trading
partners rather than averaging all 24 economies, so it tracks Hong Kong's
pre-integration path far more closely (a higher pre-period :math:`R^2`) and
estimates the post-integration GDP-growth effect more precisely than the
all-controls DiD. The exact selected group and the cell-by-cell match to Li's
released output are in :doc:`replications/fdid`.

``res`` is an
:class:`~mlsynth.utils.fdid_helpers.structures.FDIDResults`: ``res.fdid``
and ``res.did`` are the two
:class:`~mlsynth.utils.fdid_helpers.structures.FDIDMethodFit` objects, the
convenience accessors (``res.att``, ``res.att_se``, ``res.counterfactual``,
``res.donor_weights``) forward to the Forward DiD fit, and
``res.att_by_method()`` / ``res.ci_by_method()`` return both side by side.

Verification
------------

Forward DiD is validated on two fronts. **Path A** -- mlsynth reproduces the
author's public Hong Kong GDP companion replication cell by cell (FDID ATT
:math:`0.0254`, :math:`53.84\%`, pre-period :math:`R^2 = 0.843`, 9 of 24
controls). **Path B** -- the paper's own Monte Carlo (Li 2024, Web Appendix E)
reproduces cell by cell (e.g. cell :math:`(48, 24)` yields
:math:`\mathrm{PMSE} = 0.084` against the paper's :math:`0.082`), confirming
Forward DiD pays only a small efficiency cost when ordinary DiD is valid and
wins decisively when half the controls are mismatched. See the dedicated
replication page, :doc:`replications/fdid`, for the full design, code, and
cell-by-cell tables.

Core API
--------

.. automodule:: mlsynth.estimators.fdid
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.utils.fdid_helpers.config.FDIDConfig
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
