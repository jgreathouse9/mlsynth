Two-Step Synthetic Control
==========================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The synthetic control (SC) method of Abadie and Gardeazabal rests on an
identifying assumption that can only be *partially* checked: after
treatment, the synthetic control should track the treated unit's
untreated outcome. The testable part is the **SC pre-trends assumption**
-- that the synthetic control tracks the treated unit *before*
treatment. In practice this is usually verified by eyeballing a plot,
which is informal and easy to get wrong; imposing SC's restrictions when
they do not hold can bias the ATT, sometimes with the wrong sign.

Use TSSC, due to Li and Shankar [TSSC]_, when you have a **single treated
unit** observed over a panel whose outcomes may follow **nonstationary,
nonlinear trends** (sales, market share, macro series), and you want to
decide -- *formally*, not visually -- whether SC's restrictions are
appropriate or whether you should relax them. TSSC

1. runs a formal hypothesis test of the SC pre-trends assumption, and
2. recommends the member of the SC class that best balances the dual
   goal of **reducing bias** (relax restrictions that are violated) and
   **increasing efficiency** (keep restrictions that hold, since each
   correct restriction shrinks the estimator's variance).

If the SC restrictions hold, TSSC keeps the efficient SC estimator. If
they are violated, TSSC backs off to the least-restrictive modified
variant needed -- no more, no less.

.. note::

   TSSC is the only estimator in ``mlsynth`` that does not rely on a
   machine-learning step; it is a sequence of constrained least-squares
   fits plus a subsampling test.

Notation
--------

We index units by :math:`j`, with :math:`j = 1` the sole **treated** unit
and :math:`j = 2, \ldots, N` the **control** (donor) units. Time runs
over :math:`t`, partitioned by the intervention into a pre-treatment
window :math:`\mathcal{T}_1` of length :math:`T_1` and a post-treatment
window :math:`\mathcal{T}_2` of length :math:`T_2`, with
:math:`T = T_1 + T_2`. Let :math:`y_{jt}^0` and :math:`y_{jt}^1` denote
potential outcomes without and with treatment; we observe

.. math::

   y_{jt} =
   \begin{cases}
       y_{jt}^0 & j \in \{2, \ldots, N\}, \\
       y_{1t}^0 & j = 1,\ t \in \mathcal{T}_1, \\
       y_{1t}^1 & j = 1,\ t \in \mathcal{T}_2.
   \end{cases}

The regression design vector is :math:`x_t = (1, y_{2t}, \ldots,
y_{Nt})'`, so the coefficient vector :math:`\beta = (\beta_1, \beta_2,
\ldots, \beta_N)'` has :math:`\beta_1` as the **intercept** and
:math:`\beta_2, \ldots, \beta_N` as the **donor slopes**. We write
:math:`\mathbf{1}_L` and :math:`\mathbf{0}_L` for the :math:`L`-vectors of
ones and zeros, and :math:`\hat{\beta}_{\mathrm{MSC},T_1}` for the
benchmark MSC(c) estimate on the pre-treatment sample. The estimand is the
average treatment effect on the treated,

.. math::

   \mathrm{ATT} = \frac{1}{T_2} \sum_{t \in \mathcal{T}_2}
       \bigl(y_{1t} - \hat{y}_{1t}^0\bigr),
   \qquad \hat{y}_{1t}^0 = x_t' \hat{\beta},

where :math:`\hat{y}_{1t}^0` is the counterfactual untreated outcome.

The Class of Synthetic Control Methods
--------------------------------------

Each method fits :math:`y_{1t} = x_t' \beta + e_{1t}` on the
pre-treatment window by minimizing :math:`\sum_{t \in \mathcal{T}_1}
(y_{1t} - x_t'\beta)^2` subject to a subset of three restrictions:
**(1)** a zero intercept :math:`\beta_1 = 0`; **(2)** the donor weights
sum to one :math:`\sum_{j=2}^N \beta_j = 1`; **(3)** the donor weights are
non-negative :math:`\beta_j \ge 0`. The four members differ only in which
restrictions they impose:

.. list-table::
   :header-rows: 1
   :widths: 12 50 18

   * - Method
     - Restrictions
     - Intercept
   * - **SC**
     - (1), (2), (3) -- the canonical Abadie estimator
     - none
   * - **MSCa**
     - (2), (3) -- weights sum to one
     - free
   * - **MSCb**
     - (1), (3) -- no adding-up
     - none
   * - **MSCc**
     - (3) only -- the most flexible benchmark
     - free

Geometrically, SC projects the treated unit onto the *convex hull* of the
donors; MSCb and MSCc project onto a *convex cone* (non-negative weights
that need not sum to one), with MSCc additionally allowing a vertical
shift via the intercept. The flexible variants are appropriate when the
treated unit sits on a steeper trend than its donors, but that flexibility
costs efficiency -- which is precisely the trade-off Step 1 adjudicates.

Failure modes of the convex hull: when each restriction matters
---------------------------------------------------------------

The three SC restrictions -- zero intercept, weights sum to one,
weights non-negative -- together force the synthetic counterfactual
to lie strictly inside the **convex hull** of the donors' pre-period
paths. Geometrically that's a useful prior when the treated unit
*does* look like a weighted average of the donors. It is a
*catastrophic* prior when the treated unit doesn't. Two simple
violations cover almost every empirical case where vanilla SC goes
wrong:

* **Level shift -- treated above (or below) every donor's level.**
  The zero-intercept restriction pins the synthetic to a convex
  combination of the donors, which can never escape the donors'
  range. If a national chain's flagship store outsells every donor
  store by a constant multiple, SC's synthetic has to lie between
  the donors and *cannot* reach the treated level. The pre-period
  RMSE blows up; the post-period ATT inherits the same gap as a
  spurious treatment effect.

  *Fix:* add a free intercept. That is **MSCa** (keeps sum-to-one
  and non-negativity, drops the zero-intercept). An intercept can
  absorb an arbitrary vertical shift without inflating the donor
  weights.

* **Steeper trend -- treated growing faster than the fastest
  donor.** The sum-to-one restriction forces the synthetic to track
  a *weighted average* of donor trajectories, whose slope is bounded
  above by the maximum donor slope. If the treated unit's pre-trend
  is steeper than any donor's, the SC fit is uniformly behind in
  the pre-period -- and again the post-period gap is mostly
  miscalibration, not treatment effect.

  *Fix:* drop sum-to-one. That is **MSCb** (keeps the zero
  intercept and non-negativity, allows the weights to sum above
  one). Weights summing above one act as a slope amplifier:
  :math:`2 \cdot \text{donor with slope 1}` reproduces slope 2.

* **Both at once.** Common in practice -- the treated unit is
  bigger in level *and* steeper in trend than every donor. Neither
  MSCa (slope still bounded) nor MSCb (no intercept) is enough.

  *Fix:* relax both. That is **MSCc**, the most flexible variant
  in the family, retaining only non-negativity.

Side-by-side example
^^^^^^^^^^^^^^^^^^^^

The script below builds four panels with the same donor pool. Only
the treated unit changes between panels, isolating the failure mode
in question. The true ATT is zero in every panel (no treatment
effect was injected); the pre-RMSE and post-period ATT estimates
under each of the four variants tell the user which restriction is
hurting the fit, and the TSSC recommendation picks the right
variant automatically.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from mlsynth import TSSC


   def panel(treated, donors, T1=20):
       T = treated.size
       rows = [
           {"unit": "T", "t": t, "y": float(treated[t]),
            "treat": int(t >= T1)}
           for t in range(T)
       ]
       for i, d in enumerate(donors):
           rows.extend(
               {"unit": f"d{i}", "t": t, "y": float(d[t]), "treat": 0}
               for t in range(T)
           )
       return pd.DataFrame(rows)


   def report(label, df):
       res = TSSC({
           "df": df, "outcome": "y", "treat": "treat",
           "unitid": "unit", "time": "t",
           "display_graphs": False, "seed": 0,
       }).fit()
       print(f"\n{label}")
       print(f"  TSSC recommended: {res.recommended_method}")
       for m in ("SC", "MSCa", "MSCb", "MSCc"):
           v = res.variants[m]
           ic = "" if v.intercept is None else f"  intercept={v.intercept:+.2f}"
           print(f"    {m:5}  ATT = {v.att:+7.3f}   pre-RMSE = {v.rmse_pre:5.3f}{ic}")


   rng = np.random.default_rng(0)
   T, T1 = 30, 20
   t = np.arange(T)

   # Eight donors on a shallow common trend.
   donors = np.array([1.0 + 0.05 * t + 0.3 * rng.standard_normal(T)
                       for _ in range(8)])

   # (A) Treated lies inside the donor hull.
   treated_A = donors.mean(axis=0) + 0.10 * rng.standard_normal(T)
   report("(A) Inside the hull -> SC", panel(treated_A, donors))

   # (B) Treated is uniformly +8 above every donor.
   report("(B) Level shift -> MSCa", panel(treated_A + 8.0, donors))

   # (C) Treated trends 4x faster than any donor.
   treated_C = 1.0 + 0.20 * t + 0.3 * rng.standard_normal(T)
   report("(C) Steeper slope -> MSCb", panel(treated_C, donors))

   # (D) Treated is both shifted and steeper.
   treated_D = 5.0 + 0.20 * t + 0.3 * rng.standard_normal(T)
   report("(D) Shifted AND steeper -> MSCc", panel(treated_D, donors))

prints (deterministic with the seed above)::

   (A) Inside the hull -> SC
     TSSC recommended: SC
       SC     ATT =  -0.059   pre-RMSE = 0.079
       MSCa   ATT =  -0.147   pre-RMSE = 0.063  intercept=+0.06
       MSCb   ATT =  -0.189   pre-RMSE = 0.062
       MSCc   ATT =  -0.184   pre-RMSE = 0.062  intercept=+0.01

   (B) Level shift -> MSCa
     TSSC recommended: MSCa
       SC     ATT =  +7.973   pre-RMSE = 7.897
       MSCa   ATT =  -0.147   pre-RMSE = 0.063  intercept=+8.06
       MSCb   ATT =  -3.761   pre-RMSE = 1.415
       MSCc   ATT =  -0.184   pre-RMSE = 0.062  intercept=+8.01

   (C) Steeper slope -> MSCb
     TSSC recommended: MSCb
       SC     ATT =  +3.669   pre-RMSE = 1.396
       MSCa   ATT =  +2.430   pre-RMSE = 0.721  intercept=+1.23
       MSCb   ATT =  +1.720   pre-RMSE = 0.493
       MSCc   ATT =  +1.720   pre-RMSE = 0.493  intercept=-0.00

   (D) Shifted AND steeper -> MSCc
     TSSC recommended: MSCc
       SC     ATT =  +7.719   pre-RMSE = 5.303
       MSCa   ATT =  +2.408   pre-RMSE = 0.804  intercept=+5.30
       MSCb   ATT =  +0.102   pre-RMSE = 0.434
       MSCc   ATT =  +0.750   pre-RMSE = 0.332  intercept=+1.71

How to read the table:

* In **(A)**, no restriction is binding. All four variants attain
  similar pre-RMSE (~0.06-0.08), and TSSC's first-step test fails to
  reject the joint null -- so the recommendation is the most
  restrictive, most efficient member: SC.
* In **(B)**, dropping the zero intercept is *all* you need. MSCa
  (intercept ~+8) and MSCc (intercept ~+8) both hit a pre-RMSE of
  ~0.06. SC's pre-RMSE balloons to 7.9; MSCb tries to fake the level
  shift by inflating the weights past 1 and only partially succeeds
  (pre-RMSE 1.4). TSSC's test rejects the joint null but fails to
  reject sum-to-one, so it recommends MSCa.
* In **(C)**, dropping sum-to-one is what's needed. The treated's
  slope (0.20) is 4x the donors' (0.05); only MSCb / MSCc can
  amplify weights to chase the steeper trajectory. The intercept in
  MSCc is essentially zero -- the level shift wasn't the problem.
  TSSC tests through to MSCb.
* In **(D)** both restrictions bite. MSCa misses on slope, MSCb
  misses on level; only MSCc (free intercept *and* weights unbound
  above 1) cleans both. The decision path runs all the way to
  the leaf of the flowchart.

What TSSC does is automate this diagnosis: instead of eyeballing the
pre-period fit and guessing which restriction is the problem, Step 1
runs a formal sub-sampling test that rejects exactly the
restriction(s) the data refuse, and Step 2 then applies the most
restrictive variant the test couldn't reject -- buying back efficiency
wherever the data say SC's restrictions are correct.

Assumptions
-----------

The theory is developed for a nonstationary, nonlinear-trend factor model
:math:`y_{jt}^0 = c_j + d_j f_t + u_{jt}`, where :math:`f_t` is a common
trend of unknown functional form, :math:`c_j` an intercept, :math:`d_j` a
factor loading, and :math:`u_{jt}` an idiosyncratic error.

**Assumption 1 (data-generating process).** The idiosyncratic errors
:math:`u_{jt}` are zero-mean, serially uncorrelated, stationary with a
finite fourth moment and uncorrelated with the common factor; the
projection error :math:`e_{1t} = y_{1t}^0 - x_t'\beta_0` is a zero-mean,
finite-variance stationary process obeying a central limit theorem; and
:math:`T_2/T_1 \to \eta` for a finite :math:`\eta \ge 0`.

*Remark.* This says the only nonstationarity in the panel comes through
the shared trend :math:`f_t` -- the unit-specific noise is well behaved.
It is what lets a linear combination of donors soak up the treated unit's
trend and leave a stationary residual, which is exactly the parallel-
trends condition the test targets.

**Assumption 2 (trend and design regularity).** The common trend grows no
faster than a leading term :math:`g(t)` (e.g. a polynomial or :math:`\log
t`, but **not** :math:`e^t`), and the pre-treatment second-moment matrix
of the donor outcomes converges to a positive-definite limit.

*Remark.* The growth bound rules out trends so explosive that no fixed
linear combination of donors can track them; the positive-definite Gram
condition rules out perfectly collinear donors so the weights are
well-defined. Both are mild for typical marketing and macro panels.

**Parallel trends.** Two nonlinear series have *parallel trends* if their
difference is a zero-mean stationary process. The SC pre-trends
assumption is that :math:`y_{1t}` and :math:`x_t'\hat{\beta}_{\mathrm{SC}}`
are parallel for :math:`t \in \mathcal{T}_1`.

*Remark.* Under Assumptions 1-2, Li and Shankar show (Proposition 3.1)
that the MSC(c) fitted curve is **almost always** parallel to the treated
series, provided at least one donor's loading has the same sign as the
treated unit's. MSC(c) is therefore the natural benchmark against which
the SC restrictions are tested.

Step 1: Testing the SC Pre-Trends Assumption
--------------------------------------------

The key equivalence (Proposition 3.1) is that, with MSC(c) as benchmark,
**the SC pre-trends assumption holds if and only if the two SC
restrictions hold** -- the donor weights sum to one *and* the intercept is
zero. So testing pre-trends reduces to a joint linear restriction on
:math:`\hat{\beta}_{\mathrm{MSC},T_1}`:

.. math::

   H_0:\ R \beta_0 - q = \mathbf{0}_2,
   \qquad
   R = \begin{pmatrix} 0 & \mathbf{1}_{N-1}' \\ 1 & \mathbf{0}_{N-1}' \end{pmatrix},
   \quad q = (1, 0)'.

The first row tests adding-up; the second tests the zero intercept. With
:math:`\hat{d} = R\hat{\beta}_{\mathrm{MSC},T_1} - q`, the feasible
statistic is the quadratic form

.. math::

   \hat{S}_{T_1} = \bigl(\sqrt{T_1}\,\hat{d}\bigr)' \hat{V}^{-1}
       \bigl(\sqrt{T_1}\,\hat{d}\bigr).

Because the constrained estimator can sit on the boundary of its
parameter space (a weight pinned at zero), its limit is the projection of
a normal onto a convex cone -- non-standard, so the ordinary bootstrap
fails. Li and Shankar instead use **subsampling**:

.. math::

   \text{for } b = 1, \ldots, B:\quad
   \text{draw } m \text{ obs with replacement from } \mathcal{T}_1,\
   \text{refit MSC(c)} \Rightarrow \hat{\beta}^{*}_{\mathrm{MSC},m,b}.

The subsample fits give a consistent variance estimate
:math:`\hat{V} = R\,\widehat{\mathrm{Var}}^{*}\!\bigl(\sqrt{T_1}
\hat{\beta}_{\mathrm{MSC},T_1}\bigr) R'` with

.. math::

   \widehat{\mathrm{Var}}^{*} = \frac{m}{B} \sum_{b=1}^{B}
       \bigl(\hat{\beta}^{*}_{\mathrm{MSC},m,b} - \hat{\beta}_{\mathrm{MSC},T_1}\bigr)
       \bigl(\hat{\beta}^{*}_{\mathrm{MSC},m,b} - \hat{\beta}_{\mathrm{MSC},T_1}\bigr)',

and the subsampling distribution :math:`S^{*}_{m,b} = \bigl(\sqrt{m}
R(\hat{\beta}^{*}_{\mathrm{MSC},m,b} - \hat{\beta}_{\mathrm{MSC},T_1})
\bigr)' \hat{V}^{-1} \bigl(\cdots\bigr)`. Sorting the :math:`S^{*}_{m,b}`
gives the :math:`(1-\alpha)` acceptance region
:math:`[S^{*}_{m,(\alpha B/2)},\, S^{*}_{m,((1-\alpha/2)B)}]`; we **reject
:math:`H_0`** when :math:`\hat{S}_{T_1}` falls outside it.

If the joint :math:`H_0` is rejected, the source of the violation is
unclear, so we test the two restrictions singly. With :math:`R_a = (0,
\mathbf{1}_{N-1}')`, :math:`q_a = 1` for adding-up and :math:`R_b = (1,
\mathbf{0}_{N-1}')`, :math:`q_b = 0` for the intercept, the single
statistic is simply the squared scaled deviation (here :math:`\hat{V}` is
replaced by one),

.. math::

   \hat{S}_{T_1, s} = \bigl(\sqrt{T_1}\,\hat{d}_s\bigr)^2,
   \qquad \hat{d}_s = R_s \hat{\beta}_{\mathrm{MSC},T_1} - q_s,
   \quad s = a, b,

with acceptance regions read off the corresponding subsampling
distributions. TSSC then walks a decision tree: keep all SC restrictions
if the joint test is **not** rejected (use **SC**); otherwise test
adding-up -- not rejected gives **MSCa**; if rejected, test the zero
intercept -- not rejected gives **MSCb**, rejected gives **MSCc**. In
words, relax exactly the restriction(s) the data reject, stopping at the
least-flexible variant consistent with the evidence.

.. note::

   The subsample size :math:`m` is a tuning parameter
   (``subsample_size``). The paper's rule of thumb is :math:`m` between
   :math:`T_1/2` and :math:`T_1` for moderate :math:`T_1`; the bootstrap
   special case :math:`m = T_1` (the default here, used when
   ``subsample_size`` is ``None``) performs well in their simulations. If
   different :math:`m` give similar decisions, the test is reliable.

Step 2: Estimating the ATT and Its Confidence Interval
------------------------------------------------------

With the variant chosen, the ATT is the mean post-period gap between the
observed treated series and the recommended counterfactual. Each variant
also carries its own confidence interval via the subsampling procedure of
Li (2020): refit the variant on permuted size-:math:`m` pre-treatment
subsamples (whose treated outcome is regenerated from the fitted weights
plus pre-period noise) to capture donor-weight estimation error, and add
post-period idiosyncratic prediction noise. The interval is
:math:`[\mathrm{ATT} - q_{1-\alpha/2},\ \mathrm{ATT} - q_{\alpha/2}]`,
with :math:`q` the quantiles of the normalized statistic.

*Remark.* Because each correct restriction removes estimation variance,
the recommended variant typically has a **tighter** interval than the
fully flexible MSC(c) -- the efficiency half of TSSC's dual goal made
visible. ``mlsynth`` reports the CI for **all four** variants
(``att_ci_by_method()``) so this trade-off is inspectable.


Verification
------------

TSSC is validated against the authors' published numbers and their Figure-2
Monte Carlo. On Li & Shankar's Brooklyn-showroom panel the recommended variant's
ATT (:math:`+1{,}131.975`) and pre-RMSE (:math:`434.448`) match the paper to
three decimals (and Step 1 selects MSC(b), as the paper reports); the Figure-2
MSE-ratio grid reproduces, with all 16 cells below 1. See the dedicated
replication page, :doc:`replications/tssc`, for the full code, tables and
discussion.

Core API
--------

.. automodule:: mlsynth.estimators.tssc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.TSSCConfig
   :members:
   :undoc-members:

Result Containers
-----------------

The four frozen dataclasses returned by the pipeline. ``TSSC.fit()``
returns a :class:`~mlsynth.utils.tssc_helpers.structures.TSSCResults`,
whose ``variants`` map holds one
:class:`~mlsynth.utils.tssc_helpers.structures.TSSCVariantFit` per
SC-class method and whose ``selection`` records the Step-1
:class:`~mlsynth.utils.tssc_helpers.structures.TSSCRestrictionTest`
outcomes inside a
:class:`~mlsynth.utils.tssc_helpers.structures.TSSCSelection`.

.. automodule:: mlsynth.utils.tssc_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Data preparation -- pivots the long panel into the typed
:class:`~mlsynth.utils.tssc_helpers.structures.TSSCInputs`.

.. automodule:: mlsynth.utils.tssc_helpers.setup
   :members:
   :undoc-members:

Constrained-LS estimation of the four SC-class variants and the
per-variant subsampling ATT confidence interval (Step 2).

.. automodule:: mlsynth.utils.tssc_helpers.estimation
   :members:
   :undoc-members:

The Step-1 subsampling test of the SC pre-trends assumption and the
SC -> MSCa -> MSCb -> MSCc decision tree.

.. automodule:: mlsynth.utils.tssc_helpers.selection
   :members:
   :undoc-members:

Assembly of the standardized ``BaseEstimatorResults`` summary for the
recommended variant.

.. automodule:: mlsynth.utils.tssc_helpers.results_assembly
   :members:
   :undoc-members:

The observed-vs-recommended-counterfactual plot.

.. automodule:: mlsynth.utils.tssc_helpers.plotter
   :members:
   :undoc-members:

The Li & Shankar Figure 2 DGP, packaged as ``simulate_tssc_sample`` so
the Path-B replication in *Verification* runs as a one-liner.

.. automodule:: mlsynth.utils.tssc_helpers.simulation
   :members:
   :undoc-members:
