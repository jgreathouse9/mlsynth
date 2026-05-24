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

Example
-------

The block below is self-contained -- simulate one panel from a
nonlinear-trend factor model, fit TSSC, and read off the recommendation,
the ATT, and the per-method intervals.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import TSSC

   rng = np.random.default_rng(0)
   N, T, T1 = 6, 40, 30                       # 1 treated + 5 donors
   trend = np.cumsum(rng.normal(size=T))      # common nonlinear trend f_t
   loads = rng.uniform(0.5, 1.5, size=N)      # factor loadings d_j
   Y = 10 + np.outer(trend, loads) + rng.normal(scale=0.5, size=(T, N))
   Y[T1:, 0] += 3.0                           # +3 treatment effect on unit 0

   df = pd.DataFrame(
       {"unit": f"u{i}", "time": t, "y": Y[t, i],
        "treat": int(i == 0 and t >= T1)}
       for i in range(N) for t in range(T)
   )

   res = TSSC({
       "df": df, "outcome": "y", "treat": "treat",
       "unitid": "unit", "time": "time",
       "draws": 300, "seed": 1, "display_graphs": False,
   }).fit()

   print("recommended:", res.recommended_method)
   print("ATT = %.3f  95%% CI = [%.3f, %.3f]" % (res.att, *res.att_ci))
   print("ATT by method:   ", {m: round(a, 3) for m, a in res.att_by_method().items()})
   print("decision path:   ", res.selection.decision_path)

``res`` is a :class:`~mlsynth.utils.tssc_helpers.structures.TSSCResults`:
``res.variants`` holds all four ``TSSCVariantFit`` objects,
``res.selection`` records every restriction test and the decision path,
and ``res.summary`` is the standardized ``BaseEstimatorResults`` for the
recommended variant.

Empirical Illustration
----------------------

Suppose an online retailer opens a showroom in Brooklyn and we have ten
donor markets. The treated series (black) and donors (blue) are below,
with treatment at :math:`t = 76`.

.. image:: https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/examples/TSSC/treatedvsdonors.png
   :alt: Brooklyn vs Donors
   :align: center
   :width: 600px

Here :math:`j = 1` is Brooklyn and there are ten donors. Step 1 selects
MSC(b), whose pre-treatment RMSE is 434.43 and whose absolute and
percentage ATTs are 1131.97 and 24.5%.

.. image:: https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/examples/TSSC/Showroom.png
   :alt: TSSC Results
   :align: center
   :width: 600px

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
