Harmonic Synthetic Control (HSC)
================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Reach for HSC, due to Liu and Xu (2026) [HSC]_, when your outcome is
**nonstationary with unit-specific stochastic trends** and you do not know,
ex ante, whether that trend variation is *shared* across units or *idiosyncratic*
to the treated unit. This is the regime where standard synthetic control is most
fragile, and where the usual fixes force an awkward all-or-nothing choice:

- **Matching on raw levels** (plain SC, SC-with-intercept, SDID) uses all the
  trend variation for donor weighting. That is ideal when the trend is *shared*
  (a convex combination of donors reproduces it), but when the treated unit has
  its **own** stochastic trend it produces *spurious matching* — a tight
  pre-period fit that does not persist out of sample (Granger-Newbold; Phillips,
  1986), with bias that does **not** vanish as the pre-period grows.
- **Hard pre-filtering** (differencing, or the Hamilton trend-cycle split of
  :doc:`sbc`) removes the stochastic trend before matching. That neutralizes the
  idiosyncratic-trend risk, but it also **discards genuinely shared trend
  variation** that would have helped pin down the donor weights.

HSC replaces this binary choice with a **soft, data-driven allocation**: a single
parameter :math:`\rho \in [0, 1]`, chosen by rolling-origin cross-validation,
continuously interpolates between SC on differences (:math:`\rho \to 0`) and SC on
levels with an intercept/trend (:math:`\rho \to 1`). The method does not need to
know which regime it is in — cross-validation finds the allocation that predicts
best out of sample.

Concretely, use HSC for **trending, nonstationary outcomes** where the
shared-vs-idiosyncratic split is unknown:

- **Marketing / business science.** Brand or category **sales**, **market share**,
  or **traffic** after an event — where some markets share a common demand trend
  but the focal market may also drift on its own.
- **Macro / policy.** **GDP per capita**, **unemployment**, **emissions** — the
  canonical nonstationary outcomes, where whether the treated economy's trend is
  common or idiosyncratic is rarely known in advance.

The flip side: if your outcome is plausibly **stationary**, the spurious-matching
concern is muted and a conventional SCM is simpler. And if you are confident the
stochastic trend is *purely* idiosyncratic, hard differencing (or :doc:`sbc`) is a
defensible fixed choice; HSC buys robustness precisely when you are *unsure*.

Notation
--------

Let :math:`Y_{1,t}` be the treated outcome and :math:`X_{\text{pre}} \in
\mathbb{R}^{T_0 \times N}` the donor matrix over the :math:`T_0` pre-treatment
periods, with post-treatment donors :math:`X_{\text{post}}`. Donor weights
:math:`\omega` live on the simplex :math:`\Delta_N = \{\omega \ge 0,\,
\mathbf{1}^\top\omega = 1\}`. :math:`D_q` is the :math:`q`-th order difference
operator (:math:`q \in \{1, 2\}`) and :math:`K_q = D_q^\top D_q` the roughness
matrix. The pre-treatment discrepancy is :math:`r(\omega) = Y_{\text{pre}} -
X_{\text{pre}}\,\omega`.

The Outcome Decomposition
-------------------------

HSC decomposes untreated potential outcomes into three pieces (Liu-Xu Eq. (1)):

.. math::

   Y_{i,t}(0) \;=\; L_{i,t} \;+\; R_{i,t} \;+\; \varepsilon_{i,t},

where :math:`L_{i,t} = \Lambda_i^\top F_t` is a **shared** low-rank component (a
convex combination of donors that matches the treated loadings reproduces it),
:math:`R_{i,t}` is an **idiosyncratic stochastic trend** (long-run variance grows
without bound), and :math:`\varepsilon_{i,t}` is idiosyncratic short-run noise.
The difficulty is :math:`R`: independent stochastic trends produce realized
correlations that do not vanish as :math:`T_0` grows, so longer pre-periods do not
help. Whether the treated unit's trend variation is mostly shared (in :math:`L`)
or idiosyncratic (in :math:`R`) is generally unknown — which is exactly what HSC's
allocation parameter is designed to handle.

Assumptions
-----------

HSC inherits the standard synthetic-control identification conditions for the
*shared* component and adds two structural conditions on the *idiosyncratic
trend* that license the spectral metric and the forecast step. Each is stated
formally with a plain-language remark.

**Assumption 1 (three-way outcome model).** Untreated potential outcomes follow
the decomposition above, :math:`Y_{i,t}(0) = \Lambda_i^\top F_t + R_{i,t} +
\varepsilon_{i,t}`, where :math:`\Lambda_i^\top F_t` is a shared low-rank factor
structure, :math:`R_{i,t}` is an integrated (unit-root) idiosyncratic process,
and :math:`\varepsilon_{i,t}` is mean-zero short-run noise with
:math:`\mathbb{E}[\varepsilon_{i,t} \mid F_t, \Lambda_i, R_{i,t}] = 0`.

*Remark.* This is the interactive-fixed-effects model behind synthetic-control
consistency theory (Abadie et al., 2010; Bai, 2009), augmented with a
unit-specific stochastic trend :math:`R`. The augmentation is the whole point:
classical SC assumes the donor pool spans *all* of the treated unit's systematic
variation, whereas HSC explicitly allows a piece (:math:`R`) that **no** donor
combination can reproduce.

**Assumption 2 (the shared component is in the donor hull).** The treated unit's
factor loading lies in the convex hull of the donor loadings — there exists
:math:`\omega \in \Delta_N` with :math:`\sum_i \omega_i \Lambda_i = \Lambda_1`,
hence :math:`\sum_i \omega_i L_{i,t} = L_{1,t}` for all :math:`t`.

*Remark.* This is the usual "treated unit in the convex hull of donors"
requirement of Abadie et al. (2010), but imposed on the **shared component
only**. HSC does not ask the donors to reproduce the treated unit's idiosyncratic
trend — that is what the smooth component :math:`E` is for — so the hull
condition is materially weaker here than in level-matching SC.

**Assumption 3 (spectral separation of the idiosyncratic trend).** The
idiosyncratic trend :math:`R_{1,t}` is *low-frequency*: it is integrated of order
:math:`q` so that :math:`D_q R_1` is stationary and short-memory, and its energy
is concentrated at low frequencies, separated from the broadband short-run noise
:math:`\varepsilon`.

*Remark.* This is the assumption that gives the metric :math:`W_{\rho,q}` and the
smoother :math:`S_{\rho,q}` their meaning. The smoother extracts the
low-frequency part of the pre-period residual into :math:`E`; the metric
down-weights exactly that part when matching donors. If :math:`R` were itself
high-frequency, no smoother could separate it from noise and the spectral
allocation would have nothing to exploit. Under *independent* stochastic trends,
level-matching :math:`R^2` does **not** vanish as :math:`T_0 \to \infty`
(Granger–Newbold, 1974; Phillips, 1986) — the pre-period fit looks excellent but
is spurious — which is precisely the failure mode Assumption 3 lets HSC avoid.

**Assumption 4 (the smooth component is forecastable).** The smooth component
:math:`E` absorbing :math:`R_1` (plus any unmatched low-frequency residual)
follows a model the post-period forecaster extrapolates consistently — by default
an integrated AR(1), which is correctly specified when :math:`R_1` is an
ARIMA(1,1,0) trend.

*Remark.* The post-period counterfactual error decomposes into the donor-matching
error (controlled by Assumption 2) and the *forecast* error of :math:`E`. The
latter dominates at long horizons, so the relevant model is the one for
:math:`R_1`'s differences. ``forecaster="last"`` is the conservative fallback (a
driftless random walk) when even an AR(1) on the differences is suspect.

**Assumption 5 (no anticipation, no interference, clean donors).** Donor units
are untreated throughout the sample; the treated unit's pre-period outcomes are
untreated potential outcomes (no anticipation); and treatment of the focal unit
does not spill over onto donors (SUTVA).

*Remark.* These are the standard SCM design conditions and are not specific to
HSC; they are what make :math:`X_{\text{pre}}` and :math:`X_{\text{post}}` valid
controls and the pre-period a clean training window for both :math:`\omega` and
:math:`E`.

Mathematical Formulation
------------------------

The Profiled Metric (Proposition 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HSC jointly estimates donor weights :math:`\omega` and a treated-unit-specific
**smooth component** :math:`E` that absorbs low-frequency residual variation, with
the roughness of :math:`E` penalized by :math:`\|D_q E\|_2^2`. Profiling out
:math:`E` reduces the problem to a donor-weight QP under a
:math:`\rho`-dependent metric. With :math:`\lambda_\rho = \rho/(1-\rho)`,

.. math::

   S_{\rho,q} = (I_{T_0} + \lambda_\rho K_q)^{-1},
   \qquad
   W_{\rho,q} = \tfrac{1}{\rho}\,(I_{T_0} - S_{\rho,q}),

the donor weights solve a ridge-regularized simplex QP,

.. math::

   \hat\omega(\rho, q) = \arg\min_{\omega \in \Delta_N}
   \; r(\omega)^\top W_{\rho,q}\, r(\omega) + \zeta\,\|\omega\|_2^2,

and the fitted smooth component is :math:`\hat E = S_{\rho,q}\,(Y_{\text{pre}} -
X_{\text{pre}}\hat\omega)`. The smoother :math:`S_{\rho,q}` extracts the
low-frequency (trend-like) part of the residual; the metric :math:`W_{\rho,q}`
down-weights exactly that part in donor matching, which is what alleviates the
spurious-matching risk.

.. note::

   **Donor ridge.** The penalty coefficient :math:`\zeta` is set by ``ridge``.
   A float gives a relative ridge ``ridge * trace(X'WX)/N`` (default
   ``1e-6``). The string ``ridge="sdid"`` uses the data-driven SDID-style
   penalty :math:`\zeta^2 T_0` with :math:`\zeta = T_{\text{post}}^{1/4}
   \hat\sigma_{\Delta X}` (Arkhangelsky et al., 2021; Liu-Xu §7), where
   :math:`\hat\sigma_{\Delta X}` is the standard deviation of the donors'
   first differences. The SDID ridge **diversifies** the donor weights (no
   corner solutions) and is the configuration the paper uses for its empirical
   application. On the 1997 Hong Kong handover it spreads weight broadly across
   all 11 donors (largest Korea ≈ 0.16, vs ≈ 0.41 under the near-unregularized
   relative ridge) while leaving the 2003 counterfactual essentially unchanged
   — see :ref:`hsc-hong-kong` below.

The Two Endpoints
^^^^^^^^^^^^^^^^^

The metric extends continuously to the boundary, giving the two classical
estimators as special cases:

- :math:`\rho \to 0`: :math:`S = I`, :math:`W = K_q` — donor matching on **q-th
  differences**, and :math:`E` absorbs the entire level discrepancy.
- :math:`\rho \to 1`: :math:`S = P_0` (the projector onto :math:`\mathrm{Null}(K_q)`),
  :math:`W = I - P_0` — donor matching on **levels with an intercept** (:math:`q=1`)
  or **intercept plus linear trend** (:math:`q=2`).

Interior :math:`\rho` interpolates smoothly between them — a *soft spectral
transformation* of the data rather than a hard filter.

Counterfactual and Forecasting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the post-period the smooth component is extrapolated by a time-series
forecaster and added to the donor-matched component:

.. math::

   \hat Y_{1,T_0+h}(0) \;=\; X_{\text{post},h}^\top \hat\omega \;+\; \widehat{E}_{T_0+h}.

The default forecaster is a closed-form **ARIMA(1, 1, 0)** (an integrated AR(1) on
the differences of :math:`\hat E`), the correctly specified model for an
idiosyncratic ARIMA(1,1,0) stochastic trend; ``forecaster="last"`` carries the
last fitted value forward.

Selecting the Allocation
^^^^^^^^^^^^^^^^^^^^^^^^

:math:`\rho` is chosen by **rolling-origin cross-validation** (scikit-learn's
:class:`~sklearn.model_selection.TimeSeriesSplit`): for each candidate
:math:`\rho` and each expanding-window fold, HSC is fit on the training block and
scored on the held-out block by :math:`X_{\text{val}}\hat\omega + \mathrm{forecast}(\hat E)`.
The :math:`\rho` with the lowest average out-of-sample error is selected, with no
ex-ante assumption about whether the stochastic trend is shared or idiosyncratic.

Monte Carlo Study: Regime Adaptation
------------------------------------

The block below is self-contained. It draws panels from the paper's
data-generating process (Appendix C.1) under two regimes — a **shared** stochastic
trend (``rho_u=1``) and an **idiosyncratic** one (``rho_u=0``) — and compares HSC
against the two fixed strategies it interpolates: SC with an intercept (levels) and
SC on first differences. The treated unit gets *no* effect, so the post-period RMSE
is pure counterfactual error.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import cvxpy as cp
   from mlsynth import HSC

   def integ_ar1(T, phi, innov):
       d = np.zeros(T)
       for t in range(1, T):
           d[t] = phi * d[t - 1] + innov[t]
       return np.cumsum(d)

   def simulate(rng, Lam, lam0, alpha_d, N0, T0, Tpost, kappa, rho_u, phe=0.25):
       T = T0 + Tpost
       F = np.column_stack([
           np.cumsum(rng.normal(0, 2, T)),                 # random walk
           integ_ar1(T, 0.5, rng.normal(0, 2, T)),         # ARIMA(1,1,0)
           np.r_[0.0, np.cumsum(rng.normal(0, 1, T - 1))], # extra trend factor
       ])
       L = np.vstack([lam0, Lam]) @ F.T
       uc = rng.normal(0, np.sqrt(1 - phe**2), T)
       E = np.zeros((N0 + 1, T))
       for j in range(N0 + 1):
           ui = rng.normal(0, np.sqrt(1 - phe**2), T)
           E[j] = integ_ar1(T, phe, np.sqrt(rho_u) * uc + np.sqrt(1 - rho_u) * ui)
       alpha = np.concatenate([[0.0], alpha_d])
       Y = L + kappa * E + rng.normal(0, 1, (N0 + 1, T)) + alpha[:, None] + rng.normal(0, 1, T)
       return Y                                            # (N0+1, T); row 0 = treated

   def to_df(Y, T0):
       return pd.DataFrame(
           [{"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t]),
             "treat": int(j == 0 and t >= T0)}
            for j in range(Y.shape[0]) for t in range(Y.shape[1])]
       )

   def simplex_fit(X, y):
       w = cp.Variable(X.shape[1])
       cp.Problem(cp.Minimize(cp.sum_squares(y - X @ w)),
                  [w >= 0, cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
       return np.clip(w.value, 0, None)

   def sc_int(Xpre, Ypre, Xpost):       # SC with intercept (levels)
       w = simplex_fit(Xpre - Xpre.mean(0), Ypre - Ypre.mean())
       return Xpost @ w + (Ypre.mean() - Xpre.mean(0) @ w)

   def sc_diff(Xpre, Ypre, Xpost):      # SC on first differences
       w = simplex_fit(np.diff(Xpre, axis=0), np.diff(Ypre))
       return Ypre[-1] + (Xpost - Xpre[-1]) @ w

   N0, T0, Tpost, R = 20, 100, 10, 60
   m = np.random.default_rng(0)
   Lam = np.clip(m.normal(0, 0.5, (N0, 3)), -2, 2)            # donor loadings (fixed)
   S8 = m.choice(N0, 8, replace=False)
   lam0 = m.dirichlet(np.ones(8) * 0.5) @ Lam[S8]             # treated loading, in hull
   alpha_d = m.uniform(5, 15, N0)                            # donor fixed effects

   rmse = lambda e: float(np.sqrt(np.mean(np.asarray(e) ** 2)))
   for label, rho_u in [("common drift", 1.0), ("idiosyncratic", 0.0)]:
       ei, ed, eh, rhos = [], [], [], []
       for r in range(R):
           rng = np.random.default_rng(1000 + r)
           Y = simulate(rng, Lam, lam0, alpha_d, N0, T0, Tpost, kappa=2.0, rho_u=rho_u)
           Ypre, Xpre = Y[0, :T0], Y[1:, :T0].T
           Xpost, Y0post = Y[1:, T0:].T, Y[0, T0:]
           ei.append(sc_int(Xpre, Ypre, Xpost) - Y0post)
           ed.append(sc_diff(Xpre, Ypre, Xpost) - Y0post)
           res = HSC({"df": to_df(Y, T0), "outcome": "y", "unitid": "unit",
                      "time": "time", "treat": "treat"}).fit()
           eh.append(res.design.counterfactual_post - Y0post)
           rhos.append(res.selected_rho)
       print(f"{label:>14}: SC-INT {rmse(ei):5.2f} | SC-diff {rmse(ed):5.2f} | "
             f"HSC {rmse(eh):5.2f} | mean rho_hat {np.mean(rhos):.2f}")

Running it reproduces the paper's headline (and matches the standalone skeleton
bit-for-bit):

.. code-block:: text

     common drift: SC-INT  1.15 | SC-diff  1.48 | HSC  1.21 | mean rho_hat 0.86
    idiosyncratic: SC-INT 10.60 | SC-diff  6.11 | HSC  6.46 | mean rho_hat 0.48

HSC is good in **both** regimes, tracking whichever fixed estimator is best, while
each fixed estimator fails in one: SC-on-levels is excellent under a shared trend
(1.15) but **catastrophic when the trend is idiosyncratic** (10.60); SC-on-differences
is the reverse. And the cross-validated allocation moves the right way on its own —
:math:`\hat\rho \approx 0.86` (lean to levels) under a shared trend, dropping to
:math:`\approx 0.48` (lean to differencing) when the trend is idiosyncratic.

.. note::

   The small gap to the *oracle-best* fixed method in each regime (1.21 vs 1.15;
   6.46 vs 6.11) is the expected price of adaptation — cross-validation is not an
   oracle — and is what buys robustness across regimes.

Inference
---------

Setting ``run_inference=True`` runs an **Abadie placebo permutation test**: each
donor is reassigned to the treated slot, HSC is refit (including the
:math:`\rho`-CV), and the treated unit's absolute ATT is compared to the placebo
distribution to form ``results.p_value``. It is off by default because it refits
HSC once per donor; cap the cost with ``max_placebo``.

.. _hsc-hong-kong:

Empirical Application: the 1997 Hong Kong Handover
--------------------------------------------------

The paper's headline application (Liu-Xu §7) revisits the economic-integration
question of Hsiao, Ching and Wan (2012): what happened to Hong Kong's real GDP
per capita after the 1997 handover to mainland China? The panel
(``basedata/hong_kong_handover.csv``) is annual real GDP per capita for Hong Kong
and **11 OECD donors**, 1961–2010, with treatment switching on in **1997**
(:math:`T_0 = 36` pre-treatment years, 14 post). GDP per capita is the canonical
nonstationary, trending outcome for which HSC is built, and it is unknown a
priori whether Hong Kong's growth trend is shared with the donor pool or
idiosyncratic — exactly the regime HSC's cross-validated :math:`\rho` resolves.

.. code-block:: python

   import pandas as pd
   from mlsynth import HSC

   df = pd.read_csv(
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/hong_kong_handover.csv"
   )

   res = HSC({
       "df": df, "outcome": "gdp", "unitid": "country",
       "time": "year", "treat": "Handover",
       "ridge": "sdid",          # SDID-style ridge -> diversified weights
       "display_graphs": True,
   }).fit()

   print("selected rho:", round(res.selected_rho, 3))          # 0.0
   yrs = list(res.inputs.time_labels)
   i03 = yrs.index(2003)
   print("2003 counterfactual:", round(res.counterfactual_full[i03]))   # 30145
   print("2003 effect:", round(res.inputs.Y_post[i03 - res.inputs.T0]
                               - res.counterfactual_full[i03]))          # -2048
   for k, v in sorted(res.weights_by_donor.items(), key=lambda kv: -kv[1]):
       print(f"  {k:12s} {v:.3f}")

**Donor weights.** With the SDID ridge the design leans on a *broad* mix of
donors rather than a handful — the largest weight is Korea ≈ 0.16, followed by
Germany ≈ 0.13, the US ≈ 0.12 and Italy ≈ 0.11, with the remaining seven donors
each contributing 0.05–0.09. This is the diversification the paper reports;
under the near-unregularized relative ridge the same fit collapses most of the
weight onto Korea (≈ 0.41).

**The counterfactual path.** Cross-validation selects :math:`\hat\rho = 0`
(lean to differencing), consistent with Hong Kong's post-handover growth being
substantially idiosyncratic. The estimated effect is **not monotone**, and that
is the substantive finding:

.. list-table:: HSC counterfactual vs. observed real GDP per capita (USD), ``ridge="sdid"``
   :header-rows: 1
   :widths: 14 20 20 18

   * - Year
     - Observed
     - Counterfactual
     - Effect
   * - 1997
     - 26,632
     - 26,059
     - +573
   * - 1998
     - 24,857
     - 26,578
     - **-1,721**
   * - 1999
     - 25,238
     - 27,748
     - **-2,510**
   * - 2002
     - 27,210
     - 29,830
     - **-2,620**
   * - 2003
     - 28,097
     - 30,145
     - **-2,048**
   * - 2005
     - 32,401
     - 31,709
     - +693
   * - 2008
     - 36,924
     - 33,616
     - +3,308
   * - 2010
     - 38,090
     - 33,104
     - +4,986

Hong Kong falls **below** its synthetic counterfactual for roughly 1998–2004 —
the Asian Financial Crisis and the 2003 SARS epidemic — bottoming near
**-$2,600** per capita, before crossing above it from 2005 on as mainland
integration (the 2003 CEPA agreement and the post-2004 mainland boom) takes
hold, reaching **+$5,000** by 2010.

.. note::

   Because the effect changes sign, the *pooled* post-period mean ATT is small
   (≈ +$270) and the Abadie placebo test does **not** reject (``p ≈ 0.92`` with
   all 11 donors as placebos). This is the correct reading, not a defect: the
   handover's effect on GDP per capita is **dynamic** — a multi-year shortfall
   during the crisis years followed by an integration dividend — and a single
   averaged number hides it. Report the trajectory, not just the mean.

Robustness Checks
^^^^^^^^^^^^^^^^^

The 2003 counterfactual and its ≈ -$2,000 shortfall are stable across the
estimator's tuning choices. Re-fitting under the relative ridge, second-order
differencing (:math:`q = 2`), and the last-value forecaster:

.. list-table:: Hong Kong robustness: the 2003 counterfactual is stable
   :header-rows: 1
   :widths: 34 10 12 14 14

   * - Configuration
     - :math:`\hat\rho`
     - max weight
     - cf 2003
     - effect 2003
   * - ``ridge="sdid"``, ``q=1``, ARIMA(1,1,0)
     - 0.00
     - 0.16
     - 30,145
     - -2,048
   * - ``ridge=1e-6``, ``q=1``, ARIMA(1,1,0)
     - 0.50
     - 0.41
     - 29,792
     - -1,694
   * - ``ridge="sdid"``, ``q=2``, ARIMA(1,1,0)
     - 0.00
     - 0.15
     - 30,218
     - -2,121
   * - ``ridge="sdid"``, ``q=1``, ``forecaster="last"``
     - 0.00
     - 0.16
     - 30,755
     - -2,658

Across every configuration the 2003 counterfactual stays in a tight
**$29,800–$30,800** band and the 2003 effect in **-$1,700 to -$2,700**. The
ridge choice changes *how the weight is spread* (max donor weight 0.16 under the
SDID ridge vs. 0.41 under the relative ridge) far more than it changes the
counterfactual; the differencing order and forecaster barely move it. The
rolling-origin CV curve over :math:`\rho` is also informative — it is flat-to-
rising, confirming that levels-matching (high :math:`\rho`) is *penalized* here,
the signature of an idiosyncratic trend:

.. list-table:: Rolling-origin CV error by :math:`\rho` (``ridge="sdid"``)
   :header-rows: 1
   :widths: 20 30

   * - :math:`\rho`
     - CV mean squared error
   * - **0.00** (selected)
     - **1.15e6**
   * - 0.20
     - 1.20e6
   * - 0.50
     - 1.17e6
   * - 0.80
     - 1.85e6
   * - 0.97
     - 5.05e6

.. note::

   This is a **Path-A-style empirical illustration**, not a machine-precision
   replication: it reproduces the paper's *qualitative* findings on the same
   data — broadly diversified donor weights (max ≈ 0.16, no corner solution) and
   a ≈ $30,000 / ≈ -$2,000 post-handover counterfactual through 2003 — but the
   public donor panel and pre-processing differ in minor ways from the authors'.
   The validated bit-for-bit check for HSC is the Monte Carlo above, which
   matches the standalone estimation skeleton exactly.

Core API
--------

.. automodule:: mlsynth.estimators.hsc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.HSCConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``HSC.fit()`` returns an
:class:`~mlsynth.utils.hsc_helpers.structures.HSCResults`, bundling the fitted
:class:`~mlsynth.utils.hsc_helpers.structures.HSCDesign` (donor weights, the smooth
component, the cross-validated ``rho`` and its CV curve, the counterfactual), the
prepared :class:`~mlsynth.utils.hsc_helpers.structures.HSCInputs`, and an optional
:class:`~mlsynth.utils.hsc_helpers.structures.HSCInference`.

.. automodule:: mlsynth.utils.hsc_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

.. automodule:: mlsynth.utils.hsc_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.hsc_helpers.formulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.hsc_helpers.forecast
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.hsc_helpers.optimization
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.hsc_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.hsc_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.hsc_helpers.plotter
   :members:
   :undoc-members:

References
----------

Liu, Z., & Xu, Y. (2026). "The Harmonic Synthetic Control Method." See
[HSC]_.

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American Statistical
Association* 105(490):493-505.

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
(2021). "Synthetic Difference-in-Differences." *American Economic Review*
111(12):4088-4118.

Bai, J. (2009). "Panel Data Models with Interactive Fixed Effects."
*Econometrica* 77(4):1229-1279.

Granger, C. W. J., & Newbold, P. (1974). "Spurious Regressions in
Econometrics." *Journal of Econometrics* 2(2):111-120.

Hsiao, C., Ching, H. S., & Wan, S. K. (2012). "A Panel Data Approach for
Program Evaluation: Measuring the Benefits of Political and Economic
Integration of Hong Kong with Mainland China." *Journal of Applied
Econometrics* 27(5):705-740.

Phillips, P. C. B. (1986). "Understanding Spurious Regressions in
Econometrics." *Journal of Econometrics* 33(3):311-340.
