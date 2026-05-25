Relaxed / Penalized Synthetic Control (RESCM)
=============================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The classic synthetic control method (SCM) of Abadie, Diamond and Hainmueller
[ABADIE2010]_ builds a counterfactual as a **convex** combination of donor
units -- weights on the simplex, no intercept. That convex hull restriction is
what makes SCM interpretable and robust, but it is also brittle when the donor
pool is large relative to the pre-period: with :math:`N` donors and only
:math:`T_1` pre-periods, the least-squares fit underlying SCM is
under-determined once :math:`N \gtrsim T_1`, the weights become unstable, and
many "equally good" solutions exist.

``RESCM`` is a **single convex program** that nests a whole family of SCM
estimators as corner cases, so you can dial the donor-pool regularization
continuously from classic SCM all the way to difference-in-differences (equal
weights), and pick the corner that suits your data. Two branches are exposed,
each with the estimation/inference theory of its own paper:

* **Penalized branch** -- :math:`\min \tfrac{1}{2}\|y_0 - \mu - Y\omega\|_2^2 +
  P(\omega)` with :math:`P` an :math:`\ell_1` / :math:`\ell_2` /
  :math:`\ell_\infty` (or mixed) penalty. The :math:`\ell_\infty` member is the
  **L-infinity-norm SCM** of Wang, Xing and Ye [LinfSC]_, which *spreads*
  weight across donors (capping the largest weight) rather than concentrating
  it; classic Abadie SCM is the no-penalty (:math:`\lambda = 0`) simplex corner,
  and equal-weights/DiD is the heavy-:math:`\ell_\infty` limit
  [DoudchenkoImbens2017]_.
* **Relaxation branch** -- the **SCM-relaxation** of Liao, Shi and Zheng
  [RelaxSC]_: keep the simplex :math:`\omega \in \Delta_J`, but *relax* the
  exact balance first-order condition to an :math:`\ell_\infty` tolerance
  :math:`\eta`, then among all weights satisfying the relaxed condition pick the
  one minimizing an information-theoretic **divergence** :math:`D(\omega)`
  (squared :math:`\ell_2`, entropy, or empirical likelihood). The divergence
  picks a unique, stable weight (e.g. closest-to-uniform), and under a latent
  group structure recovers equal-within-group weights.

Use ``RESCM`` when you have a **single treated unit and a large donor pool**
and want either (i) a dense, stable counterfactual that does not hinge on a
handful of donors (``LINF`` / ``RELAX_*``), or (ii) a one-stop interface to
compare classic SCM against its penalized and relaxed cousins on the same
panel. Pick estimators by name through ``methods``.

Notation
--------

We use the synthetic-control canon. Unit :math:`j=0` is treated and
:math:`\mathcal{N} = \{1, \ldots, N\}` indexes the donors; :math:`\mathbf{y}_0`
is the treated outcome and :math:`\mathbf{Y} = (y_{jt})` the
:math:`T \times N` donor matrix. The intervention occurs after the pre-period
:math:`\mathcal{T}_1 = \{1, \ldots, T_1\}`; the post-period is
:math:`\mathcal{T}_2 = \{T_1+1, \ldots, T\}` with :math:`T_2 = |\mathcal{T}_2|`.
Donor weights :math:`\boldsymbol{\omega} = (\omega_1, \ldots, \omega_N)'` live
on the simplex :math:`\Delta_N = \{\boldsymbol{\omega} : \omega_j \ge 0,\,
\sum_j \omega_j = 1\}` (relaxation branch) or are penalized (penalized branch);
:math:`\mu` is an optional intercept. The counterfactual is
:math:`\hat{y}_{0t}^0 = \mu + \mathbf{Y}_{t\cdot}\,\boldsymbol{\omega}`, the
effect :math:`\hat{\Delta}_t = y_{0t} - \hat{y}_{0t}^0`, and the ATE
:math:`\bar{\Delta} = T_2^{-1}\sum_{t\in\mathcal{T}_2}\hat{\Delta}_t`.
:math:`\|\mathbf{A}\|_\infty = \max_{ij}|a_{ij}|`. With pre-period donor Gram
matrix :math:`\hat{\boldsymbol{\Sigma}} = T_1^{-1}\sum_{t\in\mathcal{T}_1}
\mathbf{Y}_{t\cdot}'\mathbf{Y}_{t\cdot}` and cross-moment
:math:`\hat{\boldsymbol{\Upsilon}} = T_1^{-1}\sum_{t\in\mathcal{T}_1}
\mathbf{Y}_{t\cdot}' y_{0t}`, the SCM least-squares first-order condition is
:math:`\hat{\boldsymbol{\Sigma}}\boldsymbol{\omega} = \hat{\boldsymbol{\Upsilon}}`.

The unified convex program
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every ``RESCM`` corner case is a special case of one program. The
**penalized** form fits the in-sample loss with a regularizer,

.. math::

   \min_{\mu,\,\boldsymbol{\omega}\in\mathcal{C}} \;
     \tfrac{1}{2}\sum_{t\in\mathcal{T}_1}
       \bigl(y_{0t} - \mu - \mathbf{Y}_{t\cdot}\boldsymbol{\omega}\bigr)^2
     + \lambda\Bigl[\alpha\|\boldsymbol{\omega}\|_1
       + (1-\alpha)\,Q(\boldsymbol{\omega})\Bigr],
   \qquad Q \in \{\tfrac{1}{2}\|\cdot\|_2^2,\; \|\cdot\|_\infty\},

with constraint set :math:`\mathcal{C}` (simplex by default), mixing
:math:`\alpha\in[0,1]`, and strength :math:`\lambda` chosen by K-fold
cross-validation. The **relaxation** form instead minimizes a divergence
subject to a relaxed balance condition,

.. math::

   \min_{\boldsymbol{\omega}\in\Delta_N} \; D(\boldsymbol{\omega})
   \quad\text{s.t.}\quad
   \bigl\|\hat{\boldsymbol{\Sigma}}\boldsymbol{\omega}
     - \hat{\boldsymbol{\Upsilon}}\bigr\|_\infty \le \eta,
   \qquad
   D \in \Bigl\{\tfrac{1}{2}\|\boldsymbol{\omega}\|_2^2,\;
     \textstyle\sum_j \omega_j\log\omega_j,\;
     -\textstyle\sum_j \log\omega_j\Bigr\},

where the three divergences are squared :math:`\ell_2`, **entropy**, and
**empirical likelihood** respectively, and :math:`\eta` is selected by
validation. Classic SCM is recovered at :math:`\lambda=0` (penalized, simplex)
or :math:`\eta = 0` (relaxation); the equal-weights/DiD estimator is the
:math:`\lambda\to\infty` :math:`\ell_\infty` limit or large-:math:`\eta` limit.

Penalized branch: L-infinity SCM (Wang, Xing & Ye)
--------------------------------------------------

**Idea.** Classic SCM tends to load heavily on one or two donors. When the
true data-generating process spreads signal across many donors, that
concentration is fragile -- a single idiosyncratic donor shock contaminates the
counterfactual. The :math:`\ell_\infty` penalty
:math:`\|\boldsymbol{\omega}\|_\infty = \max_j |\omega_j|` directly **caps the
largest weight**, so minimizing it (under the simplex) pushes the solution
toward *spreading* weight across donors -- in the limit, equal weights. The
mixed :math:`\ell_1 + \ell_\infty` penalty (``L1LINF``) trades sparsity against
spreading. Both shrink toward a stable, dense counterfactual that tolerates
:math:`N > T_1`.

**Assumptions** (Wang, Xing & Ye [LinfSC]_).

*Assumption 1 (factor model).* The outcomes follow a linear factor model
:math:`y_{jt} = \boldsymbol{\lambda}_j'\mathbf{f}_t + u_{jt}` with the treated
unit's loading in (or near) the convex hull of the donor loadings, so an
:math:`\ell_\infty`-regularized convex combination approximates the treated
factor structure.

*Assumption 2 (weak dependence).* The idiosyncratic errors :math:`u_{jt}` are
mean-zero and weakly dependent over time (strong-mixing with summable
autocovariances), permitting a HAC long-run variance and a sequential CLT for
the post-period mean.

*Assumption 3 (regularization rate).* The penalty strength :math:`\lambda`
(and mixing :math:`\alpha`) are chosen so the weight estimate is consistent for
its population target; in practice both are selected by K-fold cross-validation
on the pre-period.

*Remark.* The :math:`\ell_\infty` cap is what buys stability in high
dimensions: by refusing to let any single weight dominate, the estimator's
prediction variance stays controlled even when :math:`N \gg T_1`, where the
unconstrained least-squares (and concentrated classic SCM) solutions degrade.
This is the synthetic-control analogue of the L2-relaxation argument for dense
coefficients.

**Inference.** Wang, Xing and Ye extend the synthetic-control ATE inference of
Li [LiSCM2020]_ to the dense, weakly dependent setting. With fixed weights the
post-period gap :math:`\hat{\Delta}_t` has mean :math:`\bar{\Delta}`, and

.. math::

   \hat{Z} = \frac{\bar{\Delta}}
       {\sqrt{\hat{\rho}^2_{(1)}/T_1 + \hat{\rho}^2_{(2)}/T_2}}
   \xrightarrow{d} N(0,1),

with :math:`\hat{\rho}^2_{(1)}` the HAC long-run variance of the pre-period
prediction residuals (first-stage weight-estimation uncertainty) and
:math:`\hat{\rho}^2_{(2)}` that of the de-meaned post-period effects. This is
the two-term variance ``RESCM`` uses for **all** corner cases (see *Inference*
below).

**When to use.** Dense, factor-driven donor structure; high dimension
(:math:`N>T_1` permitted); when you want a counterfactual robust to any single
donor's idiosyncrasies rather than a sparse, concentrated fit.

Relaxation branch: SCM-relaxation (Liao, Shi & Zheng)
-----------------------------------------------------

**Idea.** Classic SCM solves a least-squares problem whose first-order
condition is :math:`\hat{\boldsymbol{\Sigma}}\boldsymbol{\omega} =
\hat{\boldsymbol{\Upsilon}}` -- exact balance of the donor moments. When
:math:`N` is large this condition is over-strict: it is satisfied (or nearly
so) by a continuum of weights, and the chosen one is arbitrary and unstable.
SCM-relaxation keeps the simplex but **relaxes** the balance condition to an
:math:`\ell_\infty` tolerance :math:`\eta`, then selects, among all admissible
weights, the one minimizing a divergence :math:`D(\boldsymbol{\omega})`:

.. math::

   \hat{\boldsymbol{\omega}} = \operatorname*{argmin}_{\boldsymbol{\omega}\in\Delta_N}
     D(\boldsymbol{\omega})
   \quad\text{s.t.}\quad
   \bigl\|\hat{\boldsymbol{\Sigma}}\boldsymbol{\omega}
     - \hat{\boldsymbol{\Upsilon}}\bigr\|_\infty \le \eta.

The divergence breaks the tie deterministically: :math:`\ell_2`
(``RELAX_L2``) picks the minimum-norm weight, **entropy** (``RELAX_ENTROPY``)
the maximum-entropy / closest-to-uniform weight, and **empirical likelihood**
(``RELAX_EL``) the EL-optimal weight. :math:`\eta=0` reduces to (a divergence-
selected) classic SCM; large :math:`\eta` admits the whole simplex and the
divergence alone determines the weight (entropy/EL :math:`\to` equal weights).

**Assumptions** (Liao, Shi & Zheng [RelaxSC]_).

*Assumption 1 (approximate factor model with group structure).* Units load on
common factors and fall into latent groups; the treated unit's loading is
spanned by the donor loadings up to an approximation error that the relaxation
tolerance :math:`\eta` absorbs.

*Assumption 2 (identifiable divergence minimizer).* The divergence
:math:`D` is strictly convex on the simplex, so the relaxed feasible set has a
unique minimizer -- this is what restores well-posedness when the exact balance
condition does not.

*Assumption 3 (weak dependence and moments).* The errors are weakly dependent
with bounded moments, so sample moments :math:`\hat{\boldsymbol{\Sigma}},
\hat{\boldsymbol{\Upsilon}}` converge and a post-period CLT applies.

*Remark.* The key result (oracle prediction) is that the relaxed weight
predicts the treated counterfactual as well as the infeasible oracle that knows
the factor structure, and under the latent group structure the divergence-
selected weight is **equal within groups** -- a transparent, interpretable
solution that the arbitrary classic-SCM tie-break does not deliver.

**Inference.** Liao, Shi and Zheng's main theory concerns *prediction*
consistency (the oracle property). For an ATE confidence interval ``mlsynth``
applies the same weak-dependence two-term HAC test as the penalized branch
(Li [LiSCM2020]_), treating the relaxed weights as the fixed first stage. See
the caveat under *Verification*.

The named corner cases
----------------------

``methods`` selects estimators by name; each resolves to one exact call of the
convex engine.

.. list-table::
   :header-rows: 1
   :widths: 16 12 50

   * - Name
     - Branch
     - Estimator
   * - ``SC``
     - penalized
     - Classic Abadie simplex SCM (:math:`\lambda=0`).
   * - ``LASSO``
     - penalized
     - :math:`\ell_1` penalty; sparse donor weights.
   * - ``RIDGE``
     - penalized
     - :math:`\ell_2` penalty; dense shrunken weights.
   * - ``ENET``
     - penalized
     - Elastic net (:math:`\ell_1 + \ell_2`); :math:`\alpha` by CV.
   * - ``LINF``
     - penalized
     - L-infinity-norm SCM [LinfSC]_; spreads weight, nests DiD.
   * - ``L1LINF``
     - penalized
     - Mixed :math:`\ell_1 + \ell_\infty` penalty.
   * - ``RELAX_L2``
     - relaxation
     - SCM-relaxation, :math:`\ell_2` divergence [RelaxSC]_.
   * - ``RELAX_ENTROPY``
     - relaxation
     - SCM-relaxation, entropy divergence.
   * - ``RELAX_EL``
     - relaxation
     - SCM-relaxation, empirical-likelihood divergence.

Inference
---------

Once the donor weights are fixed by any corner case, the post-period gap
:math:`\hat{\Delta}_t = y_{0t} - \hat{y}_{0t}^0` is a scalar series whose mean
is the ATE. ``RESCM`` reports the Li [LiSCM2020]_ two-term long-run variance
used by the L-infinity paper,

.. math::

   \widehat{\mathrm{se}}(\bar{\Delta}) =
     \sqrt{\hat{\rho}^2_{(1)}/T_1 + \hat{\rho}^2_{(2)}/T_2},

where :math:`\hat{\rho}^2_{(1)}` is the HAC long-run variance of the pre-period
prediction residuals -- carrying the **first-stage weight-estimation
uncertainty** -- and :math:`\hat{\rho}^2_{(2)}` is the HAC long-run variance of
the de-meaned post-period effects (Bartlett kernel, :math:`\lfloor
T_2^{1/4}\rfloor` lag). The pre-period term is essential for dense
penalized/relaxed weights: unlike forward-selection PDA -- whose sample
splitting makes pre/post asymptotically independent and lets a post-only
variance suffice -- the dense estimators reuse the entire pre-window to fit the
weights, so dropping :math:`\hat{\rho}^2_{(1)}` understates the standard error.

Empirical Illustration: California's Proposition 99
---------------------------------------------------

The canonical synthetic-control application [ABADIE2010]_ studies the effect of
California's 1988 Proposition 99 tobacco-control program on per-capita cigarette
sales, with 38 control states over 1970-2000. Running ``RESCM`` with the
classic ``SC`` corner alongside the two papers' headline estimators reuses the
same panel and returns each method's counterfactual, ATE, and a HAC confidence
interval.

.. code-block:: python

   import pandas as pd
   from mlsynth import RESCM

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/smoking_data.csv"
   df = pd.read_csv(url)
   df["Proposition 99"] = df["Proposition 99"].astype(int)

   res = RESCM({"df": df, "outcome": "cigsale", "treat": "Proposition 99",
                "unitid": "state", "time": "year",
                "methods": ["SC", "LINF", "RELAX_L2"], "alpha": 0.05,
                "display_graphs": True}).fit()

   for name, fit in res.fits.items():
       print(f"{name:9s} ATE {fit.att:8.3f}  SE {fit.att_se:6.3f}  "
             f"95% CI ({fit.ci[0]:7.2f},{fit.ci[1]:7.2f})  "
             f"p={fit.p_value:.3f}  donors={len(fit.donor_weights)}")

This prints::

   SC        ATE  -17.371  SE  2.304  95% CI ( -21.89, -12.86)  p=0.000  donors=6
   LINF      ATE  -17.359  SE  2.303  95% CI ( -21.87, -12.85)  p=0.000  donors=6
   RELAX_L2  ATE  -22.190  SE  4.115  95% CI ( -30.26, -14.12)  p=0.000  donors=7

The penalized corners (``SC``, ``LINF``) agree on an outcome-only effect of
about :math:`-17` cigarette packs per capita: when the donor pool fits the
pre-period well (pre-treatment :math:`R^2 \approx 0.98`, pre-RMSE
:math:`1.45`), cross-validation picks near-zero :math:`\ell_\infty` shrinkage
and ``LINF`` collapses onto classic SCM. The relaxation corner (``RELAX_L2``)
trades a little pre-fit (:math:`R^2 \approx 0.97`) for a more constrained
weight and lands at a larger effect, :math:`-22.2`, whose wider confidence
interval reflects the larger pre-period residual variance the two-term standard
error correctly propagates.

Verification
------------

.. note::

   **Empirical (Path A, Proposition 99).** ``SC``/``LINF``/``RELAX_L2`` run on
   the smoking panel (above); the penalized corners reproduce the classic
   outcome-only SCM effect (:math:`\approx -17`) and the relaxation corner the
   denser, larger estimate, both significant and consistent with the
   literature.

   **Simulation (Path B, high dimension).** A size/power Monte Carlo in the
   regime these methods target -- :math:`N=90` donors, :math:`T_1=36`
   pre-periods (:math:`N \gg T_1`), :math:`T_2=36`, three-factor AR(1) DGP,
   treated loading a convex mix of donors, idiosyncratic :math:`N(0,1)` --
   rejecting ``H0: ATE = 0`` at 5% (50 replications; :math:`\delta=0` is size,
   :math:`\delta=1` is power):

   .. list-table::
      :header-rows: 1
      :widths: 16 18 18 18

      * - :math:`\delta`
        - ``SC``
        - ``LINF``
        - ``RELAX_L2``
      * - 0 (size)
        - 0.20
        - 0.22
        - 0.28
      * - 1 (power)
        - 0.98
        - 0.98
        - 0.94

   **Estimation is unbiased and powerful** (mean ATT bias :math:`\approx 0`;
   power :math:`0.94`-:math:`0.98`), but the analytic ATE test **over-rejects**
   in this short, high-dimensional panel. A diagnostic isolates two mechanisms:
   ``SC`` genuinely over-fits the 36-period pre-window (pre-RMSE :math:`0.77` vs
   the true noise sd :math:`1.0`), so :math:`\hat{\rho}^2_{(1)}` understates
   estimation uncertainty -- it self-corrects as :math:`T_1` grows (pre-RMSE
   :math:`\to 0.94` at :math:`T_1=250`); ``RELAX_L2`` does *not* over-fit
   (pre-RMSE :math:`\approx 1.0`), so its over-rejection reflects the analytic
   influence function not capturing how strongly-relaxed dense weights feed into
   the ATE. Both papers' asymptotics require :math:`T_1` large relative to
   :math:`N`; the normal approximation is unreliable at :math:`T_1 < N`, and
   the two-term variance (which already cuts the post-only over-rejection by
   :math:`\sim 35\%`) does not fully close the gap here. For honest
   finite-sample inference in this regime, prefer a placebo / conformal
   procedure [CWZ2021]_. (Only 50 replications -- noisy; the relaxation
   :math:`\eta` is validated by CV, not fixed.)

Core API
--------

.. automodule:: mlsynth.estimators.laxscm
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.RESCMConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``RESCM.fit()`` returns a
:class:`~mlsynth.utils.laxscm_helpers.structures.RESCMResults`, whose ``fits``
maps each named corner case to a
:class:`~mlsynth.utils.laxscm_helpers.structures.RESCMMethodFit` (donor weights,
intercept, counterfactual, gap, ATE, two-term HAC standard error, CI, p-value,
nonzero donor weights, fit diagnostics, and the realized hyperparameters).
Convenience aliases (``att``, ``att_se``, ``counterfactual``,
``donor_weights``) forward to the first requested method. The prepared,
NumPy-only panel is exposed as a
:class:`~mlsynth.utils.laxscm_helpers.structures.RESCMInputs`, with units and
time addressed through an :class:`IndexSet`.

.. automodule:: mlsynth.utils.laxscm_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

The named-estimator registry: each entry maps a name to the exact convex-engine
call (branch and keyword arguments).

.. automodule:: mlsynth.utils.laxscm_helpers.specs
   :members:
   :undoc-members:

Data preparation -- the only DataFrame touchpoint: pivots to NumPy, builds the
unit/time ``IndexSet``es, and splits pre/post.

.. automodule:: mlsynth.utils.laxscm_helpers.setup
   :members:
   :undoc-members:

The weak-dependence two-term HAC ATE inference (Li 2020).

.. automodule:: mlsynth.utils.laxscm_helpers.inference
   :members:
   :undoc-members:

Run loop dispatching each corner case to the convex engine and assembling the
typed per-method fits.

.. automodule:: mlsynth.utils.laxscm_helpers.estimation
   :members:
   :undoc-members:
