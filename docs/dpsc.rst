Differentially Private Synthetic Control (DPSC)
===============================================

.. currentmodule:: mlsynth

Overview
--------

``DPSC`` is a synthetic control estimator that releases its counterfactual under
a formal privacy guarantee (Rho, Cummings & Misra 2023). It answers a question
none of the other estimators in mlsynth address: when the donor pool is
sensitive -- patient records in a clinical external-control arm, proprietary
firm-level series in a data cooperative -- how do you publish the synthetic
control, and the effect it implies, without leaking any single donor's data?

The estimator is a ridge synthetic control fitted with differentially private
empirical risk minimisation. The pre-period regression coefficients are learned
privately, the post-intervention donor matrix is privatised, and the released
counterfactual is formed from the two. Differential privacy then bounds, in a
mathematically precise sense, how much an adversary who sees the released
output can learn about any one donor.

Privacy is not free. The released estimate is a single draw of a randomised
mechanism, and the noise trades off against accuracy at a rate controlled by
the privacy budget. ``DPSC`` reports that noise honestly as the dominant source
of uncertainty.

When to use this estimator
--------------------------

* The donor pool is sensitive or proprietary -- the donors are people or
  entities whose individual series must be protected -- and the counterfactual
  or the effect will be released externally (to a regulator, a publication, a
  counterparty).
* You need a provable, quantifiable privacy guarantee (an :math:`\varepsilon`),
  not ad hoc de-identification.
* The donor pool is reasonably large and the pre-period long: the privacy noise
  averages down with panel size, so this is where differential privacy is
  affordable.

Do not reach for it for ordinary policy evaluation on public aggregates
(Proposition 99 states, country GDP, your own first-party stores). There is no
individual to protect, and the accuracy cost is then pure loss -- use
:doc:`vanillasc` instead. On a small donor pool the privacy noise is large; see
the honest caveat below.

Notation
--------

Let :math:`j = 1` denote the treated unit and
:math:`\mathcal{N}_0 \coloneqq \{2, \dots, N\}` the donor pool of cardinality
:math:`N_0`. Time runs over :math:`t \in \{1, \dots, T\}`, 1-indexed; the
intervention takes effect after period :math:`T_0`, splitting time into the
pre-period :math:`\mathcal{T}_1 \coloneqq \{t : t \le T_0\}` of length
:math:`T_0` and the post-period :math:`\mathcal{T}_2 \coloneqq \{t : t > T_0\}`.

The treated series is :math:`\mathbf{y}_1 = (y_{11}, \dots, y_{1T})^\top`, and
the donor outcomes stack into
:math:`\mathbf{Y}_0 \coloneqq [\mathbf{y}_j]_{j \in \mathcal{N}_0} \in
\mathbb{R}^{T \times N_0}` (one column per donor);
:math:`\mathbf{Y}_{0,\mathcal{T}_1} \in \mathbb{R}^{T_0 \times N_0}` is its
pre-period block. Regression weights are :math:`\mathbf{f} \in \mathbb{R}^{N_0}`
-- unlike the canonical simplex weights :math:`\mathbf{w}` of :doc:`vanillasc`,
these are unconstrained ridge coefficients. The synthetic counterfactual is
:math:`\widehat{\mathbf{y}}_1 \coloneqq \mathbf{Y}_0\,\widehat{\mathbf{f}}` with
entries :math:`\widehat{y}_{1t}`, the per-period effect is
:math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}`, and the ATT is
:math:`\widehat{\tau} \coloneqq |\mathcal{T}_2|^{-1} \sum_{t \in \mathcal{T}_2}
\tau_t`.

The privacy parameters are the budgets :math:`\varepsilon_1` (Stage 1,
coefficients) and :math:`\varepsilon_2` (Stage 2, released donors), with total
budget :math:`\varepsilon = \varepsilon_1 + \varepsilon_2`; the approximate-DP
slack is :math:`\delta`; the ridge penalty is :math:`\lambda`; and
:math:`\Delta` denotes an :math:`\ell_2` sensitivity. The significance level is
:math:`\alpha`.

Differential privacy
--------------------

A randomised mechanism :math:`\mathcal{M}` is
:math:`(\varepsilon, \delta)`-differentially private (Dwork et al. 2006) if for
every pair of neighbouring datasets :math:`\mathcal{D}, \mathcal{D}'` and every
measurable set :math:`S` of outputs,

.. math::

   \Pr[\mathcal{M}(\mathcal{D}) \in S] \le
   e^{\varepsilon}\, \Pr[\mathcal{M}(\mathcal{D}') \in S] + \delta.

Smaller :math:`\varepsilon` is stronger privacy; :math:`\delta = 0` is pure
:math:`\varepsilon`-DP. The crux for synthetic control is what "neighbouring"
means. Synthetic control regresses *vertically* -- each time point is a sample,
each donor a feature -- so changing one donor changes an entire *column* of the
design. A neighbouring dataset therefore replaces one donor's whole series
:math:`\mathbf{y}_j`, and the privacy guarantee protects donors, not the
treated unit (Rho et al. 2023, Sec 3).

Privacy is calibrated to sensitivity, the largest change in a statistic that
swapping one donor can cause. For a mechanism releasing a vector :math:`g`, the
Laplace mechanism adds noise scaled to its :math:`\ell_2` sensitivity
:math:`\Delta_g \coloneqq \max_{\mathcal{D} \sim \mathcal{D}'} \lVert
g(\mathcal{D}) - g(\mathcal{D}') \rVert_2`; here noise is drawn from the
high-dimensional Laplace distribution (a magnitude :math:`\sim
\mathrm{Lap}(\cdot)` times a uniform direction; Chaudhuri, Monteleoni & Sarwate
2011).

Identifying assumptions
-----------------------

DPSC inherits the synthetic-control identifying assumptions and adds the
conditions differential privacy needs.

1. Pre-treatment fit. There exist coefficients :math:`\mathbf{f}` under which
   the donors reproduce the treated pre-period path,
   :math:`y_{1t} \approx (\mathbf{Y}_0 \mathbf{f})_t` for
   :math:`t \in \mathcal{T}_1`.

   *Remark.* As in :doc:`vanillasc`, a good pre-fit is the empirical certificate
   of a credible counterfactual. The ridge penalty :math:`\lambda` trades a
   little pre-fit for a lower sensitivity (see below), so under privacy the fit
   is deliberately shrunk.

2. No anticipation. Treatment has no effect before :math:`T_0`, so the
   pre-period outcomes reflect the no-intervention path.

   *Remark.* Date :math:`T_0` at the first plausible response, not the nominal
   policy date; a contaminated pre-period biases the extrapolated counterfactual.

3. No interference and untreated donors (SUTVA). The treatment of unit
   :math:`1` does not change any donor's outcome, and every donor is untreated,
   so :math:`\mathbf{Y}_0` carries only no-intervention outcomes.

4. Bounded donor data. The donor observations lie in a bounded range, so the
   sensitivity :math:`\Delta` is finite and the noise calibration is valid.

   *Remark.* This is the additional condition privacy demands. It is what makes
   the influence of any single donor's series bounded, hence privatisable; the
   sensitivity bound below assumes it. Unbounded or heavy-tailed donors break
   the guarantee and must be clipped before fitting.

Note that differential privacy is a property of the release *mechanism*, not of
the data-generating process, so DPSC needs no homoskedasticity or
exchangeability assumption for its privacy guarantee -- those enter only when
one interprets the ATT causally, exactly as in the non-private synthetic
control.

Mathematical formulation
------------------------

The base estimator is a ridge synthetic control: on the pre-period,

.. math::

   \widehat{\mathbf{f}} = \operatorname*{argmin}_{\mathbf{f} \in \mathbb{R}^{N_0}}
   \; \bigl\lVert \mathbf{y}_{1,\mathcal{T}_1} -
   \mathbf{Y}_{0,\mathcal{T}_1} \mathbf{f} \bigr\rVert_2^2
   + \tfrac{\lambda}{2} \lVert \mathbf{f} \rVert_2^2 ,

and the counterfactual extrapolates it, :math:`\widehat{\mathbf{y}}_1 =
\mathbf{Y}_0 \widehat{\mathbf{f}}`. Privacy is injected in two independent
stages.

Stage 1 -- private coefficients. Two mechanisms are offered.

*Output perturbation* (Algorithm 2) fits :math:`\widehat{\mathbf{f}}` and adds
high-dimensional Laplace noise calibrated to the coefficient sensitivity

.. math::

   \Delta = \frac{4\, T_0 \sqrt{8 + N_0}}{\lambda},
   \qquad
   \widetilde{\mathbf{f}} = \widehat{\mathbf{f}}
   + \mathrm{Lap}\!\left(\tfrac{\Delta}{\varepsilon_1}\right).

The sensitivity grows with the pre-period length :math:`T_0` and the donor
count :math:`N_0` (a longer or wider design is more exposed to one donor's
change) and shrinks with :math:`\lambda` (regularisation damps any donor's
influence). Larger :math:`\lambda` therefore buys less noise at the cost of a
more biased, shrunken fit.

*Objective perturbation* (Algorithm 3) instead perturbs the objective with a
random linear term :math:`\mathbf{b}` and solves exactly,

.. math::

   \widetilde{\mathbf{f}} = \operatorname*{argmin}_{\mathbf{f}}
   \; \bigl\lVert \mathbf{y}_{1,\mathcal{T}_1} -
   \mathbf{Y}_{0,\mathcal{T}_1} \mathbf{f} \bigr\rVert_2^2
   + \tfrac{\lambda + \Delta_c}{2} \lVert \mathbf{f} \rVert_2^2
   + \mathbf{b}^\top \mathbf{f} ,

with :math:`\mathbf{b}` Laplace (pure :math:`\varepsilon`-DP, :math:`\delta =
0`) or Gaussian (:math:`(\varepsilon, \delta)`-DP), and a curvature slack
:math:`\Delta_c \ge 0` added when the base regularisation is too weak for the
requested budget (Kifer, Smith & Thakurta 2012). Because the noise enters the
objective rather than the solution, the optimiser keeps
:math:`\widetilde{\mathbf{f}}` bounded, which is why objective perturbation is
far more stable than output perturbation on typical panels.

Stage 2 -- private release. The released donor matrix is privatised,
:math:`\widetilde{\mathbf{Y}}_0 = \mathbf{Y}_0 +
\mathrm{Lap}(2\sqrt{R}/\varepsilon_2)` over the :math:`R` released rows, and the
counterfactual is :math:`\widehat{\mathbf{y}}_1 = \widetilde{\mathbf{Y}}_0\,
\widetilde{\mathbf{f}}`.

Privacy and accuracy. By the composition theorem the release is
:math:`\varepsilon`-DP with :math:`\varepsilon = \varepsilon_1 +
\varepsilon_2`. Relative to the non-private estimator, the root-mean-square
error inflates by a factor :math:`O(1/\varepsilon)` -- the unavoidable price of
privacy (Rho et al. 2023, Sec C).

Inference
---------

The privatised release is one draw of a randomised mechanism, so its dominant
uncertainty is the privacy noise itself. ``DPSC`` reports the released
counterfactual and ATT from a single seeded draw, and quantifies the noise by
the Monte Carlo standard deviation of the ATT over ``n_draws`` independent
privatised draws, giving :math:`\widehat{\tau} \pm z_{1 - \alpha/2}\,
\widehat{\mathrm{se}}`. The non-private ridge ATT (the
:math:`\varepsilon \to \infty` target) is recorded alongside, so the privacy
cost is visible directly.

A caveat, stated plainly. Output perturbation is unusable on a small donor pool:
the coefficient noise inflates :math:`\lVert \widetilde{\mathbf{f}} \rVert` and
the Stage-2 term amplifies it, so a single release can be off by many multiples
of the effect even at a nominal budget. Objective perturbation (the default) is
the viable mechanism, but even it roughly doubles the effective uncertainty at
weak privacy, and meaningful privacy (:math:`\varepsilon \approx 1`) on a
few-dozen-donor panel costs a biased estimate with a wide band. The favourable
regime is a large donor pool and a long pre-period, where the noise averages
down.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import DPSC

   df = pd.read_csv(
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/smoking_data.csv"
   )
   df["treat"] = (df["Proposition 99"]).astype(int)

   res = DPSC({
       "df":         df,
       "outcome":    "cigsale",
       "treat":      "treat",
       "unitid":     "state",
       "time":       "year",
       "mechanism":  "objective",   # or "output"
       "epsilon1":   1.0,           # Stage 1 budget (coefficients)
       "epsilon2":   1.0,           # Stage 2 budget (released donors)
       "ridge_lambda": 10.0,
       "n_draws":    500,           # privacy-noise Monte Carlo
       "seed":       0,
       "display_graphs": False,
   }).fit()

   print(res.effects.att)                              # the private ATT (one release)
   print(res.inference.standard_error)                 # the privacy noise
   print(res.inference.details["att_non_private"])     # the epsilon -> infinity target

Verification
------------

DPSC is cross-validated against the authors' own code (``srho1/dpsc``) on the
Proposition 99 panel. Under a shared Mersenne-Twister seed, mlsynth reproduces
the authors' private coefficients and private counterfactual value-for-value for
both mechanisms (the objective program solved in closed form against their
cvxpy), to solver tolerance. See the durable benchmark case
`benchmarks/cases/dpsc_prop99.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/dpsc_prop99.py>`_
and the :doc:`replications/dpsc` page.

Core API
--------

.. automodule:: mlsynth.estimators.dpsc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.utils.dpsc_helpers.config.DPSCConfig
   :members:
   :undoc-members:
   :show-inheritance:
