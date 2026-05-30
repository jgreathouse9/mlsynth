Modified Unbiased Synthetic Control
===================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Classical synthetic control fits donor weights by minimising the
pre-period prediction error of the treated unit's outcome (Abadie,
Diamond and Hainmueller 2010). The resulting estimator's theoretical
guarantees rest on an outcome model — usually a linear factor model —
that is assumed to govern the data-generating process. In an
applied study you have to defend that model, which is uncomfortable
when the panel really is "the 50 U.S. states" or "the OECD economies"
and there is no defensible random-sampling story.

Use MUSC, due to Bottmer, Imbens, Spiess and Warnick (2024) [MUSC]_,
when

* you have a **single treated unit** and a small donor pool (the
  paper's simulations include :math:`N \in \{5, 10, 50\}`);
* you are willing to commit to a **design-based** view in which
  *which* unit ends up treated is treated as random, even when the
  treatment is observational;
* you want a **finite-sample unbiased** estimator and an
  **unbiased variance estimator**, not just an asymptotic bias bound
  conditional on an assumed factor model;
* you want **randomization-based confidence intervals** that are
  exact in finite samples under the design-based assumption.

MUSC modifies the synthetic-control quadratic programme with a single
additional linear restriction — the column-sums-to-zero condition on
the weight matrix — and that one change is enough to make the
resulting average-treatment-effect estimator exactly unbiased under
random assignment of which unit is treated (Lemma 1).

.. note::

   MUSC is the only estimator in :mod:`mlsynth` that returns an
   **unbiased finite-sample variance estimator**: a closed-form, four-
   term formula (Proposition 1) computed from the weight matrix and
   the observed outcomes at the treated period. No Monte Carlo, no
   placebo loop, no asymptotic approximation.

Notation
--------

We index units by :math:`i = 1, \dots, N` (the paper labels the
treated unit's identity as a random variable; we condition on it
being unit :math:`i_*` when reporting the realised ATT). Time runs
over :math:`t = 1, \dots, T`, with the pre-treatment window
:math:`\mathcal{T}_1 = \{t < t_*\}` and the post-treatment window
:math:`\mathcal{T}_2 = \{t \ge t_*\}`. The observed outcome panel is
:math:`Y \in \mathbb{R}^{T \times N}` with element :math:`Y_{t, j}`.

The Class of Generalised Synthetic-Control Estimators
-----------------------------------------------------

Bottmer et al. write every member of the Generalised Synthetic
Control (GSC) class as a single linear functional of the outcome
matrix parametrised by an :math:`N \times (N+1)` weight matrix
:math:`M`:

.. math::

   \hat\tau(U, V, Y, M)
       \;=\; \sum_{i = 1}^{N} \sum_{t = 1}^{T} U_i V_t
       \left( M_{i, 0} + \sum_{j = 1}^{N} M_{i, j+1} Y_{j, t} \right)
       \;+\; \sum_{i, t} U_i V_t Y_{i, t},

where :math:`U_i \in \{0, 1\}` is the treated-unit indicator
(:math:`\sum_i U_i = 1`), :math:`V_t \in \{0, 1\}` is the treated-
period indicator (:math:`\sum_t V_t = 1`), :math:`M_{i, 0}` is a
per-row intercept and :math:`M_{i, j+1}` is the weight that
candidate-treated unit :math:`i` places on unit :math:`j` when used as
a control. The standard SC, DiM, DiD and MSC estimators are all
recovered by fixing different subsets of the weight matrix; the
distinguishing feature of MUSC is which restriction set
:math:`\mathcal{M}` it searches over (Table 2 of the paper).

The MUSC Weight-Matrix Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Four linear restrictions define the MUSC class:

1. :math:`M_{i, i+1} = 1` for all :math:`i` (the treated-self
   loading).
2. :math:`M_{i, j+1} \in [-1, 0]` for all :math:`i \neq j`
   (non-positive control weights, bounded below).
3. :math:`\sum_{j = 1}^{N} M_{i, j+1} = 0` for every :math:`i`
   (the SC adding-up restriction; equivalent to the canonical
   "weights sum to one" once the sign is flipped).
4. :math:`\sum_{i = 1}^{N} M_{i, j+1} = 0` for every :math:`j`
   (the **MUSC unbiasedness restriction** — this is the one
   constraint that distinguishes MUSC from the modified SC of
   Doudchenko & Imbens (2016)).

The fourth constraint is the only thing that differs from MSC. It
costs almost nothing computationally: MUSC and MSC are *the same QP*
with one extra linear equality.

The Quadratic Programme
^^^^^^^^^^^^^^^^^^^^^^^

Given the pre-treatment outcome matrix :math:`Y_{\text{pre}} \in
\mathbb{R}^{T_0 \times N}`, MUSC solves

.. math::

   \min_M
       \sum_{i = 1}^{N} \sum_{s \in \mathcal{T}_1}
       \left( M_{i, 0} + \sum_{j = 1}^{N} M_{i, j+1} Y_{j, s} \right)^2

subject to restrictions 1-4. The optimisation is convex (quadratic
objective, linear constraints) and is implemented through cvxpy with
``CLARABEL`` as the default solver. The same QP with restriction 4
dropped is the standard SC estimator (under MSC's intercept-free
modification); :mod:`mlsynth.MUSC` returns both fits so the effect of
restriction 4 is visible on the result object.

Assumptions
-----------

**Assumption 1 (Random Assignment of Units).** Conditional on the
potential outcomes :math:`Y(0)` and the treated period, the treated
unit is drawn uniformly at random from the panel.

*Remark.* This is the **design-based** view of Bottmer et al.: even
when the empirical setting (50 U.S. states) is not literally a random
sample, the analyst commits to analysing the data *as if* the
identity of the treated unit had been randomised — a posture
consistent with the placebo-based inference already used throughout
the SC literature (Abadie, Diamond & Hainmueller 2010; Firpo &
Possebom 2018). MUSC's main results require this assumption; the
randomization CIs become exact under it.

**Assumption 2 (Random Assignment of Treated Period; optional).** The
treated period is drawn uniformly at random from the candidate
intervention periods.

*Remark.* Strengthens unbiasedness from being conditional on
:math:`V` (the treated period) to being unconditional on :math:`(U,
V)`. Useful when the treated period was itself chosen for
exchangeability reasons rather than by deliberate selection.

The MUSC Bias Theorem
---------------------

**Lemma 1.** Under Assumption 1, if one of (a) the intercept
:math:`M_{i, 0}` is zero, or (b) the intercept is unconstrained and
fitted from the data, *and* the weight set :math:`\mathcal{M}`
imposes :math:`\sum_i M_{i, j+1} = 0` for every :math:`j`, then the
GSC estimator is exactly unbiased:

.. math::

   \mathbb{E}_{U} [ \hat\tau \mid V ] - \tau \;=\; 0.

*Remark.* This is the theoretical justification for MUSC. Adding the
single column-sums-to-zero restriction to the otherwise standard SC
weight matrix removes the entire bias term identified in equation
3.2. In :mod:`mlsynth.MUSC` we confirm this *empirically*: across 50
panel draws the MUSC bias is **0.000000** to machine precision on
every draw, while the same QP without the column-sum restriction
shows visible per-panel bias (see the
:ref:`Verification <musc-verification>` section below).

The Unbiased Variance Estimator (Proposition 1)
-----------------------------------------------

Under Assumption 1, the *exact* conditional variance of any GSC
estimator with a time-invariant constraint set is

.. math::

   \mathbb{V}(V, M)
       \;=\; \mathbb{E}_U\!\left[
           (\hat\tau(U, V, Y, M) - \tau)^2 \mid V
       \right]
       \;=\; \frac{1}{N} \sum_{i = 1}^{N}
           V_t \left( M_{i, 0} +
           \sum_{j = 1}^{N} M_{i, j+1} Y_{j, t} \right)^2,

and Proposition 1 of the paper gives a closed-form **unbiased
estimator** :math:`\hat{\mathbb{V}}` of this variance that depends
only on the realised outcomes at the treated period and the weight
matrix. The expression has four terms (eq. 3.3 of the paper); the
implementation in
:func:`mlsynth.utils.musc_helpers.unbiased_variance` is a direct port
of the paper's MATLAB reference ``var_gsc_intercept.m``.

For comparison the placebo-based variance estimator commonly used in
the SC literature is *biased* — its sign depends on the data (Section
3.4 of the paper) — while Proposition 1 is unconditionally unbiased
under Assumption 1.

Randomization-Based Confidence Intervals (Section 3.5)
------------------------------------------------------

The unbiased variance gives a natural Normal-approximation CI:

.. math::

   \hat\tau \;\pm\; z_{1-\alpha/2} \sqrt{\,\hat{\mathbb{V}}\,}.

For finite-sample exactness, however, Bottmer et al. propose
inverting a permutation test on the placebo distribution. For each
non-treated unit :math:`j`, the leave-one-out estimator is refit
pretending :math:`j` is treated, giving a placebo ATT
:math:`\hat\tau_j`. Under random assignment the placebo ATTs are
draws from the null distribution; the inverted ``(1 - \alpha)``
interval based on their order statistics is

.. math::

   \tau \in \big[\hat\tau - \hat\beta_{(N(1 - \alpha / 2))},\;
                   \hat\tau - \hat\beta_{(N \alpha / 2)}\big],

where :math:`\hat\beta_{(k)}` is the :math:`k`-th order statistic of
the centred placebo distribution. :class:`mlsynth.MUSC` reports both
the Normal CI (``inference.ci_normal``) and the randomization CI
(``inference.ci_randomization``); the latter is the default surfaced
as ``results.att_ci``. Table 6 of the paper shows the randomization
CIs attain nominal coverage in their CPS simulation while Normal-
approximation CIs may mildly under- or over-cover.

Example
-------

A self-contained Monte Carlo: simulate a small linear-factor panel
under :math:`H_0` (no treatment effect anywhere), fit MUSC, and
inspect the column-sum diagnostic that confirms the unbiasedness
constraint binds.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import MUSC

   # ---- one-draw factor panel (15 units x 25 periods, T0 = 20).
   rng = np.random.default_rng(0)
   N, T_pre, T_post = 15, 20, 5
   T = T_pre + T_post
   mu = rng.normal(0.0, 0.5, size=N)
   eta = rng.normal(0.0, 1.0, size=T); f = np.zeros(T)
   for t in range(1, T):
       f[t] = 0.7 * f[t - 1] + eta[t]
   lam = rng.normal(1.0, 0.3, size=N)
   eps = rng.normal(0.0, 1.0, size=(T, N))
   Y = mu[None, :] + f[:, None] * lam[None, :] + eps

   # ---- pivot into mlsynth's expected long form.
   rows = []
   for j in range(N):
       for t in range(T):
           rows.append({
               "unit": f"u{j:02d}", "time": t, "y": float(Y[t, j]),
               "treat": int(j == 0 and t >= T_pre),
           })
   df = pd.DataFrame(rows)

   res = MUSC({
       "df": df, "outcome": "y", "treat": "treat",
       "unitid": "unit", "time": "time",
       "display_graphs": False, "run_inference": True, "seed": 0,
   }).fit()

   print(f"SC   col-sum |max abs|: {res.fits['SC'].column_sum_residual:.4f}")
   print(f"MUSC col-sum |max abs|: {res.fits['MUSC'].column_sum_residual:.2e}")
   print(f"ATT       SC={res.fits['SC'].att:+.3f}  MUSC={res.fits['MUSC'].att:+.3f}")
   print(f"V̂ (Prop 1)               = {res.inference.variance:.3f}")
   print(f"95% randomization CI = ({res.att_ci[0]:+.3f}, {res.att_ci[1]:+.3f})")

The MUSC column-sum is at machine precision (~1e-14) while the SC
column-sum is materially non-zero — the unbiasedness constraint is
binding exactly. The Proposition 1 variance estimator returns a
finite, non-negative number, and the randomization CI is built from
the leave-one-out placebos.

.. _musc-verification:

Verification
------------

**Empirical replication against the authors' Lemma 1 (Path B).** The
paper's headline theoretical claim is that the MUSC ATT estimator is
**exactly unbiased** under random unit assignment (Lemma 1), while
the standard SC estimator is biased. The architectural test in
:file:`mlsynth/tests/test_musc.py::TestLemma1Replication`
reproduces this on the paper's linear-factor data-generating process
and confirms that the empirical bias of MUSC is at machine precision
on every Monte Carlo draw, while SC's bias is bounded away from zero.

.. code-block:: python

   import numpy as np
   from mlsynth.utils.musc_helpers import (
       att_for_unit, solve_musc_qp,
   )

   def factor_panel(rng, N=10, T_pre=20, T_post=3,
                      rho=0.7, sigma=1.0, mu_std=0.5):
       T = T_pre + T_post
       mu = rng.normal(0.0, mu_std, size=N)
       eta = rng.normal(0.0, 1.0, size=T); f = np.zeros(T)
       for t in range(1, T): f[t] = rho * f[t - 1] + eta[t]
       lam = rng.normal(1.0, 0.3, size=N)
       eps = rng.normal(0.0, sigma, size=(T, N))
       return mu[None, :] + f[:, None] * lam[None, :] + eps

   biases = {"SC": [], "MUSC": []}
   for rep in range(50):
       rng = np.random.default_rng(rep)
       Y = factor_panel(rng); T0 = 20
       M_sc, _   = solve_musc_qp(Y[:T0], column_balance=False)
       M_musc, _ = solve_musc_qp(Y[:T0], column_balance=True)
       # Exact expectation under random unit assignment.
       sc_atts   = np.array([att_for_unit(M_sc,   Y, i, T0)[2]
                              for i in range(Y.shape[1])])
       musc_atts = np.array([att_for_unit(M_musc, Y, i, T0)[2]
                              for i in range(Y.shape[1])])
       biases["SC"].append(sc_atts.mean())
       biases["MUSC"].append(musc_atts.mean())

   print(f"SC   bias: max|E_U[τ̂]| over 50 panels = "
          f"{np.abs(biases['SC']).max():.3e}")
   print(f"MUSC bias: max|E_U[τ̂]| over 50 panels = "
          f"{np.abs(biases['MUSC']).max():.3e}")

prints::

   SC   bias: max|E_U[τ̂]| over 50 panels = 3.5e-01
   MUSC bias: max|E_U[τ̂]| over 50 panels = 1.7e-15

MUSC's bias is at machine precision on *every* one of the 50 panel
draws — not just small on average — because the column-sum
restriction analytically annihilates the bias formula 3.2,
irrespective of the panel. SC's bias varies by panel and reaches a
maximum of ~0.35 in magnitude.

**Unbiased variance estimator validation.** Proposition 1 says
:math:`\mathbb{E}_Y[\hat{\mathbb{V}}] = \mathbb{E}_Y[\mathrm{Var}_U[\hat\tau]]`
across DGPs. The test in
:file:`mlsynth/tests/test_musc.py::TestProposition1Replication` runs
50 panels and checks that ``mean(V̂) / mean(Var_U)`` lies in
``[0.85, 1.15]``; empirically the ratio sits around ``0.97-1.00``.

Core API
--------

.. automodule:: mlsynth.estimators.musc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.MUSCConfig
   :members:
   :undoc-members:

Result Containers
-----------------

:meth:`mlsynth.MUSC.fit` returns a
:class:`~mlsynth.utils.musc_helpers.structures.MUSCResults` whose
``fits`` map holds one
:class:`~mlsynth.utils.musc_helpers.structures.MUSCVariantFit` per
restriction-set (``"SC"`` and ``"MUSC"``), and whose ``inference``
attribute is a
:class:`~mlsynth.utils.musc_helpers.structures.MUSCInference`
containing the Proposition 1 variance, the Normal-approximation CI,
the randomization-based CI, and the leave-one-out placebo ATTs.

.. automodule:: mlsynth.utils.musc_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

The long-DataFrame -> NumPy boundary, the cvxpy quadratic programme,
the Proposition 1 variance estimator, the randomization-based CI, the
result-assembly orchestration, and the treated-vs-counterfactual
plotter — one module each.

.. automodule:: mlsynth.utils.musc_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.musc_helpers.estimation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.musc_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.musc_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.musc_helpers.plotter
   :members:
   :undoc-members:

References
----------

.. [MUSC]
   Bottmer, L., Imbens, G. W., Spiess, J., and Warnick, M. (2024).
   "A Design-Based Perspective on Synthetic Control Methods."
   *Journal of Business & Economic Statistics* 42(2), 762-773.
   DOI: 10.1080/07350015.2023.2238788.
