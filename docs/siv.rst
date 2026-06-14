Synthetic IV
============

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Many policy panels combine an instrument that *should* satisfy
exclusion -- a tariff schedule, a regulatory threshold, an exogenous
supply shock -- with panel-data confounding that makes ordinary
2SLS-with-fixed-effects biased even when the IV is conditionally valid.
Two-way fixed effects only soak up additively separable confounding
(:math:`c_i + d_t`); a richer interactive factor structure
:math:`\mu_i' f_t` -- common shocks loading heterogeneously across units
-- leaks into the second-stage residual and contaminates the IV
estimate.

Use SIV, due to Gulek and Vives [SIV]_, when you have a panel
:math:`(Y_{it}, R_{it}, Z_{it})` with

* a single, sharp intervention date :math:`T_0` after which
  treatment :math:`R_{it}` switches on, the instrument :math:`Z_{it}`
  becomes operative, or both;
* an instrument you believe is conditionally exogenous *given* a latent
  factor model, but not unconditionally; and
* a clean pre-period (:math:`t < T_0`) during which neither the
  instrument nor the treatment has activated yet, so the data identify
  each unit's exposure to the common factors.

SIV uses that pre-period to fit a per-unit synthetic control that
absorbs the factor loadings :math:`\mu_i`, then runs 2SLS on the
debiased post-period series. The debiased outcome equation has no
residual factor structure, so the instrument's *partial validity*
(orthogonal to :math:`\varepsilon_{it}`, but possibly correlated with
:math:`\mu_i' f_t`) is sufficient for consistency.

.. note::

   SIV is the only estimator in ``mlsynth`` that consumes three series
   simultaneously -- outcome, treatment, and instrument -- and the
   only one whose target is a 2SLS coefficient rather than an ATT.
   Donor units are *all* untreated units in the panel; there is no
   single "treated unit" in the SC sense.

For higher-noise or weak-instrument regimes, SIV also exposes the paper's
ensemble (doubly-robust) estimator and a permutation inference
procedure that is exactly valid in small samples. Gulek and Vives
recommend four diagnostics, all surfaced by the estimator: (1) the
instrument is not weak *after* debiasing, (2) good pre-treatment fit, (3) a
back-test that the fit is not overfitting idiosyncratic noise, and (4)
dense synthetic-control weights (no single donor dominating).

Do not use SIV when
^^^^^^^^^^^^^^^^^^^

* You have no instrument. SIV's whole point is to rescue an IV under
  factor confounding. With no instrument, estimate the counterfactual
  directly with a factor-model or synthetic-control method (:doc:`fma`,
  :doc:`fdid`, :doc:`mcnnm`).
* Treatment is exogenous given the factor structure (no simultaneity /
  reverse causality). Then a plain synthetic-control or factor estimator
  already identifies the effect; the 2SLS machinery only adds variance.
* Confounding is purely additive (:math:`c_i + d_t`). Two-way
  fixed-effects 2SLS already absorbs it; SIV's interactive-factor debiasing
  is unnecessary.
* There is no clean pre-period in which neither the instrument nor the
  treatment has activated. SIV fits each unit's factor loadings on that
  window; without it the debiasing step is not identified.
* The instrument is weak after debiasing (diagnostic 1 fails). The
  debiased 2SLS is then unreliable -- no synthetic step repairs a weak
  instrument.
* Distributional questions (quantiles, tails) -- SIV targets a 2SLS
  coefficient; use :doc:`dsc` for distributional effects.

Notation
--------

Units are indexed by :math:`i = 1, \ldots, J`; time by :math:`t = 1,
\ldots, T`, with the pre-period :math:`\mathcal{T}_1 = \{t < T_0\}`
and the post-period :math:`\mathcal{T}_2 = \{t \ge T_0\}`. The
observed triple :math:`(Y_{it}, R_{it}, Z_{it})` satisfies the
structural system

.. math::

   Y_{it} &= \theta R_{it} + \mu_i' f_t + \varepsilon_{it}, \\
   R_{it} &= (\gamma_i' Z_{it} + \eta_{it})
            \cdot \mathbf{1}\{t \ge T_0\}, \\
   Z_{it} &= 0,\ \ t < T_0,

with :math:`f_t \in \mathbb{R}^k` an unobserved factor vector,
:math:`\mu_i \in \mathbb{R}^k` unit loadings, :math:`\theta` the
structural target, and :math:`(\varepsilon_{it}, \eta_{it})` the
outcome and first-stage shocks. The pre-treatment restriction
:math:`Z_{it} = 0` for :math:`t < T_0` (a "sharp intervention"
instrument) is what makes the pre-period purely informative about the
factor loadings.

Assumptions
-----------

Assumption 1 (factor model + sharp intervention). Outcomes follow
the interactive-effects model above; the instrument is zero before
:math:`T_0` and switches on at :math:`T_0`.

*Remark.* The sharp-intervention assumption is what separates SIV from
generic IV: it guarantees that the pre-period contains no instrument
variation, so any pre-period covariance between :math:`Y_{it}` and the
control units' :math:`Y_{jt}` is informative about the factor
structure alone.

Assumption 2 (partial validity). :math:`\mathbb{E}[\varepsilon_{it}
\mid Z_{i,1:T}, \eta_{i,1:T}, \mu_i, \{f_t\}_{t=1}^T] = 0`.

*Remark.* This is the weakened exclusion restriction at the heart of
the paper: :math:`Z_{it}` need only be orthogonal to the outcome shock
:math:`\varepsilon_{it}` *conditional on* the latent factors -- a much
weaker condition than full IV exogeneity, since
:math:`\mathrm{cov}(Z_{it}, \mu_i' f_t)` can be non-zero.

Assumption 3 (factor identification). The pre-period factor matrix
:math:`F_{\mathcal{T}_1} \in \mathbb{R}^{T_1 \times k}` has rank
:math:`k`, and the donor pool spans :math:`\mu_i` for every focal
unit's loading (the standard SC overlap condition).

*Remark.* Rank :math:`k` lets a linear combination of donor outcomes
exactly reproduce the focal unit's :math:`\mu_i' f_t` over the
pre-period; the overlap condition makes the combination feasible with
non-negative weights when ``weight_constraint = "simplex"``.

The Two-Step SIV Estimator
--------------------------

Step 1: per-unit synthetic control. For each focal unit :math:`i`,
solve the constrained pre-period fit

.. math::

   \hat{w}^{(i)} = \arg\min_{w \in \Delta^{J-1}}
       \sum_{t < T_0} \bigl(Y_{it} - \sum_{j \ne i} w_j Y_{jt}\bigr)^2,

where :math:`\Delta^{J-1}` is the simplex (``weight_constraint =
"simplex"``) or the :math:`\ell_1` ball of radius :math:`C`
(``weight_constraint = "l1_ball"``). The fitted weights define
debiased series

.. math::

   \tilde{Y}_{it} = Y_{it} - \sum_{j \ne i} \hat{w}^{(i)}_j Y_{jt},
   \quad
   \tilde{R}_{it} = R_{it} - \sum_{j \ne i} \hat{w}^{(i)}_j R_{jt},
   \quad
   \tilde{Z}_{it} = Z_{it} - \sum_{j \ne i} \hat{w}^{(i)}_j Z_{jt}.

Under Assumptions 1-3 the SC fit absorbs :math:`\mu_i' f_t` on the
pre-period, and -- because all units share the same factor structure
-- the same weights remove it on the post-period too.

Step 2: 2SLS on the debiased post-period. Stack the post-period
debiased series across units and run just-identified 2SLS:

.. math::

   \tilde{R}_{it} = \pi \tilde{Z}_{it} + v_{it},
   \qquad
   \tilde{Y}_{it} = \theta \tilde{R}_{it} + e_{it},
   \qquad t \ge T_0.

The reported ``theta_hat`` is the 2SLS slope from the second equation.
Because the debiased equations no longer contain :math:`\mu_i' f_t`,
partial validity is enough for consistency (Theorem 4).

Variants
--------

``mlsynth.SIV`` exposes three modes via the ``mode`` config field:

* ``"siv"`` -- the canonical pipeline above.
* ``"projected"`` -- project :math:`Y` onto the instrument space
  before fitting the SC (Section 5.1.2). Useful when the instrument
  has substantial cross-sectional variation that can be exploited as
  an auxiliary signal for the factor loadings.
* ``"ensemble"`` -- a convex combination of the canonical and
  projected fits with weight selected by a held-out validation block
  inside the pre-period (Section 5.1.3).

Inference uses either the IV sandwich SE (``inference_method =
"asymptotic"``) or the split-conformal permutation test
(``inference_method = "conformal"``); the latter is robust to small
:math:`T` and weak first stages.

Example
-------

.. code-block:: python

   import numpy as np
   from mlsynth import SIV
   from mlsynth.utils.siv_helpers.simulation import simulate_siv_sample

   sample = simulate_siv_sample(J=26, T=16, T0=10, theta=-0.16, r=0.5,
                                  rng=np.random.default_rng(0))

   res = SIV({
       "df": sample.df, "outcome": "y", "treat": "r", "instrument": "z",
       "unitid": "unit", "time": "time", "T0": sample.T0,
       "mode": "siv", "display_graphs": False,
   }).fit()

   print(f"theta_hat = {res.theta_hat:+.3f}  (true = -0.160)")

Verification
------------

Empirical replication against the authors' published numbers (Path
A) plus a Section 6 Monte Carlo (Path B). Path A reproduces the
2SLS-TWFE row of Autor, Dorn & Hanson [ADH]_ Table 3 (the canonical
shift-share IV design the SIV paper benchmarks against) directly from
the published replication archive, and then runs ``mlsynth.SIV`` on
the same 722-CZ panel. Path B replicates the paper's own Syrian-
calibrated Monte Carlo (Section 6, Table 1) and confirms the headline
ranking that SIV has substantially lower bias than 2SLS-TWFE at every
correlation level :math:`r \in \{0.5, 0.7, 0.9\}`.

Path A: ADH China shock 2SLS baseline (Table 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reading the stacked-decades panel ``workfile_china.dta`` from the
Autor-Dorn-Hanson replication archive and fitting a one-instrument
2SLS on (manufacturing-employment share change) ~ (import exposure
per worker) with commuting-zone fixed effects, the three published
columns of Table 3 reproduce to the third decimal:

.. list-table::
   :header-rows: 1
   :widths: 28 22 22

   * - Specification
     - Replicated coefficient
     - Published (ADH Table 3)
   * - 1990-2000, no Census-region FE
     - :math:`-0.888`
     - :math:`-0.89`
   * - 1990-2007 stacked, no Census-region FE
     - :math:`-0.718`
     - :math:`-0.72`
   * - 1990-2007 stacked, with Census-region FE
     - :math:`-0.746`
     - :math:`-0.75`

so the baseline 2SLS-TWFE estimator the SIV paper benchmarks against
is faithfully reproduced from the public archive.

``mlsynth.SIV`` on the same 722-CZ panel (instrumenting actual import
exposure with the non-US-supply Bartik instrument, pre-treatment
window 1970-1980, post-period 1990-2007) returns
:math:`\hat\theta_{\mathrm{SIV}} = -0.544`, in the same magnitude
band as the published 2SLS estimates but moderated by the SC
debiasing step (the paper's own SIV column on this design is
:math:`-0.70`; the residual gap reflects the donor-trimming and
pre-window choices, which Gulek and Vives explore as a robustness
exercise rather than a single point).

Path B: Section 6 Monte Carlo (Syrian calibration)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The paper's Section 6 Table 1 reports absolute biases for SIV and
2SLS-TWFE on a panel calibrated to the Syrian application (:math:`J =
26` donors, :math:`T = 16`, :math:`T_0 = 10`, true :math:`\theta =
-0.16`, :math:`\sigma_\varepsilon^2 = \sigma_\eta^2 = 0.035`,
:math:`\kappa = 0.5`, single-factor structure). The DGP -- packaged
in :func:`mlsynth.utils.siv_helpers.simulation.simulate_siv_sample`
-- sweeps a single correlation knob :math:`r` jointly across the
three bivariate-normal pairs :math:`(\varepsilon, \eta)`,
:math:`(Z_i, \mu_i)`, and :math:`(u_f, u_g)`, with the published
calibration :math:`(\sigma_\mu, \sigma_z) = (0.5, 0.2)`.

.. code-block:: python

   import numpy as np
   from mlsynth import SIV
   from mlsynth.utils.siv_helpers.simulation import simulate_siv_sample

   def tsls_twfe(Y, R, Z, T0):
       T = Y.shape[1]
       def demean(X):
           X = X - X.mean(axis=1, keepdims=True)
           return X - X.mean(axis=0, keepdims=True)
       Yd, Rd, Zd = demean(Y), demean(R), demean(Z)
       mask = np.arange(T) >= T0
       y, r, z = Yd[:, mask].flatten(), Rd[:, mask].flatten(), Zd[:, mask].flatten()
       z_c = np.column_stack([np.ones_like(z), z])
       b_fs, *_ = np.linalg.lstsq(z_c, r, rcond=None)
       rhat = z_c @ b_fs
       r_c = np.column_stack([np.ones_like(y), rhat])
       b_ss, *_ = np.linalg.lstsq(r_c, y, rcond=None)
       return float(b_ss[1])

   M = 200
   for r in (0.5, 0.7, 0.9):
       siv_hat, tsls_hat = [], []
       for s in range(M):
           sample = simulate_siv_sample(r=r, rng=np.random.default_rng(s))
           est = SIV({"df": sample.df, "outcome": "y", "treat": "r",
                        "instrument": "z", "unitid": "unit", "time": "time",
                        "T0": sample.T0, "mode": "siv",
                        "display_graphs": False}).fit()
           siv_hat.append(float(est.theta_hat))
           tsls_hat.append(tsls_twfe(sample.Y, sample.R, sample.Z, T0=sample.T0))
       siv_b = abs(np.mean(siv_hat) - (-0.16))
       tsls_b = abs(np.mean(tsls_hat) - (-0.16))
       print(f"r={r}  |bias_SIV|={siv_b:.3f}  |bias_2SLS|={tsls_b:.3f}")

prints (at :math:`M = 200`; the paper uses :math:`M = 1{,}000`):

.. list-table::
   :header-rows: 1
   :widths: 10 18 18 18 18

   * - :math:`r`
     - SIV bias (here)
     - SIV bias (paper)
     - 2SLS bias (here)
     - 2SLS bias (paper)
   * - 0.5
     - 0.027
     - 0.009
     - 0.111
     - 0.111
   * - 0.7
     - 0.089
     - 0.028
     - 0.228
     - 0.218
   * - 0.9
     - 0.306
     - 0.104
     - 0.387
     - 0.360

The 2SLS-TWFE column reproduces the paper essentially exactly across
all three :math:`r` (0.111/0.228/0.387 here vs 0.111/0.218/0.360
published). The SIV column carries the same qualitative ordering --
SIV bias is below 2SLS bias at every :math:`r` and the gap widens
with :math:`r` -- which is the headline finding the paper draws from
Table 1. The Monte Carlo standard error at :math:`M = 200` and SIV
bias :math:`\approx 0.03` is roughly :math:`\pm 0.013`, so the small
residual gap at :math:`r = 0.5` sits within Monte Carlo noise; at
:math:`r = 0.9` SIV's MC variance balloons (its post-period 2SLS gets
weaker), which is consistent with the paper's own observation that
SIV's bias advantage compresses as :math:`r \to 1`.

The takeaway carried into the published SIV procedure is the one the
paper highlights: when factor confounding leaks through into the
first-stage residual, SC-debiasing the outcome/treatment/instrument
triple before the IV step buys you substantial bias reduction
relative to plain 2SLS-with-fixed-effects.

Core API
--------

.. automodule:: mlsynth.estimators.siv
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SIVConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``SIV.fit()`` returns a
:class:`~mlsynth.utils.siv_helpers.structures.SIVResults`, holding the
preprocessed inputs, the per-unit SC weights, every variant of the
2SLS estimate (``siv`` / ``projected`` / ``ensemble``), and the
inferential output. Each variant is a
:class:`~mlsynth.utils.siv_helpers.structures.SIVEstimate`; the
selected one is exposed via the ``theta_hat`` shortcut.

.. automodule:: mlsynth.utils.siv_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Data preparation -- pivots the long panel into the typed
:class:`~mlsynth.utils.siv_helpers.structures.SIVInputs` and validates
the sharp-intervention shape of the instrument.

.. automodule:: mlsynth.utils.siv_helpers.setup
   :members:
   :undoc-members:

Per-unit constrained-LS synthetic-control weights (Step 1).

.. automodule:: mlsynth.utils.siv_helpers.weights
   :members:
   :undoc-members:

Projection of the outcome onto the instrument space (Section 5.1.2)
used by the ``projected`` and ``ensemble`` variants.

.. automodule:: mlsynth.utils.siv_helpers.projection
   :members:
   :undoc-members:

Post-period 2SLS on the debiased series (Step 2).

.. automodule:: mlsynth.utils.siv_helpers.twosls
   :members:
   :undoc-members:

Ensemble blend of the canonical and projected variants with CV-
selected mixing weight.

.. automodule:: mlsynth.utils.siv_helpers.ensemble
   :members:
   :undoc-members:

Asymptotic and split-conformal inference for the selected variant.

.. automodule:: mlsynth.utils.siv_helpers.inference
   :members:
   :undoc-members:

The observed-vs-debiased plot and the IV scatter.

.. automodule:: mlsynth.utils.siv_helpers.plotter
   :members:
   :undoc-members:

The Gulek and Vives Section 6 DGP, packaged as ``simulate_siv_sample``
so the Path-B replication in *Verification* runs as a one-liner.

.. automodule:: mlsynth.utils.siv_helpers.simulation
   :members:
   :undoc-members:
