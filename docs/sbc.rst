Synthetic Business Cycle (SBC)
==============================

.. currentmodule:: mlsynth

Overview
--------

Synthetic Business Cycle (SBC)
`arXiv:2505.22388 <https://arxiv.org/abs/2505.22388>`_ is a
trend-cycle-aware synthetic control estimator for nonstationary
macroeconomic panels. Standard SCM (:doc:`fdid`, :doc:`tssc`,
:doc:`clustersc`) implicitly assumes the untreated potential outcomes
share a common low-rank factor structure across units. When the
outcome is nonstationary — GDP per capita, exchange rates, gasoline
prices, unemployment — that assumption is fragile: a pre-treatment fit
of the treated unit on the donors can look strong purely because
both series are trending, even when the underlying processes are
independent. Shi, Xi, and Xie (2025) call this the *spurious synthetic
control problem* and propose a divide-and-conquer alternative.

SBC decomposes each unit's outcome into a **trend** and a **cycle**
via the Hamilton (2018) filter. Two distinct counterfactual ingredients
are then built from different parts of the panel:

* the **treated unit's own past** is used to forecast its trend
  forward (no donors involved), neutralizing the spurious-regression
  risk for the persistent component;
* the **donor pool's cycles** are combined via classical simplex SCM
  to impute the treated unit's cyclical fluctuations (where the
  common-factor justification of synthetic control is most defensible).

Combining the two pieces yields the post-treatment counterfactual.

Two configurable knobs are exposed:

- ``h``: Hamilton-filter forecasting horizon (paper recommends 2-4
  years; default ``h=2``).
- ``p``: Number of self-lags used by the filter (paper default
  ``p=2``).

A third option, ``weights_mode``, picks between the paper's simplex SCM
on cycles (Eq. (3); default) and an unrestricted vertical-regression
alternative.

Mathematical Formulation
------------------------

Let :math:`Y_{i,t}` denote the observed outcome for unit
:math:`i \in \{1, \dots, N+1\}` at time :math:`t \in \{1, \dots, T\}`.
Unit :math:`i=1` is the treated unit; treatment begins at
:math:`T_0 + 1`. The treated unit has potential outcomes
:math:`Y_{1,t}(1)` and :math:`Y_{1,t}(0)`; the SBC objective is to
impute the unobserved :math:`Y_{1,t}(0)` for :math:`t > T_0`.

Trend-Cycle Decomposition (Hamilton Filter)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each unit's untreated potential outcome is decomposed as

.. math::

   Y_{i,t}(0) \;=\; \tau_{i,t} \;+\; c_{i,t},

where :math:`\tau_{i,t}` captures the persistent (possibly
nonstationary) component and :math:`c_{i,t}` captures stationary
fluctuations. The trend is defined by the linear projection of
:math:`Y_{i,t}(0)` onto a constant and :math:`p` of its lagged values
shifted back by :math:`h` periods (Eq. (2) of the paper):

.. math::

   \tau_{i,t}
   \;\equiv\;
   \alpha_{i,0}
   + \alpha_{i,1} \, Y_{i, t-h}
   + \alpha_{i,2} \, Y_{i, t-h-1}
   + \cdots
   + \alpha_{i,p} \, Y_{i, t-h-p+1},
   \qquad
   c_{i,t} \;\equiv\; Y_{i,t}(0) - \tau_{i,t}.

The coefficients :math:`(\hat\alpha_{i,0}, \dots, \hat\alpha_{i,p})`
are estimated by OLS on pre-treatment data (helper:
``hamilton.fit_hamilton_filter``). The first :math:`h + p - 1`
observations have no defined target and are reported as ``NaN`` in the
returned trend and cycle vectors.

Step 2: Forecasting the Treated Trend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The treated unit's post-treatment trend is extrapolated by applying
its own AR slope coefficients to its observed lags:

.. math::

   \hat\tau_{1,t}
   \;\equiv\;
   \hat\alpha_{1,1} \, Y_{1, t-h}
   + \cdots
   + \hat\alpha_{1,p} \, Y_{1, t-h-p+1},
   \qquad
   T_0 + 1 \;\leq\; t \;\leq\; T_0 + h.

Two structural choices in the paper deserve emphasis. First, the
intercept :math:`\hat\alpha_{1,0}` is *not* applied here — the
projection uses only the slope coefficients on the treated unit's own
lags. Second, no donors are used in this step at all; the treated
trend is forecast entirely from its own history. This is what
inoculates the procedure against spurious comovement: even if the
donor pool's trends are unrelated to the treated unit's, those trends
never enter the forecast.

Step 3: Synthetic Control on Cycles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The treated unit's cyclical component is imputed via standard
synthetic control on the donors' cycles (Eq. (3)):

.. math::

   \hat c_{1,t}
   \;\equiv\;
   \sum_{i=2}^{N+1} \hat w_i \, \hat c_{i,t},
   \qquad t \;\geq\; T_0 + 1,

with weights solved on the pre-treatment cycles:

.. math::

   (\hat w_2, \dots, \hat w_{N+1})
   \;=\;
   \arg\min_{w_2, \dots, w_{N+1}}
   \sum_{t \leq T_0}
   \left( \hat c_{1,t} - \sum_{i=2}^{N+1} w_i \, \hat c_{i,t} \right)^2
   \quad
   \text{subject to}
   \quad
   w_i \geq 0, \;\; \sum_{i=2}^{N+1} w_i = 1.

No intercept is needed: the cycles are mean-zero by construction. The
``weights_mode`` config flag chooses between this simplex form
(default; ``"simplex"``) and an unrestricted vertical regression
with intercept (``"unrestricted"``) discussed in Section 2.1.

Step 4: Counterfactual Outcome and Treatment Effect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trend and cycle are combined to give the SBC counterfactual:

.. math::

   \hat Y_{1,t}(0) \;\equiv\; \hat\tau_{1,t} + \hat c_{1,t},
   \qquad T_0 + 1 \;\leq\; t \;\leq\; T_0 + h.

The estimated treatment effect at period :math:`t > T_0` is

.. math::

   \widehat{\mathrm{TE}}_t \;=\; Y_{1,t} - \hat Y_{1,t}(0),

with average treatment effect on the treated

.. math::

   \widehat{\mathrm{ATT}}
   \;=\;
   \frac{1}{T - T_0}
   \sum_{t = T_0 + 1}^{T}
   \left( Y_{1,t} - \hat Y_{1,t}(0) \right).

Why the Asymmetry?
^^^^^^^^^^^^^^^^^^

Standard SCM treats time and unit dimensions symmetrically: weights
are fit by regressing the treated outcome on contemporaneous donor
outcomes, and the time ordering of pre-intervention periods is
interchangeable. SBC intentionally breaks that symmetry. The trend is
predicted *through time* using the treated unit's own past, exploiting
the persistence of nonstationary signals. The cycle is predicted
*across units* using the donor pool, exploiting the common-factor
structure that the SC framework was originally designed for. This
divide-and-conquer separation is what gives the paper's main theoretical
result — Theorem 1, the asymptotic unbiasedness of
:math:`\hat Y_{1,t}(0)` even under independent unit-root trends and
without cointegration — its bite.

Asymptotic Properties
^^^^^^^^^^^^^^^^^^^^^

Under the cyclical factor structure (Assumption 1)

.. math::

   c_{i,t} \;=\; \lambda_i^\top f_t + \varepsilon_{i,t},

and the high-level filter accuracy condition (Assumption 2), Theorem 1
gives

.. math::

   \hat Y_{1,t}(0) - Y_{1,t}(0)
   \;=\;
   \sum_{i=2}^{N+1} \hat w_i \, (\varepsilon_{i,t} - \varepsilon_{1,t})
   + o_p(1),

so :math:`\hat Y_{1,t}(0)` is asymptotically unbiased for each
post-treatment period. In the special case where the cycles are
driven only by common factors (no idiosyncratic shocks),
:math:`\hat Y_{1,t}(0)` is consistent in addition to unbiased.

Why the Hamilton Filter?
^^^^^^^^^^^^^^^^^^^^^^^^

Section 3.3 of the paper lays out three criteria the trend-cycle
filter should satisfy:

1. **Cyclical residual must be stationary.** Otherwise the SCM step
   on cycles is again vulnerable to spurious regression. Lemma 1
   shows the Hamilton filter satisfies this for both unit-root
   processes and deterministic-trend-plus-noise processes.
2. **Consistent trend and cycle estimates.** The Hamilton filter
   readily satisfies Assumption 2 (Lemma 2): a simple OLS projection
   gives consistent estimates of the population AR coefficients.
3. **One-sided / no future leakage.** The trend extrapolation in Step
   2 depends only on past observations; symmetric filters like the
   Hodrick-Prescott smoother would compromise this.

The current implementation hard-codes the Hamilton filter; a future
extension could expose a ``filter`` parameter on :class:`SBCConfig`
for swappable alternatives.

Core API
--------

.. automodule:: mlsynth.estimators.sbc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SBCConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.sbc_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sbc_helpers.hamilton
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sbc_helpers.trend_forecast
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sbc_helpers.synthetic
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sbc_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sbc_helpers.plotter
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sbc_helpers.structures
   :members:
   :undoc-members:

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import SBC

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/smoking_data.csv"
   data = pd.read_csv(url)

   config = {
       "df": data,
       "outcome": data.columns[2],
       "treat": data.columns[-1],
       "unitid": data.columns[0],
       "time": data.columns[1],
       "h": 2,                       # Hamilton horizon (paper default)
       "p": 2,                       # AR lags (paper default)
       "weights_mode": "simplex",    # or "unrestricted"
       "display_graphs": True,
   }

   results = SBC(config).fit()

   # Headline ATT and per-donor weights
   print(f"ATT: {results.att:.3f}")
   print(results.weights_by_donor)

   # Per-period treatment effect (zero in the pre-period by construction)
   results.treatment_effect

   # Decomposition diagnostics
   results.design.treated_hamilton.coefficients   # alpha_0 .. alpha_p
   results.design.treated_hamilton.cycle_pre      # in-sample cycle
   results.design.trend_forecast                  # post-period treated trend
   results.design.cycle_forecast                  # post-period synthetic cycle
   results.design.counterfactual_post             # trend_forecast + cycle_forecast
   results.design.pre_cycle_rmse                  # SC fit RMSE on cycles

When SBC vs Classical SCM Disagree
----------------------------------

On the California smoking panel, classical SCM (and TASC, which fits a
state-space model to the *level* of the outcome) yield an ATT around
:math:`-19` packs per capita. SBC on the same panel yields an ATT around
:math:`-10`. The paper's argument is that part of the classical
estimator's apparent precision comes from a tight fit of California's
nonstationary level to a weighted combination of other states' levels
— a fit that may persist out-of-sample even when the underlying trends
are unrelated. SBC strips that spurious component out by forecasting
California's trend from its own history and using donors only for the
stationary cycle.

In short: when SBC and classical SCM disagree on a nonstationary
outcome, SBC is the more conservative answer about how much of the
post-treatment gap really reflects the intervention.
