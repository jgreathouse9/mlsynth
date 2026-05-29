Multi-Level Synthetic Control (mlSC)
====================================

.. currentmodule:: mlsynth

Overview
--------

Multi-Level Synthetic Control (mlSC) is the estimator proposed by
`Bottmer (2025) <https://leabottmer.github.io/job_market/jmp_bottmer.pdf>`_
for panels with two levels of aggregation: an aggregate level at which
treatment is assigned (e.g. states), and a finer disaggregate level at
which outcomes are also observed (e.g. counties within those states).
Such panels are common in policy evaluation — state-wide cigarette and
minimum-wage policies, for instance, are routinely studied with
county-level outcome data — and they raise a question that classical
SC does not address: at *what* level of aggregation should the
estimator operate?

Existing practice spans the full range: some studies aggregate every-
thing to the state level and apply classical SC; others enlarge the
donor pool with all disaggregated control units; still others
disaggregate both treated and control sides. Each choice has a
different bias / noise profile. The mlSC estimator turns the
aggregation choice into a data-driven regularization problem: the
disaggregate weights are allowed to vary freely, but a ridge-type
penalty shrinks them toward the classical-SC solution, with the
penalty strength selected from data.

Two structural properties distinguish mlSC from the rest of the
``mlsynth`` toolkit. **First, it operates on two long-form
DataFrames**, one at the aggregate level (``df_agg``) and one at the
disaggregate level (``df_disagg``), with an ``agg_id`` column on
``df_disagg`` mapping each disaggregate unit to its parent aggregate.
Every other estimator in the package takes a single panel. **Second,
the entire spectrum from classical SC to fully disaggregated SC is
recovered as limiting cases** of a single penalty parameter
:math:`\lambda` — at :math:`\lambda \to \infty` mlSC reduces to the
classical SC of Abadie, Diamond, and Hainmueller (2010); at
:math:`\lambda = 0` it recovers the fully-disaggregated control SC
(``dGSC-AD`` in the paper's notation); intermediate values pick a
data-driven mixture.

Mathematical Formulation
------------------------

Setup
^^^^^

Let :math:`s = 0, 1, \dots, S` index :math:`S + 1` aggregate units
(e.g. states), with :math:`s = 0` the single treated aggregate. Each
aggregate :math:`s` contains :math:`C_s` disaggregate units (e.g.
counties), indexed :math:`c = 1, \dots, C_s`. Time periods are
:math:`t = 1, \dots, T`, with treatment assigned in period
:math:`T_0 + 1` and absorbing thereafter (treatment never turns off).
Let :math:`Y_{sct}` denote the disaggregate outcome and :math:`Y_{st}`
the aggregate outcome. The two are linked by population aggregation
weights :math:`v_{sc}` summing to one within each aggregate:

.. math::

   Y_{st} \;=\; \sum_{c = 1}^{C_s} v_{sc} \, Y_{sct},
   \qquad
   \sum_{c = 1}^{C_s} v_{sc} = 1.

The default in :class:`mlsynth.config_models.MLSCConfig` is uniform
weights :math:`v_{sc} = 1 / C_s`; non-uniform population weights can
be supplied through the ``weight_col`` field.

The dGSC Family and the Aggregation Choice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Section 4 of the paper introduces the *disaggregated general SC*
(``dGSC``) class, parametrized by a weight matrix
:math:`W_{c_0 s c}` indexed by treated disaggregate units
:math:`c_0`, control aggregates :math:`s`, and control disaggregates
:math:`c`:

.. math::

   \hat\tau_{0t}^{\,\text{dGSC}}
   \;=\;
   \sum_{c_0 = 1}^{C_0} v_{0 c_0} \left(
       Y_{0 c_0 t}
       -
       \sum_{s = 1}^{S} \sum_{c = 1}^{C_s} W_{c_0 s c} \, Y_{sct}
   \right),

with :math:`W` chosen to minimize the pre-treatment squared error
subject to convex-hull constraints
:math:`\sum_{s, c} W_{c_0 s c} = 1` and :math:`W \geq 0`. The four
edge cases discussed in Sections 3-4 of the paper correspond to
constraints on :math:`W`:

* **dGSC-AA** — aggregate treated, aggregate controls (the *classical
  SC* of Abadie et al., 2010). All within-block weights are
  proportional to :math:`v_{sc}` and equal across treated
  disaggregates.
* **dGSC-AD** — aggregate treated, disaggregate controls. The
  treated unit is kept at the aggregate level (weights equal across
  :math:`c_0`); disaggregate control weights vary freely within
  convexity.
* **dGSC-DA** — disaggregate treated, aggregate controls.
* **dGSC** — disaggregate on both sides.

Proposition 1 of the paper shows that all four cases share the same
classical-SC objective, differing only in the *donor pool* implied by
the weight constraints. Section 6 demonstrates empirically that
*disaggregating the controls* is the primary source of estimation
gains; disaggregating the treated unit alone usually worsens
performance (Jensen's inequality: matching an average is easier than
matching each component). For this reason, the mlSC estimator
implemented in :mod:`mlsynth` focuses on the aggregate-treated,
flexible-disaggregate-controls regime — the dGSC-AD case — with a
penalty that shrinks toward classical SC.

The mlSC Objective
^^^^^^^^^^^^^^^^^^

With the treated unit kept aggregated (the paper's preferred variant,
Equation 5.2), the weight matrix collapses to a single vector
:math:`\omega_{sc}` over all disaggregate control units. Let
:math:`w_s = \sum_c \omega_{sc}` denote the implied aggregate weight
on donor unit :math:`s`. The mlSC estimator solves

.. math::

   \min_{\omega \,\geq\, 0,\; \mathbf{1}^\top \omega \,=\, 1}
   \quad
   \underbrace{
     \sum_{t = 1}^{T_0}
       \left(
         Y_{0t}
         -
         \sum_{s = 1}^{S} \sum_{c = 1}^{C_s}
           \omega_{sc} \, Y_{sct}
       \right)^{2}
   }_{\text{pre-treatment fit}}
   \;+\;
   \lambda \, \hat\sigma_y^2
   \underbrace{
     \sum_{s = 1}^{S} \sum_{c = 1}^{C_s}
       \left(
         \omega_{sc} - v_{sc} \, w_s
       \right)^{2}
   }_{\text{hierarchical penalty}}.

The penalty term measures how far the disaggregate weight vector
deviates from "*population-share times aggregate weight*". When
:math:`\lambda \to \infty` the penalty forces
:math:`\omega_{sc} = v_{sc} \, w_s`, which collapses the estimator to
classical SC at the aggregate level. When :math:`\lambda = 0` the
disaggregate weights are free to vary within the convex hull,
recovering the fully-disaggregated dGSC-AD estimator. The scale factor
:math:`\hat\sigma_y^2` keeps :math:`\lambda` itself scale-invariant
across panels.

The Block-Diagonal Penalty Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stacking the within-block penalty terms gives a clean quadratic
form. For each aggregate :math:`s`, let
:math:`\omega_s \in \mathbb{R}^{C_s}` be the within-block weight
vector and :math:`v_s \in \mathbb{R}^{C_s}` the corresponding
population weights. Defining :math:`A_s = I - v_s \mathbf{1}^\top`,
the within-block contribution is

.. math::

   \sum_{c = 1}^{C_s} \left( \omega_{sc} - v_{sc} \, w_s \right)^{2}
   \;=\;
   \| A_s \, \omega_s \|_2^2
   \;=\;
   \omega_s^\top Q_s \, \omega_s,

with

.. math::

   Q_s
   \;=\;
   A_s^\top A_s
   \;=\;
   I \;-\; v_s \mathbf{1}^\top \;-\; \mathbf{1} v_s^\top
   \;+\; \| v_s \|_2^{\,2} \, \mathbf{1} \mathbf{1}^\top.

The full penalty matrix
:math:`Q \in \mathbb{R}^{M \times M}`, with :math:`M = \sum_s C_s`
the total number of disaggregate control units, is block-diagonal
across :math:`s` with blocks :math:`Q_s`. :math:`Q_s` is symmetric
positive semidefinite with :math:`v_s` in its kernel (any
weight vector proportional to :math:`v_s` incurs zero penalty), so
the mlSC optimization remains a convex QP and admits standard
``cvxpy`` solvers. :mod:`mlsynth` uses ``SCS`` by default; this
matches the reference implementation and avoids any commercial-solver
dependency.

Selecting :math:`\lambda`: Heuristic and Fixed Modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two penalty-selection rules are exposed via
:class:`mlsynth.config_models.MLSCConfig.lambda_est`:

* ``'heuristic'`` — the closed-form rule from Section 5.2 of the
  paper,

  .. math::

     \hat\lambda
     \;=\;
     2 \, \hat\sigma_\varepsilon^2 \,/\, \hat\sigma_y^2,

  derived from the oracle optimal :math:`\lambda^*` in a stylized
  hierarchical random-effects model under :math:`T = S = C_s = 2`
  (Appendix B of the paper). Intuitively, the heuristic imposes a
  larger penalty when the panel is noisier, since added flexibility
  is more likely to overfit noise than to extract signal.

* ``'fixed'`` — use a user-supplied :math:`\lambda` directly. Useful
  for sensitivity analysis (sweep :math:`\lambda` and inspect how
  the aggregate weight pattern shifts between classical-SC and
  disaggregated-SC regimes) and for reproducing the paper's grid-
  search experiments.

The cross-validation-over-time rule from Section 5.2 is not yet
exposed in v1 of the :mod:`mlsynth` implementation; it is on the
short-term roadmap.

Variance Decomposition (Appendix G)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The heuristic requires plug-in estimates of
:math:`\sigma_\varepsilon^2` and :math:`\sigma_y^2`. Following the
paper's Appendix G, these are obtained from a simplified hierarchical
random-effects model :math:`Y_{sct} = \alpha_s + \eta_{sc} +
\varepsilon_{sct}` fit to the pre-treatment slice of ``df_disagg``.
For each control aggregate :math:`s` (the treated aggregate is
excluded so the post-period unobservability does not contaminate the
estimates), define

.. math::

   \hat\mu_{sc}
   \;=\;
   \frac{1}{T_0} \sum_{t = 1}^{T_0} Y_{sct},
   \qquad
   \widehat{\mathrm{Var}}(\varepsilon)_s
   \;=\;
   \frac{1}{C_s T_0}
   \sum_{c = 1}^{C_s} \sum_{t = 1}^{T_0}
   (Y_{sct} - \hat\mu_{sc})^{2},

and

.. math::

   \widehat{\mathrm{Var}}(y)_s
   \;=\;
   \frac{1}{C_s T_0}
   \sum_{c = 1}^{C_s} \sum_{t = 1}^{T_0}
   \left( Y_{sct} - \overline{Y}_{s\cdot\cdot} \right)^{2},

with :math:`\overline{Y}_{s\cdot\cdot}` the pre-treatment grand mean
over aggregate :math:`s`. The plug-in estimates are then the
non-treated averages :math:`\hat\sigma_\varepsilon^2 =
\mathrm{mean}_s \widehat{\mathrm{Var}}(\varepsilon)_s` and
:math:`\hat\sigma_y^2 = \mathrm{mean}_s \widehat{\mathrm{Var}}(y)_s`.
The implementation in
:func:`mlsynth.utils.mlsc_helpers.variance.estimate_variance_components`
matches the reference Python package and the paper's description
to the letter.

Counterfactual and Effect Summaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the optimal :math:`\omega`, the aggregate counterfactual is

.. math::

   \hat Y_{0t}(0)
   \;=\;
   \sum_{s = 1}^{S} \sum_{c = 1}^{C_s}
     \omega_{sc} \, Y_{sct},
   \qquad t = 1, \dots, T,

and the average treatment effect on the treated aggregate is

.. math::

   \widehat{\mathrm{ATT}}
   \;=\;
   \frac{1}{T - T_0}
   \sum_{t = T_0 + 1}^{T}
     \left( Y_{0t} - \hat Y_{0t}(0) \right),

reported as :attr:`MLSCResults.att`. The pre-period RMSE between
the observed treated series and its mlSC reconstruction,
:math:`\mathrm{RMSE}_{\text{pre}} = \big[\frac{1}{T_0}
\sum_{t \le T_0} (Y_{0t} - \hat Y_{0t})^2\big]^{1/2}`, is reported
as :attr:`MLSCResults.pre_rmse`.

Two-DataFrame API and Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because the aggregate-level and disaggregate-level data live at
different units of observation, they are passed as two separate
long-form DataFrames sharing the same ``outcome``, ``time``, and
``treat`` column names. ``df_disagg`` additionally carries an
``agg_id`` column mapping each disaggregate unit to its parent
aggregate, and an optional ``weight_col`` for non-uniform
:math:`v_{sc}`. The configuration validator enforces four
cross-DataFrame invariants:

1. **Treatment timing alignment.** The pre-period count :math:`T_0`
   implied by ``df_agg`` must match the one implied by ``df_disagg``;
   any disagreement raises :class:`mlsynth.exceptions.MlsynthDataError`.
2. **Treated-aggregate consistency.** Every disaggregate unit with
   ``treat = 1`` must map (via ``agg_id``) to the same aggregate
   unit, which must equal the treated aggregate in ``df_agg``.
3. **Single treatment cohort.** All treated disaggregate units must
   share the same treatment start period (treatment-never-turns-off
   is part of the mlSC framework).
4. **Aggregate-label coverage.** Every value of ``agg_id`` in
   ``df_disagg`` must appear as a ``unitid_agg`` in ``df_agg``.

These invariants are enforced once in
:func:`mlsynth.utils.mlsc_helpers.setup.prepare_mlsc_inputs`, which
reuses :func:`mlsynth.utils.datautils.dataprep` on both panels
(including the disaggregate cohorts shape) and then assembles the
matrices the rest of the pipeline consumes.

Core API
--------

.. automodule:: mlsynth.estimators.mlsc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.MLSCConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.mlsc_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.mlsc_helpers.variance
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.mlsc_helpers.penalty
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.mlsc_helpers.optimization
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.mlsc_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.mlsc_helpers.plotter
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.mlsc_helpers.structures
   :members:
   :undoc-members:

The README hierarchical-factor DGP from the author's reference package,
packaged as ``simulate_mlsc_sample`` so the *Verification* replication
runs as a one-liner.

.. automodule:: mlsynth.utils.mlsc_helpers.simulation
   :members:
   :undoc-members:

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import MLSC

   # Two long-form panels sharing outcome / time / treat column names.
   #
   #   df_agg     — one row per (state, year): the aggregate panel.
   #   df_disagg  — one row per (county, year): the disaggregate panel.
   #                Must contain an 'agg_id' column mapping each county
   #                to its parent state, and may optionally carry a
   #                'population' column for non-uniform v_sc.

   df_agg    = pd.read_csv("state_panel.csv")
   df_disagg = pd.read_csv("county_panel.csv")

   config = {
       "df_agg":         df_agg,
       "df_disagg":      df_disagg,
       "outcome":        "sales",
       "time":           "year",
       "treat":          "treated",        # binary 0/1, equal at both levels
       "unitid_agg":     "state",          # column in df_agg
       "unitid_disagg":  "county_fips",    # column in df_disagg
       "agg_id":         "state",          # column in df_disagg
       "weight_col":     "population",     # optional; defaults to 1 / C_s
       "lambda_est":     "heuristic",      # or "fixed" with lambda_val
       "display_graphs": True,
   }

   results = MLSC(config).fit()

   # Point estimate, fit diagnostic, learned penalty.
   print(results.att)                          # mean post-period aggregate gap
   print(results.pre_rmse)                     # pre-period RMSE
   print(results.design.lambda_used)           # selected lambda
   print(results.design.sigma_eps2,
         results.design.sigma_y2)              # Appendix-G variance estimates

   # Disaggregate and implied aggregate weights.
   print(results.donor_weights)                # {county_fips: omega_sc}
   print(results.aggregate_donor_weights)      # {state:        w_s = sum_c omega_sc}

   # Counterfactual trajectory (length T) and the observed - counterfactual gap.
   cf  = results.inference.counterfactual
   gap = results.inference.gap

   # Sensitivity: sweep lambda by hand.
   for lam in [0.0, 1e-2, 1e-1, 1.0, 10.0, 1e6]:
       r = MLSC({**config, "lambda_est": "fixed", "lambda_val": lam}).fit()
       print(f"lambda={lam:>8.4f}  ATT={r.att:.4f}  preRMSE={r.pre_rmse:.4f}")

Verification
------------

**Empirical replication against the author's reference code (Path A)
plus a Monte Carlo unbiasedness check (Path B).** Bottmer's published
empirical application — the effect of a state minimum-wage policy
change on county-level teen employment in Iowa — uses the Quarterly
Workforce Indicators (QWI) panel that ships with her reference
package (``tests/ia_emp_app_teen_empl.csv``). The simulation block in
her README spells out a hierarchical linear-factor DGP that's free of
proprietary data and is *the* design ``mlSC_estimator`` is calibrated
against. Both validations are wired in below.

The DGP is packaged in
:func:`mlsynth.utils.mlsc_helpers.simulation.simulate_mlsc_sample`:

.. math::

   y_{s, c, t} = (\alpha_s + \eta_{sc}) \cdot f_t + \varepsilon_{sct},
   \quad
   y_{s, t} = \frac{1}{C_s} \sum_c y_{s, c, t},

with :math:`f_t \sim \mathcal{N}(0, 1)`, :math:`\alpha_s \sim
\mathcal{N}(0, 0.8^2)`, :math:`\eta_{sc} \sim \mathcal{N}(0, 0.5^2)`,
and :math:`\varepsilon_{sct} \sim \mathcal{N}(0, 0.3^2)` at the
README defaults :math:`N_s = 10`, :math:`C_s = 10`, :math:`T = 20`.
The simulation never adds a treatment effect, so the true ATT is
exactly zero and the Monte Carlo target is unbiasedness.

Path A: value-for-value vs the reference code (one draw)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the README's exact panel (``np.random.default_rng(42)``), mlsynth's
``MLSC`` reproduces both the selected penalty and the reported ATT to
solver tolerance:

.. code-block:: python

   import numpy as np
   from mlsynth import MLSC
   from mlsynth.utils.mlsc_helpers.simulation import simulate_mlsc_sample
   from multi_level_sc_estimator.mlSC import mlSC_estimator       # reference

   s = simulate_mlsc_sample(rng=np.random.default_rng(42))
   tau_ref, lam_ref, _ = mlSC_estimator(
       s.data_s, s.data_c, s.idx, s.n_c, s.t, s.w_c,
       lambda_est="heuristic")
   res = MLSC({"df_agg": s.df_agg, "df_disagg": s.df_disagg,
                "outcome": "y", "time": "time", "treat": "treated",
                "unitid_agg": "state", "unitid_disagg": "county",
                "agg_id": "state", "lambda_est": "heuristic",
                "display_graphs": False}).fit()
   print(f"REFERENCE  tau_hat = {tau_ref:+.6f}  lambda = {lam_ref:.6f}")
   print(f"MLSYNTH    tau_hat = {res.att:+.6f}  "
         f"lambda = {res.design.lambda_used:.6f}")

prints::

   REFERENCE  tau_hat = +0.011928  lambda = 1.970185
   MLSYNTH    tau_hat = +0.011930  lambda = 1.970185

Differences: :math:`|\hat\tau_{\text{mlsynth}} - \hat\tau_{\text{ref}}|
= 1.5 \times 10^{-6}`,
:math:`|\hat\lambda_{\text{mlsynth}} - \hat\lambda_{\text{ref}}| =
4.4 \times 10^{-16}` — i.e. solver noise on the ATT and machine
precision on the penalty.

Path B: unbiasedness across 200 draws
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Looping the comparison across :math:`M = 200` draws confirms (a) both
estimators are unbiased for the true zero ATT, and (b) they remain in
near-machine agreement throughout:

.. code-block:: python

   def one_rep(seed):
       s = simulate_mlsc_sample(rng=np.random.default_rng(seed))
       tau_ref, _, _ = mlSC_estimator(
           s.data_s, s.data_c, s.idx, s.n_c, s.t, s.w_c,
           lambda_est="heuristic")
       res = MLSC({"df_agg": s.df_agg, "df_disagg": s.df_disagg,
                    "outcome": "y", "time": "time", "treat": "treated",
                    "unitid_agg": "state", "unitid_disagg": "county",
                    "agg_id": "state", "lambda_est": "heuristic",
                    "display_graphs": False}).fit()
       return float(tau_ref), float(res.att)

   import numpy as np
   tau_pairs = np.array([one_rep(s) for s in range(200)])
   tau_ref, tau_ml = tau_pairs[:, 0], tau_pairs[:, 1]
   print(f"reference: mean={tau_ref.mean():+.4f}  std={tau_ref.std(ddof=1):.4f}  "
         f"RMSE={np.sqrt((tau_ref**2).mean()):.4f}")
   print(f"mlsynth  : mean={tau_ml.mean():+.4f}   std={tau_ml.std(ddof=1):.4f}  "
         f"RMSE={np.sqrt((tau_ml**2).mean()):.4f}")
   print(f"max |Δ| = {np.abs(tau_ml - tau_ref).max():.2e}, "
         f"median |Δ| = {np.median(np.abs(tau_ml - tau_ref)):.2e}")

prints::

   reference: mean=-0.0103  std=0.1663  RMSE=0.1662
   mlsynth  : mean=-0.0103  std=0.1663  RMSE=0.1662
   max |Δ| = 5.76e-04, median |Δ| = 1.05e-05

The Monte Carlo standard error of the mean at :math:`M = 200` is
:math:`\sigma / \sqrt{200} \approx 0.012`, so the observed bias of
:math:`-0.010` is well inside one MC SE of zero — the heuristic
:math:`\hat\lambda` does not introduce systematic bias, and ``mlsynth``
and the reference impl produce statistically identical samples.

Empirical application
^^^^^^^^^^^^^^^^^^^^^

For reference, Bottmer's empirical application — Iowa's 2007 minimum-
wage hike on county-level teen employment, run on the Quarterly
Workforce Indicators panel that ships with her reference package
(``tests/ia_emp_app_teen_empl.csv``) — uses the same heuristic-lambda
configuration. Because the panel is public, mlsynth's ``MLSC`` can be
pointed at it directly through the two-DataFrame API; see the
estimator's docstring for the data-cleaning conventions the package
expects (drop counties with any NaN over the analysis window, mirror
the state-level treatment indicator on each contained county, normalize
within-state weights to sum to one).

References
----------

Bottmer, L. (2025). *Synthetic Control with Disaggregated Data.*
Stanford University job-market paper.
`PDF <https://leabottmer.github.io/job_market/jmp_bottmer.pdf>`_.

The reference Python implementation by the paper's author,
`multi-levelSC <https://pypi.org/project/multi-levelSC/>`_
(Apache-2.0), inspired several details of this rewrite — in
particular the variance-decomposition implementation and the
classical-SC warm-start used by the cvxpy solver.

Abadie, A., Diamond, A., and Hainmueller, J. (2010). "Synthetic
Control Methods for Comparative Case Studies: Estimating the Effect
of California's Tobacco Control Program." *Journal of the American
Statistical Association* 105(490):493-505.
