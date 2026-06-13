Forward-Selected Synthetic Control (FSCM)
=========================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The synthetic control method (SCM) of Abadie, Diamond and Hainmueller
[ABADIE2010]_ builds a treated unit's counterfactual as a convex combination of
donor units. Conventional practice is to start from all available donors
and let the simplex weights zero out the irrelevant ones. Cerulli [FSCM]_ argues
this is often suboptimal: the number of initial donors is itself a complexity
parameter governing a bias--variance trade-off. A richer donor pool fits the
pre-treatment window better *in sample*, but each extra donor that is only
weakly correlated with the treated unit injects variance into the
counterfactual *out of sample* -- the synthetic control overfits the
pre-period and predicts the post-period worse.

``FSCM`` resolves this by treating the donor count as a tuning parameter chosen
by out-of-sample validation. It is the right tool when you have a single
treated unit and a sizeable donor pool and suspect that not all donors
deserve to be in the comparison set -- when you want SCM to tell you *how many*
and *which* donors to use, rather than assuming "more is better." Because the
selection is greedy (forward stepwise), it scales linearly in the pool size,
unlike the :math:`2^N` exhaustive subset search.

The donor (and predictor) weights are computed by the bilevel optimization
of Malo, Eskelinen, Zhou and Kuosmanen [malo2023computing]_, implemented from
scratch in :mod:`mlsynth.utils.fscm_helpers.bilevel` -- no external QP solver
is used. Two switches control the estimator:

* ``forward_selection`` (default ``True``) -- when ``True``, run the greedy
  forward selection with rolling-origin out-of-sample validation, fitting each
  candidate donor set with the bilevel solver. When ``False``, skip selection
  entirely and take the full bilevel solve over all donors (the global SCM
  optimum), reporting the donors that carry weight.
* ``covariates`` / ``match_periods`` -- when given, the estimator runs in
  predictor mode (Abadie's predictor matching with a bilevel-optimized
  predictor-weight matrix :math:`\mathbf{V}`); when omitted, in trajectory
  mode (matching the pre-treatment outcome path). The four combinations all
  run.

Notation
--------

We use the synthetic-control canon. Unit :math:`j=0` is treated and
:math:`\mathcal{N} = \{1, \ldots, N\}` indexes the donors; :math:`\mathbf{y}_0`
is the treated outcome and :math:`\mathbf{Y} = (y_{jt})` the donor matrix. The
intervention occurs after the pre-period
:math:`\mathcal{T}_1 = \{1, \ldots, T_1\}`; the post-period is
:math:`\mathcal{T}_2` with :math:`T_2 = |\mathcal{T}_2|`. For a donor subset
:math:`U \subseteq \mathcal{N}`, the simplex weights solve

.. math::

   \boldsymbol{\omega}^*(U) = \operatorname*{argmin}_{\boldsymbol{\omega}\in\Delta_U}
     \sum_{t\in\mathcal{S}} \bigl(y_{0t} - \mathbf{Y}_{t,U}\,\boldsymbol{\omega}\bigr)^2,
   \qquad
   \Delta_U = \Bigl\{\boldsymbol{\omega}\ge 0 : \textstyle\sum_{j\in U}\omega_j = 1\Bigr\},

fit over a window :math:`\mathcal{S}`, and the root-mean-square prediction error
over an evaluation window :math:`\mathcal{E}` is
:math:`\mathrm{RMSPE}_{\mathcal{E}}(U) = \sqrt{|\mathcal{E}|^{-1}\sum_{t\in\mathcal{E}}
(y_{0t} - \mathbf{Y}_{t,U}\boldsymbol{\omega}^*(U))^2}`.

Computing the weights: bilevel optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With predictors, SCM jointly chooses predictor weights :math:`\mathbf{V}`
(a :math:`K\times K` non-negative diagonal matrix on the simplex) and donor
weights :math:`\mathbf{W}`. Malo et al. [malo2023computing]_ show this is an
optimistic bilevel program: the upper level fits the outcome, the lower
level fits the :math:`\mathbf{V}`-weighted predictors,

.. math::

   \min_{\mathbf{V},\mathbf{W}} \; \tfrac{1}{T_1}\bigl\|\mathbf{y}_0^{\text{pre}}
       - \mathbf{Y}_0^{\text{pre}}\mathbf{W}\bigr\|^2
   \quad\text{s.t.}\quad
   \mathbf{W} \in \operatorname*{argmin}_{\mathbf{W}\in\Delta}
       \bigl\|\mathbf{x}_0 - \mathbf{X}_0\mathbf{W}\bigr\|_{\mathbf{V}}^2 ,

which is NP-hard in general and is the reason off-the-shelf SCM packages can be
numerically unstable. ``mlsynth`` implements the paper's globally-convergent
iterative algorithm in three stages, short-circuiting as soon as an optimum is
certified (the paper notes the optimum is usually a corner found early):

1. Unconstrained feasibility (Section 3.1) -- solve the simplex regression
   of the treated outcome on the donors, giving the lower bound :math:`L(W^{})`
   on the upper-level loss; an LP over :math:`\mathbf{V}` checks whether some
   predictor is already matched, which certifies optimality.
2. Corner solutions (Section 3.2) -- evaluate the :math:`K` basic predictor
   weightings (all weight on one predictor) and keep the best by outcome loss.
3. Tykhonov-regularized descent (Section 3.3) -- only if a gap remains,
   descend over :math:`\mathbf{V}` for a vanishing regularization sequence.

The lower-level (and the trajectory-mode) simplex problems are solved by a
self-contained FISTA projected-gradient routine
(:func:`~mlsynth.utils.fscm_helpers.bilevel.simplex_lstsq`), which matches a
reference QP solver to ~1e-8. In predictor mode the optimal :math:`\mathbf{V}`
is computed once on the full donor pool and reused through forward
selection.

The forward stepwise algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``forward_selection=True`` (default), the donor count is chosen by
Cerulli's procedure ([FSCM]_, Table 1):

1. Start from the empty model :math:`U_0 = \varnothing`.
2. For :math:`k = 0, 1, \ldots, J-1`: among the :math:`J-k` candidate donors not
   yet selected, add the one whose inclusion minimizes the in-sample
   pre-period RMSPE, giving the nested model
   :math:`U_{k+1} = U_k \cup \{j^*\}`.
3. For each nested model :math:`U_k`, compute an out-of-sample validation
   RMSPE and select :math:`k^* = \operatorname*{argmin}_k \mathrm{CV}(U_k)`.

The selected donor set is :math:`U_{k^*}`; ``mlsynth`` then refits the weights
on the full pre-period over :math:`U_{k^*}` to form the counterfactual
:math:`\hat{y}_{0t}^0 = \mathbf{Y}_{t,U_{k^*}}\boldsymbol{\omega}^*`, the gap
:math:`\hat{\Delta}_t = y_{0t} - \hat{y}_{0t}^0`, and the ATT
:math:`\bar{\Delta} = \mathcal{E}_{\mathcal{T}_2}(\hat{\Delta}_t)`.

When ``forward_selection=False`` the selection and cross-validation are skipped:
the estimator returns the single full bilevel solve over all donors (the
global SCM optimum), reporting the weight-bearing donors. This is faster and is
the right choice when you want the canonical SCM weights rather than a
parsimonious donor subset.

Rolling-origin cross-validation. Cerulli's paper splits the pre-period once
(early half train, late half test). With short pre-periods (Proposition 99 has
only 19) a single split is noisy, so ``mlsynth`` uses an expanding-window,
one-step-ahead scheme instead: for each origin :math:`t` from
:math:`\lceil T_1\cdot\texttt{cv\_split}\rceil` to :math:`T_1-1`, weights are fit
on :math:`\{1,\ldots,t-1\}` and used to forecast period :math:`t`; the
validation score is the RMSPE of those one-step forecasts. Every late
pre-period period serves as a test point, using the data more efficiently than
a single cut.

Assumptions (forward convex hull + selection consistency)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FSCM relaxes canonical SCM's "the full donor pool must contain a
convex-hull match for the treated unit" to the much weaker "*some*
subset does," and the forward-stepwise selector is the device that
finds it. The four assumptions that make the selector behave:

A1 (forward convex-hull condition -- the identifying premise).
There exists a non-empty subset :math:`U^* \subseteq \mathcal{N}`
and simplex weights :math:`\omega^* \in \Delta_{U^*}` such that the
treated unit's pre-period trajectory is (approximately) reproduced
by the corresponding donor combination,

.. math::

   y_{0t} \;\approx\; \sum_{j \in U^*} \omega_j^*\, y_{jt}
   \quad \text{for all } t \in \mathcal{T}_1.

The classical SCM hull condition is the special case
:math:`U^* = \mathcal{N}`; FSCM operates whenever some subset
(potentially a small one) supplies the hull. If no subset of
controls can form a convex hull around the target, FSCM cannot be
used -- and no amount of forward stepwise will rescue it. The
diagnostic is the lower envelope of in-sample RMSPE across the
nested models :math:`\{U_k\}`: if it never falls below the noise
floor at any size, A1 is failing.

A2 (stable pre/post relationship). The weights that reproduce
the treated unit on the pre-period also reproduce its untreated
trajectory on the post-period -- the standard SCM identification
premise carried through to the selected subset. *Remark.* This is
what licenses using pre-period out-of-sample fit (the
cross-validation test window) as a stand-in for post-period
predictive accuracy of the counterfactual, which is never
observed.

A3 (forward-stepwise selection consistency). Under regularity
conditions on the donor pool and the pre-period length (Shi &
Huang 2023, Theorem 1) [fsPDA], the forward stepwise selection rule
recovers the oracle donor set wpa1:

.. math::

   \mathbb{P}\bigl( U_{k^*} = U^* \bigr) \;\longrightarrow\; 1
   \qquad \text{as } T_1 \to \infty.

*Remark.* This is the wpa1 selection-consistency property
that distinguishes FSCM from heuristic donor pre-screening:
greedy forward steps are not just computationally tractable
(:math:`1 + J(J+1)/2` fits vs. the exhaustive :math:`2^J`), they
asymptotically pick the *right* subset. The regularity conditions
follow Shi & Huang and assume bounded donor signals, mixing
shocks within units, and a signal-strength condition on the
oracle weights (no donor in :math:`U^*` carries vanishing weight
in :math:`\omega^*`).

A4 (informative cross-validation split). The pre-period is
long enough, and the donor / treated dynamics stable enough
across the split, that the test-RMSPE on the held-out interval
is a consistent estimate of out-of-sample prediction error.
*Remark.* mlsynth's expanding-window scheme expects the late
pre-period to resemble the post-period in distribution; if the
panel has a structural break inside the pre-window (a global
financial crisis, a regime change), this assumption fails and the
selected :math:`k^*` will reflect the break, not the donor pool.


Empirical Illustration: California's Proposition 99
---------------------------------------------------

Following Cerulli's application -- the canonical Abadie [ABADIE2010]_ study of
California's 1988 tobacco-control program -- ``FSCM`` runs on ``P99data.csv``
(per-capita cigarette sales, 38 control states, 1970--2000). The treated
indicator is California from 1989 onward.

.. code-block:: python

   import pandas as pd
   from mlsynth import FSCM

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/P99data.csv"
   df = pd.read_csv(url)
   df["treated"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)

   res = FSCM({"df": df, "outcome": "cigsale", "treat": "treated",
               "unitid": "state", "time": "year", "display_graphs": True}).fit()

   print(f"optimal donors : {res.n_selected} of {res.diagnostics['n_donors_available']}")
   print(f"selected       : {res.selected_donors}")
   print(f"ATT            : {res.att:.2f}")
   print(f"pre-period R^2 : {res.diagnostics['pre_r_squared']:.3f}")
   print(f"CV RMSPE       : optimum={res.diagnostics['cv_rmspe_at_optimum']:.3f}"
         f"  full pool={res.diagnostics['cv_rmspe_full_pool']:.3f}")

This prints::

   optimal donors : 3 of 38
   selected       : ['Montana', 'Nevada', 'Utah']
   ATT            : -20.15
   pre-period R^2 : 0.970
   CV RMSPE       : optimum=1.605  full pool=2.916

Forward selection keeps 3 of the 38 donors and estimates a drop of about
20 packs per capita, consistent with Abadie's original synthetic-control
estimate. The key diagnostic is the last line: the rolling-origin CV RMSPE at
the optimum (1.61) is far below using all 38 donors (2.92) -- the
out-of-sample evidence for Cerulli's bias--variance argument, that the smaller
donor set forecasts the treated unit *better* than the full pool. With
``display_graphs=True`` the second panel plots the CV-RMSPE curve against the
number of donors (the paper's Fig. 2--3), with the selected count marked.

Matching on the author's full predictor specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``P99data.csv`` ships with Abadie's predictor specification: the covariates
``lnincome``, ``beer``, ``age15to24``, ``retprice`` and cigarette sales in
1975, 1980 and 1988. Abadie averages the covariates over 1980--1988 (beer over
1984--1988); these aggregation windows are given through ``covariate_windows``,
and the lagged smoking values through ``match_periods``. With
``forward_selection=False`` the estimator returns the full bilevel SCM
optimum over all donors:

.. code-block:: python

   res = FSCM({"df": df, "outcome": "cigsale", "treat": "treated",
               "unitid": "state", "time": "year", "forward_selection": False,
               "covariates": ["lnincome", "beer", "age15to24", "retprice"],
               "covariate_windows": {"lnincome": (1980, 1988), "age15to24": (1980, 1988),
                                     "retprice": (1980, 1988), "beer": (1984, 1988)},
               "match_periods": [1975, 1980, 1988]}).fit()
   print(f"R^2 {res.diagnostics['pre_r_squared']:.4f}  ATT {res.att:.2f}")
   print({str(k): round(v, 3) for k, v in res.donor_weights.items() if v > 1e-3})
   # R^2 0.9787  ATT -19.68
   # {'Utah': 0.386, 'Montana': 0.257, 'Nevada': 0.206, 'Connecticut': 0.107, 'New Hampshire': 0.043}

This reproduces the global optimum reported in Malo et al.
[malo2023computing]_ (their Table 1): :math:`R^2 = 0.979` with donor weights
concentrated on Utah, Montana, Nevada, Connecticut and New Hampshire -- the
solution the standard *Synth* package fails to reach. Consistent with the
paper's central finding, the optimal predictor weights :math:`\mathbf{V}` form a
corner solution that places all weight on a single predictor: many
predictors are interchangeable at the optimum (the upper-level loss is
non-unique in :math:`\mathbf{V}`), so the bilevel solver certifies optimality at
a corner rather than spreading weight across predictors.

.. note::

   With ``forward_selection=True`` *and* predictors, the global :math:`\mathbf{V}`
   is frozen and reused across candidate donor sets. Because that corner
   :math:`\mathbf{V}` may rest on an easily-matched predictor, it can stop
   constraining the subset fits, so the rolling-origin CV may decline to prune
   the pool. For predictor matching, ``forward_selection=False`` (the full
   bilevel optimum) is therefore the more reliable choice; forward selection is
   most useful in trajectory mode.

Verification
------------

.. note::

   Empirical (Proposition 99). In trajectory mode with forward selection,
   ``FSCM`` reproduces Cerulli's regime: a small donor set (3 of 38) with a
   rolling-origin CV RMSPE minimized below the full pool -- the
   out-of-sample signature of the bias--variance trade-off -- and an ATT
   :math:`\approx -20`. With the full Abadie predictor spec and
   ``forward_selection=False``, the bilevel solver reproduces the global SCM
   optimum of Malo et al. [malo2023computing]_ exactly (:math:`R^2 = 0.979`,
   Table 1 donor weights), which the *Synth* package does not reach.

   Solver. The self-contained FISTA simplex solver agrees with a reference
   QP solver to ~1e-8 over random problems; the bilevel ``unconstrained``
   feasibility certificate, corner-solution bounds, and a determinism check are
   unit-tested (``mlsynth/tests/test_fscm_bilevel.py``). All four
   ``forward_selection`` x ``covariates`` combinations are exercised in
   ``mlsynth/tests/test_fscm.py``.

Core API
--------

.. automodule:: mlsynth.estimators.fscm
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.FSCMConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``FSCM.fit()`` returns a
:class:`~mlsynth.utils.fscm_helpers.structures.FSCMResults`, an
:class:`~mlsynth.config_models.EffectResult` on the standardized two-family
contract: the flat accessors ``res.att`` / ``res.counterfactual`` / ``res.gap``
/ ``res.donor_weights`` (the full-pool mapping) / ``res.pre_rmse`` resolve
through the standardized sub-models (``effects`` / ``time_series`` / ``weights``
/ ``fit_diagnostics``). The FSCM-specific detail is carried alongside: the
weight-bearing donor set (``selected_donors``), the raw simplex weight array
(``weights_vector``), the rich fit-diagnostics dict (``diagnostics`` -- pre-RMSE,
R\ :sup:`2`, donor counts, CV stats), and -- when forward selection ran --
a :class:`~mlsynth.utils.fscm_helpers.structures.FSCMSelectionPath` (per-size
in-sample and rolling-origin CV RMSPE, and the selection order;
``None`` when ``forward_selection=False``). The prepared, NumPy-only panel is
exposed as a :class:`~mlsynth.utils.fscm_helpers.structures.FSCMInputs`, with
units and time addressed through an :class:`IndexSet`.

.. note::

   The raw simplex weight array is ``res.weights_vector`` and the rich
   diagnostics dict is ``res.diagnostics``; the bare names ``res.weights`` /
   ``res.fit_diagnostics`` are reserved by the contract for the standardized
   :class:`~mlsynth.config_models.WeightsResults` /
   :class:`~mlsynth.config_models.FitDiagnosticsResults` sub-models.

.. automodule:: mlsynth.utils.fscm_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Data preparation -- the only DataFrame touchpoint: pivots to NumPy, builds the
unit/time ``IndexSet``es, splits pre/post, and assembles the optional covariate
arrays.

.. automodule:: mlsynth.utils.fscm_helpers.setup
   :members:
   :undoc-members:

The forward-selection / rolling-origin cross-validation engine and the
weight-fitting that dispatches to the bilevel solver (predictor mode) or the
trajectory simplex solve.

.. automodule:: mlsynth.utils.fscm_helpers.estimation
   :members:
   :undoc-members:

Plotting: the outcome paths and the donor-count CV-RMSPE selection curve.

.. automodule:: mlsynth.utils.fscm_helpers.plotter
   :members:
   :undoc-members:

Bilevel optimization (Malo et al. 2024)
---------------------------------------

A self-contained implementation of the optimistic bilevel SCM program, used as
the weight solver in predictor mode (and the simplex primitive everywhere). No
external QP solver is involved.

The simplex-constrained least-squares core: Euclidean projection onto the
probability simplex and the FISTA accelerated projected-gradient solver.

.. automodule:: mlsynth.utils.fscm_helpers.bilevel.simplex
   :members:
   :undoc-members:

The three algorithm stages -- unconstrained feasibility, corner solutions, and
Tykhonov-regularized descent.

.. automodule:: mlsynth.utils.fscm_helpers.bilevel.stages
   :members:
   :undoc-members:

The driver composing the stages, and the structured problem/solution
containers.

.. automodule:: mlsynth.utils.fscm_helpers.bilevel.solver
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.fscm_helpers.bilevel.structure
   :members:
   :undoc-members:
   :show-inheritance:
