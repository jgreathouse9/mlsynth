PROPSC — Treatment Effects on Proportions with Synthetic Controls
=================================================================

When to use it
--------------

Reach for PROPSC when the outcome is not a single number but a *composition* — a
vector of proportions that sum to a whole. Party vote shares that sum to 100
percent, a budget split across spending categories, market share across brands,
or a turnout-versus-abstention breakdown are all compositional. The defining
feature is a sum constraint: if one share goes up under treatment, another must
come down, so the treatment effects on the components must sum to zero.

The usual practice is to run a separate synthetic control for each component.
The trouble is that each fit picks its own donor weights, so every component is
compared against a *different* synthetic control. The estimated effects then
need not sum to zero, which is logically incoherent for shares of a whole — an
artefact of the method, not the data. PROPSC removes the incoherence by fitting
a single set of donor (and, for synthetic difference-in-differences, time)
weights *jointly* across all components, so every share is compared against the
same synthetic control. This "constant control comparison" makes the estimated
effects sum to zero by construction, following Bogatyrev and Stoetzer (2026).

If you only care about one component in isolation and the confounders are
specific to that component, a single-outcome estimator such as :doc:`sdid` or
:doc:`scmo` may be preferable. PROPSC is for when you want a coherent read of the
*whole* composition at once.

Notation
--------

We observe a balanced panel of :math:`N` units over :math:`T` periods. The
outcome for unit :math:`i` at time :math:`t` is a vector of :math:`K`
proportions :math:`(y_{it1}, \dots, y_{itK})` with
:math:`0 \le y_{itk} \le 1` and :math:`\sum_{k=1}^{K} y_{itk} = 1`. A block of
treated units switches on simultaneously in the last :math:`T - T_0` periods;
the remaining :math:`N_0` control units are never treated. The target is the
average treatment effect on the treated for each component,

.. math::

   \tau_k = \mathbb{E}\!\left[y_{iTk}(1) - y_{iTk}(0) \mid D_i = 1\right],
   \qquad k = 1, \dots, K,

where :math:`y_{iTk}(d)` is the potential outcome under treatment status
:math:`d`. Because the proportions sum to one in both potential states, the
effects obey the sum constraint :math:`\sum_{k=1}^{K} \tau_k = 0`.

PROPSC estimates each :math:`\tau_k` with the synthetic-DID double difference
using *common* unit weights :math:`\omega_j` (over the :math:`N_0` controls, on
the simplex) and *common* time weights :math:`\lambda_t` (over the :math:`T_0`
pre-periods), shared across all :math:`K` components:

.. math::

   \hat{\tau}_k =
   \Big( \tfrac{1}{N_1}\!\!\sum_{i : D_i = 1}\!\! y_{iTk}
         - \sum_{j=1}^{N_0} \omega_j\, y_{jTk} \Big)
   - \sum_{t=1}^{T_0} \lambda_t
     \Big( \tfrac{1}{N_1}\!\!\sum_{i : D_i = 1}\!\! y_{itk}
           - \sum_{j=1}^{N_0} \omega_j\, y_{jtk} \Big).

The weights are chosen to fit the pre-treatment trajectories jointly across all
components, so the same :math:`\omega` and :math:`\lambda` enter every
:math:`\hat{\tau}_k`.

Assumptions
-----------

1. Compositional outcome. The :math:`K` components are non-negative and sum to a
   constant (one) in every unit-time cell.

   Remark. Code an "other" or "abstention" category explicitly so the vector is
   complete; the sum-to-zero coherence only holds for the full composition.

2. Simultaneous adoption. All treated units adopt in the same period with no
   reversals, and the control units are never treated (a single treatment
   block).

   Remark. Staggered adoption is out of scope for PROPSC; use :doc:`sdid` or
   :doc:`seq_sdid` for a single outcome under staggered timing.

3. Pre-treatment fit. A convex, common combination of control units can track
   the treated units' pre-treatment paths across all components jointly.

   Remark. Common weights need only fit the composition *as a whole* well; exact
   fit for every single component is not required and rarely achievable.

4. No anticipation / stable composition. Potential outcomes in the pre-period
   are unaffected by the (future) treatment, and the set of components is fixed.

   Remark. A newly formed party with no past votes can still be carried as a
   component coded zero pre-treatment; the common weights borrow strength from
   the other components.

Inference and diagnostics
-------------------------

The default standard errors are the fixed-weights jackknife of Arkhangelsky et
al. (2021), adapted to the multivariate estimator: unit weights are held fixed
(renormalised over the retained controls) and the :math:`K`-vector of effects is
recomputed on each leave-one-unit-out panel. Set ``inference="none"`` to skip
it. The jackknife is undefined with a single treated unit (it returns ``NaN``);
the paper also discusses bootstrap and placebo alternatives.

The headline diagnostic is ``sum_constraint`` — the sum of the estimated effects
across components, which is zero to machine precision by construction and is a
direct check that the composition is coherent.

Example
-------

A one-draw compositional panel from a latent-factor DGP: :math:`K = 3`
proportions built by a softmax of unit-and-time latent factors, with a treatment
bump on the first component for a block of treated units. PROPSC recovers the
three effects and reports a sum of zero.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import PROPSC

   rng = np.random.default_rng(0)
   N, T, K, N0, T0 = 30, 8, 3, 24, 5
   load = rng.standard_normal((N, K))
   rows = []
   for i in range(N):
       for t in range(T):
           lat = 0.4 * load[i] + 0.15 * t * np.arange(K)
           treated = i >= N0
           if treated and t >= T0:
               lat[0] += 1.0                     # true effect on component 1
           p = np.exp(lat - lat.max())
           p = p / p.sum()
           row = {"unit": f"u{i:02d}", "time": t,
                  "treat": int(treated and t >= T0)}
           for k in range(K):
               row[f"share{k + 1}"] = float(p[k])
           rows.append(row)
   df = pd.DataFrame(rows)

   res = PROPSC({
       "df": df, "outcomes": ["share1", "share2", "share3"],
       "treat": "treat", "unitid": "unit", "time": "time",
       "method": "sdid", "display_graphs": False,
   }).fit()

   print("effects per component:", np.round(res.att_vector, 4))
   print("sum of effects (coherence):", round(res.sum_constraint, 12))
   print("effect on share1:", round(res.att, 4), "95% CI:", res.att_ci)

The positive effect lands on ``share1`` (the component that was bumped) and is
offset by declines in the others, so ``sum_constraint`` is zero.

Verification
------------

PROPSC reproduces the authors' R package ``propsdid`` cell by cell. The
benchmark ``benchmarks/cases/propsc_spain.py`` installs ``propsdid`` at a pinned
commit and diffs ``PROPSC.fit()`` against a live run on the paper's Spain "Just
Transition" panel, reproducing Table 2 (common-weights column) to roughly
:math:`10^{-11}`. See :doc:`replications/propsc` for the full validation,
including the Poland application.

Core API
--------

.. autoclass:: mlsynth.PROPSC
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.config_models.PROPSCConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.utils.propsc_helpers.structures.PROPSCResults
   :members:
   :undoc-members:
   :show-inheritance:

References
----------

Bogatyrev, K., and L. F. Stoetzer (2026). "Estimating Treatment Effects on
Proportions with Synthetic Controls." Political Analysis.

Arkhangelsky, D., S. Athey, D. A. Hirshberg, G. W. Imbens, and S. Wager (2021).
"Synthetic Difference-in-Differences." American Economic Review 111(12).
