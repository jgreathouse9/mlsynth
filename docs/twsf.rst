Two-Way Synthetic Forecasting (TWSF)
====================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Every other estimator in mlsynth fills in a missing cell *inside* the panel you
already have. Abadie's synthetic control asks what California would have done
without Proposition 99 over 1989 to 2000, a window you observed for every other
state. Synthetic interventions generalises which column is missing, but the cell
is still inside the observed window.

``TWSF``, due to Shen [TWSF]_, asks for a cell that is not in the panel at all:
the outcome of a unit that has never had the intervention, at a date that has
not happened yet.

Reach for it when the decision is prospective. Six markets already run the
programme, a seventh has not, and someone has to decide this month whether to
switch it on. The retrospective question -- what did the six get -- is answered
by the rest of the library. The prospective one is not.

Two things have to be true for the question to be answerable at all, and both
can be checked before fitting:

* Some units have already had the intervention, and have run under it long
  enough to show how it behaves. TWSF learns the treated regime's dynamics from
  them, so a genuinely novel intervention is out of reach.
* The horizon is short. The recoverability condition tightens as the forecast
  extends, so this is a two-to-four-week tool in daily data, not a quarterly
  planning forecast.

Why two halves are needed
-------------------------

Neither parent method can do this alone, and seeing why is the fastest route
into the algorithm.

Synthetic control cannot extrapolate. It builds a counterfactual by borrowing
donors' contemporaneous outcomes, and at :math:`T + 1` there are no donor
outcomes yet. There is nothing to borrow.

Time-series forecasting cannot counterfactualise. You can forecast the target's
own continuation from its own history, but that history is entirely untreated.
There is no treated path to continue.

What closes the gap is that the donors *have* been treated. Their
post-adoption trajectories carry the dynamics of the regime the target is about
to enter. TWSF takes the target's *position* from the cross-section and the
regime's *dynamics* from the donors.

Notation
--------

Units are :math:`i \in \{1, \dots, N\}`, with the focal unit written :math:`N`
and the treated donor pool :math:`\mathcal{I}_1` of size :math:`N_1`. Time runs
:math:`t \in \{1, \dots, T\}`. The donors adopt after :math:`T_0`, so
:math:`T_1 = T - T_0` is the treated window; the focal unit never adopts.

Outcomes follow a latent factor model,

.. math::
   Y_{it}(d) = \langle \mathbf{u}_i, \mathbf{v}_t(d) \rangle + \varepsilon_{it},

where :math:`\mathbf{u}_i` is a unit factor, :math:`\mathbf{v}_t(d)` a time
factor under intervention :math:`d \in \{0, 1\}`, and :math:`\varepsilon_{it}`
idiosyncratic noise. The estimand is the focal unit's *treated* potential
outcome past the end of the panel, :math:`Y_{N, T+h}(1)`, or a prespecified
linear summary :math:`\theta_\eta = \sum_\ell \eta_\ell Y_{N, T+\ell}(1)` over a
fixed horizon.

Assumptions
-----------

1. Latent factor model. Outcomes are generated as above, with
   :math:`\mathbf{u}_i` carrying no dependence on :math:`t` or on :math:`d`.

   Remark. The invariance of :math:`\mathbf{u}_i` to :math:`d` is what licenses
   the whole exercise. It means a cross-unit relationship learned while everyone
   is under control still holds once the intervention is on, which is why the
   unit side can be fitted on the pre-adoption window and applied to a regime
   the focal unit has never occupied.

2. Span. The focal unit's factor lies in the span of the donors',
   :math:`\mathbf{u}_N \in \mathrm{span}\{\mathbf{u}_j : j \in \mathcal{I}_1\}`.

   Remark. This is the familiar synthetic-control requirement wearing different
   clothes, and it fails the same way: a focal unit outside the donors' convex
   or linear reach cannot be reconstructed from them. A poor pre-adoption fit is
   the visible symptom.

3. Low-rank temporal structure. The treated time factors
   :math:`\mathbf{v}_t(1)` admit a linear recursion of order at most :math:`L`.

   Remark. This is the assumption that separates TWSF from imputation, and it
   has no counterpart in synthetic control. It says the treated regime evolves
   according to a rule stable enough to be learned from the donors and applied
   to the target. A sum of :math:`q` harmonics satisfies it with
   :math:`L \ge 2q`, which is also how the estimator is tested for exactness.

4. Recoverability. The terminal lag block, and its recursively shifted versions
   out to the horizon, lie in the span of the temporal training data.

   Remark. This is what tightens as the horizon grows, and it is the formal
   reason the method is for short horizons. At :math:`h = 1` it reduces to the
   one-step condition.

5. Common adoption date. Every donor adopts at the same :math:`T_0`.

   Remark. Real panels rarely satisfy this and the paper's own application does
   not. ``TWSF`` therefore warns and proceeds on the approximate mapping the
   paper uses: the unit side closes at the first adoption date, when everyone is
   still under control, and the time side opens at the last, so every donor is
   treated across the whole training window. Prefer a common-date donor pool
   when one is available.

The algorithm
-------------

Four steps, two of which are ordinary regressions.

The unit side denoises the donors' pre-adoption block by hard singular value
thresholding at rank ``k_y`` and regresses the focal unit's pre-adoption
outcomes onto it, giving weights :math:`\widehat{\beta}` over donors.

The time side cuts each donor's treated series into non-overlapping blocks of
length :math:`L + 1`. Within a block the first :math:`L` entries are a lag
vector and the last is its one-step-ahead response; stacking the donors
horizontally gives the Page matrix. Denoising at rank ``k_z`` and regressing
response on lags gives the temporal rule :math:`\widehat{\alpha}`.

The donors' terminal :math:`L` observations form :math:`\mathbf{W}`. Then
:math:`\mathbf{W}^\top \widehat{\beta}` is the focal unit's *imputed treated
state* -- what its recent history would have looked like had it already adopted
-- and the forecast is

.. math::
   \widehat{\theta} = \langle \widehat{\alpha}, \mathbf{W}^\top \widehat{\beta} \rangle .

Beyond one step the horizon is covered either recursively, iterating the
one-step rule through its companion matrix, or directly, fitting a separate rule
per lead. Recursive keeps the full temporal sample size and is the default;
direct avoids recursive error propagation but needs blocks of length
:math:`L + h` and so is often infeasible at short treated windows. The two
coincide at ``horizon=1``.

Inference and diagnostics
-------------------------

``TWSF`` reports a pointwise interval from the paper's plug-in variance. The
recursive form carries a Jacobian term that propagates one-step estimation error
through the recursion, which is why the band widens faster than the one-step
band scaled by the horizon.

Two interval kinds are available and they answer different questions. The
default ``interval="confidence"`` is for the *expected* counterfactual
trajectory. ``interval="prediction"`` adds the future innovation variance and is
what a validation exercise needs, where the comparison is against a realised
trajectory that carries its own noise.

Three diagnostics can be read off the result. ``sigma2`` in
``fit_diagnostics.additional_metrics`` is the pooled residual variance, and it
pools both sides: on panels where the donors are much larger in scale than the
focal unit the time side can dominate it and widen every interval.
``n_page_blocks`` says how much temporal training data the Page construction
actually recovered, which is the binding constraint at short treated windows.
And ``staggered_adoption`` in ``method_details.parameters_used`` records whether
assumption 5 held.

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import TWSF
   from mlsynth.config_models import TWSFConfig

   rng = np.random.default_rng(0)
   n_donors, T0, T1 = 12, 60, 160
   U = rng.standard_normal((n_donors, 2))
   w = rng.dirichlet(np.ones(n_donors))
   t = np.arange(1, T0 + T1 + 1)
   basis = np.vstack([np.sin(2 * np.pi * t / 11), np.cos(2 * np.pi * t / 11)])
   V0, V1 = rng.standard_normal((2, 2)) @ basis, rng.standard_normal((2, 2)) @ basis

   rows = []
   for i in range(n_donors):
       for k in range(T0 + T1):
           v = V1[:, k] if k >= T0 else V0[:, k]
           rows.append(dict(unit=f"d{i}", period=k + 1, y=U[i] @ v,
                            adopted=int(k >= T0)))
   for k in range(T0 + T1):
       rows.append(dict(unit="focal", period=k + 1, y=(w @ U) @ V0[:, k],
                        adopted=0))
   panel = pd.DataFrame(rows)

   results = TWSF(TWSFConfig(
       df=panel, outcome="y", unitid="unit", time="period", treat="adopted",
       target="focal", L=10, k_y=2, k_z=4, horizon=5,
       display_graphs=False)).fit()

   print(results.time_series.counterfactual_outcome)   # the forecast treated path
   print(results.weights.donor_weights)                # who the focal unit resembles
   print(results.weights.time_weights)                 # the treated regime's rule

Verification
------------

Validated by :doc:`replications/twsf`, whose durable form is the benchmark case
`benchmarks/cases/twsf_coverage_mc.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/twsf_coverage_mc.py>`_.

Core API
--------

.. autoclass:: TWSF
   :members: fit

.. autoclass:: mlsynth.config_models.TWSFConfig
   :members:
