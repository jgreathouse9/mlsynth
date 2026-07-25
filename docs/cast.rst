Confidence Intervals under Staggered Adoption (CAST)
====================================================

.. currentmodule:: mlsynth

Overview
--------

Policies rarely arrive everywhere at once. States expand Medicaid in different
years, countries join a trade bloc at different times, stores roll out a
promotion region by region. This is staggered adoption: every unit is
eventually treated (or never), but each on its own schedule.

Most panel estimators answer the pooled question -- what was the average effect
across the treated? CAST (Xia, Yan and Wainwright 2025) answers a finer one.
It returns a confidence interval for *every treated unit in every period*: not
just "expansion reduced uninsurance by 2.6 percent on average in 2019", but
"Louisiana's reduction in 2019 was 3.1 points, with an interval that excludes
zero". Aggregates are still available, and because they are built from the same
entrywise variances they come with standard errors of their own.

When to use this estimator
--------------------------

* Your treatment is staggered and you care about *which* units responded, not
  only the average. Reporting a per-unit effect with an interval is the
  natural output when the units are themselves of interest (states, countries,
  large accounts).
* You want a population-weighted total rather than an unweighted mean --
  "how many people gained coverage", not "the average percentage-point change"
  -- with a standard error attached.
* Your panel is reasonably large. The intervals are asymptotic; see the
  coverage remark under Assumption 3.

Reach for something else when you only need a pooled ATT (:doc:`sdid`,
:doc:`seq_sdid` and :doc:`ppscm` are purpose-built for that and report
event-study aggregates), or when there is no never-treated unit to anchor the
comparison.

Notation
--------

There are :math:`N` units indexed :math:`i` observed over :math:`T` periods
indexed :math:`t`. Unit :math:`i` adopts treatment at time :math:`t_i`, and
treatment is absorbing: once treated, always treated. A unit with
:math:`t_i > T` is never treated.

Write :math:`Y_{i,t}` for the observed outcome and :math:`M^\star_{i,t}` for the
mean outcome that *would* have been observed without treatment. The individual
treatment effect is

.. math::

   \tau_{i,t} = Y_{i,t} - M^\star_{i,t}, \qquad t \ge t_i .

Because :math:`Y_{i,t}` is observed, estimating :math:`\tau_{i,t}` is the same
problem as estimating the counterfactual mean :math:`M^\star_{i,t}`, and a
confidence interval for one is an interval for the other. More general targets
are the weighted effects

.. math::

   \tau_{w,t} = \sum_{i=1}^{N} w_i \, \tau_{i,t} \, \mathbf{1}(t_i \le t),

which recover the average effect on the treated when :math:`w` is uniform over
the treated units, and a population total when :math:`w` counts people.

Mathematical formulation
------------------------

The staircase
^^^^^^^^^^^^^

Sort the units by how long they are treated. The observed cells then form a
descending staircase: the never-treated units contribute a full row, the
earliest adopters the shortest. Everything above the staircase is
counterfactual and must be imputed.

The paper's device is to notice that such a staircase decomposes into
overlapping *four-block* problems. Each has the shape

.. math::

   Y = \begin{pmatrix} Y_a & Y_b \\ Y_c & ? \end{pmatrix},

where :math:`Y_a` (:math:`N_1 \times T_1`) is fully observed, :math:`Y_b` and
:math:`Y_c` extend it right and down, and the bottom-right block is the
unknown. Every unobserved cell in the original panel falls inside one such
sub-problem, so solving the four-block case solves the general one.

Estimation
^^^^^^^^^^

Assume the counterfactual means have low rank :math:`r`, that is
:math:`M^\star = U^\star \Sigma^\star (V^\star)^\top` with :math:`r` factors.
The four-block routine estimates the two subspaces from the observed blocks and
transports one onto the other:

1. the column subspace :math:`\hat U` from a rank-:math:`r` SVD of the left
   blocks :math:`[Y_a; Y_c]`;
2. the row subspace from a rank-:math:`r` SVD of the upper blocks
   :math:`[Y_a \; Y_b]`, which also gives a denoised :math:`\hat M_b`;
3. the imputation, by regressing the denoised block onto the shared subspace,

.. math::

   \hat M_d = \hat U_2 (\hat U_1^\top \hat U_1)^{-1} \hat U_1^\top \hat M_b ,

where :math:`\hat U_1` and :math:`\hat U_2` are the top and bottom row blocks
of :math:`\hat U`. If the mean matrix is exactly rank :math:`r` and the
observations are noiseless, this recovers the missing block exactly.

Inference
^^^^^^^^^

The same subspaces give the uncertainty. Residuals are formed on the observed
entries, :math:`\hat E = (Y - \hat M) \odot A` with :math:`A` the observation
indicator, and the entrywise variance is

.. math::

   \hat\gamma_{i,t} = \sum_{k \le N_1} \hat E_{k,t}^2
       \Bigl[\hat U_{i,\cdot} (\hat U_1^\top \hat U_1)^{-1} \hat U_{k,\cdot}^\top\Bigr]^2
     + \sum_{s \le T_1} \hat E_{i,s}^2
       \Bigl[\hat V_{t,\cdot} (\hat V_1^\top \hat V_1)^{-1} \hat V_{s,\cdot}^\top\Bigr]^2 ,

giving the interval
:math:`\hat M_{i,t} \pm \Phi^{-1}(1 - \alpha/2)\,\hat\gamma_{i,t}^{1/2}`.
The two terms are the noise propagated through the column and row subspace
projections respectively.

Assumptions
-----------

Assumption 1 (Absorbing staggered treatment). Each unit adopts at some
:math:`t_i` and stays treated; at least one unit is never treated.

  Remark. The never-treated units anchor the staircase -- they are the only
  rows observed for the full horizon, so they identify the row subspace over
  the post-treatment periods. mlsynth raises
  :class:`~mlsynth.exceptions.MlsynthDataError` when none is present.

Assumption 2 (Low-rank counterfactual means). :math:`M^\star` has rank
:math:`r`, small relative to :math:`N` and :math:`T`.

  Remark. The rank is the method's one substantive tuning choice. Too small
  under-fits the factor structure and leaves signal in the residual; too large
  lets the factors absorb the treatment effect itself. Set it with ``rank`` or
  choose ``rank_method``: ``"broken_stick"`` (the default, matching the
  authors' reference implementation), ``"svht"`` (Gavish-Donoho optimal hard
  threshold) or ``"ratio"`` (largest spectral gap). Inspecting the singular
  values yourself is worthwhile -- the paper selects by eye from a scree plot.

Assumption 3 (Independent sub-Gaussian noise). The errors
:math:`\varepsilon_{i,t} = Y_{i,t} - M^\star_{i,t}` are independent, mean-zero
and sub-Gaussian across both units and time.

  Remark. Independence across *time* is the demanding half. Macro panels such
  as state-year outcomes are typically serially correlated, which this rules
  out; where that is a concern the intervals should be read as optimistic.
  Coverage is also asymptotic: in our simulations on a rank-2 design, coverage
  of :math:`M^\star` was about 0.82 at :math:`N=30, T=24`, rising to 0.91 at
  :math:`N=60, T=40` and 0.94 at :math:`N=240, T=160` against a nominal 0.95.
  Small panels undercover.

Assumption 4 (Incoherence). The singular vectors of :math:`M^\star` are
delocalised, so no single unit or period dominates the factor structure.

  Remark. Standard in matrix completion. It fails when one unit is effectively
  its own factor -- an outlier so idiosyncratic that no combination of the
  others describes it.

Inference and diagnostics
-------------------------

The entrywise panel. ``effects_panel``, ``std_errors``, ``ci_lower_panel`` and
``ci_upper_panel`` are :math:`N \times T` arrays, finite exactly on the treated
cells and ``nan`` elsewhere. :py:attr:`CASTResults.effect_table` flattens them
into tidy ``(unit, period, effect, se, ci_lower, ci_upper)`` rows.

Per-period aggregates. ``period_effects`` maps each post-treatment period to
``(effect, standard error)`` for the weighted average of the units treated by
then. Supply ``weights`` for a population-weighted version, and set
``renormalize_weights=False`` to report a total rather than a mean.

The significance map. ``significance`` counts, per period, how many treated
units have a significantly positive, significantly negative, or null effect at
``alpha``; :py:meth:`CASTResults.significant_units` returns the units
themselves.

An important caveat on what is being tested. The interval covers the
counterfactual *mean* :math:`M^\star_{i,t}`, so the effect
:math:`\tau_{i,t} = Y_{i,t} - M^\star_{i,t}` still contains that cell's own
idiosyncratic noise :math:`\varepsilon_{i,t}`, which the variance deliberately
excludes. A single unit-period can therefore be flagged as significant on the
strength of its own noise draw. Per-unit significance is best read as
suggestive, with the per-period aggregates -- where the noise averages out --
carrying the evidential weight.

Example
-------

.. code-block:: python

   import numpy as np, pandas as pd
   from mlsynth import CAST

   # staggered panel: two cohorts adopt at t=16 and t=20, ten units never treated
   rng = np.random.default_rng(0)
   N, T, r = 30, 24, 2
   U, V = rng.normal(size=(N, r)), rng.normal(size=(T, r))
   Y = U @ V.T + 0.2 * rng.normal(size=(N, T))
   adopt = np.full(N, T); adopt[10:20] = 16; adopt[20:] = 20
   for i, a in enumerate(adopt):
       if a < T:
           Y[i, a:] += 3.0                      # planted effect

   df = pd.DataFrame(
       [dict(u=f"u{i:02d}", t=t, y=Y[i, t], d=int(t >= adopt[i]))
        for i in range(N) for t in range(T)]
   )

   res = CAST({"df": df, "outcome": "y", "treat": "d", "unitid": "u",
               "time": "t", "rank": 2}).fit()

   print(res.att, res.att_ci)                   # treated average, ~3.0
   print(res.period_effects[16])                # (effect, se) in the first post period
   print(res.significance[16])                  # positive / negative / null counts
   print(res.effect_table[:3])                  # entrywise rows

For a population-weighted total instead of a mean, pass per-unit weights and
turn off renormalisation::

   res = CAST({..., "weights": population.tolist(),
               "renormalize_weights": False}).fit()

Verification
------------

CAST is cross-validated against the authors' own ``CAST-panel`` package and
reproduces their Affordable Care Act analysis. The durable case lives in
`benchmarks/cases/cast_aca.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/cast_aca.py>`_
and the full replication story, including a standard-error discrepancy we found
in the reference package, is on :doc:`replications/cast`.

Core API
--------

.. autoclass:: mlsynth.CAST
   :members: fit

Configuration
-------------

.. autoclass:: mlsynth.config_models.CASTConfig
   :members:
   :exclude-members: model_config, model_fields, model_computed_fields

Results
-------

.. autoclass:: mlsynth.utils.cast_helpers.structures.CASTResults
   :members: effect_table, significant_units

Helper Modules
--------------

.. automodule:: mlsynth.utils.cast_helpers.four_block
   :members: four_block_est, staircase_blocks, entrywise_variance

.. automodule:: mlsynth.utils.cast_helpers.pipeline
   :members: run_cast, select_rank, aggregate_effects
