Parallel-Trends Supergeo Design (PANGEO)
=========================================

.. currentmodule:: mlsynth

Overview
--------

PANGEO is a prospective **experimental-design** estimator for geographic
(geo) experiments, in the lineage of **Supergeo Design** (Chen,
Doudchenko, Jiang, Stein & Ying 2023) and its scalable successor
*Optimized Supergeo Design* (Shaw 2025). It keeps the Supergeo idea --
group geos into composite "supergeos", form balanced **pairs**, randomise
treatment within each pair, and **trim no geo** -- including the
set-partitioning mixed-integer program that selects the design.

What changes is the **matching objective**. Supergeo matches supergeos on
a *scalar* aggregate (the summed response); OSD balances a handful of
*scalar covariate summaries*. Both collapse the time dimension. Two
candidate groups can have identical totals yet completely different
*shapes* over the pre-period -- which a downstream
difference-in-differences or synthetic-control analysis (that differences
trajectories) treats as not interchangeable at all.

PANGEO matches on the full pre-treatment **trajectory** instead. It
chooses the partition whose treatment and control halves are as
**parallel as possible** over the pre-period, scored by the DiD
pre-period residual sum of squares -- the *level-removed gap variance*:

.. math::

   \text{score}(A, B) = \sum_{t}\big[(\bar Y_{A,t}-\bar Y_{B,t})-\delta\big]^2,
   \qquad \delta = \overline{(\bar Y_A - \bar Y_B)} ,

for a supergeo pair split into halves :math:`A` (treatment) and
:math:`B` (control) with mean trajectories :math:`\bar Y_A, \bar Y_B`.
This is exactly the pre-period residual of a DiD fit (cf.
:func:`mlsynth.utils.selector_helpers._did_from_mean`). Minimising it
makes the halves run parallel; the **level shift** :math:`\delta` is
absorbed, so two supergeos can differ in level yet match perfectly on
*shape* -- precisely what parallel-trends DiD needs, and what scalar
matching discards.

Multi-arm support
^^^^^^^^^^^^^^^^^

A single categorical column names each geo's eligible treatment arm
(e.g. values ``A`` / ``B`` / ``C``). Arms occupy non-overlapping geos, and
PANGEO designs **each arm independently** -- only same-arm geos are
combined into supergeos. The output is a **design** (supergeo pairs +
treatment/control assignment + achieved parallelism), not a treatment
effect; PANGEO is run *before* the experiment, on historical sales.

Algorithm
---------

For each arm:

1. **Enumerate** admissible supergeo pairs over the arm's geos -- subsets
   that split into two halves each of size :math:`\le Q` -- scoring each
   by its best-split level-removed gap variance.
2. **Solve** the set-partitioning MIP (cvxpy + HiGHS): choose pairs that
   cover every geo exactly once, minimising total non-parallelism

   .. math::

      \min_{x\in\{0,1\}}\ \sum_G \text{score}(G)\,x_G
      \quad\text{s.t.}\quad M^\top x = \mathbf 1,\ \mathbf 1^\top x \ge \kappa .

3. Within each chosen pair, treatment/control is the score-minimising
   split (randomised in the field).

Setting ``max_supergeo_size = 1`` (``Q = 1``) recovers the classic
matched-pairs design; ``Q > 1`` allows composite supergeos when no single
geo matches another well -- without trimming.

Core API
--------

.. automodule:: mlsynth.estimators.pangeo
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.PANGEOConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.pangeo_helpers.parallelism
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.mip
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.simulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.structures
   :members:
   :undoc-members:

Example
-------

A seasonal, multi-arm sales panel (the bundled simulator), designed into
parallel supergeo pairs. With ``display_graphs=True`` PANGEO plots each
arm's treatment vs control aggregate pre-period trajectories.

.. code-block:: python

   from mlsynth import PANGEO
   from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel

   # 3 arms (non-overlapping geos), 6 geos each, 156 weeks of history.
   df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                  T=156, seed=0)

   res = PANGEO({
       "df": df,
       "outcome": "sales",
       "arm": "arm",                # single categorical arm column
       "unitid": "unit",
       "time": "time",
       "max_supergeo_size": 3,      # Q
   }).fit()

   for arm, design in res.arm_designs.items():
       print(f"Arm {arm}: {len(design.pairs)} pair(s), "
             f"parallel-trends R^2 = {design.mean_parallelism_r2:.3f}")
       for p in design.pairs:
           print(f"   T={p.treatment}  C={p.control}  R^2={p.parallelism_r2:.3f}")

   # res.assignment maps every geo -> 'treatment' / 'control'.

On the simulated data this returns designs with parallel-trends R^2 around
0.90-0.98 -- roughly 10-35x more parallel than a random treatment/control
split of the same geos.

References
----------

Chen, A., Doudchenko, N., Jiang, S., Stein, C., & Ying, B. (2023).
"Supergeo Design: Generalized Matching for Geographic Experiments."
arXiv:2301.12044.

Shaw, C. (2025). "Optimized Supergeo Design: A Scalable Framework for
Geographic Marketing Experiments." arXiv:2506.20499.

Li, K. T. (2023). "Frontiers: A Simple Forward Difference-in-Differences
Method." *Marketing Science* 43(2):267-279.

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American
Statistical Association* 105(490):493-505.
