Branch and bound work — SYNDES and MAREX regression gates
=========================================================

SYNDES's two-way exact backend and MAREX's exact backend both switch from
pricing every treated set to a certified branch and bound once a cluster holds
more than twenty million sets. Their tests check the answer: that the search
returns the design the enumeration would, certified. They do not check what the
answer costs, and a search can do several times the work for the same design
without a test noticing. Two benchmark cases pin that cost.

* `benchmarks/cases/syndes_bnb_work.py
  <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/syndes_bnb_work.py>`_
  runs the two-way search (:mod:`mlsynth.utils.syndes_helpers.branch`).
* `benchmarks/cases/marex_bnb_work.py
  <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/marex_bnb_work.py>`_
  runs MAREX's search (:mod:`mlsynth.utils.marex_helpers.branch`).

Neither replicates a paper. Both are regression gates, path C in the
:doc:`../benchmarks` index.

What is pinned
--------------

Both searches decide where to spend time from a cost model of constants and
counts, never a clock. The work they do is therefore the same on every machine
with the same library versions, so a case can pin it without timing noise:

* SYNDES: the nodes bounded, the lifts computed, the nodes discarded, the
  designs priced and the work in the cost model's seconds. Beside them sit the
  design, whether the search certified it, and whether it is the design the
  enumeration returns.
* MAREX: the bounds computed, the sets priced and the designs fitted. Beside
  them sit the design, its agreement with the enumeration, and its
  certificate's relative gap.

The instances
-------------

SYNDES, in about 50 seconds:

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Panel
     - Treated
     - What it exercises
   * - GeoLift, first 30 and 32 markets
     - 8, 10
     - the default costs: bounds, lifts and pruning as in production
   * - no structure, 30 units
     - 8
     - the default costs on a panel whose bounds prune early
   * - four factors, 30 units
     - 8
     - a flat panel: the bounds prune nothing and every design is priced
   * - GeoLift, no structure, four factors, 20 units
     - 8
     - pricing made expensive, so every node is bounded and lifted
   * - four factors and GeoLift, 26 units
     - ``K=None``
     - every size up to 13 under ``work_limit=5``: the swap-search cutoff,
       best first and the mirror rule of :doc:`../syndes`, under which both
       searches close within the limit

MAREX, in about 25 seconds:

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Panel
     - Treated
     - What it exercises
   * - Walmart, 45 stores
     - 6
     - the default costs: two bounds and the rest priced, as in the
       :doc:`../marex` table
   * - GeoLift, first 20 and 24 markets
     - 5, 6
     - pricing made expensive, so every node is bounded

Tolerances
----------

The pinned values were measured under ``benchmarks/constraints.txt`` (Python
3.11, numpy 2.4.6, scipy 1.17.1), the versions the pull-request suite installs.
They were checked against Python 3.13 with numpy 2.5.3 and scipy 1.18.1.

* MAREX agreed exactly on every count, design and value, so its counts are
  pinned exactly.
* SYNDES agreed exactly on every design, certified flag, number of designs
  priced and size chosen. The nodes bounded differed by one on two instances,
  and the modelled work by up to 6 percent, because L-BFGS-B takes different
  numbers of evaluations in the lift under the two scipy versions. Nodes carry
  a 5 percent band and work a 15 percent band.

The bands absorb that version noise and nothing larger. A search that started
pricing every design, stopped lifting, or bounded nodes in a worse order would
move these numbers by far more than 15 percent.

Mutation targets ``syndes-bnb-work`` and ``marex-bnb-work`` in
``tools/mutation/targets.toml`` confirm it. Their mutants change how much work
the search does and not its answer:

* SYNDES: every node priced, no node lifted, the branching order reversed.
* MAREX: the branching unit chosen last, the rounded guess left unpriced.

The cases fail with every one of them. The unit tests, which check the answer,
pass with three in place: the reversed branching order, MAREX's branching unit
chosen last and the rounded guess left unpriced. They catch the other two
through the tests that pin how the SYNDES search bounds, lifts and prunes.
