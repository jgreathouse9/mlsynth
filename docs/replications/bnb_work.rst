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
counts, never a clock, so a case can pin their work without timing noise. The
counts still rest on floating point, and one SYNDES instance's differ between
processors (see Tolerances):

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

The same versions give two sets of SYNDES counts, by processor. OpenBLAS picks
its kernels by CPU. Under each kernel a GitHub runner can select, the counts
fall into two families: the AVX2 kernels (Haswell, and Zen, which AMD runners
select) and the AVX-512 ones, whose counts are the pins. Every design and
certified flag agrees, and MAREX agrees on every count.

* ``geolift20_k8_eager`` is the one instance that leaves its bands. Its root
  lift, an L-BFGS-B solve of the regularised dual, stops after 553 evaluations
  under AVX2 and 647 under AVX-512, at bounds 1.3e-5 apart relative. Both bounds
  are valid, but the children start from different multipliers. The AVX2 search
  bounds 46 nodes, lifts 39 and prices 3 designs where the AVX-512 one bounds
  42, lifts 33 and prices 1, for the same certified design. That instance's
  bands span both families.
* Elsewhere the families agree on every count and differ by at most 1.9
  percent in work, inside the bands.

The bands are drawn for the processors the pull-request suite runs on. On an
older processor whose OpenBLAS selects the Sandybridge kernel (AVX without FMA),
the SYNDES case fails. At default costs ``geolift30_k8`` prices 6.7 times as
many designs and ``noise30_k8`` 1.72 times as many, with every design and
certified flag unchanged. Bands that wide would let a doubling of the work
through on the processors the gate runs on.

The SYNDES pins moved once for a defect in the lift. It divided a node's parts
by the largest of them, the fixed units' constant included. The constant grows
with the units fixed, so the free part shrank against the lift's regularisation
schedule, and deep nodes stopped at the trust-region bound where the relaxation
was tighter. Scaled by the free part alone, more nodes prune. ``geolift30_k8``
prices 407,330 designs where it priced 1,479,291, and ``geolift32_k10`` 8,008
where it priced 221,442. Every design and certified flag is unchanged.

The bands absorb that noise and nothing larger. A search that started
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
