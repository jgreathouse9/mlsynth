SYNDES — Doudchenko et al. (2021) BLS Monte Carlo
=================================================

.. currentmodule:: mlsynth

Path-B replication of the simulation study (Section 5, Table 1) in

   Doudchenko, Khosravi, Pouget-Abadie, Lahaie, Lubin, Mirrokni, Spiess &
   Imbens (2021). *Synthetic Design: An Optimization Approach to Experimental
   Design with Synthetic Controls.* arXiv:2112.00278.

SYNDES is an **experimental-design** method: an MIP jointly chooses which units
to treat and the synthetic-control weights, before the experiment runs. The
paper ships no public reference code (only the data link), so this is a Path-B
replication of the authors' own Monte Carlo — mlsynth's :class:`~mlsynth.SYNDES`
design modes must attain the paper's reported RMSEs and beat the randomized
baseline.

Data
----

``basedata/urate_cps.csv`` — US Bureau of Labor Statistics state unemployment
rates, **40 months × 50 states**: the exact file named in the paper's footnote 4
(``synth-inference/synthdid .../bdm/data/urate_cps.csv``).

DGP (Section 5)
---------------

Each simulation samples a 10×10 panel (10 random states, 10 consecutive months),
uses 7 pre- and 3 post-periods, and treats :math:`K` units selected by the design
on the pre-periods. A homogeneous additive effect ``0.05`` is added to the treated
units' post-periods; the ATET RMSE is computed over the post-periods, averaged
across simulations, and reported ×1000.

Result
------

100 simulations per cell (seed-fixed; can be raised toward the paper's 500 to
tighten the Monte-Carlo error):

==================  ===========  =================  ===========  =================
Method              K=3 (paper)  K=3 (mlsynth)      K=7 (paper)  K=7 (mlsynth)
==================  ===========  =================  ===========  =================
Per-unit            8.5          8.7                8.3          7.7
Two-way global      8.4          9.2                8.4          9.0
One-way global      8.5          9.2                8.5          9.0
Diff-in-means       12.1         10.2               11.5         9.8
==================  ===========  =================  ===========  =================

(ATET RMSE ×1000.) The paper's **headline reproduces**: all three optimized
design modes attain RMSEs in the paper's ``8–9`` band and beat the randomized
difference-in-means design at both :math:`K`. (The diff-in-means baseline sits a
little below the paper's value because mlsynth uses a plain difference of
means, not the paper's regularized per-unit synthetic control with random
assignment; the design-beats-randomization ordering is what the case asserts.)

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py syndes_bls

The durable case is ``benchmarks/cases/syndes_bls.py``; broader Monte-Carlo
coverage (effects, sizes, power across all three designs vs the randomized
baseline) lives in the estimator's own simulation harness — see the Verification
note on :doc:`../syndes`.

The treated-set search against SCIP
------------------------------------

``mode="two_way_global"`` defaults to ``backend="exact"``. Naming the treated set
removes every binary from the design MIP, leaving a convex program, so the design
is found by searching treated sets instead of by branch-and-bound. That default
rests on the search reaching the design SCIP would prove optimal, which is a
solver question and not a replication of any published number.

``syndes_exact_vs_mip`` pins it on the same BLS panel used above: eight cells
across two sub-panel shapes, two draws and two treated-set sizes, with SCIP asked
to prove optimality (``gap_limit=0``). The two select the same treated set in
every cell, the search's objective never exceeds SCIP's, and the largest
disagreement in objective is 2.1e-09 — the solvers' numerical tolerance, on
outcomes that are unemployment rates in :math:`[0.01, 0.17]`.

One caveat governs how the comparison is set up. ``gap_limit`` defaults to
``0.05``, so the MIP path may stop at a design five percent above the optimum,
and on some panels it does. The two agree on the treated set only once the MIP is
asked to prove optimality; otherwise the search's design is the one that is at
least as good.

`benchmarks/cases/syndes_exact_vs_mip.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/syndes_exact_vs_mip.py>`_

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case syndes_exact_vs_mip
