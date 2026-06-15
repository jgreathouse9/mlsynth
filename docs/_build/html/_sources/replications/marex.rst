MAREX — Abadie & Zhao (2026) Walmart design (independent validator)
===================================================================

.. currentmodule:: mlsynth

Independent, commit-stamped validator of **MAREX** -- mlsynth's port of Abadie &
Zhao's synthetic-control experimental-design estimator -- on the paper's Section 4
Walmart application. The authors' reference code is the R package
`jinglongzhao2/SCDesign <https://github.com/jinglongzhao2/SCDesign>`_.

This complements :doc:`lexscm`, which validates the *same* Abadie-Zhao design on
the same Walmart panel through the lexicographic LEXSCM solver; here the check
runs **MAREX's own MIQP optimizer**.

Data
----

``basedata/walmart_weekly_sales.csv`` (45 stores x 143 weeks; value-identical to
SCDesign's ``Walmart.csv``), restricted to the first **10 stores** so the exact
mixed-integer program is fast and deterministic.

Result — placebo design (Sec. 4)
--------------------------------

A placebo intervention (week 129, no real effect) must yield a design whose
synthetic treated and control units track closely pre-period and whose estimated
effect is indistinguishable from zero. MAREX's exact MIQP delivers exactly that:

==========================  ==========
Quantity                    MAREX
==========================  ==========
Treated stores selected     2
Pre-fit RMSE (% mean sales)  2.66%
Placebo effect (% mean)      0.98%
CI covers zero              yes
==========================  ==========

-- the paper's "no spurious effect" result, and the ~2.7% pre-fit tracking
matches LEXSCM on this panel.

.. note::

   **Exact MIQP, not the relaxation.** The benchmark uses MAREX's exact MIQP
   (free SCIP backend). The relaxed continuous-``z`` mode shares A&Z's objective
   (``build_objective`` is common to both) but drops the integrality that *defines*
   the selection; for a small treated count the relaxed optimum is degenerate, so
   its top-``m`` rounding is lossy and numerically non-deterministic -- unfaithful
   to the paper's exact design. The authors' R solves the full 45-store MIQP with
   Gurobi (a licence-free environment cannot run it), so this validator is Path A
   on a subset with mlsynth's own solver rather than a live R cross-validation;
   only SCDesign's ``quadprog`` SC-weight solver is open.

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py marex_walmart
