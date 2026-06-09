.. _replication-linf:

LINF / L1LINF — L-infinity-norm SC (Wang, Xing & Ye 2025)
=========================================================

:Estimator: :doc:`../rescm` — :class:`mlsynth.RESCM` (penalized branch:
   ``LINF`` / ``L1LINF``)
:Source: Wang, Le, Xin Xing, and Youhui Ye (2025), *"A L-infinity Norm
   Counterfactual and Synthetic Control Approach,"* arXiv:2510.26053 [LinfSC]_.
:Replication type: **Path A** — the paper's California Proposition 99
   application — **Path B** — the paper's Section 5 Monte Carlo — and
   **cross-validation** — mlsynth's L-infinity engine matched cell-by-cell
   against the authors' ``LinfinitySC`` code.
:Status: **Verified** — engine cross-validated; empirical and simulation
   reproduced directionally.

Method
------

The L-infinity SC replaces the classic synthetic-control simplex with an
**intercept-shifted, unconstrained** penalized regression (Wang–Xing–Ye, Eqs.
4–5): minimize :math:`\tfrac{1}{2}\lVert y-\mu-Y_0\omega\rVert^2 +
\lambda\lVert\omega\rVert_\infty` (``LINF``), or with the mixed penalty
:math:`\lambda(\alpha\lVert\omega\rVert_1+(1-\alpha)\lVert\omega\rVert_\infty)`
(``L1LINF``). Dropping the simplex and capping the *largest* weight yields a
**dense** weighting that spreads mass across the donor pool — the opposite of
SC's sparse handful — and nests equal-weights/DiD as :math:`\lambda` grows.

mlsynth realises this in the RESCM penalized branch with
``constraint_type="unconstrained"``, ``fit_intercept=True`` and
``standardize=False`` (the L-infinity penalty is scale-sensitive, so the series
are fit raw with the level carried by the intercept).

Validation strategy
-------------------

**Cross-validation (durable: ``linf_crossval_ref``).** The authors ship
``LinfinitySC`` (https://github.com/BioAlgs/LinfinitySC), whose
``our(method="inf" | "l1-inf")`` solves the same program with ``cvxopt`` (open
source — no commercial solver). mlsynth's engine and the reference minimize the
same objective up to a loss-normalization constant, so a reference
:math:`\lambda_{\text{ref}}` corresponds to mlsynth
:math:`\lambda = 2T_0\lambda_{\text{ref}}`. In the **over-determined** regime
:math:`T_0>J` the penalized minimizer is unique and the two independent solvers
agree to solver precision — mlsynth matches both ``LINF`` and ``L1LINF``
cell-by-cell (weight :math:`\ell_1` difference :math:`\approx 0.0019`). (At
:math:`J>T_0` the L-infinity interpolant is non-unique, so the Prop 99 case
validates that regime qualitatively instead.)

**Path A — Proposition 99 (durable: ``linf_prop99``).** On the
Abadie–Diamond–Hainmueller California tobacco panel (treated 1989, 38 donors,
1970–2000) the paper's headline is graphical (Figure 4/5): classic SC
concentrates on ~6 donor states, while the L-infinity method spreads weight
densely, and SC "appears to overestimate the effect." mlsynth reproduces this:
SC keeps 6 donors; ``LINF`` activates all 38 (with ~15 negative weights, off the
simplex); SC's ATT is the more negative. The paper reports no numeric ATT or
weight table, so the case pins these qualitative facts.

**Path B — Monte Carlo (durable: ``linf_sim``).** Under the paper's two-factor
DGP (Section 5; :math:`T_0=100`, :math:`J=30`, :math:`\delta=3`), the dense
L-infinity SC beats sparse classic SC at estimating the ATT when the true donor
weights are dense (DGP 2/3), while the mixed ``L1LINF`` wins when the truth is
sparse (DGP 4). mlsynth reproduces this ordering. The case uses
:math:`B=50` replications and a fixed penalty for a CI-affordable guard
(asserting the ordering, not the paper's 4-decimal :math:`B=2000` Table-4
cells); the full-:math:`B` cells become tractable once the penalized solve is
sped up.

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case linf_crossval_ref --with-reference
   python benchmarks/run_benchmarks.py --case linf_prop99
   python benchmarks/run_benchmarks.py --case linf_sim

Note on fidelity
----------------

Before this replication, mlsynth's ``LINF`` inherited the SC-family simplex
default and a penalty short-circuit that applied a squared-:math:`\ell_2` (ridge)
term for ``alpha == 0``, so it silently collapsed to classic SC rather than the
dense Wang–Xing–Ye estimator. The benchmark surfaced both gaps; the penalized
engine now applies the true L-infinity penalty under the unconstrained,
intercept-shifted program, guarded by ``test_laxscm`` regression tests.
