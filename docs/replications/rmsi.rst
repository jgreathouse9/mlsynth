RMSI — Robust Matrix Estimation with Side Information (Agarwal-Choi-Yuan 2026)
=============================================================================

.. currentmodule:: mlsynth

Reproduction of Agarwal, Choi & Yuan (2026), *Robust Matrix Estimation with Side
Information* (arXiv:2603.24833). RMSI decomposes a matrix into four complementary
components -- a (possibly nonlinear) row x column **interaction**, a
**row-characteristic-driven** part, a **column-driven** part, and a **residual
low-rank** part -- estimated by sieve projection + nuclear-norm penalization, with
MAR / MNAR (block-missing) extensions for panel causal inference. The paper ships
no reference code, so this is a Path-B reproduction of its simulations plus a
Path-A check on the tobacco application.

Path B -- Section 5.1 (robustness across component weights)
----------------------------------------------------------

The paper writes :math:`M = a_1 M_1 + a_2 M_2 + a_3 M_3 + a_4 M_4` with each
:math:`\|M_r\|_F = 2\sqrt{NT}` and :math:`\sum_r a_r = 1`, and shows the
estimator's advantage over a no-side-information baseline **varies with but is
robust across** the component weights. Over five seeds and three weight regimes
(interaction-dominant / balanced / residual-dominant), RMSI's relative recovery
error beats the no-side-info nuclear-norm baseline in **every** cell and sits near
the noise floor:

==========================  ==========
Quantity                    Value
==========================  ==========
Cells where side info wins  15 / 15
Mean relative error (RMSI)  0.29
Mean relative error (base)  0.32
==========================  ==========

Path A -- Section 5.2 (tobacco)
------------------------------

On the Proposition 99 panel (``basedata/P99data.csv``), RMSI recovers California's
tobacco ATT at **−21.4 packs/capita**, in the Abadie-Diamond-Hainmueller range
(~ −19 to −20).

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py rmsi_sim

The durable case is ``benchmarks/cases/rmsi_sim.py``; the algorithmic core
(four-component recovery, block-imputation ATT) is also pinned in
``mlsynth/tests/test_rmsi.py``.
