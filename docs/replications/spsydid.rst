.. _replication-spsydid:

SpSyDiD — Spatial Synthetic-DiD (Serenini & Masek 2024)
=======================================================

:Estimator: :doc:`../spsydid` — :class:`mlsynth.SpSyDiD`
:Source: Serenini, R., & Masek, F. (2024), *"Spatial Synthetic
   Difference-in-Differences,"* SSRN Working Paper 4736857.
:Replication type: **Path B** — the authors' State-Level Monte Carlo, as a
   **per-replication cross-validation** against the authors' own reference code.
:Status: **Fully verified** — matched per-rep to the authors' algorithm and
   recovers the paper's headline unbiasedness.

Validation strategy
-------------------

Serenini & Masek's empirical example (the Arizona 2007 LAWA effect) runs on a
CPS panel that the authors do not release. Their public replication repository,
https://github.com/serenini/spatial_SDID, instead ships the **simulation code**
plus a BLS unemployment panel for two Monte Carlo exercises. We therefore
satisfy the contract along **Path B** — and, because the authors' full code is
available, we make it a **cross-validation**: on every replication we run both
``mlsynth.SpSyDiD`` and the authors' own estimator on the *same* panel and
compare the direct ATT per-rep.

Because the reference repository carries **no licence**, its code is not
vendored into mlsynth. The benchmark clones it on demand at a pinned commit
(``e43427d``) and imports the authors' ``functions_ssdid`` weight machinery from
the clone; the spatial WLS step is reproduced from
``State_Level_Simulations.ipynb`` (cell 16). It skips gracefully when git, the
network, or ``libpysal`` is unavailable.

The DGP (State-Level Monte Carlo)
---------------------------------

* **Panel:** 49 contiguous US states, monthly unemployment 1976-2014
  (``basedata/state_unemployment.csv``); queen-contiguity :math:`W`
  (``basedata/US_no_islands_matrix.gal``), row-standardised.
* **Windows:** for each 3-year rolling window the third year is the
  post-period (:math:`T_0 = 24`, :math:`T_1 = 12`). Panels are deterministic.
* **Treatment:** Arkansas (FIPS 5) is the directly-treated state. The outcome
  is :math:`\text{UR2} = \text{perc\_unem} + \text{interaction}\cdot\text{ATT}
  + \text{spillover}\cdot\text{ATT}\cdot\rho`, with ATT set to 25% of the
  window's mean unemployment and :math:`\rho = 0.8`.

Per-replication cross-validation
--------------------------------

Over the 20 deterministic windows from 1975, mlsynth and the authors' reference
algorithm agree on the direct ATT to within :math:`\approx 0.09` pp per rep,
with a per-rep correlation of **0.996**:

.. list-table::
   :header-rows: 1
   :widths: 46 26 26

   * - Quantity (20 windows)
     - mlsynth
     - reference
   * - mean ATT bias
     - +0.023
     - +0.022
   * - max per-rep :math:`|\Delta\widehat{\tau}|` vs ref
     - 0.094
     - —
   * - per-rep correlation vs ref
     - 0.996
     - —

Both estimators recover the paper's headline finding: the **mean ATT bias is
essentially zero** (~0.02 against an ATT magnitude of ~1.7 pp), i.e. the
spatial correction cleanly removes the spillover-induced bias that contaminates
plain SDID. The small per-rep residual is the affected-unit weight convention
(mlsynth: :math:`1/N_{sp}`; reference: mean treated-unit SDID weight) — both
valid downstream of the SDID weight QPs.

Durable check
-------------

The benchmark lives in ``benchmarks/cases/spsydid_state_mc.py``::

   pip install libpysal
   python benchmarks/run_benchmarks.py --case spsydid_state_mc

It asserts per-rep agreement (max :math:`|\Delta| < 0.2`, correlation > 0.98)
and near-zero mean ATT bias for *both* implementations.

References
----------

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
(2021). "Synthetic Difference-in-Differences." *American Economic Review*
111(12):4088-4118.

Serenini, R., & Masek, F. (2024). "Spatial Synthetic Difference-in-Differences."
SSRN Working Paper 4736857.
