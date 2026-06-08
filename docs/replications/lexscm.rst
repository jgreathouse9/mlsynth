.. _replication-lexscm:

LEXSCM — Synthetic Experimental Design (Abadie-Zhao 2026; Vives-i-Bastida 2022)
===============================================================================

:Estimator: :doc:`../lexscm` — :class:`mlsynth.LEXSCM`
:Source: Abadie, A., & Zhao, J. (2026), *"Synthetic Controls for Experimental
   Design."* LEXSCM is the **lexicographic** solve of their program
   (:math:`\xi \to 0`: fit the treated units first, then their controls), due to
   Vives-i-Bastida, J. (2022), *"Synthetic Experimental Design for a UBI Pilot
   Study."*
:Replication type: **Path A** — the paper's Walmart empirical illustration —
   **and Path B** — the paper's linear-factor simulation study.
:Status: **Verified** — placebo empirical and design simulation both reproduced.

LEXSCM is a *design* estimator (it returns a
:class:`~mlsynth.config_models.DesignResult`): it chooses which units to treat
before any intervention so the treated set is representative of the population
and admits a valid synthetic control.

Path A — Walmart placebo design
-------------------------------

We reproduce the paper's empirical illustration (Sec. 4) on the Walmart
store-sales panel (``basedata/walmart_weekly_sales.csv`` — **45 stores over 143
weeks**). Following the paper we design a *placebo* experiment with a fictitious
intervention at week 129 (:math:`T_0 = 128`, the first ~100 weeks fitting, the
rest blank, 15 experimental weeks) and ``m = 2`` treated stores.

.. code-block:: python

   import pandas as pd
   from mlsynth import LEXSCM

   df = pd.read_csv("basedata/walmart_weekly_sales.csv")
   df["candidate"] = 1
   df["post"] = (df["week"] >= 129).astype(int)

   res = LEXSCM({"df": df, "outcome": "sales", "unitid": "store", "time": "week",
                 "candidate_col": "candidate", "m": 2, "post_col": "post",
                 "frac_E": 100 / 128, "top_K": 5, "n_sims": 200,
                 "n_post_grid": [5, 10, 15], "mde_horizon": "late"}).fit()
   res.selected_units            # [1, 25]
   res.report.att                # ~0.9% of mean sales (a placebo: near zero)

Because the intervention is a placebo, a correct design must track closely
pre-period and produce an effect indistinguishable from zero:

.. list-table:: Walmart placebo design (m = 2)
   :header-rows: 1
   :widths: 44 28 28

   * - Quantity
     - LEXSCM
     - Abadie-Zhao (Sec. 4)
   * - Pre-fit RMSE / mean sales
     - **2.7%**
     - small (close tracking)
   * - Placebo effect / mean sales
     - **0.9%**
     - near zero
   * - Permutation p-value
     - **0.63**
     - fails to reject (~0.93)
   * - CI covers zero
     - yes
     - yes

The durable check is ``benchmarks/cases/lexscm_walmart.py``::

   python benchmarks/run_benchmarks.py --case lexscm_walmart

(LEXSCM's lexicographic design selects stores {1, 25} and uses a moving-block
conformal band, so its permutation p differs from MAREX's MIQP design on the
same panel; both deliver the same "no spurious effect" verdict.)

Path B — the design recovers the effect (Abadie-Zhao Sec. 5)
------------------------------------------------------------

On the paper's linear-factor DGP
(:func:`mlsynth.utils.marex_helpers.simulation.generate_marex_sample`,
eqs 12a/12b) we run the full experimental-design loop: LEXSCM picks the treated
units from the pre-period untreated outcomes, the experiment realizes the
treated potential outcome on exactly those units, and the design's estimator is
compared to the true effect. The design recovers the average effect with MAE far
below its own scale, and the error shrinks as more units are treated — the
paper's Table 2 finding:

.. list-table:: Design MAE relative to the effect scale (paired draws)
   :header-rows: 1
   :widths: 30 35 35

   * - Treated cardinality
     - MAE / effect scale
     -
   * - ``m = 2``
     - **0.17**
     -
   * - ``m = 4``
     - **0.09**
     - (error decreases with m)

The durable check is ``benchmarks/cases/lexscm_design_mc.py``::

   python benchmarks/run_benchmarks.py --case lexscm_design_mc

It asserts the design recovers the effect (MAE well below scale at both
cardinalities) and that the error decreases from ``m = 2`` to ``m = 4``.

References
----------

Abadie, A., & Zhao, J. (2026). "Synthetic Controls for Experimental Design."

Vives-i-Bastida, J. (2022). "Synthetic Experimental Design for a UBI Pilot
Study."
