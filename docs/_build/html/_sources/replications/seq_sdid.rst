.. _replication-seq_sdid:

SEQ_SDID — Sequential Synthetic DiD (Arkhangelsky & Samkov 2025)
================================================================

:Estimator: :doc:`../seq_sdid` — :class:`mlsynth.SequentialSDID`
:Source: Arkhangelsky, D. & Samkov, A. (2025), *"Sequential Synthetic
   Difference in Differences,"* arXiv:2404.00164v2.
:Replication type: **Path B** — the paper's Monte Carlo (Section 5.2.2,
   "Experiment 2: Calibrated State-Level Panel"; Table 1 and Figures 4-5).
:Status: **Verified (geometry)** — the headline coverage/RMSE contrast is
   reproduced; the exact Table-1 cells require the authors' (non-public) CPS
   panel.

Validation strategy
-------------------

The paper's central empirical claim is about *inference*: when parallel trends
fail because adoption timing is correlated with unobserved interactive fixed
effects, **standard difference-in-differences is severely biased and its
confidence intervals under-cover**, whereas **Sequential SDiD stays
approximately unbiased with near-nominal coverage**. Table 1 quantifies this on
a state-by-year panel calibrated to March-CPS women's log wages: 95% CI
coverage of ~0.95 for Sequential SDiD against ~0.70 for DiD, with lower RMSE at
every lag.

That panel is not public, so the cells cannot be matched value-for-value.
Instead we re-implement the *design* from the paper's description (scenario 1,
paper only) and reproduce its **geometry**: the same qualitative ranking and an
even sharper version of the same coverage collapse.

A convenient feature of the method makes the comparison airtight: the paper's
standard-DiD comparator is *the same estimator at* :math:`\eta \to \infty` (the
"Original Results" line in Figure 1; the stacked-DiD limit of Remark 2.2),
exposed as ``mode="sdid_imputation"``. Both arms therefore share the Bayesian
bootstrap and differ only in the weighting.

The data-generating process
---------------------------

The DGP is packaged in
:mod:`mlsynth.utils.seq_sdid_helpers.simulate`. Following the paper's recipe:

* **Structural truth is fixed, only shocks are redrawn.** The authors freeze
  the estimated structural components (two-way FE plus a low-rank interactive
  fixed effect) and generate new draws by resampling the idiosyncratic AR
  shocks. :func:`~mlsynth.utils.seq_sdid_helpers.simulate.calibrate_staggered_ife`
  draws the structure once; each draw of
  :func:`~mlsynth.utils.seq_sdid_helpers.simulate.simulate_replication` redraws
  only the AR(2) noise. This is what makes the within-panel bootstrap a valid
  measure of the sampling variability the Monte Carlo averages.
* **The IFE is a differential linear trend** — the canonical rank-one
  interactive fixed effect, :math:`\lambda_i \, f_t` with :math:`f_t = t/T`.
  Adoption is tilted toward high-loading (steeper-trending) units, so treatment
  timing is correlated with the unobserved trend. DiD assumes a common trend
  and is biased; Sequential SDiD balances the loading against later-adopting
  and never-treated donors and is not.
* **Cohorts are enlarged** by replicating each unit four times (Section 5.2.1),
  so cohort aggregates concentrate.
* **Only donor-balanced cohorts are estimated.** A cohort needs at least two
  later / never-treated donor cohorts to balance its loading, so ``a_max`` is
  capped to the sixth-latest cohort (the latest cohorts are donor-starved — see
  the estimator's :doc:`../seq_sdid` "Limitations").

Reproducing Table 1's geometry
------------------------------

.. code-block:: python

   import warnings
   import numpy as np
   from mlsynth import SequentialSDID
   from mlsynth.utils.seq_sdid_helpers.simulate import (
       calibrate_staggered_ife, simulate_replication)

   design = calibrate_staggered_ife(seed=2024)
   tau, K, M, B = 1.0, 4, 40, 50

   def fit(df, mode):
       res = SequentialSDID({"df": df, "outcome": "y", "treat": "treat",
           "unitid": "unit", "time": "year", "mode": mode, "eta": 0.05,
           "K": K, "a_max": design.a_max, "n_bootstrap": B, "seed": 7,
           "display_graphs": False}).fit()
       return res.event_study.tau, res.event_study.ci

   cov = {"ssdid": [], "sdid_imputation": []}
   with warnings.catch_warnings():
       warnings.simplefilter("ignore")
       for m in range(M):
           df = simulate_replication(design, np.random.default_rng(8000 + m), tau=tau)
           for mode in cov:
               tau_hat, ci = fit(df, mode)
               cov[mode].append(((ci[:, 0] <= tau) & (tau <= ci[:, 1])).mean())
   print("SSDiD coverage", np.mean(cov["ssdid"]))
   print("DiD   coverage", np.mean(cov["sdid_imputation"]))

Results
-------

At :math:`M = 40` draws, :math:`B = 50` bootstrap reps (the paper uses
:math:`M = 1000`, :math:`B = 100`):

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Metric
     - Sequential SDiD
     - Standard DiD
     - Paper (SSDiD / DiD)
   * - 95% CI coverage
     - 0.945
     - 0.45
     - ~0.95 / ~0.70
   * - mean :math:`|\mathrm{bias}|`
     - 0.062
     - 0.305
     - —
   * - RMSE
     - 0.252
     - 0.346
     - SSDiD < DiD

What it confirms
----------------

* **Sequential SDiD delivers valid inference** — coverage 0.945, essentially
  the nominal 0.95 — under an IFE violation that breaks DiD.
* **Standard DiD's coverage collapses** to 0.45 and its bias is about five
  times larger; its CIs are unreliable in exactly the regime the method
  targets. (The collapse is sharper than the paper's ~0.70 because the
  reconstructed differential-trend violation is stronger than the CPS
  calibration; the *direction and ranking* are the paper's.)
* **Sequential SDiD has lower RMSE**, the second half of Table 1's finding.

A noiseless corollary, pinned in ``test_seq_sdid.py``, underlies the result:
on a noiseless rank-one IFE the estimator recovers the effect to machine
precision for every donor-balanced cohort, so the design's reliability is not a
tolerance artifact.

The durable check lives in ``benchmarks/cases/seq_sdid_mc.py``::

   python benchmarks/run_benchmarks.py --case seq_sdid_mc
