.. _replication-smc:

SMC — Synthetic Matching Control (Zhu 2023)
===========================================

:Estimator: :doc:`../smc` — :class:`mlsynth.SMC`
:Source: Zhu, Rong J. B. (2023), *"Synthetic Matching Control Method,"*
   arXiv:2306.02584 [SMC2023]_.
:Replication type: cross-validation against the author's reference R
   implementation (``Code_SMC.R``), plus Path A — the Basque / ETA study.
:Status: verified — the weight computation matches the reference to machine
   precision.

Validation strategy
-------------------

SMC ships an R reference implementation (``Code_SMC.R``): a function ``SMCV``
that, given a matching matrix and predictor weights, computes the per-donor
univariate coefficients :math:`\widehat{\theta}_j`, the plug-in
:math:`\widehat{\sigma}^2`, and the Mallows / :math:`C_p` box weights via
``quadprog::solve.QP``. That function is the ground truth. We reproduce it two
ways: a cell-by-cell numeric match of the weight computation, and the Basque
counterfactual it produces.

Cross-validation — machine precision
------------------------------------

We build the Abadie-Gardeazabal Basque matching matrix through
``Synth::dataprep`` (the reference's own path) and run ``SMCV`` with the
predictor weights fixed to one, so the oracle is deterministic. The mlsynth
weight computation :func:`mlsynth.utils.smc_helpers.smc_weights` is fed the
identical matrix. Every quantity agrees:

  =========================  ==================
  Quantity                   max \|mlsynth − R\|
  =========================  ==================
  :math:`\widehat{\theta}_j` (all donors)   5.3e-15
  box weights :math:`\mathbf{w}`            1.9e-14
  combined :math:`\widehat{\theta}_j w_j`   2.0e-14
  ``bias``                                  6.3e-14
  :math:`\widehat{\sigma}^2`                2.4e-14
  counterfactual (43 years)                 1.7e-13
  =========================  ==================

The synthesis QP is the load-bearing step. mlsynth solves it with an exact
active-set box solver (:func:`mlsynth.utils.smc_helpers.solver.solve_box_qp`),
the box-``[0, 1]`` analogue of the repository's simplex active-set solver. On
the Basque problem it reproduces ``solve.QP`` to ``2e-14`` with a KKT residual
below ``1.5e-14`` — tighter than ``solve.QP``'s own residual — and pins the box
bounds exactly, so a dropped donor's weight is exactly zero. Against a first-
order QP (OSQP) it is an order of magnitude faster at synthetic-control donor
sizes; against an interior-point QP (CLARABEL) it agrees on the objective but,
unlike CLARABEL, leaves no donor microscopically off its bound. The solver is
fuzz-tested against an independent cvxpy oracle over hundreds of random
problems (``mlsynth/tests/test_smc.py``).

Path A — the Basque study
-------------------------

Run through the public estimator on ``basedata/basque_data.csv`` (outcome-only
matching, treatment in 1970), SMC reproduces the Abadie-Gardeazabal result:

.. code-block:: python

   import pandas as pd
   from mlsynth import SMC

   df = pd.read_csv("basedata/basque_data.csv")
   df["treat"] = ((df["regionname"] == "Basque Country (Pais Vasco)")
                  & (df["year"] >= 1970)).astype(int)
   res = SMC({"df": df, "outcome": "gdpcap", "treat": "treat",
              "unitid": "regionname", "time": "year",
              "display_graphs": False}).fit()

gives a pre-period RMSE of about 0.048, a mean post-1969 ATT of about
:math:`-0.858`, and a 1997 gap of about :math:`-0.848` (thousand-1986-USD per
capita), with the combined donor coefficients concentrated on Murcia
(:math:`0.63`), Madrid (:math:`0.37`) and Castilla y León (:math:`0.24`). The
divergence traces the familiar economic cost of ETA terrorism. The durable case
is ``benchmarks/cases/smc_basque.py``.

A note on the covariate / predictor-weight variant
--------------------------------------------------

The paper's Basque tables (its placebo study, Table 4) use the
covariate-augmented variant with an Abadie predictor-weight (:math:`V`) search.
That search is not identified: on the Basque matching matrix a large manifold of
:math:`V` achieves essentially the same pre-outcome fit while producing post-
period effects that range over a wide band, so the reported per-region cells are
not a well-defined function of the data (a global differential-evolution search
reproduces the paper's *average* placebo MSPE but not its cells, and remains
seed-dependent). mlsynth therefore ships the deterministic Algorithm 1 as the
estimator: the :math:`C_p` penalty identifies the weights directly, so no
:math:`V` search — and no attendant non-reproducibility — is involved. Optional
``covariates`` enter at equal predictor weight.
