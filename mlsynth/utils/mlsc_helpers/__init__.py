"""Helper modules for the Multi-Level Synthetic Control (mlSC) estimator.

Implements:

    Bottmer, L. (2025). "Synthetic Control with Disaggregated Data."
    Stanford University job-market paper.

The preferred mlSC variant studied in the paper keeps the treated unit at the
aggregate level while allowing flexible weighting of disaggregated control
units, with a ridge-type penalty that shrinks the disaggregated weights
toward ``v_sc * w_s`` (population-share times aggregate weight). At penalty
infinity the estimator collapses to the classical SC; at penalty zero it
recovers the fully-disaggregated SC (dGSC-AD in the paper's notation).

Layout follows the SCDI / SPCD / LEXSCM / TASC helper packages:

    structures.py   : frozen dataclasses (MLSCInputs / Design / Inference / Results)
    setup.py        : two-DataFrame validation + dataprep-driven matrix assembly
    variance.py     : Appendix G variance decomposition (sigma_eps^2, sigma_y^2)
    penalty.py      : block-diagonal penalty matrix Q from population weights v_sc
    optimization.py : cvxpy QP with simplex constraint and classical-SC warm-start
    inference.py    : counterfactual path, ATT, pre-period RMSE
    plotter.py      : observed-vs-counterfactual chart via resultutils.plot_estimates
"""
