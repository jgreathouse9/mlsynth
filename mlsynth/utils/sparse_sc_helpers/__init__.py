"""Helper modules for the Sparse Synthetic Control (SparseSC) estimator.

Implements the L1-penalized predictor-weighting variant of Abadie,
Diamond & Hainmueller (2010), based on the MATLAB ``sparse_synth.m``
of Vives-i-Bastida and collaborators.

The estimator differs from canonical SCM in three ways:

  * **Two-level optimization.** SCM has V-weights (importance of each
    predictor) and W-weights (donor combination). SparseSC explicitly
    optimizes both via a nested loop: outer over V, inner over W on
    the simplex.
  * **L1 penalty on V.** The outer training loss adds
    ``lambda * sum(|v|)``, which yields interpretable predictor
    selection -- the V-weights collapse to zero on uninformative
    predictors as lambda increases.
  * **Held-out lambda selection.** The pre-treatment window is split
    into a training block and a validation block. Lambda is selected
    by minimum validation MSE on the outcome, not by training fit.

The first predictor's V-weight is fixed at 1 to anchor the scale of
``v`` (only ``v[1:]`` is optimized); this matches the MATLAB
convention. The inner W-weight problem is the standard simplex QP

    min over w in simplex:  w' X0' diag(v) X0 w  -  2 X1' diag(v) X0 w,

solved with cvxpy. The outer V-weight problem is a smooth bound-
constrained nonlinear program, solved with L-BFGS-B.

Layout:

    structures.py    : SparseSCInputs / Design / Inference / Results
    setup.py         : panel + predictor preparation
    inner.py         : the cvxpy W-weight QP
    objective.py     : training loss with L1 penalty + validation MSE
    optimization.py  : the lambda grid sweep + scipy minimize
    inference.py     : Abadie placebo permutation test
    plotter.py       : observed-vs-counterfactual plot
"""
