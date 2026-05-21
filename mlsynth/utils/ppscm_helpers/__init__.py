"""Helper modules for the Partially Pooled SCM estimator (Ben-Michael,
Feller & Rothstein 2022, *JRSS-B* 84(2):351-381).

The estimator extends classical synthetic control to staggered-adoption
designs. For ``J`` treated units (each adopting at its own period
``T_j``) and ``N`` never-treated controls, partially pooled SCM picks
a ``J x N`` weight matrix ``Gamma`` minimizing a weighted average of
two imbalance measures:

  * ``q_sep(Gamma)^2`` -- mean per-treated-unit pre-treatment fit
    (the imbalance from running SCM separately for each treated unit);
  * ``q_pool(Gamma)^2`` -- pre-treatment fit for the *average* treated
    unit (the imbalance from averaging the treated units first and
    then running one SCM on the average).

The objective (Eq. 6 of the paper) is

    min over Gamma in simplex^J:
        nu * q_tilde_pool(Gamma)^2 + (1 - nu) * q_tilde_sep(Gamma)^2
        + lam * ||Gamma||_F^2

where ``q_tilde`` are normalized by their values at the ``nu = 0``
(separate-SCM) solution. ``nu = 0`` recovers separate SCM; ``nu = 1``
recovers fully pooled SCM. Intermediate ``nu`` trades off the two
imbalances along a convex frontier.

This package implements the outcome-only variant of the paper
(Sections 3-4). The auxiliary-covariate extension of Section 5.2 is
out of scope.

Layout:

    structures.py    : frozen dataclasses for inputs / design / results
    setup.py         : align treated and donor pre-windows
    imbalance.py     : q_sep and q_pool (Eq. 5 of the paper) and
                        their gradients with respect to Gamma
    optimization.py  : the partially-pooled QP via cvxpy + the
                        equal-imbalance heuristic for auto-nu
    inference.py     : jackknife on the overall ATT
    plotter.py       : event-study chart with jackknife CI band
"""
