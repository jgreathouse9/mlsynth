r"""The Unit-level design's second term, Abadie and Zhao's equation (10).

Stage 1 chooses treated units so the treated aggregate reproduces the population
target. That is one of the two conditions the design needs: it says nothing
about whether each chosen unit is reproducible by the donors left over, and the
two come apart exactly where it hurts. On a 62-market panel at ``m = 8`` the
design selected the two largest markets, which no convex combination of the
remainder can reach, and gave them 58 per cent of the weight; the aggregate
estimate read 31.9 per cent against a true 8.5 and its interval missed.

Equation (10) adds

.. math::

   \xi \sum_j w_j \bigl\lVert x_j - \sum_i v_{ij} x_i \bigr\rVert^2 ,

each treated unit's own reproducibility weighted by its share of the aggregate,
with :math:`v_{ij} = 0` for :math:`i` in the treated set so treated units cannot
serve as donors for one another.

Two facts keep this cheap. The inner weights :math:`v_{\cdot j}` appear in one
term only, multiplied by the non-negative scalar :math:`w_j`, so a positive
scalar cannot move their argmin: the optimal :math:`v_{\cdot j}` is independent
of :math:`\mathbf{w}` and is the ordinary synthetic control for unit :math:`j`
against the donors outside the tuple. And on the simplex a linear term folds
into the quadratic exactly, so the existing minimum-norm-point solver takes the
penalty unchanged.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError
from .lexsearch import _afw_single


def fold_linear_into_gram(gram: np.ndarray,
                          linear: Sequence[float]) -> Tuple[np.ndarray, float]:
    r"""Rewrite ``w'Qw + c'w`` on the simplex as ``w'Mw - kappa``.

    On the simplex :math:`\mathbf{1}'\mathbf{w} = 1`, so
    :math:`c'w = (c'w)(\mathbf{1}'w) = w'\,\tfrac{1}{2}(c\mathbf{1}' +
    \mathbf{1}c')\,w` and the linear term is a quadratic form. The same identity
    makes :math:`w'\mathbf{1}\mathbf{1}'w = 1`, so adding
    :math:`\kappa\mathbf{1}\mathbf{1}'` shifts the objective by a constant and
    leaves the minimiser alone.

    That second freedom is what keeps the result usable. The rank-two term is
    indefinite, and the solver is a minimum-norm-point method whose active-set
    systems have no meaning for an indefinite form. On
    :math:`\{u : \mathbf{1}'u = 0\}` the rank-two term contributes
    :math:`(u'c)(\mathbf{1}'u) = 0`, so the form restricts to ``gram`` there and
    is already positive semidefinite; only the complementary direction needs
    lifting, and a large enough :math:`\kappa` supplies it.

    Returns the folded Gram and the constant to subtract.
    """
    Q = np.asarray(gram, dtype=float)
    c = np.asarray(linear, dtype=float).ravel()
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise MlsynthConfigError(
            f"the Gram matrix has to be square; got shape {Q.shape}.")
    if c.size != Q.shape[0]:
        raise MlsynthConfigError(
            f"the linear term carries {c.size} coefficient(s) against a "
            f"{Q.shape[0]}x{Q.shape[0]} Gram matrix.")
    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(c)):
        raise MlsynthDataError(
            "the Gram matrix or the linear term contains non-finite values.")

    if not np.any(c):
        return Q, 0.0                      # xi = 0 leaves Stage 1 exactly as it was

    m = Q.shape[0]
    ones = np.ones(m)
    folded = Q + 0.5 * (np.outer(c, ones) + np.outer(ones, c))
    kappa = 0.0
    bump = max(1.0, float(np.abs(c).max()))
    for _ in range(64):                    # lift until the form is usable
        if np.linalg.eigvalsh(folded + kappa * np.outer(ones, ones)).min() > -1e-10:
            break
        kappa = bump if kappa == 0.0 else kappa * 2.0
    else:                                  # pragma: no cover - a PSD Gram plus a
        raise MlsynthDataError(            # finite rank-two term always lifts
            "the penalised Gram matrix could not be made positive semidefinite.")
    return folded + kappa * np.outer(ones, ones), float(kappa)


def per_unit_imbalance(design: np.ndarray,
                       treated: Sequence[int],
                       donors: Sequence[int]) -> np.ndarray:
    r"""How well each treated unit's own donors reproduce it.

    One ordinary synthetic control per treated unit, fitted against ``donors``
    only, returning :math:`\lVert x_j - \sum_i v_{ij} x_i \rVert^2` for each.
    The treated units are excluded from one another's donor pools, which
    equation (10) imposes with :math:`v_{ij} = 0` for :math:`i \in \mathcal{S}`:
    without it a pair of near-identical treated markets would each reproduce the
    other perfectly while the donor pool could reach neither.

    The weights do not enter. :math:`v_{\cdot j}` is the argmin of a term scaled
    by :math:`w_j \ge 0`, and a positive scalar does not move an argmin, so this
    is computable once a candidate tuple is known and before any weight is
    solved for.
    """
    X = np.asarray(design, dtype=float)
    tr = np.asarray(treated, dtype=int).ravel()
    dn = np.asarray(donors, dtype=int).ravel()
    if X.ndim != 2:
        raise MlsynthConfigError(
            f"the design matrix has to be two-dimensional; got shape {X.shape}.")
    if dn.size == 0:
        raise MlsynthDataError(
            "no donors are left to reproduce the treated units, so the "
            "unit-level penalty has nothing to measure.")
    if not np.all(np.isfinite(X)):
        raise MlsynthDataError("the design matrix contains non-finite values.")
    overlap = set(tr.tolist()) & set(dn.tolist())
    if overlap:
        raise MlsynthConfigError(
            f"unit(s) {sorted(overlap)} appear as both treated and donor; "
            f"equation (10) excludes the treated set from every donor pool.")

    D = X[:, dn]
    gram = D.T @ D
    out = np.empty(tr.size, dtype=float)
    for k, j in enumerate(tr):
        target = X[:, j]
        # min_v ||x_j - D v||^2 = v'(D'D)v - 2 (D'x_j)'v + x_j'x_j, and the
        # linear part folds onto the simplex the same way the penalty does.
        folded, kappa = fold_linear_into_gram(gram, -2.0 * (D.T @ target))
        loss, _, _ = _afw_single(folded)
        out[k] = max(float(loss) - kappa + float(target @ target), 0.0)
    return out
