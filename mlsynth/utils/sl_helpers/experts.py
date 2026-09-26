"""The expert library of Viviano and Bradic (2023), as their own code builds it.

Four forecasters of the treated unit's counterfactual, each fit on the training
split alone and then evaluated over every period. Their ``library.R``'s
``generate_experts`` assembles them in this column order, which is the order
kept here:

1. ``lasso``  -- a penalised regression of the treated outcome on donor outcomes;
2. ``factor`` -- the leading eigenvector of the donor Gram matrix, projected onto
   the donors and then loaded onto the treated unit by least squares;
3. ``forest`` -- a random forest on donor outcomes plus any external covariates,
   the only member with an information set of its own;
4. ``did``    -- the parallel-trends prediction, which is mlsynth's own
   difference-in-differences fit over the whole donor pool.

One deliberate divergence from their code. They choose the ``lasso``'s penalty
with ``cv.glmnet(nfolds = 5)`` and no ``foldid``, so the fold assignment comes off
the RNG. On their own 30-period window that lands ``lambda.min`` in one of three
places -- and the worst of the three keeps no donors at all, collapsing the
expert to a constant -- which moves the reported effect by 40 percent depending
on nothing but the order the script's blocks were run in. The folds here are
contiguous and unshuffled, so the penalty is a function of the data. Contiguous
folds also respect time order, which a random split of a time series does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthDataError
from ..fdid_helpers import did_from_mean

#: The paper's library, in its own column order.
EXPERTS: Tuple[str, ...] = ("lasso", "factor", "forest", "did")

#: Penalty grid for the lasso expert, ``generate_experts``'s
#: ``seq(from = exp(-10), to = exp(-1), length = 79)``.
#:
#: The grid is not a detail on this data. Measured on the paper's own panel the
#: mean cross-validated error varies by a factor of only 1.064 across the whole
#: of it, so the curve barely separates the null model from the six-donor one and
#: whichever grid is searched decides the answer. Letting scikit-learn derive its
#: own grid put the penalty at 5.97e-05 keeping five donors, where the paper's
#: grid puts it at 0.368 keeping none -- a different expert, and one that then
#: carried 34 percent of the ensemble weight instead of 14.
LASSO_LAMBDAS: np.ndarray = np.linspace(np.exp(-10.0), np.exp(-1.0), 79)

#: Penalty grid for the factor expert, which in their code runs to ``exp(2)``
#: and not ``exp(-1)``: it regresses the leading factor on the donors, a much
#: better-conditioned target than the treated outcome, so the useful penalties
#: sit higher.
FACTOR_LAMBDAS: np.ndarray = np.linspace(np.exp(-10.0), np.exp(2.0), 79)


@dataclass(frozen=True)
class ExpertLibrary:
    """Predictions from the experts that fit, and why the others did not.

    Parameters
    ----------
    predictions : np.ndarray
        Counterfactual path per expert over all periods, shape ``(T, K)``.
    names : tuple of str
        The experts that fit, in the column order of ``predictions``.
    details : dict
        Per-expert provenance -- the ``lasso``'s chosen penalty, the ``factor``
        expert's rank, the ``forest``'s tree count.
    dropped : dict
        Expert name to the reason it could not be built. A dropped expert is
        recorded, never silently absent: the library's composition decides the
        estimate.
    """

    predictions: np.ndarray
    names: Tuple[str, ...]
    details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dropped: Dict[str, str] = field(default_factory=dict)


def _standardize(train: np.ndarray, full: np.ndarray):
    """Center and scale on the training block, glmnet's ``standardize = TRUE``.

    Both lasso fits below need this, and not as a stylistic choice: the penalty
    is not scale invariant, so an unstandardized fit applies a different
    effective penalty to every donor. glmnet standardizes by default and the
    authors do not turn it off. The divisor is the population standard deviation,
    which is glmnet's, and it is not interchangeable with the sample one --
    against the R reference the factor expert's path agrees to 7.0e-07 with
    ``ddof=0`` and to 6.8e-06 with ``ddof=1``, from 2.1e-02 unstandardized.
    """
    mu = train.mean(axis=0)
    sd = train.std(axis=0)                      # population, glmnet's divisor
    sd = np.where(sd > 0.0, sd, 1.0)            # a constant donor carries no scale
    return (train - mu) / sd, (full - mu) / sd


def _lasso(Yco: np.ndarray, y: np.ndarray, tr: slice, *, folds: int,
           detail: Dict[str, Any]) -> np.ndarray:
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import KFold

    X1, y1 = Yco[tr], y[tr]
    n = X1.shape[0]
    Z1, Zall = _standardize(X1, Yco)
    # Contiguous, unshuffled folds: deterministic, and they do not train on a
    # period's immediate neighbours to predict it.
    k = int(min(max(folds, 2), n))
    cv = KFold(n_splits=k, shuffle=False)
    fit = LassoCV(alphas=LASSO_LAMBDAS, cv=cv, max_iter=200000).fit(Z1, y1)
    detail["alpha"] = float(fit.alpha_)
    detail["n_selected"] = int(np.count_nonzero(fit.coef_))
    return np.asarray(fit.predict(Zall), dtype=float)


def _factor(Yco: np.ndarray, y: np.ndarray, tr: slice, *, rank: int,
            folds: int, detail: Dict[str, Any]) -> np.ndarray:
    """Their ``eigen(X X')$vectors[, 1]`` route, then loadings by least squares."""
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import KFold

    X1 = Yco[tr]
    if X1.shape[1] < 2:
        raise MlsynthDataError(
            "the factor expert needs at least two donors to decompose")
    # The leading left singular vectors of the training block are the paper's
    # eigenvectors of X X', up to sign.
    U = np.linalg.svd(X1, full_matrices=False)[0][:, :rank]
    Z1, Zall = _standardize(X1, Yco)
    k = int(min(max(folds, 2), X1.shape[0]))
    cols, alphas_used = [], []
    for j in range(U.shape[1]):
        fit = LassoCV(alphas=FACTOR_LAMBDAS, cv=KFold(n_splits=k, shuffle=False),
                      max_iter=200000).fit(Z1, U[:, j])
        cols.append(fit.predict(Zall))
        alphas_used.append(fit.alpha_)
    F = np.column_stack(cols)
    A = np.column_stack([np.ones(X1.shape[0]), F[tr]])
    coef = np.linalg.lstsq(A, y[tr], rcond=None)[0]
    detail["rank"] = int(U.shape[1])
    detail["alphas"] = [float(a) for a in alphas_used]
    return np.asarray(np.column_stack([np.ones(F.shape[0]), F]) @ coef,
                      dtype=float)


def _forest(Yco: np.ndarray, y: np.ndarray, tr: slice, *,
            covariates: Optional[np.ndarray], trees: int, max_leaf_nodes: int,
            seed: int, detail: Dict[str, Any]) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor

    design = Yco if covariates is None else np.column_stack([Yco, covariates])
    model = RandomForestRegressor(n_estimators=trees,
                                  max_leaf_nodes=max_leaf_nodes,
                                  random_state=seed, n_jobs=1)
    model.fit(design[tr], y[tr])
    detail["trees"] = int(trees)
    detail["n_features"] = int(design.shape[1])
    return np.asarray(model.predict(design), dtype=float)


def _did(Yco: np.ndarray, y: np.ndarray, tr: slice,
         detail: Dict[str, Any]) -> np.ndarray:
    """The parallel-trends prediction, from mlsynth's difference-in-differences.

    ``generate_experts`` writes this expert as one line,
    ``mean(y1) - mean(apply(X1, 2, mean)) + apply(X, 1, mean)``, and mlsynth
    already has that fit: :func:`~mlsynth.utils.fdid_helpers.did_from_mean` is
    what ``FDID.fit()`` returns in its ``did`` block, and Li's method reports the
    conventional estimate over the full donor pool alongside her forward one.
    That is the same estimator the authors use as their fourth expert, on the
    same donors, so the expert is delegated to it.

    The two intercepts are one number and not two estimators that agree. Theirs
    is the treated training mean minus the grand mean of the donor block; FDID's
    is the mean of the period-by-period difference. A mean of period differences
    over a rectangular block equals the difference of grand means, so the forms
    coincide algebraically. What separates them numerically is the order the
    sums run in: on the authors' own panel the intercepts differ by 2.8e-17 and
    the paths by 5.6e-17.

    ``did_from_mean`` reads the pre-treatment window as a prefix, which
    Algorithm 1's expert-training split is.
    """
    if tr.start:
        raise MlsynthDataError(
            "the did expert needs a training split that starts at the first "
            f"period, and this one starts at {tr.start}"
        )
    raw = did_from_mean(y, Yco.mean(axis=1), int(tr.stop))
    detail["n_donors"] = int(Yco.shape[1])
    detail["intercept"] = float(raw["Inference"]["Intercept"])
    return np.asarray(raw["Vectors"]["Counterfactual"], dtype=float)


def build_experts(Yco: np.ndarray, y: np.ndarray, train: slice,
                  names: Sequence[str] = EXPERTS, *,
                  covariates: Optional[np.ndarray] = None, seed: int = 0,
                  lasso_folds: int = 5, factor_rank: int = 1,
                  forest_trees: int = 500,
                  forest_max_leaf_nodes: int = 20) -> ExpertLibrary:
    """Fit each requested expert on ``train`` and predict over every period.

    Parameters
    ----------
    Yco : np.ndarray
        Donor outcomes, shape ``(T, N)``.
    y : np.ndarray
        Treated outcome, shape ``(T,)``.
    train : slice
        The expert-training split of Algorithm 1. No expert sees anything
        outside it while fitting, which is what makes the weighting window out
        of sample.
    names : sequence of str
        Which experts to build, in the order they should occupy columns.
    covariates : np.ndarray, optional
        Time-varying covariates, shape ``(T, m)``, read by ``forest`` only.
    seed : int
        Seed for the forest. The other three are deterministic.
    lasso_folds, factor_rank, forest_trees, forest_max_leaf_nodes
        Pinned hyperparameters. None of them is drawn.

    Returns
    -------
    ExpertLibrary

    Raises
    ------
    MlsynthDataError
        If no requested expert could be built.
    """
    Yco = np.asarray(Yco, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    cols, kept, details, dropped = [], [], {}, {}

    for name in names:
        detail: Dict[str, Any] = {}
        try:
            if Yco.shape[1] == 0:
                raise MlsynthDataError("no donor units")
            if name == "lasso":
                p = _lasso(Yco, y, train, folds=lasso_folds, detail=detail)
            elif name == "factor":
                p = _factor(Yco, y, train, rank=factor_rank,
                            folds=lasso_folds, detail=detail)
            elif name == "forest":
                p = _forest(Yco, y, train, covariates=covariates,
                            trees=forest_trees,
                            max_leaf_nodes=forest_max_leaf_nodes,
                            seed=seed, detail=detail)
            elif name == "did":
                p = _did(Yco, y, train, detail=detail)
            else:  # pragma: no cover - SLConfig's Literal rejects anything else
                raise MlsynthDataError(f"unknown expert {name!r}")
            p = np.asarray(p, dtype=float).ravel()
            if p.shape != y.shape or not np.isfinite(p).all():
                raise MlsynthDataError("produced a non-finite prediction")
        except Exception as exc:
            dropped[str(name)] = f"{type(exc).__name__}: {exc}"
            continue
        cols.append(p)
        kept.append(str(name))
        details[str(name)] = detail

    if not cols:
        raise MlsynthDataError(
            "No expert in the library could be fit, so there is nothing to "
            "weight. Reasons: "
            + "; ".join(f"{k}: {v}" for k, v in dropped.items()))
    return ExpertLibrary(predictions=np.column_stack(cols),
                         names=tuple(kept), details=details, dropped=dropped)
