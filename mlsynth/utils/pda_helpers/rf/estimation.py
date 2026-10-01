"""Random-forest donor selection for PDA (Liu, Long & Luo 2025).

The counterfactual is Hsiao, Ching & Wan's: an OLS regression of the treated
unit on a subset of controls over the pre-treatment window, extrapolated after
it. What this module supplies is the subset. A random forest grown on the
pre-treatment data ranks the controls by permutation importance, and a forward
search walks down that ranking, adding one control at a time and keeping the
prefix whose held-out mean squared error is smallest. The cost is ``2n`` forests
against the ``2^n`` subsets of the original best-subset search and the
``(n + 1/2) R - R^2/2`` of forward selection.

Three choices here are not the released code's, and each is named so the
released behaviour stays reachable.

Splitting. Section 2.2 divides the pre-treatment window into three disjoint
blocks in time order -- training, validation, testing -- citing Gu, Kelly and
Xiu (2020), on the grounds that preserving the ordering is what keeps future
data out of the validation set. ``split="temporal"`` does that and is the
default. The released ``RF.R`` takes a random 70/30 split of the pre-treatment
rows and has no validation block; ``split="random"`` reproduces it, and is what
the published estimates were computed with.

Tuning. With a validation block the number of trees, the depth and the number
of controls tried per split are chosen on it, as Section 2.2 Step 2 describes.
Under ``split="random"`` there is no validation block and nothing is tuned.

The cap. The released search runs the prefix length from 2 to ``n - 1`` with
nothing tying it to the pre-period length, while Assumption 3 of the paper
requires ``|U| / T1 -> 0``. When the search selects more controls than there are
pre-treatment periods the OLS fit interpolates the pre-period exactly, the
residual falls to rounding, and the long-run variance it is standardised by
collapses with it: measured over twenty seeds on the paper's own panels this
happens in 7 of 20 Brexit fits and 4 of 20 luxury-watch fits, returning test
statistics of -17.8 and -966336 with a p value of zero. ``k_max`` defaults to
``T0 - 2``, which leaves a residual degree of freedom; passing a larger value
reproduces the released search and sets ``cap_exceeds_pre_periods``.

Seeds. The selected set is a function of the random split and the forest, and on
the paper's panels it moves a great deal with the seed -- over twenty seeds the
luxury-watch ATE runs from -0.063 to -0.003 around a published -0.0266, and
across twenty Brexit seeds every one of the 167 controls is selected by some
seed. ``n_seeds`` re-runs the selection on consecutive seeds and reports the
spread as a diagnostic. The estimate stays the fit at ``seed``, so the estimator
remains a function of its arguments.
"""

from __future__ import annotations

from math import isqrt
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ....exceptions import MlsynthConfigError

# Smallest pre-treatment window a three-block split can be formed from: one
# period in each block, plus the two the OLS fit needs.
_MIN_T0 = 6

# A control whose pre-period standard deviation falls below this times the panel
# scale carries no information to rank and would make the OLS design singular.
_CONST_TOL = 1e-10


def _blocks(T0: int, split: str, train_fraction: float,
            validation_fraction: float, rng: np.random.Generator
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training, validation and testing row indices within the pre-period."""
    if split == "random":
        # The released RF.R: sample(1:T1, T1 * per_train), the rest held out.
        n_train = int(T0 * train_fraction)
        train = np.sort(rng.choice(T0, size=max(n_train, 1), replace=False))
        test = np.setdiff1d(np.arange(T0), train)
        return train, np.empty(0, dtype=int), test
    n_train = max(int(T0 * train_fraction), 1)
    n_val = max(int(T0 * validation_fraction), 1)
    if n_train + n_val >= T0:
        raise MlsynthConfigError(
            f"the split leaves no test block: train {n_train} + validation "
            f"{n_val} of {T0} pre-treatment periods")
    idx = np.arange(T0)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def _forest(n_estimators: int, max_depth, mtry, seed: int):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=int(n_estimators), max_depth=max_depth,
        max_features=mtry, random_state=int(seed), n_jobs=-1)


def _tune(Xtr, ytr, Xva, yva, n_estimators, seed) -> Tuple[int, Optional[int]]:
    """``(mtry, max_depth)`` minimising validation error -- Section 2.2 Step 2."""
    p = Xtr.shape[1]
    grid_mtry = sorted({max(p // 3, 1), max(isqrt(p), 1), p})
    best, choice = np.inf, (grid_mtry[0], None)
    for mtry in grid_mtry:
        for depth in (None, 5):
            rf = _forest(n_estimators, depth, mtry, seed).fit(Xtr, ytr)
            err = float(np.mean((rf.predict(Xva) - yva) ** 2))
            if err < best:
                best, choice = err, (mtry, depth)
    return choice


def _oob_importance(rf, X, y) -> np.ndarray:
    """Out-of-bag permutation importance, the quantity ``randomForest`` reports.

    For each tree the error is measured on the rows it did not draw, then
    measured again with one control's values permuted among those rows; the
    increase, averaged over trees, is that control's importance. Unlike R's
    ``importance(..., type = 1)`` the result is not divided by its standard
    error across trees, so it is the raw mean increase.
    """
    from sklearn.ensemble._forest import _generate_sample_indices
    n, p = X.shape
    rng = np.random.default_rng(0)
    total = np.zeros(p)
    counted = 0
    for est in rf.estimators_:
        try:
            drawn = _generate_sample_indices(est.random_state, n, n, None)
        except TypeError:  # pragma: no cover - older scikit-learn signature
            drawn = _generate_sample_indices(est.random_state, n, n)
        oob = np.setdiff1d(np.arange(n), np.unique(drawn))
        if oob.size < 2:
            continue
        counted += 1
        Xo, yo = X[oob], y[oob]
        # One predict per tree instead of one per (tree, control): the p noised
        # copies of the out-of-bag block are stacked and scored in one call.
        stack = np.repeat(Xo[None, :, :], p, axis=0)
        for j in range(p):
            stack[j, :, j] = Xo[rng.permutation(oob.size), j]
        pred = est.predict(stack.reshape(p * oob.size, -1)).reshape(p, oob.size)
        base = float(np.mean((est.predict(Xo) - yo) ** 2))
        total += np.mean((pred - yo[None, :]) ** 2, axis=1) - base
    return total / max(counted, 1)


def _importance(rf, X, y, kind: str, seed: int) -> np.ndarray:
    if kind == "oob":
        return _oob_importance(rf, X, y)
    from sklearn.inspection import permutation_importance
    # Equation (7) is the increase in prediction error on the held-out point
    # when one control is noised, which is this quantity on the test block.
    out = permutation_importance(rf, X, y, n_repeats=5, random_state=int(seed),
                                 n_jobs=-1)
    return np.asarray(out.importances_mean, dtype=float)


def _ols(y_pre: np.ndarray, X_pre: np.ndarray, cols: Sequence[int],
         X_all: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """OLS on ``[1, selected]`` over the pre-period, extrapolated to all rows."""
    design = np.column_stack([np.ones(X_pre.shape[0]), X_pre[:, list(cols)]])
    coef, *_ = np.linalg.lstsq(design, y_pre, rcond=None)
    beta = np.zeros(X_all.shape[1])
    beta[list(cols)] = coef[1:]
    return beta, float(coef[0]), X_all @ beta + float(coef[0])


def _one_fit(y, X, T0, *, split, train_fraction, validation_fraction,
             n_estimators, max_depth, mtry, k_max, importance, seed, live):
    """Selection and OLS refit for a single seed. Returns (selected, meta-bits)."""
    rng = np.random.default_rng(seed)
    train, validation, test = _blocks(T0, split, train_fraction,
                                      validation_fraction, rng)
    y_pre, X_pre = y[:T0], X[:T0]
    Xl = X_pre[:, live]

    mtry_used, depth_used = mtry, max_depth
    if validation.size and (mtry is None or max_depth is None):
        tuned_mtry, tuned_depth = _tune(Xl[train], y_pre[train],
                                        Xl[validation], y_pre[validation],
                                        n_estimators, seed)
        mtry_used = tuned_mtry if mtry is None else mtry
        depth_used = tuned_depth if max_depth is None else max_depth

    rf = _forest(n_estimators, depth_used, mtry_used, seed).fit(Xl[train], y_pre[train])
    score_rows = test if test.size else train
    imp = _importance(rf, Xl[score_rows], y_pre[score_rows], importance, seed)
    order = np.argsort(-imp, kind="stable")

    best_k, best_mse = 1, np.inf
    for k in range(1, min(k_max, live.size) + 1):
        cols = order[:k]
        m = _forest(n_estimators, depth_used, min(mtry_used or k, k), seed)
        m.fit(Xl[train][:, cols], y_pre[train])
        mse = float(np.mean((m.predict(Xl[score_rows][:, cols]) - y_pre[score_rows]) ** 2))
        if mse < best_mse:
            best_mse, best_k = mse, k
    selected = [int(live[j]) for j in order[:best_k]]
    return selected, order, train, validation, test, mtry_used, depth_used


def rf_select(
    y: np.ndarray, X: np.ndarray, T0: int, *,
    split: str = "temporal", train_fraction: float = 0.6,
    validation_fraction: float = 0.2, n_estimators: int = 500,
    max_depth: Optional[int] = None, mtry: Optional[int] = None,
    k_max: Optional[int] = None, importance: str = "permutation",
    seed: int = 0, n_seeds: int = 1,
) -> Tuple[List[int], np.ndarray, float, np.ndarray, Dict]:
    """Select controls by random forest, then refit PDA's OLS on them.

    Returns ``(selected, beta_full, intercept, counterfactual, meta)`` where
    ``beta_full`` is an ``N``-vector zero off the selected support and ``meta``
    carries the split, the ranking, the cap and the seed diagnostic.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    T0 = int(T0)
    if T0 < _MIN_T0:
        raise MlsynthConfigError(
            f"rfPDA needs at least {_MIN_T0} pre-treatment periods to split "
            f"into training, validation and testing blocks; got {T0}")

    # A control that never moves before treatment cannot be ranked and makes the
    # OLS design singular, so it leaves the pool before anything is fitted.
    sd = X[:T0].std(axis=0)
    live = np.flatnonzero(sd > _CONST_TOL * max(float(np.abs(y[:T0]).max()), 1.0))
    dropped = [int(j) for j in np.flatnonzero(~np.isin(np.arange(X.shape[1]), live))]
    if live.size == 0:
        raise MlsynthConfigError("rfPDA has no control with pre-treatment variation")

    cap = (T0 - 2) if k_max is None else int(k_max)
    cap = max(1, min(cap, int(live.size)))

    fit_kw = dict(split=split, train_fraction=train_fraction,
                  validation_fraction=validation_fraction,
                  n_estimators=n_estimators, max_depth=max_depth, mtry=mtry,
                  k_max=cap, importance=importance, live=live)
    selected, order, train, validation, test, mtry_used, depth_used = _one_fit(
        y, X, T0, seed=seed, **fit_kw)
    beta, const, cf = _ols(y[:T0], X[:T0], selected, X)

    spread = None
    if n_seeds > 1:
        atts, sets = [], []
        for s in range(seed, seed + int(n_seeds)):
            sel_s, *_ = _one_fit(y, X, T0, seed=s, **fit_kw)
            _, _, cf_s = _ols(y[:T0], X[:T0], sel_s, X)
            atts.append(float(np.mean((y - cf_s)[T0:])))
            sets.append(set(sel_s))
        pairs = [len(a & b) / max(len(a | b), 1)
                 for i, a in enumerate(sets) for b in sets[i + 1:]]
        atts = np.asarray(atts, dtype=float)
        spread = {
            "n_seeds": int(n_seeds),
            "att_mean": float(atts.mean()),
            "att_sd": float(atts.std(ddof=1)) if atts.size > 1 else 0.0,
            "att_min": float(atts.min()), "att_max": float(atts.max()),
            "selection_jaccard_mean": float(np.mean(pairs)) if pairs else 1.0,
            "n_selected_min": int(min(len(s) for s in sets)),
            "n_selected_max": int(max(len(s) for s in sets)),
        }

    meta = {
        "split": split, "importance": importance,
        "importance_order": [int(live[j]) for j in order],
        "train_idx": train.tolist(), "validation_idx": validation.tolist(),
        "test_idx": test.tolist(),
        "n_estimators": int(n_estimators), "mtry": mtry_used,
        "max_depth": depth_used,
        "k_max": cap if k_max is None else int(k_max),
        "cap_binds": len(selected) == cap,
        "cap_exceeds_pre_periods": cap > T0 - 2,
        "selected": selected, "dropped_constant": dropped,
        "seed": int(seed), "seed_sensitivity": spread,
    }
    return selected, beta, const, cf, meta
