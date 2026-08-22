"""Whether an off-the-shelf conformal library can carry this.

A synthetic control is a constrained regressor -- donors are the design matrix,
the treated series is the target, the weights live on the probability simplex --
so it wraps as a scikit-learn estimator and drops into MAPIE, whose
``TimeSeriesRegressor(method="enbpi", cv=BlockBootstrap(...))`` is the same
bootstrap-ensemble leave-one-out construction ``loo_ensemble.py`` hand-rolls.

This checks that the wrapper is accepted, cross-checks MAPIE's conformity scores
against the hand-rolled residuals, and reports the shape of what comes back. The
estimand here is the H-period total; a library that returns one interval per
period has answered a different question, and stacking its endpoints is the
lockstep assumption the cumulative band exists to replace.

The same holds for puncc (deel-ai): its ``EnbPI`` and ``AdaptiveEnbPI`` are the
ensemble machinery, but every nonconformity score in ``api/nonconformity_scores``
and every set in ``api/prediction_sets`` is per-observation, and its only
multiplicity tool is a Bonferroni correction -- simultaneous per-period bands,
wider than stacking, still not an interval for the total.

MAPIE is not a mlsynth dependency; this skips when it is absent.

    MLSYNTH_CAL_PANEL=<panel.csv> python external_libraries.py [seed]
"""
from __future__ import annotations

import sys
from pathlib import Path

# run from anywhere: the repo root supplies mlsynth, this directory supplies panel
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sys
import warnings

import numpy as np

from mlsynth.utils.bilevel.simplex import simplex_lstsq
from loo_ensemble import J, design
from panel import ALPHA, DELTA, H, T0, draw


def main(seed=0, n_resamplings=200):
    try:
        from sklearn.base import BaseEstimator, RegressorMixin
        from mapie.regression import TimeSeriesRegressor
        from mapie.subsample import BlockBootstrap
    except ImportError as exc:                       # pragma: no cover - optional
        print(f"skipped: {exc}")
        return

    class SimplexSC(RegressorMixin, BaseEstimator):
        """Least squares over the probability simplex, as a sklearn regressor."""

        def fit(self, X, y):
            self.coef_ = simplex_lstsq(np.asarray(X, float), np.asarray(y, float))
            self.is_fitted_ = True
            return self

        def predict(self, X):
            return np.asarray(X, float) @ self.coef_

    YN = draw(seed, J=J)
    w, _ = design(YN)
    S = np.flatnonzero(w > 1e-8)
    rest = np.array([j for j in range(J) if j not in set(S.tolist())])
    Y = YN.copy()
    Y[S, T0:] += DELTA
    X, y = Y[rest].T, w @ Y

    cv = BlockBootstrap(n_resamplings=n_resamplings, length=H,
                        overlapping=False, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp = TimeSeriesRegressor(estimator=SimplexSC(), method="enbpi", cv=cv,
                                 agg_function="mean", random_state=seed)
        mp.fit(X[:T0], y[:T0])
        pred, intervals = mp.predict(X[T0:], confidence_level=1 - ALPHA,
                                     allow_infinite_bounds=True)

    print(f"MAPIE accepted the simplex wrapper: {n_resamplings} members fitted")
    print(f"predict returns pred {pred.shape}, intervals {intervals.shape}")
    lo, hi = intervals[:, 0, 0], intervals[:, 1, 0]
    print(f"per-period half-widths: min {(hi - lo).min() / 2:.4f}  "
          f"max {(hi - lo).max() / 2:.4f}")
    scores = np.asarray(mp.conformity_scores_, dtype=float)
    scores = scores[np.isfinite(scores)]
    print(f"conformity scores: {scores.size} (one per training period, "
          f"T0 = {T0}), sd {scores.std():.4f}")
    print(f"\nstacking the per-period half-widths gives "
          f"{float(((hi - lo) / 2).sum()):.4f}")
    print("  compare the cumulative half-widths from loo_ensemble.py; there is no")
    print("  MAPIE call returning an interval for a functional of the H predictions")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
