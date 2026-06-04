"""Bilevel SCM engine for the ``VanillaSC`` estimator.

A thin, ``dataprep``-agnostic wrapper that turns the self-contained bilevel
machinery (:mod:`mlsynth.utils.fscm_helpers.bilevel`) into the *standard*
single-treated synthetic control, with a selectable predictor-weight backend:

* ``"outcome-only"`` -- no covariates: the donor weights solve the convex
  simplex least-squares fit on the pre-treatment outcomes (Abadie's outcome
  matching). Well-posed and unique up to donor collinearity.
* ``"malo"`` / ``"mscmt"`` -- covariate matching via the bilevel program
  (predictor weights ``V`` + donor weights ``W``): Malo et al. (2024) corner
  search or Becker-Kloessner (2018) global differential evolution.
* ``"penalized"`` -- Abadie-L'Hour (2021) pairwise-penalized estimator (a
  unique, sparse ``W``); works with or without covariates.

The engine takes plain NumPy arrays so it can be unit-tested in isolation and
reused outside the estimator. The ``VanillaSC`` estimator feeds it the matrices
produced by :func:`mlsynth.utils.datautils.dataprep`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..fscm_helpers.bilevel import (
    BilevelProblem,
    canonical_v_diagnostics,
    solve_bilevel,
)
from ..fscm_helpers.bilevel.simplex import simplex_lstsq

_BACKENDS = ("auto", "outcome-only", "malo", "mscmt", "penalized")
# Keyword arguments each bilevel backend accepts (others are filtered out so a
# single ``solver_kwargs`` blob can be passed regardless of backend).
_SOLVER_KWARGS = {
    "mscmt": {"lb", "maxiter", "popsize", "tol", "seed", "polish", "feas_tol"},
    "malo": {"feas_tol", "eps_corner", "refine", "refine_gap_tol"},
    "penalized": {"lam", "lam_grid", "max_iter", "tol"},
}
_EPS = 1e-10


@dataclass
class BilevelSCMResult:
    """Output of :meth:`BilevelSCM.fit`.

    Attributes
    ----------
    W : np.ndarray
        Donor weights, shape ``(J,)``, on the simplex.
    donor_weights : dict
        ``{donor_name: weight}`` for weight-bearing donors.
    V : np.ndarray or None
        Predictor (covariate) weights, shape ``(P,)``, normalised to the
        simplex; ``None`` for outcome-only matching.
    predictor_names : list of str
        Names of the ``P`` predictors (empty for outcome-only).
    backend : str
        The backend actually used.
    pre_rmspe : float
        Root mean squared prediction error on the pre-treatment outcomes.
    v_agreement : float or None
        Identification diagnostic: max abs difference between the
        ``min.loss.w`` and ``max.order`` canonical predictor weights. Small =
        ``V`` well identified; large = predictor weights fragile. ``None`` for
        outcome-only.
    diagnostics : dict
        Free-form solver diagnostics (stage, gap, lower bound, ...).
    """

    W: np.ndarray
    donor_weights: Dict[str, float]
    V: Optional[np.ndarray]
    predictor_names: List[str]
    backend: str
    pre_rmspe: float
    v_agreement: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def counterfactual(self, donor_matrix: np.ndarray) -> np.ndarray:
        """Synthetic outcome path ``donor_matrix @ W`` over all periods.

        ``donor_matrix`` has shape ``(T, J)`` (donors in columns, matching the
        order of ``W``).
        """
        return np.asarray(donor_matrix, dtype=float) @ self.W


class BilevelSCM:
    """Standard single-treated synthetic control via the bilevel solver.

    Parameters
    ----------
    backend : {"auto", "outcome-only", "malo", "mscmt", "penalized"}
        Predictor-weight backend. ``"auto"`` (default) picks ``"outcome-only"``
        when no covariates are supplied and ``"mscmt"`` when they are.
    canonical_v : bool or {"min.loss.w", "max.order"}
        Canonicalise the (non-identified) predictor weights ``V`` for the
        ``mscmt`` backend (see
        :func:`mlsynth.utils.fscm_helpers.bilevel.determine_v.canonical_v`).
        Ignored by other backends. Default ``False``.
    seed : int
        RNG seed for the ``mscmt`` differential-evolution search.
    solver_kwargs
        Extra keyword arguments forwarded to the backend (e.g. ``maxiter``,
        ``popsize`` for ``mscmt``; ``lam`` for ``penalized``).
    """

    def __init__(
        self,
        backend: str = "auto",
        *,
        canonical_v=False,
        seed: int = 0,
        **solver_kwargs: Any,
    ) -> None:
        if backend not in _BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; expected one of {_BACKENDS}."
            )
        self.backend = backend
        self.canonical_v = canonical_v
        self.seed = seed
        self.solver_kwargs = solver_kwargs

    def fit(
        self,
        y_pre: np.ndarray,
        Y0_pre: np.ndarray,
        *,
        X1: Optional[np.ndarray] = None,
        X0: Optional[np.ndarray] = None,
        donor_names: Optional[List[str]] = None,
        predictor_names: Optional[List[str]] = None,
    ) -> BilevelSCMResult:
        """Solve for the donor weights ``W``.

        Parameters
        ----------
        y_pre : np.ndarray
            Treated pre-treatment outcomes, shape ``(T0,)``.
        Y0_pre : np.ndarray
            Donor pre-treatment outcomes, shape ``(T0, J)``.
        X1 : np.ndarray, optional
            Treated predictor (covariate) values, shape ``(P,)``.
        X0 : np.ndarray, optional
            Donor predictor matrix, shape ``(P, J)``.
        donor_names : list of str, optional
            Donor labels (defaults to ``donor_0 ...``).
        predictor_names : list of str, optional
            Predictor labels.
        """
        y_pre = np.asarray(y_pre, dtype=float).ravel()
        Y0_pre = np.asarray(Y0_pre, dtype=float)
        if Y0_pre.ndim != 2 or Y0_pre.shape[0] != y_pre.shape[0]:
            raise ValueError(
                f"Y0_pre must be (T0, J) with T0={y_pre.shape[0]}, got {Y0_pre.shape}."
            )
        J = Y0_pre.shape[1]
        if donor_names is None:
            donor_names = [f"donor_{j}" for j in range(J)]
        has_cov = X1 is not None and X0 is not None

        backend = self.backend
        if backend == "auto":
            backend = "mscmt" if has_cov else "outcome-only"
        if backend in ("malo", "mscmt") and not has_cov:
            raise ValueError(
                f"backend {backend!r} needs covariates (X1/X0); for outcome-only "
                "matching use backend='outcome-only' or 'penalized'."
            )

        V = None
        v_agreement = None
        pred_names: List[str] = []
        diagnostics: Dict[str, Any] = {}

        if backend == "outcome-only" or not has_cov:
            backend = "outcome-only"
            W = simplex_lstsq(Y0_pre, y_pre)
        else:
            X1 = np.asarray(X1, dtype=float).ravel()
            X0 = np.asarray(X0, dtype=float)
            if X0.shape != (X1.shape[0], J):
                raise ValueError(
                    f"X0 must be (P, J)=({X1.shape[0]}, {J}), got {X0.shape}."
                )
            pred_names = list(predictor_names or [f"x{p}" for p in range(X1.shape[0])])
            prob = BilevelProblem(
                y1_pre=y_pre, Y0_pre=Y0_pre, X1=X1, X0=X0, predictor_names=pred_names,
            )
            # Forward only kwargs the chosen backend accepts.
            kw = {k: v for k, v in self.solver_kwargs.items()
                  if k in _SOLVER_KWARGS[backend]}
            if backend == "mscmt":
                kw.setdefault("seed", self.seed)
                kw["canonical_v"] = self.canonical_v
            sol = solve_bilevel(prob, method=backend, **kw)
            W = sol.W
            V = sol.V
            diagnostics = dict(sol.metadata)
            diagnostics["stage"] = sol.stage
            v_agreement = sol.metadata.get("v_agreement")
            if v_agreement is None:
                # Always try to surface the identification diagnostic.
                try:
                    v_agreement = canonical_v_diagnostics(prob, W)["agreement"]
                except Exception:
                    v_agreement = None
            if v_agreement is not None and not np.isfinite(v_agreement):
                v_agreement = None  # canonicalisation did not certify

        W = np.asarray(W, dtype=float).ravel()
        pre_rmspe = float(np.sqrt(np.mean((y_pre - Y0_pre @ W) ** 2)))
        donor_weights = {
            donor_names[j]: float(W[j]) for j in range(J) if abs(W[j]) > _EPS
        }
        return BilevelSCMResult(
            W=W,
            donor_weights=donor_weights,
            V=(np.asarray(V, dtype=float) if V is not None else None),
            predictor_names=pred_names,
            backend=backend,
            pre_rmspe=pre_rmspe,
            v_agreement=(float(v_agreement) if v_agreement is not None else None),
            diagnostics=diagnostics,
        )
