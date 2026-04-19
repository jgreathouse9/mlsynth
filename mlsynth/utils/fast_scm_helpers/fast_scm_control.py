# mlsynth/experimental_design/fast_scm_control.py

from typing import List
import numpy as np

from .fast_scm_bb_helpers import Solution
from .structure import SEDCandidate, WeightVectors, PredictionVectors, Losses, Identification
from .fast_scm_control_helpers import solve_control_qp

def _zero_small_weights(weights: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    w = weights.copy()
    w[np.abs(w) < threshold] = 0.0
    return w

def evaluate_candidates(
    candidates: List[Solution],
    X: np.ndarray,
    X_E: np.ndarray,
    Y: np.ndarray,
    f: np.ndarray,
    E_idx: np.ndarray,
    B_idx: np.ndarray,
    post_idx: np.ndarray,
    lambda_penalty: float
) -> List[SEDCandidate]:

    results: List[SEDCandidate] = []

    J = Y.shape[1] if Y.ndim > 1 else 1
    target = X[:, :J] @ f[:J]

    for sol in candidates:
        treated_idx = np.asarray(sol.indices, dtype=int)
        m = len(treated_idx)

        w = sol.weights[treated_idx] if len(sol.weights) != m else sol.weights

        treated_vec_E = X_E[:, treated_idx] @ w
        v = solve_control_qp(X_E, treated_vec_E, treated_idx, lambda_penalty)

        if v is None:
            continue

        synth_treated = X[:, treated_idx] @ w
        synth_control = X @ v
        effects = synth_treated - synth_control

        # Inline NMSE
        def _nmse(period_idx):
            synth_p = synth_treated[period_idx]
            targ_p = target[period_idx]
            std_t = np.maximum(np.std(X[period_idx], axis=1, ddof=1), 1e-8)
            return float(np.mean(((synth_p - targ_p) / std_t) ** 2))

        candidate = SEDCandidate(
            identification=Identification(solution=sol, treated_idx=treated_idx),
            weights=WeightVectors(treated=w.copy(), control=v.copy()),
            predictions=PredictionVectors(
                synthetic_treated=synth_treated.copy(),
                synthetic_control=synth_control.copy(),
                effects=effects.copy(),
                residuals_E=effects[E_idx].copy(),
                residuals_B=effects[B_idx].copy(),
            ),
            losses=Losses(
                loss_E=float(sol.loss),
                nmse_E=_nmse(E_idx),
                nmse_B=_nmse(B_idx),
            )
        )

        results.append(candidate)

    return results