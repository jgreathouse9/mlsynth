
# mlsynth/experimental_design/fast_scm_control.py

from typing import List, Optional
import numpy as np

from .fast_scm_bb_helpers import Solution
from .structure import SEDCandidate, WeightVectors, PredictionVectors, Losses, Identification
from .fast_scm_control_helpers import solve_control_qp
from .fast_scm_setup import IndexSet
from .conflict import neighbours
from ...exceptions import MlsynthConfigError

def _zero_small_weights(weights: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    """
    Zero out numerically insignificant weights.

    Parameters
    ----------
    weights : np.ndarray
        Input weight vector.
    threshold : float, default=1e-8
        Absolute value below which weights are set to zero.

    Returns
    -------
    w : np.ndarray
        Copy of `weights` with small-magnitude entries replaced by 0.0.

    Notes
    -----
    - Useful for cleaning up numerical noise from optimization solvers.
    - Does not renormalize the weights after thresholding.
    - Preserves the original array (operates on a copy).
    """
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
    lambda_penalty: float,
    index_set: IndexSet,
    conflict: Optional[np.ndarray] = None,
) -> List[SEDCandidate]:


    """
    Evaluate candidate treated-unit subsets by constructing synthetic controls.

    Parameters
    ----------
    candidates : list of Solution
        Candidate subsets from branch-and-bound, each containing indices and weights.
    X : np.ndarray, shape (T, N)
        Full feature matrix (outcomes + covariates).
    X_E : np.ndarray, shape (T_E, N)
        Standardized feature matrix over the estimation period.
    Y : np.ndarray, shape (T, J)
        Outcome matrix.
    f : np.ndarray, shape (N,)
        Weight vector used to construct the target series.
    E_idx : np.ndarray
        Indices for estimation period.
    B_idx : np.ndarray
        Indices for validation/backcast period.
    lambda_penalty : float
        Regularization strength for control QP.

    Returns
    -------
    results : list of SEDCandidate
        Evaluated candidates including weights, predictions, and loss metrics.

    Notes
    -----
    For each candidate subset:

    1. Construct synthetic treated unit::

        treated_vec_E = X_E[:, treated_idx] @ w

    2. Solve for synthetic control weights ``v``.
    3. Compute full time-series:

       - synthetic treated
       - synthetic control
       - treatment effects

    4. Evaluate fit using NMSE on estimation and validation periods.

    Additional Details
    ------------------
    - Target series is defined as::

        target = X[:, :J] @ f[:J]

      (weighted average over outcome units).
    - Candidates with failed QP solves are skipped.
    - Results are packaged into structured SEDCandidate objects.
    """

    

    results: List[SEDCandidate] = []

    J = Y.shape[1] if Y.ndim > 1 else 1
    target = X[:, :J] @ f[:J]

    for sol in candidates:
        treated_idx = np.asarray(sol.indices, dtype=int)
        m = len(treated_idx)

        w = sol.weights[treated_idx] if len(sol.weights) != m else sol.weights

        # Spillover "exclusion restriction": drop the treated units' conflict
        # neighbours N(S) from the donor pool so the treatment cannot contaminate
        # the synthetic control.
        spill = neighbours(conflict, treated_idx) if conflict is not None else None

        treated_vec_E = X_E[:, treated_idx] @ w
        v = solve_control_qp(X_E, treated_vec_E, treated_idx, lambda_penalty,
                             exclude_idx=spill)
        if v is None:
            # The exclusions (treated + N(S)) emptied / over-constrained the donor
            # pool for this candidate; skip it rather than crash.
            continue

        v = _zero_small_weights(v, threshold=1e-8)

        excluded_idx = treated_idx if spill is None or len(spill) == 0 \
            else np.concatenate([treated_idx, np.asarray(spill, dtype=int)])
        assert np.allclose(v[excluded_idx], 0.0, atol=1e-8), \
            "Control QP violated exclusion constraint: a treated unit or its " \
            "spillover neighbour has nonzero control weight"
        

        # label mapping (THIS is the fix)
        control_weights = {
            index_set.labels[i]: r
            for i in range(len(v))
            if (r := round(float(v[i]), 3)) > 0.001
        }

        treated_weights = {
            index_set.labels[treated_idx[i]]: r
            for i in range(len(treated_idx))
            if (r := round(float(w[i]), 3)) > 0.001
        }

        synth_treated = X[:, treated_idx] @ w
        synth_control = X @ v
        effects = synth_treated - synth_control

        # Covariates enter the design as extra rows of X (rows >= n_outcome_rows)
        # and are used in the weight-solve QP for *matching only*. All reported
        # fit metrics (RMSE / NMSE / residuals) must be computed over the OUTCOME
        # rows of the estimation block alone -- covariate values never enter an
        # RMSE in any form; their balance is reported separately as SMD.
        n_outcome_rows = Y.shape[0]
        E_out = E_idx[E_idx < n_outcome_rows]

        def _rmse(a, b, idx):
            return float(np.sqrt(np.mean((a[idx] - b[idx]) ** 2)))

        # Inline NMSE
        def _nmse(period_idx):
            synth_p = synth_treated[period_idx]
            targ_p = target[period_idx]
            std_t = np.maximum(np.std(X[period_idx], axis=1, ddof=1), 1e-8)
            return float(np.mean(((synth_p - targ_p) / std_t) ** 2))

        rmse_sc_E = _rmse(synth_treated, synth_control, E_out)
        rmse_sc_B = _rmse(synth_treated, synth_control, B_idx)

        rmse_pop_E = _rmse(synth_treated, target, E_out)
        rmse_pop_B = _rmse(synth_treated, target, B_idx)

        candidate = SEDCandidate(
            identification=Identification(
                solution=sol,
                treated_idx=treated_idx
            ),

            weights=WeightVectors(
                treated=w.copy(),
                control=v.copy()
            ),

            predictions=PredictionVectors(
                synthetic_treated=synth_treated.copy(),
                synthetic_control=synth_control.copy(),
                effects=effects.copy(),
                residuals_E=effects[E_out].copy(),
                residuals_B=effects[B_idx].copy(),
            ),

            losses=Losses(
                loss_E=float(sol.loss),
                nmse_E=_nmse(E_out),
                nmse_B=_nmse(B_idx),

                rmse_sc_E=rmse_sc_E,
                rmse_sc_B=rmse_sc_B,

                rmse_pop_E=rmse_pop_E,
                rmse_pop_B=rmse_pop_B
            ),

            # NEW structured metadata
            treated_weight_dict=treated_weights,
            control_weight_dict=control_weights
        )

        results.append(candidate)

    if not results and candidates and conflict is not None:
        raise MlsynthConfigError(
            "Every candidate design had its donor pool emptied by the spillover "
            "'exclusion restriction' (treated units plus their conflict "
            "neighbours leave too few controls). Relax the adjacency/cluster "
            "constraint or widen the donor pool."
        )

    return results


