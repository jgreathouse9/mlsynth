# optutils.py

import cvxpy as cp
import numpy as np
from typing import Optional, Literal, Callable, Dict, Any
from .opthelpers import OptHelpers


class Opt2:
    @staticmethod
    def SCopt(
        y: np.ndarray,
        X: np.ndarray,
        *,
        T0: Optional[int] = None,
        fit_intercept: bool = False,
        constraint_type: Literal["unconstrained", "simplex", "affine", "nonneg", "unit"] = "nonneg",
        objective_type: Literal["penalized", "relaxed"] = "penalized",
        relaxation_type: Literal["l2", "entropy", "el"] = "l2",
        lam: float = 0.0,
        alpha: float = 0.5,
        second_norm: Literal["L1_L2", "L1_INF"] = "L1_L2",
        tau: Optional[float] = None,
        custom_penalty_callable: Optional[Callable[[cp.Variable], cp.Expression]] = None,
        custom_objective_callable: Optional[
            Callable[[np.ndarray, np.ndarray, cp.Variable, Optional[cp.Variable]], cp.Objective]
        ] = None,
        solver: str = "CLARABEL",
        tol_abs: float = 1e-6,
        tol_rel: float = 1e-6,
        solve: bool = True,
    ) -> Dict[str, Any]:
        """
                Synthetic Control Optimization (SCopt).

                This method estimates synthetic control weights by solving a constrained optimization problem
                using CVXPY. It supports penalized objectives (loss + regularization) or relaxed objectives
                (regularization-only with relaxed balance constraints). The method can handle pre-treatment
                slicing, intercept fitting, various weight constraints, and custom objectives/penalties.

                Parameters
                ----------
                y : np.ndarray
                    Target outcome vector of shape (T,). The treated unit's outcomes over time.
                X : np.ndarray
                    Donor outcome matrix of shape (T, J). Outcomes for J donor units over T periods.
                T0 : Optional[int], default None
                    Number of pre-treatment periods. If provided, optimization uses only the first T0 periods.
                    If None, uses all periods.
                fit_intercept : bool, default False
                    If True, includes an intercept term (b0) in the model.
                constraint_type : {"unconstrained", "simplex", "affine", "nonneg", "unit"}, default "nonneg"
                    Type of constraints on weights w:
                    - "unconstrained": No constraints.
                    - "simplex": w >= 0, sum(w) == 1.
                    - "affine": sum(w) == 1 (allows negative weights).
                    - "nonneg": w >= 0.
                    - "unit": 0 <= w <= 1.
                objective_type : {"penalized", "relaxed"}, default "penalized"
                    Type of objective:
                    - "penalized": Minimize squared loss + regularization penalty.
                    - "relaxed": Minimize regularization penalty subject to relaxed balance constraints (requires tau).
                relaxation_type : {"l2", "entropy", "el"}, default "l2"
                    Regularization type for "relaxed" objective_type:
                    - "l2": Squared L2 norm.
                    - "entropy": Entropy penalty (sum(w_i * log(w_i))); requires sum-to-one constraints.
                    - "el": Empirical likelihood (sum(-log(w_i))); requires sum-to-one constraints.
                lam : float, default 0.0
                    Regularization strength for "penalized" objective_type.
                alpha : float, default 0.5
                    Elastic net mixing parameter for "penalized" (1 = L1 only, 0 = second_norm only).
                second_norm : {"l2", "inf"}, default "l2"
                    Second norm in elastic net for "penalized" when alpha < 1.
                tau : Optional[float], default None
                    Balance relaxation tolerance for "relaxed" objective_type (required if objective_type="relaxed").
                custom_penalty_callable : Optional[Callable[[cp.Variable], cp.Expression]], default None
                    Custom penalty term to add to the objective (called with w).
                custom_objective_callable : Optional[Callable[[np.ndarray, np.ndarray, cp.Variable, Optional[cp.Variable]], cp.Objective]], default None
                    Fully custom objective; overrides built-in objectives (called with y_opt, X_opt, w, b0).
                solver : str, default "CLARABEL"
                    CVXPY solver name (e.g., "ECOS", "OSQP", "SCS").
                tol_abs : float, default 1e-6
                    Absolute tolerance for solver.
                tol_rel : float, default 1e-6
                    Relative tolerance for solver.
                solve : bool, default True
                    If True, solves the problem; if False, builds but does not solve (returns unsolved problem).

                Returns
                -------
                Dict[str, Any]
                    Dictionary containing:
                    - "problem": The CVXPY Problem object (solved or unsolved).
                    - "weights": Dict with "w" (np.ndarray) and optionally "b0" (float); None if solve=False.
                    - "predictions": np.ndarray of synthetic predictions over full periods; None if solve=False.

                Raises
                ------
                ValueError
                    If incompatible parameters (e.g., entropy relaxation without sum-to-one constraints,
                    relaxed objective without tau, unknown types).

                Examples
                --------
                >>> import numpy as np
                >>> from optutils import Opt2
                >>> y = np.random.randn(10)
                >>> X = np.random.randn(10, 5)
                >>> result = Opt2.SCopt(y, X, T0=5, objective_type="penalized", lam=0.1)
                >>> weights = result["weights"]
                >>> predictions = result["predictions"]
        """

        # ---------- Slice pre-treatment ----------
        if T0 is not None:
            y_opt = y[:T0]
            X_opt = X[:T0, :]
        else:
            y_opt = y
            X_opt = X

        J = X_opt.shape[1]
        w = cp.Variable(J)
        b0 = cp.Variable() if fit_intercept else None

        # ---------- Objective ----------
        if custom_objective_callable is not None:
            objective = custom_objective_callable(y_opt, X_opt, w, b0)
        else:
            objective = OptHelpers.build_objective(
                y_opt,
                X_opt,
                w,
                b0,
                objective_type=objective_type,
                relaxation_type=relaxation_type,
                lam=lam,
                alpha=alpha,
                second_norm=second_norm,
                custom_penalty_callable=custom_penalty_callable,
            )

        # ---------- Constraints ----------
        if objective_type == "relaxed" and relaxation_type in ["entropy", "el"]:
            if constraint_type not in ["simplex", "affine"]:
                raise ValueError(
                    f"{relaxation_type} relaxation requires sum-to-one constraints "
                    "(simplex or affine constraint_type)."
                )

        balance_tau = tau if objective_type == "relaxed" else None

        constraints = OptHelpers.build_constraints(
            w=w,
            constraint_type=constraint_type,
            X=X_opt,
            y=y_opt,
            b0=b0,
            tau=balance_tau,
            objective_type=relaxation_type,
        )

        # ---------- Solver options ----------
        solver_opts = OptHelpers.get_solver_opts(
            solver=solver,
            tol_abs=tol_abs,
            tol_rel=tol_rel,
        )

        # ---------- Solve ----------
        problem = cp.Problem(objective, constraints)

        if solve:
            problem.solve(solver=solver, verbose=False, **solver_opts)


            if w.value is None:
                pass

            weights = {"w": w.value}
            if fit_intercept:
                weights["b0"] = b0.value if b0 is not None else 0.0

            intercept = weights.get("b0", 0.0)
            y_synth_full = X @ weights["w"] + intercept
        else:
            weights = None
            y_synth_full = None

        return {
            "problem": problem,
            "weights": weights,
            "predictions": y_synth_full,
        }
