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
        second_norm: Literal["l2", "inf"] = "l2",
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
