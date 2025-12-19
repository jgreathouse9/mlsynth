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
        constraint_type: Literal["unconstrained", "simplex", "affine", "nonneg"] = "nonneg",
        objective_type: Literal["penalized", "relaxed"] = "penalized",
        relaxation_type: Literal["l2", "entropy", "EL"] = "l2",
        lam: float = 0.0,
        alpha: float = 0.5,
        second_norm: Literal["l2", "linf"] = "l2",
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
        """Synthetic Control Optimization (SCopt)."""

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
            if objective_type == "penalized":
                loss = OptHelpers.squared_loss(y_opt, X_opt, w, b0, scale=False)

                if alpha == 0.0:
                    penalty = lam * OptHelpers.l2_only_penalty(w)
                else:
                    penalty = OptHelpers.elastic_net_penalty(
                        w, lam, alpha, second_norm
                    )

                extra_penalty = (
                    custom_penalty_callable(w) if custom_penalty_callable else 0.0
                )
                objective = cp.Minimize(loss + penalty + extra_penalty)

            elif objective_type == "relaxed":
                if relaxation_type == "l2":
                    base_obj = OptHelpers.l2_only_penalty(w)

                elif relaxation_type == "entropy":
                    if constraint_type not in ["simplex", "affine"]:
                        raise ValueError(
                            "Entropy relaxation requires sum-to-one constraints "
                            "(simplex or affine constraint_type)."
                        )
                    base_obj = OptHelpers.entropy_penalty(w)

                elif relaxation_type == "EL":
                    if constraint_type not in ["simplex", "affine"]:
                        raise ValueError(
                            "EL relaxation requires sum-to-one constraints "
                            "(simplex or affine constraint_type)."
                        )
                    base_obj = OptHelpers.el_penalty(w)

                else:
                    raise ValueError(
                        f"Unknown relaxation_type: {relaxation_type}"
                    )

                extra_penalty = (
                    custom_penalty_callable(w) if custom_penalty_callable else 0.0
                )
                objective = cp.Minimize(base_obj + extra_penalty)

            else:
                raise ValueError(
                    f"Unknown objective_type: {objective_type}"
                )

        # ---------- Constraints ----------
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
        solver_opts: Dict[str, Any] = {}
        solver_upper = solver.upper()

        if solver_upper in ["OSQP", "ECOS", "SCS"]:
            if solver_upper == "OSQP":
                solver_opts["eps_abs"] = tol_abs
                solver_opts["eps_rel"] = tol_rel
            else:
                solver_opts["abstol"] = tol_abs
                solver_opts["reltol"] = tol_rel

        # ---------- Solve ----------
        problem = cp.Problem(objective, constraints)

        if solve:
            problem.solve(solver=solver, verbose=False, **solver_opts)

            if w.value is None:
                raise ValueError("Optimization failed: no solution found for weights.")

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

