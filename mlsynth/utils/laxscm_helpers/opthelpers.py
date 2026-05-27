## 12/22/2025

import cvxpy as cp
import numpy as np
from typing import Optional, Literal, List, Callable, Dict, Any


class OptHelpers:
    """
    Collection of helper functions for building objectives and constraints
    in synthetic control and balancing weight optimization problems.

    These helpers do not solve optimization problems themselves.
    They return CVXPY expressions or constraint lists to be assembled
    by a higher-level solver (e.g., Opt.SCopt).
    """

    # =======================
    # Loss helpers
    # =======================

    @staticmethod
    def squared_loss(
        y: np.ndarray,
        X: np.ndarray,
        w: cp.Variable,
        b0: Optional[cp.Variable] = None,
        scale: bool = True,
    ) -> cp.Expression:
        """
        Construct a squared-error loss term.

        Parameters
        ----------
        y : np.ndarray
            Target outcome vector of shape (T,).
        X : np.ndarray
            Donor outcome matrix of shape (T, J).
        w : cp.Variable
            Weight vector of shape (J,).
        b0 : cp.Variable, optional
            Intercept term. If None, no intercept is included.
        scale : bool, default True
            If True, divide the loss by the number of observations T.

        Returns
        -------
        cp.Expression
            A CVXPY expression representing the (scaled) squared loss.
        """
        T = y.shape[0]
        residual = y - (X @ w + b0) if b0 is not None else y - X @ w
        loss = cp.sum_squares(residual)
        return loss / T if scale else loss

    # =======================
    # Penalty helpers
    # =======================

    @staticmethod
    def elastic_net_penalty(
        w: cp.Variable,
        lam: float = 0.0,
        alpha: float = 0.5,
        second_norm: Literal["L1_L2", "L1_INF"] = "L1_L2",
    ) -> cp.Expression:
        """
        Construct an elastic-net-style penalty.

        Parameters
        ----------
        w : cp.Variable
            Weight vector of shape (J,).
        lam : float, default 0.0
            Overall penalty strength.
        alpha : float, default 0.5
            Mixing parameter between L1 and second norm.
            alpha = 1   -> pure L1
            alpha = 0   -> pure second norm
        second_norm : {"l2", "inf"}, default "l2"
            Choice of second norm when alpha < 1.

        Returns
        -------
        cp.Expression
            A CVXPY expression representing the penalty term.
        """
        if lam == 0:
            return 0.0

        penalty = 0.0

        if alpha > 0:
            penalty += alpha * cp.norm(w, 1)

        if alpha < 1:
            if second_norm == "L1_L2":
                penalty += (1 - alpha) * cp.norm(w, 2)
            elif second_norm == "L1_INF":
                penalty += (1 - alpha) * cp.norm(w, "inf")
            else:
                raise ValueError(f"Unknown second_norm: {second_norm}")

        return lam * penalty

    @staticmethod
    def l2_only_penalty(w: cp.Variable) -> cp.Expression:
        """
        Construct a pure L2-squared penalty.

        Parameters
        ----------
        w : cp.Variable
            Weight vector of shape (J,).

        Returns
        -------
        cp.Expression
            Squared L2 norm of the weight vector.
        """
        return cp.norm(w, 2) ** 2

    @staticmethod
    def entropy_penalty(w: cp.Variable) -> cp.Expression:
        """
        Construct an entropy penalty: sum(w_i * log(w_i)).

        This minimizes the negative entropy (equivalent to maximizing entropy under constraints).

        Note: Requires w > 0; use with sum-to-one and non-negativity constraints.

        Parameters
        ----------
        w : cp.Variable
            Weight vector of shape (J,).

        Returns
        -------
        cp.Expression
            Sum of w_i * log(w_i).
        """
        return cp.sum(-cp.entr(w))

    @staticmethod
    def el_penalty(w: cp.Variable) -> cp.Expression:
        """
        Construct an empirical likelihood penalty: sum(-log(w_i)).

        This minimizes the negative log-likelihood under constraints.

        Note: Requires w > 0; use with sum-to-one and non-negativity constraints.

        Parameters
        ----------
        w : cp.Variable
            Weight vector of shape (J,).

        Returns
        -------
        cp.Expression
            Sum of -log(w_i).
        """
        return cp.sum(-cp.log(w))

    # =======================
    # Constraint helpers
    # =======================

    @staticmethod
    def simplex_constraints(w: cp.Variable) -> List:
        """
        Construct simplex constraints.

        Parameters
        ----------
        w : cp.Variable
            Weight vector of shape (J,).

        Returns
        -------
        list
            Constraints enforcing w >= 0 and sum(w) == 1.
        """
        return [w >= 0, cp.sum(w) == 1]

    @staticmethod
    def affine_constraints(w: cp.Variable) -> List:
        """
        Construct affine (sum-to-one) constraints.

        Parameters
        ----------
        w : cp.Variable
            Weight vector of shape (J,).

        Returns
        -------
        list
            Constraint enforcing sum(w) == 1.
        """
        return [cp.sum(w) == 1]

    @staticmethod
    def nonneg_constraints(w: cp.Variable) -> List:
        """
        Construct non-negativity constraints.

        Parameters
        ----------
        w : cp.Variable
            Weight vector of shape (J,).

        Returns
        -------
        list
            Constraint enforcing w >= 0.
        """
        return [w >= 0]

    @staticmethod
    def unit_constraints(w: cp.Variable) -> List:
        """
        Construct unit constraints.

        Parameters
        ----------
        w : cp.Variable
            Weight vector of shape (J,).

        Returns
        -------
        list
            Constraints enforcing 0 <= w_i <= 1 for all i.
        """
        return [w >= 0, w <= 1]



    @staticmethod
    def relaxed_balance_constraints(
            X: np.ndarray,
            residual: cp.Expression,
            tau: float,
            objective_type: Literal["l2", "entropy", "el"] = "l2"
    ) -> tuple[list, cp.Variable]:
        """
        Relaxed balance constraints with gamma slack variable,
        now supporting L2, entropy, and empirical likelihood (EL) forms.

        Parameters
        ----------
        X : np.ndarray
            Donor outcome matrix (T, J)
        residual : cp.Expression
            Residual vector y - Xw (or y - Xw - b0)
        tau : float
            Balance relaxation tolerance
        objective_type : {"l2", "entropy", "el"}
            Form of relaxed balance

        Returns
        -------
        constraints : list
            CVXPY constraints enforcing relaxed balance
        gam : cp.Variable
            Scalar slack variable
        """
        if objective_type not in ["l2", "entropy", "el"]:
            raise ValueError(f"Unknown objective_type: {objective_type}")

        gam = cp.Variable()  # scalar slack
        constraints = [cp.norm(X.T @ residual / X.shape[0] + gam * np.ones(X.shape[1]), "inf") <= tau]
        return constraints, gam

    @staticmethod
    def build_constraints(
            w: cp.Variable,
            constraint_type: Literal["unconstrained", "simplex", "affine", "nonneg", "unit"] = "nonneg",
            X: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            b0: Optional[cp.Variable] = None,
            tau: Optional[float] = None,
            objective_type: Literal["l2", "entropy", "el"] = "l2"
    ) -> List:
        constraints = []

        # Structural constraints
        if constraint_type == "simplex":
            constraints += OptHelpers.simplex_constraints(w)
        elif constraint_type == "affine":
            constraints += OptHelpers.affine_constraints(w)
        elif constraint_type == "nonneg":
            constraints += OptHelpers.nonneg_constraints(w)
        elif constraint_type == "unit":
            constraints += OptHelpers.unit_constraints(w)
        elif constraint_type == "unconstrained":
            pass
        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type}")

        # Relaxed balance constraint (gamma)
        if tau is not None and X is not None and y is not None:
            residual = y - (X @ w + b0 if b0 is not None else X @ w)
            relax_constraints, gam = OptHelpers.relaxed_balance_constraints(
                X, residual, tau, objective_type=objective_type
            )
            constraints += relax_constraints

        return constraints

    @staticmethod
    def build_objective(
        y: np.ndarray,
        X: np.ndarray,
        w: cp.Variable,
        b0: Optional[cp.Variable],
        objective_type: Literal["penalized", "relaxed"] = "penalized",
        relaxation_type: Literal["l2", "entropy", "el"] = "l2",
        lam: float = 0.0,
        alpha: float = 0.5,
        second_norm: Literal["L1_L2", "L1_INF"] = "L1_L2",
        custom_penalty_callable: Optional[Callable[[cp.Variable], cp.Expression]] = None,
    ) -> cp.Objective:
        """
        Construct the CVXPY objective for penalized or relaxed optimization.

        Parameters
        ----------
        y : np.ndarray
            Target outcome vector.
        X : np.ndarray
            Donor outcome matrix.
        w : cp.Variable
            Weight vector.
        b0 : cp.Variable, optional
            Intercept term.
        objective_type : {"penalized", "relaxed"}, default "penalized"
            Type of objective (loss + penalty or penalty-only with relaxed constraints).
        relaxation_type : {"l2", "entropy", "el"}, default "l2"
            Type of relaxation for "relaxed" objective.
        lam : float, default 0.0
            Penalty strength for "penalized".
        alpha : float, default 0.5
            L1 mixing parameter for elastic net in "penalized".
        second_norm : {"l2", "inf"}, default "l2"
            Second norm for elastic net in "penalized".
        custom_penalty_callable : callable, optional
            Additional custom penalty term.

        Returns
        -------
        cp.Objective
            The minimization objective.
        """
        if objective_type == "penalized":
            loss = OptHelpers.squared_loss(y, X, w, b0, scale=False)
            if alpha == 0.0:
                penalty = lam * OptHelpers.l2_only_penalty(w)
            else:
                penalty = OptHelpers.elastic_net_penalty(w, lam, alpha, second_norm)
            extra_penalty = custom_penalty_callable(w) if custom_penalty_callable else 0.0
            return cp.Minimize(loss + penalty + extra_penalty)

        elif objective_type == "relaxed":
            if relaxation_type == "l2":
                base_obj = OptHelpers.l2_only_penalty(w)
            elif relaxation_type == "entropy":
                base_obj = OptHelpers.entropy_penalty(w)
            elif relaxation_type == "el":
                base_obj = OptHelpers.el_penalty(w)
            else:
                raise ValueError(f"Unknown relaxation_type: {relaxation_type}")
            extra_penalty = custom_penalty_callable(w) if custom_penalty_callable else 0.0
            return cp.Minimize(base_obj + extra_penalty)

        else:
            raise ValueError(f"Unknown objective_type: {objective_type}")

    @staticmethod
    def get_solver_opts(
        solver: str = "CLARABEL",
        tol_abs: float = 1e-6,
        tol_rel: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Get solver-specific options.

        Parameters
        ----------
        solver : str, default "CLARABEL"
            CVXPY solver name.
        tol_abs : float, default 1e-6
            Absolute tolerance.
        tol_rel : float, default 1e-6
            Relative tolerance.

        Returns
        -------
        dict
            Solver options dictionary.
        """
        solver_opts: Dict[str, Any] = {}
        solver_upper = solver.upper()

        if solver_upper in ["OSQP", "ECOS", "SCS"]:
            if solver_upper == "OSQP":
                solver_opts["eps_abs"] = tol_abs
                solver_opts["eps_rel"] = tol_rel
            else:
                solver_opts["abstol"] = tol_abs
                solver_opts["reltol"] = tol_rel

        return solver_opts