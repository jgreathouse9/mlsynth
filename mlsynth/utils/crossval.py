import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from mlsynth.utils.optutils import Opt2
from mlsynth.utils.resultutils import effects
from typing import Any, Tuple, List, Optional, Dict, Union, Callable, Literal


class RelaxationCV(BaseEstimator, RegressorMixin):
    """
    Cross-validated Relaxed Synthetic Control Method (SCM) wrapper.

    Uses Opt2.SCopt to compute synthetic control weights under relaxed-balance
    constraints. Always enforces simplex constraints (weights sum to 1 and are
    non-negative) and does not fit an intercept.

    Parameters
    ----------
    tau : float or array-like, optional
        Relaxation tolerance(s) for relaxed-balance constraints. If None, a
        default grid is generated automatically.
    n_taus : int, default=1000
        Number of tau values to generate if tau=None.
    n_splits : int, default=10
        Number of folds for time-series cross-validation.
    nonneg : bool, default=True
        If True, enforce non-negativity (always True for simplex weights).
    solver : str, default="CLARABEL"
        CVXPY solver used for SCopt optimization.
    """

    def __init__(self, *,
                 tau=None,
                 n_taus=1000,
                 n_splits=10,
                 nonneg=True,
                 solver="CLARABEL",
                 relaxation_type: str = "l2"):
        self.tau = tau
        self.n_taus = n_taus
        self.n_splits = n_splits
        self.nonneg = nonneg
        self.solver = solver
        self.relaxation_type = relaxation_type

    def _generate_tau_grid(self, X: np.ndarray, y: np.ndarray):
        """
        Generate a geometric grid of tau values for cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Donor matrix (T x J).
        y : np.ndarray
            Treated unit outcome vector (length T).

        Notes
        -----
        The grid is descending, from max L-infinity norm of X.T @ y down to 1e-4.
        """
        lower_limit = 1e-4
        upper_limit = np.linalg.norm(X.T @ y, np.inf)
        self.taus_ = np.geomspace(upper_limit, lower_limit, self.n_taus)

    def _process_tau_grid(self, X: np.ndarray, y: np.ndarray):
        """
        Prepare tau grid depending on user input.

        Parameters
        ----------
        X : np.ndarray
            Donor matrix (T x J).
        y : np.ndarray
            Treated unit outcome vector (length T).

        Sets
        ----
        taus_ : np.ndarray
            Tau values for cross-validation.
        tau_ : float
            Selected tau if single value.
        skip_cv_ : bool
            Whether to skip cross-validation (True if tau is scalar).
        """
        if np.isscalar(self.tau):
            self.taus_ = np.array([self.tau])
            self.tau_ = self.tau
            self.skip_cv_ = True
        elif self.tau is not None:
            self.taus_ = np.sort(np.array(self.tau))[::-1]
            self.skip_cv_ = False
        else:
            self._generate_tau_grid(X, y)
            self.skip_cv_ = False

    def _solve_relax_problem(self, X: np.ndarray, y: np.ndarray, tau: float = None) -> np.ndarray:
        """
        Solve a relaxed SCM optimization problem for a given tau.

        Parameters
        ----------
        X : np.ndarray
            Donor matrix (T x J).
        y : np.ndarray
            Treated unit outcome vector (length T).
        tau : float, optional
            Relaxation tolerance. If None, uses self.tau_.

        Returns
        -------
        np.ndarray
            Optimized weight vector (length J).

        Notes
        -----
        If the solver fails, returns a zero vector of length J.
        """
        tau_val = self.tau_ if tau is None else tau
        try:
            sc_res = Opt2.SCopt(
                y=y,
                X=X,
                T0=X.shape[0],
                fit_intercept=False,
                constraint_type="simplex",
                objective_type="relaxed",
                relaxation_type=self.relaxation_type,
                lam=0.0,
                tau=tau_val,
                solver=self.solver,
                tol_abs=1e-8,
                tol_rel=1e-8
            )

            w = sc_res["weights"]["w"]

            if hasattr(w, "value") and w.value is not None:
                w = w.value.flatten()
            elif isinstance(w, np.ndarray):
                w = w.flatten()
            else:
                w = np.zeros(X.shape[1], dtype=np.float64)

        except Exception as e:
            print(f"Solver failed for tau={tau_val}: {e}")
            w = np.zeros(X.shape[1], dtype=np.float64)

        return w

    def _fit_fold(self, X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
        """
        Fit the relaxed SCM for a single fold of time-series CV.

        Parameters
        ----------
        X : np.ndarray
            Donor matrix (T x J).
        y : np.ndarray
            Treated unit outcome vector (length T).
        train_idx : np.ndarray
            Indices of training samples.
        test_idx : np.ndarray
            Indices of test samples.

        Returns
        -------
        np.ndarray
            Fold mean squared error for each tau.
        """
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        w_estimates = np.column_stack([
            np.atleast_1d(self._solve_relax_problem(X_train, y_train, tau=val))
            for val in self.taus_
        ])
        y_pred = X_test @ w_estimates
        return np.mean((y_test[:, None] - y_pred) ** 2, axis=0)

    def _cross_validate(self, X: np.ndarray, y: np.ndarray):
        """
        Perform time-series cross-validation to select optimal tau.

        Parameters
        ----------
        X : np.ndarray
            Donor matrix (T x J).
        y : np.ndarray
            Treated unit outcome vector (length T).

        Sets
        ----
        tau_ : float
            Selected tau minimizing CV MSE.
        cv_mean_mse_ : np.ndarray
            Mean squared error across folds for each tau.
        cv_performed_ : bool
            True if cross-validation was executed.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_errors = []

        for train_idx, test_idx in tscv.split(X):
            fold_errors = self._fit_fold(X, y, train_idx, test_idx)
            if fold_errors is not None:
                cv_errors.append(fold_errors)

        if len(cv_errors) == 0:
            raise ValueError("No feasible taus for any fold.")

        min_length = np.min([len(errors) for errors in cv_errors])
        cv_errors = np.array([errors[:min_length] for errors in cv_errors])
        self.taus_ = self.taus_[:min_length]

        self.cv_mean_mse_ = np.mean(cv_errors, axis=0)
        self.tau_ = self.taus_[np.argmin(self.cv_mean_mse_)]
        self.cv_performed_ = True

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the RelaxationCV model.

        Parameters
        ----------
        X : np.ndarray
            Donor matrix (T x J).
        y : np.ndarray
            Treated unit outcome vector (length T).

        Returns
        -------
        self : RelaxationCV
            Fitted instance with selected tau and computed weights.
        """
        X, y = X.astype(np.float64), y.astype(np.float64)
        self._process_tau_grid(X, y)

        if not getattr(self, "skip_cv_", False):
            self._cross_validate(X, y)
        else:
            self.cv_performed_ = False
            self.tau_ = self.taus_[0]

        self.coef_ = self._solve_relax_problem(X, y, tau=self.tau_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outcomes for new donor data using fitted weights.

        Parameters
        ----------
        X : np.ndarray
            Donor matrix (T x J) or (n_samples x J).

        Returns
        -------
        np.ndarray
            Predicted synthetic control outcomes.
        """
        X = X.astype(np.float64)
        return X @ self.coef_






def fit_relaxed_scm(
    X_pre: np.ndarray,
    y_pre: np.ndarray,
    X_post: np.ndarray,
    donor_names: list = None,
    y: np.ndarray = None,
    tau: float | list[float] | None = None,
    n_splits: int = 5,
    n_taus: int = 1000,
    solver: str = "CLARABEL",
        relaxation_type: str = "l2"
) -> dict:
    """
    Fit a Relaxed L2 Synthetic Control Method (SCM) with optional cross-validation
    over the relaxation parameter tau, and return donor weights, predictions,
    and average treatment effect on the treated (ATT) results.

    Parameters
    ----------
    X_pre : np.ndarray, shape (T0, J)
        Pre-treatment donor matrix (T0 time points, J donor units).
    y_pre : np.ndarray, shape (T0,)
        Pre-treatment outcome vector for treated unit.
    X_post : np.ndarray, shape (T1, J)
        Post-treatment donor matrix (T1 time points, J donor units).
    donor_names : list of str, optional
        Names of donor units corresponding to columns of X_pre/X_post.
        Used to label donor weights in output.
    y : np.ndarray, shape (T0+T1,), optional
        Full outcome vector of treated unit. Used for ATT computation.
        If None, ATT effects will be computed only from predictions.
    tau : float or list of floats, optional
        Relaxation tolerance(s) for relaxed-balance constraints. If None, a
        default geometric grid of taus is used for cross-validation.
    n_splits : int, default=5
        Number of folds for time-series cross-validation when selecting tau.
    n_taus : int, default=1000
        Number of tau values to generate if tau=None.
    solver : str, default="CLARABEL"
        CVXPY solver used for SCopt optimization.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'donor_weights' : dict
            Mapping of donor unit names to optimized weights. Weights < 0.001 are set to 0.
        - 'predictions' : np.ndarray, shape (T0 + T1,)
            Predicted synthetic control outcomes for pre- and post-treatment periods.
        - 'Results' : dict
            - 'Effects' : dict
                ATT / treatment effect results computed using `effects.calculate`.
            - 'Fit' : dict
                Fit diagnostics for pre- and post-treatment periods.
            - 'Vectors' : dict
                Additional vectors from `effects.calculate` (e.g., residuals, weights).
        - 'Model' : str
            Description of model used ('Relaxed SCM (l2)').
        - 'hyperparameters' : dict
            Information about tau selection and CV:
            - 'tau_used' : float, selected tau after CV
            - 'cv_performed' : bool, whether cross-validation was executed
            - 'n_splits' : int or None, number of CV folds if CV performed

    Notes
    -----
    - Both donor matrix X and outcome y are standardized (zero mean, unit variance)
      before fitting, and predictions are rescaled back to the original scale.
    - Donor weights are transformed back to original scale to sum to 1.
    - Cross-validation uses `RelaxationCV` with simplex weights and no intercept.
    - ATT/effects calculation is performed via `effects.calculate`.
    """
    # Standardize donors
    scaler_X = StandardScaler().fit(X_pre)
    X_pre_scaled = scaler_X.transform(X_pre)
    X_post_scaled = scaler_X.transform(X_post)

    # Standardize outcomes
    y_mean, y_std = y_pre.mean(), y_pre.std()
    y_pre_scaled = (y_pre - y_mean) / y_std

    # Fit relaxation SCM
    model = RelaxationCV(
        tau=tau,
        n_taus=n_taus,
        n_splits=n_splits,
        nonneg=True,
        solver=solver,
        relaxation_type=relaxation_type
    )
    model.fit(X_pre_scaled, y_pre_scaled)

    # Predictions
    y_pre_pred = model.predict(X_pre_scaled) * y_std + y_mean
    y_post_pred = model.predict(X_post_scaled) * y_std + y_mean

    # Transform weights back to original scale
    weights_scaled = model.coef_
    weights_orig = (weights_scaled / scaler_X.scale_) / np.sum(weights_scaled / scaler_X.scale_)

    donor_weights = {state: (0 if w < 0.001 else round(w, 3))
                     for state, w in zip(donor_names, weights_orig)}

    # ATT / Fit diagnostics
    attdict, fitdict, Vectors = effects.calculate(
        y, np.concatenate([y_pre_pred, y_post_pred]),
        X_pre.shape[0], X_post.shape[0]
    )

    return {
        "donor_weights": donor_weights,
        "predictions": np.concatenate([y_pre_pred, y_post_pred]),
        "Results": {
            "Effects": attdict,
            "Fit": fitdict,
            "Vectors": Vectors
        },
        "Model": f"Relaxed SCM ({relaxation_type})",
        "hyperparameters": {
            "tau_used": model.tau_,
            "cv_performed": getattr(model, "cv_performed_", False),
            "n_splits": n_splits if getattr(model, "cv_performed_", False) else None,
            "relaxation_type": relaxation_type
        }
    }





def generate_lambda_seq2(Y1, Y0, alpha, epsilon=1e-4, num=30):
    sY0 = (Y0 - np.mean(Y0, axis=0)) / np.std(Y0, axis=0, ddof=0)
    alpha = max(alpha, 0.01)  # avoid div by zero
    lam_max = np.max(np.abs(sY0.T @ Y1)) / (Y0.shape[0] * alpha)
    lam_min = lam_max * epsilon
    lam_max = min(lam_max, 20)
    lam_min = min(lam_min, 1e-4)
    lam_seq = np.exp(np.linspace(np.log(lam_max), np.log(lam_min), num))
    return lam_seq


class ElasticNetCV(BaseEstimator, RegressorMixin):
    """
    Cross-validated Elastic Net wrapper for SCM using Opt2.SCopt.

    Parameters
    ----------
    alpha : float or array-like, optional
        Mixing parameter(s) between L1 and second norm.
    lam : float or array-like, optional
        Regularization strength(s). If None, lambda sequence is generated automatically.
    second_norm : {"l2", "linf"}, default "l2"
        Second norm used in Elastic Net.
    n_splits : int, default=5
        Number of folds for time-series CV.
    solver : str, default="CLARABEL"
        CVXPY solver used in SCopt.
    """

    def __init__(self, *,
                 alpha: Optional[float | list[float]] = 0.5,
                 lam: Optional[float | list[float]] = None,
                 second_norm: Literal["l2", "linf"] = "l2",
                 n_splits: int = 5,
                 solver: str = "CLARABEL"):
        self.alpha = alpha
        self.lam = lam
        self.second_norm = second_norm
        self.n_splits = n_splits
        self.solver = solver

    def _process_grid(self, X, y):
        """Prepare alpha and lambda grids for CV."""
        self.alphas_ = np.atleast_1d(self.alpha) if self.alpha is not None else np.linspace(0.01, 0.99, 10)
        if self.lam is not None:
            self.lams_ = np.atleast_1d(self.lam)
        else:
            # Generate lambda sequence using the first alpha
            self.lams_ = generate_lambda_seq2(y, X, self.alphas_[0])

    def _solve_enet(self, X, y, lam, alpha) -> np.ndarray:
        """Fit Elastic Net using SCopt for given lam and alpha."""
        try:
            res = Opt2.SCopt(
                y=y,
                X=X,
                T0=X.shape[0],
                fit_intercept=True,
                constraint_type="simplex",
                objective_type="penalized",
                lam=lam,
                alpha=alpha,
                second_norm=self.second_norm,
                solver=self.solver,
            )
            w = res["weights"]["w"]
            if hasattr(w, "value") and w.value is not None:
                w = w.value.flatten()
            elif isinstance(w, np.ndarray):
                w = w.flatten()
            else:
                w = np.zeros(X.shape[1], dtype=np.float64)
            return w
        except Exception as e:
            print(f"SCopt failed for lam={lam}, alpha={alpha}: {e}")
            return np.zeros(X.shape[1], dtype=np.float64)

    def _fit_fold(self, X, y, train_idx, test_idx, lam, alpha):
        """Compute fold MSE for one lam/alpha combination."""
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        w = self._solve_enet(X_train, y_train, lam, alpha)
        y_pred = X_test @ w
        return np.mean((y_test - y_pred) ** 2)

    def _cross_validate(self, X, y):
        """Perform time-series CV over lambda and alpha."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        best_mse = np.inf
        best_params = {}
        for alpha in self.alphas_:
            for lam in self.lams_:
                fold_mses = []
                for train_idx, test_idx in tscv.split(X):
                    mse = self._fit_fold(X, y, train_idx, test_idx, lam, alpha)
                    fold_mses.append(mse)
                mean_mse = np.mean(fold_mses)
                if mean_mse < best_mse:
                    best_mse = mean_mse
                    best_params = {"lam": lam, "alpha": alpha}
        self.lam_ = best_params["lam"]
        self.alpha_ = best_params["alpha"]
        self.cv_performed_ = True

    def fit(self, X, y):
        """Fit Elastic Net CV model."""
        X, y = X.astype(np.float64), y.astype(np.float64)
        self._process_grid(X, y)
        self._cross_validate(X, y)
        self.coef_ = self._solve_enet(X, y, self.lam_, self.alpha_)
        return self

    def predict(self, X):
        """Predict using fitted weights."""
        X = X.astype(np.float64)
        return X @ self.coef_




def fit_en_scm(
    X_pre: np.ndarray,
    y_pre: np.ndarray,
    X_post: np.ndarray,
    donor_names: list = None,
    y: np.ndarray = None,
    alpha: float | list[float] | None = 0.5,
    lam: float | list[float] | None = None,
    n_splits: int = 5,
    second_norm: str = "l2",
    solver: str = "CLARABEL",
) -> dict:
    """
    Fit an Elastic Net Synthetic Control Method (SCM) with optional
    cross-validation over lambda and alpha, returning donor weights,
    predictions, and ATT results.

    Parameters
    ----------
    X_pre : np.ndarray, shape (T0, J)
        Pre-treatment donor matrix.
    y_pre : np.ndarray, shape (T0,)
        Pre-treatment outcome vector for treated unit.
    X_post : np.ndarray, shape (T1, J)
        Post-treatment donor matrix.
    donor_names : list of str, optional
        Names of donor units corresponding to columns of X_pre/X_post.
    y : np.ndarray, shape (T0+T1,), optional
        Full outcome vector of treated unit. Used for ATT computation.
    alpha : float or list of floats, optional
        Mixing parameter(s) for L1/L2/Linf.
    lam : float or list of floats, optional
        Regularization strength(s) for Elastic Net.
    n_splits : int, default=5
        Number of folds for time-series cross-validation.
    second_norm : {"l2", "linf"}, default "l2"
        Second norm used in Elastic Net.
    solver : str, default "CLARABEL"
        CVXPY solver.

    Returns
    -------
    result : dict
        Dictionary containing donor weights, predictions, ATT/fit results, model info, and CV hyperparameters.
    """
    # Standardize donors
    scaler_X = StandardScaler().fit(X_pre)
    X_pre_scaled = scaler_X.transform(X_pre)
    X_post_scaled = scaler_X.transform(X_post)

    # Standardize outcomes
    y_mean, y_std = y_pre.mean(), y_pre.std()
    y_pre_scaled = (y_pre - y_mean) / y_std

    # Fit Elastic Net SCM with CV
    model = ElasticNetCV(
        alpha=alpha,
        lam=lam,
        second_norm=second_norm,
        n_splits=n_splits,
        solver=solver
    )
    model.fit(X_pre_scaled, y_pre_scaled)

    # Predictions
    y_pre_pred = model.predict(X_pre_scaled) * y_std + y_mean
    y_post_pred = model.predict(X_post_scaled) * y_std + y_mean

    # Transform weights back to original scale
    weights_scaled = model.coef_
    weights_orig = (weights_scaled / scaler_X.scale_) / np.sum(weights_scaled / scaler_X.scale_)

    donor_weights = {state: (0 if w < 0.001 else round(w, 3))
                     for state, w in zip(donor_names, weights_orig)}

    # ATT / Fit diagnostics
    attdict, fitdict, Vectors = effects.calculate(
        y, np.concatenate([y_pre_pred, y_post_pred]),
        X_pre.shape[0], X_post.shape[0]
    )

    return {
        "donor_weights": donor_weights,
        "predictions": np.concatenate([y_pre_pred, y_post_pred]),
        "Results": {
            "Effects": attdict,
            "Fit": fitdict,
            "Vectors": Vectors
        },
        "Model": f"Elastic Net SCM ({second_norm})",
        "hyperparameters": {
            "alpha_used": model.alpha_,
            "lam_used": model.lam_,
            "cv_performed": getattr(model, "cv_performed_", False),
            "n_splits": n_splits if getattr(model, "cv_performed_", False) else None
        }
    }
