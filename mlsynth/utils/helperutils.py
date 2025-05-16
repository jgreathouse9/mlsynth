import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import rc_context
import os

def prenorm(X, target=100):
    """
    Normalize a vector or matrix so that the last row is equal to `target` (default: 100).

    Parameters:
        X (np.ndarray): Vector (1D) or matrix (2D) to normalize.
        target (float): Value to normalize the last row to (default is 100).

    Returns:
        np.ndarray: Normalized array.
    """
    X = np.asarray(X)
    denom = X[-1] if X.ndim == 1 else X[-1, :]

    # Check if any denominator is zero and raise a ZeroDivisionError
    if np.any(denom == 0):
        raise ZeroDivisionError("Division by zero in prenorm.")

    return X / denom * target




def ssdid_w(treated_y, donor_matrix, a, k, eta, pi=None):
    """Solve for synthetic control weights (omega) to match pre-treatment trajectory."""
    T, J = donor_matrix.shape
    if pi is None:
        pi = np.ones(J) / J

    t_max = a + k
    Y_pre_treated = treated_y[:t_max]
    Y_pre_donors = donor_matrix[:t_max, :]

    omega = cp.Variable(J)
    omega_0 = cp.Variable()

    residuals = Y_pre_donors @ omega - omega_0 - Y_pre_treated
    objective = cp.sum_squares(residuals) + eta**2 * cp.sum(cp.multiply(pi, cp.square(omega)))
    constraints = [cp.sum(omega) == 1]

    cp.Problem(cp.Minimize(objective), constraints).solve()
    return omega.value, omega_0.value

def ssdid_lambda(treated_y, donor_matrix, a, k, eta):
    """Solve for time weights (lambda) to balance the donor pre-treatment series."""
    t_max = a + k
    Y_pre_treated = treated_y[:a]
    Y_pre_donors = donor_matrix[:a, :]

    lambda_var = cp.Variable(a)
    lambda_0 = cp.Variable()

    residuals = Y_pre_treated @ lambda_var - lambda_0 - donor_matrix[t_max, :]
    objective = cp.sum_squares(residuals) + eta**2 * cp.sum_squares(lambda_var)
    constraints = [cp.sum(lambda_var) == 1]

    cp.Problem(cp.Minimize(objective), constraints).solve()
    return lambda_var.value, lambda_0.value

def ssdid_est(treated_y, donor_matrix, omega, lambda_vec, a, k):
    """Compute treatment effect estimate at horizon k by comparing gaps."""
    t_post = a + k
    y_post_treated = treated_y[t_post]
    y_post_donors = donor_matrix[t_post, :] @ omega
    gap_post = y_post_treated - y_post_donors

    y_pre_treated = treated_y[:a]
    y_pre_donors = donor_matrix[:a, :] @ omega
    gap_pre = lambda_vec @ (y_pre_treated - y_pre_donors)

    return gap_post - gap_pre


def sc_diagplot(config_list):
    from mlsynth.mlsynth import dataprep  # import inside parent in case of circularity
    if not isinstance(config_list, list):
        raise ValueError("Expected a list of configs")

    n = len(config_list)

    # Use same theme as before
    ubertheme = {
        "figure.facecolor": "white",
        "figure.dpi": 100,
        "font.size": 14,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "grid.alpha": 0.1,
        "grid.linewidth": 0.5,
        "grid.color": "#000000",
        "legend.fontsize": "small",
    }

    with rc_context(rc=ubertheme):
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)

        if n == 1:
            axes = [axes]  # make iterable

        for ax, config in zip(axes, config_list):
            data = dataprep(
                df=config["df"],
                unitid=config["unitid"],
                time=config["time"],
                outcome=config["outcome"],
                treat=config["treat"],
            )

            cohort = config.get("cohort", None)

            if "cohorts" in data:
                cohort_keys = list(data["cohorts"].keys())
                if cohort is None:
                    raise ValueError(f"Multiple cohorts found. Please specify 'cohort'.")
                d = data["cohorts"][cohort_keys[cohort]]
                treated = d["y"].mean(axis=1)
                treated_units = d["treated_units"]
                treated_unit_name = "_".join(treated_units)
                treatment_label = f"Cohort {cohort} (treated at {cohort_keys[cohort]})"
                title_label = f"{treated_unit_name}"
            else:
                d = data
                treated = d["y"]
                treated_unit_name = data["treated_unit_name"]
                treatment_label = "Treated Unit"
                title_label = treated_unit_name

            donor_matrix = d["donor_matrix"]
            donor_mean = donor_matrix.mean(axis=1)
            time_index = data["Ywide"].index

            for i in range(donor_matrix.shape[1]):
                ax.plot(time_index, donor_matrix[:, i], color="gray", linewidth=0.8, alpha=0.3, zorder=1)

            ax.plot(time_index, donor_mean, label="Donor Mean", color="blue", linewidth=2, zorder=2)
            ax.plot(time_index, treated, label=treatment_label, color="black", linewidth=2, zorder=3)
            ax.axvline(x=time_index[d["pre_periods"]], color="black", linestyle="--", linewidth=1)

            ax.set_title(title_label, loc="left")
            ax.set_xlabel(config["time"])
            ax.set_ylabel(config["outcome"])

            ax.legend()

        fig.suptitle("Treated vs Donor Trends", fontsize=16)
        plt.tight_layout()
        plt.show()
