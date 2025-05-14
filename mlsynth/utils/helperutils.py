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


def sc_diagplot(config):
    from mlsynth.mlsynth import dataprep  # Moved here to avoid circular import

    """
    Diagnostic plot showing treated unit(s) vs donor units.

    Parameters:
    - config: dictionary with required keys:
        - df, unitid, time, outcome, treat
        - cohort (optional): int, index of treated cohort
        - save (optional): bool or dict with keys:
            - filename, extension, directory, display
    """

    ubertheme = {
        "figure.facecolor": "white",
        "figure.figsize": (6, 5),
        "figure.dpi": 100,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "lines.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.size": 14,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.grid": True,
        "axes.facecolor": "white",
        "axes.linewidth": 0.1,
        "axes.titlesize": "large",
        "axes.titleweight": "bold",
        "axes.labelsize": "medium",
        "axes.labelweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.titlepad": 25,
        "axes.labelpad": 20,
        "grid.alpha": 0.1,
        "grid.linewidth": 0.5,
        "grid.color": "#000000",
        "legend.framealpha": 0.5,
        "legend.fancybox": True,
        "legend.borderpad": 0.5,
        "legend.loc": "best",
        "legend.fontsize": "small",
    }

    with rc_context(rc=ubertheme):
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
                raise ValueError(
                    f"Multiple treated cohorts found. Please specify 'cohort' index in config (0 to {len(cohort_keys) - 1})."
                )
            if not (0 <= cohort < len(cohort_keys)):
                raise IndexError(f"Invalid cohort index: {cohort}. Must be between 0 and {len(cohort_keys) - 1}.")

            cohort_key = cohort_keys[cohort]
            d = data["cohorts"][cohort_key]
            treated = d["y"].mean(axis=1)
            treated_units = d["treated_units"]
            treated_unit_name = "_".join(treated_units)
            treatment_label = f"Cohort {cohort} (treated at {cohort_key})"
            outcome = config["outcome"]

            title_label = f"Cohort {cohort} ({', '.join(treated_units)}) versus Donors:  {outcome} Trends"
        else:
            d = data
            treated = d["y"]
            treated_unit_name = data["treated_unit_name"]
            treatment_label = "Treated Unit"
            outcome = config["outcome"]
            title_label = f"{treated_unit_name} versus Donors: {outcome} Trends"

        donor_matrix = d["donor_matrix"]
        donor_mean = donor_matrix.mean(axis=1)
        time_index = data["Ywide"].index

        plt.figure(figsize=(10, 6))

        for i in range(donor_matrix.shape[1]):
            plt.plot(time_index, donor_matrix[:, i], color="grey", linewidth=0.8, alpha=0.3, zorder=1)

        plt.plot(time_index, donor_mean, label="Donor Mean", color="blue", linewidth=2.0, zorder=2)
        plt.plot(time_index, treated, label=treatment_label, color="black", linewidth=2.0, zorder=3)
        plt.axvline(x=time_index[d["pre_periods"]], color="black", linestyle="--", label="Treatment Start")

        plt.xlabel(config["time"])
        plt.ylabel(config["outcome"])
        plt.title(title_label, loc="left")
        plt.legend()
        plt.tight_layout()

        save = config.get("save", False)
        if save:
            import matplotlib
            if isinstance(save, dict):
                filename = save.get("filename", f"sc_diagplot_{treated_unit_name}")
                extension = save.get("extension", "png")
                directory = save.get("directory", os.getcwd())
                display = save.get("display", True)
            else:
                filename = f"sc_diagplot_{treated_unit_name}"
                extension = "png"
                directory = os.getcwd()
                display = True

            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, f"{filename}.{extension}")
            plt.savefig(filepath)
            print(f"Plot saved to: {filepath}")

            if display:
                plt.show()
            else:
                plt.close()
        else:
            plt.show()
            plt.close()
