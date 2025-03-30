import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib import rc_context

def plot_estimates(
    df,
    time,
    unitid,
    outcome,
    treatmentname,
    treated_unit_name,
    y,
    cf_list,
    method,
    treatedcolor,
    counterfactualcolors,
    counterfactual_names=None,
    save=False,
):
    """
    Plots observed and multiple counterfactual estimates with markers only at x-tick positions.

    Parameters:
    - df: DataFrame containing the dataset.
    - time: Column name for time variable.
    - unitid: Column name for unit identifier.
    - outcome: Outcome variable name (y-axis label).
    - treatmentname: Column name for treatment indicator.
    - treated_unit_name: The treated unit identifier.
    - y: Observed outcome for the treated unit.
    - cf_list: List of counterfactual predictions (one series per method).
    - method: Method name for title or file saving.
    - treatedcolor: Color for the observed line.
    - counterfactualcolors: List of colors for each counterfactual.
    - counterfactual_names: List of custom names for counterfactuals (optional).
    - save: Boolean or dictionary for saving the plot. Defaults to False.
    """

    ubertheme = {
        "figure.facecolor": "white",
        "figure.figsize": (11, 5),
        "figure.dpi": 100,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "lines.linewidth": 1.2,
        "patch.facecolor": "#0072B2",  # Blue shade for patches
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
        # Identify the intervention point
        intervention_point = df.loc[df[treatmentname] == 1, time].min()
        time_axis = df[df[unitid] == treated_unit_name][time].values

        plt.axvline(
            x=intervention_point,
            color='grey',
            linestyle='--',  # Dashed line
            linewidth=1.2,  # Line thickness
            label=f"{treatmentname}" # Optional label for the reference line
        )

        plt.draw()
        xticks = plt.gca().get_xticks()

        # Handle datetime x-axis
        if np.issubdtype(time_axis.dtype, np.datetime64):
            xticks = pd.to_datetime(xticks, unit="D")
            time_axis = pd.to_datetime(time_axis)

        # Find the closest x values to tick positions
        valid_x = np.intersect1d(time_axis, xticks)
        valid_y = [y[np.where(time_axis == vx)[0][0]] for vx in valid_x]

        # Plot each counterfactual
        for idx, cf in enumerate(cf_list):
            label = (
                counterfactual_names[idx] if counterfactual_names else f"Artificial {idx + 1}"
            )
            color = counterfactualcolors[idx % len(counterfactualcolors)]
            plt.plot(time_axis, cf, label=label, linestyle="-", color=color)

        # Plot observed outcomes
        plt.plot(time_axis, y, label=f"{treated_unit_name}", color=treatedcolor)

        # Add labels, title, legend, and grid
        plt.xlabel(time)
        plt.title(f"Causal Impact on {outcome}", loc="left")
        plt.legend()

        # Save or display the plot
        if save:
            if isinstance(save, dict):
                filename = save.get("filename", f"{method}_{treated_unit_name}")
                extension = save.get("extension", "png")
                directory = save.get("directory", os.getcwd())
            else:
                filename = f"{method}_{treated_unit_name}"
                extension = "png"
                directory = os.getcwd()

            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, f"{filename}.{extension}")
            plt.savefig(filepath)
            print(f"Plot saved to: {filepath}")

        if not save or (isinstance(save, dict) and save.get("display", True)):
            plt.show()
            plt.close()





class effects:
    @staticmethod
    def calculate(y, y_counterfactual, t1, t2):
        ATT = np.mean(y[t1:] - y_counterfactual[t1:])
        ATT_percentage = 100 * ATT / np.mean(y_counterfactual[t1:])
        u1 = y[:t1] - y_counterfactual[:t1]
        omega_1_hat = (t2 / t1) * np.mean(u1**2)
        omega_2_hat = np.mean(u1**2)
        std_omega_hat = np.sqrt(omega_1_hat + omega_2_hat)
        ATT_std = np.sqrt(t2) * ATT / std_omega_hat

        # Calculate Total Treatment Effect (TTE)
        TTE = np.sum(y[t1:] - y_counterfactual[t1:])

        # Calculate R-squared for the fit
        R2 = 1 - (np.mean((y[:t1] - y_counterfactual[:t1]) ** 2)) / (
            np.mean((y[:t1] - np.mean(y[:t1])) ** 2)
        )

        # Fit dictionary
        Fit_dict = {
            "T0 RMSE": round(
                np.sqrt(np.mean((y[:t1] - y_counterfactual[:t1]) ** 2)), 3
            ),
            "T1 RMSE": round(np.std(y[t1:] - y_counterfactual[t1:]), 3),
            "R-Squared": round(R2, 3),
            "Pre-Periods": t1,
            "Post-Periods": len(y[t1:]),
        }

        # Effects dictionary
        Effects_dict = {
            "ATT": round(ATT, 3),
            "Percent ATT": round(ATT_percentage, 3),
            "SATT": round(ATT_std, 3),
            "TTE": round(TTE, 3),  # Add TTE to the returned dictionary
        }

        gap = y - y_counterfactual

        second_column = np.arange(gap.shape[0]) - t1 + 1

        gap_matrix = np.column_stack((gap, second_column))

        # Vectors subdictionary
        Vector_dict = {
            "Observed Unit": np.round(y.reshape(-1, 1), 3),
            "Counterfactual": np.round(y_counterfactual.reshape(-1, 1), 3),
            "Gap": np.round(gap_matrix, 3),
        }

        return Effects_dict, Fit_dict, Vector_dict
