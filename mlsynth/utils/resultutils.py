import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

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
    # Identify the intervention point
    intervention_point = df.loc[df[treatmentname] == 1, time].min()
    time_axis = df[df[unitid] == treated_unit_name][time].values

    # Plot intervention point
    plt.axvline(
        x=intervention_point,
        color="black",
        linestyle="-",
        linewidth=2,
        label=f"{treatmentname}, {intervention_point}",
    )

    # Plot observed outcomes without markers
    plt.plot(time_axis, y, label=f"{treated_unit_name}", linewidth=3, color=treatedcolor)

    # Draw plot to determine tick positions
    plt.draw()
    xticks = plt.gca().get_xticks()

    # Handle datetime x-axis
    if np.issubdtype(time_axis.dtype, np.datetime64):
        xticks = pd.to_datetime(xticks, unit="D")  # Convert float ticks to datetime
        time_axis = pd.to_datetime(time_axis)  # Ensure time_axis is also datetime

    # Find the closest x values to tick positions
    valid_x = np.intersect1d(time_axis, xticks)
    valid_y = [y[np.where(time_axis == vx)[0][0]] for vx in valid_x]

    # Plot markers only at tick positions for observed data (same color as line)
    plt.scatter(valid_x, valid_y, color=treatedcolor, edgecolor="black", zorder=3)

    # Plot each counterfactual without markers
    for idx, cf in enumerate(cf_list):
        label = (
            counterfactual_names[idx] if counterfactual_names else f"Artificial {idx + 1}"
        )
        color = counterfactualcolors[idx % len(counterfactualcolors)]
        plt.plot(time_axis, cf, label=label, color=color, linestyle="-", linewidth=2)

        # Find valid y values for counterfactuals at tick positions
        valid_cf_y = [cf[np.where(time_axis == vx)[0][0]] for vx in valid_x]

        # Plot markers only at tick positions for counterfactuals (same color as line)
        plt.scatter(valid_x, valid_cf_y, color=color, edgecolor="black", zorder=3)

    # Add labels, title, legend, and grid
    plt.xlabel(time)
    plt.ylabel(outcome)
    plt.title("Observed vs. Prediction")
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

    # Clear the figure only once at the end
    plt.clf()




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
