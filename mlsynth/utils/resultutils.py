import numpy as np
import matplotlib.pyplot as plt
import os


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
    customization=None,
    save=False,
):
    """
    Plots observed and multiple counterfactual estimates with customizable options.

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
    - customization: Dictionary for customizing plot elements. Keys can include:
        - 'reference_line_color': Color for the reference line.
        - 'reference_line_style': Line style for the reference line.
        - 'reference_line_width': Width of the reference line.
        - 'treated_line_style': Line style for the treated line.
        - 'treated_line_width': Width of the treated line.
        - 'plot_title': Custom title for the plot.
        - 'xlabel': Custom x-axis label.
        - 'ylabel': Custom y-axis label.
        - 'grid': Boolean to toggle grid.
    - save: Boolean or dictionary for saving the plot. Defaults to False.
            If a dictionary, keys can include:
                - 'filename': Custom file name (without extension).
                - 'extension': File format (e.g., 'png', 'pdf').
                - 'directory': Directory to save the plot.
    """
    import os
    import matplotlib.pyplot as plt

    # Extract customization options or set defaults
    customization = customization or {}
    ref_color = customization.get("reference_line_color", "#7DF9FF")
    ref_style = customization.get("reference_line_style", "-")
    ref_width = customization.get("reference_line_width", 2)
    treated_style = customization.get("treated_line_style", "-")
    treated_width = customization.get("treated_line_width", 3)
    plot_title = customization.get("plot_title", "Observed vs. Prediction")
    xlabel = customization.get("xlabel", time)
    ylabel = customization.get("ylabel", outcome)
    grid = customization.get("grid", True)

    # Identify the intervention point
    intervention_point = df.loc[df[treatmentname] == 1, time].min()
    time_axis = df[df[unitid] == treated_unit_name][time].values

    # Plot intervention point
    plt.axvline(
        x=intervention_point,
        color=ref_color,
        linestyle=ref_style,
        linewidth=ref_width,
        label=f"{treatmentname}, {intervention_point}",
    )

    # Plot observed outcomes
    plt.plot(
        time_axis,
        y,
        label=f"{treated_unit_name}",
        linewidth=treated_width,
        linestyle=treated_style,
        color=customization.get("treated_color", "black"),
    )

    # Plot each counterfactual
    for idx, cf in enumerate(cf_list):
        label = (
            customization.get("counterfactual_names", [f"Artificial {idx + 1}"])[idx]
            if "counterfactual_names" in customization
            else f"Artificial {idx + 1}"
        )
        color = customization.get("counterfactual_colors", ["green", "red"])[
            idx % len(customization.get("counterfactual_colors", ["green", "red"]))
        ]
        plt.plot(
            time_axis,
            cf,
            label=label,
            color=color,
            linestyle=customization.get("counterfactual_line_styles", ["--", ":"])[
                idx % len(customization.get("counterfactual_line_styles", ["--", ":"]))
            ],
            linewidth=2,
        )

    # Add labels, title, legend, and grid
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.legend()
    if grid:
        plt.grid(True)

    # Save or display the plot
    if save:
        if isinstance(save, dict):
            # Extract options from the dictionary
            filename = save.get("filename", f"{method}_{treated_unit_name}")
            extension = save.get("extension", "png")
            directory = save.get("directory", os.getcwd())
        else:
            # Use default filename and extension
            filename = f"{method}_{treated_unit_name}"
            extension = "png"
            directory = os.getcwd()

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Construct the full file path
        filepath = os.path.join(directory, f"{filename}.{extension}")
        plt.savefig(filepath)
        print(f"Plot saved to: {filepath}")

    if not save or (
        isinstance(save, dict) and "display" in save and save["display"]
    ):
        plt.show()

    # Clear the plot to avoid overlap in subsequent plots
    #plt.clf()


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
