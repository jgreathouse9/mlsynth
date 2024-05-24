import numpy as np
import matplotlib.pyplot as plt

def plot_estimates(df, time, unitid, outcome, treatmentname, treated_unit_name, y, cf, method, treatedcolor, counterfactualcolor,  rmse=None, att=None, save=False):
    intervention_point = df.loc[df[treatmentname] == 1, time].min()
    time_axis = df[df[unitid] == treated_unit_name][time].values

    plt.axvline(x=intervention_point, color="black", linestyle="-", linewidth=2.5,
                label=treatmentname + ", " + str(intervention_point))
    plt.plot(time_axis, y, label=f'Observed {treated_unit_name}', linewidth=3,
             color=treatedcolor, marker='o')
    plt.plot(time_axis, cf, label=f'Synthetic {treated_unit_name}', color=counterfactualcolor,
             linestyle="--", linewidth=1, marker='o', markersize=6)

    plt.xlabel(time)
    plt.ylabel(outcome)
    plt.title(fr'{method}, $\bar{{\tau}}$ = {att:.3f}, RMSE = {rmse:.3f}')
    plt.legend()  # Adjust the location of the legend as needed
    plt.grid(True)

    if save:
        plt.savefig(f"{method}_{treated_unit_name}.png")
    else:
        plt.show()


class effects:
    @staticmethod
    def calculate(y, y_counterfactual, t1, t2):
        ATT = np.mean(y[t1:] - y_counterfactual[t1:])
        ATT_percentage = 100 * ATT / np.mean(y_counterfactual[t1:])
        u1 = y[:t1] - y_counterfactual[:t1]
        omega_1_hat = (t2 / t1) * np.mean(u1 ** 2)
        omega_2_hat = np.mean(u1 ** 2)
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
            "T0 RMSE": round(np.sqrt(np.mean((y[:t1] - y_counterfactual[:t1]) ** 2)), 3),
            "T1 RMSE": round(np.std(y[t1:] - y_counterfactual[t1:]), 3),
            "R-Squared": round(R2, 3),
            "Pre-Periods": t1,
            "Post-Periods": len(y[t1:])
        }

        # Effects dictionary
        Effects_dict = {
            "ATT": round(ATT, 3),
            "Percent ATT": round(ATT_percentage, 3),
            "SATT": round(ATT_std, 3),
            "TTE": round(TTE, 3)  # Add TTE to the returned dictionary
        }

        gap = y - y_counterfactual

        second_column = np.arange(gap.shape[0]) - t1 + 1

        gap_matrix = np.column_stack((gap, second_column))

        # Vectors subdictionary
        Vector_dict = {
            "Observed Unit": np.round(y.reshape(-1, 1), 3),
            "Counterfactual": np.round(y_counterfactual.reshape(-1, 1), 3),
            "Gap": np.round(gap_matrix, 3)
        }

        return Effects_dict, Fit_dict, Vector_dict