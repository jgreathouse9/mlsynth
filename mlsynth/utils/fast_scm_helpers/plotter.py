from typing import List, Optional, Dict, Any, Union
import os
import numpy as np
import matplotlib.pyplot as plt


def lexplot(
        results,
        save_plot_config: Union[bool, Dict[str, str]] = False,
) -> None:
    """
    Plotting utility for LEXSCM results.
    Merged Version: Metadata-safe (V1) + Datetime-robust (V2).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # =========================================================
    # 1. Metadata and X-Axis Setup (Robust Handling)
    # =========================================================
    # Use time-specific suffixes to prevent covariate row pollution
    n_fit = results.panel.time.n_fit_time
    n_blank = results.panel.time.n_blank_time
    n_post = results.panel.time.n_post

    # Crucial: Convert labels to an array for value-based vertical lines
    x_axis_labels = np.asarray(results.panel.time.index.labels)
    
    synthetic_treated = results.search.winner.predictions.synthetic_treated
    synthetic_control = results.search.winner.predictions.synthetic_control
    population_mean = results.panel.population_mean

    outcome_name = results.panel.outcome
    
    # Boundary indices for slicing
    est_end = n_fit
    blank_end = n_fit + n_blank

    # =========================================================
    # 2. Performance Metrics
    # =========================================================
    def rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    est_slice = slice(0, est_end)
    blank_slice = slice(est_end, blank_end)

    rmse_est_pop = rmse(population_mean[est_slice], synthetic_treated[est_slice])
    rmse_blank_pop = rmse(population_mean[blank_slice], synthetic_treated[blank_slice])
    rmse_est_ctrl = rmse(synthetic_control[est_slice], synthetic_treated[est_slice])

    # =========================================================
    # 3. Plot Construction
    # =========================================================
    fig, ax = plt.subplots(figsize=(11, 5))

    # --- Vertical Lines (Fixed for Datetime/Value-based X) ---
    # We reference x_axis_labels[index] instead of the raw index
    if est_end < len(x_axis_labels):
        ax.axvline(
            x=x_axis_labels[est_end],
            linestyle="--",
            color="gray",
            linewidth=1.2,
            alpha=0.8,
            label="End of estimation",
        )

    if n_post > 0 and blank_end < len(x_axis_labels):
        ax.axvline(
            x=x_axis_labels[blank_end],
            linestyle="--",
            color="black",
            linewidth=1.5,
            label="Treatment start",
        )

    # --- Series Plotting ---
    # In-sample / Pre-period (Solid)
    ax.plot(x_axis_labels[:blank_end], population_mean[:blank_end],
            color="black", linewidth=2.5, label="Population Mean")

    ax.plot(x_axis_labels[:blank_end], synthetic_treated[:blank_end],
            color="tab:blue", linewidth=2, label="Synthetic Treated")

    ax.plot(x_axis_labels[:blank_end], synthetic_control[:blank_end],
            color="tab:red", linewidth=2, label="Synthetic Control")

    # Out-of-sample / Post-period (Dashed)
    if n_post > 0:
        ax.plot(x_axis_labels[blank_end:], population_mean[blank_end:],
                color="black", linewidth=2.5, alpha=0.7)

        ax.plot(x_axis_labels[blank_end:], synthetic_treated[blank_end:],
                color="tab:blue", linewidth=1.5, linestyle="--", alpha=0.8)

        ax.plot(x_axis_labels[blank_end:], synthetic_control[blank_end:],
                color="tab:red", linewidth=1.5, linestyle="--", alpha=0.8)

    # =========================================================
    # 4. Annotation and Titles
    # =========================================================
    best = results.search.winner
    tuplename = best.identification.tuple_id
    m_val = len(best.treated_weight_dict)

    title = (
        f"{outcome_name} | {tuplename} | m={m_val}\n"
        f"RMSE(Fit): {rmse_est_pop:.3f} | RMSE(Blank): {rmse_blank_pop:.3f} | "
        f"RMSE(T vs C): {rmse_est_ctrl:.3f}"
    )

    if n_post > 0 and hasattr(best, "inference"):
        inf = best.inference
        # Use getattr for safer access
        ate = getattr(inf, "ate", None)
        pval = getattr(inf, "p_value", None)
        ci_l = getattr(inf, "ci_lower", None)
        ci_u = getattr(inf, "ci_upper", None)

        if ate is not None and pval is not None:
            title += f"\nATT: {ate:.3f} | p-val: {pval:.3g}"
            if ci_l is not None:
                title += f" | 95% CI: [{ci_l:.2f}, {ci_u:.2f}]"

    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel(outcome_name)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(alpha=0.25)
    
    # Auto-rotates dates if they are crowded
    fig.autofmt_xdate()
    plt.tight_layout()

    # =========================================================
    # 5. Export Logic
    # =========================================================
    if save_plot_config:
        # (Export logic remains identical to your Version 1)
        if isinstance(save_plot_config, dict):
            filename = save_plot_config.get("filename", f"{outcome_name}_lexplot")
            extension = save_plot_config.get("extension", "png")
            directory = save_plot_config.get("directory", ".")
            display = save_plot_config.get("display", True)
        else:
            filename = f"{outcome_name}_lexplot"; extension = "png"; directory = "."; display = True

        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{filename}.{extension}")
        plt.savefig(filepath, bbox_inches="tight", dpi=150)
        if display: plt.show()
    else:
        plt.show()

    plt.close()
