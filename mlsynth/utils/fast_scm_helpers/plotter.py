from typing import List, Optional, Dict, Any, Union
    import os
    import numpy as np
    import matplotlib.pyplot as plt

def lexplot(
    results,
    save_plot_config: Union[bool, Dict[str, str]] = False,
) -> None:



    # =========================================================
    # Extract core components
    # =========================================================
    time_info = results.time

    n_fit = time_info.n_fit
    n_blank = time_info.n_blank
    n_post = time_info.n_post

    synthetic_treated = results.best_candidate.predictions.synthetic_treated
    synthetic_control = results.best_candidate.predictions.synthetic_control
    population_mean = results.y_pop_mean_t

    outcome_name = results.outcome

    # Ensure x is array-like (timestamps/datetime supported)
    x = np.asarray(time_info.index.labels)

    best = results.best_candidate

    # =========================================================
    # Helper metrics
    # =========================================================
    def rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    est_end = n_fit
    blank_end = n_fit + n_blank
    post_end = n_fit + n_blank + n_post

    # =========================================================
    # Plot setup
    # =========================================================
    fig, ax = plt.subplots(figsize=(11, 5))

    # =========================================================
    # Vertical lines (FIXED FOR DATETIME X-AXIS)
    # =========================================================

    if est_end < len(x):
        ax.axvline(
            x=x[est_end],
            linestyle="--",
            color="gray",
            linewidth=1.5,
            label="End of estimation",
        )

    if n_post > 0 and blank_end < len(x):
        ax.axvline(
            x=x[blank_end],
            linestyle="--",
            color="black",
            linewidth=1.5,
            label="Treatment start",
        )

    # =========================================================
    # SERIES (split in-sample vs OOS)
    # =========================================================

    # --- in-sample (solid) ---
    ax.plot(
        x[:blank_end],
        population_mean[:blank_end],
        color="black",
        linewidth=3,
        label="Population Mean",
    )

    ax.plot(
        x[:blank_end],
        synthetic_treated[:blank_end],
        color="tab:blue",
        linewidth=2,
        label="Synthetic Treated",
    )

    ax.plot(
        x[:blank_end],
        synthetic_control[:blank_end],
        color="tab:red",
        linewidth=2,
        label="Synthetic Control",
    )

    # --- out-of-sample (dashed + thinner) ---
    if n_post > 0:

        ax.plot(
            x[blank_end:],
            population_mean[blank_end:],
            color="black",
            linewidth=3,
            linestyle="-",
            alpha=0.8,
        )

        ax.plot(
            x[blank_end:],
            synthetic_treated[blank_end:],
            color="tab:blue",
            linewidth=1.5,
            linestyle="--",
            alpha=0.9,
        )

        ax.plot(
            x[blank_end:],
            synthetic_control[blank_end:],
            color="tab:red",
            linewidth=1.5,
            linestyle="--",
            alpha=0.9,
        )

    # =========================================================
    # METRICS FOR TITLE
    # =========================================================

    est_slice = slice(0, est_end)
    blank_slice = slice(est_end, blank_end)

    rmse_est_pop = rmse(
        population_mean[est_slice],
        synthetic_treated[est_slice],
    )

    rmse_blank_pop = rmse(
        population_mean[blank_slice],
        synthetic_treated[blank_slice],
    )

    rmse_est_ctrl = rmse(
        synthetic_control[est_slice],
        synthetic_treated[est_slice],
    )

    tuplename = best.identification.tuple_id
    w = results.bnb_metadata["top_tuples"][0].full_weights
    m = np.sum(w > 0)

    # =========================================================
    # TITLE
    # =========================================================

    title = (
        f"{outcome_name} | {tuplename} | m={m}\n"
        f"RMSE(est, pop): {rmse_est_pop:.3f} | "
        f"RMSE(blank, pop): {rmse_blank_pop:.3f} | "
        f"RMSE(T vs C, est): {rmse_est_ctrl:.3f}"
    )

    if n_post > 0 and hasattr(best, "inference"):

        ci_l = getattr(best.inference, "ci_lower", None)
        ci_u = getattr(best.inference, "ci_upper", None)
        pval = getattr(best.inference, "p_value", None)
        ate = getattr(best.inference, "ate", None)

        if ci_l is not None and ci_u is not None and ate is not None:
            title += (
                f"\nATT={ate:.3f} | "
                f"CI=[{ci_l:.3f}, {ci_u:.3f}] | "
                f"p={pval:.3g}"
            )

    ax.set_title(title, loc="left")

    # =========================================================
    # LABELS
    # =========================================================

    ax.set_xlabel("Time")
    ax.set_ylabel(outcome_name)

    ax.legend()
    ax.grid(alpha=0.3)

    # Optional: cleaner datetime formatting
    fig.autofmt_xdate()

    # =========================================================
    # SAVE / SHOW
    # =========================================================

    if save_plot_config:

        if isinstance(save_plot_config, dict):
            filename = save_plot_config.get(
                "filename",
                f"{outcome_name}_lexplot",
            )

            extension = save_plot_config.get(
                "extension",
                "png",
            )

            directory = save_plot_config.get(
                "directory",
                ".",
            )

            display = save_plot_config.get(
                "display",
                True,
            )

        else:
            filename = f"{outcome_name}_lexplot"
            extension = "png"
            directory = "."
            display = True

        os.makedirs(directory, exist_ok=True)

        filepath = os.path.join(
            directory,
            f"{filename}.{extension}",
        )

        plt.savefig(
            filepath,
            bbox_inches="tight",
            dpi=150,
        )

        if display:
            plt.show()

    else:
        plt.show()

    plt.close()
