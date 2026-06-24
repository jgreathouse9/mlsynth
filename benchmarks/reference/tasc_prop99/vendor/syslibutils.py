import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def hsvt(df, rank: int = 2, p: float = 1.0):
    """
    Input:
        df: matrix of interest
        rank: rank of output matrix
        p: the fraction of observed entries. 0<p<=1
    Output:
        thresholded matrix
    """

    u, s, v = np.linalg.svd(df, full_matrices=False)
    s[rank:].fill(0)
    vals = np.dot(u * s, v)
    if p <= 0:
        raise ValueError("p should be greater than 0")
    elif p < 1:
        vals = vals / p
    return pd.DataFrame(vals, index=df.index, columns=df.columns)


def get_energy(s):
    s2 = np.power(s, 2)
    spectrum = np.cumsum(s2) / np.sum(s2)
    return spectrum


def singval_test(test_df, n=50, show=True):
    # test_df: df
    (U, s, Vh) = np.linalg.svd(test_df.values)
    s2 = np.power(s, 2)
    spectrum = np.cumsum(s2) / np.sum(s2)
    cumulative_singval = np.cumsum(s) / np.sum(s)

    if min(test_df.shape) < n:
        n = min(test_df.shape)

    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(np.arange(1, n + 1), spectrum[:n], marker="o")
        # ax1.plot(np.arange(1, n + 1), cumulative_singval[:n], marker="o")
        ax1.grid()
        ax1.set_xticks(np.arange(1, n + 1, 5))
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_xlabel("Rank (Decreasing order of singular values")
        ax1.set_ylabel("Energy")
        ax1.set_title("Cumulative Energy")

        # ax2.plot(np.arange(1, n + 1), s[:n], marker="o")
        # ax2.grid()
        # ax2.set_xticks(np.arange(1, n + 1, 5))
        # ax2.set_xlabel("Rank (Decreasing order of singular values")
        # ax2.set_ylabel("Singval")
        # ax2.set_title("Singular Value Spectrum")
        ax2.plot(np.arange(1, n + 1), cumulative_singval[:n], marker="o")
        ax2.grid()
        ax2.set_xticks(np.arange(1, n + 1, 5))
        ax2.set_xlabel("Rank (Decreasing order of singular values")
        ax2.set_ylabel("Cumulative Singval")
        ax2.set_title("Singular Value Percentage")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the figure
        plt.show()
    return s


def get_approx_rank(s, threshold=0.95):
    cumulative_singval = np.cumsum(s) / np.sum(s)
    index = next(i for i, value in enumerate(cumulative_singval) if value > threshold)
    return index + 1


def unif_sphere(d: int, n: int):
    # returns n samples of d dimensional random vector, uniform over unit sphere
    # output type numpy array n by d

    output = np.zeros(shape=(n, d))
    for i in range(n):
        x = np.random.normal(0, 1, size=d)
        norm = np.sqrt(np.sum(x**2))
        output[i] = x / norm

    return output


def laplace_sample(d: int, n: int, b: float):
    # Lap(b), n-samples in d-dimensional vector
    lap = np.random.laplace(loc=0.0, scale=b, size=n)
    uni = unif_sphere(d, n)
    return np.multiply(lap, uni.T).T


def linear_sample(drop_percent, plac, colname, seed=True):
    """Sample IDs based on linearly decreasing prob distribution, no replacement.

    Assuming a set X of size N is monotonically increasing,
    P(x_i) = (N - i - 1)/T_N, where T_N is the N-th triangular number and i in [1, N]
    Ex: X = [4, 2, 3, 5]; reorder to [2, 3, 4, 5]
    P(X = x_1 = 2) = (4 - 1 - 1) / 10 = .4
    P(X) = [0.4, 0.3, 0.2, 0.1]
    """
    idxs = np.arange(len(plac), 0, -1)
    P = idxs / np.sum(idxs)

    id_cov_list = np.stack((plac.index.tolist(), plac[colname].tolist()), axis=1)
    sorted_ids = id_cov_list[id_cov_list[:, 1].argsort()][:, 0]

    if seed:
        np.random.seed(0)
    return np.random.choice(
        sorted_ids,
        round((100 - drop_percent) / 100 * len(plac)),
        replace=False,
        p=P,
    )


def exp_sample(drop_percent, plac, colname, seed=True):
    """Sample IDs based on exponential prob distribution, no replacement."""
    D = np.exp(-plac[colname])
    D /= np.sum(D)
    if seed:
        np.random.seed(0)
    return np.random.choice(
        D.index.tolist(),
        size=round((100 - drop_percent) / 100 * len(D)),
        replace=False,
        p=D.tolist(),
    )


def mse(true: int, preds: list) -> int:
    """Custom MSE for 1 true, multiple predictions."""
    return np.mean([(true - pred) ** 2 for pred in preds])


def winloss(stats, true_ATE, key="Counterfactual"):
    """Winloss for biased placebo or placebo reconstruction test.

    Params:
        stats: A dict with keys "Naive", "Naive_shift", "Counterfactual" or "Reconstructed"
        true_ATE: True ATE
    """
    c = 0
    c_shift = 0
    for i in range(len(stats["Naive"])):
        if abs(stats["Naive"][i] - true_ATE) >= abs(stats[key][i] - true_ATE):
            c += 1

        if abs(stats["Naive_shift"][i] - true_ATE) >= abs(stats[key][i] - true_ATE):
            c_shift += 1

    print(f"Naive: Wins={c}, Losses={len(stats[key])-c}")
    print(f"Naive_shift: Wins={c_shift}, Losses={len(stats[key])-c_shift}")


def plot_hist(
    data,
    title="",
    xlabel=None,
    gt=None,
    xlim=None,
    ylim=None,
    long=False,
    fname=None,
    bins=10,
):
    #     print("Total:", len(data))
    mean = np.mean(data).item()
    median = np.median(data).item()
    height, bound = np.histogram(data, bins=bins)
    plt.bar(
        x=bound[:-1] + (bound[1] - bound[0]) / 2,
        height=height,
        width=(bound[1] - bound[0]) * 0.9,
        facecolor="blue",
        alpha=0.5,
    )
    #     plt.hist(
    #         data,
    #         bins=10,
    #         range=xlim,
    #         color="blue",
    #         alpha=0.5,
    #         ec='black'
    #     )
    plt.axvline(mean, color="red", label=f"mean = {np.round(mean, 2)}", alpha=0.8)
    plt.axvline(
        median, color="purple", label=f"median = {np.round(median, 2)}", alpha=0.8
    )

    if gt:
        plt.axvline(
            gt, color="green", label=f"ground truth = {np.round(gt, 2)}", alpha=0.8
        )

    if xlim:
        plt.xlim(xlim)
        plt.xticks(list(range(xlim[0], xlim[1], 2)))

    if ylim:
        plt.ylim(ylim)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()

    if fname:
        plt.savefig(fname, dpi=300)
    plt.show()


def plot_bar(
    pred_ATEs: dict,
    true_ATE: float,
    range: list = [0, 20],
    fpath=None,
):
    start, end = range
    x = np.arange(start, end)
    width = 0.2

    plt.figure().set_figwidth(12)
    plt.bar(x - 0.2, pred_ATEs["Naive"][start:end], width, label="Naive")
    plt.bar(
        x, pred_ATEs["Naive_shift"][start:end], width, label="Naive (bias corrected)"
    )
    plt.bar(
        x + 0.2, pred_ATEs["Counterfactual"][start:end], width, label="Counterfactual"
    )
    plt.axhline(true_ATE, label="Ground Truth", color="purple", alpha=0.7)

    plt.xticks(x)
    plt.xlabel("Placebo Subsample")
    plt.ylabel("ATE")
    plt.legend(bbox_to_anchor=(1, 1.4))
    plt.title("Biased Placebo ATEs (first 20 subsamples)")

    if fpath:
        plt.savefig(fpath, bbox_inches="tight", dpi=300)
    plt.show()


def scale_df(
    data: pd.DataFrame, method: str = "minmax", min: float = 0, max: float = 100
) -> tuple:
    """
    Scale a DataFrame using either min-max scaling or standardization followed by min-max scaling.

    Args:
        data (pd.DataFrame): The input DataFrame to be scaled.
        method (str): The scaling method to use ('minmax' or 'standardize'). Default is 'minmax'.
        min (float): The minimum value for min-max scaling. Default is 0.
        max (float): The maximum value for min-max scaling. Default is 100.

    Returns:
        tuple: The scaled DataFrame and the scaler used.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")

    if data.empty:
        raise ValueError("Input DataFrame is empty")

    numerical_cols = data.select_dtypes(exclude="object").columns

    if method == "minmax":
        scaler = MinMaxScaler((min, max))
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        return data, scaler

    elif method == "standardize":
        scaler_standardize = StandardScaler()
        scaler_minmax = MinMaxScaler((min, max))

        def custom_scaler(new_data: pd.DataFrame) -> pd.DataFrame:
            standardized = scaler_standardize.transform(new_data)
            return scaler_minmax.transform(standardized)

        data[numerical_cols] = scaler_standardize.fit_transform(data[numerical_cols])
        data[numerical_cols] = scaler_minmax.fit_transform(data[numerical_cols])
        return data, custom_scaler

    else:
        raise ValueError(f"Unknown method: {method}")


def plot_model_weights(
    weights,
    feature_names=None,
    title="Model Weights",
    figsize=(10, 12),
    color="skyblue",
    edgecolor="navy",
    top_n=None,
):
    """
    Plot the weights of a linear model as a horizontal bar chart, showing only the top N features.

    Parameters:
    - model: A fitted model object with a `coef_` attribute
    - feature_names: List of feature names. If None, will use index numbers
    - title: Title for the plot
    - figsize: Tuple specifying figure size
    - color: Color for the bars
    - edgecolor: Color for the edges of the bars
    - top_n: Number of top features to display. If None, all features are shown

    Returns:
    - fig, ax: The figure and axis objects
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(weights))]

    # Sort weights and feature names together
    sorted_pairs = sorted(
        zip(weights, feature_names), key=lambda x: abs(x[0]), reverse=True
    )

    # Select top N features if specified
    if top_n is not None:
        sorted_pairs = sorted_pairs[:top_n]

    sorted_weights, sorted_features = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(sorted_weights))
    ax.barh(y_pos, sorted_weights, align="center", color=color, edgecolor=edgecolor)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Weight")
    ax.set_title(title)

    # Add weight values at the end of each bar
    for i, v in enumerate(sorted_weights):
        ax.text(v, i, f" {v:.3f}", va="center")

    plt.tight_layout()
    return fig, ax
