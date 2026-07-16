import numpyro as npy
from typing import Optional, Dict, Any
import os
import pickle
import jax.numpy as jnp
import jax.random as jr
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as npy_dist
import numpy as np
from typing import Callable


def bayesian_scm(target: jnp.ndarray, donors: jnp.ndarray, T0: int, key_seed: int = 123):
    """
    Fit a Bayesian Synthetic Control Model (SCM) for a treated unit using a
    Dirichlet prior over donor weights. This function is used within the
    donor-screening stage described in:

        O’Riordan, Michael and Gilligan-Lee, Ciarán M. (2025).
        "Spillover Detection for Donor Selection in Synthetic Control Models."
        Journal of Causal Inference, 13(1), 20240036.
        https://doi.org/10.1515/jci-2024-0036

    Model specification:
        y_pre ~ Normal(intercept + donors_pre @ w, σ)
        w ~ Dirichlet(α · 1_K),     α ~ Gamma(0.5, 0.5)
        intercept ~ Normal(0, 10)
        σ ~ HalfNormal(1)

    Posterior predictive draws are returned for both pre- and post-treatment
    periods.

    Parameters
    ----------
    target : (T,) array
        Outcome series for the treated unit.
    donors : (T, K) array
        Outcome matrix for K donor units.
    T0 : int
        Pre-treatment cutoff.
    key_seed : int
        Seed for the JAX PRNGKey.

    Returns
    -------
    trace : dict
        Posterior samples of weights, intercept, and noise.
    ppc_pre : dict
        Posterior predictive distribution for the pre-treatment period.
    ppc_post : dict
        Posterior predictive distribution for the post-treatment period.
    model : callable
        NumPyro model function for further predictive sampling.
    """
    key = jr.PRNGKey(key_seed)

    # Split pre- and post-treatment
    target_pre = target[:T0]
    donors_pre = donors[:T0, :]
    donors_post = donors[T0:, :]

    # Define the model
    def model(control_units, treated_unit=None):
        num_units = jnp.shape(control_units)[1]
        concentration = npy.sample("concentration", npy_dist.Gamma(0.5, 0.5)) * jnp.ones(num_units)
        weights = npy.sample("weights", npy_dist.Dirichlet(concentration=concentration))
        intercept = npy.sample("intercept", npy_dist.Normal(0, 10))
        counterfactual = intercept + jnp.matmul(control_units, weights)
        noise = npy.sample("noise", npy_dist.HalfNormal(scale=1.0))
        with npy.handlers.condition(data={"obs": treated_unit}):
            npy.sample("obs", npy_dist.Normal(counterfactual, noise))

    # Run MCMC
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=1000,
        num_samples=1500,
        num_chains=1,
        progress_bar=False
    )
    mcmc.run(key, control_units=donors_pre, treated_unit=target_pre)
    trace = mcmc.get_samples()

    # Posterior predictive on both pre and post
    predictive = Predictive(model, trace)
    ppc_pre = predictive(key, control_units=donors_pre, treated_unit=None)
    ppc_post = predictive(key, control_units=donors_post, treated_unit=None)

    return trace, ppc_pre, ppc_post, model



# -------------------------------
# Private helper: check for saved results
# -------------------------------
def _load_if_exists(save_path):
    """
    Check if a pickle file exists at the specified path and load it if present.

    This helper is used to avoid re-running expensive computations by
    returning previously saved results.

    Parameters
    ----------
    save_path : str
        Path to the pickle file.

    Returns
    -------
    object or None
        The Python object loaded from the pickle file if it exists;
        otherwise, returns None.

    Side Effects
    ------------
    Prints a message indicating that the file exists and is being loaded.
    """
    if os.path.exists(save_path):
        print(f"File '{save_path}' exists. Loading saved results...")
        with open(save_path, "rb") as f:
            return pickle.load(f)
    return None

# -------------------------------
# Private helper: screen a single donor
# -------------------------------
def _screen_single_donor(i, donors, T0, k, threshold, weights, key, donor_names, bayesian_scm_func):
    """
    Determine whether the `i`-th donor should be retained using Bayesian SCM
    forward predictive validation with exponentially weighted scoring.

    The procedure treats donor `i` as a pseudo-treated unit and fits a Bayesian
    Synthetic Control Model (SCM) using all remaining donors as controls. After
    fitting, the model generates `k`-step-ahead posterior predictive draws for the
    pseudo-treated unit. For each post-`T0` period, the method checks whether the
    donor’s actual outcome falls inside the corresponding Bayesian credible interval.

    To emphasize periods closest to the treatment boundary—where predictive accuracy
    is most diagnostic—the method uses an exponentially decaying weighting scheme.
    If `I_t` is an indicator that the actual outcome at horizon `t` lies within
    its predictive interval, and `w_t` are normalized exponential weights, the
    donor’s predictive validity score is

        S_i = sum_t w_t * I_t.

    Early post-treatment periods thus receive more influence, while later periods
    (naturally noisier and less informative) contribute less. A donor is retained
    if and only if this weighted score exceeds the supplied threshold.

    Parameters
    ----------
    i : int
        Index of the donor to evaluate.
    donors : jnp.ndarray, shape (T, D)
        Matrix of donor unit outcomes over time.
    T0 : int
        Last pre-treatment time index.
    k : int
        Number of post-treatment periods used for predictive checking.
    threshold : float
        Minimum required weighted score for retaining the donor.
    weights : jnp.ndarray, shape (k,)
        Exponentially decayed, normalized weights applied to each post-treatment period.
    key : jax.random.PRNGKey
        Base PRNG key for model fitting.
    donor_names : list of str
        Names of the donor units.
    bayesian_scm_func : callable
        Function implementing the Bayesian SCM fit.

    Returns
    -------
    bool
        True if the donor passes the screening test; False otherwise.

    Side Effects
    ------------
    Prints the donor name, its weighted predictive score, and whether it was kept
    or excluded.
    """

    key, subkey = jr.split(key)
    fake_target = donors[:, i]
    fake_donors = jnp.delete(donors, i, axis=1)

    trace, _, _, model = bayesian_scm_func(
        target=fake_target,
        donors=fake_donors,
        T0=T0,
        key_seed=int(jr.randint(subkey, (), 0, 1_000_000))
    )

    predictive = Predictive(model, trace)
    post_pred = predictive(
        subkey,
        control_units=fake_donors[T0:T0+k, :],
        treated_unit=None
    )["obs"]

    actual_vec = fake_target[T0:T0+k]
    lo = jnp.quantile(post_pred, (1 - threshold) / 2, axis=0)
    hi = jnp.quantile(post_pred, 1 - (1 - threshold) / 2, axis=0)
    inside = (actual_vec >= lo) & (actual_vec <= hi)

    weighted_score = jnp.sum(weights * inside)
    valid_i = weighted_score >= threshold

    print(
        f"{donor_names[i]:<20} | "
        f"score={weighted_score:.3f} | "
        f"{'KEPT' if valid_i else 'EXCLUDED'}"
    )
    return valid_i

# -------------------------------
# Private helper: screen all donors
# -------------------------------
def _screen_all_donors(donors, T0, k, threshold, weights, key, donor_names, bayesian_scm_func):
    """
    Perform Bayesian donor screening for all donors in the dataset.

    Iterates over each donor, treating it as a pseudo-treated unit, and
    applies `_screen_single_donor` to determine whether it should be retained
    based on forward predictive validation using a Bayesian SCM. Returns a
    boolean array indicating which donors pass the screening.

    Parameters
    ----------
    donors : jnp.ndarray, shape (T, D)
        Matrix of donor unit outcomes over time.
    T0 : int
        Last pre-treatment time index.
    k : int
        Number of post-treatment periods used for predictive checking.
    threshold : float
        Minimum weighted proportion of post-treatment periods that must fall inside the predictive interval to keep a donor.
    weights : jnp.ndarray, shape (k,)
        Exponentially decayed weights for post-treatment periods.
    key : jax.random.PRNGKey
        Base PRNG key for model fitting.
    donor_names : list of str
        Names of the donor units.
    bayesian_scm_func : callable
        Function implementing the Bayesian SCM fit.

    Returns
    -------
    jnp.ndarray, shape (D,)
        Boolean array where True indicates the donor passed the screening and should be kept,
        and False indicates it was excluded.
    """
    D = donors.shape[1]
    valid = jnp.ones(D, dtype=bool)
    for i in range(D):
        valid_i = _screen_single_donor(i, donors, T0, k, threshold, weights, key, donor_names, bayesian_scm_func)
        valid = valid.at[i].set(valid_i)
    return valid

# -------------------------------
# Private helper: organize screening results
# -------------------------------
def _organize_screening_results(valid, donors, donor_names, y, T0):
    """
    Organize and partition donor screening results.

    Separates donors into those that were retained (kept) and those that were
    excluded based on the screening boolean array. Also partitions the outcome
    series and donor matrices into pre- and post-treatment periods for
    downstream SCM analysis.

    Parameters
    ----------
    valid : array_like, shape (D,)
        Boolean array indicating which donors passed screening (True = kept).
    donors : jnp.ndarray, shape (T, D)
        Full donor outcome matrix.
    donor_names : list of str
        Names of the donor units.
    y : jnp.ndarray, shape (T,)
        Outcome series of the treated unit.
    T0 : int
        Last pre-treatment time index.

    Returns
    -------
    dict
        Dictionary containing:
        - kept_indices : list[int]
            Indices of donors that were retained.
        - excluded_indices : list[int]
            Indices of donors that were excluded.
        - kept_names : list[str]
            Names of retained donors.
        - excluded_names : list[str]
            Names of excluded donors.
        - X_inc : jnp.ndarray
            Matrix of retained donor outcomes.
        - Z_exc : jnp.ndarray or None
            Matrix of excluded donor outcomes, or None if none were excluded.
        - y_pre : jnp.ndarray
            Pre-treatment outcome series for the treated unit.
        - y_post : jnp.ndarray
            Post-treatment outcome series for the treated unit.
        - X_inc_pre : jnp.ndarray
            Pre-treatment outcomes for retained donors.
        - X_inc_post : jnp.ndarray
            Post-treatment outcomes for retained donors.
        - Z_exc_pre : jnp.ndarray or None
            Pre-treatment outcomes for excluded donors, or None if none were excluded.
        - Z_exc_post : jnp.ndarray or None
            Post-treatment outcomes for excluded donors, or None if none were excluded.
    """
    kept_indices = [i for i in range(donors.shape[1]) if bool(valid[i])]
    excluded_indices = [i for i in range(donors.shape[1]) if not bool(valid[i])]

    kept_names = [donor_names[i] for i in kept_indices]
    excluded_names = [donor_names[i] for i in excluded_indices]

    X_inc = donors[:, kept_indices]
    Z_exc = donors[:, excluded_indices] if excluded_indices else None

    y_pre = y[:T0]
    y_post = y[T0:]

    X_inc_pre = X_inc[:T0, :]
    X_inc_post = X_inc[T0:, :]

    if Z_exc is not None:
        Z_exc_pre = Z_exc[:T0, :]
        Z_exc_post = Z_exc[T0:, :]
    else:
        Z_exc_pre = Z_exc_post = None

    return {
        "kept_indices": kept_indices,
        "excluded_indices": excluded_indices,
        "kept_names": kept_names,
        "excluded_names": excluded_names,
        "X_inc": X_inc,
        "Z_exc": Z_exc,
        "y_pre": y_pre,
        "y_post": y_post,
        "X_inc_pre": X_inc_pre,
        "X_inc_post": X_inc_post,
        "Z_exc_pre": Z_exc_pre,
        "Z_exc_post": Z_exc_post,
    }

# -------------------------------
# Private helper: fit final SCM using kept donors
# -------------------------------
def _fit_final_scm(y, X_inc, T0, key, bayesian_scm_func, save=False):
    """
    Fit a Bayesian Synthetic Control Model (SCM) using only the retained donors.

    This function runs the Bayesian SCM on the treated unit using the subset
    of donors that passed the screening process. It computes posterior
    predictive distributions for both pre- and post-treatment periods and
    summarizes the results as means and 95% credible intervals. A "safe" version
    of the results suitable for pickling is also prepared.

    Parameters
    ----------
    y : jnp.ndarray, shape (T,)
        Outcome series of the treated unit.
    X_inc : jnp.ndarray, shape (T, K_inc)
        Outcome matrix for retained (kept) donor units.
    T0 : int
        Last pre-treatment time index.
    key : jax.random.PRNGKey
        Base PRNG key for model fitting.
    bayesian_scm_func : callable
        Function implementing the Bayesian SCM fit, e.g., `bayesian_scm`.
    save : bool, optional
        Whether to prepare a picklable version of the results. Default is False.

    Returns
    -------
    results : dict
        Dictionary containing the full SCM outputs:
        - trace : posterior samples of model parameters
        - ppc_pre : posterior predictive samples for pre-treatment period
        - ppc_post : posterior predictive samples for post-treatment period
        - model : the NumPyro model function
        - in_mean, in_lower, in_upper : mean and 95% credible interval for pre-treatment
        - out_mean, out_lower, out_upper : mean and 95% credible interval for post-treatment

    safe_scm : dict
        Pickle-safe version of the SCM results, containing only arrays and summary statistics:
        - in_mean, in_lower, in_upper : summary of pre-treatment predictions
        - out_mean, out_lower, out_upper : summary of post-treatment predictions
        - in_samples : posterior predictive samples for pre-treatment
        - out_samples : posterior predictive samples for post-treatment

    Notes
    -----
    The "safe_scm" dictionary removes non-pickleable objects such as the model
    function and raw MCMC trace, while retaining all information necessary for
    downstream analyses or saving to disk.
    """
    sc_trace, sc_ppc_pre, sc_ppc_post, sc_model = bayesian_scm_func(
        target=y,
        donors=X_inc,
        T0=T0,
        key_seed=int(jr.randint(key, (), 0, 500_000))
    )

    in_sample = sc_ppc_pre["obs"] if "obs" in sc_ppc_pre else sc_ppc_pre
    out_sample = sc_ppc_post["obs"] if "obs" in sc_ppc_post else sc_ppc_post

    in_mean = jnp.mean(in_sample, axis=0)
    in_lower = jnp.percentile(in_sample, 2.5, axis=0)
    in_upper = jnp.percentile(in_sample, 97.5, axis=0)

    out_mean = jnp.mean(out_sample, axis=0)
    out_lower = jnp.percentile(out_sample, 2.5, axis=0)
    out_upper = jnp.percentile(out_sample, 97.5, axis=0)

    safe_scm = {
        "in_mean": in_mean,
        "in_lower": in_lower,
        "in_upper": in_upper,
        "out_mean": out_mean,
        "out_lower": out_lower,
        "out_upper": out_upper,
        "in_samples": np.array(in_sample),
        "out_samples": np.array(out_sample),
    }

    results = {
        "trace": sc_trace,
        "ppc_pre": sc_ppc_pre,
        "ppc_post": sc_ppc_post,
        "model": sc_model,
        "in_mean": in_mean,
        "in_lower": in_lower,
        "in_upper": in_upper,
        "out_mean": out_mean,
        "out_lower": out_lower,
        "out_upper": out_upper,
    }

    return results, safe_scm

# -------------------------------
# Main function: spotify_donor_screen
# -------------------------------
def spotify_donor_screen(
        donors,
        y,
        T0,
        k=5,
        threshold=0.8,
        lambda_decay=0.5,
        key=jr.PRNGKey(0),
        donor_names=None,
        bayesian_scm_func=bayesian_scm,
        save=False,
        save_path="scscm_screened_results.pkl"
):
    """
    Perform Bayesian donor screening and fit a final synthetic control model (SCM).

    This function implements the Bayesian forward predictive validation procedure
    for donor selection described in O’Riordan & Gilligan-Lee (2025). Each donor
    unit is temporarily treated as the target and checked against the predictive
    posterior distribution generated by a Bayesian SCM fit to the other donors.
    Donors that sufficiently reproduce their own post-treatment trajectory are
    retained. After screening, the SCM is fit using only the retained donors,
    and posterior predictive summaries are returned.

    Parameters
    ----------
    donors : jnp.ndarray, shape (T, D)
        Outcome matrix for D donor units over T time periods.
    y : jnp.ndarray, shape (T,)
        Outcome series of the treated unit.
    T0 : int
        Last pre-treatment period index (0-based).
    k : int, optional
        Number of post-treatment periods used for predictive validation. Default is 5.
    threshold : float, optional
        Minimum exponentially-weighted proportion of post-treatment periods
        that must fall inside the predictive interval for a donor to be retained.
        Default is 0.8.
    lambda_decay : float, optional
        Exponential decay parameter for time weights, giving more weight to
        periods closer to treatment. Default is 0.5.
    key : jax.random.PRNGKey, optional
        PRNG key for all model fits. Default is `jr.PRNGKey(0)`.
    donor_names : list of str, optional
        Optional names for donor units. Defaults to ["Unit0", "Unit1", ...].
    bayesian_scm_func : callable, optional
        Function implementing the Bayesian SCM. Must accept signature
        `(target, donors, T0, key_seed)` and return
        `(trace, ppc_pre, ppc_post, model)`. Default is `bayesian_scm`.
    save : bool, optional
        If True, save a pickle containing the picklable SCM summary to `save_path`.
    save_path : str, optional
        Path for saving pickled results if `save=True`. Default is
        `"scscm_screened_results.pkl"`.

    Returns
    -------
    results : dict
        Dictionary containing:
        - "Clean Donors": retained donor names, indices, and outcome matrix.
        - "Excluded Donors": screened-out donor names, indices, and matrix (or None).
        - "Partitions": pre- and post-treatment partitions of treated and donor outcomes.
        - "SCM": dictionary of the Bayesian SCM fit including posterior samples, predictive
          distributions, and summary statistics (means and 95% credible intervals).

    Notes
    -----
    - If a file exists at `save_path`, the function loads and returns it immediately,
      skipping screening and SCM fitting.
    - The procedure uses exponential weighting of post-treatment periods when
      determining whether a donor is retained.
    - The returned SCM dictionary includes both full posterior samples and summary
      statistics for downstream analysis.

    References
    ----------
    O’Riordan, M., & Gilligan-Lee, C. M. (2025).
        "Spillover detection for donor selection in synthetic control models."
        Journal of Causal Inference, 13(1), 20240036.
        https://doi.org/10.1515/jci-2024-0036
    """

    # Check if saved results exist
    loaded = _load_if_exists(save_path)
    if loaded is not None:
        return loaded

    T, D = donors.shape
    assert T0 + k <= T, "Not enough post-treatment periods for k"
    if donor_names is None:
        donor_names = [f"Unit{i}" for i in range(D)]

    # Exponential-decay weights
    raw_weights = jnp.exp(-lambda_decay * jnp.arange(k))
    weights = raw_weights / raw_weights.sum()

    # -------------------------------
    # Screening
    # -------------------------------
    valid = _screen_all_donors(donors, T0, k, threshold, weights, key, donor_names, bayesian_scm_func)
    screened_results = _organize_screening_results(valid, donors, donor_names, y, T0)

    # -------------------------------
    # Fit final SCM using kept donors
    # -------------------------------
    scm_results, safe_scm = _fit_final_scm(y, screened_results["X_inc"], T0, key, bayesian_scm_func, save=save)

    # -------------------------------
    # Assemble final dictionary
    # -------------------------------
    results = {
        "Clean Donors": {
            "names": screened_results["kept_names"],
            "indices": screened_results["kept_indices"],
            "matrix": screened_results["X_inc"],
        },
        "Excluded Donors": {
            "names": screened_results["excluded_names"],
            "indices": screened_results["excluded_indices"],
            "matrix": screened_results["Z_exc"],
        },
        "Partitions": {
            "y_pre": screened_results["y_pre"],
            "y_post": screened_results["y_post"],
            "X_inc_pre": screened_results["X_inc_pre"],
            "X_inc_post": screened_results["X_inc_post"],
            "Z_exc_pre": screened_results["Z_exc_pre"],
            "Z_exc_post": screened_results["Z_exc_post"],
        },
        "SCM": scm_results
    }

    # -------------------------------
    # Save option
    # -------------------------------
    if save:
        with open(save_path, "wb") as f:
            pickle.dump(safe_scm, f)

    return results


from typing import Tuple

def compute_standardization_stats(pre_array: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation using only pre-treatment data.

    Parameters
    ----------
    pre_array : np.ndarray
        Array of shape (T0, d) or (T0,) containing pre-treatment observations.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    mean : np.ndarray
        Mean computed over the pre-treatment period. Same shape as a row vector.
    std : np.ndarray
        Standard deviation computed over the pre-treatment period.
    """
    mean = pre_array.mean(axis=0, keepdims=True)
    std = pre_array.std(axis=0, keepdims=True) + eps
    return mean, std


def standardize_vector_pre_only(
    y_pre: np.ndarray,
    y_post: np.ndarray,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Z-score standardization for a single treated outcome vector,
    using *only pre-treatment periods* to compute mean and std.

    Parameters
    ----------
    y_pre : np.ndarray
        Vector of shape (T0,) for pre-treatment periods.
    y_post : np.ndarray
        Vector of shape (T_post,) for post-treatment periods.
    eps : float, optional
        Small constant to prevent division by zero.

    Returns
    -------
    y_pre_std : np.ndarray
        Standardized pre-treatment outcome.
    y_post_std : np.ndarray
        Standardized post-treatment outcome.
    y_mean : float
        Pre-treatment mean.
    y_std : float
        Pre-treatment standard deviation.
    """
    y_mean = float(y_pre.mean())
    y_std = float(y_pre.std() + eps)

    y_pre_std = (y_pre - y_mean) / y_std
    y_post_std = (y_post - y_mean) / y_std

    return y_pre_std, y_post_std, y_mean, y_std



def standardize_matrix_pre_only(
    M_pre: np.ndarray,
    M_post: np.ndarray,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Column-wise Z-score standardization of donor or proxy matrices,
    using only pre-treatment periods to compute mean and std.

    Parameters
    ----------
    M_pre : np.ndarray
        Pre-treatment matrix of shape (T0, d).
    M_post : np.ndarray
        Post-treatment matrix of shape (T_post, d).
    eps : float, optional
        Small constant added to std to prevent division by zero.

    Returns
    -------
    M_pre_std : np.ndarray
        Standardized pre-treatment matrix.
    M_post_std : np.ndarray
        Standardized post-treatment matrix.
    M_mean : np.ndarray
        Row vector of column means, shape (1, d).
    M_std : np.ndarray
        Row vector of column stds, shape (1, d).
    """
    M_mean = M_pre.mean(axis=0, keepdims=True)
    M_std = M_pre.std(axis=0, keepdims=True) + eps

    M_pre_std = (M_pre - M_mean) / M_std
    M_post_std = (M_post - M_mean) / M_std

    return M_pre_std, M_post_std, M_mean, M_std


def build_proximal_scm_model(
    y_pre_std: jnp.ndarray,
    y_post: jnp.ndarray,
    X_pre_std: jnp.ndarray,
    X_post_std: jnp.ndarray,
    Z_pre_std: jnp.ndarray,
    J: int,
    K: int,
    y_mean: float,
    y_std: float,
    concentration: float = 1.0,
) -> Callable:
    """
    Construct the NumPyro model for proximal Bayesian SCM, all in standardized space.

    Parameters
    ----------
    y_pre_std : jnp.ndarray
        Standardized treated outcome (pre-treatment). Shape (T0,).
    y_post : jnp.ndarray
        Treated outcome in *original* units (post-treatment). Shape (T1,).
    X_pre_std : jnp.ndarray
        Standardized clean donors (pre). Shape (T0, J).
    X_post_std : jnp.ndarray
        Standardized clean donors (post). Shape (T1, J).
    Z_pre_std : jnp.ndarray
        Standardized proxy donors (pre). Shape (T0, K).
    J : int
        Number of “clean” donors included.
    K : int
        Number of proxy donors excluded.
    y_mean : float
        Pre-treatment mean of y.
    y_std : float
        Pre-treatment std of y.
    concentration : float
        Dirichlet concentration parameter for the synthetic control weights.

    Returns
    -------
    model : Callable
        NumPyro model function ready to be passed to NUTS().
    """

    def model():
        # Outcome model (standardized)
        alpha = npy.sample("alpha", npy_dist.Normal(0, 10))
        beta = npy.sample("beta", npy_dist.Dirichlet(jnp.full(J, concentration)))
        sigma_y = npy.sample("sigma_y", npy_dist.HalfNormal(5.0))

        mu_y = alpha + X_pre_std @ beta
        npy.sample("y_obs", npy_dist.Normal(mu_y, sigma_y), obs=y_pre_std)

        # Proximal debiasing system
        gamma = npy.sample("gamma", npy_dist.Normal(0, 10).expand([J]))
        lam = npy.sample("lam", npy_dist.Normal(0, 10).expand([K, J]))
        sigma_x = npy.sample("sigma_x", npy_dist.HalfNormal(5.0).expand([J]))

        mu_x = gamma + Z_pre_std @ lam
        npy.sample("X_obs", npy_dist.Normal(mu_x, sigma_x), obs=X_pre_std)

        # Counterfactual in standardized space
        cf_pre_std = alpha + X_pre_std @ beta
        cf_post_std = alpha + X_post_std @ beta

        # Rescale back to original outcome units
        cf_pre = cf_pre_std * y_std + y_mean
        cf_post = cf_post_std * y_std + y_mean

        npy.deterministic("cf_pre", cf_pre)
        npy.deterministic("cf_post", cf_post)
        npy.deterministic("cf_full", jnp.concatenate([cf_pre, cf_post]))

        # Treatment effect (original units)
        npy.deterministic("treatment_effect", y_post - cf_post)

    return model


def run_mcmc(
    model: Callable,
    seed: int = 0,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 1,
) -> Dict[str, Any]:
    """
    Run NUTS MCMC for a given NumPyro model.

    Parameters
    ----------
    model : Callable
        NumPyro model function produced by `build_proximal_scm_model`.
    seed : int
        RNG seed.
    num_warmup : int
        Number of warmup iterations.
    num_samples : int
        Number of posterior samples per chain.
    num_chains : int
        Number of parallel MCMC chains.

    Returns
    -------
    samples : dict
        Posterior samples.
    """
    rng_key = jr.PRNGKey(seed)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(rng_key)
    return mcmc.get_samples()



def extract_posterior_results(samples: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """
    Summarize posterior draws from proximal Bayesian SCM.

    Parameters
    ----------
    samples : dict
        Dictionary of posterior draws returned by run_mcmc().

    Returns
    -------
    results : dict
        Counterfactual means, intervals, treatment effects, and raw samples.
    """
    cf_full = samples["cf_full"]
    te = samples["treatment_effect"]

    return {
        "counterfactual_mean": cf_full.mean(axis=0),
        "counterfactual_lower": jnp.percentile(cf_full, 2.5, axis=0),
        "counterfactual_upper": jnp.percentile(cf_full, 97.5, axis=0),

        "treatment_effect_mean": te.mean(axis=0),
        "treatment_effect_lower": jnp.percentile(te, 2.5, axis=0),
        "treatment_effect_upper": jnp.percentile(te, 97.5, axis=0),

        "counterfactual_samples": cf_full,
        "weights": samples["beta"],
        "posterior_samples": samples,
    }







def bscm_proximal(
        y_pre: jnp.ndarray,
        y_post: jnp.ndarray,
        X_donors_pre: jnp.ndarray,  # included ("clean") donors
        X_donors_post: jnp.ndarray,
        Z_proxies_pre: jnp.ndarray,  # excluded ("spillover") donors → proxies
        Z_proxies_post: jnp.ndarray,
        concentration: float = 0.4,
        num_warmup: int = 1000,
        num_samples: int = 3000,
        num_chains: int = 4,
        seed: int = 0,
) -> Dict[str, jnp.ndarray]:
    """
    Bayesian Proximal Synthetic Control.

    Implements Equation (5) of O’Riordan & Gilligan-Lee (2025), which
    combines (i) a Bayesian synthetic control outcome model with Dirichlet
    weights and (ii) a proximal debiasing model using excluded donors as
    proxies. All estimation is performed in z-scored space, and posterior
    counterfactuals are returned on the original outcome scale.

    Parameters
    ----------
    y_pre : jnp.ndarray
        Treated unit's pre-treatment outcomes (T0,).
    y_post : jnp.ndarray
        Treated unit's post-treatment outcomes (T1,).
    X_donors_pre : jnp.ndarray
        Included donor outcomes, pre-treatment (T0, J).
    X_donors_post : jnp.ndarray
        Included donor outcomes, post-treatment (T1, J).
    Z_proxies_pre : jnp.ndarray
        Excluded donor outcomes (proxies), pre-treatment (T0, K).
    Z_proxies_post : jnp.ndarray
        Excluded donor outcomes, post-treatment (T1, K).
    concentration : float
        Dirichlet concentration for donor weight prior.
    num_warmup : int
        Number of NUTS warmup iterations.
    num_samples : int
        Number of posterior draws per chain.
    num_chains : int
        Number of MCMC chains.
    seed : int
        Random seed for NumPyro.

    Returns
    -------
    dict
        Dictionary containing:
        - counterfactual_mean : posterior mean counterfactual path.
        - counterfactual_lower/upper : 95% credible interval.
        - treatment_effect_mean : posterior mean treatment effect.
        - treatment_effect_lower/upper : treatment effect CI.
        - counterfactual_samples : all posterior draws.
        - weights : posterior draws of donor weights β.
        - posterior_samples : all MCMC samples from NumPyro.

    References
    ----------
    O’Riordan, M., & Gilligan-Lee, C. M. (2025).
        *Spillover detection for donor selection in synthetic control models*.
        Journal of Causal Inference, 13(1), 20240036.
        https://doi.org/10.1515/jci-2024-0036
    """

    T0 = y_pre.shape[0]
    J = X_donors_pre.shape[1]  # number of clean donors
    K = Z_proxies_pre.shape[1]  # number of proxy donors

    # =================================================================
    # 1. Z-SCORE STANDARDIZATION (pre-treatment only)
    # =================================================================
    # y standardization
    y_pre_std, y_post_std, y_mean, y_std = standardize_vector_pre_only(y_pre, y_post)

    # X donors standardization
    X_pre_std, X_post_std, X_mean, X_std = standardize_matrix_pre_only(
        X_donors_pre, X_donors_post
    )

    # Z proxies standardization
    Z_pre_std, Z_post_std, Z_mean, Z_std = standardize_matrix_pre_only(
        Z_proxies_pre, Z_proxies_post
    )

    model = build_proximal_scm_model(
        y_pre_std=y_pre_std,
        y_post=y_post,
        X_pre_std=X_pre_std,
        X_post_std=X_post_std,
        Z_pre_std=Z_pre_std,
        J=J,
        K=K,
        y_mean=y_mean,
        y_std=y_std,
        concentration=concentration,
    )

    samples = run_mcmc(model, seed=seed)
    results = extract_posterior_results(samples)
    return results
