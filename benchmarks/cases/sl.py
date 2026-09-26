"""SL: the size and power of Viviano and Bradic's own test on their own design,
plus the two corrections their replication code needs and the structural facts
about the expert library that decide what an SL number means.

Path B. The design is their ``Factor_model`` DGP (``libraries/library.R:226``,
``DGP = 1``) driven through ``simulate_data_helper``: a common factor and a common
``theta`` shared by the treated unit and every donor, donor-specific AR(1) errors
at ``rho = 0.6``, loadings and means ``lambda_j = mu_j = j / p``, and a planted
constant effect added to the treated unit's post-treatment periods. Their Table 1
sweeps that effect over ``c(0, 0.1, ..., 1.5)`` and reports rejection rates, which
is what is pinned here: size at effect zero, power at the top of their grid, and
monotonicity in between.

Path A is not pinned. The paper's empirical application needs a Tennessee MEDCOST
series their package does not ship -- all four of its state matrices have 50
columns and none is 47 -- so reproducing it needs BRFSS microdata that is not
vendored here. That comparison was run and is reported on
``docs/replications/sl.rst``; it is evidence, not a check, and the replication
page says so.

Two of the pins below exist because the authors' code is wrong in ways that move
published numbers, and a regression toward either would be invisible otherwise:

* ``critical_values_move_with_eta``. Their ``function4boot_TE`` refits the
  ensemble with ``eta = 1`` hard-coded (``library.R:214``) while the observed
  statistic uses ``1/(sqrt(88) var(y)) = 51.43``, so the null distribution
  describes a near-equal-weighted ensemble and the statistic an exponentially
  weighted one. Measured on their two raw blocks that inflates the 10 percent
  critical value by 19 and 61 percent.
* ``lasso_expert_is_deterministic``. Their ``cv.glmnet(nfolds = 5)`` draws the
  penalty's folds from the RNG. On their 30-period window ``lambda.min`` lands in
  one of three places, the worst keeping no donors at all, which moves the
  reported effect by 40 percent with nothing but the order the script's blocks ran
  in.

Two more record what the ensembling actually does, measured on every run:
``effective_k_at_paper_eta`` and ``error_participation_ratio``. At the paper's own
learning rate the weighting averages instead of selecting, and the four experts'
errors span roughly one direction, so averaging them cannot cancel much.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# The size arm and the recovery arm are pinned as numbers, so they get enough
# draws to be measurements: at a true size of 0.05, 40 replicates carry a Monte
# Carlo standard error of 0.034, which is most of the quantity. The middle effect
# only has to sit between the other two, so it stays cheap.
N_SIM = {0.0: 120, 0.5: 40, 1.5: 120}
EFFECTS = (0.0, 0.5, 1.5)
N_PERIODS, T0, P_DONORS, RHO = 80, 70, 10, 0.6
N_BOOT = 500


def _ar1(rho: float, innov: np.ndarray) -> np.ndarray:
    """AR(1) driven by ``innov``, started from its stationary distribution."""
    out = np.empty(innov.size)
    out[0] = innov[0] / np.sqrt(max(1.0 - rho ** 2, 1e-12))
    for t in range(1, innov.size):
        out[t] = rho * out[t - 1] + innov[t]
    return out


def _panel(seed: int, effect: float) -> pd.DataFrame:
    """Their ``sim_process_factor_model`` with ``DGP = 1``, then the effect added.

    ``u_t`` reuses the last donor's innovations, which is what their code does
    (``library.R:239`` passes ``innov = xi_j`` after the donor loop has ended);
    it is kept so the design is theirs and not a tidied version of it.
    """
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(N_PERIODS)
    theta = rng.standard_normal(N_PERIODS)
    lam = mu = np.arange(1, P_DONORS + 1) / P_DONORS
    sd = 1.0 - RHO ** 2
    X = np.empty((N_PERIODS, P_DONORS))
    xi = None
    for j in range(P_DONORS):
        xi = rng.normal(0.0, sd, N_PERIODS)
        X[:, j] = lam[j] * factor + mu[j] + _ar1(RHO, xi) + theta
    y = 0.5 + 0.5 * factor + theta + _ar1(RHO, xi)
    y[T0:] += effect
    Y = np.column_stack([y, X])
    units = np.repeat(np.arange(P_DONORS + 1), N_PERIODS)
    times = np.tile(np.arange(N_PERIODS), P_DONORS + 1)
    return pd.DataFrame({
        "unit": units, "time": times, "y": Y.T.ravel(),
        "D": ((units == 0) & (times >= T0)).astype(int)})


def _cfg(df: pd.DataFrame, **kw) -> dict:
    base = dict(df=df, outcome="y", treat="D", unitid="unit", time="time",
                n_boot=N_BOOT, seed=0, display_graphs=False)
    base.update(kw)
    return base


def run() -> dict:
    from mlsynth import SL
    from mlsynth.utils.sl_helpers.experts import EXPERTS, build_experts
    from mlsynth.utils.sl_helpers.inference import block_bootstrap_test
    from mlsynth.utils.sl_helpers.weights import exponential_weights

    # --- size, power and recovery on their Table 1 design --------------------
    reject, mae, eff_k, part, n_used = {}, {}, [], [], {}
    for effect in EFFECTS:
        p, err = [], []
        for s in range(N_SIM[effect]):
            r = SL(_cfg(_panel(2000 + s, effect))).fit()
            p.append(r.fit.p_value)
            err.append(abs(r.effects.att - effect))
            if effect == 0.0:
                eff_k.append(r.fit.effective_k)
                part.append(r.fit.error_participation_ratio)
        reject[effect] = float(np.mean(np.asarray(p) < 0.05))
        mae[effect] = float(np.mean(err))
        n_used[effect] = len(p)

    monotone = float(reject[0.0] <= reject[0.5] <= reject[1.5])

    # --- the two corrections -------------------------------------------------
    df = _panel(2000, 1.0)
    fit = SL(_cfg(df)).fit().fit
    P, y = fit.predictions, np.asarray(
        SL(_cfg(df)).fit().time_series.observed_outcome, dtype=float)
    weight = np.arange(fit.train_periods, T0)
    post = np.arange(T0, N_PERIODS)
    kw = dict(train=slice(0, fit.train_periods), weight=weight, post=post,
              n_boot=400, block=3, seed=0)
    lo = block_bootstrap_test(P, y, eta=1.0, **kw)
    hi = block_bootstrap_test(P, y, eta=500.0, **kw)
    eta_wired = float(lo.critical_values[0.05] != hi.critical_values[0.05])

    Yco = np.column_stack([
        df.loc[df.unit == u].sort_values("time").y.to_numpy(float)
        for u in range(1, P_DONORS + 1)])
    yt = df.loc[df.unit == 0].sort_values("time").y.to_numpy(float)
    paths = [build_experts(Yco, yt, slice(0, fit.train_periods), ("lasso",),
                           seed=s).predictions for s in (0, 1, 7, 2026)]
    lasso_spread = float(max(np.max(np.abs(q - paths[0])) for q in paths[1:]))

    # --- eta's two limits ----------------------------------------------------
    zero = SL(_cfg(df, eta=0.0)).fit().fit
    avg_gap = float(np.max(np.abs(zero.counterfactual
                                  - zero.predictions.mean(axis=1))))
    sharp = SL(_cfg(df, eta=1e8)).fit().fit

    return {
        "size_at_5pct": reject[0.0],
        "power_at_effect_1p5": reject[1.5],
        "power_is_monotone_in_the_effect": monotone,
        "recovery_mean_abs_error_at_1p5": mae[1.5],
        "effective_k_at_paper_eta": float(np.mean(eff_k)),
        "error_participation_ratio": float(np.mean(part)),
        "critical_values_move_with_eta": eta_wired,
        "lasso_expert_is_deterministic": lasso_spread,
        "eta_zero_is_the_simple_average": avg_gap,
        "eta_large_selects_one_expert": float(sharp.effective_k),
        "experts_built": float(len(EXPERTS)),
        "size_replicates": float(n_used[0.0]),
    }


EXPECTED = {
    # Size is 0.092 here and 0.067 on a different 120-seed range, pooling to
    # 0.079 over 240 draws against a nominal 0.05 with a standard error of 0.014.
    # So the test over-rejects mildly on a panel this size, which Theorem 3.1's
    # asymptotic guarantee does not forbid: the split leaves 28 weighting periods
    # and the bootstrap pool is 38 long at block 3. The tolerance catches a size
    # blowout and survives a boundary p-value flipping on a different BLAS.
    "size_at_5pct": (0.0917, 0.05),
    # Power at the top of their grid. The loose end still demands 0.91.
    "power_at_effect_1p5": (0.9917, 0.08),
    "power_is_monotone_in_the_effect": (1.0, 0.0),
    # The planted effect is 1.5, so even the loose end keeps the error under a
    # fifth of it. Flat in the effect size, since it is the ensemble's own
    # prediction error and not anything the effect changes.
    "recovery_mean_abs_error_at_1p5": (0.2474, 0.08),
    # The averaging regime, on the paper's own design and its own learning rate.
    # Pinned to stay above 3.47 of 4: an SL fit that silently started selecting
    # would be a different estimator.
    "effective_k_at_paper_eta": (3.7231, 0.25),
    # ...and the errors spanning roughly one direction, which is why the
    # averaging buys little. Pinned to stay under 1.72 of 4.
    "error_participation_ratio": (1.3643, 0.35),
    # The two corrections. Both are booleans or exact zeros, so both are pinned
    # tight: a regression here would restore a defect that moves a published
    # number without failing anything else.
    "critical_values_move_with_eta": (1.0, 0.0),
    "lasso_expert_is_deterministic": (0.0, 1e-12),
    # Equation 12's two limits, both exact.
    "eta_zero_is_the_simple_average": (0.0, 1e-10),
    "eta_large_selects_one_expert": (1.0, 1e-3),
    "experts_built": (4.0, 0.0),
    "size_replicates": (120.0, 0.0),
}
