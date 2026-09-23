r"""RSC property case: the rank condition, and the two hyperparameters.

Path C (property; scenario 1 -- paper only). Three results from Amjad, Shah &
Shen (2018), *Robust Synthetic Control* (JMLR 19:1-51), Section 4.

Theorem 6 is the one synthetic control has been leaning on without saying
so. The method assumes the treated signal is a donor combination in the
pre-period, :math:`M_1^- = (M^-)^\top\beta^*`, and then uses that relation
to forecast the post-period. Whether it still holds there is a separate
question, and the answer is: it does provided
:math:`\operatorname{rank}(M^-) = \operatorname{rank}(M)`. The authors note
the point "has been amiss in the literature, potentially implicitly believed
or assumed" since Abadie and Gardeazabal (2003).

Theorem 3 bounds the pre-intervention error for any singular-value threshold
:math:`\mu`, and its two leading terms move in opposite directions: the
retained set :math:`S` grows as :math:`\mu` falls, which shrinks the unmodelled
signal :math:`\lambda^*` and grows the captured noise
:math:`|S|\sigma^2/T_0`. The paper calls choosing between them the Goldilocks
principle.

Section 4.3, "Benefits of regularization", reads Theorems 3 and 7 together to
make a claim about the *other* hyperparameter, the ridge penalty
:math:`\eta` of equation (18). Theorem 3's bound carries
:math:`+\eta\|\beta^*\|^2/T_0`, which the paper reads as "the
pre-intervention error increases linearly with respect to the choice of
:math:`\eta`"; Theorem 7's second term is controlled by
:math:`\|\widehat\beta(\eta) - \beta^*\|`, which it reads as "a larger
value of :math:`\eta` reduces the post-intervention error". Together:
"employing ridge regression introduces extraneous bias into our model,
yielding a higher pre-intervention error. In exchange, regularization reduces
the post-intervention error." Both halves are directional and measurable.

Provenance
----------

Amjad, M., Shah, D. & Shen, D. (2018), *"Robust Synthetic Control"*, JMLR
19(22):1-51 -- Theorem 3 and its bias-variance reading (Section 4.2.1),
Theorem 6 (Section 4.3). The threshold sweep runs on the Section 5.3
latent-variable DGP, shared with ``rsc_synth_error``.

Bounds with unnamed constants
-----------------------------

Theorems 3 and 7 and Corollary 4 all read ":math:`\le C_1(\cdot) + C_2(\cdot)`"
with :math:`C_1, C_2` "universal positive constants" and no values given, so
none of them can be checked as a numerical bound -- there is no number to
compare against. What can be checked is the structure each asserts: which
way a quantity moves as a knob turns, and whether an equality that is
supposed to hold does. That is what this case measures, and it is why
Theorem 6, which carries no constants at all, is the headline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsynth import CLUSTERSC
from mlsynth.utils.clustersc_helpers.simulation import (
    simulate_rank_shift_panel,
    simulate_rsc_panel,
)

# --- Theorem 6 design -------------------------------------------------------
RANK_N, RANK_T, RANK_T0, RANK_R = 12, 60, 40, 3
RANK_SEEDS = range(8)
RANK_NOISE = 0.05

# --- Theorem 3 sweep (the Section 5.3 DGP, true signal rank about 3) --------
SWEEP_RANKS = (1, 2, 3, 5, 10, 25)
SWEEP_TARGETS = 3
SWEEP_NOISE = 1.0

# --- Section 4.3 eta sweep, at a well-set and an over-permissive threshold --
ETAS = (0.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
ETA_RANKS = (3, 25)
ETA_SEEDS = (0, 1)
ETA_TARGETS = 3


def _theorem6_gap(dormant: bool, seed: int) -> float:
    r"""Post-period failure of a pre-period-fitted relation, on the truth.

    No estimator and no noise: fits :math:`\beta` to reproduce
    :math:`M_1^-` exactly, then measures
    :math:`\max_t |M_{1t} - (M^+)^\top\beta|` after the intervention. Zero
    means the relation extrapolated.
    """
    p = simulate_rank_shift_panel(dormant_factor=dormant, N=RANK_N, T=RANK_T,
                                  T0=RANK_T0, n_factors=RANK_R, noise=0.0,
                                  seed=seed)
    M, T0 = p.means, p.T0
    beta = np.linalg.lstsq(M[1:, :T0].T, M[0, :T0], rcond=None)[0]
    return float(np.abs(M[1:, T0:].T @ beta - M[0, T0:]).max())


def _rsc_post_rmse(dormant: bool, seed: int) -> float:
    """What the rank condition costs the estimator, on a noisy panel."""
    p = simulate_rank_shift_panel(dormant_factor=dormant, N=RANK_N, T=RANK_T,
                                  T0=RANK_T0, n_factors=RANK_R,
                                  noise=RANK_NOISE, seed=seed)
    cf, _ = _fit_pcr(p.observed, p.T0, target=0, rank=RANK_R)
    return float(np.sqrt(np.mean((cf[p.T0:] - p.means[0, p.T0:]) ** 2)))


def _fit_pcr(X: np.ndarray, T0: int, target: int, rank: int,
             eta: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """RSC (HSVT-denoise the donors, then OLS) through the public API.

    ``eta`` is the paper's equation (18) penalty at ``q = 2``; mlsynth spells
    the same objective ``lambda_penalty * ||w||_p ** q``, so ``p = q = 2``
    matches it (held by ``tests/test_pcr_ridge.py``). Returns the
    counterfactual and the fitted weights.
    """
    wide = pd.DataFrame(X.T)
    wide.columns = range(X.shape[0])
    long = (wide.reset_index()
            .melt(id_vars="index", var_name="unit", value_name="y")
            .rename(columns={"index": "time"}))
    long["treat"] = ((long["unit"] == target) & (long["time"] >= T0)).astype(int)
    cfg = {
        "df": long, "outcome": "y", "treat": "treat",
        "unitid": "unit", "time": "time",
        "method": "pcr", "clustering": False, "pcr_objective": "OLS",
        "rank": rank, "rank_method": "fixed", "project_denoised": True,
        "compute_shen_ci": False, "display_graphs": False,
    }
    if eta:
        cfg |= {"lambda_penalty": eta, "p": 2.0, "q": 2.0}
    res = CLUSTERSC(cfg).fit()
    weights = res.weights.donor_weights
    beta = np.array([weights[str(j)] for j in sorted(int(k) for k in weights)])
    return np.asarray(res.counterfactual).ravel(), beta


def _sweep_cell(rank: int) -> tuple[float, float]:
    """Pre- and post-intervention MSE against the truth at one threshold.

    Theorem 3 bounds the error against the true mean, not the training
    residual, so both halves are scored against ``panel.means``.
    """
    panel = simulate_rsc_panel(N=100, T=2000, T0=1600, noise=SWEEP_NOISE, seed=0)
    M, T0 = panel.means, panel.T0
    pre, post = [], []
    for ti in range(SWEEP_TARGETS):
        cf, _ = _fit_pcr(panel.observed, T0, target=ti, rank=rank)
        pre.append(float(np.mean((cf[:T0] - M[ti, :T0]) ** 2)))
        post.append(float(np.mean((cf[T0:] - M[ti, T0:]) ** 2)))
    return float(np.mean(pre)), float(np.mean(post))


def _eta_cell(rank: int, eta: float) -> dict:
    r"""One (threshold, penalty) cell, scored the paper's way.

    ``pre`` is equation (24)'s MSE and ``post`` equation (33)'s RMSE, both
    against the true signal :math:`M` -- which is what Theorems 3 and 7 bound,
    and is not the training error. ``train`` is the fit to the observed
    :math:`Y`, the quantity the paper's intuition for the eta term is actually
    about, so the two can be told apart. ``beta_gap`` is Theorem 7's own
    :math:`\|\widehat\beta(\eta) - \beta^*\|`, with :math:`\beta^*` the
    pre-period relation on the noise-free signal.
    """
    pre, train, post, beta_gap, beta_norm = [], [], [], [], []
    for seed in ETA_SEEDS:
        panel = simulate_rsc_panel(N=100, T=2000, T0=1600,
                                   noise=SWEEP_NOISE, seed=seed)
        M, Y, T0 = panel.means, panel.observed, panel.T0
        for ti in range(ETA_TARGETS):
            cf, beta = _fit_pcr(Y, T0, target=ti, rank=rank, eta=eta)
            donors = [j for j in range(M.shape[0]) if j != ti]
            beta_star = np.linalg.lstsq(M[donors, :T0].T, M[ti, :T0],
                                        rcond=None)[0]
            pre.append(np.mean((cf[:T0] - M[ti, :T0]) ** 2))
            train.append(np.mean((cf[:T0] - Y[ti, :T0]) ** 2))
            post.append(np.sqrt(np.mean((cf[T0:] - M[ti, T0:]) ** 2)))
            beta_gap.append(np.linalg.norm(beta - beta_star))
            beta_norm.append(np.linalg.norm(beta))
    return {"pre": float(np.mean(pre)), "train": float(np.mean(train)),
            "post": float(np.mean(post)),
            "beta_gap": float(np.mean(beta_gap)),
            "beta_norm": float(np.mean(beta_norm))}


def _increasing(values, tol: float = 0.02) -> float:
    """Non-decreasing to within ``tol`` of the first value.

    The eta columns are flat over part of the grid, at a level where the
    conic solver's own accuracy shows. A strict test would report noise; the
    slack is far below the effects being separated (a 19 per cent fall at one
    threshold against a 2 per cent rise at the other).
    """
    v = list(values)
    slack = tol * abs(v[0])
    return float(all(b >= a - slack for a, b in zip(v, v[1:])))


def _decreasing_seq(values, tol: float = 0.02) -> float:
    v = list(values)
    slack = tol * abs(v[0])
    return float(all(b <= a + slack for a, b in zip(v, v[1:])))


def run() -> dict:
    out: dict[str, float] = {}

    # --- Theorem 6, on the noise-free mean matrix ---------------------------
    kept = [_theorem6_gap(False, s) for s in RANK_SEEDS]
    lost = [_theorem6_gap(True, s) for s in RANK_SEEDS]
    out["thm6_gap_rank_preserved_max"] = max(kept)
    out["thm6_gap_rank_deficient_min"] = min(lost)
    out["thm6_gap_rank_deficient_mean"] = float(np.mean(lost))
    # The hypothesis itself, so the design is pinned alongside the conclusion.
    p_keep = simulate_rank_shift_panel(dormant_factor=False, N=RANK_N, T=RANK_T,
                                       T0=RANK_T0, n_factors=RANK_R,
                                       noise=0.0, seed=0)
    p_lose = simulate_rank_shift_panel(dormant_factor=True, N=RANK_N, T=RANK_T,
                                       T0=RANK_T0, n_factors=RANK_R,
                                       noise=0.0, seed=0)
    out["rank_pre_preserved"] = float(
        np.linalg.matrix_rank(p_keep.means[:, :RANK_T0]))
    out["rank_full_preserved"] = float(np.linalg.matrix_rank(p_keep.means))
    out["rank_pre_deficient"] = float(
        np.linalg.matrix_rank(p_lose.means[:, :RANK_T0]))
    out["rank_full_deficient"] = float(np.linalg.matrix_rank(p_lose.means))

    # --- and what it costs the estimator ------------------------------------
    rsc_keep = float(np.mean([_rsc_post_rmse(False, s) for s in RANK_SEEDS]))
    rsc_lose = float(np.mean([_rsc_post_rmse(True, s) for s in RANK_SEEDS]))
    out["rsc_post_rmse_rank_preserved"] = rsc_keep
    out["rsc_post_rmse_rank_deficient"] = rsc_lose
    out["rsc_post_rmse_ratio"] = rsc_lose / rsc_keep

    # --- Theorem 3: the threshold's bias-variance tradeoff ------------------
    sweep = {r: _sweep_cell(r) for r in SWEEP_RANKS}
    for r, (pre, post) in sweep.items():
        out[f"sweep_pre_mse_r{r}"] = pre
        out[f"sweep_post_mse_r{r}"] = post
    posts = [sweep[r][1] for r in SWEEP_RANKS]
    best = int(np.argmin(posts))
    out["sweep_argmin_rank"] = float(SWEEP_RANKS[best])
    # A U-shape: the error falls to the optimum and rises after it.
    out["sweep_u_shaped"] = float(
        all(b <= a for a, b in zip(posts[:best + 1], posts[1:best + 1]))
        and all(b >= a for a, b in zip(posts[best:], posts[best + 1:])))
    out["sweep_post_penalty_underfit"] = posts[0] / posts[best]
    out["sweep_post_penalty_overfit"] = posts[-1] / posts[best]

    # --- Section 4.3: what the ridge penalty does at each threshold --------
    eta = {(r, e): _eta_cell(r, e) for r in ETA_RANKS for e in ETAS}
    for r in ETA_RANKS:
        for e in ETAS:
            tag = f"r{r}_eta{int(e)}"
            out[f"eta_pre_{tag}"] = eta[(r, e)]["pre"]
            out[f"eta_train_{tag}"] = eta[(r, e)]["train"]
            out[f"eta_post_{tag}"] = eta[(r, e)]["post"]
        col = lambda k: [eta[(r, e)][k] for e in ETAS]           # noqa: E731
        # The training error rises in eta at every threshold, which is what
        # "handicaps its ability to fit the data" predicts. Whether the error
        # against the signal follows it is the separate question below.
        out[f"eta_train_rises_r{r}"] = _increasing(col("train"))
        out[f"eta_train_ratio_r{r}"] = col("train")[-1] / col("train")[0]
        # Shrinkage is happening, so a flat error column is not an inert knob.
        out[f"eta_beta_norm_falls_r{r}"] = _decreasing_seq(col("beta_norm"))
        # The two directional claims of Section 4.3, at this threshold.
        pres, posts_e, gaps = col("pre"), col("post"), col("beta_gap")
        out[f"eta_pre_rises_r{r}"] = _increasing(pres)
        out[f"eta_pre_improvement_r{r}"] = pres[0] / min(pres)
        out[f"eta_post_improvement_r{r}"] = posts_e[0] / min(posts_e)
        # Theorem 7's own term. Farebrother (1976), which the paper invokes,
        # asserts some eta > 0 improves it; this records by how much.
        out[f"eta_beta_gap_improvement_r{r}"] = gaps[0] / min(gaps)
        # And where the bound's eta-dependent term still improves past the
        # eta at which the error it bounds has already turned around. A
        # one-step flip in either argmin leaves this unchanged.
        out[f"eta_beta_gap_outlasts_post_r{r}"] = float(
            np.argmin(gaps) > np.argmin(posts_e))
    out["eta_post_best_r25"] = float(
        ETAS[int(np.argmin([eta[(25, e)]["post"] for e in ETAS]))])
    return out


# Every cell is deterministic -- fixed seeds, no resampling -- so these
# reproduce exactly. The tolerances say how far a quantity may drift before
# the claim behind it has changed.
EXPECTED = {
    # --- Theorem 6, on the noise-free mean matrix -------------------------
    # Where the ranks agree, a relation fitted on the pre-period reproduces
    # the post-period to machine precision, over all eight designs.
    "thm6_gap_rank_preserved_max": (1.8e-15, 1e-10),
    # Where they do not, it fails on every one of them.
    "thm6_gap_rank_deficient_min": (0.114, 0.02),
    "thm6_gap_rank_deficient_mean": (0.493, 0.05),
    # The hypothesis, pinned so the conclusion cannot be read off a design
    # that stopped satisfying it.
    "rank_pre_preserved": (3.0, 0.0),
    "rank_full_preserved": (3.0, 0.0),
    "rank_pre_deficient": (2.0, 0.0),
    "rank_full_deficient": (3.0, 0.0),

    # --- what the condition costs RSC on a noisy panel --------------------
    "rsc_post_rmse_rank_preserved": (0.0244, 0.005),
    "rsc_post_rmse_rank_deficient": (0.342, 0.05),
    "rsc_post_rmse_ratio": (14.03, 3.0),

    # --- Theorem 3: the threshold's bias-variance tradeoff ----------------
    # Post-intervention error against the truth, swept over the retained
    # rank. Falls steeply out of the underfit regime, then climbs as the
    # threshold admits noise -- the Goldilocks principle, measured.
    "sweep_post_mse_r1": (0.2472, 0.03),
    "sweep_post_mse_r2": (0.0236, 0.004),
    "sweep_post_mse_r3": (0.0239, 0.004),
    "sweep_post_mse_r5": (0.0247, 0.004),
    "sweep_post_mse_r10": (0.0264, 0.004),
    "sweep_post_mse_r25": (0.0344, 0.006),
    "sweep_u_shaped": (1.0, 0.0),
    # The minimum sits at rank 2, with rank 3 within 1.4% of it -- the
    # Section 5.3 signal is "approximately rank 3" and its third component
    # is weak beside this noise level, so the two are effectively tied. The
    # tolerance admits either without admitting 1 or 5.
    "sweep_argmin_rank": (2.5, 0.5),
    "sweep_post_penalty_underfit": (10.47, 2.0),
    "sweep_post_penalty_overfit": (1.458, 0.3),
    # The pre-intervention error, which Theorem 3 bounds directly, traces
    # the same shape.

    # --- Section 4.3: the ridge penalty, at each threshold ----------------
    # MSE (eq. 24) on the pre window and RMSE (eq. 33) on the post, both
    # against the true signal M -- what Theorems 3 and 7 bound. "train" is
    # the fit to the observed Y, which is what the paper's intuition for the
    # eta term is about, kept alongside so the two can be told apart.
    # retained rank 3: the threshold matches the signal's approximate rank
    "eta_pre_r3_eta0": (0.0235, 0.004),
    "eta_train_r3_eta0": (1.0193, 0.004),
    "eta_post_r3_eta0": (0.1568, 0.010),
    "eta_pre_r3_eta100": (0.0234, 0.004),
    "eta_train_r3_eta100": (1.0193, 0.004),
    "eta_post_r3_eta100": (0.1567, 0.010),
    "eta_pre_r3_eta300": (0.0234, 0.004),
    "eta_train_r3_eta300": (1.0193, 0.004),
    "eta_post_r3_eta300": (0.1569, 0.010),
    "eta_pre_r3_eta1000": (0.0235, 0.004),
    "eta_train_r3_eta1000": (1.0197, 0.004),
    "eta_post_r3_eta1000": (0.1585, 0.010),
    "eta_pre_r3_eta3000": (0.0257, 0.004),
    "eta_train_r3_eta3000": (1.0224, 0.004),
    "eta_post_r3_eta3000": (0.1697, 0.010),
    "eta_pre_r3_eta10000": (0.0415, 0.004),
    "eta_train_r3_eta10000": (1.0390, 0.004),
    "eta_post_r3_eta10000": (0.2224, 0.010),

    # retained rank 25: an order of magnitude too permissive
    "eta_pre_r25_eta0": (0.0341, 0.004),
    "eta_train_r25_eta0": (1.0087, 0.004),
    "eta_post_r25_eta0": (0.1902, 0.010),
    "eta_pre_r25_eta100": (0.0331, 0.004),
    "eta_train_r25_eta100": (1.0087, 0.004),
    "eta_post_r25_eta100": (0.1876, 0.010),
    "eta_pre_r25_eta300": (0.0315, 0.004),
    "eta_train_r25_eta300": (1.0088, 0.004),
    "eta_post_r25_eta300": (0.1835, 0.010),
    "eta_pre_r25_eta1000": (0.0284, 0.004),
    "eta_train_r25_eta1000": (1.0101, 0.004),
    "eta_post_r25_eta1000": (0.1756, 0.010),
    "eta_pre_r25_eta3000": (0.0276, 0.004),
    "eta_train_r25_eta3000": (1.0154, 0.004),
    "eta_post_r25_eta3000": (0.1766, 0.010),
    "eta_pre_r25_eta10000": (0.0418, 0.004),
    "eta_train_r25_eta10000": (1.0355, 0.004),
    "eta_post_r25_eta10000": (0.2238, 0.010),
    # The paper: "the pre-intervention error increases linearly with respect
    # to the choice of eta". At the well-set threshold it does (flat, then
    # steeply up). At the over-permissive one it falls 19 per cent first.
    "eta_pre_rises_r3": (1.0, 0.0),
    "eta_pre_rises_r25": (0.0, 0.0),
    "eta_pre_improvement_r3": (1.0054, 0.03),
    "eta_pre_improvement_r25": (1.2353, 0.08),

    # The paper: "a larger value of eta reduces the post-intervention error."
    # At the well-set threshold the best cell beats eta = 0 by 0.01 per cent,
    # which is nothing; at the over-permissive one it beats it by 7.7.
    "eta_post_improvement_r3": (1.0001, 0.02),
    "eta_post_improvement_r25": (1.0829, 0.05),
    # A clear interior minimum, with the neighbouring cells within 0.6 per
    # cent, so the tolerance admits a one-step move without admitting either
    # end of the grid.
    "eta_post_best_r25": (1000.0, 2000.0),

    # The intuition offered for the eta term is about the fit to Y, and about
    # that it is right: the training error rises in eta at both thresholds.
    # Equation (24) is against M, not Y, and at rank 25 the two move
    # oppositely on the same fits.
    "eta_train_rises_r3": (1.0, 0.0),
    "eta_train_rises_r25": (1.0, 0.0),
    "eta_train_ratio_r3": (1.0193, 0.01),
    "eta_train_ratio_r25": (1.0266, 0.01),

    # The shrinkage is real, so a flat error column is not an inert knob.
    "eta_beta_norm_falls_r3": (1.0, 0.0),
    "eta_beta_norm_falls_r25": (1.0, 0.0),

    # Theorem 7's own term. Farebrother (1976), which the paper invokes for
    # the post-intervention half, holds at both thresholds: some eta > 0
    # improves ||beta_hat(eta) - beta*||, by 11 per cent at rank 3 and 51 at
    # rank 25.
    "eta_beta_gap_improvement_r3": (1.1248, 0.05),
    "eta_beta_gap_improvement_r25": (2.0319, 0.15),
    # And at both thresholds that term is still improving at an eta where the
    # error it bounds has already turned around -- the distance between a
    # bound and the quantity under it.
    "eta_beta_gap_outlasts_post_r3": (1.0, 0.0),
    "eta_beta_gap_outlasts_post_r25": (1.0, 0.0),

    "sweep_pre_mse_r1": (0.1601, 0.02),
    "sweep_pre_mse_r2": (0.0221, 0.004),
    "sweep_pre_mse_r25": (0.0326, 0.006),
}
