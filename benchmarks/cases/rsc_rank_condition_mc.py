r"""RSC property case: the rank condition, and the threshold tradeoff.

Path C (property; scenario 1 -- paper only). Two results from Amjad, Shah &
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
    cf = _fit_pcr(p.observed, p.T0, target=0, rank=RANK_R)
    return float(np.sqrt(np.mean((cf[p.T0:] - p.means[0, p.T0:]) ** 2)))


def _fit_pcr(X: np.ndarray, T0: int, target: int, rank: int) -> np.ndarray:
    """RSC (HSVT-denoise the donors, then OLS) through the public API."""
    wide = pd.DataFrame(X.T)
    wide.columns = range(X.shape[0])
    long = (wide.reset_index()
            .melt(id_vars="index", var_name="unit", value_name="y")
            .rename(columns={"index": "time"}))
    long["treat"] = ((long["unit"] == target) & (long["time"] >= T0)).astype(int)
    res = CLUSTERSC({
        "df": long, "outcome": "y", "treat": "treat",
        "unitid": "unit", "time": "time",
        "method": "pcr", "clustering": False, "pcr_objective": "OLS",
        "rank": rank, "rank_method": "fixed", "project_denoised": True,
        "compute_shen_ci": False, "display_graphs": False,
    }).fit()
    return np.asarray(res.counterfactual).ravel()


def _sweep_cell(rank: int) -> tuple[float, float]:
    """Pre- and post-intervention MSE against the truth at one threshold.

    Theorem 3 bounds the error against the true mean, not the training
    residual, so both halves are scored against ``panel.means``.
    """
    panel = simulate_rsc_panel(N=100, T=2000, T0=1600, noise=SWEEP_NOISE, seed=0)
    M, T0 = panel.means, panel.T0
    pre, post = [], []
    for ti in range(SWEEP_TARGETS):
        cf = _fit_pcr(panel.observed, T0, target=ti, rank=rank)
        pre.append(float(np.mean((cf[:T0] - M[ti, :T0]) ** 2)))
        post.append(float(np.mean((cf[T0:] - M[ti, T0:]) ** 2)))
    return float(np.mean(pre)), float(np.mean(post))


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
    "sweep_pre_mse_r1": (0.1601, 0.02),
    "sweep_pre_mse_r2": (0.0221, 0.004),
    "sweep_pre_mse_r25": (0.0326, 0.006),
}
