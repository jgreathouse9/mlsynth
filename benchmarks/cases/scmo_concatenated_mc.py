"""Path B benchmark: concatenated multi-outcome SCM (Tian-Lee-Panchenko 2026, Table 1).

Reproduces the simulation that motivates the concatenated SCMO: under a factor
model whose outcomes share the unit predictors, matching the synthetic control on
*more* related outcomes sharpens identification of the latent predictors and so
reduces the post-treatment bias of the single-outcome SC -- at the cost of a
larger (but truthful) pre-treatment fit error, which rises toward the noise floor
instead of overfitting to near-zero. The same table carries the paper's second
comparison: a ridge-augmented SC on the same ten-outcome design, which cuts the
bias further where the pre-treatment fit is imperfect.

The reported cells are, for each ``T0`` in {1, 5, 10} and each outcome count
``K`` in {1, 5, 10}: the average pre-treatment RMSPE ("pre"), the average
absolute post-period gap on outcome 1 ("bias"), and the standard deviation of
that signed gap ("sd"), over ``M`` null (``tau = 0``) draws. ``K = 1`` is the
conventional single-outcome SC (the ``separate`` scheme); ``K = 5, 10`` are the
``concatenated`` multi-outcome SC; the fourth arm is ``K = 10`` again with
``augment="ridge"``.

Provenance
----------
* DGP: :func:`mlsynth.utils.scmo_helpers.simulation.simulate_tian` -- the
  Section-3 factor model of Tian-Lee-Panchenko (2026), identical to the
  ``Simulation1.R`` DGP of the Sun et al. (2025) replication package (N = 30,
  f = 4 predictors, 1 post-period, tau = 0).
* Headline: Tian-Lee-Panchenko (2026, Econometrics Journal) Table 1, reproduced
  cell-by-cell by the authors' code as ``Output/sim_tab1.txt``::

        T0   K=1 (pre/bias/sd)  K=5               K=10              Augmented (K=10)
        1    0.04 / 1.23 / 1.57  0.38 / 1.21 / 1.53  0.62 / 1.12 / 1.42  0.49 / 1.12 / 1.41
        5    0.46 / 1.21 / 1.53  0.95 / 1.04 / 1.31  1.02 / 1.00 / 1.26  0.95 / 0.97 / 1.22
        10   0.77 / 1.13 / 1.42  1.05 / 1.01 / 1.27  1.09 / 0.98 / 1.23  1.03 / 0.95 / 1.19

  The paper uses 5,000 reps; we use M = 250 (tolerances absorb the MC gap:
  bias SE ~ 1.4/sqrt(250) ~ 0.09).
* The augmented arm is ``Simulation1.R``'s fourth method, which stacks the ten
  outcomes' pre-periods into one series and hands it to ``augsynth`` (ridge
  progfunc, no unit fixed effects). mlsynth reaches it as
  ``augment="ridge"`` on the concatenated scheme, with the penalty selected by
  cross-validation. The two implementations differ in one step -- mlsynth
  standardizes the matching columns by their cross-unit SD, augsynth works on
  the raw stacked series -- so the augmented cells are pinned with the wider
  tolerances recorded below. The paper's reading of the column, that the
  augmentation lowers the bias of the ten-outcome SC without collapsing the
  pre-treatment fit, is pinned as a paired count over the same draws.
"""
from __future__ import annotations

import warnings

import numpy as np

M = 250
SEED = 321
T0_VALUES = (1, 5, 10)
K_VALUES = (1, 5, 10)
AUG_K = 10          # the augmented arm matches on all ten outcomes


def _stats(draws: list) -> tuple:
    """(mean pre-fit RMSPE, mean |post gap|, SD of the signed post gap)."""
    a = np.asarray(draws)
    return (float(a[:, 0].mean()), float(np.abs(a[:, 1]).mean()),
            float(a[:, 1].std(ddof=1)))


def _grid() -> dict:
    from mlsynth import SCMO
    from mlsynth.utils.scmo_helpers.simulation import simulate_tian, to_panel

    out: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for T0 in T0_VALUES:
            rng = np.random.default_rng(SEED)
            acc = {K: [] for K in K_VALUES}
            acc["aug"] = []
            for _ in range(M):
                Ys, N, TT, tr = simulate_tian(rng, T0, max(K_VALUES))
                for K in K_VALUES:
                    df = to_panel(Ys[:K], N, TT, tr)
                    cfg = {"df": df, "outcome": "y0", "treat": "treat",
                           "unitid": "unit", "time": "time",
                           "schemes": ["separate" if K == 1 else "concatenated"],
                           "display_graphs": False}
                    if K > 1:
                        cfg["addout"] = [f"y{k}" for k in range(1, K)]
                    fit = SCMO(cfg).fit()._primary
                    acc[K].append((fit.pre_rmse, float(np.asarray(fit.gap)[-1])))
                    if K == AUG_K:
                        # Same draw, same design, ridge-augmented: a paired
                        # comparison against the plain K = 10 arm.
                        aug = SCMO({**cfg, "augment": "ridge"}).fit()._primary
                        acc["aug"].append(
                            (aug.pre_rmse, float(np.asarray(aug.gap)[-1])))
            for key in (*K_VALUES, "aug"):
                out[(T0, key)] = _stats(acc[key])
    return out


def run() -> dict:
    g = _grid()
    res: dict = {}
    for T0 in T0_VALUES:
        for K in K_VALUES:
            pre, bias, sd = g[(T0, K)]
            res[f"pre_T{T0}_K{K}"] = pre
            res[f"bias_T{T0}_K{K}"] = bias
            res[f"sd_T{T0}_K{K}"] = sd
        pre, bias, sd = g[(T0, "aug")]
        res[f"aug_pre_T{T0}"] = pre
        res[f"aug_bias_T{T0}"] = bias
        res[f"aug_sd_T{T0}"] = sd
    # Headline geometry: bias falls and pre-fit rises as outcomes are added.
    res["bias_falls_with_K"] = float(all(
        g[(T0, 1)][1] >= g[(T0, 10)][1] for T0 in T0_VALUES))
    res["prefit_rises_with_K"] = float(all(
        g[(T0, 1)][0] <= g[(T0, 10)][0] for T0 in T0_VALUES))
    # The augmented column: lower bias than the plain ten-outcome SC on the same
    # draws, at every T0, with the pre-treatment fit still above the K = 1 floor.
    res["aug_reduces_bias"] = float(sum(
        g[(T0, "aug")][1] <= g[(T0, AUG_K)][1] for T0 in T0_VALUES))
    res["aug_keeps_prefit"] = float(sum(
        g[(T0, "aug")][0] > g[(T0, 1)][0] for T0 in T0_VALUES))
    return res


# Stochastic (M=250 vs the paper's 5,000). Pre-fit reproduces tightly (it is the
# deterministic geometry of the SC fit) -> +-0.08; bias absorbs MC noise -> +-0.12;
# the SD of the gap is a second moment over the same 250 draws -> +-0.15. The
# augmented arm additionally carries the standardization difference against
# augsynth described above -> +-0.15 on its pre-fit and bias, +-0.2 on its SD;
# measured, its nine cells land within 0.07 of the printed table, the same
# distance as the three plain arms.
# The two counts are paired comparisons on identical draws. aug_keeps_prefit has
# wide margins (0.51 vs 0.04, 0.96 vs 0.46, 1.05 vs 0.79). aug_reduces_bias is
# the thin one: the augmentation buys 0.014 / 0.023 / 0.030 of bias at
# T0 = 1 / 5 / 10, the same size as the gain the paper prints (0.00 / 0.03 /
# 0.03), and the count admits ties, so only a sign flip fails it.
EXPECTED = {
    "pre_T1_K1": (0.04, 0.08), "bias_T1_K1": (1.23, 0.12), "sd_T1_K1": (1.57, 0.15),
    "pre_T1_K5": (0.38, 0.08), "bias_T1_K5": (1.21, 0.12), "sd_T1_K5": (1.53, 0.15),
    "pre_T1_K10": (0.62, 0.08), "bias_T1_K10": (1.12, 0.12), "sd_T1_K10": (1.42, 0.15),
    "pre_T5_K1": (0.46, 0.08), "bias_T5_K1": (1.21, 0.12), "sd_T5_K1": (1.53, 0.15),
    "pre_T5_K5": (0.95, 0.08), "bias_T5_K5": (1.04, 0.12), "sd_T5_K5": (1.31, 0.15),
    "pre_T5_K10": (1.02, 0.08), "bias_T5_K10": (1.00, 0.12), "sd_T5_K10": (1.26, 0.15),
    "pre_T10_K1": (0.77, 0.08), "bias_T10_K1": (1.13, 0.12), "sd_T10_K1": (1.42, 0.15),
    "pre_T10_K5": (1.05, 0.08), "bias_T10_K5": (1.01, 0.12), "sd_T10_K5": (1.27, 0.15),
    "pre_T10_K10": (1.09, 0.08), "bias_T10_K10": (0.98, 0.12), "sd_T10_K10": (1.23, 0.15),
    "aug_pre_T1": (0.49, 0.15), "aug_bias_T1": (1.12, 0.15), "aug_sd_T1": (1.41, 0.2),
    "aug_pre_T5": (0.95, 0.15), "aug_bias_T5": (0.97, 0.15), "aug_sd_T5": (1.22, 0.2),
    "aug_pre_T10": (1.03, 0.15), "aug_bias_T10": (0.95, 0.15), "aug_sd_T10": (1.19, 0.2),
    "bias_falls_with_K": (1.0, 0.0),
    "prefit_rises_with_K": (1.0, 0.0),
    "aug_reduces_bias": (3.0, 0.0),
    "aug_keeps_prefit": (3.0, 0.0),
}
