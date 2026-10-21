"""Path-B benchmark: the base MAREX design is unbiased on Abadie and Zhao's Section 5.

Abadie, A. and Zhao, J. "Synthetic Controls for Experimental Design."
arXiv:2108.02196 [ABADIE2024], Section 5 and Table 2.

Scenario 3 (full repo), path B, with the simulator cross-validated draw for
draw. The authors' ``SCDesign`` repository (github.com/jinglongzhao2/SCDesign)
runs simulation ``i`` of 1000 as ``set.seed(i)`` followed by
``Generate_Model_Primitives`` ("2. Many Simulations (...)/Main_LazyRun.R",
lines 48-98 and 103-263; ``R_job.qsub`` submits the array job
``#$ -t 1-1000`` and the driver seeds with the task id).
:func:`mlsynth.utils.marex_helpers.simulation.generate_section5_sample` ports
R's ``set.seed``, Mersenne-Twister and inversion ``rnorm`` and replays that
function, so this case runs the paper's own 1000 panels. Two checks pin the
port against R's output in ``benchmarks/reference/marex_scdesign_sim/``: every
value of the 12 panels R captured, and the 1000-seed effect path R printed.

Why the bias of a design is known exactly
-----------------------------------------
The estimator is the paper's equation (8),
``tau_hat_t = w'Y_I,t - v'Y_N,t``, against ``tau_t = f'(Y_I,t - Y_N,t)``
with ``f_j = 1/J``. In this DGP every coefficient vector is drawn afresh each
period on Uniform(0, 10), and ``w``, ``v`` and ``f`` each sum to one, so the
period intercepts cancel. The design reads only the fitting periods, which are
independent of the post-period coefficients and noise. Given the design,

    E[tau_hat_t - tau_t | w, v] = 5 (w - v)' s,    s_j = sum_k Z_kj + sum_f mu_fj,

the same in every post period, where 5 is the coefficients' mean. The
simulator returns the unobserved ``mu``, so ``5 (w - v)'s`` is computed for
every design: the bias that design carries.

Over the DGP that bias averages to zero exactly. Reflecting the covariates,
``Z -> 1 - Z`` and ``mu -> 1 - mu``, and the noise, ``eps -> -eps``, leaves
the DGP's distribution unchanged and maps ``Y_N,jt`` to ``a_t - Y_N,jt`` with
``a_t`` common to all units. MAREX's design is invariant to that map, so the
same ``(w, v)`` is chosen and ``(w - v)'s`` changes sign, since
``(w - v)'1 = 0``. The case checks the invariance directly.

What is pinned
--------------
1. The simulator against R: the largest deviation over the 12 captured panels
   (``Y_N``, ``Y_I``, ``Z``) and over the 1000-seed effect path.
2. Table 2 at the paper's M = 1000, for the constrained design at ``m = 1``
   and ``m = 3``: MAE and RMSE of ``tau_hat_t`` against ``tau_t``.
3. Unbiasedness: the mean signed error, pooled over the five post periods, as
   a t statistic against its Monte Carlo standard error, and the largest
   per-period t.
4. The design's own bias ``5 (w - v)'s``: its mean as a t statistic, and its
   RMS, the bias a single experiment carries.
5. The blank-period check: the slope of ``post_fit.power.placebo_bias``, the
   mean of ``w'Y_N,t - v'Y_N,t`` over the blank periods, on the design's bias.
   The blank periods carry coefficients with the same mean, so the slope is
   one when the check estimates that bias without distortion.
6. The reflection: on 20 panels at each cardinality, the largest change in
   the weights and the largest ``|b + b_reflected|``.

The unconstrained design (``m`` from 1 to 14) takes 0.9 s a panel, about 15
minutes for 1000, and runs only from the command line::

    python -m benchmarks.cases.marex_section5_bias unconstrained

The design is MAREX's ``standard`` objective on the exact backend, which
selects the same designs as the MIQP on SCIP on these panels at a hundredth of
the time.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

M_SIMS = 1000            # the paper's
N_REFLECT = 20
J, T, T0, T_FIT = 15, 30, 25, 20
#: The mean of the Uniform(0, 10) coefficients.
COEF_MEAN = 5.0
_REF_CASE = "marex_scdesign_sim"

_CFG = {"outcome": "y", "unitid": "unit", "time": "time", "T0": T0,
        "blank_periods": T0 - T_FIT, "T_post": T - T0, "design": "standard",
        "backend": "exact", "inference": False, "display_graph": False}


def _long(Y_N: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"unit": np.repeat(np.arange(1, J + 1), T),
                         "time": np.tile(np.arange(1, T + 1), J),
                         "y": Y_N.reshape(-1)})


def _design(Y_N: np.ndarray, setting):
    """Aggregate treated and control weights, and the blank-period gap."""
    from mlsynth import MAREX

    cfg = dict(_CFG)
    if setting == "unconstrained":
        cfg["m_min"], cfg["m_max"] = 1, J - 1
    else:
        cfg["m_eq"] = int(setting)
    res = MAREX({"df": _long(Y_N), **cfg}).fit()
    g = res.globres
    return (np.asarray(g.treated_weights_agg, dtype=float),
            np.asarray(g.control_weights_agg, dtype=float),
            float(res.post_fit.power.placebo_bias))


def _se(x: np.ndarray) -> float:
    return float(x.std(ddof=1) / np.sqrt(x.size))


def _t(x: np.ndarray) -> float:
    return float(x.mean()) / _se(x)


def cell(samples, setting) -> dict:
    """One cardinality over the given samples."""
    err, bias, blank = [], [], []
    for s in samples:
        w, v, gap = _design(s.Y_N, setting)
        err.append(w @ s.Y_I[:, T0:] - v @ s.Y_N[:, T0:] - s.tau[T0:])
        bias.append(COEF_MEAN * (w - v) @ (s.Z.sum(axis=1) + s.mu.sum(axis=1)))
        blank.append(gap)
    err, bias, blank = np.array(err), np.array(bias), np.array(blank)
    pooled = err.mean(axis=1)
    per_period = err.mean(axis=0) / (err.std(axis=0, ddof=1) / np.sqrt(len(err)))
    return {
        "mae": float(np.mean(np.abs(err).mean(axis=1))),
        "rmse": float(np.mean(np.sqrt((err ** 2).mean(axis=1)))),
        "bias_mean": float(pooled.mean()),
        "bias_se": _se(pooled),
        "bias_t": _t(pooled),
        "bias_t_max": float(np.abs(per_period).max()),
        "design_bias_mean": float(bias.mean()),
        "design_bias_se": _se(bias),
        "design_bias_t": _t(bias),
        "design_bias_rms": float(np.sqrt(np.mean(bias ** 2))),
        "design_bias_share": float(np.mean(bias ** 2) / np.mean(err ** 2)),
        "blank_slope": float(np.polyfit(bias, blank, 1)[0]),
        "blank_corr": float(np.corrcoef(bias, blank)[0, 1]),
    }


def _reflection(samples, setting):
    """Largest weight change and largest ``|b + b_reflected|`` under the map."""
    dw, db = 0.0, 0.0
    rng = np.random.default_rng(0)
    for s in samples:
        a = rng.uniform(50.0, 150.0, T)
        w, v, _ = _design(s.Y_N, setting)
        wr, vr, _ = _design(a[None, :] - s.Y_N, setting)
        load = s.Z.sum(axis=1) + s.mu.sum(axis=1)
        reflected = (s.Z.shape[1] + s.mu.shape[1]) - load
        dw = max(dw, float(np.abs(w - wr).max()), float(np.abs(v - vr).max()))
        db = max(db, abs(COEF_MEAN * ((w - v) @ load + (wr - vr) @ reflected)))
    return dw, db


def _simulator_against_r(ref) -> tuple:
    from mlsynth.utils.marex_helpers.simulation import generate_section5_sample

    values, weights = ref["values"], ref["weights"]
    dev = 0.0
    for seed in range(1, int(values["n_panels"]) + 1):
        s = generate_section5_sample(seed)
        for j in range(J):
            for t in range(T):
                dev = max(dev, abs(s.Y_N[j, t] - weights[f"yn_s{seed}_u{j+1}_t{t+1}"]))
            for t in range(T0, T):
                dev = max(dev, abs(s.Y_I[j, t] - weights[f"yi_s{seed}_u{j+1}_t{t+1}"]))
            for k in range(s.Z.shape[1]):
                dev = max(dev, abs(s.Z[j, k] - weights[f"z_s{seed}_k{k+1}_u{j+1}"]))
    return dev, values


def run() -> dict:
    from benchmarks.reference import load_reference
    from mlsynth.utils.marex_helpers.simulation import generate_section5_sample

    out: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dev, values = _simulator_against_r(load_reference(_REF_CASE))
        out["simulator_max_abs_dev"] = dev
        samples = [generate_section5_sample(seed) for seed in range(1, M_SIMS + 1)]
        tau = np.mean([s.tau[T0:] for s in samples], axis=0)
        out["tau_path_max_abs_dev"] = float(max(
            abs(tau[k] - values[f"tau_bar_t{T0 + k + 1}"]) for k in range(T - T0)))

        for setting in (1, 3):
            for key, val in cell(samples, setting).items():
                out[f"{key}_m{setting}"] = val
            dw, db = _reflection(samples[:N_REFLECT], setting)
            out[f"reflection_weight_change_m{setting}"] = dw
            out[f"reflection_bias_sum_m{setting}"] = db
    return out


# Tolerances.
#
# The case is deterministic: the panels are the authors' seeds 1 to 1000 and the
# exact backend has no randomness, so every number repeats exactly.
#
# simulator_max_abs_dev compares the port with the 12 panels R captured, which R
# printed to ten decimals: the measured 5.0e-11 is that rounding, and 1e-9
# admits it and nothing more. tau_path_max_abs_dev compares the 1000-seed effect
# path with the one R printed to six decimals: measured 3.2e-7, tolerance 1e-6.
#
# The Table 2 cells are measured at M = 1000 on the paper's own panels:
# 2.878 / 3.409 at m = 1 and 1.219 / 1.457 at m = 3, against the paper's
# 2.93 / 3.45 and 1.26 / 1.49. The panels are the same, so the gap is the
# design: the paper's design also matches the observed covariates, MAREX's here
# only the pre-period outcomes. Each tolerance brackets the paper's value and
# sits far inside the spread between cardinalities.
#
# bias_t is the pooled mean signed error over its Monte Carlo standard error:
# measured -0.28 at m = 1 and 0.34 at m = 3. The target is zero and the band
# two standard errors, which is the claim: the estimator's error averages to
# zero over the paper's 1000 simulations. bias_t_max is the largest of the five
# per-period t statistics (1.20 and 1.73); three is a two-sided 5% bound
# across five periods. design_bias_t is the same claim for 5 (w - v)'s
# (-0.52 and 1.03).
#
# design_bias_rms is how far one design's bias sits from zero: 1.387 at m = 1
# and 0.414 at m = 3. These are regression guards on the design; the spread
# between the two cardinalities is threefold.
#
# blank_slope is 0.973 at m = 1 and 0.978 at m = 3, with standard errors near
# 0.03 and 0.06; 0.15 is about two and a half of the larger.
#
# The reflection numbers are floating-point zeros: the weights move by at most
# 6.3e-15 and the two biases cancel to 2.5e-14.
EXPECTED = {
    "simulator_max_abs_dev": (0.0, 1e-9),
    "tau_path_max_abs_dev": (0.0, 1e-6),
    "mae_m1": (2.878, 0.06),
    "rmse_m1": (3.409, 0.06),
    "mae_m3": (1.219, 0.05),
    "rmse_m3": (1.457, 0.05),
    "bias_t_m1": (0.0, 2.0),
    "bias_t_m3": (0.0, 2.0),
    "bias_t_max_m1": (0.0, 3.0),
    "bias_t_max_m3": (0.0, 3.0),
    "design_bias_t_m1": (0.0, 2.0),
    "design_bias_t_m3": (0.0, 2.0),
    "design_bias_rms_m1": (1.387, 0.1),
    "design_bias_rms_m3": (0.414, 0.05),
    "blank_slope_m1": (1.0, 0.15),
    "blank_slope_m3": (1.0, 0.15),
    "reflection_weight_change_m1": (0.0, 1e-9),
    "reflection_weight_change_m3": (0.0, 1e-9),
    "reflection_bias_sum_m1": (0.0, 1e-9),
    "reflection_bias_sum_m3": (0.0, 1e-9),
}


if __name__ == "__main__":  # pragma: no cover - the long cell, run by hand
    from mlsynth.utils.marex_helpers.simulation import generate_section5_sample

    settings = [s if s == "unconstrained" else int(s) for s in sys.argv[1:]] or [1, 3]
    warnings.simplefilter("ignore")
    samples = [generate_section5_sample(seed) for seed in range(1, M_SIMS + 1)]
    for setting in settings:
        print(setting, {k: round(v, 4) for k, v in cell(samples, setting).items()},
              flush=True)
