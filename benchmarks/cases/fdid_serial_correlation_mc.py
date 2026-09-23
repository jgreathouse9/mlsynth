r"""FDID property case: the standard error under serially correlated errors.

Path C (property; scenario 1 -- paper only). Measures where Proposition
2.1's variance formula stops applying, and by how much.

Li's standard error is built from the *marginal* variance of the
parallel-trends residual, :math:`\sigma^2 = \mathbb{E}[v_t^2]`. The
estimator's sampling error is
:math:`\bar v_{\text{post}} - \bar v_{\text{pre}}`, a difference of two block
means, and a block mean's variance is governed by the *long-run* variance
:math:`\sum_k \gamma_k`. The two coincide exactly when :math:`v_t` is
serially uncorrelated -- which Online Appendix A's Assumption 2(ii) (iid
:math:`\epsilon_{it}`) and Assumption 3(i) (iid :math:`f_t`) impose, so
under the appendix's own conditions the formula is right.

Assumption 2.1 in the main text asks only that :math:`v_t` be "a weakly
dependent process with zero mean and finite variance", and the appendix
remarks that both iid assumptions "can be easily relaxed to weakly dependent
processes". The estimator survives that relaxation. The standard error does
not: nothing in :math:`\Omega_1 + \Omega_2` estimates an autocovariance.

Provenance
----------

Li, Kathleen T. (2023), *"Frontiers: A Simple Forward
Difference-in-Differences Method"*, Marketing Science 43(2), Proposition 2.1
and Online Appendix A (Assumptions 2.1, 2, 3 and the relaxation remark).

Why the paper's own designs cannot show this
--------------------------------------------

The four Web Appendix E DGPs have serially correlated factors, so a
mismatched subset does give a serially correlated residual. But the residual
is :math:`\varepsilon_{tr,t} - \bar\varepsilon_{Ut} + (c_0 - \bar c_U)
\mathbf{1}'f_t`, and at the subset the forward search selects
:math:`\bar c_U = c_0`, killing the factor term and leaving an iid residual.
The design is self-protecting exactly where the standard error needs
testing, which is Assumption 3(ii)'s :math:`\bar\lambda_U = 0` doing its
work. So this case uses
:func:`~mlsynth.utils.fdid_helpers.simulation.simulate_fdid_serial_sample`,
which is DGP 2 with the treated unit's shock made AR(1) at unit marginal
variance -- serial dependence that survives the selection.

What is measured
----------------

Coverage of the reported 95% interval and the dispersion of the studentised
ATT, across :math:`\rho`, at :math:`T_1 = 400` so that the post-selection
effect the :doc:`normality case <fdid_normality_mc>` measures is long gone
and only the serial correlation is left.

The same draws are then re-fitted with ``inference="hac"``, which estimates
the residual's autocovariances on the pre-period and puts them through the
exact finite-block variance of both means. Selection never consults a
standard error, so the two fits share their subsets and point estimates and
differ only in the interval.

The dispersion is also scored against a closed-form prediction,
:func:`~mlsynth.utils.fdid_helpers.population.long_run_inflation`, so the
gap is identified and not merely reported. That prediction is first-order:
it prices the long-run variance of the two block means, and leaves out the
downward bias in :math:`\widehat\sigma^2` from demeaning an autocorrelated
series on :math:`T_1` periods, which is why the measured dispersion runs a
few per cent above it and by more as :math:`\rho` rises.
"""

from __future__ import annotations

import numpy as np

from mlsynth import FDID
from mlsynth.utils.fdid_helpers.inference import hac_lag
from mlsynth.utils.fdid_helpers.population import long_run_inflation
from mlsynth.utils.fdid_helpers.simulation import simulate_fdid_serial_sample

N = 20
T1 = 400
T2 = 10
M = 600            # standard error on a coverage rate near 0.9 is 0.012
RHOS = (0.0, 0.3, 0.5, 0.7, 0.9)


def _cell(rho: float, seed: int = 0) -> dict:
    """Coverage and dispersion of the reported inference at one ``rho``.

    Both standard errors are measured on the same draws. Selection does not
    consult a standard error, so the two fits differ only in the interval.
    """
    covered, z, selected = [], [], []
    covered_hac, z_hac, se_ratio = [], [], []
    for j in range(M):
        rng = np.random.default_rng(seed + j)
        s = simulate_fdid_serial_sample(rho=rho, N=N, T1=T1, T2=T2, rng=rng)
        cfg = {"df": s.df, "outcome": "y", "treat": "treat",
               "unitid": "unit", "time": "time",
               "display_graphs": False, "verbose": False}
        r = FDID(cfg).fit().fdid
        h = FDID({**cfg, "inference": "hac"}).fit().fdid
        covered.append(r.ci[0] <= 0.0 <= r.ci[1])     # true ATT is zero
        z.append(r.att / r.att_se)
        selected.append(len(r.selected_names))
        covered_hac.append(h.ci[0] <= 0.0 <= h.ci[1])
        z_hac.append(h.att / h.att_se)
        se_ratio.append(h.att_se / r.att_se)
    return {
        "cov95": float(np.mean(covered)),
        "sd": float(np.std(z)),
        "n_selected": float(np.mean(selected)),
        "cov95_hac": float(np.mean(covered_hac)),
        "sd_hac": float(np.std(z_hac)),
        "se_ratio": float(np.mean(se_ratio)),
    }


def _decreasing(values, slack: float = 0.01) -> float:
    v = list(values)
    return float(all(b <= a + slack for a, b in zip(v, v[1:])))


def run() -> dict:
    out: dict[str, float] = {}
    cells = {rho: _cell(rho) for rho in RHOS}
    for rho, m in cells.items():
        tag = f"{rho:.1f}".replace(".", "")
        out[f"cov95_rho{tag}"] = m["cov95"]
        out[f"sd_rho{tag}"] = m["sd"]
        # Measured dispersion over the closed-form prediction. The prediction
        # uses the population subset size N//2; n_selected below records what
        # the search actually returned, so a drift there stays visible.
        out[f"sd_over_predicted_rho{tag}"] = m["sd"] / long_run_inflation(
            rho, n=N // 2, T1=T1, T2=T2)
        out[f"cov95_hac_rho{tag}"] = m["cov95_hac"]
        out[f"sd_hac_rho{tag}"] = m["sd_hac"]
        out[f"se_ratio_hac_rho{tag}"] = m["se_ratio"]
    out["hac_lag"] = float(hac_lag(T1, T2))
    out["cov_decreasing"] = _decreasing(m["cov95"] for m in cells.values())
    out["n_selected_rho00"] = cells[0.0]["n_selected"]
    out["n_selected_rho09"] = cells[0.9]["n_selected"]
    return out


# Tolerances. Seeded end to end, so these reproduce exactly; the tolerances
# size the spread a different set of draws would give. At M = 600 a coverage
# rate near 0.9 has standard error 0.012 and a dispersion has sd/sqrt(2M),
# about 0.03 at sd = 1 and 0.08 at sd = 2.8. Tolerances are roughly three of
# those.
EXPECTED = {
    # --- Coverage of the reported 95% interval collapses in rho -----------
    # The estimator stays consistent throughout; only the interval fails.
    "cov95_rho00": (0.942, 0.04),
    "cov95_rho03": (0.877, 0.04),
    "cov95_rho05": (0.787, 0.04),
    "cov95_rho07": (0.678, 0.04),
    "cov95_rho09": (0.533, 0.04),
    "cov_decreasing": (1.0, 0.0),

    # --- The studentised ATT spreads out by the same factor ---------------
    "sd_rho00": (1.016, 0.10),
    "sd_rho03": (1.317, 0.10),
    "sd_rho05": (1.608, 0.10),
    "sd_rho07": (2.047, 0.10),
    "sd_rho09": (2.790, 0.12),

    # --- and the closed form says why -------------------------------------
    # Dividing the measured dispersion by long_run_inflation leaves a ratio
    # that barely moves: 1.016 at rho = 0 (the post-selection residual the
    # normality case measures at this T1, unrelated to serial correlation)
    # rising only to 1.058 at rho = 0.9. So the long-run variance accounts
    # for essentially the whole 2.79x blow-up, and what it does not account
    # for is the downward bias in sigma-hat from demeaning an autocorrelated
    # series -- which grows in rho, as the drift across these five does.
    "sd_over_predicted_rho00": (1.016, 0.05),
    "sd_over_predicted_rho03": (1.019, 0.05),
    "sd_over_predicted_rho05": (1.024, 0.05),
    "sd_over_predicted_rho07": (1.036, 0.05),
    "sd_over_predicted_rho09": (1.058, 0.05),

    # --- inference="hac" prices the autocovariances in --------------------
    # The same draws, the same selected subsets, the same point estimates.
    # Coverage holds across the whole range where the analytic interval
    # collapses, and the studentised statistic goes back to unit dispersion.
    # The lag is min(T2 - 1, T1 // 10) = min(9, 40) = 9, so the post block's
    # sum is exhaustive: lag k enters it with weight 1 - k/T2, zero at k = 10.
    "hac_lag": (9.0, 0.0),
    "cov95_hac_rho00": (0.947, 0.04),
    "cov95_hac_rho03": (0.935, 0.04),
    "cov95_hac_rho05": (0.938, 0.04),
    "cov95_hac_rho07": (0.933, 0.04),
    "cov95_hac_rho09": (0.920, 0.04),

    "sd_hac_rho00": (0.987, 0.10),
    "sd_hac_rho03": (1.049, 0.10),
    "sd_hac_rho05": (1.061, 0.10),
    "sd_hac_rho07": (1.077, 0.10),
    "sd_hac_rho09": (1.116, 0.10),

    # What it costs where there is nothing to correct: 2.8 per cent of width
    # at rho = 0, which coverage does not notice (0.947 against 0.942). What
    # it buys at rho = 0.9 is an interval 2.52 times wider -- close to the
    # 2.79 the dispersion column says is missing, the shortfall being the
    # lag-9 truncation of a sequence still at 0.9^10 = 0.35.
    "se_ratio_hac_rho00": (1.028, 0.10),
    "se_ratio_hac_rho03": (1.268, 0.10),
    "se_ratio_hac_rho05": (1.532, 0.10),
    "se_ratio_hac_rho07": (1.919, 0.10),
    "se_ratio_hac_rho09": (2.517, 0.12),

    # The prediction uses the population matched-half size N//2 = 10; the
    # search returns fewer. It makes no difference at rho = 0, where the
    # inflation is 1 for any n, and little elsewhere since the 1/n term is
    # small beside the unit-variance shock. Pinned so a drift stays visible.
    "n_selected_rho00": (8.43, 0.5),
    "n_selected_rho09": (8.66, 0.5),
}
