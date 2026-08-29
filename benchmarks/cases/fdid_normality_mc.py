r"""FDID property case: asymptotic normality of the ATT (Li 2023, Prop. 2.1).

Path C (property; scenario 1 -- paper only). Proposition 2.1 states that

.. math::

   \bigl|\Pr\bigl(\sqrt{T_2}(\widehat{ATT} - ATT)/\widehat\sigma \le a\bigr)
     - \Phi(a)\bigr| \to 0 \quad \text{for all } a \in \mathbb{R},

as :math:`T_1, T_2 \to \infty`, with
:math:`\widehat\sigma^2 = T_1^{-1}\sum_{t \le T_1}\hat v_t^2` the pre-period
residual variance of the *selected* donor subset. The true ATT is zero in
the Web Appendix E designs, so the statistic is computable from the public
result surface: ``sqrt(T2) * res.fdid.att / res.fdid.pre_rmse``.

Provenance
----------

Li, Kathleen T. (2023), *"Frontiers: A Simple Forward
Difference-in-Differences Method"*, Marketing Science 43(2), Proposition
2.1 and Online Appendix B; Web Appendix E for the DGPs, shared with
``fdid_table5`` and ``fdid_selection_mc``.

The two statistics
------------------

The proposition's statistic divides by :math:`\widehat\sigma` alone.
mlsynth's reported standard error carries one extra finite-sample term,
:math:`\widehat{se} = \widehat\sigma\sqrt{1 + T_2/T_1}/\sqrt{T_2}`, which
prices in the error from estimating the level shift on :math:`T_1`
pre-periods. The two agree in the limit precisely because Assumption 4(ii)
sends :math:`T_2\log N/T_1 \to 0`, and this case measures both so the
correction's contribution is visible instead of assumed.

Two regimes
-----------

The first holds Assumption 4 by fixing :math:`T_2 = 10` and growing
:math:`T_1`; the statistic should approach a standard normal and the
interval its nominal coverage.

The second violates Assumption 4(ii) by setting :math:`T_2 = T_1/2`, so
:math:`T_2\log N/T_1` never falls. This is not an attempt to contradict the
proposition, which claims nothing when its hypothesis fails; it measures
what the hypothesis is buying. The answer is exact: the paper's statistic
settles at :math:`\sqrt{1 + T_2/T_1} = \sqrt{1.5} \approx 1.2247` instead
of 1, which is the level-shift estimation term the assumption would have
sent to zero. The library's standard error carries that factor already, so
its statistic converges where the paper's does not.

Reading the numbers
-------------------

A Kolmogorov-Smirnov distance measured on ``M`` draws has a floor near
:math:`0.86/\sqrt{M}` even when the statistic is exactly normal -- 0.027 at
``M = 1000`` and 0.038 at ``M = 500``. Distances at that level say the
design cannot resolve any remaining gap, not that the gap is zero, so the
dispersion and coverage keys carry the convergence and the KS keys bound it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from mlsynth import FDID
from mlsynth.utils.fdid_helpers.simulation import simulate_fdid_sample

DGP = 2              # the design whose donor pool is half mismatched
N = 20
M_A4 = 1000          # Assumption-4 regime; KS floor ~ 0.027
M_VIOL = 500         # violation regime; KS floor ~ 0.038, and its headline
                     # metric is a dispersion, which M = 500 already resolves

# Assumption 4 holds: T2 fixed, T1 growing, so T2 log N / T1 -> 0.
GRID_A4 = ((50, 10), (100, 10), (200, 10), (400, 10), (800, 10))
# Assumption 4(ii) fails: T2 = T1 / 2 holds T2 log N / T1 constant.
GRID_VIOL = ((100, 50), (400, 200), (1600, 800))


def _ks(z: np.ndarray) -> float:
    """One-sample Kolmogorov-Smirnov distance from the standard normal."""
    z = np.sort(z)
    empirical = np.arange(1, z.size + 1) / z.size
    return float(np.max(np.abs(empirical - norm.cdf(z))))


def _cell(T1: int, T2: int, M: int, seed: int = 0) -> dict:
    """Draw ``M`` panels and studentise the ATT two ways.

    ``z_paper`` is Proposition 2.1's statistic. ``z_mlsynth`` is the same
    quantity through the library's reported standard error, which adds the
    ``sqrt(1 + T2/T1)`` finite-sample factor. Coverage is of the 95%
    interval the estimator returns, against the designs' true ATT of zero.
    """
    z_paper, z_mlsynth, covered = [], [], []
    for j in range(M):
        rng = np.random.default_rng(seed + j)
        s = simulate_fdid_sample(dgp=DGP, N=N, T1=T1, T2=T2, rng=rng)
        r = FDID({"df": s.df, "outcome": "y", "treat": "treat",
                  "unitid": "unit", "time": "time",
                  "display_graphs": False, "verbose": False}).fit().fdid
        z_paper.append(np.sqrt(T2) * r.att / r.pre_rmse)
        z_mlsynth.append(r.att / r.att_se)
        covered.append(r.ci[0] <= 0.0 <= r.ci[1])
    z_paper = np.asarray(z_paper)
    return {
        "sd": float(z_paper.std()),
        "sd_mlsynth": float(np.std(z_mlsynth)),
        "ks": _ks(z_paper),
        "ks_mlsynth": _ks(np.asarray(z_mlsynth)),
        "cov95": float(np.mean(covered)),
    }


def _decreasing(values, slack: float = 0.02) -> float:
    v = list(values)
    return float(all(b <= a + slack for a, b in zip(v, v[1:])))


def _increasing(values, slack: float = 0.02) -> float:
    v = list(values)
    return float(all(b >= a - slack for a, b in zip(v, v[1:])))


def run() -> dict:
    out: dict[str, float] = {}

    # --- Assumption 4 holds ----------------------------------------------
    a4 = {T1: _cell(T1, T2, M_A4) for T1, T2 in GRID_A4}
    for T1, m in a4.items():
        out[f"a4_sd_T{T1}"] = m["sd"]
        out[f"a4_cov95_T{T1}"] = m["cov95"]
    out["a4_ks_T50"] = a4[50]["ks"]
    out["a4_ks_T800"] = a4[800]["ks"]
    # The convergence, as geometry: dispersion falls to 1 and coverage rises
    # to nominal, both monotonically.
    out["a4_sd_decreasing"] = _decreasing(m["sd"] for m in a4.values())
    out["a4_cov_increasing"] = _increasing(m["cov95"] for m in a4.values())

    # --- Assumption 4(ii) fails -------------------------------------------
    viol = {T1: _cell(T1, T2, M_VIOL) for T1, T2 in GRID_VIOL}
    for T1, T2 in GRID_VIOL:
        out[f"viol_sd_T{T1}"] = viol[T1]["sd"]
        out[f"viol_sd_mlsynth_T{T1}"] = viol[T1]["sd_mlsynth"]
    # The paper's statistic settles at sqrt(1 + T2/T1) instead of 1, which is
    # the level-shift estimation term Assumption 4(ii) would have sent to zero.
    # Dividing the measured dispersion by that predicted inflation lands on 1,
    # which identifies the gap instead of merely reporting it -- and the
    # library's standard error carries exactly that factor, so its statistic
    # converges where the paper's does not.
    out["viol_sd_over_predicted_T1600"] = (
        viol[1600]["sd"] / np.sqrt(1.0 + 800.0 / 1600.0))
    out["viol_ks_mlsynth_T1600"] = viol[1600]["ks_mlsynth"]
    out["viol_cov95_T1600"] = viol[1600]["cov95"]
    return out


# Tolerances. The case is seeded end to end, so it reproduces these exactly;
# the tolerances size the spread a different set of draws would give. A
# dispersion estimated on M draws has standard error about sd/sqrt(2M), so
# 0.022 at M = 1000 and 0.032 at M = 500; a coverage rate near 0.95 has
# 0.0069 and 0.0097. Tolerances are roughly three of those.
EXPECTED = {
    # --- Assumption 4 holds: the statistic converges on a standard normal ---
    # Dispersion falls to 1 from above. The excess at small T1 is the
    # pre-period residual variance being estimated on the very subset chosen
    # to minimise it, which biases sigma-hat down and the statistic out.
    "a4_sd_T50": (1.329, 0.07),
    "a4_sd_T100": (1.130, 0.07),
    "a4_sd_T200": (1.112, 0.07),
    "a4_sd_T400": (1.019, 0.07),
    "a4_sd_T800": (0.998, 0.07),
    # Coverage of the reported 95% interval climbs to nominal alongside it.
    "a4_cov95_T50": (0.889, 0.025),
    "a4_cov95_T100": (0.931, 0.025),
    "a4_cov95_T200": (0.938, 0.025),
    "a4_cov95_T400": (0.947, 0.025),
    "a4_cov95_T800": (0.952, 0.025),
    # The distributional distance itself, at the ends of the grid. The value
    # at T1 = 800 is close to the M = 1000 floor of 0.027, so it bounds the
    # remaining gap and does not measure it.
    "a4_ks_T50": (0.075, 0.02),
    "a4_ks_T800": (0.035, 0.02),
    "a4_sd_decreasing": (1.0, 0.0),
    "a4_cov_increasing": (1.0, 0.0),

    # --- Assumption 4(ii) fails: T2 = T1/2 --------------------------------
    # The paper's statistic settles at sqrt(1 + T2/T1) = sqrt(1.5) = 1.2247
    # instead of 1, and the measured 1.224 at the largest cell identifies that
    # constant to three decimals. The library's standard error divides by the
    # same factor, so its statistic converges to 1 across the same grid.
    "viol_sd_T100": (1.401, 0.10),
    "viol_sd_T400": (1.237, 0.10),
    "viol_sd_T1600": (1.224, 0.10),
    "viol_sd_mlsynth_T100": (1.144, 0.10),
    "viol_sd_mlsynth_T400": (1.010, 0.10),
    "viol_sd_mlsynth_T1600": (1.000, 0.10),
    # The identification, stated directly: dividing out the predicted
    # inflation lands on 1.
    "viol_sd_over_predicted_T1600": (1.000, 0.10),
    # And the corrected statistic is at the M = 500 KS floor of 0.038, with
    # coverage within a standard error and a half of nominal.
    "viol_ks_mlsynth_T1600": (0.040, 0.02),
    "viol_cov95_T1600": (0.936, 0.035),
}
