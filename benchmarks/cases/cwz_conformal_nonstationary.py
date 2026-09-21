r"""CWZ conformal test (JASA 2021): what trending factors do to its exactness.

Path B. Chernozhukov, Wuthrich & Zhu (2021), online supplement Section G and
Tables I.2, I.3 and I.4. The companion case ``cwz_conformal_mc`` pins the
stationary size study at one cell, where the test is exact and there is nothing
to choose between specifications. This case covers the half of the supplement
where that stops being true.

The claim under test
--------------------

The moving-block conformal test is exact when the residuals are exchangeable,
and close to exact when they are merely stationary -- that is the paper's
Theorem D.1 and the reason the method is advertised as robust to a misspecified
counterfactual. The supplement then removes stationarity: the second factor
becomes :math:`F_{2t} \sim N(t, 1)` instead of :math:`N(0, 1)`, so every donor
carries a deterministic trend scaled by its own loading. Tables I.3 and I.4
report what happens, and the answer is that robustness to misspecification does
not survive:

======  ============================  ==========  ==============
DGP     treated unit's weights        stationary  trending
======  ============================  ==========  ==============
1       ``(1/J, ..., 1/J)``                 0.11            0.11
2       ``(1/3, 1/3, 1/3, 0, ...)``         0.11            0.11
3       ``-(1/J, ..., 1/J)``                0.10            0.98
4       ``(1, -1, 0, ..., 0)``              0.11            0.62
======  ============================  ==========  ==============

Read at :math:`T_0 = 100`, :math:`J = 20`, :math:`\rho = 0.6`, nominal level
0.1, synthetic-control column. DGPs 1 and 2 are inside the simplex, so the
control can represent the treated unit's trend and the test holds its level.
DGPs 3 and 4 are not -- DGP3 asks for negative weights and DGP4 for a
difference -- so the fitted control carries a trend the treated unit does not,
the pre-period residual is a ramp, and the cyclic shifts of a ramp are nothing
like the ramp. The test rejects a true null 98% of the time.

The direction is the part to hold on to, because it inverts the usual reading
of a size table. Under DGP3 the distortion *grows* with the pre-period, 0.53 at
:math:`T_0 = 20` to 0.83 at 50 to 0.98 at 100: more data makes the test worse,
since a longer window gives the uncancelled trend more room. Under DGP4 it
*falls* with the donor pool, 0.62 at :math:`J = 20` to 0.34 at 50 to 0.20 at
100, because a wider pool lets the simplex approximate a difference it cannot
represent exactly. Neither movement is Monte Carlo error and both are pinned.

Why the stationary cells are here too
-------------------------------------

A size of 0.98 is only evidence about trending factors if the same cell is
exact without them. The stationary arm runs the identical DGP, estimator and
seed policy with :math:`F_{2t} \sim N(0, 1)`, and Table I.2 says it should come
back to the nominal level. Without that pair the finding would be consistent
with the port being broken.

Exact size is arithmetic, not approximation
-------------------------------------------

With one post period the reference set is the :math:`T = T_0 + 1` cyclic
shifts, so the p-value is uniform on :math:`\{1/T, ..., T/T\}` under
exchangeability and the size is :math:`\lfloor \alpha T \rfloor / T` exactly --
0.0952 at :math:`T_0 = 20`, 0.0980 at 50, 0.0990 at 100. That is why Table I.1's
entries are 0.09 and 0.10 and never 0.11, and it is pinned here with no
tolerance at all: it is a property of the reference set's size, and a port that
enumerated the wrong number of shifts would miss it.

Power against the oracle bound
------------------------------

Figure I.2's setting: :math:`T_0 = 19`, :math:`J = 50`, :math:`\rho = 0.6`,
stationary, alternatives 0 through 6. The oracle bound is the power of a test
that knows the marginal law of :math:`u_t`, which here is standard normal, so it
is closed form and needs no reference run::

    oracle(theta) = P(|N(theta, 1)| > z) with z = Phi^{-1}(1 - alpha/2)

Under DGP1 the paper reports power close to that bound, and it is: the largest
gap over the seven alternatives is 0.027. Under DGP4 the constraints the
synthetic control imposes are wrong, and the supplement's conclusion is that the
extra restrictions cost power when they do not hold. They cost a great deal --
at an effect of four standard deviations the oracle rejects 99.1% of the time,
correct specification 98.2%, and DGP4 63.2%.

The supplement prints those power curves as a figure, not a table, so the DGP4
levels below are a calibration and not a published target; what the paper states
in words, and what is pinned sharply, is the ordering. The DGP1 cells are
different: there the target is the oracle bound, which is closed form.

Provenance
----------

* Paper: Chernozhukov, Wuthrich & Zhu (2021), JASA 116(536):1849-1864.
* Supplement: ``online_supplements/supplement_conformal_final.pdf``, Section G
  and Tables I.2-I.4, and ``replication_package_final/``'s
  ``simulations_conformal_final.R``, whose ``sim()`` is ported in
  ``benchmarks/cwz_common.py`` and already validated against a live
  ``scinference`` run by ``cwz_conformal_mc``.
* No R is needed: the targets are printed and the oracle is closed form.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from benchmarks.cwz_common import moving_block_pvalue, simulate_conformal_panel

ALPHA = 0.1
N_REPS = 500
SEED = 20260921

# Table I.4 / I.2, synthetic-control column, rho = 0.6. Keyed (dgp, T0, J).
TRENDING = {
    (1, 20, 20): 0.10, (1, 50, 20): 0.11, (1, 100, 20): 0.11,
    (1, 100, 50): 0.12, (1, 100, 100): 0.11,
    (2, 20, 20): 0.12, (2, 50, 20): 0.11, (2, 100, 20): 0.11,
    (2, 100, 50): 0.11, (2, 100, 100): 0.11,
    (3, 20, 20): 0.53, (3, 50, 20): 0.83, (3, 100, 20): 0.98,
    (3, 100, 50): 0.97, (3, 100, 100): 0.97,
    (4, 20, 20): 0.21, (4, 50, 20): 0.39, (4, 100, 20): 0.62,
    (4, 100, 50): 0.34, (4, 100, 100): 0.20,
}
STATIONARY = {
    (1, 100, 20): 0.11, (2, 100, 20): 0.11,
    (3, 50, 20): 0.10, (3, 100, 20): 0.10,
    (4, 50, 20): 0.11, (4, 100, 20): 0.11,
}

POWER_T0, POWER_J, POWER_RHO = 19, 50, 0.6
ALTERNATIVES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def oracle_power(alternative: float, alpha: float = ALPHA) -> float:
    """The supplement's ``oracle.power``: a test that knows the law of ``u_t``.

    ``pfoldnorm(qfoldnorm(1 - alpha), mean = alt, lower.tail = FALSE)`` in
    VGAM, which is the probability that a folded ``N(alt, 1)`` exceeds the
    ``1 - alpha`` quantile of a folded standard normal -- and that quantile is
    the two-sided normal critical value.
    """
    z = norm.ppf(1.0 - alpha / 2.0)
    return float(norm.sf(z - alternative) + norm.cdf(-z - alternative))


def exact_size(pre_periods: int, post_periods: int = 1, alpha: float = ALPHA) -> float:
    """``floor(alpha * T) / T``: the level a rank test on ``T`` shifts can hit."""
    n_shifts = pre_periods + post_periods
    return math.floor(alpha * n_shifts) / n_shifts


def _rejection_rate(dgp: int, pre_periods: int, n_donors: int, *,
                    stationary: bool, alternative: float, rho: float) -> float:
    # Entropy as a tuple, so no arithmetic can collide two cells onto one
    # stream; every cell is an independent draw and the rates are comparable.
    rng = np.random.default_rng(
        [SEED, dgp, pre_periods, n_donors,
         int(round(alternative * 10)), int(stationary)]
    )
    hits = 0
    for _ in range(N_REPS):
        y, Y0 = simulate_conformal_panel(
            dgp, pre_periods=pre_periods, post_periods=1, n_donors=n_donors,
            rho_donor=rho, rho_treated=rho, stationary=stationary,
            alternative=alternative, rng=rng,
        )
        hits += moving_block_pvalue(y, Y0, pre_periods) <= ALPHA
    return hits / N_REPS


def run() -> dict:
    out: dict = {}

    trending = {
        key: _rejection_rate(*key, stationary=False, alternative=0.0, rho=0.6)
        for key in TRENDING
    }
    stationary = {
        key: _rejection_rate(*key, stationary=True, alternative=0.0, rho=0.6)
        for key in STATIONARY
    }

    # The two specifications the simplex can represent hold their level under a
    # trend; the two it cannot do not. Aggregated so one drifting cell shows.
    well = [abs(trending[k] - v) for k, v in TRENDING.items() if k[0] in (1, 2)]
    out["trend_size_max_gap_specified"] = round(float(max(well)), 3)
    out["trend_size_max_specified"] = round(
        float(max(trending[k] for k in TRENDING if k[0] in (1, 2))), 3)

    # DGP3 under a trend: the distortion grows with the pre-period.
    for T0 in (20, 50, 100):
        out[f"trend_dgp3_T{T0}"] = round(trending[(3, T0, 20)], 3)
    out["trend_dgp3_rises_with_T0"] = float(
        trending[(3, 20, 20)] < trending[(3, 50, 20)] < trending[(3, 100, 20)])

    # DGP4 under a trend: the distortion falls as the donor pool widens.
    for J in (20, 50, 100):
        out[f"trend_dgp4_J{J}"] = round(trending[(4, 100, J)], 3)
    out["trend_dgp4_falls_with_J"] = float(
        trending[(4, 100, 100)] < trending[(4, 100, 50)] < trending[(4, 100, 20)])

    # The same cells without the trend, which is what makes the above a finding
    # about trending factors and not about the port.
    out["stat_size_max_gap"] = round(
        float(max(abs(stationary[k] - v) for k, v in STATIONARY.items())), 3)
    out["stat_dgp3_T100"] = round(stationary[(3, 100, 20)], 3)
    out["stat_dgp4_T100"] = round(stationary[(4, 100, 20)], 3)
    # and the pair the whole case turns on
    out["dgp3_T100_trend_minus_stat"] = round(
        trending[(3, 100, 20)] - stationary[(3, 100, 20)], 3)

    # Exact size is arithmetic on the number of shifts, not an approximation.
    out["exact_size_T20"] = round(exact_size(20), 4)
    out["exact_size_T50"] = round(exact_size(50), 4)
    out["exact_size_T100"] = round(exact_size(100), 4)
    out["exact_size_below_nominal"] = float(
        all(exact_size(t) <= ALPHA for t in (19, 20, 50, 100)))

    # Power against the oracle bound, Figure I.2's setting.
    for alt in ALTERNATIVES:
        out[f"oracle_alt{int(alt)}"] = round(oracle_power(alt), 3)
    gaps1, gaps4 = [], []
    for alt in ALTERNATIVES:
        p1 = _rejection_rate(1, POWER_T0, POWER_J, stationary=True,
                             alternative=alt, rho=POWER_RHO)
        p4 = _rejection_rate(4, POWER_T0, POWER_J, stationary=True,
                             alternative=alt, rho=POWER_RHO)
        if alt in (2.0, 4.0):
            out[f"power_dgp1_alt{int(alt)}"] = round(p1, 3)
            out[f"power_dgp4_alt{int(alt)}"] = round(p4, 3)
        gaps1.append(oracle_power(alt) - p1)
        gaps4.append(oracle_power(alt) - p4)
    # Under correct specification the test comes close to the bound; under DGP4
    # the constraints are wrong and cost power, which is the supplement's point.
    out["power_max_gap_to_oracle_dgp1"] = round(float(max(gaps1)), 3)
    out["power_max_gap_to_oracle_dgp4"] = round(float(max(gaps4)), 3)
    out["dgp4_loses_power_to_dgp1"] = float(max(gaps4) > max(gaps1))
    # The ordering the supplement states in words, at one alternative, as a
    # single number: how much more of the oracle's power the wrong constraints
    # give up than the right ones do.
    out["power_loss_ratio_alt4"] = round(
        float((oracle_power(4.0) - out["power_dgp4_alt4"])
              / max(oracle_power(4.0) - out["power_dgp1_alt4"], 1e-9)), 1)
    return out


# 500 draws a cell against the supplement's 5,000. At a rate near 0.1 the
# standard error is 0.013 and near 0.5 it is 0.022, so the size bands below are
# 0.05 -- under four of the first and between two and three of the second.
# The closed-form quantities carry no tolerance at all.
EXPECTED = {
    # Tables I.2 and I.4: the specifications the simplex can represent hold
    # their level whether or not the factors trend.
    "trend_size_max_gap_specified": (0.02, 0.05),
    "trend_size_max_specified": (0.13, 0.05),
    "stat_size_max_gap": (0.02, 0.05),
    # Table I.4, DGP3: 0.53 / 0.83 / 0.98 as the pre-period grows.
    "trend_dgp3_T20": (0.53, 0.08),
    "trend_dgp3_T50": (0.83, 0.08),
    "trend_dgp3_T100": (0.98, 0.06),
    "trend_dgp3_rises_with_T0": (1.0, 0.0),
    # Table I.4, DGP4: 0.62 / 0.34 / 0.20 as the donor pool widens.
    "trend_dgp4_J20": (0.62, 0.08),
    "trend_dgp4_J50": (0.34, 0.08),
    "trend_dgp4_J100": (0.20, 0.06),
    "trend_dgp4_falls_with_J": (1.0, 0.0),
    # Table I.2, the same cells without the trend.
    "stat_dgp3_T100": (0.10, 0.05),
    "stat_dgp4_T100": (0.11, 0.05),
    # 0.98 against 0.10 at one cell, which is the case in one number.
    "dgp3_T100_trend_minus_stat": (0.88, 0.10),
    # Arithmetic on the reference set, so no tolerance.
    "exact_size_T20": (0.0952, 1e-9),
    "exact_size_T50": (0.0980, 1e-9),
    "exact_size_T100": (0.0990, 1e-9),
    "exact_size_below_nominal": (1.0, 0.0),
    # The oracle bound, closed form.
    "oracle_alt0": (0.100, 1e-9),
    "oracle_alt1": (0.264, 1e-3),
    "oracle_alt2": (0.639, 1e-3),
    "oracle_alt3": (0.912, 1e-3),
    "oracle_alt4": (0.991, 1e-3),
    "oracle_alt5": (1.000, 1e-3),
    "oracle_alt6": (1.000, 1e-3),
    # Figure I.2. Under DGP1 the target is the oracle bound itself, since the
    # supplement says the test comes close to it.
    "power_dgp1_alt2": (0.639, 0.12),
    "power_dgp1_alt4": (0.991, 0.10),
    "power_max_gap_to_oracle_dgp1": (0.03, 0.10),
    # Under DGP4 the supplement prints a figure, so these are calibration: the
    # levels record what the wrong constraints cost, and the two entries below
    # them are the claim the paper makes in words.
    "power_dgp4_alt2": (0.246, 0.10),
    "power_dgp4_alt4": (0.632, 0.10),
    "power_max_gap_to_oracle_dgp4": (0.51, 0.12),
    "dgp4_loses_power_to_dgp1": (1.0, 0.0),
    "power_loss_ratio_alt4": (40.0, 25.0),
}
