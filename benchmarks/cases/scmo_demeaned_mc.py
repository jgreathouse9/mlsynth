"""Path B benchmark: demeaned multi-outcome SCM (Tian-Lee-Panchenko 2026, Table B.1).

Reproduces the Online Appendix simulation, the one that motivates matching on
demeaned outcomes (Appendix B.1.1) and reports the size of the permutation test.
The DGP gives each outcome a large level of its own and places the treated unit
with a parameter ``d``: at ``d = 1`` it is as likely as a donor to take an
extreme predictor value, so it often falls outside the donors' convex hull; as
``d`` falls it moves inside and the pre-treatment fit improves.

Four estimators, all matching on the two observed predictors alongside the
outcomes, as the authors' ``Simulation2.R`` does:

* ``conventional`` -- one outcome, matched in levels;
* ``demeaned`` -- one outcome, matched after centering each unit on its own
  pre-treatment mean (the paper's "Multi-Outcome SC (K = 1)" column);
* ``multi3`` / ``multi10`` -- the same, on three and ten related outcomes.

Four statistics per cell: the treated unit's pre-treatment RMSPE ("pre"), the
average absolute post-period gap ("bias"), the standard deviation of that signed
gap ("sd"), and the rejection rate of the 10% permutation test ("rej") -- the
share of draws in which the treated unit's post-to-pre-treatment RMSPE ratio
ranks in the top three of the thirty units. Under the null DGP (``tau = 0``) a
rejection rate above 0.10 is size distortion.

Provenance
----------
* DGP: :func:`mlsynth.utils.scmo_helpers.simulation.simulate_tian_demeaned` --
  eq. B.2.5 of the Online Appendix, the ``Simulation2.R`` model (N = 30, r = 2
  observed and f = 4 unobserved predictors, one post-period, tau = 0,
  outcome-level means drawn N(0, 10^2)).
* Inference: ``inference="placebo"``, the Abadie permutation test on the
  post-to-pre-treatment RMSPE ratio (Online Appendix B.3.3).
* Headline: Tian-Lee-Panchenko (2026) Online Appendix Table B.1, all 144 cells.
  At ``d = 1, T0 = 5``, reading pre / bias / sd / rej across the four
  estimators: 1.65 / 1.94 / 2.91 / 0.10, 0.51 / 1.43 / 1.81 / 0.10,
  0.82 / 1.32 / 1.67 / 0.10, 0.99 / 1.22 / 1.54 / 0.10.
* The appendix draws three findings from the table, each pinned here as a count
  over the nine (d, T0) settings or the three T0 at one d:

  1. demeaning improves the fit and lowers the bias, most at ``d = 1``;
  2. the permutation test is close to its nominal 10% at ``d = 1``, and its size
     distortion grows as ``d`` falls, where the treated unit's pre-treatment
     RMSPE is small and its ratio large; demeaning alleviates that distortion;
  3. more pre-treatment periods or more outcomes also reduce the distortion.

  The paper uses 5,000 draws per setting; we use M = 100, so the tolerances are
  wide, and widest on the conventional arm: with the treated unit outside the
  donors' convex hull its gap is heavy-tailed, and at this draw count neither
  its mean nor its dispersion settles down (the normal-theory standard error,
  0.29 on a gap SD of 2.9, understates what the tails do). The counts are what
  carry the findings; the cells are pinned to show the levels agree.
"""
from __future__ import annotations

import warnings

import numpy as np

M = 100
SEED = 321
T0_VALUES = (5, 10, 20)
D_VALUES = (1.0, 0.5, 0.0)
ALPHA = 0.1               # the 10% test: top 3 of 30 units
N_PREDICTORS = 2
# (arm, number of outcomes, demeaned)
ARMS = (
    ("conventional", 1, False),
    ("demeaned", 1, True),
    ("multi3", 3, True),
    ("multi10", 10, True),
)
_D_TAG = {1.0: "d10", 0.5: "d05", 0.0: "d00"}


def _spec(K: int, T0: int) -> dict:
    """Stack the ``K`` outcomes over the pre-periods and pin the two observed
    predictors to a single period, so each enters the matching matrix once."""
    return {"year": list(range(T0)), "vars": {
        **{f"y{k}": f"y{k}" for k in range(K)},
        **{f"z{j}": {"column": f"z{j}", "year": 0} for j in range(N_PREDICTORS)}}}


def _grid() -> dict:
    from mlsynth import SCMO
    from mlsynth.utils.scmo_helpers.simulation import simulate_tian_demeaned, to_panel

    out: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d in D_VALUES:
            for T0 in T0_VALUES:
                rng = np.random.default_rng(SEED)
                acc = {arm: [] for arm, _K, _dm in ARMS}
                for _ in range(M):
                    Ys, N, TT, treated, Z = simulate_tian_demeaned(
                        rng, T0, max(K for _a, K, _d in ARMS), d=d)
                    for arm, K, demean in ARMS:
                        df = to_panel(Ys[:K], N, TT, treated, predictors=Z)
                        fit = SCMO({
                            "df": df, "outcome": "y0", "treat": "treat",
                            "unitid": "unit", "time": "time", "spec": _spec(K, T0),
                            "schemes": ["concatenated"], "demean": demean,
                            "inference": "placebo", "display_graphs": False,
                        }).fit()._primary
                        acc[arm].append((fit.pre_rmse,
                                         float(np.asarray(fit.gap)[-1]),
                                         float(fit.p_value)))
                for arm, _K, _dm in ARMS:
                    a = np.asarray(acc[arm])
                    out[(d, T0, arm)] = (
                        float(a[:, 0].mean()),                    # pre-treatment fit
                        float(np.abs(a[:, 1]).mean()),            # absolute bias
                        float(a[:, 1].std(ddof=1)),               # SD of the gap
                        float(np.mean(a[:, 2] <= ALPHA + 1e-9)),  # rejection rate
                    )
    return out


def run() -> dict:
    g = _grid()
    res: dict = {}
    for d in D_VALUES:
        for T0 in T0_VALUES:
            for arm, _K, _dm in ARMS:
                pre, bias, sd, rej = g[(d, T0, arm)]
                tag = f"{arm}_{_D_TAG[d]}_T{T0}"
                res[f"pre_{tag}"] = pre
                res[f"bias_{tag}"] = bias
                res[f"sd_{tag}"] = sd
                res[f"rej_{tag}"] = rej

    def cell(d, T0, arm):
        return g[(d, T0, arm)]

    # Finding 1: demeaning improves the pre-treatment fit everywhere, and lowers
    # the bias where the treated unit is as extreme as the donors (d = 1).
    res["demeaning_improves_fit"] = float(sum(
        cell(d, T0, "demeaned")[0] < cell(d, T0, "conventional")[0]
        for d in D_VALUES for T0 in T0_VALUES))
    res["demeaning_cuts_bias_at_d1"] = float(sum(
        cell(1.0, T0, "demeaned")[1] < cell(1.0, T0, "conventional")[1]
        for T0 in T0_VALUES))
    # Finding 2: the test holds its nominal size at d = 1 and distorts as d
    # falls; demeaning alleviates the distortion.
    res["size_near_nominal_at_d1"] = float(sum(
        abs(cell(1.0, T0, arm)[3] - ALPHA) <= 0.06
        for T0 in T0_VALUES for arm, _K, _dm in ARMS))
    res["distortion_grows_as_d_falls"] = float(sum(
        cell(0.0, T0, "conventional")[3] > cell(1.0, T0, "conventional")[3]
        for T0 in T0_VALUES))
    res["demeaning_cuts_distortion"] = float(sum(
        cell(d, T0, "multi10")[3] < cell(d, T0, "conventional")[3]
        for d in (0.5, 0.0) for T0 in T0_VALUES))
    # Finding 3: more pre-periods, or more outcomes, also reduce the distortion.
    res["more_periods_cut_distortion"] = float(
        cell(0.0, 20, "conventional")[3] < cell(0.0, 5, "conventional")[3])
    res["more_outcomes_cut_distortion"] = float(sum(
        cell(d, T0, "multi10")[3] <= cell(d, T0, "demeaned")[3]
        for d in (0.5, 0.0) for T0 in T0_VALUES))
    # The conventional arm's gap is the dispersed one at d = 1, by a margin that
    # dwarfs the Monte Carlo error on its SD.
    res["demeaning_cuts_dispersion_at_d1"] = float(sum(
        cell(1.0, T0, "conventional")[2] > max(
            cell(1.0, T0, arm)[2] for arm, _K, dm in ARMS if dm)
        for T0 in T0_VALUES))
    return res


# Stochastic (M=100 vs the paper's 5,000). The cells carry the printed Table B.1
# values, and the tolerances are the Monte Carlo error at this draw count. They
# are lopsided because the conventional arm is: at d = 1 with few pre-periods
# the treated unit is often outside the donors' convex hull, so its gap is
# heavy-tailed and both its mean and its SD are estimated poorly at M = 100.
# Measured against the printed table, that arm lands within 0.22 on the
# pre-treatment fit, 0.34 on the bias and 0.54 on the SD of the gap, while the
# three demeaned arms land within 0.04, 0.17 and 0.19. The tolerances below
# leave headroom over those measurements; the rejection rate is within 0.07
# everywhere. The counts are the paper's findings and are exact -- including the
# dispersion ranking, which at d = 1 separates the conventional arm from every
# demeaned one by more than a whole unit of SD.
_PRE_TOL = {"conventional": 0.3, "demeaned": 0.15, "multi3": 0.15, "multi10": 0.15}
_SPREAD_TOL = {"conventional": 0.5, "demeaned": 0.3, "multi3": 0.3, "multi10": 0.3}
_SD_TOL = {**_SPREAD_TOL, "conventional": 0.8}
_REJ_TOL = 0.10

# Table B.1 as printed: {(d, T0): {arm: (pre, bias, sd, rej)}}.
_TABLE = {
    (1.0, 5): {"conventional": (1.65, 1.94, 2.91, 0.10), "demeaned": (0.51, 1.43, 1.81, 0.10),
               "multi3": (0.82, 1.32, 1.67, 0.10), "multi10": (0.99, 1.22, 1.54, 0.10)},
    (1.0, 10): {"conventional": (1.63, 1.64, 2.47, 0.10), "demeaned": (0.83, 1.27, 1.61, 0.10),
                "multi3": (1.04, 1.19, 1.50, 0.10), "multi10": (1.14, 1.12, 1.40, 0.10)},
    (1.0, 20): {"conventional": (1.62, 1.52, 2.36, 0.10), "demeaned": (1.03, 1.18, 1.49, 0.10),
                "multi3": (1.15, 1.11, 1.41, 0.10), "multi10": (1.20, 1.08, 1.36, 0.10)},
    (0.5, 5): {"conventional": (0.44, 1.10, 1.40, 0.36), "demeaned": (0.23, 1.16, 1.47, 0.32),
               "multi3": (0.56, 1.08, 1.36, 0.15), "multi10": (0.77, 1.01, 1.26, 0.12)},
    (0.5, 10): {"conventional": (0.71, 1.03, 1.29, 0.24), "demeaned": (0.54, 1.08, 1.35, 0.19),
                "multi3": (0.80, 1.01, 1.26, 0.14), "multi10": (0.91, 0.95, 1.18, 0.12)},
    (0.5, 20): {"conventional": (0.86, 0.95, 1.20, 0.17), "demeaned": (0.77, 0.99, 1.25, 0.15),
                "multi3": (0.92, 0.92, 1.16, 0.12), "multi10": (0.99, 0.89, 1.11, 0.10)},
    (0.0, 5): {"conventional": (0.24, 1.05, 1.32, 0.57), "demeaned": (0.15, 1.09, 1.37, 0.48),
               "multi3": (0.48, 1.04, 1.31, 0.19), "multi10": (0.71, 0.99, 1.23, 0.13)},
    (0.0, 10): {"conventional": (0.54, 0.98, 1.23, 0.34), "demeaned": (0.45, 1.03, 1.29, 0.25),
                "multi3": (0.72, 0.96, 1.20, 0.15), "multi10": (0.86, 0.90, 1.13, 0.13)},
    (0.0, 20): {"conventional": (0.73, 0.92, 1.16, 0.23), "demeaned": (0.68, 0.96, 1.21, 0.18),
                "multi3": (0.86, 0.90, 1.13, 0.14), "multi10": (0.93, 0.87, 1.09, 0.12)},
}

EXPECTED = {}
for _d in D_VALUES:
    for _T0 in T0_VALUES:
        for _arm, _K, _dm in ARMS:
            _pre, _bias, _sd, _rej = _TABLE[(_d, _T0)][_arm]
            _tag = f"{_arm}_{_D_TAG[_d]}_T{_T0}"
            EXPECTED[f"pre_{_tag}"] = (_pre, _PRE_TOL[_arm])
            EXPECTED[f"bias_{_tag}"] = (_bias, _SPREAD_TOL[_arm])
            EXPECTED[f"sd_{_tag}"] = (_sd, _SD_TOL[_arm])
            EXPECTED[f"rej_{_tag}"] = (_rej, _REJ_TOL)
EXPECTED.update({
    "demeaning_improves_fit": (9.0, 0.0),
    "demeaning_cuts_bias_at_d1": (3.0, 0.0),
    "size_near_nominal_at_d1": (12.0, 0.0),
    "distortion_grows_as_d_falls": (3.0, 0.0),
    "demeaning_cuts_distortion": (6.0, 0.0),
    "more_periods_cut_distortion": (1.0, 0.0),
    "more_outcomes_cut_distortion": (6.0, 0.0),
    "demeaning_cuts_dispersion_at_d1": (3.0, 0.0),
})
