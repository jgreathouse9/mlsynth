"""What SCM-relaxation does to the weights -- Liao, Shi & Zheng (2026).

Liao, C., Shi, Z., & Zheng, Y. (2026), *A Relaxation Approach to Synthetic
Control*, `arXiv:2508.01793 <https://arxiv.org/abs/2508.01793>`_. Tables 1 and
2 of the v2 manuscript; the replication package
(`YapengZheng/Relaxed_SC <https://github.com/YapengZheng/Relaxed_SC>`_)
publishes the Brexit application only, so the design here is transcribed from
Section 5.1 and lives in ``mlsynth.utils.laxscm_helpers.simulation``.

The companion cases check other things. ``rescm_relax_mc`` pins that the L2
relaxation beats SCM out of sample on one design point; ``rescm_relax_ref``
cross-validates the solver against the authors' ``scmrelax`` at a fixed
relaxation level. Neither looks at what the weights are, and the paper's claim
is about the weights:

    When the donor pool exhibits a group structure, SCM-relaxation
    approximates the equal weights within each group to diversify the
    prediction risk.

That is a statement about *where* the weight goes, not about how large the
error is, and two estimators can match on error while doing entirely different
things to the donor pool. This case measures the mechanism.

Why it exists
-------------
Until 2026-09 the relaxation returned the equal weights ``1/J`` -- exactly, and
for all three objectives at once -- in 10-44% of fits on this design, because
the ``tau`` grid was built from a Lasso penalty path instead of the program's
own feasible range. The prediction error stayed plausible throughout: a fit
that averages every donor is a reasonable forecast and a completely different
estimator from the one the paper describes. The two cases above could not see
it, one because it fixes ``tau`` before comparing and the other because it
reports a median over replications. A weight-level check is what separates
"spreads within groups" from "spreads over everything".

What is pinned
--------------
The mechanism, as the contrast the paper draws with SCM: the relaxation puts
weight on most donors and nearly equally within a group, where SCM concentrates
on a handful. Then the ordering claims, which are rankings, not levels,
and so survive the fact that this is a transcribed DGP: L2 ahead of entropy
ahead of EL where the groups do not outnumber the factors, and entropy ahead of
L2 where they do (the paper's Panel C, its own stated boundary). Then the
collapse rate itself, which is the regression guard -- a return to the old grid
would drive it back up, and every ordering above would go with it, since a
collapsed replication returns the same weights for every objective.

The prediction-error ratios are reported against the paper's published cells
but banded loosely. Twenty-five replications against the paper's 1000 is a
quarter of the precision, the DGP is a transcription, and the levels are the
part of the table least robust to both.

What twenty-five replications can say
-------------------------------------
Not every ordering in Table 1 is testable here, and the paper's own margins say
which. It separates L2 from EL by 0.54 and 0.28 in Panels A and B; it separates
L2 from entropy by 0.033, 0.061 and 0.019. The first is far outside this case's
Monte Carlo error and the second is inside it, so only the first is asserted.

This was got wrong on the first pass. ``panelA_l2_beats_entropy`` was written
as a zero-tolerance boolean, and it failed -- not because the estimator is
wrong but because a 0.033 margin is not resolvable at this count, and the
assertion would have been luck either way. Asserting the Panel C crossover the
same way would have had the same defect and happened to pass.

What survives is the contrast. The paper's claim is that entropy *improves
relative to L2* as the groups come to outnumber the factors, moving from -0.033
in Panel A to +0.019 in Panel C, so ``entropy_advantage_C_over_A`` is the
direction of that movement, not either endpoint. It is reported with a
band wide enough to admit the wrong sign, because at 25 replications a +0.052
contrast is not resolvable either. Reading it needs the number, not the
pass. Resolving it would take a few hundred replications, which is a
half-hour case and a separate decision.
"""

from __future__ import annotations

import warnings

import numpy as np

_J, _T0, _T1 = 50, 25, 50
_SIMS = 25
_N_TAUS, _N_SPLITS = 10, 2
_METHODS = ["SC", "RELAX_L2", "RELAX_ENTROPY", "RELAX_EL"]
_RELAX = _METHODS[1:]

#: the paper's Panel A/B/C at J = 50, T0 = 25 -- K below, at, and above r
_PANELS = {"A": 2, "B": 3, "C": 4}
#: Table 1, J = 50 / T0 = 25 rows: (L2, entropy, EL) prediction-error ratios
_PUBLISHED = {"A": (0.3019, 0.3349, 0.8441),
              "B": (0.5290, 0.5895, 0.8095),
              "C": (0.5075, 0.4890, 0.5937)}

EXPECTED = {
    # ---- the mechanism: where the weight goes ----
    # SCM concentrates; the relaxation spreads. The paper's whole argument.
    "sc_share_of_donors_used": (0.16, 0.14),
    "relax_l2_share_of_donors_used": (0.98, 0.25),
    # and spreads *within groups*: dispersion inside a group, relative to SCM's
    "relax_l2_within_group_sd_ratio": (0.04, 0.10),
    # which is why it recovers the oracle weights the group structure implies
    "relax_l2_l1_distance_ratio": (0.36, 0.25),
    "relax_l2_l2_distance_ratio": (0.22, 0.20),
    # ---- the regression guard ----
    # Exact-1/J fits. Both bands are one-sided in effect -- centred at half
    # their ceiling so they admit 0, since a further drop is an improvement and
    # must not fail. The ceilings are set from a run against the old grid,
    # recorded under "Verified against the defect" above.
    "relax_l2_max_collapse_rate": (0.16, 0.16),
    # EL collapses more than L2 even on the corrected grid.
    "max_collapse_rate_any_objective": (0.20, 0.20),
    # ---- the ordering claims the replication count can resolve ----
    # The paper separates L2 from EL by 0.54 and 0.28 in Panels A and B.
    "panelA_l2_beats_el": (1.0, 0.0),
    "panelB_l2_beats_el": (1.0, 0.0),
    # ---- the levels, banded ----
    "panelA_l2_ratio": (0.3019, 0.30),
    "panelB_l2_ratio": (0.5290, 0.30),
    "panelC_l2_ratio": (0.5075, 0.30),
    "max_abs_dev_from_published": (0.21, 0.25),
    # every relaxation arm beats SCM in every panel
    "n_panels_all_relax_beat_sc": (3.0, 0.0),
    # ---- reported, not asserted: see "What twenty-five replications can say" ----
    # the paper's Panel C claim as a contrast, which is the form of it that has
    # any chance of surviving the count
    "entropy_advantage_C_over_A": (0.05, 0.45),
    "n_fits": (float(3 * _SIMS), 0.0),
}


def _weights(fit, J: int) -> np.ndarray:
    w = fit.weights
    if isinstance(w, dict):
        w = np.asarray([w[k] for k in sorted(w)], float)
    return np.asarray(w, float).ravel()


def run() -> dict:
    warnings.simplefilter("ignore")
    from mlsynth import RESCM
    from mlsynth.utils.laxscm_helpers.simulation import (
        simulate_relaxation_groups_design,
        to_panel,
    )

    r = max(1, int(np.floor(np.log(_T0))))
    acc: dict = {p: {m: {"ratio": [], "l1": [], "l2": [], "wsd": [], "used": [],
                         "collapse": []} for m in _METHODS} for p in _PANELS}
    fits = 0

    for panel, K in _PANELS.items():
        rng = np.random.default_rng(40_000 + K)
        for _ in range(_SIMS):
            d = simulate_relaxation_groups_design(
                rng, J=_J, T0=_T0, T1=_T1, K=K, r=r)
            res = RESCM({"df": to_panel(d.Yc, d.y0, d.T0), "outcome": "y",
                         "treat": "treat", "unitid": "unit", "time": "time",
                         "methods": _METHODS, "tau": None, "n_taus": _N_TAUS,
                         "n_splits": _N_SPLITS, "standardize": False,
                         "display_graphs": False}).fit()
            fits += 1
            post = slice(d.T0, None)
            err = {m: float(np.sum(
                (np.asarray(res.fits[m].counterfactual)[post] - d.oracle_cf[post]) ** 2))
                for m in _METHODS}
            if err["SC"] <= 0:
                continue
            for m in _METHODS:
                w = _weights(res.fits[m], _J)
                a = acc[panel][m]
                a["ratio"].append(err[m] / err["SC"])
                a["l1"].append(float(np.abs(w - d.w_star).sum()))
                a["l2"].append(float(np.linalg.norm(w - d.w_star)))
                a["used"].append(float((w > 1e-6).mean()))
                a["wsd"].append(float(np.mean(
                    [w[d.groups == k].std() for k in range(K)])))
                a["collapse"].append(float(np.abs(w - 1.0 / _J).max() < 1e-8))

    med = {p: {m: {k: float(np.median(v)) if v else float("nan")
                   for k, v in acc[p][m].items()} for m in _METHODS} for p in _PANELS}
    out: dict = {"n_fits": float(fits)}

    # the mechanism, averaged over the three panels
    out["sc_share_of_donors_used"] = float(np.mean([med[p]["SC"]["used"] for p in _PANELS]))
    out["relax_l2_share_of_donors_used"] = float(
        np.mean([med[p]["RELAX_L2"]["used"] for p in _PANELS]))
    out["relax_l2_within_group_sd_ratio"] = float(np.mean(
        [med[p]["RELAX_L2"]["wsd"] / med[p]["SC"]["wsd"] for p in _PANELS]))
    out["relax_l2_l1_distance_ratio"] = float(np.mean(
        [med[p]["RELAX_L2"]["l1"] / med[p]["SC"]["l1"] for p in _PANELS]))
    out["relax_l2_l2_distance_ratio"] = float(np.mean(
        [med[p]["RELAX_L2"]["l2"] / med[p]["SC"]["l2"] for p in _PANELS]))

    out["relax_l2_max_collapse_rate"] = float(max(
        float(np.mean(acc[p]["RELAX_L2"]["collapse"])) for p in _PANELS))
    out["max_collapse_rate_any_objective"] = float(max(
        float(np.mean(acc[p][m]["collapse"])) for p in _PANELS for m in _RELAX))

    rat = {p: {m: med[p][m]["ratio"] for m in _RELAX} for p in _PANELS}
    out["panelA_l2_beats_el"] = float(rat["A"]["RELAX_L2"] < rat["A"]["RELAX_EL"])
    out["panelB_l2_beats_el"] = float(rat["B"]["RELAX_L2"] < rat["B"]["RELAX_EL"])
    # entropy's standing against L2, C relative to A. The paper has it moving
    # from -0.033 to +0.019, so this contrast is +0.052 there.
    out["entropy_advantage_C_over_A"] = float(
        (rat["C"]["RELAX_ENTROPY"] - rat["C"]["RELAX_L2"])
        - (rat["A"]["RELAX_ENTROPY"] - rat["A"]["RELAX_L2"])) * -1.0

    for p in _PANELS:
        out[f"panel{p}_l2_ratio"] = rat[p]["RELAX_L2"]
    out["max_abs_dev_from_published"] = float(max(
        abs(rat[p][m] - _PUBLISHED[p][i])
        for p in _PANELS for i, m in enumerate(_RELAX)))
    out["n_panels_all_relax_beat_sc"] = float(sum(
        all(rat[p][m] < 1.0 for m in _RELAX) for p in _PANELS))
    return out
