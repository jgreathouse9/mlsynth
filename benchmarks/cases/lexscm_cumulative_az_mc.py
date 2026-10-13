r"""Path C: what the cumulative paths cover on Abadie and Zhao's own DGP.

``lexscm_cumulative_coverage`` prices the block-sum pivot on a two-component
simulation written for that purpose. This case asks the same question on the
data-generating process the design method itself was published with -- Abadie
and Zhao (2026) Section 5, Assumption 1, equations 12a/12b, as implemented in
``mlsynth.utils.marex_helpers.simulation.generate_marex_sample`` and already
used by ``lexscm_design_mc`` and ``marex_section5_mc``.

That DGP supplies both potential outcomes, so every treated unit has a known
effect and the per-unit estimand is not a construct of this case:
``tau_j = sum_t (Y^I_jt - Y^N_jt)`` over the post window.

The experimental-design loop is the paper's. LEXSCM picks ``m = 3`` treated
units from the untreated potential outcomes ``Y^N`` alone, so the design is
fixed before any intervention; the experiment then realises ``Y^I`` on exactly
those units in the post window; each treated unit is fitted to its own
synthetic control over the design's control pool, and the cumulative paths are
read at the final horizon against a nominal 0.90.

Four things come out, and the third is the one to read.

1. The aggregate covers better than the per-unit paths, on every arm. The
   aggregate borrows across the treated group and a single unit cannot.

2. Both fall short of nominal on this DGP, and further on the larger panel
   with the longer horizon. The covariates are uniform with random loadings,
   which puts a good share of units outside the convex hull of the others, and
   a cumulative total over fifteen periods accumulates a level offset fifteen
   times while its interval grows more slowly.

3. ``approximability`` sorts the units that are covered from the units that
   are not. Conditioning on the gate, coverage is near 0.82 on both arms --
   stable across panel size and horizon -- while units the gate refuses cover
   0.61 on the paper's dimensions and 0.34 on the longer horizon. The gate is
   not a formality attached to the interval; it is the variable that decides
   whether the interval means anything.

4. ``center=True`` costs the aggregate and does not clearly buy the per-unit
   paths. Subtracting each unit's blank-window mean takes aggregate coverage
   from 0.793 to 0.673 while the per-unit share moves 0.680 to 0.698, which is
   inside this arm's Monte Carlo error. The offsets it removes also sit in the
   residuals the null is built from, so taking them out contracts the null, and
   at 150 draws the bias it removes is not large enough to show against the
   width it costs. Its own docstring reports a per-unit gain on a four-unit
   design; on this DGP only the aggregate's loss is resolved.

Runtime is the binding constraint: each draw runs a full design search plus
one simplex solve per treated unit and per held-out series, so the arms are
sized at 150 draws, not the 1000 a tighter tolerance would want. The
tolerances below absorb that.

Provenance: no external coverage table to match -- Abadie and Zhao report the
design's MAE, not the cumulative band's coverage, and the band is mlsynth's own
construction. The DGP is theirs; the estimand, the loop and the measurement are
this case's, and the numbers are pinned against themselves. Companions:
``lexscm_cumulative_coverage`` (the same question on a purpose-built DGP) and
``conformal_window_count`` (the calibration-window count in the same family).
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd

_M = 3                  # treated units per design
_LEVEL = 0.90
_N_DRAWS = 150


def _one(J, T0, T, seed, center):
    """One design-then-realise draw; returns the aggregate and per-unit hits."""
    from mlsynth import LEXSCM
    from mlsynth.utils.marex_helpers.simulation import generate_marex_sample
    from mlsynth.utils.solvers.active_set import solve_simplex_qp
    from mlsynth.utils.fast_scm_helpers.post_inference import (
        approximability, unit_level_cumulative,
    )

    rng = np.random.default_rng(seed)
    sample = generate_marex_sample(J=J, T=T, T0=T0, rng=rng)
    names = [f"u{j:02d}" for j in range(J)]
    post_flag = [int(t >= T0) for t in range(T)]

    # the design sees untreated potential outcomes only
    frame = pd.DataFrame([
        {"market": names[j], "week": t, "y": sample.Y_N[j, t],
         "eligible": 1, "post": post_flag[t]}
        for j in range(J) for t in range(T)
    ])
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        res = LEXSCM(dict(df=frame, outcome="y", unitid="market", time="week",
                          candidate_col="eligible", post_col="post",
                          m=_M, top_K=10, verbose=False)).fit()

    layout, winner = res.panel.time, res.search.winner
    chosen = list(winner.treated_weight_dict)
    w = np.array([winner.treated_weight_dict[k] for k in chosen], float)
    w = w / w.sum()
    pool = [names.index(k) for k in winner.control_weight_dict]
    fit_w = slice(0, layout.n_fit)
    blank_w = slice(layout.n_fit, layout.n_fit + layout.n_blank)
    post_w = slice(T - layout.n_post, T)

    # the experiment realises Y^I on exactly the chosen units
    observed = sample.Y_N.T.copy()
    for k in chosen:
        observed[post_w, names.index(k)] = sample.Y_I[names.index(k), post_w]

    def _fit_against(target_col, donors):
        v = solve_simplex_qp(observed[fit_w][:, donors], observed[fit_w, target_col])
        return np.asarray(v[0] if isinstance(v, tuple) else v, dtype=float)

    post_gaps, blank_gaps = [], []
    for k in chosen:
        j = names.index(k)
        v = _fit_against(j, pool)
        post_gaps.append(observed[post_w, j] - observed[post_w][:, pool] @ v)
        blank_gaps.append(observed[blank_w, j] - observed[blank_w][:, pool] @ v)
    extra = []
    for j in range(J):
        if names[j] in chosen:
            continue
        donors = [c for c in pool if c != j]
        if not donors:
            continue
        v = _fit_against(j, donors)
        extra.append(observed[blank_w, j] - observed[blank_w][:, donors] @ v)

    P = np.column_stack(post_gaps)
    B = np.column_stack(blank_gaps)
    gates = [approximability(B[:, i]).ok for i in range(len(chosen))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = unit_level_cumulative(P, B, w, level=_LEVEL,
                                    extra_pool=extra or None, center=center)

    truth = {k: float((sample.Y_I[names.index(k), post_w]
                       - sample.Y_N[names.index(k), post_w]).sum())
             for k in chosen}
    agg_truth = float(sum(truth[k] * w[i] for i, k in enumerate(chosen)))
    a = out.aggregate[-1]
    units = []
    for i, k in enumerate(chosen):
        q = out.per_unit[i][-1]
        units.append((bool(q.lower <= truth[k] <= q.upper), gates[i]))
    return bool(a.lower <= agg_truth <= a.upper), units


def _arm(J, T0, T, seed0, center=False):
    agg_hits = unit_hits = n_units = 0
    passed, refused = [], []
    for d in range(_N_DRAWS):
        agg, units = _one(J, T0, T, seed0 + d, center)
        agg_hits += agg
        for hit, ok in units:
            unit_hits += hit
            n_units += 1
            (passed if ok else refused).append(hit)
    return dict(
        agg=agg_hits / _N_DRAWS,
        unit=unit_hits / n_units,
        gate_pass=float(np.mean(passed)) if passed else float("nan"),
        gate_fail=float(np.mean(refused)) if refused else float("nan"),
    )


def run() -> dict:
    paper = _arm(J=15, T0=25, T=30, seed0=7_000)
    large = _arm(J=30, T0=60, T=75, seed0=8_000)
    centred = _arm(J=30, T0=60, T=75, seed0=8_000, center=True)
    return {
        "paper_aggregate": paper["agg"],
        "paper_per_unit": paper["unit"],
        "paper_aggregate_edge": paper["agg"] - paper["unit"],
        "large_aggregate": large["agg"],
        "large_per_unit": large["unit"],
        "large_aggregate_edge": large["agg"] - large["unit"],
        "gate_pass_paper": paper["gate_pass"],
        "gate_fail_paper": paper["gate_fail"],
        "gate_pass_large": large["gate_pass"],
        "gate_fail_large": large["gate_fail"],
        "gate_separation_large": large["gate_pass"] - large["gate_fail"],
        "centred_aggregate": centred["agg"],
        "centred_per_unit": centred["unit"],
        "centring_costs_the_aggregate": large["agg"] - centred["agg"],
    }


# Filled from a run of this module. Coverage is a share of 150 draws for the
# aggregate cells (Monte Carlo standard error at most 0.041) and of 450
# unit-draws for the per-unit cells, correlated within a draw, so those get the
# same tolerance and not a tighter one. The tolerance is about two of those
# standard errors; differences of two shares carry both errors and get 0.14.
#
# `gate_separation_large` is the cell that carries the claim: units the
# approximability gate admits cover roughly half again as often as the units it
# refuses, on the same draws and the same designs. A change that broke the gate
# -- or made it fire at random -- collapses this toward zero and trips here
# before it trips anything else.
EXPECTED = {
    "paper_aggregate": (0.880, 0.09),
    "paper_per_unit": (0.778, 0.09),
    "paper_aggregate_edge": (0.102, 0.14),
    "large_aggregate": (0.793, 0.09),
    "large_per_unit": (0.680, 0.09),
    "large_aggregate_edge": (0.113, 0.14),
    "gate_pass_paper": (0.820, 0.09),
    "gate_fail_paper": (0.642, 0.11),
    "gate_pass_large": (0.848, 0.09),
    "gate_fail_large": (0.379, 0.11),
    "gate_separation_large": (0.469, 0.14),
    "centred_aggregate": (0.673, 0.09),
    "centred_per_unit": (0.698, 0.09),
    "centring_costs_the_aggregate": (0.120, 0.14),
}
