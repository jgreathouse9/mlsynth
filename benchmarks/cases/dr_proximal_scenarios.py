"""PROXIMAL against every scenario in the DR_Proximal_SC simulation directory.

Qiu, Shi, Miao, Dobriban and Tchetgen Tchetgen (2024) ship their Monte Carlo as
eight scenarios, each a ``generate_data.R`` carrying the DGP and eight
estimators, plus a ``simulation.R`` that preprocesses the panel before
estimating. This runs mlsynth's ``PROXIMAL``, through its public API, on the
panels their code produces, and compares against what their estimators return on
those same panels.

Path: cross-validation against an authoritative reference implementation. The
reference script sources each scenario's ``generate_data.R`` in its own
environment and copies that scenario's preprocessing block verbatim, so the
panel carried in the bundle is the one their estimators actually received.
Storing the preprocessed panel rather than the raw draw is deliberate: it keeps
the whole of their preprocessing on the R side, where this case cannot disagree
with it by porting it wrong.

Their preprocessing is not uniform, which is most of why the eight scenarios are
different tests and not one test repeated. ``normal``, ``nonstationary``,
``short_post`` and ``small_T`` take a raw-power quadratic expansion of the proxy
blocks with each column scaled by its largest absolute value. ``detrend`` first
pools every proxy series, regresses on ``poly(t, 2)``, and subtracts the fitted
trend from ``W``, ``Z`` and ``Y`` alike. ``just_identified`` does none of it.
``select_donor`` runs ``Synth`` to decide which pooled columns act as outcome
proxies and which as treatment proxies, so the assignment is estimated.

What is compared
----------------
Their ``correct.DR`` and ``correct.q`` are mlsynth's ``DR`` and ``PIPW``. Both
are driven through ``PROXIMAL(...).fit()`` on a long panel whose donor units
carry ``W`` as their outcome and ``Z`` as a donor-proxy variable.

Across the 19 compared cells the point estimates match to ``5e-06`` for ``DR``
and ``1.4e-05`` for ``PIPW``. The standard errors sit up to 16 percent apart, and
the reason is a variance-estimator convention, established rather than assumed:
their ``correct.q``
takes ``sandwich::vcovHAC`` over the full GMM parameter vector with an Andrews
automatic bandwidth, while mlsynth uses a Bartlett kernel at
``floor(4 (T_post/100)^(2/9))`` with no finite-sample adjustment. Recomputing
their own GMM fit under mlsynth's settings closes the gap, on the first
``normal`` panel::

    their default   QS kernel, Andrews bw, adjust=TRUE     0.1787608416
                    QS kernel, Andrews bw, adjust=FALSE    0.1762404217
                    Bartlett,  Andrews bw, adjust=FALSE    0.1769125271
    mlsynth's       Bartlett,  fixed bw=4,  adjust=FALSE   0.1750005994
    mlsynth                                                0.1749891868

Three settings and 6.5e-05 of residual, so the sandwich itself agrees and the
deviation is the kernel, the bandwidth rule and the adjustment. They are pinned
as measured so the convention cannot drift unnoticed, which is not the same as
calling them agreement.

``dr_pipw_internal_se_gap`` is the cell that needs no reference at all. In the
just-identified design the treatment bridge solves its moment exactly, the DR
correction cancels, and the two are the same estimator against the same moments,
so their standard errors must coincide. It reads 5e-10. Before the sandwich
index in ``estimate_pipw`` was corrected it read 0.44.

What is excluded, and why
-------------------------
Three exclusions, each named in a module constant so a reader can check the list
instead of inferring it from an absence.

``over_identified`` carries five outcome proxies against ten treatment proxies,
so the GMM ``DR`` and ``PIPW`` solve is not square and they do not apply.
mlsynth's over-identified route is ``DR-OID``, and it is a different estimator,
which was checked rather than assumed. Their ``correct.DR`` there stacks both
bridges into one five-block GMM and solves them jointly; ``DR-OID`` implements
Shi et al. (2026)'s formulation. Run on the same two panels they give 2.049 and
2.007 against their 1.564 and 1.709, with a true effect of 2 -- both consistent
for the same estimand and 0.3 to 0.5 apart in finite sample, which is not a
tolerance question. The scenario's reference values sit in the bundle for anyone
who wants to pursue it.

``short_post`` (``T_post = 22``) and ``small_T`` (``T_post = 20``) meet the shape
condition and strain a numerical one, so they are measured separately below
instead of being compared cell by cell.

``select_donor_r2`` is their own degenerate draw: ``Synth`` picks the wrong proxy
assignment, their ``correct.DR`` returns 9.82 against a true effect of 2, and its
reported standard error is 103977 with a convergence flag of 0. mlsynth declines
to fit it at all, which is the third exclusion and also the first cell below.

The short post-period designs
-----------------------------
``psi = E_post[(1, W)]`` is a mean over the post-treatment periods, so with
twenty of them the exponential tilt that has to reproduce it may sit far out in
``beta`` or not exist in finite sample. On five of these eight cells mlsynth's
root search cannot bring the moment residual under ``1e-6`` and
``fit_treatment_bridge`` says so; their own ``convergence`` flag is 1 on two of
those, so their solver reports failure there too and their simulation counts the
number anyway.

On the three cells where mlsynth does solve the moment, the two implementations
still land 3e-02 to 1.2e-01 apart, which is three to four orders of magnitude
worse than the 1e-6 they agree to everywhere else. The reason is not two roots of
one system. Refitting their ``correct.q`` on these panels and evaluating the
bridge moment at the ``beta`` it returns gives residuals of 8.3e-03 to 4.9e-02 on
all eight cells -- their fit does not solve that moment either.

The two are solving different problems once it cannot be solved. Their estimator
is one joint GMM: ``gmm(..., wmatrix = "ident")`` minimises the summed squared
norm of every moment block at once, so when the bridge block is unsatisfiable the
optimiser trades it off against the others and returns the compromise. mlsynth
solves the bridge exactly as its own step and substitutes the result, so it has
nothing to trade and says so instead. Where the moment has a solution both routes
land on it and the two agree to 1e-6, which is the other 19 cells; where it does
not, one compromises and the other declines.

That makes the equivalence this case pins conditional, and the condition is
solvability. It is a difference in construction, not a defect on either side.

So this scenario group is pinned by what can be checked: how many cells mlsynth
declines, and how far apart the rest land. Averaging these into the main cells
would report agreement the case does not have.

Reference bundle: ``benchmarks/reference/dr_proximal_scenarios/`` (regenerate
with ``Rscript benchmarks/reference/dr_proximal_scenarios/reference.R``).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from mlsynth.exceptions import MlsynthEstimationError

from benchmarks.reference import load_reference

_CASE = "dr_proximal_scenarios"
_PANELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reference", _CASE, "panels.tsv.gz")

#: The scenarios compared cell by cell. Square proxy blocks, so the
#: just-identified GMM that DR and PIPW solve applies, and a treatment-bridge
#: solve that converges. ``short_post`` and ``small_T`` meet the first condition
#: and not the second -- see the module docstring.
SCENARIOS = ("normal", "detrend", "nonstationary", "just_identified",
             "select_donor")

#: The short post-period designs, measured separately: see the module docstring.
SHORT_POST = tuple(f"{s}_r{r}" for s in ("short_post", "small_T")
                   for r in range(1, 5))

#: Their own degenerate draw: ``Synth`` assigns the wrong proxies, their
#: ``correct.DR`` returns 9.82 against a true effect of 2 and reports a standard
#: error of 103977, and its convergence flag still reads 0.
DEGENERATE = ("select_donor_r2",)


def _panels() -> dict:
    df = pd.read_csv(_PANELS, sep="\t")
    out = {}
    for cell, blk in df.groupby("cell", sort=False):
        blk = blk.sort_values("t")
        out[str(cell)] = (
            blk["y"].to_numpy(dtype=float),
            np.array([np.fromstring(s, sep=" ") for s in blk["W"]]),
            np.array([np.fromstring(s, sep=" ") for s in blk["Z"]]),
        )
    return out


def _long(Y: np.ndarray, W: np.ndarray, Z: np.ndarray, T0: int) -> pd.DataFrame:
    """The panel as the public API takes it.

    Each donor unit carries one column of ``W`` as its outcome and the matching
    column of ``Z`` as its proxy, which is the shape ``prepare_proximal_inputs``
    pivots back into the two matrices the bridges consume.
    """
    T = len(Y)
    recs = [{"unit": "treated", "time": t + 1, "y": float(Y[t]), "proxy": 0.0,
             "treat": int(t >= T0)} for t in range(T)]
    for j in range(W.shape[1]):
        recs += [{"unit": f"d{j + 1}", "time": t + 1, "y": float(W[t, j]),
                  "proxy": float(Z[t, j]), "treat": 0} for t in range(T)]
    return pd.DataFrame(recs)


def _fit(panel: pd.DataFrame, n_donors: int):
    from mlsynth import PROXIMAL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PROXIMAL({
            "df": panel, "outcome": "y", "treat": "treat", "unitid": "unit",
            "time": "time", "donors": [f"d{j + 1}" for j in range(n_donors)],
            "vars": {"donorproxies": ["proxy"]},
            "methods": ["DR", "PIPW"], "display_graphs": False,
        }).fit()


def run() -> dict:
    ref = load_reference(_CASE)["values"]
    reps = int(ref["n_reps"])
    panels = _panels()

    dr_phi_err = pipw_phi_err = 0.0
    dr_se_dev = pipw_se_dev = 0.0
    internal_se_gap = 0.0
    n = 0
    covered = []

    for scen in SCENARIOS:
        for r in range(1, reps + 1):
            cell = f"{scen}_r{r}"
            if cell in DEGENERATE:
                continue
            Y, W, Z = panels[cell]
            T0 = int(ref[f"T0_{cell}"])
            res = _fit(_long(Y, W, Z, T0), W.shape[1])
            sub = res.sub_method_results
            dr, pipw = sub["DR"], sub["PIPW"]

            dr_phi_err = max(dr_phi_err, abs(dr.effects.att - ref[f"correctDR_phi_{cell}"]))
            pipw_phi_err = max(pipw_phi_err, abs(pipw.effects.att - ref[f"correctq_phi_{cell}"]))

            dr_se, pipw_se = dr.inference.standard_error, pipw.inference.standard_error
            internal_se_gap = max(internal_se_gap, abs(dr_se / pipw_se - 1.0))
            dr_se_dev = max(dr_se_dev, abs(dr_se / ref[f"correctDR_se_{cell}"] - 1.0))
            pipw_se_dev = max(pipw_se_dev, abs(pipw_se / ref[f"correctq_se_{cell}"] - 1.0))
            covered.append(cell)
            n += 1

    # The short post-period group, plus their degenerate draw.
    declined, gaps = 0, []
    for cell in SHORT_POST + DEGENERATE:
        Y, W, Z = panels[cell]
        try:
            sub = _fit(_long(Y, W, Z, int(ref[f"T0_{cell}"])), W.shape[1]).sub_method_results
        except MlsynthEstimationError:
            declined += 1
            continue
        gaps.append(abs(sub["DR"].effects.att - ref[f"correctDR_phi_{cell}"]))

    return {
        "n_cells": float(n),
        "short_post_cells": float(len(SHORT_POST) + len(DEGENERATE)),
        "short_post_declined": float(declined),
        "short_post_max_abs_phi_err": max(gaps) if gaps else 0.0,
        "n_scenarios": float(len(SCENARIOS)),
        "dr_max_abs_phi_err": dr_phi_err,
        "pipw_max_abs_phi_err": pipw_phi_err,
        "dr_max_rel_se_dev": dr_se_dev,
        "pipw_max_rel_se_dev": pipw_se_dev,
        "dr_pipw_internal_se_gap": internal_se_gap,
    }


# Deterministic: the panels are fixed in the bundle and both estimators solve a
# just-identified GMM in closed form from a Newton solve with a fixed start. No
# seeds enter at run time.
#
# The point-estimate tolerance is 1e-4 against an observed ~2e-06. It is loose
# enough for the bridge solve to land on a slightly different root across BLAS
# builds and tight enough that any disagreement about the estimator itself
# blows through it.
#
# The standard-error deviations are pinned as measured, not at zero, because
# they are a variance-estimator convention: their sandwich::vcovHAC with an
# Andrews bandwidth against a fixed-lag Bartlett kernel here. Pinning them keeps
# the convention from drifting silently while stating that it is not agreement.
#
# dr_pipw_internal_se_gap is the invariant that has nothing to do with R. In the
# just-identified design the treatment bridge solves its moment exactly, the DR
# correction cancels, and the two are the same estimator against the same
# moments -- so their standard errors must coincide. This cell read 0.44 before
# the sandwich index was corrected.
EXPECTED = {
    "n_cells": (19.0, 0.0),
    "n_scenarios": (5.0, 0.0),
    "dr_max_abs_phi_err": (0.0, 1e-4),
    "pipw_max_abs_phi_err": (0.0, 1e-4),
    "dr_max_rel_se_dev": (0.16, 0.08),
    "pipw_max_rel_se_dev": (0.07, 0.06),
    "dr_pipw_internal_se_gap": (0.0, 1e-6),
    # The short post-period group. Six of nine cells are declined because the
    # bridge moment cannot be solved to 1e-6; the three that solve land far
    # enough from their code to be reported, not absorbed. Both cells are pinned
    # as measured, with one cell of slack on the count.
    "short_post_cells": (9.0, 0.0),
    "short_post_declined": (6.0, 1.0),
    "short_post_max_abs_phi_err": (0.12, 0.06),
}
