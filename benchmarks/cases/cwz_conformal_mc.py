"""CWZ conformal test (JASA 2021): the paper's own simulation study.

Path B, scenario 3. Chernozhukov, Wüthrich & Zhu (2021), Section 4: the
empirical rejection rate of the moving-block conformal test under a true null,
which is its size and should sit near the nominal level. The design is the
authors' own ``sim()`` from the online supplement
(``replication_package_final/simulations_conformal_final.R``), ported in
``benchmarks/cwz_common.py``; the reference side is a live run of that design
driving ``scinference``, captured in
``benchmarks/reference/cwz_conformal_mc/``.

Two claims are separated here because they can fail independently, and a single
end-to-end rate cannot tell them apart.

The inference is identical. The reference dumps its first ten panels for two
cells alongside the p-value it computed on each; mlsynth is handed those exact
panels and has to return those exact p-values. No Monte-Carlo error enters this
comparison at all -- the moving-block reference set is the ``T`` cyclic shifts,
enumerated on both sides -- so the tolerance is floating point and the check is
a defect detector, not a noise budget.

The DGP port is faithful. mlsynth draws its own panels from its own port and has
to land on the reference's rejection rate up to Monte-Carlo error. This is what
would catch a mis-transcribed factor loading or an AR(1) started at the wrong
variance, neither of which the per-draw check can see.

The cells span the four weight vectors at both error structures. The
:math:`\\rho = 0.6` half is the one that matters: serially correlated errors are
the assumption boundary the moving-block scheme exists for, and a scheme that
had become an i.i.d. permutation would over-reject there while looking fine at
:math:`\\rho = 0`. DGPs 3 and 4 put the treated unit outside the donor
hull, so the synthetic control is misspecified by construction and the test is
still exact -- the paper's point, and the reason those cells are kept.
"""
from __future__ import annotations

import os

import numpy as np

from benchmarks.compare import BenchmarkSkipped
from benchmarks.cwz_common import (
    DGPS, load_reference_draws, moving_block_pvalue, simulate_conformal_panel,
)
from benchmarks.reference import reference_value

_REF = os.path.join(os.path.dirname(__file__), "..", "reference", "cwz_conformal_mc")

ALPHA = 0.1
T0, T1, J = 20, 1, 20
# The reference runs 500 draws per cell (the paper runs 5000); mlsynth runs the
# same 500 so the two rates are comparable at the same Monte-Carlo scale.
N_REPS = 500
SEED = 20260814


def _size(dgp: int, rho: float) -> float:
    """Rejection rate under a true null: mlsynth's own draws, mlsynth's own test."""
    rng = np.random.default_rng(SEED + 1000 * dgp + int(100 * rho))
    rejections = 0
    for _ in range(N_REPS):
        y, Y0 = simulate_conformal_panel(
            dgp, pre_periods=T0, post_periods=T1, n_donors=J,
            rho_donor=rho, rho_treated=rho, stationary=True,
            alternative=0.0, rng=rng,
        )
        rejections += moving_block_pvalue(y, Y0, T0) <= ALPHA
    return rejections / N_REPS


def _tag(rho: float) -> str:
    return "rho0" if rho == 0 else "rho06"


def run() -> dict:
    out: dict = {}

    # The seam: the reference's own panels, its own p-values.
    for rho in (0.0, 0.6):
        path = os.path.abspath(os.path.join(_REF, f"draws_{_tag(rho)}.tsv"))
        if not os.path.exists(path):
            raise BenchmarkSkipped(f"draws_{_tag(rho)}.tsv not available")
        for i, (y, Y0) in enumerate(load_reference_draws(path), start=1):
            out[f"draw_{_tag(rho)}_{i:02d}_p"] = moving_block_pvalue(y, Y0, T0)

    # The design: mlsynth's own draws, compared as a rate.
    for rho in (0.0, 0.6):
        for dgp in DGPS:
            out[f"size_dgp{dgp}_{_tag(rho)}"] = _size(dgp, rho)
    return out


def comparison() -> dict:
    """mlsynth against ``scinference`` on the authors' simulation design.

    Ten seed-matched panels per error structure, where both sides see the
    identical data and the p-values must agree exactly, and the eight size cells,
    where each side draws its own panels and the rates agree up to Monte-Carlo
    error.
    """
    got = run()
    rows = []
    for rho in (0.0, 0.6):
        for i in (1, 5, 10):                     # a readable subset of the ten
            key = f"draw_{_tag(rho)}_{i:02d}_p"
            rows.append({"quantity": f"p-value, rho={rho}, draw {i}",
                         "mlsynth": round(got[key], 6),
                         "reference": round(reference_value("cwz_conformal_mc", key), 6)})
    for rho in (0.0, 0.6):
        for dgp in DGPS:
            key = f"size_dgp{dgp}_{_tag(rho)}"
            rows.append({"quantity": f"size, DGP {dgp}, rho={rho}",
                         "mlsynth": round(got[key], 4),
                         "reference": round(reference_value("cwz_conformal_mc", key), 4)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC",
                         "config": {"inference": "conformal",
                                    "conformal_type": "block", "alpha": ALPHA}},
        "reference": {"impl": "R package scinference (conformal) driven by the "
                              "authors' own simulation design",
                      "version": "scinference v1.0.0 (567c688)"},
    }


_r = lambda k: reference_value("cwz_conformal_mc", k)

# Two tolerances, because two different things are claimed.
#
# The per-draw p-values are exact. Both sides enumerate the same T = 21 cyclic
# shifts of the same residual path, and the only step that is not integer
# arithmetic is the simplex solve, where mlsynth's QP and limSolve's
# lsei(type=2) agree to ~1e-8 on these panels. A p-value here lives on the grid
# k/21, whose spacing is 0.0476, so 1e-9 is far tighter than the finest possible
# real difference: any disagreement at all means a different reference set, a
# different statistic, or a different refit.
#
# The size cells are Monte Carlo on both sides. Each is a proportion from 500
# independent draws, so one side's standard error at p = 0.1 is
# sqrt(0.1 * 0.9 / 500) = 0.0134, and the difference of two independent rates has
# 0.019. The band is 0.05, a shade under three of those: loose enough that a
# re-seed does not flip a cell, tight enough that the failure this case exists to
# catch cannot hide in it -- an i.i.d. permutation scheme substituted for the
# moving-block one over-rejects at rho = 0.6 by far more than 0.05, and a test
# that had lost its calibration entirely would miss by more still.
EXPECTED = {
    **{f"draw_{tag}_{i:02d}_p": (_r(f"draw_{tag}_{i:02d}_p"), 1e-9)
       for tag in ("rho0", "rho06") for i in range(1, 11)},
    **{f"size_dgp{dgp}_{tag}": (_r(f"size_dgp{dgp}_{tag}"), 0.05)
       for tag in ("rho0", "rho06") for dgp in DGPS},
}
