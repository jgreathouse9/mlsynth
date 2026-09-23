"""mlsynth's default PDA path on Shi & Huang's (2023) Table-1 design.

Forward selection and the cross-validated LASSO on the four-factor Monte Carlo
DGP of Shi & Huang (2023), *"Forward-selected panel data approach for program
evaluation,"* J. Econometrics 234(2), 512-535
(:func:`mlsynth.utils.pda_helpers.simulation.simulate_pda_panel`). One treated
unit, ``N = 100`` controls of which four are relevant, ``T1 = T2 = 100``; the
test is the post-selection t-test at the 5% level, "size" the D1 (null)
rejection rate and "power" the D5 (mean-1 shift) rate.

The paper's own table, cell by cell and at its own tuning, is ``fspda_table1``.
What this case covers is the path a user gets by default -- ``LassoCV`` for the
penalty and Li & Bell's (2017) two-component variance for the test -- on the
same design, so that the cost of the defaults is measured and not assumed.

What the defaults cost
----------------------
The two rules select differently. Forward selection stops at 5.5 donors on
average and the cross-validated LASSO takes 13.5, against the paper's 7 and 11
for the same design at its own tuning.

The size behaves differently, and the difference is the penalty rule and not the
estimator class. Shi & Huang report the LASSO's null rejection rate inflating
under dynamic factors -- 0.058 to 0.184 at this length -- while forward
selection's holds. Holding the variance at Li & Bell's and changing only the
penalty reproduces that split: under the modified BIC the null rejection rate
goes 0.013 (i.i.d.) to 0.087 (dynamic) at ``T1 = 100``, and under
cross-validation it goes 0.070 to 0.045. The inflation is a property of their
selection rule, not of L1 selection.

Both rules are fully powered against the D5 alternative at this length, as the
paper reports.

Path B (scenario 3: the authors' full repository). The DGP is traced to
``simulation/nonsparse/FS.simulation.dense.R`` in ``zhentaoshi/fsPDA`` at
``5f4542c`` and uses their committed loadings' distribution.
``fs_intercept=False`` (the default) is required for valid fs size on the
mean-zero factor data -- an intercept absorbs a spurious level and breaks it.
"""
from __future__ import annotations

import warnings

import numpy as np

from mlsynth.utils.pda_helpers.simulation import simulate_pda_panel

T1 = 100
M_SIZE = 200      # reps for the size (D1) cells (binomial SE ~ 0.017 at p~0.06)
M_POWER = 60      # reps for the power (D5) cells


def _fit(sample):
    from mlsynth import PDA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PDA({
            "df": sample.df, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "methods": ["fs", "LASSO"], "fs_intercept": False,
            "display_graphs": False,
        }).fit()


def _n_sel(fit) -> int:
    sel = fit.selected_donors
    return len(sel) if sel is not None else int(fit.beta.size)


def _sweep(shock, dynamic, M, base_seed):
    """Return (mean fs donors, mean lasso donors, fs rej rate, lasso rej rate)."""
    fs_don, la_don, fs_rej, la_rej = [], [], 0, 0
    for i in range(M):
        s = simulate_pda_panel(T1=T1, shock=shock, dynamic_factors=dynamic,
                               seed=base_seed + i)
        r = _fit(s)
        fs, la = r.fits["fs"], r.fits["lasso"]
        fs_don.append(_n_sel(fs)); la_don.append(_n_sel(la))
        fs_rej += int(fs.p_value < 0.05); la_rej += int(la.p_value < 0.05)
    return (float(np.mean(fs_don)), float(np.mean(la_don)),
            fs_rej / M, la_rej / M)


def run() -> dict:
    # i.i.d. factors: size (D1) and donor counts
    fs_d_iid, la_d_iid, fs_sz_iid, la_sz_iid = _sweep("D1", False, M_SIZE, 5000)
    # i.i.d. factors: power (D5)
    _, _, fs_pw, la_pw = _sweep("D5", False, M_POWER, 9000)
    # dynamic factors: size (D1)
    _, _, fs_sz_dyn, la_sz_dyn = _sweep("D1", True, M_SIZE, 7000)
    return {
        "fs_donors_iid": fs_d_iid,
        "lasso_donors_iid": la_d_iid,
        "fs_size_iid": fs_sz_iid,
        "lasso_size_iid": la_sz_iid,
        "fs_power": fs_pw,
        "lasso_power": la_pw,
        "fs_size_dyn": fs_sz_dyn,
        "lasso_size_dyn": la_sz_dyn,
        # robust ordering indicators (1.0 == holds)
        "lasso_over_selects": float(la_d_iid > fs_d_iid + 2.0),
        "both_fully_powered": float(fs_pw >= 0.95 and la_pw >= 0.95),
    }


# Deterministic: every replication is seeded, so re-running returns identical
# numbers. The cells below are the measured values; the tolerances absorb the
# binomial Monte Carlo noise at these M and nothing else. At M_SIZE = 200 the
# standard error of a size cell is sqrt(p(1-p)/200) -- 0.021 at p = 0.095, 0.015
# at p = 0.045 -- so 0.05 is between two and three of them; the donor counts are
# means over 200 draws and carry far less. The power cells sit at 1.0 with M = 60
# and get 0.06, which is the point below which 60 draws could not all reject.
#
# These are mlsynth's defaults on the paper's design, not the paper's cells, so
# the published values are quoted alongside as context and not as targets.
EXPECTED = {
    "fs_donors_iid": (5.48, 0.8),        # paper 7, at its own tuning
    "lasso_donors_iid": (13.55, 2.0),    # over-selects; paper 11
    "fs_size_iid": (0.095, 0.05),        # paper 0.059
    "lasso_size_iid": (0.070, 0.05),     # paper 0.058
    "fs_power": (1.0, 0.06),             # paper 1.00
    "lasso_power": (1.0, 0.06),          # paper 1.00
    "fs_size_dyn": (0.095, 0.05),        # fs stays sized under dyn; paper 0.088
    "lasso_size_dyn": (0.045, 0.05),     # cross-validation does not inflate here
    "lasso_over_selects": (1.0, 0.0),
    "both_fully_powered": (1.0, 0.0),
}
