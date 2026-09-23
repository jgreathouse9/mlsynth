"""CWZ debiased SC t-test: Table 1's relative asymptotic efficiency.

Path B, scenario 3. Table 1 of Chernozhukov, Wüthrich & Zhu's t-test paper: the
relative asymptotic efficiency of the :math:`K`-fold debiased confidence
interval, the ratio of its limiting expected length as :math:`K \\to \\infty` to
its length at finite :math:`K`. It depends on nothing but :math:`K`, the level,
and the pre-to-post ratio :math:`c_0 = T_0 / T_1`, so the whole table is a
closed form and the reference is the formula, not a fit.

This is the rule behind ``ttest_K="auto"``. mlsynth's ``select_K`` climbs
:math:`K` until the RAE meets a target, so the number that decides how many
folds a user gets is this one, and the case exists because that rule was
exercised at every level except against the table it comes from.

The reference is the authors' own ``RAE.R`` from the JPE replication package,
run live and captured in ``benchmarks/reference/cwz_rae/``, at their own
:math:`c_0 = 30/16` — the carbon tax panel's shape — and :math:`\\alpha = 0.1`.

What the table says, and what the case therefore pins: two folds cost most of
the efficiency (RAE 0.33), three recovers two thirds of it (0.64), and the
returns flatten past six (0.86 rising to 0.92 at ten). That shape is why the
paper's guidance settles on a small :math:`K` and why ``select_K`` treats three
as the floor.
"""
from __future__ import annotations

from benchmarks.reference import reference_value

ALPHA = 0.1
C0 = 30.0 / 16.0
K_VALUES = tuple(range(2, 11))


def run() -> dict:
    from mlsynth.utils.inferutils import rae

    out = {f"rae_K{K}": float(rae(C0, K, alpha=ALPHA)) for K in K_VALUES}
    base = out["rae_K3"]
    for K in (4, 5, 6):
        out[f"gain_K{K}_over_K3"] = 100.0 * (out[f"rae_K{K}"] - base)
    return out


def comparison() -> dict:
    """mlsynth's ``inferutils.rae`` against the authors' ``RAE.R``, Table 1."""
    got = run()
    rows = [{"quantity": f"RAE, K={K}",
             "mlsynth": round(got[f"rae_K{K}"], 6),
             "reference": round(reference_value("cwz_rae", f"rae_K{K}"), 6)}
            for K in K_VALUES]
    rows += [{"quantity": f"gain over K=3, K={K} (pp)",
              "mlsynth": round(got[f"gain_K{K}_over_K3"], 4),
              "reference": round(reference_value("cwz_rae", f"gain_K{K}_over_K3"), 4)}
             for K in (4, 5, 6)]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "mlsynth.utils.inferutils.rae",
                         "config": {"c0": C0, "alpha": ALPHA}},
        "reference": {"impl": "the authors' RAE.R (JPE replication package), live run",
                      "version": "closed form; no packages"},
    }


_r = lambda k: reference_value("cwz_rae", k)

# A closed form on both sides, evaluated with the same standard special
# functions: R's qt / gamma against scipy's t.ppf and the log-gamma difference
# mlsynth uses to keep the ratio finite at large K. Nothing here is estimated,
# resampled or solved, so the only difference possible is floating point in the
# quantile and gamma evaluations, which is why the band is 1e-9 and not a
# display-rounding allowance. The paper prints these to two decimals; matching to
# two decimals would not distinguish this formula from a neighbouring one, and
# the point of running theirs is to be able to ask for more.
EXPECTED = {
    **{f"rae_K{K}": (_r(f"rae_K{K}"), 1e-9) for K in K_VALUES},
    **{f"gain_K{K}_over_K3": (_r(f"gain_K{K}_over_K3"), 1e-7) for K in (4, 5, 6)},
}
