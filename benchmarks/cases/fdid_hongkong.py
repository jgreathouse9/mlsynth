"""Cross-validation: Forward DiD against both of Li's released implementations.

Li's headline application uses a confidential retailer panel, but she released
MATLAB and R code and a public companion dataset -- the Hsiao, Ching and Wan
(2012) Hong Kong GDP panel. Both of her implementations are vendored and run
live: ``Fun_FDID.R`` under Rscript, and ``FDID_Matlab.m`` with ``FDID_newR2.m``
under Octave at ``benchmarks/octave/fdid_hongkong.m``. Nothing here is
transcribed from her readme.

Running both buys three things.

Her two implementations are checked against each other, which nobody else does.
They agree to exactly 0.0 on every quantity both compute.

The MATLAB script computes inference her R script does not: the standardized
ATT, the standard error, the two-sided p-value and the 95 percent interval, for
the forward and the conventional fit alike. Those are now pinned, so the case
covers the uncertainty and not only the point estimate.

And the Octave run reproduces every figure her readme prints, to every digit she
printed: ``ATT_FDID = 0.025405``, ``ATT_std_FDID = 5.4941``,
``p_value_forward_DID = 3.9274e-08``, ``CI_95_FDID = [0.016342, 0.034468]``,
``ATT_DID = 0.031721``, ``ATT_std_DID = 3.8647``, ``p_value_DID = 0.00011122``,
nine controls selected.

One deliberate divergence, decomposed rather than tolerated. Li hardcodes 1.96
as the normal quantile (her ``FDID_Matlab.m`` line 49); mlsynth uses
``norm.ppf(0.975) = 1.959963984540``. That shifts each interval endpoint by
``3.601546e-05 * se``, which is 1.665e-07 on the forward fit and 2.956e-07 on the
conventional one -- matching the observed gaps of 1.67e-07 and 2.96e-07. So the
endpoints are pinned against Li's formula evaluated at mlsynth's own quantile,
where they agree to floating point, and the gap to her 1.96 interval is pinned
separately as the predicted quantity. A tolerance wide enough to swallow it would
have hidden a real drift of the same size.

Before this case ran at full precision, ``did_from_mean`` rounded its whole
return dict on the way out and ``results_assembly`` read the rounded vectors into
the typed result, so the tolerances here were 5e-4 and 2e-3 -- sized by mlsynth's
own display convention and nothing else. Li's code rounds nothing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlsynth import FDID

from benchmarks.reference import reference_value

# basedata/HongKong.csv lives at the repo root.
_DATA = Path(__file__).resolve().parents[2] / "basedata" / "HongKong.csv"


def _fit():
    df = pd.read_csv(_DATA)
    return FDID(
        {
            "df": df,
            "outcome": "GDP",
            "treat": "Integration",
            "unitid": "Country",
            "time": "Time",
            "display_graphs": False,
            "verbose": False,
        }
    ).fit()


Z_EXACT = 1.959963984540054   # scipy.stats.norm.ppf(0.975)
Z_LI = 1.96                   # her FDID_Matlab.m lines 49-50 and 77-78


def run() -> dict:
    res = _fit()
    f, d = res.fdid, res.did
    r = lambda k: reference_value("fdid_hongkong", k)

    out = {
        # Li's own two implementations against each other.
        "li_r_vs_matlab_max_abs_diff": max(
            abs(r(k) - r("m_" + k)) for k in
            ("fdid_att", "fdid_att_pct", "fdid_r2_pre", "fdid_n_controls",
             "did_att", "did_att_pct", "did_r2_pre")),
        # mlsynth against MATLAB, on everything both compute the same way.
        "fdid_att": float(f.att),
        "fdid_att_pct": float(f.att_percent),
        "fdid_r2_pre": float(f.r_squared),
        "fdid_n_controls": float(len(f.selected_names)),
        "fdid_satt": float(f.satt),
        "fdid_se": float(f.att_se),
        "fdid_p_value": float(f.p_value),
        "did_att": float(d.att),
        "did_att_pct": float(d.att_percent),
        "did_r2_pre": float(d.r_squared),
        "did_satt": float(d.satt),
        "did_se": float(d.att_se),
        "did_p_value": float(d.p_value),
    }

    # The interval, decomposed. First: mlsynth's endpoints against Li's formula
    # evaluated at mlsynth's own quantile, which is the like-for-like check.
    out["fdid_ci_residual_at_our_z"] = max(
        abs(float(f.ci[0]) - (float(f.att) - Z_EXACT * float(f.att_se))),
        abs(float(f.ci[1]) - (float(f.att) + Z_EXACT * float(f.att_se))))
    out["did_ci_residual_at_our_z"] = max(
        abs(float(d.ci[0]) - (float(d.att) - Z_EXACT * float(d.att_se))),
        abs(float(d.ci[1]) - (float(d.att) + Z_EXACT * float(d.att_se))))
    # Second: the gap to her 1.96 interval is (1.96 - z) * se, and nothing else.
    for tag, fit in (("fdid", f), ("did", d)):
        predicted = (Z_LI - Z_EXACT) * float(fit.att_se)
        observed = abs(float(fit.ci[0]) - r(f"m_{tag}_ci_lo"))
        out[f"{tag}_ci_gap_is_the_z_difference"] = abs(observed - predicted)
    return out


def comparison() -> dict:
    """mlsynth ``FDID`` against Li's MATLAB, quantity by quantity.

    Pairs the point estimates and the inference -- the standardized ATT, the
    standard error, the p-value and both interval endpoints -- for the forward
    and the conventional fit. The MATLAB side is the live captured run at
    ``benchmarks/reference/fdid_hongkong/``; the endpoints differ by her
    hardcoded 1.96 against the exact quantile, which the case pins separately.
    """
    res = _fit()
    f, d = res.fdid, res.did
    pairs = [
        ("FDID/ATT", float(f.att), "m_fdid_att"),
        ("FDID/%ATT", float(f.att_percent), "m_fdid_att_pct"),
        ("FDID/R2_pre", float(f.r_squared), "m_fdid_r2_pre"),
        ("FDID/n_controls", float(len(f.selected_names)), "m_fdid_n_controls"),
        ("FDID/SATT", float(f.satt), "m_fdid_satt"),
        ("FDID/SE", float(f.att_se), "m_fdid_se"),
        ("FDID/p_value", float(f.p_value), "m_fdid_p_value"),
        ("FDID/CI_lo", float(f.ci[0]), "m_fdid_ci_lo"),
        ("FDID/CI_hi", float(f.ci[1]), "m_fdid_ci_hi"),
        ("DID/ATT", float(d.att), "m_did_att"),
        ("DID/%ATT", float(d.att_percent), "m_did_att_pct"),
        ("DID/R2_pre", float(d.r_squared), "m_did_r2_pre"),
        ("DID/SATT", float(d.satt), "m_did_satt"),
        ("DID/SE", float(d.att_se), "m_did_se"),
        ("DID/p_value", float(d.p_value), "m_did_p_value"),
        ("DID/CI_lo", float(d.ci[0]), "m_did_ci_lo"),
        ("DID/CI_hi", float(d.ci[1]), "m_did_ci_hi"),
    ]
    rows = [{"quantity": q, "mlsynth": v,
             "reference": reference_value("fdid_hongkong", k)}
            for q, v, k in pairs]
    cfg = {"outcome": "GDP", "treat": "Integration", "unitid": "Country",
           "time": "Time"}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "FDID", "config": cfg},
        "reference": {
            "impl": ("Kathleen T. Li's FDID_Matlab.m + FDID_newR2.m under "
                     "Octave, and Fun_FDID.R under Rscript; both live runs"),
            "version": "Li (2024), Marketing Science, DOI 10.1287/mksc.2022.0212"},
    }


# Forward selection is deterministic, so both captured runs are exact re-runs,
# and the targets are read from the bundle rather than written here. The
# tolerances are what the implementations actually support: they used to be 5e-4
# and 2e-3 because mlsynth rounded its own output to 3-4 decimals on the way out,
# which is 7 to 9 orders of magnitude looser than the algorithms agree.
_fd = lambda k: reference_value("fdid_hongkong", k)
EXPECTED = {
    # Li's R and her MATLAB are bit-identical on every shared quantity.
    "li_r_vs_matlab_max_abs_diff": (0.0, 0.0),
    # The forward fit: measured at 5e-14 to 3e-13, pinned with BLAS headroom.
    "fdid_att": (_fd("m_fdid_att"), 1e-11),
    "fdid_att_pct": (_fd("m_fdid_att_pct"), 1e-11),
    "fdid_r2_pre": (_fd("m_fdid_r2_pre"), 1e-11),
    "fdid_n_controls": (_fd("m_fdid_n_controls"), 0.0),
    "fdid_satt": (_fd("m_fdid_satt"), 1e-11),
    "fdid_se": (_fd("m_fdid_se"), 1e-11),
    "fdid_p_value": (_fd("m_fdid_p_value"), 1e-11),
    # The conventional fit is a little looser: the donor average is summed in a
    # different order than R's and Octave's, so the ATT lands 1.9e-11 away and
    # the percentage amplifies that by 100/mean(yhat_post), about 3150x.
    "did_att": (_fd("m_did_att"), 1e-9),
    "did_att_pct": (_fd("m_did_att_pct"), 1e-6),
    "did_r2_pre": (_fd("m_did_r2_pre"), 1e-8),
    "did_satt": (_fd("m_did_satt"), 1e-8),
    "did_se": (_fd("m_did_se"), 1e-10),
    "did_p_value": (_fd("m_did_p_value"), 1e-11),
    # The interval agrees with Li's formula at our own quantile to floating point.
    "fdid_ci_residual_at_our_z": (0.0, 1e-15),
    "did_ci_residual_at_our_z": (0.0, 1e-15),
    # ...and the whole gap to her 1.96 interval is (1.96 - z) * se. Pinned at
    # zero residual: if the endpoints ever drift for any other reason, the
    # prediction stops accounting for the gap and this fails. The residual cannot
    # beat the ATT disagreement it sits on top of, so the conventional arm's
    # tolerance is set by its own 1.9e-11 summation difference and not by the
    # quantile arithmetic, which is exact.
    "fdid_ci_gap_is_the_z_difference": (0.0, 1e-12),
    "did_ci_gap_is_the_z_difference": (0.0, 1e-10),
}
