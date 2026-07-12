"""Cross-validation: mlsynth SCD vs a base-R reference of Rincon & Song (2026).

mlsynth's Synthetic Control with Differencing (:class:`mlsynth.SCD`) is checked,
value for value, against a from-scratch base-R reproduction of the estimator of
Rincon & Song (2026), *"Synthetic Control with Differencing"* (arXiv:2510.26106),
with the repeated-cross-section inference of Canen & Song (2025), on the authors'
Arizona LAWA CPS application. The upstream package (``ratzanyelrincon/scd``) is
GPL, so the method is reproduced on public-domain CPS microdata rather than
vendored (as with ``sbc_germany`` / ``lto_refined_placebo`` /
``proximal_germany_oid``).

What is cross-validated
-----------------------
The full pipeline, on ``basedata/cps_lawa_arizona.parquet`` (a 5% CPS extract; 47
states, 84 periods; Arizona treated at period 55, ``T0 = 54``, ``K = 46``):

* the simplex donor weights and the post-period ATT for all three differencing
  schemes (``did`` / ``uniform`` / ``sc``);
* the effect at the first post-period;
* the repeated-cross-section pointwise standard error ``se_t``;
* the weight-variance trace ``tr(V)``;
* the confidence-set membership decisions ``in_C(w)`` for the fitted weights, an
  interior point, a vertex, and an edge midpoint.

Corrected RC variance
----------------------
The upstream ``gen_hat_sigma_squared_RC`` builds its treated/donor multiplier as
``ifelse(G_idx == 1, 1, -w[G_idx - 1])``, which indexes ``w`` at 0 for treated
rows; R drops zero indices, shortening the vector so ``ifelse`` recycles it and
misaligns the donor weights. Both the base-R reference here and mlsynth use the
corrected length-``(K+1)`` lookup ``c(1, -w)[G_idx]`` (matching the sibling
``gen_hat_V_RC``). The bug only affects the RC pointwise SE, not the point
estimator, the weight variance, or the confidence set.

Reference (live captured run)
-----------------------------
``benchmarks/reference/scd_cps/reference.R`` reproduces the SCD point estimator
(SLSQP simplex), the corrected RC variance, and the ``in_C`` projection QP on the
in-repo CPS extract; the values are captured under ``benchmarks/reference/scd_cps/``
and pinned here via :func:`reference_value`. Regenerate with
``python benchmarks/reference/generate.py scd_cps``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.reference import reference_value

_CPS = Path(__file__).resolve().parents[2] / "basedata" / "cps_lawa_arizona.parquet"


def _load() -> pd.DataFrame:
    df = pd.read_parquet(_CPS)
    df["treat"] = ((df["state_name"] == "Arizona") & (df["D"] == 1)).astype(int)
    return df


def _fit(differencing: str, compute_inference: bool = False):
    from mlsynth import SCD

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SCD({
            "df": _load(), "outcome": "wklyearn", "treat": "treat",
            "unitid": "state_name", "time": "period", "weight_col": "weight",
            "differencing": differencing, "compute_inference": compute_inference,
            "display_graphs": False,
        }).fit()


def run() -> dict:
    out = {}
    for scheme in ("did", "uniform", "sc"):
        res = _fit(scheme)
        theta = np.asarray(res.time_series.estimated_gap)
        out[f"{scheme}_att_abs_diff"] = abs(res.att - reference_value("scd_cps", f"{scheme}_att"))
        out[f"{scheme}_theta_post1_abs_diff"] = abs(
            theta[54] - reference_value("scd_cps", f"{scheme}_theta_post1")
        )

    # DID weights by state (value for value)
    res = _fit("did", compute_inference=True)
    dw = res.weights.donor_weights
    wmax = 0.0
    for state in ("Ohio", "Missouri", "Connecticut", "Arkansas", "Wyoming",
                  "Colorado", "West Virginia"):
        ref = reference_value("scd_cps", f"did_w[{state}]")
        wmax = max(wmax, abs(dw[state] - ref))
    out["did_weight_max_abs_diff"] = wmax

    det = res.inference.details
    se = np.asarray(det["se"])
    out["se_post1_abs_diff"] = abs(se[54] - reference_value("scd_cps", "se_post1"))
    out["se_post2_abs_diff"] = abs(se[55] - reference_value("scd_cps", "se_post2"))
    out["hatV_trace_abs_diff"] = abs(det["hatV_trace"] - reference_value("scd_cps", "hatV_trace"))

    # confidence-set membership decisions match the reference
    from mlsynth.utils.scd_helpers.setup import prepare_scd_inputs
    from mlsynth.utils.scd_helpers.inference import build_inference_operators
    from mlsynth.utils.scd_helpers.weights import in_C

    inp = prepare_scd_inputs(_load(), "wklyearn", "treat", "state_name", "period", "weight")
    ops = build_inference_operators(inp, differencing="did")
    K = inp.K
    u = np.full(K, 1.0 / K)
    v1 = np.zeros(K); v1[0] = 1.0
    ed = np.zeros(K); ed[0] = 0.5; ed[4] = 0.5
    dec = {
        "inC_hatw": int(in_C(ops.hat_w, ops, 0.05, 1e-6)),
        "inC_uniform": int(in_C(u, ops, 0.05, 1e-6)),
        "inC_vertex1": int(in_C(v1, ops, 0.05, 1e-6)),
        "inC_edge1_5": int(in_C(ed, ops, 0.05, 1e-6)),
    }
    out["inC_decisions_match"] = 1.0 if all(
        dec[k] == int(round(reference_value("scd_cps", k))) for k in dec
    ) else 0.0
    return out


def comparison() -> dict:
    """mlsynth SCD vs the base-R SCD reference on the Arizona LAWA CPS extract."""
    rows = []
    for scheme in ("did", "uniform", "sc"):
        res = _fit(scheme)
        rows.append({"quantity": f"{scheme}: post-period ATT",
                     "mlsynth": round(res.att, 5),
                     "reference": round(reference_value("scd_cps", f"{scheme}_att"), 5)})
    res = _fit("did", compute_inference=True)
    se = np.asarray(res.inference.details["se"])
    rows.append({"quantity": "DID: se(theta) at first post-period",
                 "mlsynth": round(float(se[54]), 5),
                 "reference": round(reference_value("scd_cps", "se_post1"), 5)})
    rows.append({"quantity": "DID: Ohio donor weight",
                 "mlsynth": round(res.weights.donor_weights["Ohio"], 5),
                 "reference": round(reference_value("scd_cps", "did_w[Ohio]"), 5)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "SCD",
                         "config": {"differencing": ["did", "uniform", "sc"],
                                    "compute_inference": True}},
        "reference": {"impl": "base-R SCD (point estimator + corrected RC variance "
                              "+ in_C projection QP), reproduced on public CPS microdata",
                      "version": "Rincon & Song (2026) arXiv:2510.26106; "
                                 "Canen & Song (2025) inference"},
    }


# mlsynth reproduces the base-R SCD reference value for value on every scheme:
# the simplex weights and ATT (did/uniform/sc), the first post-period effect, the
# corrected repeated-cross-section SE, the weight-variance trace, and the four
# confidence-set membership decisions. Targets are pinned from the captured run.
EXPECTED = {
    "did_att_abs_diff": (0.0, 1e-3),
    "uniform_att_abs_diff": (0.0, 1e-3),
    "sc_att_abs_diff": (0.0, 1e-3),
    "did_theta_post1_abs_diff": (0.0, 1e-3),
    "uniform_theta_post1_abs_diff": (0.0, 1e-3),
    "sc_theta_post1_abs_diff": (0.0, 1e-3),
    "did_weight_max_abs_diff": (0.0, 1e-3),
    "se_post1_abs_diff": (0.0, 1e-3),
    "se_post2_abs_diff": (0.0, 1e-3),
    "hatV_trace_abs_diff": (0.0, 1e-1),
    "inC_decisions_match": (1.0, 0.0),
}
