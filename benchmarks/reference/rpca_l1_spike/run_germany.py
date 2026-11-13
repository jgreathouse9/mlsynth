"""The Bayani West-Germany cluster: the rank-k l1 fit beside RPCA-SC's denoisers.

Panel: Bayani's own West-Germany cluster (11 donors, 1960-2003) from the
vendored dissertation data, the same one benchmarks/cases/clustersc_rpca_germany.py
cross-validates. Weights are RPCA-SC's non-negative least squares.

Two exercises:

1. The real design (treatment 1990): ATT, pre-period RMSE, top weights, and for
   the l1 fit the criticality certificate, at k = 1..6.
2. An in-time placebo: a fake treatment in 1980 on the 1960-1989 window, so the
   ten "post" years are untreated and the counterfactual's RMSE against West
   Germany's actual GDP is prediction error.

Usage: python run_germany.py OUT.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

from benchmarks.reference.rpca_sc_reference import WEST_GERMANY_CLUSTER, _VENDOR  # noqa: E402
from l1rpca import certify, l1_lowrank  # noqa: E402
from mlsynth.utils.clustersc_helpers.rpca.hqf import hqf_decompose  # noqa: E402
from mlsynth.utils.clustersc_helpers.rpca.pcp import pcp_decompose  # noqa: E402
from mlsynth.utils.clustersc_helpers.rpca.weights import solve_nnls  # noqa: E402
from mlsynth.utils.pcr.core import hsvt  # noqa: E402
from mlsynth.utils.pcr.rank import select_rank  # noqa: E402

KS = (1, 2, 3, 4, 5, 6)


def load():
    wide = pd.read_csv(_VENDOR / "Data_Germany.csv", index_col=0)
    years = np.array([int(c) for c in wide.columns])
    out = wide.loc[WEST_GERMANY_CLUSTER].to_numpy(float)
    tr = WEST_GERMANY_CLUSTER.index("West Germany")
    donors = [c for c in WEST_GERMANY_CLUSTER if c != "West Germany"]
    return years, np.delete(out, tr, axis=0), out[tr], donors


def denoisers(M):
    fits = {
        "pcp_1000": pcp_decompose(M).low_rank,
        "pcp_conv": pcp_decompose(M, max_iter=20000).low_rank,
        "hqf": hqf_decompose(M).low_rank,
    }
    r = int(select_rank(M, method="usvt"))
    fits[f"hsvt_usvt(r={r})"] = hsvt(M, max(r, 1))[0]
    certs = {}
    for k in KS:
        f = l1_lowrank(M, k)
        fits[f"l1_k{k}"] = f.low_rank
        c = certify(M, f.low_rank, k)
        certs[f"l1_k{k}"] = {"status": c.status, "t_star": c.t_star,
                             "critical": c.critical, "residual_support": c.support_share}
    return fits, certs


def evaluate(fits, treated, n_pre, donors):
    rows = {}
    for name, L in fits.items():
        w = solve_nnls(L[:, :n_pre].T, treated[:n_pre])
        cf = L.T @ w
        top = sorted(((d, float(v)) for d, v in zip(donors, w) if v > 1e-4), key=lambda x: -x[1])
        rows[name] = {"post_gap_mean": float(np.mean((treated - cf)[n_pre:])),
                      "post_rmse": float(np.sqrt(np.mean((treated - cf)[n_pre:] ** 2))),
                      "pre_rmse": float(np.sqrt(np.mean((treated - cf)[:n_pre] ** 2))),
                      "weight_sum": float(w.sum()), "top_weights": top[:4]}
    return rows


def main(out):
    years, M, treated, donors = load()
    n_pre = int((years < 1990).sum())
    fits, certs = denoisers(M)
    real = evaluate(fits, treated, n_pre, donors)

    keep = years < 1990
    years_p, M_p, tr_p = years[keep], M[:, keep], treated[keep]
    fits_p, certs_p = denoisers(M_p)
    placebo = evaluate(fits_p, tr_p, int((years_p < 1980).sum()), donors)

    res = {"real_1990": real, "real_1990_certificates": certs,
           "placebo_1980": placebo, "placebo_1980_certificates": certs_p}
    Path(out).write_text(json.dumps(res, indent=2) + "\n")
    print(f"{'method':18s} {'ATT 1990':>9s} {'pre RMSE':>9s} {'wsum':>5s} | {'placebo RMSE':>12s}")
    for name in real:
        print(f"{name:18s} {real[name]['post_gap_mean']:9.0f} {real[name]['pre_rmse']:9.1f} "
              f"{real[name]['weight_sum']:5.2f} | {placebo[name]['post_rmse']:12.1f}")
    for k in KS:
        c, cp_ = certs[f"l1_k{k}"], certs_p[f"l1_k{k}"]
        print(f"l1 k={k}: certificate 1990 {c['status']} t*={c['t_star']:.3f} "
              f"support={c['residual_support']:.2f} | 1980 window {cp_['status']} t*={cp_['t_star']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
