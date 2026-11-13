"""Path B: the paper's Table 1, and mlsynth's HQF against it, change by change.

Table 1 (Wang, Li, So & Liu 2023): m = n = 500, rank m/50 = 10, U, V standard
normal, Gaussian-mixture noise (10% of entries at 10x the standard deviation)
at SNR 3-15 dB; RPCA-HQF with maxiter 100, ip 3, lam 0.001 (main.m); RMSE of
the low-rank estimate against M; "average results by 20 independent runs".

Arms, all from the same matrix and the same random U initialisation per run:

  authors        port.rpca_hqf with the authors' settings: their code, ported
  +proper_prox   lam I and + lam U_p (mlsynth's update)
  +recompute     no monotone min, no 10x start (mlsynth's threshold path)
  +reset_stall   stagnation counter resets (mlsynth's stopping rule)
  mlsynth@paper  mlsynth's hqf_decompose with rank 10, ip 3, lam 0.001,
                 max_iter 100 -- the library at the paper's settings
  mlsynth@default hqf_decompose as CLUSTERSC calls it: ip 1,
                 lam 1/sqrt(max(m,n)), max_iter 1000, rank by cumvar 0.999

The library draws its own U; it is pinned to the arm's U0 by replacing
`np.random.default_rng` inside `hqf` for the call, so the comparison is on
identical starts.

Usage: python run_tableone.py RUNS OUT.parquet [M]
"""
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import port  # noqa: E402
from mlsynth.utils.clustersc_helpers.rpca import hqf as hqf_mod  # noqa: E402

PAPER = {500: {3: (0.1508, 0.0027), 6: (0.1074, 0.0019), 9: (0.0757, 0.0006),
               12: (0.0534, 0.0006), 15: (0.0383, 0.0008)},
         1000: {3: (0.2141, 0.0020), 6: (0.1514, 0.0012), 9: (0.1072, 0.0008),
                12: (0.0757, 0.0006), 15: (0.0536, 0.0004)}}


class _FixedRng:
    def __init__(self, U0):
        self.U0 = U0

    def random(self, shape):
        assert tuple(shape) == self.U0.shape
        return self.U0.copy()


def _library(M_noise, U0, **kw):
    with mock.patch.object(hqf_mod.np.random, "default_rng", lambda *_a, **_k: _FixedRng(U0)):
        return hqf_mod.hqf_decompose(M_noise, **kw)


def one(m, snr, run):
    rng = np.random.default_rng([m, snr, run])
    rak = m // 50
    U = rng.standard_normal((m, rak))
    V = rng.standard_normal((m, rak))
    M = U @ V.T
    M_noise = M + port.gaussian_noise_gm(M, snr, rng)
    U0 = rng.random((m, rak))
    score = lambda X: float(np.linalg.norm(X - M) / np.sqrt(m * m))
    out = {}
    base = dict(maxiter=100, ip=3.0, lam_1=0.001, U0=U0)
    out["authors"] = port.rpca_hqf(M_noise, rak, **base)
    out["+proper_prox"] = port.rpca_hqf(M_noise, rak, **base, proper_prox=True)
    out["+recompute"] = port.rpca_hqf(M_noise, rak, **base, proper_prox=True,
                                      monotone=False, start10=False)
    out["+reset_stall"] = port.rpca_hqf(M_noise, rak, **base, proper_prox=True,
                                        monotone=False, start10=False, reset_stall=True)
    r = _library(M_noise, U0, rank=rak, ip=3.0, lam=0.001, max_iter=100)
    out["mlsynth@paper"] = (r.low_rank, r.iterations)
    U0_lib = rng.random((m, hqf_mod._select_rank(M_noise, 0.999)))
    r = _library(M_noise, U0_lib)
    out["mlsynth@default"] = (r.low_rank, r.iterations)
    rows = [dict(m=m, snr=snr, run=run, arm=k, rmse=score(X), iters=int(it))
            for k, (X, it) in out.items()]
    rows.append(dict(m=m, snr=snr, run=run, arm="default_rank", rmse=float("nan"),
                     iters=int(r.rank_used)))
    # the library at the paper's settings against the port with the same changes
    rows.append(dict(m=m, snr=snr, run=run, arm="lib_vs_port_maxabs",
                     rmse=float(np.max(np.abs(out["mlsynth@paper"][0] - out["+reset_stall"][0]))),
                     iters=0))
    return rows


if __name__ == "__main__":
    runs, out = int(sys.argv[1]), Path(sys.argv[2])
    sizes = [int(sys.argv[3])] if len(sys.argv) > 3 else [500]
    rows, t0 = [], time.perf_counter()
    for m in sizes:
        for snr in (3, 6, 9, 12, 15):
            for run in range(runs):
                rows += one(m, snr, run)
            pd.DataFrame(rows).to_parquet(out, index=False)
            print(f"m={m} snr={snr} done {time.perf_counter() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    arms = ["authors", "+proper_prox", "+recompute", "+reset_stall", "mlsynth@paper", "mlsynth@default"]
    tab = df[df.arm.isin(arms)].pivot_table(index=["m", "snr"], columns="arm", values="rmse",
                                            aggfunc="mean")[arms]
    tab.insert(0, "paper", [PAPER[m][s][0] for m, s in tab.index])
    print(tab.round(4).to_string())
    print("\nmax |library - port with the same changes| per cell:")
    print(df[df.arm == "lib_vs_port_maxabs"].groupby(["m", "snr"]).rmse.max().to_string())
    print("\ndefault rank picked (true 10/20):")
    print(df[df.arm == "default_rank"].groupby(["m", "snr"]).iters.median().to_string())
