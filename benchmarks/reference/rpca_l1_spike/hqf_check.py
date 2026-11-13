"""HQF as shipped against Wang, Li, So & Liu (2023) Algorithm 1, on the spike's panels.

Output is in hqf_check.log. Run from any directory: python hqf_check.py
"""
import sys, warnings, numpy as np, pandas as pd
warnings.simplefilter("ignore")
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2]))
import run_sim as RS
from mlsynth.utils.clustersc_helpers.rpca.hqf import hqf_decompose
from mlsynth.utils.clustersc_helpers.rpca.pcp import pcp_decompose

def hqf_paper(X, r, zeta=3.0, lam=1e-3, Im=100, eta=1e-6, seed=0):
    """Wang, Li, So & Liu (2023) Algorithm 1 as printed (with the U/V update signs that minimise (14)-(15))."""
    m, n = X.shape
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((r, n))
    for _ in range(3):                                   # PowerFactorization, In = 3
        U = X @ np.linalg.pinv(V); V = np.linalg.pinv(U) @ X
    e = np.inf; S = np.zeros_like(X); Rn_prev = np.inf
    for _ in range(Im):
        R = X - U @ V
        sig = 1.4815 * np.median(np.abs(R - np.median(R)))
        e = min(zeta * sig, e)                          # eq. (13)
        S = np.where(np.abs(R) >= e, R, 0.0)            # eq. (10)-(11)
        D = X - S
        U = (D @ V.T + lam * U) @ np.linalg.inv(V @ V.T + lam * np.eye(r))
        V = np.linalg.inv(U.T @ U + lam * np.eye(r)) @ (U.T @ D + lam * V)
        Rn = np.linalg.norm(X - U @ V - S)
        if abs(Rn_prev - Rn) / np.sqrt(m * n) < eta: break
        Rn_prev = Rn
    return U @ V

rows = []
for (sig, pat) in [(0.0,"sparse5"),(0.0,"sparse15"),(0.2,"sparse5"),(0.2,"sparse15"),(0.2,"spells"),(0.5,"sparse5"),(0.5,"spells"),(0.2,"none")]:
    for s in range(10):
        Yd, yt, st, Sd = RS.panel(7000 + s, sig, pat)
        fits = {"hqf_lib(ip=1,cumvar)": hqf_decompose(Yd, random_state=s).low_rank,
                "hqf_lib(ip=3,cumvar)": hqf_decompose(Yd, ip=3.0, random_state=s).low_rank,
                "hqf_lib(ip=3,rank3)": hqf_decompose(Yd, ip=3.0, rank=3, random_state=s).low_rank,
                "hqf_paper(rank3)": hqf_paper(Yd, 3, seed=s),
                "pcp": pcp_decompose(Yd, max_iter=20000).low_rank}
        for k, L in fits.items():
            cf = RS._fit(L, yt)
            rows.append(dict(sigma=sig, pattern=pat, method=k, post=np.sqrt(np.mean((cf[RS.T0:] - st[RS.T0:])**2))))
    r_used = hqf_decompose(Yd).rank_used
    print(sig, pat, "lib cumvar rank on last seed:", r_used, flush=True)
d = pd.DataFrame(rows)
print(d.pivot_table(index=["sigma","pattern"], columns="method", values="post", aggfunc="median").round(3).to_string())
