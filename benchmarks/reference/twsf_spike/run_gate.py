"""The Path-B gate: does v2's specified DGP give nominal coverage?

Reports, per (n, h): bias, RMSE, the ratio of empirical SD to mean plug-in SE
(diagnostic 2 -- is the variance formula right?), and coverage at nominal 90%.
Monte Carlo standard errors are clustered over the 100 latent replications, as
simulations.tex specifies.
"""
import warnings; warnings.filterwarnings("ignore")
import pathlib, sys, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dgp_v2 as D, twsf as T

N_LATENT = int(sys.argv[1]) if len(sys.argv) > 1 else 100
N_NOISE = int(sys.argv[2]) if len(sys.argv) > 2 else 10
NS = (25, 50, 75, 100, 125, 150)
HS = (1, 5, 10)
Z90 = 1.6448536269514722                            # Phi^{-1}(0.95)

A0, A1 = D.loadings()
res = {(n, h): {"err": [], "V": [], "cov": [], "cluster": []} for n in NS for h in HS}

t_start = time.time()
for s in range(N_LATENT):
    U, uN, V0, V1, Ts = D.latent(1000 + s, A0, A1)
    for n in NS:
        for r in range(N_NOISE):
            y0, Yd, Yp, tpath, d = D.panel(n, U, uN, V0, V1, Ts,
                                           noise_seed=900000 + s * 1000 + r)
            try:
                fit = T.fit_once(y0, Yd, Yp, d["L"], D.R_Y, D.R_Z)
            except Exception:
                continue
            for h in HS:
                theta, V = T.eval_horizon(fit, h)
                truth = float(np.mean(tpath[:h]))
                e = theta - truth
                cell = res[(n, h)]
                cell["err"].append(e); cell["V"].append(V)
                cell["cov"].append(abs(e) <= Z90 * V); cell["cluster"].append(s)
    if (s + 1) % 10 == 0:
        print(f"  latent {s+1}/{N_LATENT}  [{time.time()-t_start:.0f}s]", flush=True)

def clustered_se(x, cl):
    x, cl = np.asarray(x, float), np.asarray(cl)
    means = np.array([x[cl == c].mean() for c in np.unique(cl)])
    return float(means.std(ddof=1) / np.sqrt(means.size))

print(f"\nTWSF v2 Path-B gate: {N_LATENT} latent x {N_NOISE} noise = "
      f"{N_LATENT*N_NOISE} replications, nominal coverage 0.90\n")
hdr = (f"{'n':>5} {'h':>3} | {'bias':>9} {'(mcse)':>8} | {'RMSE':>9} | "
       f"{'emp SD':>9} {'plug SE':>9} {'SD/SE':>7} | {'coverage':>9} {'(mcse)':>8}")
print(hdr); print("-" * len(hdr))
for h in HS:
    for n in NS:
        c = res[(n, h)]
        if not c["err"]:
            print(f"{n:>5} {h:>3} |  (no valid replications)"); continue
        e = np.array(c["err"]); V = np.array(c["V"]); cov = np.array(c["cov"], float)
        print(f"{n:>5} {h:>3} | {e.mean():9.4f} {clustered_se(e, c['cluster']):8.4f} | "
              f"{np.sqrt((e**2).mean()):9.4f} | {e.std(ddof=1):9.4f} {V.mean():9.4f} "
              f"{e.std(ddof=1)/V.mean():7.3f} | {cov.mean():9.3f} "
              f"{clustered_se(cov, c['cluster']):8.3f}")
    print()

