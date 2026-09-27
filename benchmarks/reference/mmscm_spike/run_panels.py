"""All three of the paper's panels: MMSCM variants against competent baselines."""
import warnings; warnings.filterwarnings("ignore")
import os, sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
_REF = os.environ.get("MMSCM_REPO")   # a clone of github.com/MasaKat0/mmscm
if _REF:
    sys.path.insert(0, _REF)
import mmscm_oracle as O
try:
    from mmscm import MMSCM as RefMMSCM
except ImportError:
    RefMMSCM = None

B = str(pathlib.Path(__file__).resolve().parents[3] / "basedata") + "/"
G = 5

def panel_smoking():
    d = pd.read_csv(B + "smoking_data.csv")
    w = d.pivot(index="year", columns="state", values="cigsale").sort_index()
    return "Prop 99 (California, 1989)", w, "California", 1988, \
           d[["state", "year", "cigsale"]], ("state", "cigsale", "year")

def panel_basque():
    d = pd.read_csv(B + "basque_data.csv")
    d = d[d.regionname != "Spain (Espana)"]
    w = d.pivot(index="year", columns="regionname", values="gdpcap").sort_index()
    w = w.dropna(axis=1)
    return "Basque (ETA, 1970)", w, "Basque Country (Pais Vasco)", 1969, \
           d[["regionname", "year", "gdpcap"]], ("regionname", "gdpcap", "year")

def panel_germany():
    d = pd.read_stata(B + "repgermany.dta")
    w = d.pivot(index="year", columns="country", values="gdp").sort_index()
    return "German reunification (1990)", w, "West Germany", 1989, \
           d[["country", "year", "gdp"]], ("country", "gdp", "year")

for name, wd, treated, last_pre, longdf, (uid, out, tvar) in (
        panel_smoking(), panel_basque(), panel_germany()):
    cols = [treated] + [c for c in wd.columns if c != treated]
    Y = wd[cols].to_numpy(float); yrs = wd.index.to_numpy()
    T0 = int((yrs <= last_pre).sum()); pre = slice(0, T0); post = slice(T0, None)
    A, m0, _ = O.moment_design(Y[:T0], G)
    print(f"\n=== {name}:  J={Y.shape[1]-1} donors, T0={T0}, T1={Y.shape[0]-T0} ===")

    def row(label, w, bias=True):
        cf = O.counterfactual(Y, w, T0, bias=bias)
        att = float(np.mean(Y[post, 0] - cf[post]))
        rm = float(np.sqrt(np.mean((Y[pre, 0] - cf[pre]) ** 2)))
        print(f"  {label:<42} ATT={att:9.3f}  pre-RMSE={rm:8.3f}  active={int((w>1e-6).sum()):2d}")

    try:
        if RefMMSCM is None:
            raise ImportError("set MMSCM_REPO to a clone of MasaKat0/mmscm")
        m = RefMMSCM(longdf, "MMSCM", uid, treated, out, tvar, last_pre, poly=G+1)
        m.train_param(); row("reference mmscm.py (SLSQP)", m.res.x)
    except Exception as e:
        print(f"  reference mmscm.py failed: {type(e).__name__}: {str(e)[:60]}")
    for wt in ("unit", "scaled"):
        v = O.moment_weights(m0, wt)
        for sel in ("minnorm", "pathfit"):
            row(f"MMSCM v={wt:<6} tie-break={sel}", O.fit(Y[:T0], G, sel, v=v)["w"])
    row("Abadie SC", O.abadie_sc(Y[:T0]), bias=False)
    w_dm = O.demeaned_sc(Y[:T0])
    cf = Y[:, 1:] @ w_dm; cf = cf + float(np.mean(Y[:T0, 0] - cf[:T0]))
    print(f"  {'demeaned SC (Ferman & Pinto)':<42} "
          f"ATT={np.mean(Y[post,0]-cf[post]):9.3f}  "
          f"pre-RMSE={np.sqrt(np.mean((Y[pre,0]-cf[pre])**2)):8.3f}  active={int((w_dm>1e-6).sum()):2d}")
