"""Link A (aggregation) and Link C (method) candidates for the HZ ladder."""
import sys, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from methods import simulate, pca_counterfactual, cce_counterfactual, select_r_by_cv
from sklearn.linear_model import Lasso, LassoCV

def refit(y1, Yco, T0, keep):
    if keep.size == 0:
        return np.full(len(y1), y1[:T0].mean())
    X = np.column_stack([np.ones(T0), Yco[:T0][:, keep]])
    c, *_ = np.linalg.lstsq(X, y1[:T0], rcond=None)
    return np.column_stack([np.ones(len(y1)), Yco[:, keep]]) @ c

def pda_cv(y1, Yco, T0, seed):
    f = LassoCV(cv=max(3, min(10, T0 // 3)), max_iter=20000,
                random_state=seed).fit(Yco[:T0], y1[:T0])
    return refit(y1, Yco, T0, np.flatnonzero(np.abs(f.coef_) > 0)), f.alpha_

def pda_alpha(y1, Yco, T0, alpha):
    m = Lasso(alpha=alpha, max_iter=20000).fit(Yco[:T0], y1[:T0])
    return refit(y1, Yco, T0, np.flatnonzero(np.abs(m.coef_) > 0))

def pda_split(y1, Yco, T0, rng, G=2):
    """M2: random split into G groups, select within each, pool, reselect."""
    idx = rng.permutation(Yco.shape[1])
    pooled = []
    for g in np.array_split(idx, G):
        f = LassoCV(cv=max(3, min(10, T0 // 3)), max_iter=20000,
                    random_state=0).fit(Yco[:T0][:, g], y1[:T0])
        pooled.extend(g[np.abs(f.coef_) > 0])
    pooled = np.array(sorted(set(pooled)), dtype=int)
    if pooled.size == 0:
        return np.full(len(y1), y1[:T0].mean())
    f = LassoCV(cv=max(3, min(10, T0 // 3)), max_iter=20000,
                random_state=0).fit(Yco[:T0][:, pooled], y1[:T0])
    return refit(y1, Yco, T0, pooled[np.abs(f.coef_) > 0])

GRID = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
CELLS = (("dgp6", 30, 1.181, 1.717), ("dgp6", 50, 0.739, 1.712),
         ("dgp7", 30, 1.315, 1.873), ("dgp7", 50, 0.857, 1.800))
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 120

for dgp, T0, paper_pda, paper_pca in CELLS:
    rng = np.random.default_rng(0)
    acc = {f"alpha={a}": [] for a in GRID}
    acc["LassoCV"] = []; acc["A1 |mean e|"] = []
    acc["M2 split"] = []; acc["PCA r=2"] = []; acc["PCA r=CV"] = []
    rcv = []
    for _ in range(REPS):
        y1, Yco = simulate(dgp, 30, T0, 10, rng)
        post = slice(T0, T0 + 10)
        cf, _ = pda_cv(y1, Yco, T0, 0)
        e = y1[post] - cf[post]
        acc["LassoCV"].append(np.abs(e).mean())
        acc["A1 |mean e|"].append(abs(e.mean()))
        for a in GRID:
            c2 = pda_alpha(y1, Yco, T0, a)
            acc[f"alpha={a}"].append(np.abs(y1[post] - c2[post]).mean())
        s = pda_split(y1, Yco, T0, rng)
        acc["M2 split"].append(np.abs(y1[post] - s[post]).mean())
        p2 = pca_counterfactual(y1, Yco, T0, 2)
        acc["PCA r=2"].append(np.abs(y1[post] - p2[post]).mean())
        r = select_r_by_cv(Yco, T0); rcv.append(r)
        pc = pca_counterfactual(y1, Yco, T0, r)
        acc["PCA r=CV"].append(np.abs(y1[post] - pc[post]).mean())
    print(f"\n=== {dgp} T0={T0}  reps={REPS}   paper: PDA {paper_pda:.3f}  PCA {paper_pca:.3f}")
    best = min(GRID, key=lambda a: np.mean(acc[f"alpha={a}"]))
    for k in ["LassoCV"] + [f"alpha={a}" for a in GRID] + ["M2 split", "A1 |mean e|", "PCA r=2", "PCA r=CV"]:
        v = np.mean(acc[k]); tag = ""
        if k.startswith("alpha") and k == f"alpha={best}": tag = "  <- best on path"
        print(f"   {k:14} {v:8.3f}{tag}")
    print(f"   best-on-path {np.mean(acc[f'alpha={best}']):.3f} vs paper PDA {paper_pda:.3f}"
          f"  -> paper is {'REACHABLE' if np.mean(acc[f'alpha={best}']) <= paper_pda else 'BELOW my whole path'}")
    print(f"   mean CV r = {np.mean(rcv):.2f}")
