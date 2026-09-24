"""Section 7.1 on Kato & Ohda's own DGP, with the baseline they never run."""
import warnings; warnings.filterwarnings("ignore")
import os, sys, pathlib, time, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mmscm_oracle as O
import dgp_verbatim as D

METHODS = ("MMSCM as solved", "MMSCM pathfit", "uniform 1/J", "Abadie SC", "demeaned SC")

def one(Y, T0, G):
    out = {}
    Jn = Y.shape[1] - 1
    w_mm = O.fit(Y[:T0], G, "none")["w"]
    out["_dist_to_uniform"] = float(np.abs(w_mm - 1.0 / Jn).sum())
    att = lambda w, bias=True: float(np.mean(
        Y[T0:, 0] - O.counterfactual(Y, w, T0, bias=bias)[T0:]))
    out["MMSCM as solved"] = att(w_mm)
    out["MMSCM pathfit"] = att(O.fit(Y[:T0], G, "pathfit")["w"])
    J = Y.shape[1] - 1
    out["uniform 1/J"] = att(np.full(J, 1.0 / J))   # no fitting at all
    out["Abadie SC"] = att(O.abadie_sc(Y[:T0]), bias=False)
    out["demeaned SC"] = att(O.demeaned_sc(Y[:T0]))
    return out

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
print(f"Kato & Ohda section 7.1, their own DGP (T0=50, T1=1000, tau=20), reps={REPS}")
print("cells are MSE of the estimated effect; the truth is 20\n")
hdr = f"{'J':>3} {'G':>3} | " + " | ".join(f"{m:^18}" for m in METHODS)
print(hdr); print("-" * len(hdr))
store = {}
for J in (5, 15, 30):
    for G in (2, 5, 10):
        t0 = time.time()
        errs = {m: [] for m in METHODS}
        rng = np.random.default_rng(9000 + J * 100 + G)
        for _ in range(REPS):
            Y, T0, _ = D.draw_panel(J, rng)
            try:
                r = one(Y, T0, G)
            except Exception:
                continue
            for m in METHODS:
                errs[m].append(r[m] - D.TREATMENT_EFFECT)
            errs.setdefault("_dist", []).append(r["_dist_to_uniform"])
        store[(J, G, "_dist")] = (float(np.mean(errs["_dist"])), 0.0)
        cells = []
        for m in METHODS:
            e = np.array(errs[m])
            store[(J, G, m)] = (float(np.mean(e**2)), float(np.mean(np.abs(e))))
            cells.append(f"{np.mean(e**2):10.1f}")
        print(f"{J:>3} {G:>3} | " + " | ".join(f"{c:^18}" for c in cells) + f"  [{time.time()-t0:.0f}s]")

print("\n=== how often does each method win the cell (lowest MSE)? ===")
wins = {m: 0 for m in METHODS}
for J in (5, 15, 30):
    for G in (2, 5, 10):
        best = min(METHODS, key=lambda m: store[(J, G, m)][0])
        wins[best] += 1
for m in METHODS:
    print(f"  {m:<18} {wins[m]}/9 cells")
print("\n=== how far are MMSCM's weights from uniform? (L1, max 2.0) ===")
for J in (5, 15, 30):
    print(f"  J={J:<3} " + "  ".join(
        f"G={G}: {store[(J,G,'_dist')][0]:.3f}" for G in (2, 5, 10)))

print("\n=== is MMSCM distinguishable from plain averaging? ===")
for J in (5, 15, 30):
    for G in (2, 5, 10):
        a = store[(J, G, "MMSCM as solved")][0]; u = store[(J, G, "uniform 1/J")][0]
        print(f"  J={J:<3} G={G:<3} MMSCM={a:10.1f}  uniform={u:10.1f}  ratio={a/u:6.3f}")

print("\n=== does MMSCM improve as G grows, as the paper reports? ===")
for J in (5, 15, 30):
    row = [store[(J, G, 'MMSCM as solved')][0] for G in (2, 5, 10)]
    row2 = [store[(J, G, 'MMSCM pathfit')][0] for G in (2, 5, 10)]
    print(f"  J={J:<3} as-solved G=2,5,10: " + " ".join(f"{v:9.1f}" for v in row)
          + "   | pathfit: " + " ".join(f"{v:9.1f}" for v in row2))
