"""One simulation draw; the three closed forms against what is measured.

    python closed_form_plot.py

    naive      - oracle = -v_k * pi
    iterative  - oracle =  v_k * e - v_k * l1 * tau
    iscm       - oracle =  v_k * (e + delta * l1) / (1 - v_k * l1)

l1 is controlled directly by building k*'s replacement as a convex combination
of its clean rebuild and the treated aggregate, so the leak can be swept.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlsynth.utils.solvers.simplex import simplex_lstsq
from dgps import DGPS
from run import design

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
COL = {"naive": "#2a78d6", "iterative": "#eb6834", "iscm": "#1baf7a"}

# ---- one draw -------------------------------------------------------------
YN, _, T0 = DGPS["marex_native"](0)
T, J = YN.shape
w, v_full = design(YN, T0, max(2, J // 6))
treated = np.flatnonzero(w > 1e-8)
controls = np.flatnonzero(v_full > 1e-12)
kstar = int(np.argmax(v_full)); vk = float(v_full[kstar])
clean = np.array([j for j in controls if j != kstar])
scale = float(np.median(YN[:T0].std(axis=0)))
post = slice(T0, T)

fit = simplex_lstsq(YN[:T0, clean], YN[:T0, kstar])


def run(pi, l1, tau):
    """Measured errors for one (pi, l1, tau), plus the quantities the forms use."""
    Y = YN.copy()
    Y[post, treated] += tau                      # treatment
    Y[post, kstar] += pi                         # contamination
    A = Y @ w                                    # treated aggregate

    YNt = YN.copy(); YNt[post, treated] += tau   # uncontaminated counterpart
    oracle = float(np.mean((YNt @ w)[post] - YNt[post] @ v_full))
    naive = float(np.mean(A[post] - Y[post] @ v_full))

    # k*'s replacement: clean rebuild blended with the treated aggregate at l1
    C = Y[:, clean] @ fit
    R = (1 - l1) * C + l1 * A
    Yi = Y.copy(); Yi[post, kstar] = R[post]
    iterative = float(np.mean(A[post] - Yi[post] @ v_full))

    gamma = float(np.mean(Y[post, kstar] - R[post]))
    Om = np.array([[1.0, -vk], [-l1, 1.0]])
    iscm = float(np.linalg.solve(Om, np.array([naive, gamma]))[0])

    # e: rebuild error of the UNTREATED path at this l1. The blend must use the
    # UNTREATED treated-aggregate; using the treated one folds l1*tau into e and
    # the leak gets counted twice.
    CN = YN[:, clean] @ fit
    RN = (1 - l1) * CN + l1 * (YN @ w)
    e = float(np.mean(YN[post, kstar] - RN[post]))
    delta = oracle - tau
    return dict(naive=naive - oracle, iterative=iterative - oracle,
                iscm=iscm - oracle, e=e, delta=delta, oracle=oracle)


def forms(r, pi, l1, tau):
    return dict(naive=-vk * pi,
                iterative=vk * r["e"] - vk * l1 * tau,
                iscm=vk * (r["e"] + r["delta"] * l1) / (1 - vk * l1))


TAU = scale
fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.0), facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF)
    a.grid(True, color="#e6e5e2", lw=0.8, zorder=0)
    a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): a.spines[sp].set_color("#d9d8d4")
    a.tick_params(colors=INK2, labelsize=10)

# --- A: sweep pi at l1 = 0 -------------------------------------------------
pis = np.linspace(0, 4 * scale, 17)
meas = {k: [] for k in COL}; pred = {k: [] for k in COL}
for pi in pis:
    r = run(pi, 0.0, TAU); f = forms(r, pi, 0.0, TAU)
    for k in COL: meas[k].append(r[k]); pred[k].append(f[k])
for k in COL:
    ax[0].plot(pis / scale, pred[k], color=COL[k], lw=2, zorder=2)
    ax[0].plot(pis / scale, meas[k], ls="none", marker="o", ms=8, mfc=COL[k],
               mec=SURF, mew=2, zorder=3)
ax[0].set_title("A.  No leak (l₁ = 0): the naive bias is −vₖ·π",
                color=INK, fontsize=12, loc="left", pad=12)
ax[0].set_xlabel("contamination π  (panel scales)", color=INK2, fontsize=10)
ax[0].set_ylabel("estimate − oracle", color=INK2, fontsize=10)
ax[0].annotate("naive", (pis[-1] / scale, meas["naive"][-1]), xytext=(-6, -16),
               textcoords="offset points", ha="right", color=INK, fontsize=10)
ax[0].annotate("iterative = iSCM", (pis[-1] / scale, meas["iterative"][-1]),
               xytext=(-6, 10), textcoords="offset points", ha="right",
               color=INK, fontsize=10)

# --- B: sweep l1 at fixed pi ----------------------------------------------
l1s = np.linspace(0, 0.8, 17); PI = 2 * scale
meas = {k: [] for k in COL}; pred = {k: [] for k in COL}
for l1 in l1s:
    r = run(PI, l1, TAU); f = forms(r, PI, l1, TAU)
    for k in COL: meas[k].append(r[k]); pred[k].append(f[k])
for k in COL:
    ax[1].plot(l1s, pred[k], color=COL[k], lw=2, zorder=2)
    ax[1].plot(l1s, meas[k], ls="none", marker="o", ms=8, mfc=COL[k],
               mec=SURF, mew=2, zorder=3)
ax[1].set_title("B.  Treated markets in the rebuild: iterative leaks −vₖ·l₁·τ",
                color=INK, fontsize=12, loc="left", pad=12)
ax[1].set_xlabel("l₁  (treated weight in k*'s rebuild)", color=INK2, fontsize=10)
ax[1].set_ylabel("estimate − oracle", color=INK2, fontsize=10)
for k, lab, dy in (("naive", "naive", 10), ("iterative", "iterative", -16),
                   ("iscm", "iSCM", 10)):
    ax[1].annotate(lab, (l1s[-1], meas[k][-1]), xytext=(-6, dy),
                   textcoords="offset points", ha="right", color=INK, fontsize=10)

# --- C: predicted vs measured over a grid ---------------------------------
P, Mv = {k: [] for k in COL}, {k: [] for k in COL}
for pi in np.linspace(0, 4 * scale, 7):
    for l1 in np.linspace(0, 0.8, 7):
        for tm in (0.5, 1.0, 2.0):
            r = run(pi, l1, tm * scale); f = forms(r, pi, l1, tm * scale)
            for k in COL: Mv[k].append(r[k]); P[k].append(f[k])
allv = np.concatenate([np.array(P[k]) for k in COL] + [np.array(Mv[k]) for k in COL])
lim = [allv.min() - 0.3, allv.max() + 0.3]
ax[2].plot(lim, lim, color="#b9b8b4", lw=1.5, ls="--", zorder=1)
for k in COL:
    ax[2].plot(P[k], Mv[k], ls="none", marker="o", ms=8, mfc=COL[k], mec=SURF,
               mew=1.5, alpha=0.85, zorder=2)
ax[2].set_xlim(lim); ax[2].set_ylim(lim)
worst = max(np.abs(np.array(P[k]) - np.array(Mv[k])).max() for k in COL)
ax[2].set_title(f"C.  441 settings: closed form vs measured (max |gap| {worst:.1e})",
                color=INK, fontsize=12, loc="left", pad=12)
ax[2].set_xlabel("closed form", color=INK2, fontsize=10)
ax[2].set_ylabel("measured", color=INK2, fontsize=10)
ax[2].annotate("45°", (lim[1], lim[1]), xytext=(-24, -16),
               textcoords="offset points", color=INK2, fontsize=10)

h = [plt.Line2D([], [], color=COL[k], lw=2, marker="o", ms=8, mec=SURF, mew=1.5)
     for k in COL]
fig.legend(h, ["naive", "iterative", "iSCM"], loc="lower center", ncol=3,
           frameon=False, fontsize=11, labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f"Closed forms on one draw  (marex_native, seed 0; vₖ = {vk:.3f}, "
             f"τ = {TAU:.2f}, lines are the formulas, dots are measured)",
             color=INK, fontsize=13, y=1.00, x=0.012, ha="left")
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
out = "results/closed_form.png"
fig.savefig(out, dpi=170, facecolor=SURF, bbox_inches="tight")
print(f"vk={vk:.4f} tau={TAU:.4f} worst_abs_gap={worst:.3e}")
for k in COL:
    print(f"  {k:10} max|form-measured| = {np.abs(np.array(P[k])-np.array(Mv[k])).max():.3e}")
print("saved", out)
