"""Path C: does Proposition 4.3 hold, cross-sectionally and in panel geometry?

Part 1 reproduces the paper's claim on the authors' own LaLonde data, which is
the cross-validation against the reference implementation.

Part 2 asks the question mlsynth actually cares about: the same claim in the
geometry a synthetic control lives in, where the source units are donors and
the features are pre-treatment periods. Panels invert the aspect ratio the
paper works in -- a donor pool is wide relative to its pre-window as often as
not -- so the check sweeps both sides of J = T0.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from prop43 import (  # noqa: E402
    augmented_estimate, gamma_map, generalized_ridge,
    l2_balancing_weights, ols_plugin,
)

LALONDE = pathlib.Path(
    "/tmp/claude-0/-home-user-mlsynth/9a535672-c663-54d2-bdc6-d437774bd5f3"
    "/scratchpad/balance-equiv-jrssb/data"
)


def _equivalence(Xp, Yp, phi_q, lam, delta):
    """Augmented estimate and its Proposition 4.3 single-ridge counterpart."""
    d = Xp.shape[1]
    evals, _ = np.linalg.eigh(Xp.T @ Xp / Xp.shape[0])
    lam_vec = np.full(d, lam)

    beta_lam = generalized_ridge(Xp, Yp, lam_vec)
    w = l2_balancing_weights(Xp, phi_q, delta)
    aug = augmented_estimate(Xp, Yp, phi_q, w, beta_lam)

    plug = float(phi_q @ generalized_ridge(Xp, Yp, gamma_map(evals, lam_vec, delta)))
    return aug, plug


def factor_panel(n_donors, T0, seed, rank=3):
    """A donor pool and one treated unit from a low-rank factor model."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T0 + 1, rank))          # last row is the post period
    L = rng.normal(size=(n_donors + 1, rank))
    Y = F @ L.T + 0.1 * rng.normal(size=(T0 + 1, n_donors + 1))
    y_pre, Y0_pre = Y[:T0, 0], Y[:T0, 1:]
    Y0_post = Y[T0, 1:]
    return y_pre, Y0_pre, Y0_post


def part1_lalonde():
    print("Part 1 -- the authors' LaLonde data (cross-sectional)")
    print(f"{'features':>9} {'lambda':>8} {'delta':>8} {'augmented':>15}"
          f" {'gen-ridge':>15} {'abs diff':>10}")
    worst = 0.0
    for tag, label in (("_small", 11), ("", 171)):
        X = np.loadtxt(LALONDE / f"nsw_psid_X{tag}.csv", delimiter=",",
                       skiprows=1, usecols=range(1, label + 1))
        Y = np.loadtxt(LALONDE / f"nsw_psid_Y{tag}.csv", delimiter=",",
                       skiprows=1, usecols=(1,))
        T = np.loadtxt(LALONDE / f"nsw_psid_T{tag}.csv", delimiter=",",
                       skiprows=1, usecols=(1,))
        Xp, Yp, phi_q = X[T == 0], Y[T == 0], X[T == 1].mean(axis=0)
        for lam in (0.1, 1.0, 10.0, 100.0):
            for delta in (0.01, 0.5, 5.0, 50.0):
                a, p = _equivalence(Xp, Yp, phi_q, lam, delta)
                worst = max(worst, abs(a - p))
                if delta in (0.5, 50.0) and lam in (1.0, 100.0):
                    print(f"{label:>9} {lam:8.2f} {delta:8.2f} {a:15.6f}"
                          f" {p:15.6f} {abs(a - p):10.2e}")
        ols = ols_plugin(Xp, Yp, phi_q)
        a0, _ = _equivalence(Xp, Yp, phi_q, 1.0, 1e-12)
        print(f"  {label:>3} features, delta -> 0: augmented {a0:.6f}"
              f"  vs OLS {ols:.6f}  diff {abs(a0 - ols):.2e}")
    print(f"  worst |augmented - gen-ridge| over 32 cells: {worst:.3e}\n")
    return worst


def part2_panel():
    print("Part 2 -- panel geometry (donors = units, pre-periods = features)")
    print(f"{'donors':>7} {'T0':>5} {'lambda':>8} {'delta':>8} {'augmented':>13}"
          f" {'gen-ridge':>13} {'abs diff':>10}")
    worst = 0.0
    collapse = []
    for n_donors, T0 in ((40, 12), (12, 40), (30, 30)):
        y_pre, Y0_pre, Y0_post = factor_panel(n_donors, T0, seed=11)
        Xp, Yp, phi_q = Y0_pre.T, Y0_post, y_pre   # donors x pre-periods
        for lam in (0.01, 1.0, 100.0):
            for delta in (0.001, 0.1, 10.0):
                a, p = _equivalence(Xp, Yp, phi_q, lam, delta)
                worst = max(worst, abs(a - p))
                if delta in (0.1,) and lam in (0.01, 100.0):
                    print(f"{n_donors:>7} {T0:>5} {lam:8.2f} {delta:8.3f}"
                          f" {a:13.6f} {p:13.6f} {abs(a - p):10.2e}")
        ols = ols_plugin(Xp, Yp, phi_q)
        a0, _ = _equivalence(Xp, Yp, phi_q, 1.0, 1e-12)
        collapse.append((n_donors, T0, a0, ols, abs(a0 - ols)))
    print(f"  worst |augmented - gen-ridge| over 27 cells: {worst:.3e}")
    print("\n  delta -> 0 collapse to the OLS plug-in:")
    for n_donors, T0, a0, ols, diff in collapse:
        note = "" if n_donors >= T0 else "   (J < T0: OLS underdetermined)"
        print(f"    J={n_donors:>3} T0={T0:>3}  augmented {a0:12.6f}"
              f"  OLS {ols:12.6f}  diff {diff:9.2e}{note}")
    return worst


if __name__ == "__main__":
    w1 = part1_lalonde()
    w2 = part2_panel()
    print(f"\nProposition 4.3 holds to {max(w1, w2):.2e} in both geometries.")
