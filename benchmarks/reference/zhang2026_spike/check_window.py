"""Does the length of the pseudo-distance window matter?

The paper builds its distance on 28 pre-treatment years (GFR 1937-1964, from
ICPSR 36603). The supplied Bailey (2012) replication package starts in 1959,
which leaves 6 common pre-1965 years. This prices that difference two ways:
distance quality, and the estimator that consumes it.
"""

from __future__ import annotations

import numpy as np

from dgp import additive_fe
from distance import pseudo_distance


def offdiag(M: np.ndarray) -> np.ndarray:
    return M[~np.eye(M.shape[0], dtype=bool)]


CONTEXTS = ((6, "Bailey only, common pre-1965"),
            (11, "Bailey only, per-cohort G3 max"),
            (19, "paper robustness, 1946-1964"),
            (28, "paper main, 1937-1964"),
            (200, "asymptotic reference"))

if __name__ == "__main__":
    print("pseudo-distance quality vs pre-period length (design 8.1; limit slope = 12.5)")
    print(f"{'T0':>4} {'context':<34} {'slope':>7} {'corr':>6} {'med err / med d':>16}")
    for T0, ctx in CONTEXTS:
        sl, co, re = [], [], []
        for seed in range(6):
            rng = np.random.default_rng(100 + seed)
            p = additive_fe(300, T0, n_post=1, rng=rng)
            d = pseudo_distance(p.Y[:, :T0], "zhang_range")
            tgt = np.abs(p.alpha[:, None] - p.alpha[None, :])
            x, y = offdiag(tgt), offdiag(d)
            sl.append(x @ y / (x @ x))
            co.append(np.corrcoef(x, y)[0, 1])
            re.append(np.median(np.abs(y - 12.5 * x)) / np.median(12.5 * x))
        print(f"{T0:>4} {ctx:<34} {np.mean(sl):7.2f} {np.mean(co):6.3f} {np.mean(re):16.2f}")
