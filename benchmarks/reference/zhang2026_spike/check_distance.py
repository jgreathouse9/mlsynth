"""Two checks the paper's own text makes falsifiable.

1. Population limit. For the Section 8.1 design, g(alpha, Gamma) = alpha/(1-rho)
   plus a term common to all units, so g(a1,.) - g(a2,.) = 5(a1-a2) is constant
   in Gamma and eq. (4.2) collapses to

       d(a_i, a_j) = sup_{a1,a2} |5(a1-a2) * 5(a_i-a_j)| = 25 * 0.5 * |a_i-a_j|
                   = 12.5 |a_i - a_j|.

   So d-hat should converge to 12.5 |alpha_i - alpha_j| as T0 grows.

2. Trend robustness (Section 3 remark). Adding a common lambda_t must leave
   zhang_range unchanged and must move feng_maxabs.
"""

from __future__ import annotations

import numpy as np

from dgp import additive_fe
from distance import pseudo_distance


def offdiag(M: np.ndarray) -> np.ndarray:
    return M[~np.eye(M.shape[0], dtype=bool)]


print("check 1 -- convergence to 12.5 |alpha_i - alpha_j|")
print(f"  {'T0':>5}  {'slope':>7}  {'corr':>6}  {'median |d-hat - 12.5 dalpha|':>30}")
for T0 in (20, 50, 200, 800):
    rng = np.random.default_rng(11)
    panel = additive_fe(120, T0, n_post=1, rng=rng)
    d = pseudo_distance(panel.Y[:, :T0], "zhang_range")
    target = np.abs(panel.alpha[:, None] - panel.alpha[None, :])
    x, y = offdiag(target), offdiag(d)
    slope = float(x @ y / (x @ x))
    corr = float(np.corrcoef(x, y)[0, 1])
    print(f"  {T0:5d}  {slope:7.3f}  {corr:6.3f}  {np.median(np.abs(y - 12.5 * x)):30.4f}")

print()
print("check 2 -- invariance to an additive common time trend")
rng = np.random.default_rng(7)
panel = additive_fe(120, 60, n_post=1, rng=rng)
block = panel.Y[:, :60]
trend = np.linspace(0.0, 40.0, 60)          # a large, nonstationary common trend
for metric in ("zhang_range", "feng_maxabs"):
    base = pseudo_distance(block, metric)
    moved = pseudo_distance(block + trend, metric)
    rel = np.abs(moved - base).max() / np.abs(base).max()
    print(f"  {metric:12s}  max relative change {rel:.3e}")
