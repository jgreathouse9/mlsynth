"""PDA Path-B: Shi & Wang (2024) L2-relaxation size/power simulation (Table 2).

Reproduces the geometry of the L2-relaxation PDA paper's single-treated-unit
size/power study (Shi & Wang 2024, Table 2a): on a **dense strong-factor** DGP
with ``N = 100`` controls, the post-selection test's empirical **size
approaches the nominal 5% as the sample grows** (it over-rejects in small
samples), and the test is **powerful** once there is a real effect.

The DGP is
:func:`mlsynth.utils.pda_helpers.simulation.simulate_shiwang_panel` (Sec. 5.1-5.2):
four unit-variance factors (i.i.d. / AR(1) 0.9 / MA(2) / ARMA(1,1)), strong
loadings ``U([-0.5,-0.3] U [0.3,0.5])`` on every unit, idiosyncratic ``N(0,0.5)``;
the treated unit gets shock ``D1`` (zero, for size) or ``D4`` (a +0.3 mean shift,
for power). The L2-relaxation is tuned over a coarse log-spaced ``tau`` grid
(the full per-fit CV would make the Monte Carlo too slow) with the authors'
standardisation.

  =================  ===============  ==================
  Quantity           mlsynth l2       Shi & Wang Table 2a
  =================  ===============  ==================
  size (D1), T1=50   ~0.20            0.142
  size (D1), T1=200  ~0.03            0.072
  power (D4), T1=50  ~0.57            0.570
  =================  ===============  ==================

The size **shrinks toward nominal as T1 grows** (the paper's point) and the
power matches closely; the small-sample size level runs a little high at this
modest ``M`` and coarse grid. Path B (scenario 1): the DGP is re-implemented
from the paper; the case asserts the geometry, not exact cells.
"""
from __future__ import annotations

import warnings

import numpy as np

from mlsynth.utils.pda_helpers.simulation import simulate_shiwang_panel

N = 100
M = 30
_GRID = np.logspace(-4.0, 0.0, 12)        # coarse CV grid (runtime)


def _rejection_rate(T1, shock, base_seed):
    from mlsynth.utils.pda_helpers.l2.estimation import fit_l2
    from mlsynth.utils.pda_helpers.l2.inference import l2_ate_inference
    rej = 0
    for i in range(M):
        y, Yc, T0 = simulate_shiwang_panel(N=N, T1=T1, shock=shock,
                                           seed=base_seed + i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, cf, _ = fit_l2(y, Yc.T, T0, tau_grid=_GRID, standardize=True)
            _, _, _, p = l2_ate_inference(y, cf, T0)
        rej += int(p < 0.05)
    return rej / M


def run() -> dict:
    size_50 = _rejection_rate(50, "D1", 5000)
    size_200 = _rejection_rate(200, "D1", 7000)
    power_50 = _rejection_rate(50, "D4", 9000)
    return {
        "l2_size_t50": size_50,
        "l2_size_t200": size_200,
        "l2_power_t50_d4": power_50,
        "size_shrinks_with_T": float(size_50 > size_200),
    }


# Deterministic (seeded). Tolerances absorb the binomial MC noise at M=30
# (size SE ~ sqrt(.15*.85/30) ~ 0.065) and the gap to Shi & Wang's M=10000 +
# the coarse tau grid. The robust, reproduced facts are: the size shrinks
# toward nominal as T1 grows (over-rejecting in small samples, like the paper),
# and the test is powerful (D4 power ~ the paper's 0.570).
EXPECTED = {
    "l2_size_t50": (0.20, 0.13),          # paper 0.142 (over-rejects, small T1)
    "l2_size_t200": (0.10, 0.09),         # paper 0.072 (shrinks toward nominal)
    "l2_power_t50_d4": (0.567, 0.18),     # paper 0.570
    "size_shrinks_with_T": (1.0, 0.0),
}
