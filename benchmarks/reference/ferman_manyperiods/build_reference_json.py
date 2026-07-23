"""Build ``reference.json`` for the ``ferman_manyperiods`` benchmark.

Two provenance streams feed the bundle:

1. The pinned Monte-Carlo targets are Ferman (2021, JASA 116(536):1764-1772)
   Table 1, the SC-estimator columns (1-4) and the OLS-unconstrained se(alpha)
   row (columns 5-8), transcribed verbatim from p. 1771. They were independently
   cross-checked by aggregating the authors' supplementary ``results.csv``
   (40,000 rows = 5,000 reps x 8 (J, T0) cells): the aggregation reproduces every
   Table 1 SC cell to <= 0.001 (e.g. Panel A E[mu01] = 0.760/0.817/0.905/0.929;
   se(alpha) = 1.288/1.194/1.084/1.073). The authors' supplement is *referenced*,
   not redistributed.

2. The identity-draw reference weights are generated clean-room here: the paper's
   factor-model DGP (main.R/aux.R) drawn once with a fixed NumPy seed, solved by
   R's ``solve.QP`` algorithm (Goldfarb-Idnani) via the ``quadprog`` package --
   exactly ``aux.R``'s ``synth_control_est``. The case regenerates the identical
   draw (same seed, same DGP) and checks mlsynth's outcome-only VanillaSC recovers
   these weights value-for-value.

Regenerate:  python benchmarks/reference/ferman_manyperiods/build_reference_json.py
(requires ``quadprog``; a build-time-only dependency, not used by the case.)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --- Ferman (2021) Table 1, p. 1771 (transcribed) ---------------------------
# SC estimator, Panel A (T0 = J+5) and Panel B (T0 = 2J), J = 4/10/50/100.
_VALUES = {
    # Panel A -- SC: sum of weights on the treated unit's own factor group (mu01),
    # on the other group (mu02), and se(alpha) (the one-period-ahead effect SD).
    "A_J4_Emu01": 0.760, "A_J10_Emu01": 0.817, "A_J50_Emu01": 0.905, "A_J100_Emu01": 0.929,
    "A_J4_Emu02": 0.240, "A_J10_Emu02": 0.183, "A_J50_Emu02": 0.095, "A_J100_Emu02": 0.071,
    "A_J4_sealpha": 1.288, "A_J10_sealpha": 1.194, "A_J50_sealpha": 1.084, "A_J100_sealpha": 1.073,
    # Panel A -- OLS unconstrained (columns 5-8): se(alpha) grows with J.
    "A_J4_ols_sealpha": 1.586, "A_J10_ols_sealpha": 1.984,
    "A_J50_ols_sealpha": 3.791, "A_J100_ols_sealpha": 5.220,
    # Panel B -- SC.
    "B_J4_Emu01": 0.753, "B_J10_Emu01": 0.831, "B_J50_Emu01": 0.922, "B_J100_Emu01": 0.944,
    "B_J4_Emu02": 0.247, "B_J10_Emu02": 0.169, "B_J50_Emu02": 0.078, "B_J100_Emu02": 0.056,
    # Identity draw descriptor (see below).
    "idraw_J": 10, "idraw_T0": 20, "idraw_seed": 12345,
}

# --- clean-room factor-model DGP (Ferman main.R) + solve.QP (aux.R) ----------
_RHO, _VAR_U, _VAR_EPS, _K = 0.5, 1 - 0.5 ** 2, 1.0, 2


def _ar1(T, rng):
    y = [rng.normal(0.0, np.sqrt(_VAR_U / (1 - _RHO ** 2)))]
    for t in range(T):
        y.append(_RHO * y[t] + rng.normal(0.0, np.sqrt(_VAR_U)))
    return np.array(y[1:])


def _gen(J, T0, rng, T1=1):
    T = T0 + T1
    F = np.column_stack([_ar1(T, rng) for _ in range(_K)])
    Mu = np.zeros((J + 1, _K)); h = J // 2
    Mu[: 1 + h, 0] = 1.0
    Mu[1 + h:, 1] = 1.0
    eps = rng.normal(0.0, np.sqrt(_VAR_EPS), size=(T, J + 1))
    y = F @ Mu.T + eps
    return y[:T0], y[T0:], Mu


def _solveqp(y_pre, Y0_pre):
    import quadprog
    X = Y0_pre
    D = X.T @ X + 1e-8 * np.eye(X.shape[1])
    d = X.T @ y_pre
    J = X.shape[1]
    C = np.column_stack([np.ones(J), np.eye(J)])
    b = np.concatenate([[1.0], np.zeros(J)])
    return np.maximum(quadprog.solve_qp(D, d, C, b, meq=1)[0], 0.0)


def main() -> None:
    rng = np.random.default_rng(_VALUES["idraw_seed"])
    yb, ya, _ = _gen(_VALUES["idraw_J"], _VALUES["idraw_T0"], rng)
    w_ref = _solveqp(yb[:, 0], yb[:, 1:])
    _VALUES["idraw_effect"] = float(ya[0, 0] - w_ref @ ya[0, 1:])
    out = {"values": _VALUES, "idraw_weights": [float(x) for x in w_ref]}
    path = Path(__file__).resolve().parent / "reference.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}  (idraw sum(w)={w_ref.sum():.6f}, effect={_VALUES['idraw_effect']:.6f})")


if __name__ == "__main__":
    main()
