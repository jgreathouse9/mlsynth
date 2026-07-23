"""Path B (Monte Carlo): Ferman (2021, JASA) factor-structure recovery of the SC
estimator with many periods and many controls.

Validates mlsynth's outcome-only ``VanillaSC`` against the Monte Carlo of Ferman,
B. (2021), *"On the Properties of the Synthetic Control Estimator with Many
Periods and Many Controls"*, Journal of the American Statistical Association
116(536):1764-1772 -- Table 1 (p. 1771), the SC-estimator columns.

The setup.  Potential outcomes follow a linear factor model
``y_t = Mu F_t + eps_t`` with ``K = 2`` Gaussian AR(1) factors (``rho = 0.5``).
The treated unit and the first half of the donors load only on factor 1; the
second half of the donors load only on factor 2. To reconstruct the treated
unit's factor structure, the synthetic control must place (asymptotically) all of
its weight on the first-group donors. Treatment has no effect, so the estimated
one-period-ahead effect *is* the bias, and its Monte-Carlo SD is ``se(alpha)``.

Ferman's result (Proposition 3.1 / Corollary 3.1).  As both the number of donors
``J`` and pre-treatment periods ``T0`` grow, weights diluted across the growing
donor pool let the SC unit recover the treated factor structure *even though the
pre-treatment fit stays imperfect*: the weight on the treated unit's own factor
group ``E[mu01] -> 1`` (misallocation ``E[mu02] -> 0``) and ``se(alpha)`` shrinks
toward the idiosyncratic-shock SD. The paper's foil is unconstrained OLS: it also
drives ``E[mu01] -> 1`` but its ``se(alpha)`` *grows* with ``J`` (the simplex
constraint is what buys SC its precision).

What this case checks.

1. Identity.  On one fixed draw, mlsynth's outcome-only ``VanillaSC``
   (``BilevelSCM('outcome-only')``, the same solver ``VanillaSC(backend=...)``
   dispatches to) reproduces the weights of R's ``solve.QP`` -- exactly the
   paper's ``aux.R`` ``synth_control_est`` -- value-for-value (baked in the
   reference bundle). mlsynth *is* Ferman's SC estimator.

2. Reproduction.  Running mlsynth's own solver over the paper's 4x2 (J, T0) grid
   (200 reps/cell, seed 47) reproduces Table 1's SC columns: ``E[mu01]`` climbs
   0.75 -> 0.94 and strictly increases in ``J``; ``E[mu02] -> ~0.06``; ``se(alpha)``
   shrinks from ``J=4`` to ``J=100`` while the OLS-unconstrained ``se(alpha)`` grows
   ~3x -- so SC's ``se(alpha)`` is about a fifth of OLS's at ``J=100`` (Panel A).

The pinned targets are the *published* Table 1 numbers (reference bundle,
independently cross-checked against the authors' supplementary ``results.csv``);
tolerances absorb the Monte-Carlo gap between this 200-rep run and the paper's
5,000-rep study. Deterministic (seeded).
"""
from __future__ import annotations

import warnings

import numpy as np

from benchmarks.reference import load_reference

_REF = load_reference("ferman_manyperiods")
_V = _REF["values"]

# --- factor-model DGP, ported verbatim from Ferman (2021) main.R -------------
_RHO, _VAR_U, _VAR_EPS, _K = 0.5, 1 - 0.5 ** 2, 1.0, 2
_GRID = [("A", J, J + 5) for J in (4, 10, 50, 100)] + \
        [("B", J, 2 * J) for J in (4, 10, 50, 100)]
_NREPS, _SEED = 200, 47


def _ar1(T, rng):
    """Gaussian AR(1) started from its stationary distribution (aux.R)."""
    y = [rng.normal(0.0, np.sqrt(_VAR_U / (1 - _RHO ** 2)))]
    for t in range(T):
        y.append(_RHO * y[t] + rng.normal(0.0, np.sqrt(_VAR_U)))
    return np.array(y[1:])


def _gen(J, T0, rng, T1=1):
    """One panel: treated + first J//2 donors load on factor 1, rest on factor 2."""
    T = T0 + T1
    F = np.column_stack([_ar1(T, rng) for _ in range(_K)])
    Mu = np.zeros((J + 1, _K)); h = J // 2
    Mu[: 1 + h, 0] = 1.0
    Mu[1 + h:, 1] = 1.0
    eps = rng.normal(0.0, np.sqrt(_VAR_EPS), size=(T, J + 1))
    y = F @ Mu.T + eps
    return y[:T0], y[T0:], Mu


def _sc_weights(y_pre, Y0_pre):
    """mlsynth's outcome-only SC solver -- the fast path VanillaSC dispatches to
    (bit-identical to a full VanillaSC.fit()), skipping the dataframe/dataprep
    overhead so the Monte Carlo is affordable."""
    from mlsynth.utils.bilevel import BilevelSCM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(BilevelSCM("outcome-only").fit(y_pre, Y0_pre).W, float)


def _ols_effect(y_pre, Y0_pre, y1_post, Y0_post):
    """Unconstrained OLS foil (Ferman Table 1 columns 5-8)."""
    b = np.linalg.lstsq(Y0_pre, y_pre, rcond=None)[0]
    return float(y1_post - Y0_post @ b)


def _monte_carlo():
    rng = np.random.default_rng(_SEED)
    cells = {}
    for panel, J, T0 in _GRID:
        wl1, sc_eff, ols_eff = [], [], []
        for _ in range(_NREPS):
            yb, ya, Mu = _gen(J, T0, rng)
            w = _sc_weights(yb[:, 0], yb[:, 1:])
            wl1.append(w[Mu[1:, 0].astype(bool)].sum())
            sc_eff.append(float(ya[0, 0] - w @ ya[0, 1:]))
            ols_eff.append(_ols_effect(yb[:, 0], yb[:, 1:], ya[0, 0], ya[0, 1:]))
        wl1 = np.asarray(wl1); sc_eff = np.asarray(sc_eff); ols_eff = np.asarray(ols_eff)
        cells[(panel, J)] = {
            "Emu01": float(wl1.mean()), "Emu02": float(1.0 - wl1.mean()),
            "sealpha": float(np.sqrt(np.mean(sc_eff ** 2))),
            "ols_sealpha": float(np.sqrt(np.mean(ols_eff ** 2))),
        }
    return cells


def _identity_dev():
    """max |mlsynth outcome-only weights - baked R solve.QP weights| on the fixed
    draw -- the value-for-value tie to aux.R synth_control_est."""
    rng = np.random.default_rng(int(_V["idraw_seed"]))
    yb, _, _ = _gen(int(_V["idraw_J"]), int(_V["idraw_T0"]), rng)
    w = _sc_weights(yb[:, 0], yb[:, 1:])
    return float(np.max(np.abs(w - np.asarray(_REF["idraw_weights"], float))))


def run() -> dict:
    cells = _monte_carlo()
    A = {J: cells[("A", J)] for J in (4, 10, 50, 100)}
    B = {J: cells[("B", J)] for J in (4, 10, 50, 100)}
    emu01_A = [A[J]["Emu01"] for J in (4, 10, 50, 100)]
    emu01_B = [B[J]["Emu01"] for J in (4, 10, 50, 100)]
    return {
        # 1. identity: mlsynth == R solve.QP (aux.R synth_control_est)
        "idraw_vs_solveqp_weight_dev": _identity_dev(),
        # 2. reproduction: factor recovery grows with J (Proposition 3.1)
        "B_J4_Emu01": B[4]["Emu01"],
        "B_J100_Emu01": B[100]["Emu01"],
        "A_J100_Emu01": A[100]["Emu01"],
        "B_J100_Emu02": B[100]["Emu02"],
        "factor_recovery_increasing": float(
            all(np.diff(emu01_A) > 0) and all(np.diff(emu01_B) > 0)),
        # 3. precision: SC se(alpha) shrinks, OLS se(alpha) grows (Corollary 3.1)
        "sc_sealpha_shrinks": float(A[4]["sealpha"] - A[100]["sealpha"]),
        "ols_sealpha_grows": float(A[100]["ols_sealpha"] - A[4]["ols_sealpha"]),
        "sc_over_ols_sealpha_J100": float(A[100]["sealpha"] / A[100]["ols_sealpha"]),
    }


def comparison() -> dict:
    """mlsynth's outcome-only SC vs Ferman (2021) Table 1, quantity by quantity."""
    cells = _monte_carlo()
    rows = []
    # factor recovery E[mu01] across the grid (Panel A then Panel B)
    for panel, J, T0 in _GRID:
        rows.append({"quantity": f"E[mu01] Panel{panel} J={J}",
                     "mlsynth": round(cells[(panel, J)]["Emu01"], 3),
                     "reference": _V[f"{panel}_J{J}_Emu01"]})
    # precision contrast at the endpoints (Panel A): SC se(alpha) shrinks, OLS grows
    for J in (4, 100):
        rows.append({"quantity": f"se(alpha) SC PanelA J={J}",
                     "mlsynth": round(cells[("A", J)]["sealpha"], 3),
                     "reference": _V[f"A_J{J}_sealpha"]})
        rows.append({"quantity": f"se(alpha) OLS PanelA J={J}",
                     "mlsynth": round(cells[("A", J)]["ols_sealpha"], 3),
                     "reference": _V[f"A_J{J}_ols_sealpha"]})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC", "config": {"backend": "outcome-only"}},
        "reference": {"impl": "Ferman (2021) JASA Table 1 (SC columns 1-4, OLS se col 5-8)",
                      "version": "doi:10.1080/01621459.2021.1965613"},
    }


# Targets are Ferman (2021) Table 1 (reference bundle). The E[mu] tolerances
# absorb the Monte-Carlo gap between this 200-rep run and the paper's 5,000-rep
# study (worst observed cell |Δ| ~ 0.017); the se(alpha)-contrast tolerances are
# wider because the OLS effect SD is itself high-variance (the paper's point).
# The identity dev is a value-for-value order-statistic match (float/solver noise
# only). factor_recovery_increasing is a strict-monotonicity flag (holds by wide
# margins). Deterministic at seed 47.
EXPECTED = {
    "idraw_vs_solveqp_weight_dev": (0.0, 1e-5),
    "B_J4_Emu01": (_V["B_J4_Emu01"], 0.03),
    "B_J100_Emu01": (_V["B_J100_Emu01"], 0.02),
    "A_J100_Emu01": (_V["A_J100_Emu01"], 0.02),
    "B_J100_Emu02": (_V["B_J100_Emu02"], 0.02),
    "factor_recovery_increasing": (1.0, 0.0),
    "sc_sealpha_shrinks": (_V["A_J4_sealpha"] - _V["A_J100_sealpha"], 0.13),
    "ols_sealpha_grows": (_V["A_J100_ols_sealpha"] - _V["A_J4_ols_sealpha"], 1.6),
    "sc_over_ols_sealpha_J100": (_V["A_J100_sealpha"] / _V["A_J100_ols_sealpha"], 0.10),
}
