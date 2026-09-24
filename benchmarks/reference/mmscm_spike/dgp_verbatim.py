"""Kato & Ohda's own section 7.1 DGP, transcribed from their notebook.

Source: ``Figure2_Figure3_Simulation_Treatment_Effect.ipynb``, cell 3. Their
loop is reproduced exactly, including three details the paper's text does not
mention and which change what the simulation measures:

1. ``means`` and ``variances`` are drawn once per trial, not per period, so
   units differ persistently. The paper reads as though they are redrawn.
2. The outcome coordinate's mean and variance are bumped by ``N(0, 10)``
   *inside the loop over units*, so each period applies ``J + 1`` increments.
   The outcome is therefore a random walk with per-period step sd
   ``10 * sqrt(J + 1)`` -- which is not stationary, and their Assumption 5.7
   requires the error process to be stationary and strongly mixing.
3. ``data = data[data[1] != "2"]`` drops donor 2 after generation. The treated
   unit's mixture still uses that component, so the mixture assumption the
   estimator relies on is violated by the very panel it is tested on.

``total_period = 1050`` with ``intervention = 50`` gives T0 = 50 and T1 = 1000,
against the "T0 = 30, T1 = 100" of the paper's text.
"""
import numpy as np

DIM = 5
TREATMENT_EFFECT = 20.0


def draw_panel(n_donors: int, rng: np.random.Generator, total_period: int = 1050,
               intervention: int = 50, drop_unit_2: bool = True):
    """Return ``(Y, T0, w_star)``; ``Y`` is ``total_period x (1 + J)``, treated first."""
    means = rng.standard_normal((n_donors, DIM))
    variances = rng.uniform(1, 20, size=(n_donors, DIM))
    params = rng.uniform(0, 1, size=n_donors)
    params = params / params.sum()

    y_treated = np.empty(total_period)
    y_donors = np.empty((total_period, n_donors))

    for t in range(total_period):
        for i in range(-1, n_donors):
            sd = np.sqrt(np.maximum(variances, 1e-300))
            if i == -1:
                k = rng.choice(n_donors, p=params)
                val = means[k, 0] + sd[k, 0] * rng.standard_normal()
                if (t + 1) > intervention:
                    val += TREATMENT_EFFECT
                y_treated[t] = val
            else:
                y_donors[t, i] = means[i, 0] + sd[i, 0] * rng.standard_normal()
            # their bump, applied once per (t, unit)
            means[:, 0] += rng.normal(0, 10, size=n_donors)
            variances[:, 0] += rng.normal(0, 10, size=n_donors)
            variances[variances < 0.1] = 0.1

    Y = np.column_stack([y_treated, y_donors])
    if drop_unit_2 and n_donors > 2:
        keep = [0] + [1 + j for j in range(n_donors) if j != 2]
        Y = Y[:, keep]
    return Y, intervention, params
