"""Path B -- Masini & Medeiros (2021) Tables 2-3, the partial-resampling test's size.

The size designs of *"Counterfactual Analysis With Artificial Controls:
Inference, High Dimensions, and Nonstationarity"*, JASA 116(536), 1773-1788,
Section 5. Their Equations (15)-(17): 200 units over 100 periods, six of them
loading on a common factor and the rest pure noise, the treated unit first. The
factor is trend-stationary ``F_t = t + U^F_t`` in Table 2 and a driftless unit
root ``F_t = F_{t-1} + U^F_t`` in Table 3. No intervention is added, so the
rejection rate at a nominal level is the test's size. The statistic is
:math:`\\phi(x) = \\|x\\|_2` and the penalty is chosen by BIC, as they specify.

Each table reports three arms, and the three are what make this design
diagnostic: ``True`` uses the pseudo-true :math:`\\theta_0` with no estimation at
all, ``Oracle`` runs OLS on the five relevant units, and ``LASSO`` selects from
all 199 controls. Reading across them separates the test from the first stage
that feeds it.

What reproduces
---------------
The two arms without selection reproduce. ``True`` comes back at 0.055 against
their 0.0583, and ``Oracle`` at 0.080 and 0.0775 against their 0.0755 and
0.0770 -- inside Monte Carlo error at this replication count. The single
post-intervention period, which is the claim the procedure exists to support,
holds: with ``T1 = 1`` the ``True`` arm sizes at 0.0625 against their 0.0562,
so a test built on one post-treatment observation is correctly sized.

What does not
-------------
The ``LASSO`` arm rejects at 0.140 where they report 0.0637, and at 0.118 where
they report 0.0824 -- a factor of 2.2 and 1.4. The divergence is confined to
the arm that selects; nothing else in the design moves.

The mechanism is measurable. Under the null the observed statistic and the
null draws estimate the same quantity, so their ratio is the finite-sample
error the theorem's partial asymptotics send to zero. That ratio is 0.99 for
``True``, 1.08 for ``Oracle`` and 1.19 for ``LASSO``, and the size rises with
it monotonically, 0.055 to 0.080 to 0.140. The null draws are in-sample
residuals from one fit on the whole pre-period while the observed statistic is
an out-of-sample gap, and selecting five donors out of 199 buys more in-sample
fit than regressing on the five true ones. Their reported cells put the
``LASSO`` arm at or below ``Oracle``, which requires their first stage to shrink
about as much as an OLS on the true support.

What was eliminated
-------------------
The divergence survived five checks. The design was transcribed a second time
in MATLAB syntax and drawn under Octave, and the Python estimator on those
panels returns the same rejection rates (0.135 / 0.130 against the 0.140 /
0.118 here), so it is not the data-generating transcription. The resampling
test was cross-checked against the authors' ``ressampling.m`` run verbatim
under Octave and agrees to machine precision, and it sizes exactly when no
estimation intervenes, so it is not the test. Widening the penalty path from a
1e-2 to a 1e-6 ratio of smallest to largest penalty moves the size by 0.01 and
the ratio not at all. Building their Section 4 grid literally -- ``lambda_max =
||(1/T0) sum Y_t X_t||_inf``, exponential to ``lambda_min = 0.001`` over 100
points -- returns 0.143. Coordinate descent reaches its tolerance on every draw.

The remaining candidate is the first stage's effective shrinkage, and the
paper's simulation code is not in the replication package -- only the empirical
application is -- so it cannot be settled from what shipped. The empirical
application reproduces value-for-value (``arco_lasa``).

One property of their baseline design bears on any follow-up: all six loading
units take ``mu_i = 1`` on a common trend, so the relevant controls are 0.9988
correlated at ``T = 100`` and 0.99995 at ``T = 500``. The restricted eigenvalue
their Theorem 1 rates depend on degenerates as the sample grows, and the
selected-donor count falls from 5.8 to 3.2 over that range.
"""
from __future__ import annotations

import numpy as np

from benchmarks.masini_common import (
    partial_resampling,
    simulate_masini_panel,
    wlasso,
)

S0 = 5          # relevant controls, as the paper sets it
M = 400         # replications per cell (the paper runs 10,000)
T = 100
T1_MAIN = 3


def _phi(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x ** 2)))


def _cell(design: str, T1: int, arm: str, seed: int) -> tuple:
    """Rejection rate at the 5% level and the observed/null statistic ratio."""
    T0 = T - T1
    p_values, null_means, observed = [], [], []
    for m in range(M):
        Z = simulate_masini_panel(np.random.default_rng(seed + m), design=design, T=T)
        y, X = Z[:, 0], Z[:, 1:]

        if arm == "true":                       # theta_0: 1/s0 on the relevant units
            beta = np.zeros(X.shape[1])
            beta[:S0] = 1.0 / S0
            counterfactual = X @ beta
        elif arm == "oracle":                   # OLS on the s0 relevant units
            A = np.column_stack([np.ones(T0), X[:T0, :S0]])
            b, *_ = np.linalg.lstsq(A, y[:T0], rcond=None)
            counterfactual = b[0] + X[:, :S0] @ b[1:]
        else:                                   # LASSO over all 199 controls
            b0, beta, _, _ = wlasso(y[:T0], X[:T0])
            counterfactual = b0 + X @ beta

        gap = y[T0:] - counterfactual[T0:]
        p, _, _, draws = partial_resampling(
            _phi, y[:T0] - counterfactual[:T0], gap)
        p_values.append(p)
        null_means.append(draws.mean())
        observed.append(_phi(gap))

    p = np.asarray(p_values)
    return (float(np.mean(p <= 0.05)),
            float(np.mean(observed) / np.mean(null_means)))


def run() -> dict:
    seed3 = 4000 + 97 * T1_MAIN
    seed1 = 4000 + 97 * 1

    true_det, ratio_true = _cell("deterministic", T1_MAIN, "true", seed3)
    orac_det, ratio_orac = _cell("deterministic", T1_MAIN, "oracle", seed3)
    lass_det, ratio_lass = _cell("deterministic", T1_MAIN, "lasso", seed3)

    true_sto, _ = _cell("stochastic", T1_MAIN, "true", seed3)
    orac_sto, _ = _cell("stochastic", T1_MAIN, "oracle", seed3)
    lass_sto, _ = _cell("stochastic", T1_MAIN, "lasso", seed3)

    true_t1, _ = _cell("deterministic", 1, "true", seed1)
    lass_t1, _ = _cell("deterministic", 1, "lasso", seed1)

    return {
        "size_true_det": true_det,
        "size_oracle_det": orac_det,
        "size_lasso_det": lass_det,
        "size_true_stoch": true_sto,
        "size_oracle_stoch": orac_sto,
        "size_lasso_stoch": lass_sto,
        "size_true_t1_1": true_t1,
        "size_lasso_t1_1": lass_t1,
        "ratio_true": ratio_true,
        "ratio_oracle": ratio_orac,
        "ratio_lasso": ratio_lass,
        # ordering indicators (1.0 == holds)
        "ratio_monotone": float(ratio_true < ratio_orac < ratio_lass),
        "lasso_exceeds_oracle": float(lass_det > orac_det + 0.03),
    }


# Every replication is seeded, so re-running returns identical numbers. At
# M = 400 the binomial standard error of a size cell is 0.011 at p = 0.055 and
# 0.017 at p = 0.14, so 0.04 on the size cells is between two and four of them
# and absorbs the Monte Carlo noise and nothing else. The ratios are means over
# 400 draws and carry far less, so they get 0.03.
#
# The `true` and `oracle` cells are pinned against the paper's own published
# values, which they reproduce. The `lasso` cells are this port's measurements,
# with the published values quoted alongside as context and not as targets --
# see the module docstring for the five checks that did not close the gap.
EXPECTED = {
    "size_true_det": (0.0583, 0.04),      # the paper's cell
    "size_oracle_det": (0.0755, 0.04),    # the paper's cell
    "size_lasso_det": (0.140, 0.04),      # measured; paper 0.0637
    "size_true_stoch": (0.0611, 0.04),    # the paper's cell
    "size_oracle_stoch": (0.0770, 0.04),  # the paper's cell
    "size_lasso_stoch": (0.118, 0.04),    # measured; paper 0.0824
    "size_true_t1_1": (0.0562, 0.04),     # the paper's cell; one post period
    "size_lasso_t1_1": (0.098, 0.04),     # measured; paper 0.0583
    "ratio_true": (0.993, 0.03),
    "ratio_oracle": (1.078, 0.03),
    "ratio_lasso": (1.190, 0.03),
    "ratio_monotone": (1.0, 0.0),
    "lasso_exceeds_oracle": (1.0, 0.0),
}
