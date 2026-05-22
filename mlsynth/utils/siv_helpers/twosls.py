"""2SLS / Wald estimators on debiased SIV series.

The SIV paper considers three estimator variants once the SC step
has produced ``\\tilde Y, \\tilde R, \\tilde Z``:

* ``"siv"``   — fully debiased: ``\\hat\\theta = (\\sum \\tilde Z
                   \\tilde R)^{-1} \\sum \\tilde Z \\tilde Y``.
* ``"siv_z"`` — debiases only the instrument: ``(\\sum \\tilde Z R)^{-1}
                   \\sum \\tilde Z Y``.
* ``"siv_yr"`` — debiases only outcome and treatment, leaves Z raw:
                   ``(\\sum Z \\tilde R)^{-1} \\sum Z \\tilde Y``.

The orchestrator always computes the canonical ``"siv"`` variant
and additionally returns the others as diagnostic estimates for the
user. Each variant is just-identified, so the IV/2SLS formula
collapses to the ratio above.
"""

from __future__ import annotations

import numpy as np

from .structures import SIVEstimate


def _stack_post(*arrays: np.ndarray, T0: int) -> tuple[np.ndarray, ...]:
    """Take ``[:, T0:]`` and flatten to ``(J * T1,)`` for every input."""
    return tuple(a[:, T0:].reshape(-1) for a in arrays)


def two_sls_just_identified(
    Y: np.ndarray, R: np.ndarray, Z: np.ndarray,
    T0: int, variant: str,
) -> SIVEstimate:
    """Compute a single 2SLS estimate on debiased series.

    Parameters
    ----------
    Y, R, Z : np.ndarray
        Either the raw ``(J, T)`` outcomes/treatment/instrument or
        their SC-debiased versions, as the caller chose for the given
        variant.
    T0 : int
        Last pre-treatment column (exclusive); only ``[T0:]`` enters
        the moment conditions.
    variant : str
        Tag stored on the returned estimate for downstream consumers.

    Returns
    -------
    SIVEstimate
        Just-identified 2SLS point estimate plus heteroskedasticity-
        robust standard error and the first-stage diagnostics.
    """

    y_flat, r_flat, z_flat = _stack_post(Y, R, Z, T0=T0)
    n = y_flat.size

    # Reduced form: y ~ z (just-identified)
    z_y = float(z_flat @ y_flat)
    z_r = float(z_flat @ r_flat)
    z_z = float(z_flat @ z_flat)
    if abs(z_z) < 1e-12 or abs(z_r) < 1e-12:
        return SIVEstimate(
            variant=variant,
            theta_hat=float("nan"),
            se=float("nan"),
            pi_hat=float("nan"),
            beta_first_stage=float("nan"),
            f_stat=float("nan"),
            n_post_obs=int(n),
        )

    pi_hat = z_y / z_z                  # reduced-form coefficient
    beta_first = z_r / z_z              # first-stage coefficient
    theta_hat = z_y / z_r               # = pi_hat / beta_first

    # First-stage residuals + F statistic
    first_stage_resid = r_flat - beta_first * z_flat
    rss = float(first_stage_resid @ first_stage_resid)
    tss = float(r_flat @ r_flat)
    if tss > 0:
        f_stat = (tss - rss) / max(rss / (n - 1), 1e-12)
    else:
        f_stat = float("nan")

    # Heteroskedasticity-robust SE for the IV estimator
    iv_resid = y_flat - theta_hat * r_flat
    # Sandwich: theta has asymptotic variance
    #   (Z'R / n)^{-1} (Z' diag(eps^2) Z / n) (R'Z / n)^{-1}
    score = z_flat * iv_resid
    s_zz = float(score @ score)
    denom = (z_r / n) ** 2
    if denom > 0:
        avar = s_zz / (n * n * denom)
        se = float(np.sqrt(max(avar, 0.0)))
    else:
        se = float("nan")

    return SIVEstimate(
        variant=variant,
        theta_hat=float(theta_hat),
        se=se,
        pi_hat=float(pi_hat),
        beta_first_stage=float(beta_first),
        f_stat=float(f_stat),
        n_post_obs=int(n),
    )
