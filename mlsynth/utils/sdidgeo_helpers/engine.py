"""The SDID scoring engine for SDIDGEO.

One synthetic difference-in-differences fit on a pseudo-experiment, plus the
placebo standard error its p-value is read against.

SDID (Arkhangelsky, Athey, Hirshberg, Imbens & Wager 2021) estimates the ATT as
a doubly weighted difference in differences: unit weights ``omega`` match the
donors to the treated pre-period trajectory, and time weights ``lambda`` match
the pre-periods to the post-period donor average. The estimate is

.. math::

   \\hat\\tau = \\big(\\bar y_{post} - \\lambda' y_{pre}\\big)
              - \\big(\\omega' \\bar Y_{0,post} - \\lambda' Y_{0,pre}\\omega\\big),

which rearranges to a per-period path: the counterfactual is
``Y0 @ omega + bias_correction`` with ``bias_correction = lambda' y_pre -
lambda' Y0_pre omega``, and the effect at ``t`` is the gap against it. That is
the same decomposition
:func:`mlsynth.utils.sdid_helpers.cohort.estimate_cohort_sdid_effects` uses, so
the design is scored by the estimator that will analyse the experiment.

Two structural properties make the power sweep cheap, and both are consequences
of where the weights get their information:

* :func:`~mlsynth.utils.sdid_helpers.weights.unit_weights` reads the treated
  *pre*-period trajectory, :func:`~mlsynth.utils.sdid_helpers.weights.fit_time_weights`
  reads only donors, and
  :func:`~mlsynth.utils.sdid_helpers.weights.compute_regularization` reads only
  donors and counts. No program sees the treated post block, so injecting an
  effect there cannot move ``omega`` or ``lambda`` and

  .. math:: \\hat\\tau(e) = \\hat\\tau(0) + e \\cdot \\bar y_{post}

  holds exactly. The effect-size sweep is arithmetic, not refitting.
* The placebo draws reassign *control* units as pseudo-treated, so the injected
  effect never enters them. One placebo run covers the whole sweep.

Both are pinned in ``tests/test_sdidgeo.py`` against brute-force recomputation.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..sdid_helpers.weights import (
    compute_regularization, fit_time_weights, unit_weights)


@dataclass
class SDIDFit:
    """One SDID fit on a pseudo-experiment's pre-period.

    Attributes
    ----------
    omega : np.ndarray, shape (J,)
        Unit (donor) weights; non-negative and summing to one.
    lam : np.ndarray, shape (T0,)
        Time weights over the placement's pre-period.
    bias_correction : float
        ``lambda' y_pre - lambda' Y0_pre omega``, the difference-in-differences
        level term that turns ``Y0 @ omega`` into a counterfactual path.
    counterfactual : np.ndarray, shape (T,)
        ``Y0 @ omega + bias_correction`` over every period of the panel.
    pre_rmspe : float
        Root mean squared pre-period gap against the counterfactual.
    scaled_l2 : float
        ``pre_rmspe`` as a fraction of the treated series' own pre-period
        standard deviation, so it compares across candidates of different size.
    zeta : float
        The unit-weight ridge (:func:`compute_regularization`). SDID's analogue
        of augmented SCM's penalty, but closed form instead of cross-validated.
    """

    omega: np.ndarray
    lam: np.ndarray
    bias_correction: float
    counterfactual: np.ndarray
    pre_rmspe: float
    scaled_l2: float
    zeta: float

    def tau_path(self, y) -> np.ndarray:
        """Per-period effect ``y - counterfactual`` over every period."""
        return np.asarray(y, dtype=float).ravel() - self.counterfactual


def sdid_fit_once(y, Y0, n_pre: int, start: int, end: int,
                  n_tr: int = 1) -> SDIDFit:
    """Fit SDID once on the placement ``[:n_pre]`` / ``[start:end+1]``.

    Parameters
    ----------
    y : array-like, shape (T,)
        Treated (aggregated candidate) outcomes over the full panel.
    Y0 : array-like, shape (T, J)
        Donor outcomes over the full panel.
    n_pre : int
        Number of pre-treatment periods.
    start, end : int
        Inclusive 0-indexed bounds of the pseudo-treatment block.
    n_tr : int
        Number of treated units, which enters the ridge as
        ``(n_tr * T_post)^(1/4)`` (``synthdid``'s ``eta.omega``).

    Returns
    -------
    SDIDFit

    Raises
    ------
    MlsynthConfigError
        If the window is malformed or a weight program fails to converge.
    MlsynthDataError
        If the donor pool is empty.
    """
    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    if Y0.ndim != 2 or Y0.shape[0] != y.shape[0]:
        raise MlsynthConfigError(
            f"Y0 must be (T, J) with T={y.shape[0]}; got shape {Y0.shape}.")
    if Y0.shape[1] == 0:
        raise MlsynthDataError("SDID needs at least one donor unit.")
    if not (0 < n_pre <= start <= end < y.shape[0]):
        raise MlsynthConfigError(
            f"invalid placement: n_pre={n_pre}, window=[{start}, {end}] for a "
            f"panel of length {y.shape[0]}.")

    y_pre, Y0_pre = y[:n_pre], Y0[:n_pre]
    zeta = compute_regularization(Y0_pre, int(end - start + 1), int(n_tr))
    _, omega = unit_weights(Y0_pre, y_pre, zeta)
    _, lam = fit_time_weights(Y0_pre, Y0[start:end + 1].mean(axis=0))
    if omega is None or lam is None:  # pragma: no cover - solver failure
        raise MlsynthConfigError(
            "an SDID weight program failed to converge on this placement.")
    omega = np.asarray(omega, dtype=float).ravel()
    lam = np.asarray(lam, dtype=float).ravel()

    bias_correction = float(lam @ y_pre - lam @ (Y0_pre @ omega))
    counterfactual = Y0 @ omega + bias_correction

    pre_gap = y_pre - counterfactual[:n_pre]
    pre_rmspe = float(np.sqrt(np.mean(pre_gap ** 2)))
    spread = float(np.std(y_pre))
    scaled_l2 = pre_rmspe / spread if spread > 0 else float("nan")

    return SDIDFit(omega=omega, lam=lam, bias_correction=bias_correction,
                   counterfactual=counterfactual, pre_rmspe=pre_rmspe,
                   scaled_l2=scaled_l2, zeta=float(zeta))


def sdid_att(fit: SDIDFit, y, start: int, end: int) -> float:
    """The SDID ATT over ``[start, end]``: the mean per-period effect."""
    return float(np.mean(fit.tau_path(y)[start:end + 1]))


def placebo_sigma(y, Y0, n_pre: int, start: int, end: int, *,
                  n_draws: int = 200, n_tr: int = 1, seed: int = 0) -> float:
    """Placebo standard error of the ATT (Arkhangelsky et al., Algorithm 4).

    Reassigns ``n_tr`` donors as pseudo-treated, drops them from the donor pool,
    reruns the full SDID fit, and takes the standard deviation of the resulting
    ATTs. This is the only one of the paper's three variance procedures defined
    for a single treated series, which is what a candidate test region collapses
    to, and it is what ``synthdid`` defaults to for that case.

    The treated series is never read, so the returned sigma is the null
    distribution's scale and does not depend on any injected effect.

    Returns
    -------
    float
        The placebo standard error, or ``nan`` when the donor pool is too small
        to draw from or too few draws converge.
    """
    Y0 = np.asarray(Y0, dtype=float)
    n_donors = Y0.shape[1]
    if n_donors <= n_tr + 1:
        return float("nan")

    rng = np.random.default_rng(seed)
    taus: List[float] = []
    for _ in range(int(n_draws)):
        idx = rng.choice(n_donors, size=int(n_tr), replace=False)
        mask = np.ones(n_donors, dtype=bool)
        mask[idx] = False
        pseudo_y = Y0[:, idx].mean(axis=1)
        try:
            fit = sdid_fit_once(pseudo_y, Y0[:, mask], n_pre, start, end,
                                n_tr=n_tr)
        except (MlsynthConfigError, MlsynthDataError, ValueError):
            continue  # a degenerate draw contributes nothing
        taus.append(sdid_att(fit, pseudo_y, start, end))

    if len(taus) < 3:
        return float("nan")
    return float(np.std(np.asarray(taus), ddof=1))


def normal_p_value(tau: float, sigma: Optional[float]) -> float:
    """Two-sided p-value ``2 (1 - Phi(|tau| / sigma))`` for the null of no effect.

    SDID's variance procedures deliver a standard error, so detection is a
    normal-approximation test, not the conformal permutation test an augmented
    SCM design would use.
    """
    from math import erfc, sqrt

    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return float(erfc(abs(float(tau)) / (float(sigma) * sqrt(2.0))))
