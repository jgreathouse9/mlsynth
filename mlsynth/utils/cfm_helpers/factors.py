"""Factor extraction + selection for CFM.

Common factors are estimated from the control units by principal components
on the time-demeaned control panel (Bai 2009, Section 8: demeaning separates
unit-specific intercepts from the common factors). Factors are returned in
the Bai normalization ``F'F / T = I_r``.

The number of factors is chosen by one of:

* ``"er"`` / ``"gr"`` -- the Ahn & Horenstein (2013) eigenvalue-ratio and
  growth-ratio estimators (the paper's primary criteria);
* ``"bai_ng"`` -- the Bai & Ng (2002) ``IC_p2`` information criterion;
* a user-supplied ``n_factors``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthEstimationError
from ..fma_helpers.bai_ng import nbpiid

_SELECTION = {"er", "gr", "bai_ng"}


def ahn_horenstein(
    eigenvalues: np.ndarray, max_factors: int = 8
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """Ahn & Horenstein (2013) eigenvalue-ratio and growth-ratio estimators.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Non-increasing, non-negative eigenvalues of the control covariance.
    max_factors : int
        Largest factor count to consider.

    Returns
    -------
    r_er, r_gr : int
        Factor counts selected by ER and GR.
    ER, GR : np.ndarray
        The ER and GR curves (index ``k`` corresponds to ``k + 1`` factors).
    """
    mu = np.asarray(eigenvalues, dtype=float)
    mu = mu[mu > 0]
    kmax = int(min(max_factors, len(mu) - 2))
    if kmax < 1:
        raise MlsynthEstimationError(
            "Too few positive eigenvalues to run Ahn-Horenstein selection."
        )
    # V(k) = sum_{j > k} mu_j, with 0-based index (V[k] excludes mu[0..k]).
    V = np.array([mu[k + 1:].sum() for k in range(len(mu))])
    ER = np.array([mu[k] / mu[k + 1] for k in range(kmax)])
    GR = np.empty(kmax)
    for k in range(kmax):
        num = np.log(V[k - 1] / V[k]) if k >= 1 else np.log(mu.sum() / V[0])
        GR[k] = num / np.log(V[k] / V[k + 1])
    r_er = int(np.argmax(ER)) + 1
    r_gr = int(np.argmax(GR)) + 1
    return r_er, r_gr, ER, GR


def _bai_factors(control_panel: np.ndarray, r: int) -> np.ndarray:
    """Top-``r`` principal-component factors in the Bai normalization.

    ``F = sqrt(T) * U_r`` where ``U`` holds the left singular vectors of the
    time-demeaned control panel, so ``F'F / T = I_r``.
    """
    T = control_panel.shape[0]
    Xc = control_panel - control_panel.mean(axis=0, keepdims=True)
    U, _, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :r] * np.sqrt(T)


def extract_cfm_factors(
    control_panel: np.ndarray,
    selection: str = "er",
    n_factors: Optional[int] = None,
    max_factors: int = 10,
) -> Tuple[int, np.ndarray, str, np.ndarray]:
    """Estimate the factor structure from the control panel.

    Parameters
    ----------
    control_panel : np.ndarray
        ``(T, N_co)`` matrix of control-unit outcomes.
    selection : {"er", "gr", "bai_ng"}
        Factor-count criterion. Ignored when ``n_factors`` is supplied.
    n_factors : int, optional
        Override the data-driven selection.
    max_factors : int
        Upper bound on the number of factors considered.

    Returns
    -------
    n_selected : int
        Final number of factors used.
    factors : np.ndarray
        ``(T, r)`` estimated factors in the Bai normalization.
    source : str
        ``"ER"``, ``"GR"``, ``"BaiNg"``, or ``"user"``.
    eigenvalues : np.ndarray
        Eigenvalues of the demeaned control covariance (non-increasing).
    """
    if n_factors is None and selection not in _SELECTION:
        raise MlsynthConfigError(
            f"selection must be one of {sorted(_SELECTION)}; got {selection!r}."
        )

    T, N_co = control_panel.shape
    upper = int(min(max_factors, T - 1, N_co))
    if upper < 1:
        raise MlsynthEstimationError(
            f"Control panel too small to extract factors: T={T}, N_co={N_co}."
        )

    Xc = control_panel - control_panel.mean(axis=0, keepdims=True)
    svals = np.linalg.svd(Xc, compute_uv=False)
    eigenvalues = (svals ** 2) / T  # eigenvalues of Xc'Xc / T

    if n_factors is not None:
        if not (1 <= n_factors <= upper):
            raise MlsynthConfigError(
                f"n_factors must lie in [1, {upper}]; got {n_factors}."
            )
        return int(n_factors), _bai_factors(control_panel, n_factors), "user", eigenvalues

    if selection in ("er", "gr"):
        r_er, r_gr, _, _ = ahn_horenstein(eigenvalues, max_factors=upper)
        r = r_er if selection == "er" else r_gr
        source = "ER" if selection == "er" else "GR"
    else:  # bai_ng
        r, _, _ = nbpiid(
            input_panel_data=control_panel,
            max_factors_to_test=upper,
            criterion_selector_code=2,       # Bai-Ng (2002) IC_p2
            preprocessing_method_code=1,     # demean
        )
        r = int(r)
        source = "BaiNg"

    r = int(max(1, min(r, upper)))
    return r, _bai_factors(control_panel, r), source, eigenvalues
