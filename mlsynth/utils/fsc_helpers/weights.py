"""FSC and ridge augmented FSC weights.

Both are the scalar synthetic control applied to a *stacked* design. Section 3.1
matches on the pre-treatment objects in :math:`\\mathcal H`,

.. math::

   \\hat\\gamma^{\\mathrm{scm}} \\in \\arg\\min_{\\gamma \\in \\Delta^{N-1}}
     \\sum_{t \\le T_0} \\Bigl\\lVert Y_{1t} - \\sum_i \\gamma_i Y_{it}
     \\Bigr\\rVert_{\\mathcal H}^2 ,

which, once each object is carried into :math:`\\mathcal H` by the isometry and
evaluated on a common grid, is the ordinary simplex least-squares problem on the
``(period, grid point)`` pairs stacked into one long vector. So the base solve is
mlsynth's own exact QP, :func:`mlsynth.utils.bilevel.ridge_augment.simplex_qp`,
unchanged.

Everything in this module therefore works on *embedded* coordinates: the caller
has already applied :math:`\\Psi` (see
:func:`mlsynth.utils.fsc_helpers.project.frobenius_weights` for the one space
where that is not the identity), so the norm here is the plain Euclidean norm on
the sampled coordinates.

Section 3.2's augmentation is likewise the ridge-augmented SCM of Ben-Michael,
Feller & Rothstein (2021) with the pre-period axis replaced by the
``(period, basis coefficient)`` axis:

.. math::

   \\hat\\gamma^{\\mathrm{aug}}_i = \\hat\\gamma^{\\mathrm{scm}}_i
     + (r_{1\\cdot} - r_{0\\cdot}' \\hat\\gamma^{\\mathrm{scm}})'
       (r_{0\\cdot}' r_{0\\cdot} + \\lambda I)^{-1} r_{i\\cdot} .

The correction vanishes when the simplex fit is perfect and as
:math:`\\lambda \\to \\infty`, and it is what allows negative weights.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from ...exceptions import MlsynthEstimationError
from ..bilevel.ridge_augment import simplex_qp

# R's optimise() default tolerance, .Machine$double.eps^0.25. Used here for the
# same reason the authors' scripts do: the cross-validation objective is flat
# near its minimum, so a tighter tolerance buys precision on a quantity that
# does not need it.
_BRENT_TOL = float(np.finfo(float).eps ** 0.25)


def stacked_design(values: np.ndarray, n_pre: int
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """``(treated, donors)`` with the pre-periods and grid stacked into rows."""
    treated = values[0, :n_pre].ravel()
    donors = values[1:, :n_pre].reshape(values.shape[0] - 1, -1).T
    return treated, donors


def fsc_weights(values: np.ndarray, n_pre: int) -> np.ndarray:
    """Simplex weights of eq. (1)."""
    treated, donors = stacked_design(values, n_pre)
    return np.asarray(simplex_qp(donors, treated), dtype=float).ravel()


def center_pre_periods(values: np.ndarray, n_pre: int) -> np.ndarray:
    """Subtract the donor mean object at each pre-period (section 3.2).

    The simplex fit is invariant to this -- the shift cancels because the
    weights sum to one -- but the ridge correction is not.
    """
    centered = values.copy()
    centered[:, :n_pre] -= centered[1:, :n_pre].mean(axis=0, keepdims=True)
    return centered


def trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    """Quadrature weights approximating ``int f`` from samples on ``grid``."""
    grid = np.asarray(grid, dtype=float)
    weights = np.zeros_like(grid)
    step = np.diff(grid)
    weights[:-1] += step / 2.0
    weights[1:] += step / 2.0
    return weights


def basis_coefficients(values: np.ndarray, basis: Optional[np.ndarray],
                       grid: Optional[np.ndarray]) -> np.ndarray:
    """Inner products ``r_{i,t,k} = <phi_k, X_it>`` over the argument grid.

    The integral is approximated by the trapezoid rule, so the coefficients are
    a genuine :math:`L^2` inner product whatever the grid spacing.
    ``basis=None`` is the Euclidean case, where the coordinates already are the
    coefficients in the standard basis and no expansion is needed.

    A uniform rescaling of the basis only reparameterises the penalty -- scaling
    ``r`` by ``c`` and ``lambda`` by ``c^2`` leaves eq. (9) unchanged -- so the
    normalisation convention here is immaterial to a cross-validated fit.
    """
    if basis is None:
        return values
    return values @ (basis * trapezoid_weights(grid)[:, None])


def augmented_weights(values: np.ndarray, n_pre: int, lam: float, *,
                      basis: Optional[np.ndarray] = None,
                      grid: Optional[np.ndarray] = None) -> np.ndarray:
    """Ridge augmented weights of eq. (9)."""
    centered = center_pre_periods(values, n_pre)
    gamma = fsc_weights(centered, n_pre)
    R = basis_coefficients(centered, basis, grid)
    r0 = R[1:, :n_pre].reshape(R.shape[0] - 1, -1)
    r1 = R[0, :n_pre].ravel()
    gram = r0.T @ r0 + lam * np.eye(r0.shape[1])
    residual = r1 - r0.T @ gamma
    try:
        correction = np.linalg.solve(gram, residual)
    except np.linalg.LinAlgError:  # pragma: no cover - lambda >= 0 keeps it PSD
        correction = np.linalg.lstsq(gram, residual, rcond=None)[0]
    return gamma + r0 @ correction


def cross_validation_error(lam: float, values: np.ndarray, n_pre: int, *,
                           basis: Optional[np.ndarray] = None,
                           grid: Optional[np.ndarray] = None) -> float:
    """Leave-one-pre-period-out cross-validation error of Remark 5."""
    total = 0.0
    for k in range(n_pre):
        keep = [t for t in range(values.shape[1]) if t != k]
        w = augmented_weights(values[:, keep], n_pre - 1, lam,
                              basis=basis, grid=grid)
        total += float(np.sum((values[0, k] - w @ values[1:, k]) ** 2))
    return total


def penalty_scale(values: np.ndarray, n_pre: int, *,
                  basis: Optional[np.ndarray] = None,
                  grid: Optional[np.ndarray] = None) -> float:
    """Natural size of the ridge penalty for this design.

    The penalty in eq. (9) is added to ``r_0' r_0``, so what counts as "small"
    is set by that matrix, which scales with the square of the outcome and with
    whatever normalisation the basis carries. An absolute search interval
    therefore means something different on every dataset: the authors' scripts
    search (0, 10), which suits their fertility panel and is four orders of
    magnitude too coarse for their mortality panel once the coefficients are a
    genuine L2 inner product.

    Reported as the mean eigenvalue of the Gram matrix, so a relative penalty of
    1 shrinks about as hard as the design's own average curvature. This is the
    same idea as ``augsynth``'s data-driven lambda grid, reduced to one number
    because the search here is a scalar minimisation rather than a grid.
    """
    centered = center_pre_periods(values, n_pre)
    R = basis_coefficients(centered, basis, grid)
    r0 = R[1:, :n_pre].reshape(R.shape[0] - 1, -1)
    return float(np.sum(r0 * r0) / r0.shape[1])


def select_lambda(values: np.ndarray, n_pre: int, bounds, *,
                  basis: Optional[np.ndarray] = None,
                  grid: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Cross-validate the penalty, searching on the design's own scale.

    ``bounds`` are read as multiples of :func:`penalty_scale`, which makes the
    default interval mean the same thing whatever the units of the outcome.

    Returns
    -------
    (absolute, relative) : tuple of float
        The penalty to use in eq. (9), and the multiple of the design scale it
        corresponds to.

    Raises
    ------
    MlsynthEstimationError
        Fewer than two pre-treatment periods, where leaving one out leaves
        nothing to fit the weights on.
    """
    if n_pre < 2:
        raise MlsynthEstimationError(
            f"cross-validating the ridge penalty needs at least two "
            f"pre-treatment periods (it is leave-one-out); the panel has "
            f"{n_pre}. Pass ridge_lambda explicitly, or set augment=False.")
    scale = penalty_scale(values, n_pre, basis=basis, grid=grid)
    if not scale > 0.0:
        # The centered design is identically zero, which happens whenever there
        # is a single donor: subtracting the donor mean leaves nothing. The
        # ridge correction is then zero for every penalty, so there is nothing
        # to cross-validate and augmentation degrades to a no-op.
        return 0.0, 0.0
    result = minimize_scalar(
        lambda ratio: cross_validation_error(ratio * scale, values, n_pre,
                                             basis=basis, grid=grid),
        bounds=tuple(bounds), method="bounded", options={"xatol": _BRENT_TOL})
    return float(result.x) * scale, float(result.x)


def synthetic_objects(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average of the donor objects, ``(T, M)``."""
    return np.einsum("j,jtm->tm", np.asarray(weights, dtype=float), values[1:])


def pre_treatment_fit(values: np.ndarray, n_pre: int,
                      synthetic: np.ndarray) -> float:
    """``sqrt(sum_{t <= T0} ||Y_1t - synthetic_t||^2)`` on embedded coordinates."""
    gap = values[0, :n_pre] - synthetic[:n_pre]
    return float(np.sqrt(np.sum(gap ** 2)))
