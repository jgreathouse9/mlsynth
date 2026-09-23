"""The refit rule a test-inversion conformal procedure uses.

Chernozhukov, Wuthrich & Zhu (2021) invert a sharp null: subtract a candidate
effect from the treated post-treatment outcomes, refit the control on the
adjusted series, and ask whether the post-block residual conforms with the
pre-block ones. The refit is the only point at which the procedure touches an
estimator, so the rule it uses is what ties the inference to the estimate.

Getting that rule wrong does not produce a wrong-looking number. It produces a
test with no power, which looks like a large p-value and reads like a null
result. The mechanism is worth stating because it is not obvious: an
unconstrained refit handed a treated series with a large post-period effect will
part-absorb the effect by re-levelling, and the level it adds is spread over the
whole matching window -- including the pre-period, which was never touched. Both
halves of the test then scale with the effect. The observed statistic grows, the
reference distribution built from the pre-period residuals grows with it, and
their ratio -- which is all the p-value sees -- stalls. On the Swedish carbon
tax panel the p-value is 0.348 whether the injected effect is 5 or 100.

Convexity is what stops it. A synthetic control with non-negative weights
summing to one is bounded per period by the donor pool, so its fitted values
cannot follow an arbitrarily large shift, and the pre-period residuals stop
responding once the weights reach a vertex. The effect then has nowhere to go
but the post-block residual, which is where the test looks for it.

Hence two rules, chosen by the caller to match the estimator whose effect is
being tested:

``"sc"``
    Simplex synthetic control. ``estimation_method = "sc"`` in the authors'
    ``scinference`` package, ``progfunc = "None"`` in ``augsynth``. Outcome-only.

``"ridge"``
    Ridge-augmented SCM -- ``augsynth``'s ``progfunc = "Ridge"``, the refit its
    conformal mode is defined on. The augmentation leaves the simplex on
    purpose: that is what makes it a bias correction, and it is also what lets
    it absorb. Correct when the point estimate is itself ridge-augmented, and
    then the absence of power against a large level shift is a property of the
    published method rather than of this implementation.

References
----------
Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021). An Exact and Robust Conformal
Inference Method for Counterfactual and Synthetic Controls. Journal of the
American Statistical Association, 116(536), 1849-1864.

Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The Augmented Synthetic
Control Method. Journal of the American Statistical Association, 116(536),
1789-1803.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError

#: The refit rules :func:`conformal_refit_gaps` accepts, in the order they are
#: documented. ``"sc"`` is first because it is the rule the conformal procedure
#: was published on; ``"ridge"`` remains the default of the augsynth-facing
#: entry points, which reproduce that package.
CONFORMAL_REFIT_RULES = ("sc", "ridge")


def conformal_refit_gaps(
    y: np.ndarray,
    Y0: np.ndarray,
    *,
    rule: str,
    Z0: Optional[np.ndarray] = None,
    z1: Optional[np.ndarray] = None,
    ridge_kwargs: Optional[Dict[str, Any]] = None,
    warm_start: Optional[np.ndarray] = None,
    return_base: bool = False,
):
    """Refit the control on the whole given window and return treated-minus-fit.

    The entire outcome path is passed as the matching window, which is what the
    conformal refit means: every period is balanced, and the residuals are the
    gaps over the whole window. Callers slice the post block out afterwards.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Treated outcomes over the matching window, already adjusted by the
        candidate effect if one is being tested.
    Y0 : np.ndarray, shape (T, J)
        Donor outcomes over the same window.
    rule : {"sc", "ridge"}
        Which control to refit. See the module docstring; there is no default,
        because a silent one is how the estimate and the inference drift apart.
    Z0, z1 : np.ndarray, optional
        Auxiliary covariates, donor ``(J, K)`` and treated ``(K,)``. Only the
        ``"ridge"`` rule accepts them -- the simplex rule is outcome-only, and
        accepting covariates it would then ignore would produce a fit the caller
        did not ask for with no output showing it.
    ridge_kwargs : dict, optional
        Hyper-parameters forwarded to
        :func:`~mlsynth.utils.bilevel.ridge_augment.ridge_augment_weights`.
        Ignored by the ``"sc"`` rule, which has none.
    warm_start : np.ndarray, optional
        Previous simplex weights to seed the solve with; the two rules both
        solve a simplex QP at their base, so a grid sweep can chain them.
    return_base : bool, default False
        Also return the base simplex weights, for that chaining. Under the
        ``"sc"`` rule the base weights are the weights.

    Returns
    -------
    np.ndarray, shape (T,)
        The gaps, or ``(gaps, base_weights)`` when ``return_base`` is set.

    Raises
    ------
    MlsynthConfigError
        Unknown ``rule``, covariates supplied to the ``"sc"`` rule, or an empty
        donor pool.
    MlsynthDataError
        Treated and donor windows of different lengths.
    """
    if rule not in CONFORMAL_REFIT_RULES:
        raise MlsynthConfigError(
            f"conformal refit rule must be one of {CONFORMAL_REFIT_RULES}; "
            f"got {rule!r}."
        )
    y = np.asarray(y, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    if Y0.ndim != 2:
        raise MlsynthDataError(
            f"donor outcomes must be 2-D (T, J); got shape {Y0.shape}."
        )
    if Y0.shape[0] != y.shape[0]:
        raise MlsynthDataError(
            f"treated and donor windows must have the same length; got "
            f"{y.shape[0]} treated periods and {Y0.shape[0]} donor periods."
        )
    if Y0.shape[1] == 0:
        raise MlsynthConfigError("conformal refit needs at least one donor; got 0.")

    if rule == "sc":
        if Z0 is not None or z1 is not None:
            raise MlsynthConfigError(
                "the 'sc' refit rule matches on outcomes only and cannot use "
                "covariates; pass rule='ridge' to include them, or drop Z0/z1."
            )
        from ..bilevel.ridge_augment import simplex_qp

        w = np.asarray(simplex_qp(Y0, y, warm_start=warm_start), dtype=float)
        gaps = y - Y0 @ w
        return (gaps, w) if return_base else gaps

    from ..bilevel.ridge_augment import ridge_augment_weights

    ra = ridge_augment_weights(
        y, Y0, Z0=Z0, z1=z1, warm_start=warm_start, **(ridge_kwargs or {})
    )
    gaps = y - Y0 @ ra.W
    return (gaps, ra.W_base) if return_base else gaps
