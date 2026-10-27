"""The fitting window of SPILLSYNTH's per-unit fits.

The iterative and inclusive methods fit every per-unit synthetic control on the
periods before treatment. With ``holdout`` periods held out, every fit stops at
``T0 - holdout`` and the correction starts there, so the held-out periods are a
placebo window for the corrected estimator (#932).
"""

from __future__ import annotations

from ...exceptions import MlsynthDataError


def fit_end(T0: int, holdout: int, method: str) -> int:
    """End (exclusive) of the fitting window, ``T0 - holdout``.

    Raises
    ------
    MlsynthDataError
        If ``holdout`` is negative, or leaves no pre-treatment period to fit.
    """
    if holdout < 0:
        raise MlsynthDataError(
            f"SPILLSYNTH method={method!r}: holdout must be non-negative, "
            f"got {holdout}."
        )
    if holdout >= T0:
        raise MlsynthDataError(
            f"SPILLSYNTH method={method!r}: holding out {holdout} of the "
            f"{T0} pre-treatment periods leaves no period to fit on."
        )
    return T0 - holdout
