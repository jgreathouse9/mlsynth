"""CFPT/scpi uncertainty quantification for synthetic controls.

Cattaneo, M. D., Feng, Y., Palomba, F., & Titiunik, R. (2025).
*"Uncertainty Quantification in Synthetic Controls with Staggered Treatment
Adoption."* arXiv:2210.05026.

Non-asymptotic prediction intervals that decompose the prediction error of a
synthetic-control estimator into an *in-sample* part (SC-weight estimation) and
an *out-of-sample* part (post-treatment sampling noise), for four causal
predictands -- TSUS, TAUS, TSUA, TAUA -- plus simultaneous (uniform across
post-periods) bands.

Module layout:

* :mod:`.moments` -- conditional-moment estimation from pre-period residuals.
* :mod:`.intervals` -- out-of-sample bands (all four predictands +
  simultaneous) and the reusable in-sample simulation bound.
* :mod:`.structures` -- :class:`SCPIBand`, :class:`SCPIResults`.
"""

from __future__ import annotations

from .intervals import in_sample_band_gaussian, out_of_sample_intervals
from .moments import conditional_moments, unit_moments
from .structures import SCPIBand, SCPIResults

__all__ = [
    "SCPIBand",
    "SCPIResults",
    "conditional_moments",
    "in_sample_band_gaussian",
    "out_of_sample_intervals",
    "unit_moments",
]
