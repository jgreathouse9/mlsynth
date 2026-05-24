"""Single Proxy Synthetic Control (SPSC) (Park & Tchetgen Tchetgen, 2025).

Uses only the donor outcomes -- a single proxy type -- with the treated
unit's own (optionally B-spline-detrended) outcome as the instrument, via a
synthetic-control bridge function. Provides the ridge-GMM point estimator
with a GMM/HAC standard error (``estimation``) and conformal pointwise
prediction intervals for the per-period effect (``conformal``).
"""

from .estimation import estimate_spsc
from .conformal import conformal_intervals

__all__ = ["estimate_spsc", "conformal_intervals"]
