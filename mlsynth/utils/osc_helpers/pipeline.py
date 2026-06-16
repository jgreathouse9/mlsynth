"""OSC orchestrator: regularized nuisance -> orthogonalized ATT -> Series-HAC
fixed-smoothing inference. Mirrors the reference ``OrthoganilzedSCE`` end to end.
"""
from __future__ import annotations

import numpy as np


def orthogonalized_sce(pre_y0, pre_yj, Z, post_y0, post_yj, *,
                       alpha: float = 0.05, beta0: float = 0.0,
                       include_constant: bool = True):
    """Run the full Orthogonalized Synthetic Control estimate + inference.

    Returns
    -------
    dict with ``beta``, ``pvalue``, ``ci`` (lo, hi), ``df`` (smoothing K),
    ``control_weights`` (delta), ``instrument_weights`` (eta).
    """
    raise NotImplementedError
