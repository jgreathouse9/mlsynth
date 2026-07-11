"""Frozen input container for DPSC."""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict


class DPSCInputs(BaseModel):
    """Prepared inputs for DPSC (built by ``prepare_dpsc_inputs``).

    Notation follows the docs: the treated series ``y_treated`` is
    :math:`\\mathbf{y}_1`; ``donor_matrix`` is
    :math:`\\mathbf{Y}_0 \\in \\mathbb{R}^{T \\times N_0}`; ``T0`` is the number
    of pre-treatment periods :math:`T_0`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    y_treated: np.ndarray               # (T,) treated-unit outcome path
    donor_matrix: np.ndarray            # (T, N0) donor outcomes, one column per donor
    T0: int                             # number of pre-treatment periods
    time_labels: np.ndarray             # (T,)
    treated_name: Any
    donor_names: Tuple[Any, ...]        # (N0,) donor order matching donor_matrix columns
