"""Frozen input container for BEAST."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict


class BEASTInputs(BaseModel):
    """Prepared inputs for the BEAST estimator (built by ``prepare_beast_inputs``)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    X: np.ndarray                       # (N, 1 + M + L): [const, covariates, outcome lags]
    d: np.ndarray                       # (N,) treatment indicator per unit
    Y: np.ndarray                       # (T, N) wide outcome matrix (unit order == X rows)
    y_treated: np.ndarray               # (T,) treated-unit outcome path
    feature_names: Tuple[str, ...]      # length 1 + M + L (without the constant label)
    pre: int                            # number of pre-treatment periods
    time_labels: np.ndarray             # (T,)
    treated_name: Any
    unit_names: Tuple[Any, ...]         # (N,) unit order matching X rows
    treated_index: int
