"""The reference key's four assumption gates, read at a design as diagnostics.

The certified search ranks designs by the relative key alone (treatment.py); the
gates are reported on its design and rank nothing. The reference implementation's
key is (corr_test, aa_test, bb_test, dw_test, round(corr, 2), 1 / required_impact)
(mlsynth.utils.tbr_helpers.design.objective.score_split).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from mlsynth.utils.tbr_helpers.design.objective import score_split

GATES = ("corr_test", "aa_test", "bb_test", "dw_test")


def gates_of(y: np.ndarray, X: np.ndarray, controls: Sequence[int], n_test: int) -> Dict[str, bool]:
    """The four gates at the design whose treated aggregate is ``y`` and whose
    controls are the columns ``controls`` of ``X``, by the library's own scoring."""
    s = score_split(np.asarray(y, dtype=float), np.asarray(X, dtype=float)[:, sorted(controls)].sum(axis=1),
                    objective="reference", n_test=n_test)
    return {g: bool(s.detail[g]) for g in GATES}
