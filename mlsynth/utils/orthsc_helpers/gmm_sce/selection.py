"""Andrews--Lu (2001) downward-testing control/instrument selection for GMM-SCE.

Implements Steps 1--6 of Fry (2024, Section 3): start from the controls that the
just-identified initial estimate gives positive weight, then -- while the
Sargan--Hansen over-identification test rejects -- greedily promote the
remaining unit closest (in pre-treatment mean-squared difference) to the treated
unit from instrument to control, until the test no longer rejects or every
candidate is a control. The test uses the slow-shrinking significance level
``exp(-sqrt(T0))`` so the procedure is consistent (Andrews and Lu 2001;
Potscher 1983).

The reference ``GMM-SCE.R`` ``ModelSelection`` carries an index quirk in its loop
(it shrinks the candidate matrix but keeps the original count); this follows the
paper's stated algorithm, and the unambiguous core -- the GMM solve and its
J-statistic -- is cross-validated against the R in the benchmark suite.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.stats import chi2

from .solver import gmm_sc_weights


def select_controls(
    pre_y0: np.ndarray,
    pre_yn0: np.ndarray,
    pre_yn1: Optional[np.ndarray] = None,
    *,
    weight_tol: float = 1e-5,
    include_constant: bool = True,
) -> List[int]:
    """Choose which of the ``N0`` candidate units serve as controls.

    Parameters
    ----------
    pre_y0
        Treated unit pre-treatment outcomes, shape ``(T0,)``.
    pre_yn0
        Candidate units that may be controls or instruments, shape ``(T0, N0)``.
    pre_yn1
        Optional units that are always instruments (never controls), shape
        ``(T0, N1)`` -- e.g. neighbours of the treated unit, or units treated
        later. ``None`` if there are none.
    weight_tol
        A candidate gets a positive initial weight (and so enters the starting
        control set) if its weight exceeds this.
    include_constant
        Pass through to the GMM solve (the mean-matching constant instrument).

    Returns
    -------
    list of int
        Sorted column indices into ``pre_yn0`` chosen as controls.
    """
    YN0 = np.atleast_2d(np.asarray(pre_yn0, dtype=float))
    y0 = np.asarray(pre_y0, dtype=float).ravel()
    if YN0.shape[0] != y0.shape[0]:
        YN0 = YN0.T
    T0, N0 = YN0.shape
    YN1 = None if pre_yn1 is None else np.atleast_2d(np.asarray(pre_yn1, float))
    if YN1 is not None and YN1.shape[0] != T0:
        YN1 = YN1.T
    pvalue = float(np.exp(-np.sqrt(T0)))

    # Step 1-2: initial just-identified estimate; start from its positive-weight
    # units (P in the paper).
    yk_all = YN0 if YN1 is None else np.hstack([YN0, YN1])
    w_init = gmm_sc_weights(y0, YN0, yk_all, include_constant=include_constant)["weights"]
    control_idx = [i for i in range(N0) if w_init[i] > weight_tol]
    if not control_idx:  # pragma: no cover - simplex always gives some mass
        control_idx = [int(np.argmax(w_init))]

    # Step 3-6: downward testing.
    while True:
        instr_idx = [i for i in range(N0) if i not in control_idx]
        if not instr_idx and YN1 is None:
            break  # no instruments left to test against; accept current set
        YJ = YN0[:, control_idx]
        if YN1 is None:
            YK = YN0[:, instr_idx]
        elif instr_idx:
            YK = np.hstack([YN0[:, instr_idx], YN1])
        else:
            YK = YN1
        res = gmm_sc_weights(y0, YJ, YK, include_constant=include_constant)
        J = len(control_idx)
        K = res["n_instruments"]                    # includes the constant
        df = max(K - J, 1)
        critical_value = float(chi2.isf(pvalue, df))   # upper-tail
        if res["jstatistic"] <= critical_value or len(control_idx) >= N0:
            break
        # Step 5: promote the remaining unit nearest the treated unit.
        mse = {i: float(np.mean((y0 - YN0[:, i]) ** 2)) for i in instr_idx}
        control_idx = sorted(control_idx + [min(mse, key=mse.get)])
    return sorted(control_idx)
