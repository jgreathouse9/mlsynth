"""Ferman & Pinto (2021, QE) Monte Carlo -- mlsynth's SC / demeaned-SC reproduce
Table 1's theoretical properties on the CPS-calibrated factor-model DGP.

The golden, dashboard-facing version cross-checks mlsynth's estimators against the
authors' own ``quadprog`` QPs, live via ``Rscript``, and also reproduces the Panel
A/B bias magnitudes (``benchmarks/cases/ferman_pinto_mc.py``). This fast offline
test runs a short Monte Carlo (no R) and asserts the qualitative findings:

* no break: original SC is biased at an extreme fixed effect, while demeaned SC
  and DID are ~unbiased, and demeaned SC is more efficient than DID;
* a factor-1 break: all three are biased, but SC and demeaned SC far below DID.
"""
from __future__ import annotations

import numpy as np

from benchmarks.cases import ferman_pinto_mc as F

_NREPS = 60


def test_no_break_sc_biased_demeaned_and_did_unbiased():
    """Panel A (2nd-largest fixed effect, no break): SC biased; demeaned & DID
    ~unbiased; demeaned SC more efficient than DID (Propositions 1-3)."""
    s = F._panel_stats("A", "fe", "hi", None, 120, seed=101, nreps=_NREPS)
    bias, se = s["bias"], s["se"]
    assert abs(bias[0]) > 0.12          # original SC biased (fails to recover level)
    assert abs(bias[1]) < 0.12          # demeaned SC ~unbiased
    assert abs(bias[2]) < 0.20          # DID ~unbiased
    assert se[1] < se[2]                # demeaned SC standard error below DID's


def test_break_all_biased_with_sc_and_demeaned_below_did():
    """Panel C (2nd-largest factor-1 loading, break in factor 1): selection on
    unobservables biases all three, but SC and demeaned SC far below DID."""
    s = F._panel_stats("C", "f1", "hi", 0, 120, seed=202, nreps=_NREPS)
    b = np.abs(s["bias"])
    assert b.min() > 0.3                # all three biased
    assert b[0] < b[2] and b[1] < b[2]  # SC and demeaned SC below DID


def test_calibration_bundle_shapes():
    """The baked DGP environment has the paper's dimensions (51 units, 4 factors)."""
    assert F._N == 51
    assert F._LAM.shape == (51, 4)
    assert F._FAC_AR.shape == (4, 3)
    assert F._RES_AR.shape == (51, 3)
