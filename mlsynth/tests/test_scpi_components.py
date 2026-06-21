"""The SCPI metadata exposes the components needed for exact cross-unit
aggregation, and those components reproduce the internal ATT band.

Guards the additive refactor of ``vanillasc_helpers/scpi.py`` that surfaces the
per-simulation in-sample draws and the out-of-sample per-row bands.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel import BilevelSCM
from mlsynth.utils.vanillasc_helpers.scpi import scpi_intervals


def _panel(seed=0, J=8, T0=20, T1=6, r=3):
    rng = np.random.default_rng(seed)
    T = T0 + T1
    F = rng.standard_normal((T, r))
    Lam = rng.standard_normal((J + 1, r))
    Y = F @ Lam.T + rng.standard_normal((T, J + 1)) * 0.3   # (T, J+1)
    y = Y[:, 0].copy()
    Y0 = Y[:, 1:]
    y[T0:] += 2.0                                            # planted effect
    return y, Y0, T0


def _fit_W(y, Y0, pre, names):
    eng = BilevelSCM("outcome-only")
    return eng.fit(y[:pre], Y0[:pre], X1=None, X0=None,
                   donor_names=names, predictor_names=[]).W


def test_metadata_exposes_components_with_right_shapes():
    y, Y0, pre = _panel()
    W = _fit_W(y, Y0, pre, [f"d{j}" for j in range(Y0.shape[1])])
    sc = scpi_intervals(y, Y0, pre, W, sims=300, u_alpha=0.05, e_alpha=0.05, seed=1)
    md = sc.metadata
    T_post = md["n_post"]
    assert md["insample_draws_lo"].shape == (300, T_post + 1)
    assert md["insample_draws_hi"].shape == (300, T_post + 1)
    assert md["oos_lb_rows"].shape == (T_post + 1,)
    assert md["oos_ub_rows"].shape == (T_post + 1,)
    assert md["cf"].shape == (T_post,)
    assert md["obs"].shape == (T_post,)


def test_components_reproduce_att_band():
    """The exposed averaged-row draws + out-of-sample row, recombined the way
    scpi_intervals does internally, reproduce att_lower / att_upper exactly."""
    y, Y0, pre = _panel(seed=2)
    W = _fit_W(y, Y0, pre, [f"d{j}" for j in range(Y0.shape[1])])
    sc = scpi_intervals(y, Y0, pre, W, sims=400, u_alpha=0.05, e_alpha=0.05, seed=7)
    md = sc.metadata
    n = md["n_post"]
    lo_avg = md["insample_draws_lo"][:, n]      # averaged-predictand row
    hi_avg = md["insample_draws_hi"][:, n]
    w_lb = float(np.nanquantile(lo_avg, 0.05 / 2.0))
    w_ub = float(np.nanquantile(hi_avg, 1.0 - 0.05 / 2.0))
    e_lb = float(md["oos_lb_rows"][n])
    e_ub = float(md["oos_ub_rows"][n])
    obs_avg = float(md["obs"].mean())
    cf_avg = float(md["cf"].mean())
    att_lower = obs_avg - (cf_avg + w_ub + e_ub)
    att_upper = obs_avg - (cf_avg + w_lb + e_lb)
    assert att_lower == pytest.approx(md["att_lower"], abs=1e-9)
    assert att_upper == pytest.approx(md["att_upper"], abs=1e-9)
