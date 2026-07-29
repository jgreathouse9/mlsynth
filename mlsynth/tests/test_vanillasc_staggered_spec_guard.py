"""``staggered_spec`` must not be silently ignored on a single-treated panel.

``run_vanillasc`` routes to the staggered engine only when ``dataprep`` returns
cohorts, i.e. when the panel has more than one treated unit. A single-treated
panel falls through to the ordinary path, which never reads
``config.staggered_spec`` -- so every field on the spec, including
``w_constr``, was accepted and discarded. Silent acceptance is the failure
mode the repo's fail-early contract exists to prevent: the caller believes a
constraint family is in force when the fit is the plain outcome-only one.

The single-treated route to covariate matching is ``covariates`` /
``covariate_windows`` with a predictor-weight ``backend``, so the guard raises
and says so rather than trying to honour the spec.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError


def _panel(n_units: int = 6, n_periods: int = 20, n_treated: int = 1,
           seed: int = 0) -> pd.DataFrame:
    """Balanced panel; the first ``n_treated`` units adopt at period 15."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n_periods + 1)
    rows = []
    for i in range(n_units):
        y = 100 + 3.0 * t + rng.normal(0, 1.0, n_periods) + 5 * i
        treated = np.zeros(n_periods, dtype=int)
        if i < n_treated:
            treated[t >= 15] = 1
            y = y + treated * 6.0
        rows.append(pd.DataFrame({"unit": f"u{i}", "period": t, "y": y,
                                  "x": y * 0.5 + rng.normal(0, 0.5, n_periods),
                                  "treated": treated}))
    return pd.concat(rows, ignore_index=True)


def _fit(df: pd.DataFrame, **extra):
    cfg = {"df": df, "outcome": "y", "treat": "treated", "unitid": "unit",
           "time": "period", "display_graphs": False, "inference": False}
    cfg.update(extra)
    return VanillaSC(cfg).fit()


SPEC = {"features": ["y"], "w_constr": "lasso"}


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------
def test_single_treated_with_staggered_spec_raises():
    """One treated unit plus a staggered_spec is a contradiction, not a no-op."""
    with pytest.raises(MlsynthConfigError):
        _fit(_panel(n_treated=1), staggered_spec=SPEC)


def test_guard_message_names_the_cause_and_the_fix():
    """The error has to be actionable: say the panel is single-treated, and
    point at the knob that does work there."""
    with pytest.raises(MlsynthConfigError) as exc:
        _fit(_panel(n_treated=1), staggered_spec=SPEC)
    msg = str(exc.value)
    assert "staggered_spec" in msg
    assert "1 treated unit" in msg
    assert "covariates" in msg


def test_guard_fires_before_any_fitting():
    """The failure is reported, not swallowed and papered over with a result.

    A degenerate spec on a single-treated panel must still raise the config
    error rather than falling through to a plain fit that happens to succeed.
    """
    with pytest.raises(MlsynthConfigError):
        _fit(_panel(n_treated=1), staggered_spec={"features": ["y", "x"],
                                                  "constant": True})


@pytest.mark.parametrize("w_constr", ["simplex", "lasso", "ols", "ridge"])
def test_guard_is_independent_of_the_constraint_family(w_constr):
    """Every w_constr was equally ignored, so every one must be caught."""
    with pytest.raises(MlsynthConfigError):
        _fit(_panel(n_treated=1),
             staggered_spec={"features": ["y"], "w_constr": w_constr})


# ---------------------------------------------------------------------------
# No collateral damage
# ---------------------------------------------------------------------------
def test_single_treated_without_the_spec_is_unaffected():
    res = _fit(_panel(n_treated=1))
    assert np.isfinite(res.effects.att)
    assert res.weights is not None


def test_multi_treated_with_the_spec_still_fits():
    """The guard must key on the panel, not on the presence of the spec."""
    res = _fit(_panel(n_treated=2), staggered_spec=SPEC)
    assert np.isfinite(res.effects.att)


def test_single_treated_covariate_route_still_works():
    """The alternative the message recommends has to actually run."""
    res = _fit(_panel(n_treated=1), covariates=["x"], backend="mscmt",
               mscmt_maxiter=5, mscmt_popsize=4, seed=0)
    assert np.isfinite(res.effects.att)
