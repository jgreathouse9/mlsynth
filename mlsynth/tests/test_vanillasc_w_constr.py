"""Weight-constraint families for the single-treated VanillaSC.

``w_constr`` -- scpi's weight-constraint family (Cattaneo, Feng, Palomba and
Titiunik) -- was reachable only through ``staggered_spec``, so it applied only
to panels with two or more treated units. Which constraint the estimator uses
is a modelling choice, not a property of how many units were treated, so it is
exposed at the top level and honoured on the single-treated path.

The families are genuinely different estimators: ``simplex`` pins the weights
to the unit simplex (no extrapolation), ``lasso`` bounds their L1 norm,
``ridge`` their L2 norm, and ``ols`` leaves them free. The regression these
tests exist to prevent is all four silently collapsing to one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC
from mlsynth.config_models import VanillaSCConfig
from mlsynth.exceptions import MlsynthConfigError
from pydantic import ValidationError

FAMILIES = ["simplex", "lasso", "ridge", "ols", "L1-L2"]


def _panel(n_donors: int = 5, n_periods: int = 24, treated_offset: float = 0.0,
           seed: int = 3) -> pd.DataFrame:
    """Balanced single-treated panel; ``treated_offset`` lifts the treated unit
    off the donors' convex hull, which is where the families separate."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n_periods + 1)
    base = 50 + 2.0 * t
    rows = []
    donor_paths = []
    for j in range(n_donors):
        y = base * (0.8 + 0.1 * j) + rng.normal(0, 0.6, n_periods)
        donor_paths.append(y)
        rows.append(pd.DataFrame({"unit": f"d{j}", "period": t, "y": y,
                                  "x": y * 0.4 + rng.normal(0, 0.3, n_periods),
                                  "treated": 0}))
    y1 = np.mean(donor_paths, axis=0) + treated_offset + rng.normal(0, 0.6, n_periods)
    treat = (t >= 18).astype(int)
    y1 = y1 + treat * 9.0
    rows.append(pd.DataFrame({"unit": "T", "period": t, "y": y1,
                              "x": y1 * 0.4 + rng.normal(0, 0.3, n_periods),
                              "treated": treat}))
    return pd.concat(rows, ignore_index=True)


def _fit(df: pd.DataFrame, **extra):
    cfg = {"df": df, "outcome": "y", "treat": "treated", "unitid": "unit",
           "time": "period", "display_graphs": False, "inference": False}
    cfg.update(extra)
    return VanillaSC(cfg).fit()


def _w(res, df: pd.DataFrame) -> np.ndarray:
    """Weights in donor-label order, zeros for donors the fit dropped."""
    d = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
    names = sorted(u for u in df.unit.unique() if u != "T")
    return np.array([d.get(n, 0.0) for n in names], dtype=float)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_each_family_fits(family):
    res = _fit(_panel(), w_constr=family)
    assert np.isfinite(res.effects.att)
    assert res.weights.donor_weights
    assert np.all(np.isfinite(np.asarray(res.time_series.counterfactual_outcome,
                                         dtype=float)))


# ---------------------------------------------------------------------------
# The families must actually differ -- this is the regression under test
# ---------------------------------------------------------------------------
def test_families_are_not_all_the_same_fit():
    df = _panel(treated_offset=14.0)          # treated sits above the donor hull
    ws = {f: _w(_fit(df, w_constr=f), df) for f in FAMILIES}
    assert not np.allclose(ws["simplex"], ws["ols"], atol=1e-6)
    assert not np.allclose(ws["simplex"], ws["ridge"], atol=1e-6)
    atts = {f: float(_fit(df, w_constr=f).effects.att) for f in FAMILIES}
    assert len({round(a, 6) for a in atts.values()}) > 1


def test_simplex_reproduces_the_default_fit():
    """The default path already solves the simplex program; naming it must not
    change the answer."""
    df = _panel()
    assert _w(_fit(df, w_constr="simplex"), df) == pytest.approx(
        _w(_fit(df), df), abs=1e-4)


# ---------------------------------------------------------------------------
# Each family's defining invariant
# ---------------------------------------------------------------------------
def test_simplex_weights_are_on_the_simplex():
    w = _w(_fit(_panel(treated_offset=14.0), w_constr="simplex"), _panel())
    assert w.min() >= -1e-8
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_lasso_bounds_the_l1_norm():
    df = _panel(treated_offset=14.0)
    w = _w(_fit(df, w_constr="lasso"), df)
    assert np.abs(w).sum() <= 1.0 + 1e-6


def test_ols_is_not_confined_to_the_simplex():
    """Free weights: on a treated unit off the hull, OLS leaves the simplex."""
    df = _panel(treated_offset=14.0)
    w = _w(_fit(df, w_constr="ols"), df)
    assert (w.min() < -1e-6) or (abs(w.sum() - 1.0) > 1e-3)


def test_ridge_bounds_the_l2_norm():
    df = _panel(treated_offset=14.0)
    res = _fit(df, w_constr="ridge")
    q = res.method_details.parameters_used["w_constr_Q"]
    assert np.linalg.norm(_w(res, df)) <= float(q) + 1e-6


def test_explicit_budget_is_honoured():
    """A dict spec passes Q straight through, as in scpi's w_constr."""
    df = _panel(treated_offset=14.0)
    w = _w(_fit(df, w_constr={"name": "lasso", "Q": 0.4}), df)
    assert np.abs(w).sum() <= 0.4 + 1e-6


def test_method_details_record_the_constraint():
    res = _fit(_panel(), w_constr="ridge")
    assert res.method_details.parameters_used["w_constr"] == "ridge"


# ---------------------------------------------------------------------------
# Failures -- reported, never silently resolved
# ---------------------------------------------------------------------------
def test_unknown_family_rejected_at_config_time():
    df = _panel()
    with pytest.raises(ValidationError):
        VanillaSCConfig(df=df, outcome="y", treat="treated", unitid="unit",
                        time="period", w_constr="bogus")


def test_unknown_family_in_dict_rejected():
    df = _panel()
    with pytest.raises(ValidationError):
        VanillaSCConfig(df=df, outcome="y", treat="treated", unitid="unit",
                        time="period", w_constr={"name": "bogus"})


@pytest.mark.parametrize("clash,message", [
    ({"covariates": ["x"]}, "covariates"),
    ({"backend": "mscmt"}, "backend"),
    ({"oracle_weights": {"d0": 1.0}}, "oracle_weights"),
])
def test_incompatible_combinations_raise(clash, message):
    """The constraint families are solved by the scpi program, not the bilevel
    predictor-weight engine; combining them would silently pick one."""
    df = _panel()
    with pytest.raises((MlsynthConfigError, ValidationError)) as exc:
        _fit(df, w_constr="lasso", **clash)
    assert message in str(exc.value)


def test_w_constr_and_staggered_spec_conflict():
    df = _panel()
    with pytest.raises((MlsynthConfigError, ValidationError)):
        _fit(df, w_constr="lasso",
             staggered_spec={"features": ["y"], "w_constr": "ridge"})


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_single_donor():
    df = _panel(n_donors=1)
    res = _fit(df, w_constr="lasso")
    assert np.isfinite(res.effects.att)
    assert len(res.weights.donor_weights) == 1


def test_default_is_unchanged_when_w_constr_is_absent():
    res = _fit(_panel())
    assert res.method_details.parameters_used.get("w_constr") is None
