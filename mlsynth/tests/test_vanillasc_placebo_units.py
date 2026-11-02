"""In-space placebo: every unit's gaps, weights and ratio are on the result.

``inference="placebo"`` refits VanillaSC with each donor treated. Its details
keep, for the treated unit and for every donor that was refit, the gap path,
the donor weights and the post/pre RMSPE ratio, so a caller can rebuild the
test, combine it with other designs' tests, or invert it. With
``placebo_includes_treated=True`` the treated unit stays in each placebo
unit's donor pool, as in Abadie, Diamond and Hainmueller (2010) and Ferman,
Pinto and Possebom (2020); by default it is left out.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.bilevel import BilevelSCM

T, T0, N = 16, 11, 7
UNITS = [f"u{u}" for u in range(N)]
DESIGNS = {
    "outcome-only": dict(backend="outcome-only"),
    "covariates": dict(backend="malo", covariates=["x0", "x1"]),
}


def panel(seed=0, effect=-4.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(2, N))
    rows = []
    for u in range(N):
        level, slope = 20.0 + 3.0 * u + rng.normal(), 0.8 + 0.1 * rng.normal()
        for t in range(T):
            treated = int(u == 0 and t >= T0)
            rows.append({"unit": f"u{u}", "time": t, "treat": treated,
                         "y": level + slope * t + rng.normal(scale=0.6) + effect * treated,
                         "x0": X[0, u], "x1": X[1, u]})
    return pd.DataFrame(rows)


def fit(df, design="outcome-only", **kw):
    return VanillaSC(dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
                          display_graphs=False, inference="placebo",
                          **DESIGNS[design], **kw)).fit()


def standalone(df, unit, design, keep_treated):
    """``unit`` treated from T0; the actually treated unit dropped, or kept untreated."""
    d = df.copy() if keep_treated else df[df.unit != "u0"].copy()
    d["treat"] = ((d.unit == unit) & (d.time >= T0)).astype(int)
    return VanillaSC(dict(df=d, outcome="y", treat="treat", unitid="unit", time="time",
                          display_graphs=False, inference=False, **DESIGNS[design])).fit()


def gap_of(result):
    s = result.time_series
    return (np.asarray(s.observed_outcome, float).ravel()
            - np.asarray(s.counterfactual_outcome, float).ravel())


def rmspe_ratio(gap):
    return np.sqrt(np.mean(gap[T0:] ** 2)) / np.sqrt(np.mean(gap[:T0] ** 2))


@pytest.mark.parametrize("design", list(DESIGNS))
@pytest.mark.parametrize("with_treated", [False, True])
def test_every_unit_has_its_gaps_weights_and_ratio(design, with_treated):
    res = fit(panel(), design, placebo_includes_treated=with_treated)
    d = res.inference.details
    assert set(d["unit_gaps"]) == set(d["unit_weights"]) == set(d["unit_ratios"]) == set(UNITS)
    assert d["placebo_includes_treated"] is with_treated
    np.testing.assert_allclose(d["unit_gaps"]["u0"], gap_of(res))
    assert d["unit_ratios"]["u0"] == pytest.approx(d["treated_rmspe_ratio"])
    assert all(len(g) == T for g in d["unit_gaps"].values())


def test_a_donor_placebo_is_a_fit_without_the_treated_unit():
    df = panel()
    d = fit(df).inference.details
    for unit in ("u2", "u5"):
        assert "u0" not in d["unit_weights"][unit]
        np.testing.assert_allclose(d["unit_gaps"][unit],
                                   gap_of(standalone(df, unit, "outcome-only", False)),
                                   atol=1e-6)


@pytest.mark.parametrize("design", list(DESIGNS))
def test_the_treated_unit_joins_the_pools_on_request(design):
    df = panel()
    d = fit(df, design, placebo_includes_treated=True).inference.details
    for unit in ("u2", "u5"):
        assert set(d["unit_weights"][unit]) == set(UNITS) - {unit}
        np.testing.assert_allclose(d["unit_gaps"][unit],
                                   gap_of(standalone(df, unit, design, True)), atol=1e-6)


@pytest.mark.parametrize("with_treated", [False, True])
def test_the_weights_rebuild_the_gaps(with_treated):
    df = panel()
    d = fit(df, placebo_includes_treated=with_treated).inference.details
    Y = df.pivot(index="time", columns="unit", values="y")
    for unit, weights in d["unit_weights"].items():
        synthetic = sum(w * Y[donor].to_numpy() for donor, w in weights.items())
        np.testing.assert_allclose(Y[unit].to_numpy() - synthetic, d["unit_gaps"][unit],
                                   atol=1e-8)


def test_including_the_treated_unit_changes_only_the_placebos():
    df = panel()
    a = fit(df).inference.details
    b = fit(df, placebo_includes_treated=True).inference.details
    np.testing.assert_allclose(a["unit_gaps"]["u0"], b["unit_gaps"]["u0"])
    assert a["unit_weights"]["u0"] == pytest.approx(b["unit_weights"]["u0"])


@settings(max_examples=8, deadline=None)
@given(st.integers(0, 10_000), st.booleans())
def test_the_pvalue_ranks_the_recorded_ratios(seed, with_treated):
    res = fit(panel(seed), placebo_includes_treated=with_treated)
    d = res.inference.details
    ratios = np.array([d["unit_ratios"][u] for u in UNITS])
    for unit in UNITS:
        assert d["unit_ratios"][unit] == pytest.approx(rmspe_ratio(np.array(d["unit_gaps"][unit])))
    finite = ratios[np.isfinite(ratios)]
    assert res.inference.p_value == pytest.approx(np.mean(finite >= d["unit_ratios"]["u0"]))
    assert d["n_placebos"] == finite.size - 1


@pytest.mark.parametrize("inference", [False, "lto", "placebo_cs", "scpi"])
def test_the_flag_needs_the_in_space_placebo(inference):
    with pytest.raises(MlsynthConfigError, match="placebo_includes_treated"):
        VanillaSC(dict(df=panel(), outcome="y", treat="treat", unitid="unit", time="time",
                       display_graphs=False, inference=inference,
                       placebo_includes_treated=True))


def test_a_skipped_placebo_has_no_unit_entry(monkeypatch):
    df = panel()
    y3 = df[df.unit == "u3"].sort_values("time").y.to_numpy()[:T0]
    real = BilevelSCM.fit

    def failing(engine, y_pre, Y0_pre, *a, **kw):
        if np.array_equal(np.asarray(y_pre, float), y3):
            raise MlsynthEstimationError("injected refit failure")
        return real(engine, y_pre, Y0_pre, *a, **kw)

    monkeypatch.setattr(BilevelSCM, "fit", failing)
    with pytest.warns(UserWarning, match="u3"):
        res = fit(df, "covariates", placebo_includes_treated=True)
    d = res.inference.details
    assert "u3" in d["skipped_placebos"] and "u3" not in d["unit_gaps"]
    assert set(d["unit_gaps"]) == set(UNITS) - {"u3"}
