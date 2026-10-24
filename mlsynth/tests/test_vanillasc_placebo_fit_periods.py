"""VanillaSC's placebo tests refit the estimator that produced the treated unit's fit.

A placebo test ranks the treated unit's post/pre RMSPE ratio among the ratios the same
procedure gives each donor (Abadie, Diamond and Hainmueller 2010), so each donor's
synthetic control has to be fit the way the treated unit's was. Two inputs of that fit
were left out of the refits:

- the outcome-fit periods. ``fit_window`` (MSCMT's ``times.dep``) and ``fit_exclude``
  restrict the periods the treated unit's weights are fit on, and the in-space placebo,
  the leave-two-out test and the placebo-inverted confidence set refit every donor on the
  whole pre-period. MSCMT's own placebo study refits each unit with the same
  ``times.dep`` (``synthPlacebo`` in ``R/mscmt.r``);
- the predictors, in the confidence set's refits, so a predictor backend's refits raised
  and the set was reported as not computed for an unrelated reason.

A refit that raised was dropped without a word, which is how a numerical failure became a
p-value over fewer placebos. It is now reported. The ratio itself is unchanged: both RMSPEs
are taken over the whole pre-period and the post-period, for every unit.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import mlsynth.utils.solvers.minnorm as minnorm
from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.bilevel import BilevelSCM
from mlsynth.utils.vanillasc_helpers.lto import lto_cumulative_interval, lto_interval, lto_placebo_test

T, T0 = 20, 14                                  # periods 0..19, treated from 14


def _panel(n_donors=6, seed=0, effect=-4.0, k=2):
    """Unit levels and trends with noise; ``k`` time-constant covariates, so their window
    means are the values themselves."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(k, n_donors + 1))
    rows = []
    for u in range(n_donors + 1):
        level, slope = 20.0 + 3.0 * u + rng.normal(), 0.8 + 0.1 * rng.normal()
        for t in range(T):
            y = level + slope * t + rng.normal(scale=0.6)
            treated = int(u == 0 and t >= T0)
            rows.append({"unit": f"u{u}", "time": t, "y": y + effect * treated, "treat": treated,
                         **{f"x{i}": X[i, u] for i in range(k)}})
    return pd.DataFrame(rows)


def _outcomes(df):
    """(T, N) outcomes, the treated unit first, in the order dataprep reports the donors."""
    return df.pivot(index="time", columns="unit", values="y")[sorted(df.unit.unique(), key=lambda s: int(s[1:]))]


def _fit(df, **kw):
    cfg = dict(df=df, outcome="y", treat="treat", unitid="unit", time="time", display_graphs=False)
    cfg.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(cfg).fit()


SPECS = [
    pytest.param({"fit_window": (4, 13)}, list(range(4, 14)), id="window"),
    pytest.param({"fit_exclude": [2, 7, 11]}, [r for r in range(T0) if r not in (2, 7, 11)], id="exclude"),
    pytest.param({"fit_window": (3, 12), "fit_exclude": [5, 9]}, [3, 4, 6, 7, 8, 10, 11, 12],
                 id="window-and-exclude"),
]


class _Spy:
    """Records what every ``BilevelSCM.fit`` call is handed, then runs it."""

    def __init__(self, monkeypatch, fail_on=None):
        self.calls, self.fail_on = [], fail_on
        real = BilevelSCM.fit

        def fit(engine, y_pre, Y0_pre, *a, X1=None, X0=None, **kw):
            self.calls.append((np.array(y_pre, float), np.array(Y0_pre, float),
                               None if X1 is None else np.array(X1, float),
                               None if X0 is None else np.array(X0, float)))
            if self.fail_on is not None and self.fail_on(np.asarray(y_pre, float)):
                raise MlsynthEstimationError("injected refit failure")
            return real(engine, y_pre, Y0_pre, *a, X1=X1, X0=X0, **kw)

        monkeypatch.setattr(BilevelSCM, "fit", fit)

    def units(self, Y, rows):
        """For every call, the unit whose outcomes on ``rows`` it fit, or None if it fit other rows."""
        out = []
        for y, *_ in self.calls:
            match = [u for u in range(Y.shape[1]) if len(y) == len(rows) and np.array_equal(y, Y[rows, u])]
            out.append(match[0] if match else None)
        return out


# --------------------------------------------------------------------------- smoke
@pytest.mark.parametrize("spec, rows", SPECS)
@pytest.mark.parametrize("mode", ["placebo", "lto", "placebo_cs"])
def test_a_placebo_test_with_restricted_fit_periods_returns_its_statistic(spec, rows, mode):
    res = _fit(_panel(), inference=mode, alpha=0.2, placebo_cs_precision=8, **spec)
    inf = res.inference
    if mode == "placebo_cs":                          # bounds, or the inversion's own reason for none
        assert inf.method.startswith("placebo-inverted confidence set")
        assert (inf.ci_lower is not None) or ("unavailable_reason" in inf.details)
    else:
        assert 0.0 <= inf.p_value <= 1.0
    if mode == "placebo":
        assert inf.details["n_placebos"] == 6


# --------------------------------------------------------------------------- unit: the periods of each refit
@pytest.mark.parametrize("spec, rows", SPECS)
def test_the_outcome_only_placebo_family_is_solved_on_the_fit_periods(spec, rows, monkeypatch):
    """The batched leave-one-out solve gets the donors' outcomes on the treated fit's periods."""
    df = _panel()
    seen = []
    real = minnorm.solve_simplex_loo_exact
    monkeypatch.setattr(minnorm, "solve_simplex_loo_exact", lambda M, **kw: seen.append(M.copy()) or real(M, **kw))
    _fit(df, inference="placebo", **spec)
    Y = _outcomes(df).to_numpy()
    assert len(seen) == 1 and np.array_equal(seen[0], Y[rows][:, 1:])


@pytest.mark.parametrize("spec, rows", SPECS)
@pytest.mark.parametrize("backend, extra", [("penalized", {"penalized_lambda": 0.1}),
                                            ("exact", {"covariates": ["x0", "x1"]})])
def test_each_placebo_refit_fits_the_treated_fit_s_periods(spec, rows, backend, extra, monkeypatch):
    """The loop that refits one donor at a time: every fit it makes is a unit's outcomes on those rows."""
    df = _panel()
    spy = _Spy(monkeypatch)
    _fit(df, inference="placebo", backend=backend, **extra, **spec)
    units = spy.units(_outcomes(df).to_numpy(), rows)
    assert None not in units
    assert sorted(units) == list(range(7))           # the treated unit's fit, then one refit per donor


@pytest.mark.parametrize("spec, rows", SPECS)
def test_each_leave_two_out_refit_fits_the_treated_fit_s_periods(spec, rows, monkeypatch):
    df = _panel(n_donors=4)
    spy = _Spy(monkeypatch)
    _fit(df, inference="lto", alpha=0.2, **spec)
    units = spy.units(_outcomes(df).to_numpy(), rows)
    assert None not in units
    assert units.count(0) >= 1 + 6                    # the main fit and the treated unit in each of 6 pairs


@pytest.mark.parametrize("spec, rows", SPECS)
def test_each_confidence_set_refit_fits_the_periods_and_the_predictors(spec, rows, monkeypatch):
    """Every unit's refit uses the treated fit's rows and the predictors of the unit and its pool."""
    df = _panel()
    spy = _Spy(monkeypatch)
    _fit(df, inference="placebo_cs", backend="exact", covariates=["x0", "x1"], alpha=0.2,
         placebo_cs_precision=8, **spec)
    Y = _outcomes(df).to_numpy()
    units = spy.units(Y, rows)
    assert None not in units and sorted(units) == list(range(7))
    X = df.groupby("unit")[["x0", "x1"]].first().loc[[f"u{u}" for u in range(7)]].to_numpy().T
    Xs = X / X.std(axis=1, ddof=1, keepdims=True)
    for u, (_, _, X1, X0) in zip(units, spy.calls):
        assert np.allclose(X1, Xs[:, u]) and np.allclose(X0, Xs[:, [v for v in range(7) if v != u]])


@pytest.mark.parametrize("entry", [lto_placebo_test, lto_interval, lto_cumulative_interval])
def test_the_leave_two_out_entry_points_take_the_fit_periods(entry):
    """Called directly: refits use ``fit_periods``, and leaving it out keeps every pre-period."""
    rng = np.random.default_rng(1)
    Y0, y = rng.normal(size=(12, 4)), rng.normal(size=12)
    calls = []

    class Engine:
        def fit(self, y_pre, Y0_pre, X1=None, X0=None, warm_start=None):
            calls.append(len(y_pre))
            return BilevelSCM("outcome-only").fit(y_pre, Y0_pre)

    kw = {} if entry is lto_placebo_test else {"alpha": 0.5}
    entry(Engine(), y, Y0, 8, fit_periods=[2, 3, 5, 7], warm_start=False, **kw)
    assert set(calls) == {4}
    calls.clear()
    entry(Engine(), y, Y0, 8, warm_start=False, **kw)
    assert set(calls) == {8}


# --------------------------------------------------------------------------- unit: failures are reported
def test_a_placebo_whose_refit_raises_is_reported_and_left_out(monkeypatch):
    df = _panel()
    Y = _outcomes(df).to_numpy()
    _Spy(monkeypatch, fail_on=lambda y: len(y) == T0 and np.array_equal(y, Y[:T0, 3]))
    with pytest.warns(UserWarning, match=r"u3.*injected refit failure"):
        res = VanillaSC(dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
                             display_graphs=False, inference="placebo", backend="penalized",
                             penalized_lambda=0.1)).fit()
    d = res.inference.details
    assert d["n_placebos"] == 5
    assert list(d["skipped_placebos"]) == ["u3"] and "injected refit failure" in d["skipped_placebos"]["u3"]


def test_a_placebo_test_whose_every_refit_raises_is_not_computed(monkeypatch):
    """No placebo, no reference distribution: a p-value of 1 would be the treated unit ranked against itself."""
    df = _panel()
    Y = _outcomes(df).to_numpy()
    _Spy(monkeypatch, fail_on=lambda y: not np.array_equal(y, Y[:T0, 0]))
    with pytest.warns(UserWarning, match="not computed"):
        res = VanillaSC(dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
                             display_graphs=False, inference="placebo", backend="penalized",
                             penalized_lambda=0.1)).fit()
    assert res.inference.p_value is None
    assert res.inference.details["computed"] is False and "u1" in res.inference.details["reason"]


def test_a_leave_two_out_pair_whose_refit_raises_is_reported(monkeypatch):
    df = _panel(n_donors=4)
    Y = _outcomes(df).to_numpy()
    _Spy(monkeypatch, fail_on=lambda y: len(y) == T0 and np.array_equal(y, Y[:T0, 2]))
    with pytest.warns(UserWarning, match=r"leave-two-out.*u2.*injected refit failure"):
        res = VanillaSC(dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
                             display_graphs=False, inference="lto", alpha=0.2)).fit()
    d = res.inference.details
    assert d["n_pairs"] == 3                          # the 3 of 6 pairs that leave u2 out
    assert len(d["skipped_pairs"]) == 3 and all("u2" in p[:2] for p in d["skipped_pairs"])


def test_a_confidence_set_whose_refit_raises_names_the_unit(monkeypatch):
    df = _panel()
    Y = _outcomes(df).to_numpy()
    _Spy(monkeypatch, fail_on=lambda y: len(y) == T0 and np.array_equal(y, Y[:T0, 4]))
    with pytest.warns(UserWarning, match=r"u4.*injected refit failure"):
        res = VanillaSC(dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
                             display_graphs=False, inference="placebo_cs", alpha=0.2,
                             placebo_cs_precision=8, backend="penalized", penalized_lambda=0.1)).fit()
    assert res.inference.ci_lower is None
    assert "u4" in res.inference.details["unavailable_reason"]


def test_a_confidence_set_with_a_predictor_backend_is_computed():
    """The refits used to omit the predictors, so every one raised and the set was 'not computed'."""
    res = _fit(_panel(), inference="placebo_cs", backend="exact", covariates=["x0", "x1"], alpha=0.2,
               placebo_cs_precision=8)
    assert "Firpo-Possebom" in res.inference.method
    assert res.inference.ci_lower is not None or "unavailable_reason" in res.inference.details


def test_a_leave_two_out_test_whose_every_pair_raises_is_not_computed(monkeypatch):
    df = _panel(n_donors=4)
    Y = _outcomes(df).to_numpy()
    _Spy(monkeypatch, fail_on=lambda y: len(y) == T0 and not np.array_equal(y, Y[:T0, 0]))
    with pytest.warns(UserWarning, match="not computed"):
        res = VanillaSC(dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
                             display_graphs=False, inference="lto", alpha=0.2)).fit()
    assert res.inference.p_value is None and "every pair's refit raised" in res.inference.details["reason"]


@pytest.mark.parametrize("mode", ["placebo", "lto", "placebo_cs"])
def test_a_w_constr_fit_says_its_placebo_test_was_not_computed(mode):
    """A w_constr fit has no engine to refit; the test used to rank the treated unit against itself (p = 1)."""
    with pytest.warns(UserWarning, match="w_constr"):
        res = VanillaSC(dict(df=_panel(), outcome="y", treat="treat", unitid="unit", time="time",
                             display_graphs=False, inference=mode, w_constr="simplex", alpha=0.2)).fit()
    inf = res.inference
    assert inf.p_value is None and inf.ci_lower is None
    reason = inf.details.get("reason") or inf.details.get("unavailable_reason")
    assert "w_constr" in reason and "scpi" in reason


@pytest.mark.parametrize("rows", [[], [3, 12], [-1, 2]])
def test_the_leave_two_out_test_refuses_fit_periods_outside_the_pre_period(rows):
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="fit_periods"):
        lto_placebo_test(BilevelSCM("outcome-only"), rng.normal(size=12), rng.normal(size=(12, 4)), 8,
                         fit_periods=rows)


# --------------------------------------------------------------------------- edge cases
@pytest.mark.parametrize("mode", ["placebo", "lto", "placebo_cs"])
def test_a_window_spanning_the_pre_period_changes_nothing(mode):
    df = _panel(seed=3)
    a = _fit(df, inference=mode, alpha=0.2, placebo_cs_precision=8)
    b = _fit(df, inference=mode, alpha=0.2, placebo_cs_precision=8, fit_window=(0, T0 - 1))
    assert a.inference.p_value == b.inference.p_value
    assert a.inference.ci_lower == b.inference.ci_lower and a.inference.details.keys() == b.inference.details.keys()


def test_a_one_period_fit_still_refits_every_donor():
    res = _fit(_panel(), inference="placebo", fit_window=(T0 - 1, T0 - 1))
    assert res.inference.details["n_placebos"] == 6 and 0.0 < res.inference.p_value <= 1.0


def test_a_two_donor_panel_refits_both_donors_on_the_window():
    res = _fit(_panel(n_donors=2), inference="placebo", fit_window=(6, 13))
    assert res.inference.details["n_placebos"] == 2


def test_west_germany_s_exact_placebo_refits_every_donor_on_the_validation_years(monkeypatch):
    """Abadie and Shang's specification, validated on 1981-1990: each of the 16 refits fits those ten years."""
    from benchmarks.cases.vanillasc_exact import config
    spy = _Spy(monkeypatch)
    res = VanillaSC(config("germany", inference="placebo")).fit()
    assert res.inference.details["n_placebos"] == 16
    assert len(spy.calls) == 17 and {len(y) for y, *_ in spy.calls} == {10}


# --------------------------------------------------------------------------- property: an independent oracle
def _simplex_fit(z, Z):
    """min ||z - Z w|| over the simplex by every support's equality-constrained fit; small pools only."""
    m = Z.shape[1]
    best = (np.inf, None)
    for k in range(1, m + 1):
        for S in itertools.combinations(range(m), k):
            A = Z[:, S]
            K = np.block([[2.0 * A.T @ A, np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])
            w = np.linalg.solve(K, np.r_[2.0 * A.T @ z, 1.0])[:k]
            if w.min() >= -1e-12:
                val = float(np.sum((z - A @ w) ** 2))
                if val < best[0]:
                    full = np.zeros(m)
                    full[list(S)] = w
                    best = (val, full)
    return best[1]


def _ratio(y, cf):
    g = y - cf
    return np.sqrt(np.mean(g[T0:] ** 2)) / np.sqrt(np.mean(g[:T0] ** 2))


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), n_donors=st.integers(2, 5), lo=st.integers(0, 6), hi=st.integers(8, T0 - 1),
       drop=st.sets(st.integers(0, T0 - 1), max_size=3))
def test_property_the_placebo_p_value_is_the_rank_among_refits_on_the_fit_periods(seed, n_donors, lo, hi, drop):
    """Each donor fit from the others on the fit periods by an independent simplex solve, the ratios
    over the whole pre-period: the treated unit's rank among them is the reported p-value."""
    rows = [r for r in range(lo, hi + 1) if r not in drop]
    assume(len(rows) >= n_donors + 1)
    df = _panel(n_donors=n_donors, seed=seed)
    kw = {"fit_window": (lo, hi)}
    if drop & set(range(lo, hi + 1)):
        kw["fit_exclude"] = sorted(drop & set(range(lo, hi + 1)))
    res = _fit(df, inference="placebo", **kw)
    Y = _outcomes(df).to_numpy()
    ratios = []
    for u in range(n_donors + 1):
        pool = [v for v in range(1, n_donors + 1) if v != u]
        ratios.append(_ratio(Y[:, u], Y[:, pool] @ _simplex_fit(Y[rows, u], Y[np.ix_(rows, pool)])))
    ratios = np.array(ratios)
    assume(np.min(np.abs(ratios[1:] - ratios[0])) > 1e-6 * ratios[0])
    assert res.inference.details["n_placebos"] == n_donors
    assert res.inference.details["treated_rmspe_ratio"] == pytest.approx(ratios[0], rel=1e-6)
    assert res.inference.p_value == pytest.approx(np.mean(ratios >= ratios[0]))
