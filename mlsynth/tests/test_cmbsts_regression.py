"""CMBSTS: the control regression reaches the counterfactual (issue #1074).

The root-cause ladder, as tests. On a panel where every series doubles at the
intervention and nothing is treated, the counterfactual has to follow the
controls into the post window. It did not: the sampler drew the state path
from the raw outcome, so a random-walk level absorbed the pre-period signal
and the regression was fit to the residual it left (coefficients near zero),
and a sampler that alternated level and coefficients could not leave that
mode. The model the paper states has ``Y = Z alpha + X beta + eps``; its
posterior carries the regression.

Rung 0: the known-answer panel. Rung 1: the outputs that moved with it (the
coefficients, the in-sample fit, the counterfactual with and without controls).
Rung 2: the sampler's components (the coefficient draw with the states
integrated out, the state draw given the coefficients). Rung 3: the invariant,
a common shock carried by the controls is not an effect. Rung 4: the
contracts at the edges. Rung 5 is in ``tools/mutation/targets.toml``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import CMBSTS
from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.cmbsts_helpers import engine as E


def shock_panel(seed=7, k=2.0, T0=63, T1=14, n_controls=6, loading=1.5, group=None):
    """The issue's panel: controls and the treated series share a base that jumps by ``k``.

    With the defaults it is the issue's reproduction draw for draw. No effect is
    added, so the true ATT is zero and the true counterfactual is the observed
    post-period path. ``group`` adds a jointly modelled series with that loading.
    """
    rng = np.random.default_rng(seed)
    T = T0 + T1
    base = 200 + 40 * np.sin(np.arange(T) / 6.0) + rng.normal(0, 8, T)
    jump = np.where(np.arange(T) >= T0, k, 1.0)
    controls = {f"c{i}": base * jump * (1 + 0.05 * i) + rng.normal(0, 6, T)
                for i in range(n_controls)}
    treated = loading * base * jump + rng.normal(0, 6, T)
    series = {"treated": treated, **controls}
    if group is not None:
        series["g"] = group * base * jump + np.random.default_rng(seed + 1).normal(0, 6, T)
    rows = [dict(unit=u, time=t, sales=v[t], treat=int(u == "treated" and t >= T0))
            for u, v in series.items() for t in range(T)]
    return pd.DataFrame(rows), series, list(controls)


def fit(panel, controls, *, group=None, niter=600, burn=150, seed=0):
    return CMBSTS(dict(df=panel, outcome="sales", unitid="unit", time="time", treat="treat",
                       group_units=group, control_units=controls, components=["trend"],
                       niter=niter, burn=burn, seed=seed, ci_alpha=0.10,
                       display_graphs=False)).fit()


def loading_on_the_base(coefficients, series=0):
    """The fitted loading on the common base: controls carry it as 1 + 0.05 i."""
    return sum(b[series] * (1 + 0.05 * int(name[1:])) for name, b in coefficients.items())


@pytest.fixture(scope="module")
def issue():
    panel, series, controls = shock_panel()
    return fit(panel, controls, niter=1000, burn=100), series, panel


# --- rung 0: the known-answer panel ---------------------------------------


def test_the_counterfactual_follows_the_controls_through_a_common_shock(issue):
    res, series, _ = issue
    inf = res.inference_detail
    truth = series["treated"][63:].mean()
    assert np.asarray(inf.counterfactual_post)[:, 0].mean() == pytest.approx(truth, rel=0.05)
    assert inf.att_lower[0] < 0 < inf.att_upper[0]


# --- rung 1: the outputs that moved with it -----------------------------------


def test_the_controls_change_the_counterfactual(issue):
    res, series, panel = issue
    alone = fit(panel[panel.unit == "treated"], None, niter=1000, burn=100)
    shift = series["treated"][63:].mean() - series["treated"][:63].mean()
    with_controls = np.asarray(res.inference_detail.counterfactual_post)[:, 0].mean()
    without = np.asarray(alone.inference_detail.counterfactual_post)[:, 0].mean()
    assert with_controls - without > 0.5 * shift


def test_the_posterior_reports_the_regression_coefficients(issue):
    res, _, _ = issue
    coef = res.posterior.coefficients
    assert list(coef) == list(res.posterior.inclusion_probs) == [f"c{i}" for i in range(6)]
    assert all(len(b) == 1 for b in coef.values())
    assert loading_on_the_base(coef) == pytest.approx(1.5, abs=0.2)


def test_the_in_sample_fit_includes_the_regression(monkeypatch):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2)) + 4
    Y = x @ np.array([[1.0], [2.0]]) + rng.normal(size=(40, 1))
    T, Z, dist, M = E.build_ssm(1, ["trend"])
    monkeypatch.setattr(E, "ffbs", lambda y, T, *a, **k: np.zeros((y.shape[0], T.shape[0])))
    out = E.run_gibbs(Y, T, Z, dist, M, np.eye(1), np.eye(1), 3, 60, 20, rng, X_pre=x)
    fitted = np.mean([x @ beta for (_, _, _, beta) in out["draws"]], axis=0)
    assert np.abs(fitted).max() > 1.0
    np.testing.assert_allclose(out["prefit_mean"], fitted)


# --- rung 2: the sampler's components ------------------------------------------


@pytest.mark.parametrize("d, p", [(1, 2), (2, 2), (2, 3)])
def test_the_coefficient_posterior_integrates_the_states_out(d, p):
    """Against GLS on the stacked observations with their exact covariance."""
    rng = np.random.default_rng(3)
    n, P1 = 12, 1e3
    X = rng.normal(size=(n, p)) + 3
    Y = 2 * rng.normal(size=(n, d)) + 5
    T, Z, dist, M = E.build_ssm(d, ["trend"])
    Qd = np.array([[0.7, 0.2], [0.2, 0.5]])[:d, :d]
    Se = np.array([[1.3, -0.4], [-0.4, 0.9]])[:d, :d]
    Q = E.build_Q(M, dist, {"level": Qd})
    H = 2.0 * np.linalg.inv(X.T @ X)
    mean, cov = E.regression_posterior(Y, X, T, Z, Q, Se, H, P1=P1)
    # level: alpha_t = alpha_0 + sum_{s<t} eta_s with alpha_0 ~ N(0, P1 I)
    C = np.zeros((n * d, n * d))
    for s in range(n):
        for t in range(n):
            C[s * d:(s + 1) * d, t * d:(t + 1) * d] = (P1 * np.eye(d) + min(s, t) * Qd
                                                       + (Se if s == t else 0))
    D = np.vstack([np.kron(np.eye(d), X[t][None, :]) for t in range(n)])  # D vec(beta) = beta' x_t
    prec = D.T @ np.linalg.solve(C, D) + np.kron(np.linalg.inv(Se), np.linalg.inv(H))
    np.testing.assert_allclose(cov, np.linalg.inv(prec), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(mean, np.linalg.solve(prec, D.T @ np.linalg.solve(C, Y.reshape(-1))),
                               rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("p, d", [(1, 1), (3, 1), (1, 2), (3, 2)])
def test_a_drawn_vector_is_unstacked_by_columns(p, d):
    B = np.arange(1.0, p * d + 1).reshape(p, d)
    np.testing.assert_array_equal(E._unvec(B.T.reshape(-1), d), B)     # vec stacks columns


def test_the_state_path_is_drawn_given_the_regression(monkeypatch):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 1)) + 5
    Y = 3 * x + rng.normal(size=(30, 1))
    T, Z, dist, M = E.build_ssm(1, ["trend"])
    seen, real = [], E.ffbs

    def recording(y, *a, **k):
        seen.append(np.array(y, copy=True))
        return real(y, *a, **k)

    monkeypatch.setattr(E, "ffbs", recording)
    monkeypatch.setattr(E, "regression_posterior",
                        lambda Yp, Xp, *a, **k: (np.full(Xp.shape[1] * Yp.shape[1], 2.5),
                                                 1e-14 * np.eye(Xp.shape[1] * Yp.shape[1])))
    E.run_gibbs(Y, T, Z, dist, M, np.eye(1), np.eye(1), 3, 20, 5, rng, X_pre=x)
    later = seen[1:]
    assert later and all(np.allclose(s, Y) or np.allclose(s, Y - 2.5 * x, atol=1e-5) for s in later)
    assert any(np.allclose(s, Y - 2.5 * x, atol=1e-5) for s in later)


# --- rung 3: the invariant ------------------------------------------------------


@settings(max_examples=4, deadline=None)
@given(k=st.floats(1.6, 3.0), seed=st.integers(0, 50))
def test_a_common_shock_through_the_controls_is_not_an_effect(k, seed):
    panel, series, controls = shock_panel(seed=seed, k=k, T0=40, T1=8, n_controls=3)
    res = fit(panel, controls, niter=400, burn=100)
    shift = series["treated"][40:].mean() - series["treated"][:40].mean()
    assert abs(res.inference_detail.att_mean[0]) < 0.25 * shift


# --- rung 4: the edges ------------------------------------------------------------


def test_a_single_control_carries_the_shock():
    panel, series, controls = shock_panel(n_controls=1)
    res = fit(panel, controls)
    truth = series["treated"][63:].mean()
    assert np.asarray(res.inference_detail.counterfactual_post)[:, 0].mean() == pytest.approx(truth, rel=0.1)


def test_every_jointly_modelled_series_follows_the_controls():
    panel, series, controls = shock_panel(group=1.2)
    res = fit(panel, controls, group=["g"])
    inf = res.inference_detail
    for k, name in enumerate(["treated", "g"]):
        shift = series[name][63:].mean() - series[name][:63].mean()
        assert abs(inf.att_mean[k]) < 0.25 * shift
    assert all(len(b) == 2 for b in res.posterior.coefficients.values())


def test_without_regressors_there_are_no_coefficients():
    panel, _, _ = shock_panel(n_controls=1)
    res = fit(panel[panel.unit == "treated"], None, niter=200, burn=50)
    assert res.posterior.coefficients is None and res.posterior.inclusion_probs is None


def test_duplicated_controls_are_reported_as_an_estimation_failure():
    panel, _, _ = shock_panel(n_controls=1)
    twin = panel[panel.unit == "c0"].assign(unit="c0_twin")
    with pytest.raises(MlsynthEstimationError, match="CMBSTS estimation failed"):
        fit(pd.concat([panel, twin]), ["c0", "c0_twin"], niter=50, burn=10)


# --- the coefficient prior's scale ------------------------------------------------


def test_the_package_prior_keeps_a_single_control_out_of_the_forecast():
    """CausalMBSTS's H = (X'X)^-1 (c = 1): the regression is worth one noise term."""
    panel, series, controls = shock_panel(n_controls=1)
    res = CMBSTS(dict(df=panel, outcome="sales", unitid="unit", time="time", treat="treat",
                      control_units=controls, components=["trend"], niter=600, burn=150,
                      seed=0, display_graphs=False, g_prior_scale=1.0)).fit()
    shift = series["treated"][63:].mean() - series["treated"][:63].mean()
    assert res.inference_detail.att_mean[0] > 0.75 * shift


@pytest.mark.parametrize("given, expected", [(None, 100 * 63), (2.5, 2.5)])
def test_the_prior_scale_reaches_the_sampler(monkeypatch, given, expected):
    from mlsynth.utils.cmbsts_helpers import pipeline

    seen = {}
    real = pipeline.run_gibbs

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(pipeline, "run_gibbs", spy)
    panel, _, controls = shock_panel(n_controls=1)
    cfg = dict(df=panel, outcome="sales", unitid="unit", time="time", treat="treat",
               control_units=controls, components=["trend"], niter=20, burn=5, seed=0,
               display_graphs=False)
    if given is not None:
        cfg["g_prior_scale"] = given
    CMBSTS(cfg).fit()
    assert seen["g_scale"] == expected


@pytest.mark.parametrize("over, match", [
    (dict(g_prior_scale=10.0), "g_prior_scale scales the coefficient prior"),
    (dict(g_prior_scale=0.0, control_units=["c0"]), "greater than 0"),
    (dict(g_prior_scale=-1.0, control_units=["c0"]), "greater than 0"),
])
def test_the_prior_scale_is_validated(over, match):
    from mlsynth.exceptions import MlsynthConfigError

    panel, _, _ = shock_panel(n_controls=1)
    cfg = dict(df=panel, outcome="sales", unitid="unit", time="time", treat="treat",
               components=["trend"], niter=20, burn=5, display_graphs=False, **over)
    with pytest.raises(MlsynthConfigError, match=match):
        CMBSTS(cfg)
