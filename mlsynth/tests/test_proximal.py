from mlsynth.exceptions import MlsynthEstimationError, MlsynthDataError, MlsynthConfigError
import pytest
import pandas as pd
import numpy as np
from typing import Any, Dict
from unittest.mock import patch
from pydantic import ValidationError

from mlsynth.estimators.proximal import PROXIMAL
from mlsynth.config_models import PROXIMALConfig
from mlsynth.utils.proximal_helpers.structures import (
    PROXIMALResults,
    ProximalMethodFit,
)


@pytest.fixture
def sample_proximal_data(request: Any) -> pd.DataFrame:
    """Sample panel for PROXIMAL tests.

    Parameterize with ``{"with_surrogates": True}`` to add surrogate proxy
    columns.
    """
    include_surrogate_data = hasattr(request, "param") and request.param.get("with_surrogates", False)

    n_total_units = 5  # Unit 1 treated, 2/3 donors, 4/5 potential surrogates
    n_periods = 10
    treatment_start_period = 7  # 6 pre-periods, 4 post-periods

    units = np.repeat(np.arange(1, n_total_units + 1), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_total_units)

    np.random.seed(789)
    outcomes, donor_proxy_data, surrogate_specific_proxy_data = [], [], []
    for i in range(n_total_units):
        base_trend = np.linspace(start=10 + i * 2, stop=25 + i * 2, num=n_periods)
        noise = np.random.normal(0, 0.7, n_periods)
        outcomes.extend(base_trend + noise)
        donor_proxy_data.extend(np.random.rand(n_periods) * 5 + base_trend * 0.3)
        if include_surrogate_data:
            surrogate_specific_proxy_data.extend(np.random.rand(n_periods) * 3 + base_trend * 0.2)

    data = {
        "UnitIdentifier": units,
        "TimeIdx": times,
        "OutcomeValue": outcomes,
        "IsTreated": np.zeros(n_total_units * n_periods, dtype=int),
        "DonorProxyVar1": donor_proxy_data,
    }
    if include_surrogate_data:
        data["SurrogateSpecificProxyVar1"] = surrogate_specific_proxy_data

    df = pd.DataFrame(data)
    df.loc[(df["UnitIdentifier"] == 1) & (df["TimeIdx"] >= treatment_start_period), "IsTreated"] = 1
    return df


def _base(df, **extra):
    cfg = {
        "df": df, "outcome": "OutcomeValue", "treat": "IsTreated",
        "unitid": "UnitIdentifier", "time": "TimeIdx", "donors": [2, 3],
        "display_graphs": False,
    }
    cfg.update(extra)
    return cfg


# --- Construction ---

def test_proximal_creation(sample_proximal_data: pd.DataFrame) -> None:
    estimator = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["PI"], vars={"donorproxies": ["DonorProxyVar1"]})))
    assert estimator.outcome == "OutcomeValue"
    assert estimator.donors == [2, 3]
    assert estimator.methods == ["PI"]


def test_proximal_creation_from_dict(sample_proximal_data: pd.DataFrame) -> None:
    estimator = PROXIMAL(_base(
        sample_proximal_data, methods=["PI"], vars={"donorproxies": ["DonorProxyVar1"]}))
    assert isinstance(estimator.config, PROXIMALConfig)


# --- Helpers ---

def _validate_method_fit(fit: ProximalMethodFit, name: str, T: int) -> None:
    assert isinstance(fit, ProximalMethodFit)
    assert fit.name == name
    assert fit.counterfactual.shape == (T,)
    assert fit.gap.shape == (T,)
    assert fit.time_varying_effect.shape == (T,)
    assert isinstance(fit.att, float) and np.isfinite(fit.att)
    assert fit.att_se is None or isinstance(fit.att_se, float)
    assert isinstance(fit.pre_rmse, float) and isinstance(fit.post_rmse, float)
    assert isinstance(fit.alpha_weights, np.ndarray)
    assert isinstance(fit.donor_weights, dict)


# --- Two-family result contract (EffectResult conformance) ---

def test_proximal_result_is_effect_result(sample_proximal_data: pd.DataFrame) -> None:
    """PROXIMAL returns an EffectResult and lifts the headline (first-requested)
    variant into the standardized sub-models, exactly like the TSSC dispatcher.
    """
    from mlsynth.config_models import (
        BaseEstimatorResults, EffectResult, MlsynthResult,
    )

    res = PROXIMAL(_base(
        sample_proximal_data, methods=["PI", "SPSC"],
        vars={"donorproxies": ["DonorProxyVar1"]})).fit()

    assert isinstance(res, MlsynthResult)
    assert isinstance(res, BaseEstimatorResults)
    assert isinstance(res, EffectResult)

    # Standardized sub-models, populated from the headline variant (PI).
    assert res.effects is not None and res.effects.att is not None
    assert res.time_series is not None
    assert res.time_series.counterfactual_outcome is not None
    assert res.weights is not None
    assert res.method_details is not None and res.method_details.method_name == "PI"

    # Flat read-contract agrees with the headline fit and the lifted sub-model.
    assert isinstance(res.att, float)
    assert res.att == pytest.approx(res.effects.att)
    assert res.att == pytest.approx(res.pi.att)
    ci = res.att_ci
    assert ci is None or (len(ci) == 2 and ci[0] <= ci[1])


def test_proximal_sub_method_results_per_variant(sample_proximal_data: pd.DataFrame) -> None:
    """Every run variant is promoted to its own EffectResult under
    ``sub_method_results`` so cross-method tooling can consume any of them.
    """
    from mlsynth.config_models import EffectResult

    res = PROXIMAL(_base(
        sample_proximal_data, methods=["PI", "SPSC"],
        vars={"donorproxies": ["DonorProxyVar1"]})).fit()

    sub = res.sub_method_results
    assert set(sub) == {"PI", "SPSC"}
    for name, sr in sub.items():
        assert isinstance(sr, EffectResult)
        assert sr.method_details.method_name == name
        assert sr.effects.att == pytest.approx(res.methods[name].att)
        assert sr.att == pytest.approx(res.methods[name].att)


def test_proximal_spsc_conformal_rides_in_inference_details(
    sample_proximal_data: pd.DataFrame,
) -> None:
    """SPSC's per-period conformal (effect) band is surfaced under
    ``inference.details``, not mis-typed as a counterfactual band.
    """
    res = PROXIMAL(_base(
        sample_proximal_data, methods=["SPSC"],
        spsc_conformal=True)).fit()
    assert res.inference is not None
    assert res.inference.details is not None
    assert set(res.inference.details) >= {"periods", "lower", "upper"}
    # It is an effect band, so it must NOT populate the counterfactual band.
    assert not res.time_series.has_prediction_interval


def test_proximal_pioid_band_maps_to_time_series(
    sample_proximal_data: pd.DataFrame,
) -> None:
    """PIOID's genuine counterfactual band flows into the standardized
    ``time_series`` per-period band fields.
    """
    res = PROXIMAL(_base(
        sample_proximal_data, methods=["PIOID"],
        outcome_instruments=[4, 5], pioid_band=True)).fit()
    ts = res.time_series
    assert ts.has_prediction_interval
    assert ts.counterfactual_lower is not None and ts.counterfactual_upper is not None
    assert ts.prediction_interval_kind is not None


# --- PIOID over-identification (Hansen J) test ---------------------------------
#
# Validated against the authors' own linear interactive-fixed-effects DGP
# (shixu0830/SyntheticControl, Simulation/myfunctions_LinearSetting.R): log-trend
# factors ``log(1:t) + N(0, sd)``, identity control loadings, treated loads
# all-ones, true.beta = 2. Their sim is just-identified (n.Z = n.W); we keep the
# same factor/loading/error structure and add extra proxies to exercise the
# over-identifying restrictions, and contaminate one proxy for the power check.

def _authors_arrays(rng, m=3, n_proxy=6, t0=100, sd=1.0, dist="iid", ar=0.1, bad=0.0):
    """One draw of the authors' linear IFEM setting -> (Y, W, Z, T0)."""
    t = 2 * t0
    lam = np.column_stack(
        [np.log(np.arange(1, t + 1)) + rng.normal(0, sd, t) for _ in range(m)])

    def err():
        if dist == "iid":
            return rng.normal(0, sd, t)
        e = np.empty(t); e[0] = rng.normal()
        for k in range(1, t):
            e[k] = ar * e[k - 1] + rng.normal()
        return e

    eps_Y = err()
    Y = lam.sum(axis=1) + eps_Y + 2.0 * (np.arange(t) >= t0)   # true.beta = 2
    W = np.column_stack([lam[:, j] + err() for j in range(m)])
    Z = np.column_stack(
        [lam[:, i % m] + err() + (bad * eps_Y if i == 0 else 0.0)
         for i in range(n_proxy)])
    return Y, W, Z, t0


def _authors_panel(m=2, n_proxy=4, t0=60, seed=0, bad=0.0):
    """Same DGP as a long panel + (donor ids, instrument ids) for PROXIMAL."""
    rng = np.random.default_rng(seed)
    Y, W, Z, T0 = _authors_arrays(rng, m=m, n_proxy=n_proxy, t0=t0, bad=bad)
    t = 2 * t0
    cols = {"trt": Y}
    for j in range(m):
        cols[f"w{j}"] = W[:, j]
    for j in range(n_proxy):
        cols[f"z{j}"] = Z[:, j]
    rows = [{"unit": u, "time": k, "y": s[k], "treat": int(u == "trt" and k >= T0)}
            for u, s in cols.items() for k in range(t)]
    return (pd.DataFrame(rows), [f"w{j}" for j in range(m)],
            [f"z{j}" for j in range(n_proxy)])


def _rej_rate(bad, n=250, seed=0, hac_lag=0, dist="iid"):
    from mlsynth.utils.proximal_helpers.pi.overid import overid_j_test
    rng = np.random.default_rng(seed)
    pvals = np.array([
        overid_j_test(*_authors_arrays(rng, dist=dist, bad=bad), hac_lag)[2]
        for _ in range(n)])
    return float(np.mean(pvals < 0.05)), float(np.nanmean(pvals))


class TestPIOIDOveridJTest:
    def test_df_equals_excess_moments(self) -> None:
        from mlsynth.utils.proximal_helpers.pi.overid import overid_j_test
        rng = np.random.default_rng(0)
        Y, W, Z, T0 = _authors_arrays(rng, m=3, n_proxy=6)
        J, df, p = overid_j_test(Y, W, Z, T0, 0)
        assert df == 6 - 3                      # n_instruments - n_donors
        assert np.isfinite(J) and 0.0 <= p <= 1.0

    def test_just_identified_returns_nan(self) -> None:
        from mlsynth.utils.proximal_helpers.pi.overid import overid_j_test
        rng = np.random.default_rng(0)
        Y, W, Z, T0 = _authors_arrays(rng, m=3, n_proxy=3)   # d == p
        J, df, p = overid_j_test(Y, W, Z, T0, 0)
        assert df == 0 and np.isnan(J) and np.isnan(p)

    def test_size_controlled_valid_proxies(self) -> None:
        # Valid over-identified proxies: rejection near nominal, p-values roughly
        # uniform (mean ~ 0.5). lag 0 is the authors' classical i.i.d. setting.
        rej, mean_p = _rej_rate(bad=0.0, seed=1)
        assert rej < 0.15                       # not over-rejecting (true ~0.05)
        assert mean_p > 0.35                     # p-values not piled near 0

    def test_power_against_invalid_proxy(self) -> None:
        # One proxy correlated with the treated error (exclusion violation).
        rej, _ = _rej_rate(bad=1.5, seed=2)
        assert rej > 0.45                        # clearly rejects (true ~0.75)

    def test_size_controlled_under_ar_errors(self) -> None:
        rej, _ = _rej_rate(bad=0.0, seed=3, dist="AR")
        assert rej < 0.15

    def test_pioid_surfaces_overid_j_test(self) -> None:
        df, donors, instr = _authors_panel(m=2, n_proxy=4, seed=5)
        res = PROXIMAL(_base(
            df, outcome="y", treat="treat", unitid="unit", time="time",
            donors=donors, methods=["PIOID"], outcome_instruments=instr)).fit()
        p = res.pioid
        assert p.overid_j_df == 4 - 2
        assert np.isfinite(p.overid_j_stat) and 0.0 <= p.overid_j_pvalue <= 1.0

    def test_just_identified_fit_has_no_j_test(self) -> None:
        df, donors, instr = _authors_panel(m=2, n_proxy=2, seed=6)   # d == p
        res = PROXIMAL(_base(
            df, outcome="y", treat="treat", unitid="unit", time="time",
            donors=donors, methods=["PIOID"], outcome_instruments=instr)).fit()
        assert res.pioid.overid_j_stat is None
        assert res.pioid.overid_j_df is None

    def test_overid_test_toggle_off(self) -> None:
        df, donors, instr = _authors_panel(m=2, n_proxy=4, seed=7)
        res = PROXIMAL(_base(
            df, outcome="y", treat="treat", unitid="unit", time="time",
            donors=donors, methods=["PIOID"], outcome_instruments=instr,
            pioid_overid_test=False)).fit()
        assert res.pioid.overid_j_stat is None

    def test_simplex_fit_has_no_j_test(self) -> None:
        df, donors, instr = _authors_panel(m=2, n_proxy=4, seed=8)
        res = PROXIMAL(_base(
            df, outcome="y", treat="treat", unitid="unit", time="time",
            donors=donors, methods=["PIOID"], outcome_instruments=instr,
            pioid_simplex=True)).fit()
        assert res.pioid.overid_j_stat is None


def test_proximal_empty_result_is_valid_effect_result() -> None:
    """A container with no run variant stays a valid (empty) EffectResult
    rather than fabricating sub-models.
    """
    from mlsynth.config_models import EffectResult
    from mlsynth.utils.proximal_helpers.structures import PROXIMALInputs

    inputs = PROXIMALInputs(
        y=np.linspace(1.0, 3.0, 6),
        donor_outcomes=np.zeros((6, 2)),
        donor_proxies=None,
        surrogate_outcomes=None,
        surrogate_proxies=None,
        T=6, T0=3, bandwidth=1,
        time_labels=np.arange(6), treated_unit_name="t", donor_names=["d0", "d1"],
    )
    res = PROXIMALResults(inputs=inputs, pi=None, pis=None, pipost=None)
    assert isinstance(res, EffectResult)
    assert res.effects is None  # no primary -> nothing lifted
    assert res.sub_method_results is None


# --- Method selection: exactly what's asked runs ---

def test_proximal_pi_only(sample_proximal_data: pd.DataFrame) -> None:
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["PI"], vars={"donorproxies": ["DonorProxyVar1"]}))).fit()
    assert isinstance(results, PROXIMALResults) and results.mode == "proximal"
    assert list(results.methods.keys()) == ["PI"]
    assert results.pis is None and results.pipost is None and results.spsc is None
    n = sample_proximal_data["TimeIdx"].nunique()
    _validate_method_fit(results.pi, "PI", n)
    assert results.att == results.pi.att
    assert set(results.donor_weights.keys()) == {2, 3}


@pytest.mark.parametrize("sample_proximal_data", [{"with_surrogates": True}], indirect=True)
def test_proximal_all_methods(sample_proximal_data: pd.DataFrame) -> None:
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["PI", "PIS", "PIPost", "SPSC"],
        surrogates=[4, 5],
        vars={"donorproxies": ["DonorProxyVar1"], "surrogatevars": ["SurrogateSpecificProxyVar1"]}))).fit()
    assert list(results.methods.keys()) == ["PI", "PIS", "PIPost", "SPSC"]
    n = sample_proximal_data["TimeIdx"].nunique()
    for name, fit in results.methods.items():
        _validate_method_fit(fit, name, n)
    assert set(results.att_by_method()) == {"PI", "PIS", "PIPost", "SPSC"}


def test_proximal_selected_variant_is_first_method(sample_proximal_data: pd.DataFrame) -> None:
    """With PI not requested, convenience accessors forward to the first method."""
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["SPSC"], spsc_detrend=False))).fit()
    assert results.pi is None
    assert results.att == results.spsc.att


# --- SPSC: single proxy, no proxies required ---

@pytest.mark.parametrize("detrend", [False, True])
def test_proximal_spsc_no_proxies(sample_proximal_data: pd.DataFrame, detrend: bool) -> None:
    """Methods=['SPSC'] runs with only donors -- no donorproxies/surrogates."""
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["SPSC"], spsc_detrend=detrend))).fit()
    assert list(results.methods.keys()) == ["SPSC"]
    fit = results.spsc
    assert np.isfinite(fit.att)
    assert fit.metadata["variant"] == ("SPSC-DT" if detrend else "SPSC-NoDT")
    assert fit.metadata["detrend"] is detrend


def test_proximal_spsc_conformal(sample_proximal_data: pd.DataFrame) -> None:
    """Conformal intervals attach to the SPSC fit when requested."""
    n = sample_proximal_data["TimeIdx"].nunique()
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["SPSC"], spsc_detrend=False,
        spsc_conformal=True, spsc_conformal_periods=[7, 8]))).fit()
    cf = results.spsc.metadata.get("conformal")
    assert cf is not None
    assert list(cf["periods"]) == [7, 8]
    assert cf["lower"].shape == (2,) and cf["upper"].shape == (2,)
    assert np.all(cf["upper"] >= cf["lower"])


@pytest.mark.parametrize("detrend,prefix", [(True, "SPSC-DT"), (False, "SPSC-NoDT")])
def test_proximal_spsc_nonparametric_variant(
    sample_proximal_data: pd.DataFrame, detrend: bool, prefix: str
) -> None:
    """spsc_basis_degree>1 runs the nonparametric sieve and labels the variant."""
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["SPSC"], spsc_detrend=detrend,
        spsc_basis_degree=3))).fit()
    fit = results.spsc
    assert np.isfinite(fit.att)
    assert fit.metadata["basis_degree"] == 3
    assert fit.metadata["variant"] == f"{prefix}-NP3"


def test_proximal_spsc_basis_degree_must_be_positive(
    sample_proximal_data: pd.DataFrame,
) -> None:
    """The config rejects a non-positive sieve degree (Pydantic ge=1)."""
    with pytest.raises(Exception):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["SPSC"],
                               spsc_basis_degree=0))


# --- DR + PIPW (doubly robust proximal) ---

def test_proximal_dr_pipw_path(sample_proximal_data: pd.DataFrame) -> None:
    """Methods=['DR','PIPW'] run off donors + donorproxies (W, Z)."""
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["DR", "PIPW"],
        vars={"donorproxies": ["DonorProxyVar1"]}))).fit()
    assert list(results.methods.keys()) == ["DR", "PIPW"]
    assert results.dr is not None and results.pipw is not None
    n = sample_proximal_data["TimeIdx"].nunique()
    assert isinstance(results.dr.att, float)
    assert results.dr.counterfactual.shape == (n,) and np.all(np.isfinite(results.dr.counterfactual))
    # PIPW is a weighting estimator: no imputed counterfactual trajectory.
    assert np.all(np.isnan(results.pipw.counterfactual))
    assert isinstance(results.pipw.att, float)


def test_proximal_dr_requires_donorproxies(sample_proximal_data: pd.DataFrame) -> None:
    with pytest.raises(MlsynthConfigError, match="donorproxies"):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["DR"], vars={}))
    with pytest.raises(MlsynthConfigError, match="donorproxies"):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["PIPW"], vars={}))


# --- Config validation (input consistency per requested method) ---

def test_proximal_methods_required(sample_proximal_data: pd.DataFrame) -> None:
    with pytest.raises(ValidationError):
        PROXIMALConfig(**_base(sample_proximal_data, vars={"donorproxies": ["DonorProxyVar1"]}))


def test_proximal_unknown_method_rejected(sample_proximal_data: pd.DataFrame) -> None:
    with pytest.raises(MlsynthConfigError, match="Unknown PROXIMAL method"):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["NOPE"]))


def test_proximal_empty_donors_rejected(sample_proximal_data: pd.DataFrame) -> None:
    cfg = _base(sample_proximal_data, methods=["PI"], vars={"donorproxies": ["DonorProxyVar1"]})
    cfg["donors"] = []
    with pytest.raises(ValidationError):
        PROXIMALConfig(**cfg)


def test_proximal_pi_requires_donorproxies(sample_proximal_data: pd.DataFrame) -> None:
    with pytest.raises(MlsynthConfigError, match="donorproxies"):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["PI"], vars={}))
    with pytest.raises(MlsynthConfigError, match="donorproxies"):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["PI"], vars={"donorproxies": []}))


def test_proximal_spsc_needs_no_proxies(sample_proximal_data: pd.DataFrame) -> None:
    """SPSC-only must NOT require donorproxies/surrogatevars."""
    cfg = PROXIMALConfig(**_base(sample_proximal_data, methods=["SPSC"]))
    assert cfg.methods == ["SPSC"]


@pytest.mark.parametrize("sample_proximal_data", [{"with_surrogates": True}], indirect=True)
def test_proximal_pis_requires_surrogates(sample_proximal_data: pd.DataFrame) -> None:
    # surrogatevars missing
    with pytest.raises(MlsynthConfigError, match="surrogatevars"):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["PI", "PIS", "PIPost"],
                               surrogates=[4, 5], vars={"donorproxies": ["DonorProxyVar1"]}))
    # surrogates list empty
    with pytest.raises(MlsynthConfigError, match="surrogates"):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["PIS"], surrogates=[],
                               vars={"donorproxies": ["DonorProxyVar1"], "surrogatevars": ["SurrogateSpecificProxyVar1"]}))


# --- Fit-time data errors ---

def test_proximal_donorproxy_col_missing_in_df(sample_proximal_data: pd.DataFrame) -> None:
    df_missing = sample_proximal_data.drop(columns=["DonorProxyVar1"])
    estimator = PROXIMAL(PROXIMALConfig(**_base(
        df_missing, methods=["PI"], vars={"donorproxies": ["DonorProxyVar1"]})))
    with pytest.raises(MlsynthEstimationError, match="DonorProxyVar1"):
        estimator.fit()


def test_proximal_no_valid_donors_raises(sample_proximal_data: pd.DataFrame) -> None:
    estimator = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["PI"], donors=[10, 11],
        vars={"donorproxies": ["DonorProxyVar1"]})))
    with pytest.raises(MlsynthDataError, match="donor units are present"):
        estimator.fit()


def test_proximal_insufficient_pre_periods(sample_proximal_data: pd.DataFrame) -> None:
    df_short = sample_proximal_data.copy()
    df_short["IsTreated"] = 0
    df_short.loc[(df_short["UnitIdentifier"] == 1) & (df_short["TimeIdx"] >= 2), "IsTreated"] = 1
    estimator = PROXIMAL(PROXIMALConfig(**_base(
        df_short, methods=["PI"], vars={"donorproxies": ["DonorProxyVar1"]})))
    with pytest.raises(MlsynthEstimationError, match=r"(Singular matrix|Not enough pre-treatment)"):
        estimator.fit()


@pytest.mark.parametrize(
    "sample_proximal_data, nan_column",
    [
        ({"with_surrogates": False}, "OutcomeValue"),
        ({"with_surrogates": False}, "DonorProxyVar1"),
        ({"with_surrogates": True}, "SurrogateSpecificProxyVar1"),
    ],
    indirect=["sample_proximal_data"],
)
def test_proximal_fit_with_nans(sample_proximal_data: pd.DataFrame, nan_column: str) -> None:
    """Balance() imputes NaNs at the start of fit, so estimation proceeds."""
    df_with_nans = sample_proximal_data.copy()
    idx_to_nan = df_with_nans[df_with_nans["UnitIdentifier"] == 2].index[0]
    df_with_nans.loc[idx_to_nan, nan_column] = np.nan

    has_surr = "SurrogateSpecificProxyVar1" in df_with_nans.columns
    vars_ = {"donorproxies": ["DonorProxyVar1"]}
    if has_surr:
        vars_["surrogatevars"] = ["SurrogateSpecificProxyVar1"]
    methods = ["PI", "PIS", "PIPost"] if has_surr else ["PI"]

    results = PROXIMAL(PROXIMALConfig(**_base(
        df_with_nans, methods=methods, surrogates=[4, 5] if has_surr else [], vars=vars_))).fit()
    assert results.pi is not None
    if has_surr:
        assert results.pis is not None and results.pipost is not None


# --- Plotting ---

@pytest.mark.parametrize("display_graphs_flag", [True, False])
@patch("mlsynth.estimators.proximal.plot_proximal")
def test_proximal_plotting(mock_plot: Any, sample_proximal_data: pd.DataFrame, display_graphs_flag: bool) -> None:
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["PI"], vars={"donorproxies": ["DonorProxyVar1"]},
        display_graphs=display_graphs_flag, save=False))).fit()
    if display_graphs_flag:
        mock_plot.assert_called_once()
        (called_with,) = mock_plot.call_args.args
        assert called_with is results
    else:
        mock_plot.assert_not_called()
