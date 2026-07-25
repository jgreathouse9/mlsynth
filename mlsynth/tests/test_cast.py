"""Tests for the CAST estimator (config, setup, pipeline, estimator, plotting).

Value-for-value fidelity to the authors' ``CAST-panel`` package on the ACA
panel is pinned in the golden benchmark; here we lock behaviour on
self-contained synthetic panels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.config_models import CASTConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.cast_helpers.setup import prepare_cast_inputs


def make_panel(N=30, T=24, r=2, cohorts=((10, 20, 16), (20, 30, 20)),
               effect=3.0, noise=0.2, seed=0):
    """Staggered panel from a rank-r factor model with a planted effect.

    ``cohorts`` is a tuple of ``(start_unit, end_unit, adoption_period)``.
    """
    rng = np.random.default_rng(seed)
    U, V = rng.normal(size=(N, r)), rng.normal(size=(T, r))
    Y = U @ V.T + noise * rng.normal(size=(N, T))
    adopt = np.full(N, T)
    for lo, hi, t0 in cohorts:
        adopt[lo:hi] = t0
    for i, a in enumerate(adopt):
        if a < T:
            Y[i, a:] += effect
    rows = [dict(u=f"u{i:02d}", t=t, y=Y[i, t], d=int(t >= adopt[i]))
            for i in range(N) for t in range(T)]
    return pd.DataFrame(rows), Y, adopt


@pytest.fixture(scope="module")
def panel_df():
    return make_panel()[0]


def _cfg(df, **kw):
    return CASTConfig(df=df, outcome="y", treat="d", unitid="u", time="t", **kw)


# ------------------------------------------------------------------ config
def test_config_defaults(panel_df):
    c = _cfg(panel_df)
    assert c.rank is None and c.rank_method == "broken_stick"
    assert c.alpha == 0.05 and c.renormalize_weights is True


def test_config_rejects_rank_over_max(panel_df):
    with pytest.raises(MlsynthConfigError):
        _cfg(panel_df, rank=20, max_rank=5)


def test_config_forbids_extra_fields(panel_df):
    with pytest.raises(Exception):
        _cfg(panel_df, not_a_field=1)


@pytest.mark.parametrize("field,bad", [("rank_method", "bogus"), ("alpha", 1.5)])
def test_config_rejects_bad_values(panel_df, field, bad):
    with pytest.raises(Exception):
        _cfg(panel_df, **{field: bad})


def test_config_rejects_bad_weights(panel_df):
    with pytest.raises(MlsynthConfigError):
        _cfg(panel_df, weights=[])
    with pytest.raises(MlsynthConfigError):
        _cfg(panel_df, weights=["a", "b"])


# ------------------------------------------------------------------- setup
def test_setup_builds_staggered_panel(panel_df):
    inp = prepare_cast_inputs(panel_df, "y", "d", "u", "t")
    assert inp.Y.shape == (30, 24) and inp.N == 30 and inp.T == 24
    assert inp.A.shape == inp.Y.shape
    assert set(np.unique(inp.adoption_index)) == {16, 20, 24}   # 24 == never
    assert len(inp.treated_units) == 20 and len(inp.never_treated) == 10
    assert inp.treated_mask.sum() == (24 - 16) * 10 + (24 - 20) * 10


def test_setup_handles_single_cohort():
    df, _, _ = make_panel(N=12, T=20, cohorts=((6, 12, 14),))
    inp = prepare_cast_inputs(df, "y", "d", "u", "t")
    assert set(np.unique(inp.adoption_index)) == {14, 20}


def test_setup_handles_single_treated_unit():
    """One treated unit takes dataprep's non-cohort path (a block design)."""
    df, _, _ = make_panel(N=10, T=18, cohorts=((0, 1, 12),))
    inp = prepare_cast_inputs(df, "y", "d", "u", "t")
    assert inp.Y.shape == (10, 18)
    assert len(inp.treated_units) == 1 and len(inp.never_treated) == 9
    assert inp.adoption_index[list(inp.unit_names).index("u00")] == 12
    assert inp.treated_mask.sum() == 18 - 12


def test_single_treated_unit_end_to_end():
    df, _, _ = make_panel(N=14, T=20, cohorts=((0, 1, 14),), effect=3.0, noise=0.2)
    r = CAST(_cfg(df, rank=2)).fit()
    assert r.att == pytest.approx(3.0, abs=0.6)
    assert int(r.inputs.treated_mask.sum()) == 20 - 14


def test_setup_rejects_missing_outcomes():
    df, _, _ = make_panel(N=10, T=16, cohorts=((5, 10, 12),))
    df.loc[3, "y"] = np.nan
    with pytest.raises(MlsynthDataError):
        prepare_cast_inputs(df, "y", "d", "u", "t")


def test_setup_rejects_too_few_units():
    df, _, _ = make_panel(N=2, T=12, cohorts=((1, 2, 8),))
    with pytest.raises(MlsynthDataError):
        prepare_cast_inputs(df, "y", "d", "u", "t")


def test_setup_rejects_no_never_treated():
    df, _, _ = make_panel(N=10, T=18, cohorts=((0, 5, 12), (5, 10, 14)))
    with pytest.raises(MlsynthDataError):
        prepare_cast_inputs(df, "y", "d", "u", "t")


def test_setup_rejects_no_treated():
    df, _, _ = make_panel(N=8, T=14, cohorts=())
    with pytest.raises(MlsynthDataError):
        prepare_cast_inputs(df, "y", "d", "u", "t")


def test_setup_rejects_treatment_in_first_period():
    df, _, _ = make_panel(N=12, T=16, cohorts=((6, 12, 0),))
    with pytest.raises(MlsynthDataError):
        prepare_cast_inputs(df, "y", "d", "u", "t")


# --------------------------------------------------------------- estimator
import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
from mlsynth import CAST                                     # noqa: E402
from mlsynth.config_models import BaseEstimatorResults        # noqa: E402


def test_recovers_planted_effect():
    df, _, _ = make_panel(effect=3.0, noise=0.2)
    r = CAST(_cfg(df, rank=2)).fit()
    assert r.rank == 2
    assert r.att == pytest.approx(3.0, abs=0.3)
    lo, hi = r.att_ci
    assert lo < 3.0 < hi                                    # interval covers truth


def test_result_contract_surface():
    df, _, _ = make_panel()
    r = CAST(_cfg(df, rank=2)).fit()
    assert isinstance(r, BaseEstimatorResults)
    assert np.isfinite(r.att)
    assert r.counterfactual.shape == (24,) and r.gap.shape == (24,)
    assert r.weights is not None
    assert r.effects_panel.shape == (30, 24)
    assert r.std_errors.shape == (30, 24)


def test_entrywise_panels_defined_exactly_on_treated_cells():
    df, _, _ = make_panel()
    r = CAST(_cfg(df, rank=2)).fit()
    treated = r.inputs.treated_mask
    for panel in (r.effects_panel, r.std_errors, r.ci_lower_panel, r.ci_upper_panel):
        assert np.all(np.isfinite(panel[treated]))
        assert np.all(np.isnan(panel[~treated]))
    assert np.all(r.ci_lower_panel[treated] <= r.ci_upper_panel[treated])


def test_effect_table_is_tidy():
    df, _, _ = make_panel()
    r = CAST(_cfg(df, rank=2)).fit()
    rows = r.effect_table
    assert len(rows) == int(r.inputs.treated_mask.sum())
    assert set(rows[0]) == {"unit", "period", "effect", "se", "ci_lower", "ci_upper"}


def test_significance_counts_partition_the_treated():
    df, _, _ = make_panel(effect=3.0, noise=0.2)
    r = CAST(_cfg(df, rank=2)).fit()
    for period, s in r.significance.items():
        assert s["positive"] + s["negative"] + s["null"] == s["n_treated"]
    # a large planted positive effect is detected
    first = min(r.significance)
    assert r.significance[first]["positive"] > 0


def test_significant_units_accessor():
    df, _, _ = make_panel(effect=3.0, noise=0.2)
    r = CAST(_cfg(df, rank=2)).fit()
    period = min(r.period_effects)
    sig = r.significant_units(period, level=0.05)
    assert len(sig) > 0
    assert all(v > 0 for v in sig.values())


def test_intervals_cover_the_counterfactual_mean():
    """The interval targets the counterfactual *mean* M*, not the realized Y.

    Coverage approaches the nominal level as the panel grows, which is the
    guarantee of Theorem 1. Checked directly against the known mean matrix.
    """
    from scipy.stats import norm
    from mlsynth.utils.cast_helpers.four_block import (
        entrywise_variance, impute_missing,
    )

    N, T, r, z = 120, 80, 2, norm.ppf(0.975)
    covered = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        U, V = rng.normal(size=(N, r)), rng.normal(size=(T, r))
        M = U @ V.T
        Y = M + 0.3 * rng.normal(size=(N, T))
        adopt = np.full(N, T)
        adopt[N // 3:2 * N // 3] = int(T * 0.67)
        adopt[2 * N // 3:] = int(T * 0.83)
        A = np.ones((N, T), dtype=int)
        for i, a in enumerate(adopt):
            if a < T:
                A[i, a:] = 0
        treated = ~A.astype(bool)
        se = np.sqrt(entrywise_variance(Y, A, r))
        covered.append((np.abs(impute_missing(Y, A, r) - M)[treated]
                        <= z * se[treated]).mean())
    assert np.mean(covered) > 0.90                          # near nominal 0.95


def test_null_effect_flagged_far_less_than_a_real_effect():
    """A planted effect is detected much more often than pure noise.

    Note the significance test is on ``tau = Y - M*``, which retains the
    treated cell's own idiosyncratic noise (the interval covers ``M*``, not
    ``Y``), so the null rate is not the nominal alpha -- see the estimator's
    documentation. What must hold is the separation.
    """
    df_null, _, _ = make_panel(effect=0.0, noise=0.3, seed=1)
    df_real, _, _ = make_panel(effect=3.0, noise=0.3, seed=1)
    rate = {}
    for tag, d in (("null", df_null), ("real", df_real)):
        r = CAST(_cfg(d, rank=2, alpha=0.01)).fit()
        tot = sum(s["n_treated"] for s in r.significance.values())
        flagged = sum(s["positive"] + s["negative"] for s in r.significance.values())
        rate[tag] = flagged / tot
    assert rate["real"] > 0.95                              # real effect: nearly all
    assert rate["real"] > 2 * rate["null"]                  # clear separation


@pytest.mark.parametrize("method", ["broken_stick", "svht", "ratio"])
def test_auto_rank_selection(method):
    df, _, _ = make_panel()
    r = CAST(_cfg(df, rank=None, rank_method=method)).fit()
    assert 1 <= r.rank <= 10
    assert np.isfinite(r.att)


def test_weighted_aggregate_differs_from_equal_weights():
    df, _, _ = make_panel()
    equal = CAST(_cfg(df, rank=2)).fit()
    w = np.arange(1.0, 31.0)
    weighted = CAST(_cfg(df, rank=2, weights=list(w))).fit()
    p = min(equal.period_effects)
    assert equal.period_effects[p][0] != weighted.period_effects[p][0]


def test_unnormalized_weights_give_a_total():
    """With renormalize_weights=False the aggregate is a population total."""
    df, _, _ = make_panel(effect=3.0, noise=0.05)
    w = [100.0] * 30
    r = CAST(_cfg(df, rank=2, weights=w, renormalize_weights=False)).fit()
    p = min(r.period_effects)
    n_treated = r.significance[p]["n_treated"]
    assert r.period_effects[p][0] == pytest.approx(3.0 * 100.0 * n_treated, rel=0.2)


def test_rejects_weights_of_wrong_length():
    df, _, _ = make_panel()
    with pytest.raises(MlsynthEstimationError):
        CAST(_cfg(df, rank=2, weights=[1.0, 2.0])).fit()


def test_rejects_rank_too_large():
    df, _, _ = make_panel(N=12, T=10, cohorts=((6, 12, 8),))
    with pytest.raises(MlsynthEstimationError):
        CAST(_cfg(df, rank=10, max_rank=12)).fit()


def test_estimator_rejects_bad_config():
    df, _, _ = make_panel()
    with pytest.raises(MlsynthConfigError):
        CAST(dict(df=df, outcome="y", treat="d", unitid="u", time="t",
                  rank=99, max_rank=5))


def test_estimator_wraps_pydantic_validation_error():
    df, _, _ = make_panel()
    with pytest.raises(MlsynthConfigError):
        CAST(dict(df=df, outcome="y", treat="d", unitid="u", time="t", alpha="oops"))


def test_prediction_interval_is_populated():
    df, _, _ = make_panel()
    r = CAST(_cfg(df, rank=2)).fit()
    ts = r.time_series
    assert ts.prediction_interval_kind == "cast:entrywise"
    assert ts.has_prediction_interval


def test_plotting_smoke():
    df, _, _ = make_panel()
    r = CAST(_cfg(df, rank=2, display_graphs=False)).fit()
    from mlsynth.utils.cast_helpers.plotter import plot_cast
    fig = plot_cast(r, show=False)
    assert len(fig.axes) == 2


def test_display_graphs_path():
    df, _, _ = make_panel()
    r = CAST(_cfg(df, rank=2, display_graphs=True)).fit()
    assert np.isfinite(r.att)
