"""``objective="relative"``: TBR designs ranked by relative MDE.

The reference key ``(corr_gate, aa_gate, bb_gate, dw_gate, round(corr, 2),
1 / required_impact)`` compares designs lexicographically. Its power term is in
the outcome's own units, so across treatment groups it favours the ones with
the least sales. ``objective="relative"`` scores a split by the single scalar

    level / required_impact,

``level`` the mean of the treated aggregate over the scoring window, which is
the inverse of the relative MDE ``required_impact / (n_test * level)`` up to
the constant ``n_test``. What these tests pin:

* the key is that scalar and nothing else -- no gate, no correlation -- and the
  detail reports the impact, the level, the relative MDE, the correlation and
  the fit;
* for a fixed treatment group it orders control groups as the required impact
  does, so control matching is unchanged in kind;
* across treatment groups it can prefer a larger group the reference's power
  term ranks below a smaller one;
* the batched screen computes the same key, and the local search climbs it and
  confirms every move with the direct fit;
* a treatment-eligible geo with a non-positive level is refused at fit and
  named, and a geo that cannot be treated is not checked;
* calibrated inference matches its placebo controls under it;
* the default ``objective="reference"`` and ``objective="paper"`` are
  unchanged, against golden values computed on ``origin/main`` (65c26cc4).
"""
from __future__ import annotations

import math
import os
import warnings

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mlsynth import TBR
from mlsynth.config_models import TBRConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.tests.test_tbrmm import market_panel
from mlsynth.utils.tbr_helpers import calibration as cal
from mlsynth.utils.tbr_helpers.design.batched import design_cache, score_designs
from mlsynth.utils.tbr_helpers.design.local import local_search, neighbourhood
from mlsynth.utils.tbr_helpers.design.objective import (check_positive_levels,
                                                        required_impact,
                                                        score_split)
from mlsynth.utils.tbr_helpers.design.pipeline import build_inputs as design_inputs
from mlsynth.utils.tbr_helpers.design.search import ROLES, greedy_search
from mlsynth.utils.tbr_helpers.posterior import fit_pretest

ALL = frozenset(ROLES)
_GEOLIFT = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                        "geolift_test_data.csv")

# The goldens were computed on origin/main (65c26cc4). numpy's summation order
# differs across CI Python versions, so they are compared at 1e-9 relative,
# which any change to the scoring path exceeds by orders of magnitude.
GOLDEN_RTOL = 1e-9


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def _panel(n_geos=10, n_periods=40, seed=0):
    """Geos of differing size sharing a common factor, periods by geos."""
    rng = np.random.default_rng(seed)
    size = 100.0 * rng.lognormal(0.0, 0.8, n_geos)
    common = 1.0 + 0.6 * rng.normal(size=(n_periods, 1))
    idio = 1.0 + 0.12 * rng.normal(size=(n_periods, n_geos))
    return size * (0.5 * common + 0.5 * idio)


def _geolift_df(n_last=90):
    df = pd.read_csv(_GEOLIFT).rename(columns={"location": "geo"})
    dates = sorted(df["date"].unique())[-n_last:]
    return df[df["date"].isin(dates)]


def _geolift_window(n_last=90):
    wide = _geolift_df(n_last).pivot(index="date", columns="geo",
                                     values="Y").sort_index()
    return wide.to_numpy(float), list(wide.columns)


def _long(Y):
    T, N = Y.shape
    return pd.DataFrame([dict(geo=f"g{j:02d}", date=t, Y=float(Y[t, j]))
                         for j in range(N) for t in range(T)])


def _cfg(df, **over):
    kw = dict(df=df, unitid="geo", time="date", outcome="Y",
              max_treatment_size=2, n_test=6)
    kw.update(over)
    return TBRConfig(**kw)


def _split(Y, T, C):
    return Y[:, sorted(T)].sum(axis=1), Y[:, sorted(C)].sum(axis=1)


def _rel(Y, T, C, n_test):
    return score_split(*_split(Y, T, C), objective="relative", n_test=n_test)


def _rows(N, designs):
    UT = np.zeros((len(designs), N))
    UC = np.zeros((len(designs), N))
    for r, (T, C) in enumerate(designs):
        UT[r, list(T)] = 1.0
        UC[r, list(C)] = 1.0
    return UT, UC


def _random_designs(rng, N, m):
    out = []
    while len(out) < m:
        perm = rng.permutation(N)
        k = int(rng.integers(1, max(2, N // 3) + 1))
        c = int(rng.integers(1, N - k + 1))
        out.append((sorted(perm[:k].tolist()), set(perm[k:k + c].tolist())))
    return out


# --------------------------------------------------------------------------- #
# score_split under "relative"
# --------------------------------------------------------------------------- #
def test_the_relative_key_is_level_over_required_impact():
    """Unit: one scalar, the treated aggregate's mean over its required impact,
    and the impact is the reference's own for the same split."""
    Y = _panel(8, 40, 1)
    y, x = _split(Y, [0, 3], [1, 2, 5, 6])
    rel = score_split(y, x, objective="relative", n_test=7)
    ref = score_split(y, x, objective="reference", n_test=7)
    impact = ref.detail["required_impact"]
    level = float(np.mean(y))
    assert len(rel.key) == 1
    assert rel.key == (rel.value,)
    assert rel.value == pytest.approx(level / impact, rel=1e-13)
    assert rel.detail["required_impact"] == pytest.approx(impact, rel=1e-13)
    assert rel.detail["level"] == pytest.approx(level, rel=1e-13)
    assert rel.detail["relative_mde"] == pytest.approx(
        impact / (7 * level), rel=1e-13)
    assert rel.detail["corr"] == pytest.approx(ref.detail["corr"], rel=1e-13)
    assert rel.detail["r2"] == pytest.approx(ref.detail["r2"], rel=1e-13)


def test_the_relative_detail_carries_no_gate():
    Y = _panel(8, 40, 2)
    rel = _rel(Y, [0], [1, 2, 3], 5)
    assert set(rel.detail) == {"required_impact", "level", "relative_mde",
                               "corr", "r2"}


def test_a_split_that_fails_the_correlation_gate_is_still_ranked_on_power():
    """Unit: the reference gates a weakly correlated split to the bottom; the
    relative key has no gate, so it is the same scalar either way."""
    rng = np.random.default_rng(3)
    x = 100.0 + rng.normal(size=40)
    y = 50.0 + rng.normal(size=40)              # unrelated to x
    ref = score_split(y, x, objective="reference", n_test=5)
    rel = score_split(y, x, objective="relative", n_test=5)
    assert not ref.detail["corr_test"]
    assert abs(rel.detail["corr"]) < 0.8
    assert rel.key == (pytest.approx(np.mean(y) / ref.detail["required_impact"],
                                     rel=1e-13),)


def test_relative_ranks_a_large_noisy_group_above_a_small_quiet_one():
    """Unit: the reference's power term is in sales units and prefers the small
    group; the relative key divides by the level and prefers the large one.
    That is the only change the objective makes across treatment groups."""
    rng = np.random.default_rng(4)
    f = 1.0 + 0.3 * rng.normal(size=60)
    x = 2000.0 * f + rng.normal(size=60)
    big = 1000.0 * f + 10.0 * rng.normal(size=60)     # 100x the sales, 10x the noise
    small = 10.0 * f + 1.0 * rng.normal(size=60)
    ref_big = score_split(big, x, objective="reference", n_test=8)
    ref_small = score_split(small, x, objective="reference", n_test=8)
    rel_big = score_split(big, x, objective="relative", n_test=8)
    rel_small = score_split(small, x, objective="relative", n_test=8)
    assert ref_small.value > ref_big.value
    assert rel_big.key > rel_small.key
    assert rel_big.detail["relative_mde"] < rel_small.detail["relative_mde"]


def test_an_exact_fit_scores_infinitely_well():
    """Edge: zero residual scale is zero required impact, an unbounded score."""
    x = np.arange(1.0, 21.0)
    rel = score_split(3.0 + 2.0 * x, x, objective="relative", n_test=4)
    assert rel.value == math.inf
    assert rel.key == (math.inf,)


def _zero_mean(n, seed):
    """Integer values summing to zero, so the mean is exactly zero."""
    v = np.random.default_rng(seed).integers(-50, 50, size=n - 1).astype(float)
    return np.append(v, -v.sum())


@pytest.mark.parametrize("shift", [0.0, -200.0])
def test_a_non_positive_treated_level_is_refused_by_the_scorer(shift):
    """Failure: a relative MDE has no meaning at a level of zero or below."""
    Y = _panel(6, 30, 5)
    _, x = _split(Y, [0], [1, 2])
    y = _zero_mean(30, 5) + shift
    with pytest.raises(MlsynthDataError, match="level"):
        score_split(y, x, objective="relative", n_test=5)


def test_a_window_too_short_is_refused_under_relative():
    with pytest.raises(MlsynthDataError, match="3 pretest"):
        score_split(np.array([1.0, 2.0]), np.array([2.0, 3.0]),
                    objective="relative", n_test=1)


# --------------------------------------------------------------------------- #
# the positivity check
# --------------------------------------------------------------------------- #
def test_the_positivity_check_names_the_offending_geos():
    Y = _panel(6, 30, 6)
    Y[:, 2] = _zero_mean(30, 6)          # a mean of zero
    Y[:, 4] -= 2.0 * Y[:, 4].mean()      # a negative mean
    with pytest.raises(MlsynthDataError) as err:
        check_positive_levels(Y, list(range(6)), [f"g{j}" for j in range(6)])
    msg = str(err.value)
    assert "'g2'" in msg and "'g4'" in msg
    assert "'g0'" not in msg and "'g1'" not in msg


def test_the_positivity_check_reads_only_the_candidates():
    Y = _panel(6, 30, 7)
    Y[:, 4] -= 2.0 * Y[:, 4].mean()
    check_positive_levels(Y, [0, 1, 2, 3, 5], [f"g{j}" for j in range(6)])


def test_the_positivity_check_refuses_a_missing_level():
    Y = _panel(4, 10, 8)
    Y[3, 1] = np.nan
    with pytest.raises(MlsynthDataError, match="'g1'"):
        check_positive_levels(Y, [0, 1, 2, 3], [f"g{j}" for j in range(4)])


# --------------------------------------------------------------------------- #
# the batched screen
# --------------------------------------------------------------------------- #
def test_batched_relative_keys_equal_score_split_on_the_geolift_window():
    Y, _ = _geolift_window()
    designs = _random_designs(np.random.default_rng(9), Y.shape[1], 200)
    keys = score_designs(design_cache(Y, n_test=14), *_rows(Y.shape[1], designs),
                         objective="relative")
    assert keys.shape == (200, 1)
    for row, (T, C) in zip(keys, designs):
        assert row[0] == pytest.approx(_rel(Y, T, C, 14).key[0], rel=1e-10)


def test_a_degenerate_design_is_scored_by_the_direct_relative_fit():
    """Edge: a constant control aggregate is a computed zero in the Gram form;
    the screen hands it to score_split under the same objective."""
    Y = _panel(6, 30, 10)
    Y[:, 5] = 40.0
    keys = score_designs(design_cache(Y, n_test=5), *_rows(6, [([0], {5})]),
                         objective="relative")
    assert keys.shape == (1, 1)
    assert keys[0, 0] == _rel(Y, [0], {5}, 5).key[0]


def test_the_default_screen_is_still_the_reference_key():
    Y = _panel(8, 40, 11)
    rows = _rows(8, [([0], {1, 2, 3})])
    cache = design_cache(Y, n_test=6)
    assert np.array_equal(score_designs(cache, *rows),
                          score_designs(cache, *rows, objective="reference"))
    assert score_designs(cache, *rows).shape == (1, 6)


def test_the_screen_refuses_the_paper_objective():
    """Failure: the paper's p-values are not quadratic forms in the rows."""
    Y = _panel(6, 30, 12)
    with pytest.raises(MlsynthConfigError, match="paper"):
        score_designs(design_cache(Y, n_test=5), *_rows(6, [([0], {1, 2})]),
                      objective="paper")


# --------------------------------------------------------------------------- #
# the local search
# --------------------------------------------------------------------------- #
def _greedy(Y, roles, K, n_test, objective="relative"):
    return greedy_search(Y, roles, max_treatment_size=K, n_test=n_test,
                         objective=objective)


def test_the_local_search_climbs_the_relative_key_and_confirms_each_step():
    """Unit: every key on the trace is the direct fit's relative key, the trace
    climbs, and the returned design is one the direct fit scores at its key."""
    Y = _panel(10, 40, 13)
    roles = [ALL] * 10
    g = _greedy(Y, roles, 2, 6)[-1]
    out = local_search(Y, roles, g, n_test=6, restarts=3, seed=0,
                       objective="relative")
    assert all(len(k) == 1 for k in out.trace)
    assert all(b > a for a, b in zip(out.trace, out.trace[1:]))
    assert out.trace[0] == g.score.key
    direct = _rel(Y, out.treatment, out.control, 6)
    assert out.score.key == direct.key
    assert out.trace[-1] == out.score.key
    assert out.score.key >= g.score.key


def test_the_relative_local_search_ends_at_a_local_optimum():
    Y = _panel(10, 40, 14)
    roles = [ALL] * 10
    g = _greedy(Y, roles, 2, 6)[-1]
    out = local_search(Y, roles, g, n_test=6, restarts=0, seed=0,
                       objective="relative")
    for T, C in neighbourhood(out.treatment, set(out.control), roles):
        assert _rel(Y, T, C, 6).key <= out.score.key


def test_the_relative_local_search_moves_off_the_greedy_on_geolift():
    """The relative climb improves the relative key on the GeoLift markets."""
    Y, _ = _geolift_window()
    roles = [ALL] * Y.shape[1]
    starts = _greedy(Y, roles, 2, 14)
    for g in starts:
        out = local_search(Y, roles, g, n_test=14, restarts=1, seed=g.k,
                           objective="relative")
        assert out.score.key > g.score.key


def test_the_local_search_refuses_the_paper_objective():
    Y = _panel(6, 30, 15)
    roles = [ALL] * 6
    g = _greedy(Y, roles, 1, 5, objective="paper")[0]
    with pytest.raises(MlsynthConfigError, match="local search.*paper"):
        local_search(Y, roles, g, n_test=5, restarts=0, seed=0,
                     objective="paper")


# --------------------------------------------------------------------------- #
# the configuration
# --------------------------------------------------------------------------- #
def test_the_objective_options():
    df = _long(_panel(6, 30, 16))
    assert _cfg(df).objective == "reference"
    for o in ("reference", "paper", "relative"):
        assert _cfg(df, objective=o).objective == o
    with pytest.raises((MlsynthConfigError, ValidationError)):
        _cfg(df, objective="absolute")


def test_the_local_search_accepts_the_relative_objective():
    df = _long(_panel(6, 30, 17))
    assert _cfg(df, search="local", objective="relative").search == "local"


def test_the_local_search_still_refuses_the_paper_objective():
    df = _long(_panel(6, 30, 18))
    with pytest.raises(MlsynthConfigError, match="paper"):
        _cfg(df, search="local", objective="paper")


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #
def test_a_relative_search_through_tbr():
    """Smoke: every design is scored by the relative key, and the recommended
    design has the smallest relative MDE across sizes."""
    df = _long(_panel(10, 40, 19))
    res = TBR(_cfg(df, objective="relative", max_treatment_size=3)).fit()
    assert res.objective == "relative"
    for d in res.designs:
        assert d.objective_value == pytest.approx(
            d.detail["level"] / d.detail["required_impact"], rel=1e-12)
        assert d.detail["relative_mde"] == pytest.approx(
            1.0 / (6 * d.objective_value), rel=1e-12)
        assert all(len(k) == 1 for k in d.matching_trace)
        assert np.isfinite(d.objective_value) and d.objective_value > 0
    best = min(res.designs, key=lambda d: d.detail["relative_mde"])
    assert res.recommended == best
    assert "relative_mde" in res.power.columns and "level" in res.power.columns


def test_the_relative_level_is_the_treated_groups_mean_on_the_window():
    df = _long(_panel(10, 40, 20))
    res = TBR(_cfg(df, objective="relative")).fit()
    wide = df.pivot(index="date", columns="geo", values="Y")
    for d in res.designs:
        level = wide[d.treatment_units].sum(axis=1).mean()
        assert d.detail["level"] == pytest.approx(level, rel=1e-12)


def test_a_relative_local_search_through_tbr():
    df = _long(_panel(10, 40, 21))
    greedy = TBR(_cfg(df, objective="relative")).fit()
    res = TBR(_cfg(df, objective="relative", search="local",
                   local_restarts=2)).fit()
    for d, g in zip(res.designs, greedy.designs):
        assert d.search == "local"
        assert len(d.greedy_key) == 1
        assert d.greedy_key == tuple(g.matching_trace[-1])
        assert tuple(d.matching_trace[-1]) >= d.greedy_key
        assert d.objective_value == pytest.approx(
            d.detail["level"] / d.detail["required_impact"], rel=1e-12)


def _negative_geo(df, geo, *, rows=None):
    df = df.copy()
    sel = (df.geo == geo) if rows is None else (df.geo == geo) & rows
    df.loc[sel, "Y"] = -df.loc[sel, "Y"]
    return df


def test_a_treatment_eligible_geo_with_a_non_positive_level_is_refused():
    """Failure: refused at fit, and the message names the geo."""
    df = _negative_geo(market_panel(n_units=8, T=40, seed=1), "m03")
    with pytest.raises(MlsynthDataError, match="m03"):
        TBR(_cfg(df, objective="relative")).fit()


def test_the_reference_objective_does_not_check_the_level():
    df = _negative_geo(market_panel(n_units=8, T=40, seed=1), "m03")
    TBR(_cfg(df)).fit()


def test_a_geo_that_cannot_be_treated_may_have_any_level():
    """Edge: the check reads the treatment-eligible geos, after the business
    constraints fold forbidden geos out of the treatment role."""
    df = _negative_geo(market_panel(n_units=8, T=40, seed=1), "m03")
    df["can_treat"] = (df.geo != "m03").astype(int)
    TBR(_cfg(df, objective="relative", treatment_eligible_col="can_treat")).fit()
    TBR(_cfg(df.drop(columns="can_treat"), objective="relative",
             not_to_be_treated=["m03"])).fit()


def test_the_level_is_checked_on_the_scoring_window_only():
    """Edge: a realised window the search does not score is not checked."""
    df = market_panel(n_units=8, T=40, seed=1)
    df["post"] = (df.date >= 30).astype(int)
    df = _negative_geo(df, "m03", rows=df.post == 1)
    res = TBR(_cfg(df, objective="relative", post_col="post")).fit()
    assert res.n_periods_scored == 30


# --------------------------------------------------------------------------- #
# calibrated inference
# --------------------------------------------------------------------------- #
def _searched_calibrated(df, **over):
    kw = dict(df=df, unitid="geo", time="date", outcome="Y",
              max_treatment_size=2, n_test=5, display_graphs=False,
              inference="calibrated", n_placebos=8)
    kw.update(over)
    return TBRConfig(**kw)


def _quiet_fit(config):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TBR(config).fit()


def test_calibrated_inference_matches_placebo_controls_under_relative():
    """The placebos' control groups are matched by the relative climb, so the
    calibration is the one calibrate() returns under that objective."""
    df = market_panel(n_units=10, T=60, seed=3)
    config = _searched_calibrated(df, objective="relative")
    res = _quiet_fit(config)
    Y, units, roles = design_inputs(config)
    rec = res.recommended
    T = [units.index(u) for u in rec.treatment_units]
    C = [units.index(u) for u in rec.control_units]

    def run(objective):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cal.calibrate(Y, T, C, blank_periods=15, horizon=5,
                                 eligibility=roles, rules=None,
                                 objective=objective, n_test=5, n_placebos=8,
                                 seed=config.calibration_seed)

    relative, reference = run("relative"), run("reference")
    assert res.calibration.k == pytest.approx(relative.k, rel=1e-12)
    assert [c for _, c in relative.placebos] != [c for _, c in reference.placebos]
    assert np.isfinite(res.recommended.calibrated_required_impact)


def test_a_named_calibrated_split_under_relative_runs_and_checks_the_level():
    df = market_panel(n_units=8, T=60, seed=4)
    df["post"] = (df.date >= 55).astype(int)
    df["tr"] = (df.geo == "m00").astype(int)
    df["ct"] = (df.geo != "m00").astype(int)
    kw = dict(unitid="geo", time="date", outcome="Y", treatment_col="tr",
              control_col="ct", post_col="post", display_graphs=False,
              inference="calibrated", n_placebos=8, objective="relative")
    res = _quiet_fit(TBRConfig(df=df, **kw))
    assert res.calibration is not None
    # Negative on the design window [0, 40), positive over the pre-period
    # [0, 55): the placebos are scored on the design window, so it is refused.
    bad = _level_split(df, "m05", 40, 55, before=-1.0, after=8.0)
    with pytest.raises(MlsynthDataError, match="m05"):
        _quiet_fit(TBRConfig(df=bad, **kw))


def _level_split(df, geo, cut, end, *, before, after):
    """Scale ``geo``'s outcome by ``before`` on ``[0, cut)`` and by ``after``
    on ``[cut, end)``, and check the two windows' means differ in sign."""
    df = df.copy()
    rows = df.geo == geo
    df.loc[rows & (df.date < cut), "Y"] *= before
    df.loc[rows & (df.date >= cut) & (df.date < end), "Y"] *= after
    y = df[rows].sort_values("date").Y.to_numpy()
    assert np.sign(y[:cut].mean()) != np.sign(y[:end].mean())
    return df


def test_the_searched_level_is_checked_on_the_design_window():
    """Edge: under calibrated inference the search scores the pre-period minus
    its blank windows, and that is the window the level is checked on. A geo
    positive there and negative over the whole pre-period is accepted."""
    df = market_panel(n_units=10, T=60, seed=3)
    df = _level_split(df, "m04", 45, 60, before=1.0, after=-20.0)
    res = _quiet_fit(_searched_calibrated(df, objective="relative"))
    assert res.n_periods_scored == 45


# --------------------------------------------------------------------------- #
# the default and the paper objective are unchanged
# --------------------------------------------------------------------------- #
# Computed on origin/main (65c26cc4): the GeoLift panel's last 90 days,
# max_treatment_size=4, n_test=14. Each entry is the treatment group in the
# order the climb added it, the control group, the required impact and the key.
GOLDEN_REFERENCE_GREEDY = {
    "recommended": 1,
    "designs": [
        (["dallas"],
         ["baltimore", "boston", "chicago", "cleveland", "denver", "houston",
          "kansas city", "los angeles", "memphis", "orlando", "philadelphia",
          "reno", "salt lake city", "san francisco", "washington"],
         1001.2738162137614,
         (1.0, 1.0, 1.0, 1.0, 0.99, 0.0009987278043297105)),
        (["dallas", "denver"],
         ["baltimore", "baton rouge", "boston", "chicago", "cleveland",
          "houston", "kansas city", "los angeles", "memphis", "orlando", "reno",
          "san francisco", "washington"],
         2220.958753431646,
         (1.0, 1.0, 1.0, 1.0, 0.99, 0.0004502559979805482)),
        (["dallas", "denver", "cincinnati"],
         ["atlanta", "baltimore", "baton rouge", "chicago", "cleveland",
          "houston", "jacksonville", "kansas city", "los angeles", "memphis",
          "minneapolis", "nashville", "new york", "orlando", "philadelphia",
          "reno", "san diego", "san francisco", "washington"],
         2310.1969704288385,
         (1.0, 1.0, 1.0, 1.0, 0.99, 0.00043286352324077863)),
        (["dallas", "denver", "cincinnati", "chicago"],
         ["atlanta", "baltimore", "baton rouge", "cleveland", "houston",
          "kansas city", "memphis", "minneapolis", "nashville", "new york",
          "portland", "reno", "san diego", "washington"],
         2712.7196500232535,
         (1.0, 1.0, 1.0, 1.0, 0.99, 0.00036863374362751714)),
    ],
}

GOLDEN_REFERENCE_LOCAL = {
    "recommended": 4,
    "designs": [
        (["dallas"],
         ["baltimore", "boston", "chicago", "denver", "kansas city",
          "los angeles", "memphis", "orlando", "reno", "san francisco",
          "washington"],
         957.0691530126519,
         (1.0, 1.0, 1.0, 1.0, 0.99, 0.001044856577868183)),
        (["austin", "dallas"],
         ["baltimore", "boston", "chicago", "denver", "kansas city", "memphis",
          "miami", "oakland", "portland", "washington"],
         1177.8436507366682,
         (1.0, 1.0, 1.0, 1.0, 0.99, 0.0008490091188032995)),
        (["austin", "dallas", "minneapolis"],
         ["baltimore", "baton rouge", "boston", "chicago", "cincinnati",
          "columbus", "denver", "jacksonville", "kansas city", "memphis",
          "miami", "nashville", "new orleans", "new york", "oakland",
          "philadelphia", "salt lake city", "washington"],
         1523.5117843926291,
         (1.0, 1.0, 1.0, 1.0, 0.99, 0.0006563782507259469)),
        (["atlanta", "boston", "minneapolis", "oklahoma city"],
         ["austin", "cincinnati", "columbus", "indianapolis", "miami",
          "new orleans", "oakland", "portland", "reno", "salt lake city",
          "san diego", "tucson", "washington"],
         3588.857322794908,
         (1.0, 1.0, 1.0, 1.0, 1.0, 0.0002786402216795919)),
    ],
}

GOLDEN_PAPER = {
    "recommended": 2,
    "designs": [
        (["cincinnati"],
         ["atlanta", "baltimore", "baton rouge", "chicago", "columbus",
          "houston", "las vegas", "miami", "nashville", "new york", "portland",
          "san antonio", "san diego", "san francisco"],
         None, (0.8677435322466841,)),
        (["cincinnati", "denver"],
         ["atlanta", "baltimore", "baton rouge", "chicago", "columbus",
          "dallas", "houston", "jacksonville", "las vegas", "los angeles",
          "miami", "nashville", "portland", "san diego", "san francisco"],
         None, (0.972511722182802,)),
        (["cincinnati", "denver", "memphis"],
         ["atlanta", "baltimore", "baton rouge", "chicago", "dallas", "houston",
          "indianapolis", "jacksonville", "las vegas", "los angeles", "miami",
          "nashville", "orlando", "portland", "san diego", "san francisco"],
         None, (0.9671846718895779,)),
        (["cincinnati", "denver", "memphis", "dallas"],
         ["atlanta", "baltimore", "baton rouge", "chicago", "houston",
          "jacksonville", "las vegas", "los angeles", "miami", "nashville",
          "orlando", "portland", "san diego", "san francisco"],
         None, (0.9332887537314056,)),
    ],
}


def _geolift_fit(**over):
    return TBR(TBRConfig(df=_geolift_df(), unitid="geo", time="date",
                         outcome="Y", max_treatment_size=4, n_test=14,
                         **over)).fit()


@pytest.mark.parametrize("over,golden", [
    ({}, GOLDEN_REFERENCE_GREEDY),
    ({"search": "local", "local_restarts": 2}, GOLDEN_REFERENCE_LOCAL),
    ({"objective": "paper"}, GOLDEN_PAPER),
], ids=["reference-greedy", "reference-local", "paper"])
def test_the_existing_objectives_match_main_on_geolift(over, golden):
    res = _geolift_fit(**over)
    assert res.recommended.k == golden["recommended"]
    assert len(res.designs) == len(golden["designs"])
    for d, (treated, controls, impact, key) in zip(res.designs, golden["designs"]):
        assert d.treatment_units == treated
        assert d.control_units == controls
        got = tuple(float(v) for v in d.matching_trace[-1])
        assert len(got) == len(key)
        assert got == pytest.approx(key, rel=GOLDEN_RTOL)
        assert d.objective_value == pytest.approx(key[-1], rel=GOLDEN_RTOL)
        if impact is not None:
            assert d.detail["required_impact"] == pytest.approx(
                impact, rel=GOLDEN_RTOL)


def test_the_relative_search_on_geolift_compares_sizes_by_relative_mde():
    """The recommended design is the one with the smallest relative MDE, and
    every design passes the level check on this all-positive panel."""
    res = _geolift_fit(objective="relative")
    mdes = [d.detail["relative_mde"] for d in res.designs]
    assert res.recommended.detail["relative_mde"] == min(mdes)
    for d in res.designs:
        Y, units = _geolift_window()
        T = [units.index(u) for u in d.treatment_units]
        C = [units.index(u) for u in d.control_units]
        assert d.detail["required_impact"] == pytest.approx(
            required_impact(math.sqrt(fit_pretest(*_split(Y, T, C)).sigma_sq),
                            90, 14), rel=1e-12)
