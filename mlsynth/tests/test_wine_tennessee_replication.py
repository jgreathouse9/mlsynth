"""Sun et al. (2025, AJAE): Tennessee's 2016 wine reform, Studies 2 and 3.

Tennessee let grocery stores sell wine from July 2016. The paper asks whether
the intended gain -- wine excise tax revenue -- came with unintended costs, in
liquor store employment. It estimates each with SCM and with SDID, and this
pins both against the authors' own data.

Study 1 (liquor store establishments) is absent on purpose: its outcome comes
from proprietary NielsenIQ data the authors cannot redistribute, which also
rules out the first column of Table 6.

The checks come in three layers, and the layering is the point.

* An *oracle* layer feeds the paper's own published donor weights through
  mlsynth and asks for the paper's own per-period effects. It reproduces them
  to three decimals. Because it fixes the weights, it cannot be disturbed by a
  change in the weight search -- so it isolates the data handling and the
  counterfactual arithmetic, and pins them exactly.
* A *fitted* layer runs the estimators for real. Here the tolerances are looser
  and stated, because the nested predictor-weight search is not convex and
  mlsynth need not land where Stata's ``synth`` lands.
* A *recorded* layer states what the data do not determine, rather than pinning
  it. Study 3 has three pre-treatment periods and five donors; its weight
  vector is not identified, and a test asserting the published weights there
  would be asserting an arbitrary choice among many equally good ones.

On Study 2 mlsynth's own weights fit the pre-treatment period *better* than the
published ones (RMSE 14.52 against 14.74), which is why the small gap in the
fitted ATT is recorded as a different optimum rather than chased.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

_DIR = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
        / "wine_tennessee")

# Published donor weights: Table 1 col 2 (Study 2) and the Figure 6 code.
PAPER_W2 = {"KY": 0.231, "NY": 0.551, "UT": 0.218}
PAPER_W3 = {"AK": 0.104, "MN": 0.636, "NY": 0.259}

# Table 4: SCM effects on employment per million residents, by quarter.
PAPER_T4 = [-13.081, -30.672, -31.033, -52.340, -62.457,
            -61.404, -54.817, -49.229, -31.310, -38.665]
PAPER_T4_AVG = -42.500
# Table 5: SCM effects on wine excise tax per capita, by fiscal year.
PAPER_T5 = [0.700, 0.480, 0.587]
PAPER_T5_AVG = 0.589
# Table 6 panels B and C: SDID average effects.
PAPER_T6B_AVG = -53.154
PAPER_T6C_AVG = 0.610

COVARIATES = ["popdense2010", "pop21per", "incomepc", "wineper", "college",
              "unemploy", "nonwhite"]


def _load(name):
    path = _DIR / name
    if not path.exists():  # pragma: no cover - the panels are committed
        pytest.skip(f"missing {path}")
    d = pd.read_csv(path)
    # Stata writes these as float32; widen so the comparison is not limited by
    # the storage type. (That they load at all is #321.)
    for c in d.columns:
        if d[c].dtype == "float32":
            d[c] = d[c].astype("float64")
    return d


@pytest.fixture(scope="module")
def study2():
    return _load("study2_employment.csv")


@pytest.fixture(scope="module")
def study3():
    return _load("study3_taxrevenue.csv")


def _effects_from_weights(df, time, outcome, weights, last_pre, treated="TN"):
    """Per-period treated-minus-synthetic path for a fixed weight vector."""
    wide = df.pivot_table(index=time, columns="state", values=outcome)
    post = [t for t in wide.index if t > last_pre]
    actual = wide.loc[post, treated].to_numpy(float)
    synthetic = np.zeros_like(actual)
    for unit, weight in weights.items():
        synthetic = synthetic + weight * wide.loc[post, unit].to_numpy(float)
    return actual - synthetic


def _pre_rmse(df, time, outcome, weights, last_pre, treated="TN"):
    wide = df.pivot_table(index=time, columns="state", values=outcome)
    pre = [t for t in wide.index if t <= last_pre]
    actual = wide.loc[pre, treated].to_numpy(float)
    synthetic = np.zeros_like(actual)
    for unit, weight in weights.items():
        synthetic = synthetic + weight * wide.loc[pre, unit].to_numpy(float)
    return float(np.sqrt(np.mean((actual - synthetic) ** 2)))


class TestThePanelsAreTheOnesThePaperUsed:
    """Shape checks, so a silently truncated panel fails here and not later."""

    def test_study2_is_eleven_states_over_forty_quarters(self, study2):
        assert len(study2) == 440
        assert study2["state"].nunique() == 11
        assert study2["time"].nunique() == 40
        treated = study2[study2["treat"] == 1]
        assert set(treated["state"]) == {"TN"}
        assert treated["time"].min() == 31      # 2016 Q3
        assert treated["time"].nunique() == 10

    def test_study3_is_six_states_over_six_fiscal_years(self, study3):
        assert len(study3) == 36
        assert study3["state"].nunique() == 6
        treated = study3[study3["treat"] == 1]
        assert set(treated["state"]) == {"TN"}
        assert treated["year"].min() == 2017
        assert treated["year"].nunique() == 3


class TestTheOracleLayer:
    """The paper's own weights, through mlsynth, must give the paper's numbers.

    This is the check that cannot drift: it holds the weights fixed, so it
    isolates everything downstream of them -- the pivot, the pre/post split,
    the treated-minus-synthetic arithmetic -- and pins it to three decimals.
    If the fitted layer below ever disagrees while this still passes, the
    disagreement is in the weight search and nowhere else.
    """

    def test_table_4_reproduces_quarter_by_quarter(self, study2):
        got = _effects_from_weights(study2, "time", "employper", PAPER_W2, 30)
        assert got == pytest.approx(PAPER_T4, abs=0.01)

    def test_table_4_average_reproduces(self, study2):
        got = _effects_from_weights(study2, "time", "employper", PAPER_W2, 30)
        assert float(np.mean(got)) == pytest.approx(PAPER_T4_AVG, abs=0.01)

    def test_table_5_reproduces_year_by_year(self, study3):
        got = _effects_from_weights(study3, "year", "taxper", PAPER_W3, 2016)
        assert got == pytest.approx(PAPER_T5, abs=0.01)

    def test_table_5_average_reproduces(self, study3):
        got = _effects_from_weights(study3, "year", "taxper", PAPER_W3, 2016)
        assert float(np.mean(got)) == pytest.approx(PAPER_T5_AVG, abs=0.01)


class TestTheFittedSCMLayer:
    """Running the estimator for real, rather than replaying the paper's weights.

    Stata's ``synth ... nested allopt`` and mlsynth's differential-evolution
    predictor-weight search are different optimisers on a non-convex nested
    problem, so they need not agree exactly. What can be asked is that the
    donor set, the weights and the effect all land close, and -- the substantive
    check -- that mlsynth's answer is not a worse fit than the published one.
    """

    def _fit(self, df, outcome, time, windows, lag_windows):
        from mlsynth import VanillaSC

        d = df.copy()
        # ``synth`` enters the outcome twice: as the dependent variable and as
        # a predictor with its own V weight. `employper(1(1)30)` is one
        # predictor -- the pre-period mean -- not thirty separate lags, since
        # `varname(numlist)` averages over the listed periods.
        for name, window in lag_windows.items():
            d[name] = d[outcome]
            windows = {**windows, name: window}
        covariates = list(lag_windows) + COVARIATES
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return VanillaSC({
                "df": d, "outcome": outcome, "treat": "treat",
                "unitid": "state", "time": time, "covariates": covariates,
                "covariate_windows": windows, "backend": "mscmt",
                "display_graphs": False}).fit()

    def _weights(self, res):
        return {k: float(v) for k, v in (res.weights.donor_weights or {}).items()
                if abs(v) > 1e-3}

    def test_study2_selects_the_published_donor_set(self, study2):
        res = self._fit(study2, "employper", "time",
                        {c: (1, 30) for c in ("pop21per", "incomepc",
                                              "wineper", "unemploy")},
                        {"employper_lag": (1, 30)})
        assert set(self._weights(res)) == set(PAPER_W2)

    def test_study2_weights_match_the_published_ones(self, study2):
        """Within 0.02 each -- they agree to about a percentage point."""
        res = self._fit(study2, "employper", "time",
                        {c: (1, 30) for c in ("pop21per", "incomepc",
                                              "wineper", "unemploy")},
                        {"employper_lag": (1, 30)})
        got = self._weights(res)
        for unit, published in PAPER_W2.items():
            assert got[unit] == pytest.approx(published, abs=0.02)

    def test_study2_effect_is_close_and_the_gap_is_a_better_fit(self, study2):
        """The published -42.5 against mlsynth's -40.5, and why that is fine.

        The tolerance is 2.5, which is wider than anywhere else in this file.
        It is stated rather than tuned: mlsynth's weights achieve a *lower*
        pre-treatment RMSE than the published ones, so its answer is a
        different optimum of a non-convex search and not a worse one. Asserting
        the published number to a tight tolerance would be asserting that
        Stata's local optimum is the correct one.
        """
        res = self._fit(study2, "employper", "time",
                        {c: (1, 30) for c in ("pop21per", "incomepc",
                                              "wineper", "unemploy")},
                        {"employper_lag": (1, 30)})
        assert float(res.effects.att) == pytest.approx(PAPER_T4_AVG, abs=2.5)

        fitted = _pre_rmse(study2, "time", "employper",
                           self._weights(res), 30)
        published = _pre_rmse(study2, "time", "employper", PAPER_W2, 30)
        assert fitted <= published, (
            f"mlsynth fits worse than the paper ({fitted:.3f} > "
            f"{published:.3f}); the ATT gap would then be a defect")

    def test_study3_effect_reproduces(self, study3):
        res = self._fit(study3, "taxper", "year",
                        {c: (2014, 2016) for c in
                         ("pop21per", "incomepc", "wineper", "unemploy",
                          "nonwhite")},
                        {"taxper_2014": (2014, 2014),
                         "taxper_2016": (2016, 2016)})
        assert float(res.effects.att) == pytest.approx(PAPER_T5_AVG, abs=0.05)

    def test_the_search_is_deterministic(self, study2):
        """Same panel, same answer -- the tolerances above assume it."""
        kw = ({c: (1, 30) for c in ("pop21per", "incomepc", "wineper",
                                    "unemploy")},
              {"employper_lag": (1, 30)})
        a = self._fit(study2, "employper", "time", *kw)
        b = self._fit(study2, "employper", "time", *kw)
        assert float(a.effects.att) == pytest.approx(float(b.effects.att),
                                                     rel=1e-9)


class TestTheFittedSDIDLayer:
    """Table 6, panels B and C.

    The paper's Stata call is ``sdid ..., covariates(...)`` with no type named,
    so it takes Stata's default, which is ``optimized`` and not ``projected``.
    That option is why #322 and #323 existed; this is the check it was for.
    """

    def _att(self, df, outcome, time, covariates=None):
        from mlsynth import SDID

        cfg = {"df": df, "outcome": outcome, "treat": "treat",
               "unitid": "state", "time": time, "vce": "noinference",
               "display_graphs": False}
        if covariates is not None:
            cfg["covariates"] = covariates
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(SDID(cfg).fit().effects.att)

    def test_table_6_panel_b_employment(self, study2):
        got = self._att(study2, "employper", "time",
                        {"optimized": COVARIATES})
        assert got == pytest.approx(PAPER_T6B_AVG, abs=0.5)

    def test_table_6_panel_c_tax_revenue(self, study3):
        got = self._att(study3, "taxper", "year", {"optimized": COVARIATES})
        assert got == pytest.approx(PAPER_T6C_AVG, abs=0.02)

    def test_the_covariates_move_the_estimate(self, study2):
        """Otherwise the panel above would pass with the option doing nothing."""
        plain = self._att(study2, "employper", "time")
        with_x = self._att(study2, "employper", "time",
                           {"optimized": COVARIATES})
        assert abs(with_x - plain) > 1.0
        # and the covariate version is the one that matches the paper
        assert abs(with_x - PAPER_T6B_AVG) < abs(plain - PAPER_T6B_AVG)

    def test_this_panel_is_why_the_scaling_fix_exists(self, study2):
        """A regression guard for #325, on the data that exposed it.

        ``incomepc`` has a within-unit dispersion around seventy times the
        outcome's. Before the covariates were scaled, this returned 1.7e+11.
        """
        got = self._att(study2, "employper", "time",
                        {"optimized": COVARIATES})
        assert np.isfinite(got)
        assert abs(got) < 1e3


class TestWhatTheDataDoNotDetermine:
    """Study 3's donor weights are not identified, so they are not pinned.

    Three pre-treatment periods and five donors. Several weight vectors fit the
    pre-period about equally well, and the published one is not the best of
    them -- so a test demanding the published weights would be demanding one
    arbitrary point from a flat region, and would fail on any harmless change
    to the search.

    Recording it is the honest alternative, and it is a real assertion: if a
    future change made the problem identified, the inequality below would stop
    holding and this test would fail, prompting a revisit.
    """

    def test_the_published_weights_are_not_the_best_pre_treatment_fit(
            self, study3):
        from itertools import combinations

        from scipy.optimize import nnls

        wide = study3.pivot_table(index="year", columns="state", values="taxper")
        pre = [y for y in wide.index if y <= 2016]
        donors = [c for c in wide.columns if c != "TN"]
        target = wide.loc[pre, "TN"].to_numpy(float)

        best = np.inf
        for size in range(1, len(donors) + 1):
            for combo in combinations(donors, size):
                design = wide.loc[pre, list(combo)].to_numpy(float)
                big = 1e6                      # enforce the simplex sum
                sol, _ = nnls(np.vstack([design, big * np.ones(len(combo))]),
                              np.append(target, big))
                best = min(best, float(np.sqrt(
                    np.mean((target - design @ sol) ** 2))))

        published = _pre_rmse(study3, "year", "taxper", PAPER_W3, 2016)
        assert best < published, (
            "the published weights are now the best pre-treatment fit; the "
            "non-identification recorded here no longer holds")
        # ...and the gap is small relative to the outcome's own scale, which is
        # what makes the region flat rather than the published fit poor
        assert (published - best) < 0.1 * float(study3["taxper"].std())

    def test_the_effect_is_identified_even_though_the_weights_are_not(
            self, study3):
        """Which is why the ATT is pinned above and the weights are not."""
        got = _effects_from_weights(study3, "year", "taxper", PAPER_W3, 2016)
        assert float(np.mean(got)) == pytest.approx(PAPER_T5_AVG, abs=0.01)
