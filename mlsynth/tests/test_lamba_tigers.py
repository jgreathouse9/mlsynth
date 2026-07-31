"""Lamba et al. (2023) tiger reserves: mlsynth against current ``tidysynth``.

The paper fits one synthetic control per Indian tiger reserve -- outcome is
cumulative forest loss over 2001-2020, donors are the untreated protected areas
in the same conservation zone, and thirteen predictors are averaged over
2001..treatment year. Forty-five reserves, treatment years spread over 2007-2015,
donor pools of twenty to forty-six. Nothing in mlsynth's existing synthetic
control benchmarks has that shape: Prop 99 and Basque are single-treated, single
adoption date, one donor pool.

This file is deliberately split into what must agree and what does not, because
the two are different in kind.

The seam must agree
    ``tidysynth`` and mlsynth are handed the same panel, and the thirteen-row
    predictor matrix each derives from it is checked column by column against
    the frozen reference. That matrix is a deterministic function of the data --
    means over stated windows -- so any disagreement is a construction bug, and
    it is asserted tightly.

The solve does not
    Given identical predictor matrices the two implementations still return
    different donor weights, because Abadie's nested search for the predictor
    weights ``V`` is not identified and the two optimisers land in different
    places. mlsynth reaches a *lower* pre-treatment MSPE than ``tidysynth`` on
    ten of the eleven reserves the reference can fit, so gating agreement would
    gate mlsynth to a worse optimum on most of the panel. Those quantities are
    pinned against mlsynth's own values instead -- a regression guard, not a
    cross-check -- and the reference values are asserted only where the two
    genuinely must coincide.

That split follows ``benchmarks/cases/vanillasc_xval_references.py``, which made
the same distinction on Prop 99 for the same reason.

A note on the published numbers, which this file does NOT target. The authors'
script calls ``grab_signficance``, a misspelling ``tidysynth`` corrected on
2021-11-30, so the paper predates a 2022-08-31 fix for time-ordering problems
that its own commit message says distorted package output. Under the version the
authors used, their code reproduces Table 1 almost exactly (6556.7 ha against a
published 6560); under the current package the same script gives materially
different numbers. mlsynth is checked against the corrected package, since
pinning to a version the maintainer has since fixed would pin the defect. See
``docs/replications/lamba_tigers.rst``.

Gold from ``benchmarks/reference/lamba_tigers/reference.R``; CI has no R.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

_ROOT = Path(__file__).resolve().parents[2]
_GOLD = _ROOT / "benchmarks" / "reference" / "lamba_tigers"
_DATA = _ROOT / "basedata" / "tiger_reserves.csv"

#: The thirteen predictors, in the order the reference emits them.
COVARIATES = ["loss", "population", "precipitation", "baselineForest", "AgeOfPA",
              "elevation", "slope", "aspect", "area", "distanceToCity",
              "ppp2000", "meanAGB", "roadLength"]


def _skip_if_missing():
    if not _GOLD.joinpath("summary.csv").exists() or not _DATA.exists():
        pytest.skip("lamba_tigers reference bundle not present")


@pytest.fixture(scope="module")
def raw():
    _skip_if_missing()
    d = pd.read_csv(_DATA)
    d.loc[d.Zone == "SB", "Zone"] = "CEEG"          # the authors' fold-in
    d["closest.city"] = d["closest.city"] / 60.0    # minutes -> hours
    return d


@pytest.fixture(scope="module")
def summary():
    _skip_if_missing()
    return pd.read_csv(_GOLD / "summary.csv")


def build_panel(raw, name):
    """The paper's per-reserve panel: the reserve plus its same-zone donors.

    Kept in the test rather than imported from the benchmark case so that a
    change to the case cannot silently redefine what is being checked.
    """
    t = raw[raw.name == name].iloc[0]
    ty = int(t["TR.year"])
    donors = raw[(raw["TR.year"] == 0) & (raw.Zone == t.Zone)]
    units = pd.concat([raw[raw.name == name], donors])
    rows = []
    for _, r in units.iterrows():
        for y in range(2001, 2021):
            rows.append({
                "name": r["name"], "year": y,
                "loss": 100 * r[f"Loss.{y}"],
                "precipitation": r[f"Rainfall.{y - 1}"],
                "population": r[f"Population.{y - 1}"],
                "baselineForest": np.log(100 * r["Forest.2000"]),
                "elevation": r["elevation"], "slope": r["slope"],
                "aspect": r["aspect"], "area": np.log(r["PAArea"] * 1e-4),
                "distanceToCity": r["closest.city"], "ppp2000": r["ppp2000"],
                "roadLength": r["Road.length"] / 1000.0,
                "meanAGB": r["AGB.mean"], "AgeOfPA": r["agePA"],
            })
    p = pd.DataFrame(rows)
    p["treat"] = ((p.name == name) & (p.year >= ty)).astype(int)
    return p, ty, len(donors)


def predictor_matrix(panel, ty):
    """The 13 x n_units predictor matrix, as the paper specifies it.

    Twelve covariates are means over ``2001..ty`` inclusive -- the treatment year
    is inside the averaging window, which is the authors' choice and is kept.
    ``AgeOfPA`` is the single value at ``ty``.
    """
    win = panel[panel.year.between(2001, ty)]
    out = win.groupby("name")[[c for c in COVARIATES if c != "AgeOfPA"]].mean()
    age = panel[panel.year == ty].set_index("name")["AgeOfPA"]
    out["AgeOfPA"] = age
    return out[COVARIATES]


def _weights(res):
    return {str(k): float(v) for k, v in res.weights.donor_weights.items()}


def _averted(panel, ty, name, weights):
    """Avoided forest loss in 2020: synthetic minus observed, in hectares.

    This is the paper's quantity and it is NOT ``effects.att``. The ATT averages
    ``observed - synthetic`` over every post-treatment year; the paper reports
    the terminal-year gap with the opposite sign. Using the ATT here gives a
    number that is wrong in both magnitude and sign, which is exactly the
    mistake this helper exists to prevent.
    """
    Y = panel.pivot(index="year", columns="name", values="loss")
    donors = [c for c in Y.columns if c != name]
    w = np.array([weights.get(u, 0.0) for u in donors], dtype=float)
    return float(Y.loc[2020, donors].values @ w - Y.loc[2020, name])


def _fit(panel, ty, **kw):
    from mlsynth import VanillaSC

    wins = {c: (2001, ty) for c in COVARIATES}
    wins["AgeOfPA"] = (ty, ty)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC({
            "df": panel, "outcome": "loss", "treat": "treat", "unitid": "name",
            "time": "year", "covariates": COVARIATES, "covariate_windows": wins,
            "fit_window": (2001, ty - 1), "display_graphs": False,
            "inference": False, **kw}).fit()


class TestTheFixtureIsWhatThePaperDescribes:
    """Asserted, not assumed -- the panel is the premise of everything below."""

    def test_treated_and_donor_counts_match_the_paper(self, raw):
        """45 treated reserves and 117 untreated, as reported in Results."""
        treated = raw[(raw["TR.year"] > 2006) & (raw["TR.year"] < 2016)]
        assert len(treated) == 45
        assert int((raw["TR.year"] == 0).sum()) == 117

    def test_treatment_years_are_staggered(self, summary):
        """The shape this case exists to cover: not a single adoption date."""
        assert summary.treatment_year.nunique() >= 5

    def test_donor_pools_are_zone_restricted_and_vary(self, summary):
        """Each reserve draws only on its own conservation zone, so the pool
        size differs by reserve -- unlike Prop 99, where it is fixed."""
        assert summary.n_donors.nunique() > 1
        assert summary.n_donors.min() >= 10

    def test_the_headline_reserve_has_the_donor_count_from_figure_1(self, raw):
        """Fig. 1 panel A reports 44 donor reserves for the Central India zone."""
        _, _, n = build_panel(raw, "Nawegaon-Nagzira Tiger Reserve")
        assert n == 44


class TestTheSeam:
    """The predictor matrix, before any solver runs.

    This is a deterministic function of the panel, so it must agree exactly.
    Checking it here means a later disagreement in the weights cannot be blamed
    on the panel.
    """

    @pytest.mark.parametrize("slug,name", [
        ("Nawegaon_Nagzira_Tiger_Reserve", "Nawegaon-Nagzira Tiger Reserve"),
        ("Udanti_WLS", "Udanti WLS"),
        ("Pilibhit_Tiger_Reserve", "Pilibhit Tiger Reserve"),
    ])
    def test_treated_predictors_match_tidysynth(self, raw, summary, slug, name):
        f = _GOLD / f"predictors_treated_{slug}.csv"
        if not f.exists():  # pragma: no cover - reference is committed
            pytest.skip(f"missing {f}")
        ref = pd.read_csv(f).set_index("variable").iloc[:, 0]
        panel, ty, _ = build_panel(raw, name)
        got = predictor_matrix(panel, ty).loc[name]
        for c in COVARIATES:
            assert got[c] == pytest.approx(float(ref[c]), rel=1e-9), c

    def test_donor_predictors_match_tidysynth_elementwise(self, raw):
        """Every donor, every predictor -- not just the treated column."""
        name = "Nawegaon-Nagzira Tiger Reserve"
        f = _GOLD / "predictors_controls_Nawegaon_Nagzira_Tiger_Reserve.csv"
        if not f.exists():  # pragma: no cover - reference is committed
            pytest.skip(f"missing {f}")
        ref = pd.read_csv(f).set_index("variable")
        panel, ty, _ = build_panel(raw, name)
        got = predictor_matrix(panel, ty)
        for donor in ref.columns:
            for c in COVARIATES:
                assert got.loc[donor, c] == pytest.approx(
                    float(ref.loc[c, donor]), rel=1e-9), f"{donor}/{c}"

    def test_the_averaging_window_includes_the_treatment_year(self, raw):
        """The authors average over 2001..ty inclusive. Stated as behaviour
        because excluding ty is the natural reading and would shift every
        predictor slightly, which the seam test above would then catch without
        explaining why."""
        name = "Nawegaon-Nagzira Tiger Reserve"
        panel, ty, _ = build_panel(raw, name)
        incl = predictor_matrix(panel, ty).loc[name, "loss"]
        excl = panel[(panel.name == name)
                     & panel.year.between(2001, ty - 1)]["loss"].mean()
        assert abs(incl - excl) > 1e-6


class TestMlsynthFitsTheReserves:
    """The estimator runs on every reserve and produces sane output."""

    def test_every_reserve_fits(self, raw, summary):
        for name in summary.reserve:
            panel, ty, _ = build_panel(raw, name)
            res = _fit(panel, ty)
            assert np.isfinite(float(res.effects.att))

    def test_avoided_loss_is_not_the_att(self, raw):
        """The paper's quantity is the terminal-year gap, not the post-period
        average, and it carries the opposite sign.

        Pinned because getting this wrong produces a plausible-looking number:
        ``effects.att`` on the headline reserve is about -1491 where the avoided
        loss is +2825. An earlier draft of the benchmark case made exactly that
        substitution and the reference comparison is what caught it.
        """
        name = "Nawegaon-Nagzira Tiger Reserve"
        panel, ty, _ = build_panel(raw, name)
        res = _fit(panel, ty)
        avoided = _averted(panel, ty, name, _weights(res))
        att = float(res.effects.att)
        assert avoided > 0 > att
        assert abs(avoided - att) > 1000.0

    def test_avoided_loss_matches_the_terminal_counterfactual(self, raw):
        """The helper agrees with the estimator's own reported counterfactual,
        so the two ways of computing it cannot silently diverge."""
        name = "Nawegaon-Nagzira Tiger Reserve"
        panel, ty, _ = build_panel(raw, name)
        res = _fit(panel, ty)
        cf = float(np.asarray(res.time_series.counterfactual_outcome).ravel()[-1])
        obs = float(panel[(panel.name == name) & (panel.year == 2020)].loss.iloc[0])
        assert _averted(panel, ty, name, _weights(res)) == pytest.approx(
            cf - obs, abs=1e-6)

    def test_weights_are_a_simplex(self, raw):
        panel, ty, _ = build_panel(raw, "Nawegaon-Nagzira Tiger Reserve")
        w = np.array(list(_fit(panel, ty).weights.donor_weights.values()))
        assert w.min() >= -1e-9
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_the_fit_is_deterministic(self, raw):
        """Two runs at the same seed agree exactly. Without this, the
        pinned values below would be untrustworthy."""
        panel, ty, _ = build_panel(raw, "Udanti WLS")
        a = float(_fit(panel, ty, seed=42).effects.att)
        b = float(_fit(panel, ty, seed=42).effects.att)
        assert a == b

    def test_the_seed_moves_the_answer_slightly_on_some_reserves(self, raw):
        """The mscmt ``V`` search is a stochastic global optimiser, and on these
        short pre-periods it does not always land in the same place.

        Recorded as behaviour rather than asserted away. Udanti moves about
        20 ha (1.4 percent of its effect) across seeds while Nawegaon-Nagzira
        and Similipal-Hadagarh are seed-exact, so the pinned values in the
        benchmark case need tolerances wide enough to cover it -- which is the
        practical reason this test exists. A drift materially larger than this
        would mean the reported effect is not well defined and should fail.
        """
        panel, ty, _ = build_panel(raw, "Udanti WLS")
        vals = [_averted(panel, ty, "Udanti WLS", _weights(_fit(panel, ty, seed=s)))
                for s in (0, 7, 42, 2024)]
        spread = max(vals) - min(vals)
        assert 0.0 < spread < 0.02 * abs(np.mean(vals))

    def test_the_headline_reserve_is_seed_stable(self, raw):
        """The contrast that keeps the test above honest: seed sensitivity is a
        property of particular panels, not of the backend in general.

        Not bit-identical -- the spread is about 8e-6 ha on an effect of 2825 --
        but eleven orders of magnitude below Udanti's, which is the distinction
        being drawn.
        """
        name = "Nawegaon-Nagzira Tiger Reserve"
        panel, ty, _ = build_panel(raw, name)
        vals = [_averted(panel, ty, name, _weights(_fit(panel, ty, seed=s)))
                for s in (0, 7, 42)]
        assert max(vals) - min(vals) < 1e-3


class TestMlsynthFitsBetterThanTidysynth:
    """Why the donor weights are not gated against the reference.

    Abadie's outer problem chooses the predictor weights ``V`` to minimise the
    pre-treatment outcome MSPE. On these panels ``tidysynth`` returns a ``V``
    that puts almost no weight on the outcome's own history -- 1.2 percent on
    ``loss`` against 58 percent on ``elevation`` for the headline reserve -- and
    lands on a pre-treatment fit several times worse than mlsynth's. Asserting
    agreement on the weights would therefore hold mlsynth to a worse solution of
    the problem both are solving.

    Recorded as a test rather than a comment so that if a future change makes
    mlsynth the worse of the two, it fails here rather than quietly passing.
    """

    def _pre_mspe(self, panel, ty, weights):
        Y = panel.pivot(index="year", columns="name", values="loss")
        name = panel.loc[panel.treat == 1, "name"].iloc[0]
        donors = [c for c in Y.columns if c != name]
        pre = [y for y in Y.index if y < ty]
        w = np.array([weights.get(u, 0.0) for u in donors], dtype=float)
        return float(((Y.loc[pre, name] - Y.loc[pre, donors].values @ w) ** 2).mean())

    #: The one reserve where ``tidysynth`` fits the pre-period better than
    #: mlsynth (364.4 against 334.5). Named rather than hidden behind a
    #: threshold, so that if a future change fixes or worsens it the test says
    #: which reserve moved.
    KNOWN_EXCEPTION = "Similipal-Hadagarh WLS/Tiger Reserve"

    def test_mlsynth_fits_better_on_all_but_one_reserve(self, raw, summary):
        """Ten of the eleven reserves the reference could fit.

        This is the quantity that justifies not gating the donor weights
        against the reference, so it is measured over every reserve rather than
        spot-checked. Asserting the count exactly -- rather than 'most' -- means
        a change in either direction fails and has to be looked at.
        """
        better, worse = [], []
        for name in summary.reserve:
            panel, ty, _ = build_panel(raw, name)
            mine = self._pre_mspe(panel, ty, _weights(_fit(panel, ty)))
            ref = float(summary.loc[summary.reserve == name, "pre_mspe"].iloc[0])
            (better if mine < ref else worse).append(name)
        assert len(better) == 10, f"better on {len(better)}: worse = {worse}"
        assert worse == [self.KNOWN_EXCEPTION]

    def test_the_headline_reserve_is_fit_far_better(self, raw, summary):
        """Where it matters most, the margin is not marginal: 3490 against
        12521, a factor of 3.6."""
        name = "Nawegaon-Nagzira Tiger Reserve"
        panel, ty, _ = build_panel(raw, name)
        mine = self._pre_mspe(panel, ty, _weights(_fit(panel, ty)))
        ref = float(summary.loc[summary.reserve == name, "pre_mspe"].iloc[0])
        assert mine < ref / 3.0


class TestFailuresAreReported:
    """Degenerate variants of this design raise rather than return nonsense."""

    def test_a_reserve_with_no_donors_in_its_zone(self, raw):
        from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError

        panel, ty, _ = build_panel(raw, "Udanti WLS")
        solo = panel[panel.treat == 1].copy()
        with pytest.raises((MlsynthDataError, MlsynthEstimationError)):
            _fit(solo, ty)

    def test_unknown_covariate_is_rejected(self, raw):
        from mlsynth import VanillaSC
        from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError

        panel, ty, _ = build_panel(raw, "Udanti WLS")
        with pytest.raises((MlsynthConfigError, MlsynthDataError)):
            VanillaSC({
                "df": panel, "outcome": "loss", "treat": "treat",
                "unitid": "name", "time": "year", "covariates": ["ghost"],
                "display_graphs": False, "inference": False}).fit()
