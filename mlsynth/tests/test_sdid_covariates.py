"""SDID with time-varying covariates: the Kranz (2022) two-step adjustment.

`SDID` admits only unit and time fixed effects. Kranz's `xsynthdid` adds
time-varying covariates by adjusting the outcome first and then running ordinary
SDID on the adjusted outcome:

1. fit ``y ~ x | unit + time`` on the rows with no treatment;
2. take the covariate coefficients only -- the fixed effects stay in, because
   SDID handles those itself;
3. subtract ``X @ beta`` from the outcome, evaluated on the *whole* panel;
4. run SDID on the result.

Three things about that are easy to get wrong and are pinned separately below:
the regression is fit on untreated rows but applied to all rows; only the
covariate coefficients are removed, not the fixed effects; and ``add_mean``
defaults to off, so the adjusted outcome does not preserve the original mean.

Tested at two levels. The seam -- ``beta`` and the adjusted outcome vector --
isolates the regression and the projection from the SDID solve, so a defect is
localised before the estimate is involved. The endpoint then checks the
estimate. A wrong regression and a wrong solve both move the endpoint; only the
seam tells them apart.

Gold from a live ``xsynthdid`` 0.1.0 run on the package README's own worked
example, where the author printed the expected values (39.760 unadjusted, 49.398
adjusted, true effect 50). See ``benchmarks/reference/sdid_kranz/reference.R``.

The endpoint is checked against ``synthdid`` run to convergence rather than
against those two printed numbers, which are its projected-gradient solver at
its default stopping rule. ``TestSynthdidsEarlyStop`` records the difference and
why it is not a tolerance to be widened.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

_GOLD = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
         / "sdid_kranz")


def _gold_values():
    txt = (_GOLD / "gold.txt").read_text().splitlines()
    out = {}
    for line in txt:
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


@pytest.fixture(scope="module")
def panel():
    p = _GOLD / "panel.csv"
    if not p.exists():  # pragma: no cover - gold is committed
        pytest.skip(f"missing {p}")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def gold():
    return _gold_values()


class TestTheFixtureIsDiagnostic:
    """The premise the rest of the file rests on, asserted not assumed."""

    def test_the_covariate_actually_moves_the_estimate(self, gold):
        """If adjusting made no difference, every test below would pass trivially."""
        raw = float(gold["tau_unadjusted"])
        adj = float(gold["tau_adjusted"])
        assert abs(adj - raw) > 5.0

    def test_the_adjustment_moves_it_toward_the_truth(self, gold):
        """The point of the method, and a guard against a sign error."""
        true = float(gold["true_tau"])
        raw = float(gold["tau_unadjusted"])
        adj = float(gold["tau_adjusted"])
        assert abs(adj - true) < abs(raw - true)

    def test_the_estimation_rows_are_not_the_whole_panel(self, panel):
        """The regression is fit on untreated rows only; the panel has both."""
        n_untreated = int((panel.treat_exp == 0).sum())
        assert 0 < n_untreated < len(panel)

    def test_the_estimation_rows_are_an_unbalanced_panel(self, panel):
        """Which is why the within-transformation has to be iterated.

        Controls contribute every period; treated units contribute only their
        pre-periods. On an unbalanced panel a single pass of
        subtract-unit-means-then-time-means does not project onto the space
        orthogonal to both, so the coefficient it yields is not the two-way
        fixed-effects one.
        """
        untreated = panel[panel.treat_exp == 0]
        per_unit = untreated.groupby("i").size()
        assert per_unit.nunique() > 1

    def test_one_pass_demeaning_would_give_the_wrong_coefficient(self, panel, gold):
        """Stated as behaviour so the cheaper implementation cannot creep back.

        A single pass is exact on a balanced panel, so it is a tempting
        simplification and it would still look roughly right.
        """
        from mlsynth.utils.sdid_helpers.covariates import _absorb

        u = panel[panel.treat_exp == 0]
        uc, uu = pd.factorize(u["i"], sort=True)
        tc, tu = pd.factorize(u["t"], sort=True)

        def one_pass(v):
            v = np.asarray(v, dtype=float).copy()
            for codes, n in ((uc, len(uu)), (tc, len(tu))):
                cnt = np.bincount(codes, minlength=n).astype(float)
                v = v - (np.bincount(codes, weights=v, minlength=n) / cnt)[codes]
            return v

        naive = float(np.linalg.lstsq(
            one_pass(u["x"].to_numpy())[:, None], one_pass(u["y"].to_numpy()),
            rcond=None)[0][0])
        iterated = float(np.linalg.lstsq(
            _absorb(u["x"].to_numpy(dtype=float)[:, None], uc, tc, len(uu), len(tu)),
            _absorb(u["y"].to_numpy(dtype=float), uc, tc, len(uu), len(tu)),
            rcond=None)[0][0])
        target = float(gold["beta_x"])
        assert abs(iterated - target) < 1e-8
        assert abs(naive - target) > 1e-4


class TestTheSeam:
    """The regression and the projection, before SDID is involved."""

    def test_beta_matches_fixest(self, panel, gold):
        """The two-way fixed-effects coefficient on ``x``.

        The cheapest scalar that can detect a wrong regression: if this is off,
        nothing downstream can be right, and it fails without the SDID solve
        having run.
        """
        from mlsynth.utils.sdid_helpers.covariates import twfe_covariate_beta

        beta = twfe_covariate_beta(
            panel[panel.treat_exp == 0], outcome="y", covariates=["x"],
            unit="i", time="t")
        np.testing.assert_allclose(beta, [float(gold["beta_x"])],
                                   rtol=0, atol=1e-8)

    def test_adjusted_outcome_matches_r_elementwise(self, panel):
        """All 900 rows, not a summary of them."""
        from mlsynth.utils.sdid_helpers.covariates import adjust_outcome_for_covariates

        got = adjust_outcome_for_covariates(
            panel, unit="i", time="t", outcome="y", treat="treat_exp",
            covariates=["x"])
        np.testing.assert_allclose(got, panel["y.adj"].to_numpy(),
                                   rtol=0, atol=1e-8)

    def test_the_adjustment_is_applied_to_treated_rows_too(self, panel):
        """Fit on untreated rows, applied everywhere.

        Stated as behaviour because restricting the projection to the
        estimation rows is the natural misreading, and it would leave the
        treated rows untouched.
        """
        from mlsynth.utils.sdid_helpers.covariates import adjust_outcome_for_covariates

        got = adjust_outcome_for_covariates(
            panel, unit="i", time="t", outcome="y", treat="treat_exp",
            covariates=["x"])
        treated = panel.treat_exp == 1
        assert treated.any()
        assert np.abs(got[treated.to_numpy()]
                      - panel.loc[treated, "y"].to_numpy()).max() > 1e-6

    def test_fixed_effects_are_not_subtracted(self, panel):
        """Only the covariate effect comes out; SDID handles unit/time itself.

        If the fixed effects were removed too, the adjusted outcome would be
        approximately double-demeaned and its unit means would collapse toward
        a constant. They do not.
        """
        from mlsynth.utils.sdid_helpers.covariates import adjust_outcome_for_covariates

        got = adjust_outcome_for_covariates(
            panel, unit="i", time="t", outcome="y", treat="treat_exp",
            covariates=["x"])
        spread = pd.Series(got).groupby(panel["i"].to_numpy()).mean().std()
        assert spread > 1.0

    def test_add_mean_shifts_without_changing_differences(self, panel):
        """``add_mean=True`` re-centres and must not otherwise move anything."""
        from mlsynth.utils.sdid_helpers.covariates import adjust_outcome_for_covariates

        kw = dict(unit="i", time="t", outcome="y", treat="treat_exp",
                  covariates=["x"])
        a = adjust_outcome_for_covariates(panel, **kw)
        b = adjust_outcome_for_covariates(panel, add_mean=True, **kw)
        shift = b - a
        np.testing.assert_allclose(shift, np.full_like(shift, shift[0]),
                                   rtol=0, atol=1e-10)
        assert abs(b.mean() - panel["y"].mean()) < 1e-8


class TestTheEndpoint:
    """The SDID estimate itself, on both outcomes.

    Checked against ``synthdid`` run to convergence, not against its default
    stopping rule. ``synthdid_estimate`` solves the weight programs by projected
    gradient and stops once the objective decrease falls under
    ``min.decrease = 1e-5 * noise.level``; mlsynth solves them exactly. On this
    panel the two are not the same number, and the gap is pinned separately in
    ``TestSynthdidsEarlyStop`` below so it stays a recorded fact.
    """

    def _att(self, df, outcome, **extra):
        import warnings

        from mlsynth import SDID

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SDID({
                "df": df, "outcome": outcome, "treat": "treat_exp",
                "unitid": "i", "time": "t", "vce": "noinference",
                "display_graphs": False, **extra}).fit()
        return float(res.effects.att)

    def test_unadjusted_matches_r(self, panel, gold):
        """The control. Pinned so a failure names which half broke."""
        got = self._att(panel, "y")
        assert got == pytest.approx(
            float(gold["tau_unadjusted_converged"]), abs=1e-6)

    def test_adjusted_matches_r(self, panel, gold):
        """SDID on R's own ``y.adj``, isolating the solve from the adjustment."""
        got = self._att(panel, "y.adj")
        assert got == pytest.approx(
            float(gold["tau_adjusted_converged"]), abs=1e-6)

    def test_covariates_through_the_public_api(self, panel, gold):
        """The whole point: ``covariates=`` reproduces the adjusted estimate."""
        got = self._att(panel, "y", covariates={"adjust": ["x"]})
        assert got == pytest.approx(
            float(gold["tau_adjusted_converged"]), abs=1e-6)

    def test_the_public_api_matches_adjusting_by_hand(self, panel):
        """``covariates=`` is exactly SDID on the adjusted outcome.

        Independent of any gold: whatever the reference says, passing the
        covariate must equal pre-adjusting the column yourself. This is what
        would catch the adjustment being applied at the wrong point in the
        pipeline -- after ``dataprep``, or only to the donors.
        """
        assert self._att(panel, "y", covariates={"adjust": ["x"]}) == pytest.approx(
            self._att(panel, "y.adj"), abs=1e-9)


class TestSynthdidsEarlyStop:
    """Why the endpoint is checked against converged R and not the README.

    The README prints 39.760 and 49.398, and those are ``synthdid``'s numbers at
    its default ``min.decrease``. Run to ``min.decrease = 1e-12`` the same code
    gives 39.7605113 and 49.3922910, and mlsynth -- which solves the weight
    programs exactly rather than by early-stopped projected gradient -- agrees
    with the converged values to 1e-7.

    The gap is entirely on the adjusted outcome, and the reason is visible in
    the weights: adjusting away ``x`` leaves a residual so small relative to the
    ridge (``zeta.omega = 218.6``) that the penalty dominates, ``omega`` sits
    within 4e-6 of the uniform simplex point, and the objective there is flat
    enough that stopping early costs 0.0052 in the estimate for 6.4e-5 in the
    objective. mlsynth's solution attains the strictly lower objective of the
    two, so this is R stopping short, not mlsynth drifting.

    Pinned as behaviour because the natural response to "we are 0.005 from the
    published number" is to widen the tolerance, which would silently accept a
    real regression of the same size later.
    """

    def test_the_two_stopping_rules_disagree_on_the_adjusted_outcome(self, gold):
        published = float(gold["tau_adjusted"])
        converged = float(gold["tau_adjusted_converged"])
        assert abs(published - converged) == pytest.approx(0.0052, abs=5e-4)

    def test_they_agree_on_the_unadjusted_outcome(self, gold):
        """The contrast that localises it to the ridge-dominated case."""
        assert abs(float(gold["tau_unadjusted"])
                   - float(gold["tau_unadjusted_converged"])) < 1e-4

    def test_both_are_close_enough_to_the_truth_that_neither_is_broken(self, gold):
        """Neither number is wrong in any way that matters for the method."""
        for key in ("tau_adjusted", "tau_adjusted_converged"):
            assert abs(float(gold[key]) - float(gold["true_tau"])) < 1.0


class TestTheDefaultIsUnchanged:
    """Adding the option must not move any existing result."""

    def test_omitting_covariates_is_bit_identical(self, panel):
        import warnings

        from mlsynth import SDID

        def att(**extra):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(SDID({
                    "df": panel, "outcome": "y", "treat": "treat_exp",
                    "unitid": "i", "time": "t", "vce": "noinference",
                    "display_graphs": False, **extra}).fit().effects.att)

        assert att() == att(covariates=None)


class TestTheConfigRejectsBadInputEarly:
    """Caught at construction, before any fitting happens."""

    def _cfg(self, panel, **extra):
        from mlsynth import SDID

        return SDID({
            "df": panel, "outcome": "y", "treat": "treat_exp", "unitid": "i",
            "time": "t", "vce": "noinference", "display_graphs": False, **extra})

    def test_unknown_column_is_named_in_the_error(self, panel):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="ghost"):
            self._cfg(panel, covariates={"adjust": ["ghost"]})

    def test_non_numeric_covariate_is_rejected(self, panel):
        """A string column would silently become dummies rather than a slope."""
        from mlsynth.exceptions import MlsynthConfigError

        d = panel.copy()
        d["label"] = "a"
        with pytest.raises(MlsynthConfigError, match="numeric"):
            self._cfg(d, covariates={"adjust": ["label"]})

    def test_empty_list_is_rejected(self, panel):
        """``[]`` most likely means a caller built the list wrong; ``None`` is
        how you ask for no adjustment, and it stays available."""
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="empty"):
            self._cfg(panel, covariates={"adjust": []})

    def test_the_outcome_cannot_be_its_own_covariate(self, panel):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="outcome"):
            self._cfg(panel, covariates={"adjust": ["y"]})

    def test_the_treatment_cannot_be_its_own_covariate(self, panel):
        """Adjusting for the treatment indicator would subtract the effect."""
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="treatment"):
            self._cfg(panel, covariates={"adjust": ["treat_exp"]})

    def test_covariates_and_subgroup_are_refused_together(self, panel):
        """SC-DDD collapses the panel over the subgroup dimension, so there is no
        single (unit, time) row for the adjustment to be defined on. Refused
        rather than silently applied to one of the two panels."""
        from mlsynth.exceptions import MlsynthConfigError

        d = panel.copy()
        d["band"] = np.where(d.index % 2 == 0, "young", "old")
        with pytest.raises(MlsynthConfigError, match="subgroup"):
            self._cfg(d, covariates={"adjust": ["x"]}, subgroup="band",
                      target_subgroup="young")


class TestFailuresAreReported:
    """Invalid input raises a translated error rather than a numpy one."""

    def test_unknown_covariate_column(self, panel):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import adjust_outcome_for_covariates

        with pytest.raises(MlsynthDataError, match="ghost"):
            adjust_outcome_for_covariates(
                panel, unit="i", time="t", outcome="y", treat="treat_exp",
                covariates=["ghost"])

    def test_all_rows_treated_leaves_nothing_to_fit_on(self, panel):
        from mlsynth.exceptions import MlsynthEstimationError
        from mlsynth.utils.sdid_helpers.covariates import adjust_outcome_for_covariates

        d = panel.copy()
        d["treat_exp"] = 1
        with pytest.raises(MlsynthEstimationError, match="untreated"):
            adjust_outcome_for_covariates(
                d, unit="i", time="t", outcome="y", treat="treat_exp",
                covariates=["x"])

    @pytest.mark.parametrize("absent", ["unit", "time", "outcome", "treat"])
    def test_each_structural_column_is_checked(self, panel, absent):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import adjust_outcome_for_covariates

        kw = dict(unit="i", time="t", outcome="y", treat="treat_exp",
                  covariates=["x"])
        kw[absent] = "ghost"
        with pytest.raises(MlsynthDataError, match="ghost"):
            adjust_outcome_for_covariates(panel, **kw)

    def test_nan_in_the_covariate_is_reported_not_propagated(self, panel):
        """Left alone, lstsq would return NaN coefficients and the adjusted
        outcome would be silently all-NaN."""
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import adjust_outcome_for_covariates

        d = panel.copy()
        d.loc[0, "x"] = np.nan
        with pytest.raises(MlsynthDataError, match="NaN"):
            adjust_outcome_for_covariates(
                d, unit="i", time="t", outcome="y", treat="treat_exp",
                covariates=["x"])

    def test_bad_rows_mask_shape(self, panel):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import adjust_outcome_for_covariates

        with pytest.raises(MlsynthDataError, match="shape"):
            adjust_outcome_for_covariates(
                panel, unit="i", time="t", outcome="y", treat="treat_exp",
                covariates=["x"], rows=np.ones(5, dtype=bool))

    def test_unknown_rows_column(self, panel):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import adjust_outcome_for_covariates

        with pytest.raises(MlsynthDataError, match="ghost"):
            adjust_outcome_for_covariates(
                panel, unit="i", time="t", outcome="y", treat="treat_exp",
                covariates=["x"], rows="ghost")


class TestTheRegressionOnItsOwn:
    """``twfe_covariate_beta`` at its edges, away from the adjustment."""

    def test_missing_column_is_named(self, panel):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import twfe_covariate_beta

        with pytest.raises(MlsynthDataError, match="ghost"):
            twfe_covariate_beta(panel, outcome="y", covariates=["ghost"],
                                unit="i", time="t")

    def test_no_rows(self, panel):
        from mlsynth.exceptions import MlsynthEstimationError
        from mlsynth.utils.sdid_helpers.covariates import twfe_covariate_beta

        with pytest.raises(MlsynthEstimationError, match="no rows"):
            twfe_covariate_beta(panel.iloc[:0], outcome="y", covariates=["x"],
                                unit="i", time="t")

    def test_fewer_rows_than_covariates(self, panel):
        from mlsynth.exceptions import MlsynthEstimationError
        from mlsynth.utils.sdid_helpers.covariates import twfe_covariate_beta

        with pytest.raises(MlsynthEstimationError, match="identify"):
            twfe_covariate_beta(panel.iloc[:1], outcome="y", covariates=["x"],
                                unit="i", time="t")

    def test_non_finite_outcome(self, panel):
        from mlsynth.exceptions import MlsynthDataError
        from mlsynth.utils.sdid_helpers.covariates import twfe_covariate_beta

        d = panel.copy()
        d.loc[0, "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="NaN"):
            twfe_covariate_beta(d, outcome="y", covariates=["x"], unit="i",
                                time="t")


class TestTheEstimationRowsCanBeChosen:
    """``rows=`` overrides which observations ``beta`` is fit on."""

    def test_an_explicit_mask_reproduces_the_default(self, panel):
        """The default is spelled out as a mask, so the two must coincide.

        Pins what "untreated" means: every row with ``treat == 0``, which
        includes control units during the treatment period -- not just the
        pre-period, which is the other plausible reading.
        """
        from mlsynth.utils.sdid_helpers.covariates import adjust_outcome_for_covariates

        kw = dict(unit="i", time="t", outcome="y", treat="treat_exp",
                  covariates=["x"])
        default = adjust_outcome_for_covariates(panel, **kw)
        explicit = adjust_outcome_for_covariates(
            panel, rows=(panel.treat_exp == 0).to_numpy(), **kw)
        np.testing.assert_allclose(default, explicit, rtol=0, atol=1e-12)

    def test_a_column_name_selects_the_same_rows_as_its_values(self, panel):
        from mlsynth.utils.sdid_helpers.covariates import adjust_outcome_for_covariates

        d = panel.copy()
        d["fit_here"] = (d.treat_exp == 0)
        kw = dict(unit="i", time="t", outcome="y", treat="treat_exp",
                  covariates=["x"])
        np.testing.assert_allclose(
            adjust_outcome_for_covariates(d, rows="fit_here", **kw),
            adjust_outcome_for_covariates(d, rows=d.fit_here.to_numpy(), **kw),
            rtol=0, atol=1e-12)

    def test_restricting_to_the_pre_period_changes_beta(self, panel):
        """Confirms ``rows`` is actually used rather than ignored."""
        from mlsynth.utils.sdid_helpers.covariates import adjust_outcome_for_covariates

        kw = dict(unit="i", time="t", outcome="y", treat="treat_exp",
                  covariates=["x"])
        pre_only = adjust_outcome_for_covariates(
            panel, rows=(panel.t <= 15).to_numpy(), **kw)
        assert np.abs(pre_only - adjust_outcome_for_covariates(
            panel, **kw)).max() > 1e-6
