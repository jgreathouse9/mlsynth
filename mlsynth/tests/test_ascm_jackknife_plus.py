"""jackknife+ for ridge ASCM, against augsynth's own intermediates.

``augsynth``'s ``inf_type="jackknife+"`` is jackknife-plus over *pre-treatment
periods*, not over units: each pre-period is held out in turn, the ASCM is
refitted without it, and the interval is built from how badly the refit predicts
the period it could not see. It is neither the Barber et al. jackknife-plus over
observations nor the delete-one-unit jackknife ``PPSCM`` already implements.

Tests are ordered as ``agents_tests.md`` prescribes for a differential port: the
per-drop seam first, the assembled bounds only after. A failure in the first
class localises the fault to one refit; a failure only in the second means the
refits agree and the assembly does not. Asserting the endpoint alone would tell
us *that* the port disagrees and never *where*.

Gold in ``benchmarks/reference/ascm_jackknife_plus/`` comes from a live run of
augsynth 0.2.0 (pinned commit) on its bundled Kansas panel -- the same panel
where ``benchmarks/cases/ascm_kansas.py`` already reproduces the ridge ASCM point
estimate exactly, so the fit is not a confound and any disagreement here is the
inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthEstimationError

_GOLD = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
         / "ascm_jackknife_plus")
_DATA = Path(__file__).resolve().parents[2] / "basedata" / "kansas_ascm.csv"

#: augsynth's published ridge ASCM estimate for Kansas. Asserted here so that if
#: the panel prep below ever drifts from the benchmark's, this file fails loudly
#: rather than silently comparing gold against a different fit.
KANSAS_RIDGE_ATT = -0.04006291
TREATED_FIPS = 20
T_INT = 2012.25


def _kansas():
    """``(y_pre, Y0_pre, y_post, Y0_post)`` as augsynth sees the Kansas panel."""
    d = pd.read_csv(_DATA)
    piv = d.pivot(index="fips", columns="year_qtr",
                  values="lngdpcapita").sort_index()
    times = np.array(sorted(d["year_qtr"].unique()))
    n_pre = int((times < T_INT).sum())
    units = piv.index.to_numpy()
    Y = piv.to_numpy()
    trt = units == TREATED_FIPS
    y, Y0 = Y[trt][0], Y[~trt]
    return (y[:n_pre], Y0[:, :n_pre].T, y[n_pre:], Y0[:, n_pre:])


def _skip_without(path):
    if not path.exists():  # pragma: no cover - the gold is committed
        pytest.skip(f"missing gold at {path}")


@pytest.fixture(scope="module")
def kansas():
    _skip_without(_DATA)
    return _kansas()


@pytest.fixture(scope="module")
def perdrop_gold():
    p = _GOLD / "gold_perdrop.csv"
    _skip_without(p)
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def fitted(kansas):
    """mlsynth's jackknife+ run once, reused across assertions (89 refits)."""
    from mlsynth.utils.bilevel.jackknife_plus import jackknife_plus
    y_pre, Y0_pre, y_post, Y0_post = kansas
    return jackknife_plus(y_pre, Y0_pre, y_post, Y0_post, alpha=0.05)


class TestPanelPrep:
    """Guard the inputs before comparing anything computed from them."""

    def test_shapes(self, kansas):
        y_pre, Y0_pre, y_post, Y0_post = kansas
        assert y_pre.shape == (89,)
        assert Y0_pre.shape == (89, 49)
        assert y_post.shape == (16,)
        assert Y0_post.shape == (49, 16)

    def test_prep_reproduces_augsynths_point_estimate(self, kansas):
        """If this fails, the panel is wrong and every other test is vacuous."""
        from mlsynth.utils.bilevel.ridge_augment import ridge_augment_weights
        y_pre, Y0_pre, y_post, Y0_post = kansas
        w = ridge_augment_weights(y_pre, Y0_pre).W
        att = float(np.mean(y_post - Y0_post.T @ w))
        assert att == pytest.approx(KANSAS_RIDGE_ATT, abs=1e-6)


class TestPerDropSeam:
    """The intermediates: one refit per held-out pre-period."""

    def test_one_entry_per_pre_period(self, fitted, perdrop_gold):
        assert fitted.held_out_errors.shape == (89,)
        assert perdrop_gold.tdrop.nunique() == 89

    def test_held_out_errors_match_augsynth(self, fitted, perdrop_gold):
        """The error at the period the refit could not see.

        This is the quantity the whole interval is built from, so it is the
        first place a port can go wrong and the cheapest place to catch it.

        Tolerance is ``1e-6``, not machine epsilon, and the reason is
        structural rather than sloppy: the base simplex fit underneath each
        refit is an equality- and inequality-constrained QP, solved here by
        mlsynth's own exact solver and in the reference by ``quadprog``/osqp.
        Two exact solvers agree on the optimum to their own tolerances, not
        bit-for-bit, so ~1e-7 is the floor for anything downstream of the QP.
        Observed worst case here is 3.5e-8 with a median of 7.5e-10.
        """
        expected = (perdrop_gold.sort_values(["tdrop", "pos"])
                    .groupby("tdrop").err.first().to_numpy())
        np.testing.assert_allclose(fitted.held_out_errors, expected,
                                   rtol=0, atol=1e-6)

    def test_per_drop_predictions_match_augsynth(self, fitted, perdrop_gold):
        """Each refit's counterfactual over the post periods, plus its mean.

        ``pos`` runs 1..17: sixteen post periods then the appended average.
        """
        exp = (perdrop_gold.sort_values(["tdrop", "pos"])
               .pivot(index="tdrop", columns="pos", values="est").to_numpy())
        assert exp.shape == (89, 17)
        # 1e-6 for the QP-solver reason given above; worst case here is 1.7e-7.
        np.testing.assert_allclose(fitted.per_drop_estimates, exp,
                                   rtol=0, atol=1e-6)

    def test_the_appended_column_is_the_mean_of_the_others(self, fitted):
        """Invariant, independent of the gold."""
        est = fitted.per_drop_estimates
        np.testing.assert_allclose(est[:, -1], est[:, :-1].mean(axis=1),
                                   rtol=0, atol=1e-12)

    def test_refits_reuse_the_full_fit_penalty(self, kansas):
        """augsynth pins lambda into ``extra_args`` and refits at it.

        ``augsynth.R`` sets ``extra_args$lambda <- augsynth$lambda`` for the
        ridge path, so no refit re-runs cross-validation. Re-selecting per drop
        would give 89 different penalties and a different interval, so this
        pins the choice rather than leaving it to the default.
        """
        from mlsynth.utils.bilevel.jackknife_plus import jackknife_plus
        from mlsynth.utils.bilevel.ridge_augment import ridge_augment_weights
        y_pre, Y0_pre, y_post, Y0_post = kansas
        full_lambda = ridge_augment_weights(y_pre, Y0_pre).lambda_
        got = jackknife_plus(y_pre, Y0_pre, y_post, Y0_post)
        assert got.lambda_ == pytest.approx(full_lambda, rel=1e-12)


class TestAssembledBounds:
    """The bounds, once the per-drop seam is known to agree."""

    @staticmethod
    def _gold(conservative):
        p = _GOLD / (f"gold_jackknife_plus_conservative_"
                     f"{str(conservative).lower()}.csv")
        _skip_without(p)
        return pd.read_csv(p)

    @pytest.mark.parametrize("conservative", [False, True])
    def test_bounds_match_augsynth(self, kansas, conservative):
        """Both settings, because they assemble from different rows.

        The default takes quantiles of ``est +/- |err|``; the conservative
        branch takes the min and max of ``est`` widened by a quantile of
        ``|err|``. A port validated under one is validated under neither.
        """
        from mlsynth.utils.bilevel.jackknife_plus import jackknife_plus
        y_pre, Y0_pre, y_post, Y0_post = kansas
        g = self._gold(conservative)
        got = jackknife_plus(y_pre, Y0_pre, y_post, Y0_post, alpha=0.05,
                             conservative=conservative)
        post = g[g.lb.notna()]
        # The two branches earn different tolerances, and the gap is itself a
        # finding. Taking quantiles across 89 drops averages the QP-solver
        # difference away, so the default branch agrees to ~4e-9. The
        # conservative branch instead takes the min and max over drops, which
        # deliberately selects the most extreme refit -- precisely where the
        # solver difference is largest -- so it lands at ~1.8e-7. Looser is
        # correct here, not lax: an interval built from order statistics cannot
        # be more reproducible than its worst-case term.
        atol = 1e-6 if conservative else 1e-7
        np.testing.assert_allclose(got.lower, post.lb.to_numpy(),
                                   rtol=0, atol=atol)
        np.testing.assert_allclose(got.upper, post.ub.to_numpy(),
                                   rtol=0, atol=atol)

    def test_the_two_settings_actually_differ(self, kansas):
        """Otherwise the parametrised test above proves half of what it claims."""
        from mlsynth.utils.bilevel.jackknife_plus import jackknife_plus
        y_pre, Y0_pre, y_post, Y0_post = kansas
        a = jackknife_plus(y_pre, Y0_pre, y_post, Y0_post, conservative=False)
        b = jackknife_plus(y_pre, Y0_pre, y_post, Y0_post, conservative=True)
        assert np.abs(a.lower - b.lower).max() > 1e-6

    def test_conservative_is_not_reliably_wider(self, kansas):
        """The argument's name misleads, and the port must not "fix" that.

        ``conservative`` widens by the ``1 - alpha`` quantile of the absolute
        held-out errors, where the default reaches the ``1 - alpha/2`` quantile
        of ``est + |err|``. When the errors are right-skewed the default's
        deeper reach into that tail can dominate the wider min/max envelope, and
        the "conservative" interval comes out tighter -- as it does here, at 16
        of 17 periods. Pinned so nobody later "corrects" the branch to enforce
        an ordering augsynth does not have.
        """
        from mlsynth.utils.bilevel.jackknife_plus import jackknife_plus
        y_pre, Y0_pre, y_post, Y0_post = kansas
        a = jackknife_plus(y_pre, Y0_pre, y_post, Y0_post, conservative=False)
        b = jackknife_plus(y_pre, Y0_pre, y_post, Y0_post, conservative=True)
        narrower = ((b.upper - b.lower) < (a.upper - a.lower)).sum()
        assert narrower == 16

    def test_conservative_width_is_the_envelope_plus_two_error_quantiles(
            self, kansas):
        """The exact construction, which is what is actually invariant."""
        from mlsynth.utils.bilevel.jackknife_plus import jackknife_plus
        y_pre, Y0_pre, y_post, Y0_post = kansas
        b = jackknife_plus(y_pre, Y0_pre, y_post, Y0_post, conservative=True,
                           alpha=0.05)
        est = b.per_drop_estimates
        q = np.quantile(np.abs(b.held_out_errors), 0.95, method="linear")
        expected = (est.max(axis=0) - est.min(axis=0)) + 2.0 * q
        np.testing.assert_allclose(b.upper - b.lower, expected,
                                   rtol=0, atol=1e-12)

    def test_lengths_cover_post_periods_plus_the_average(self, fitted):
        assert fitted.lower.shape == (17,)
        assert fitted.upper.shape == (17,)

    def test_bounds_bracket_the_estimate(self, fitted):
        assert np.all(fitted.lower <= fitted.upper)

    def test_interval_is_reflected_through_the_observed_path(self, kansas):
        """``lb = y1 - ub_cf`` and ``ub = y1 - lb_cf`` -- the flip.

        Reflecting a counterfactual interval onto the ATT scale reverses the
        bounds. Getting this wrong yields an interval of the right width about
        the wrong centre, which end-to-end agreement on width would not catch.
        """
        from mlsynth.utils.bilevel.jackknife_plus import jackknife_plus
        y_pre, Y0_pre, y_post, Y0_post = kansas
        r = jackknife_plus(y_pre, Y0_pre, y_post, Y0_post)
        y1 = np.append(y_post, y_post.mean())
        np.testing.assert_allclose(r.lower, y1 - r.counterfactual_upper,
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(r.upper, y1 - r.counterfactual_lower,
                                   rtol=0, atol=1e-12)


class TestQuantileConvention:
    """R's ``quantile`` default is type 7; numpy's linear method matches it.

    Pinned because a type mismatch shifts every bound by a fraction of an
    order statistic -- small, systematic, and invisible without a reference.
    """

    def test_matches_r_type7_on_a_known_vector(self):
        from mlsynth.utils.bilevel.jackknife_plus import _quantile
        x = np.array([1.0, 2.0, 3.0, 4.0])
        # R: quantile(c(1,2,3,4), c(0.025, 0.5, 0.975)) -> 1.075 2.5 3.925
        np.testing.assert_allclose(
            [_quantile(x, 0.025), _quantile(x, 0.5), _quantile(x, 0.975)],
            [1.075, 2.5, 3.925], rtol=0, atol=1e-12)


class TestFailures:
    """Invalid input is refused with a translated error, not a bare numpy one."""

    def test_alpha_outside_the_unit_interval_is_refused(self, kansas):
        from mlsynth.utils.bilevel.jackknife_plus import jackknife_plus
        y_pre, Y0_pre, y_post, Y0_post = kansas
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(MlsynthEstimationError, match="alpha"):
                jackknife_plus(y_pre, Y0_pre, y_post, Y0_post, alpha=bad)

    def test_too_few_pre_periods_is_refused_with_a_reason(self):
        """Dropping a period must leave enough to refit on."""
        from mlsynth.utils.bilevel.jackknife_plus import jackknife_plus
        rng = np.random.default_rng(0)
        with pytest.raises(MlsynthEstimationError, match="pre-treatment"):
            jackknife_plus(rng.normal(size=2), rng.normal(size=(2, 3)),
                           rng.normal(size=4), rng.normal(size=(3, 4)))

    def test_mismatched_donor_counts_are_refused(self):
        from mlsynth.utils.bilevel.jackknife_plus import jackknife_plus
        rng = np.random.default_rng(1)
        with pytest.raises(MlsynthEstimationError, match="donor"):
            jackknife_plus(rng.normal(size=10), rng.normal(size=(10, 5)),
                           rng.normal(size=4), rng.normal(size=(3, 4)))

    def test_a_transposed_pre_period_matrix_is_refused(self):
        """``Y0_pre`` is (T0, J); handed (J, T0) the maths would run silently on
        the wrong axis for a square-ish panel, so the shape is checked."""
        from mlsynth.utils.bilevel.jackknife_plus import jackknife_plus
        rng = np.random.default_rng(3)
        with pytest.raises(MlsynthEstimationError, match="Y0_pre"):
            jackknife_plus(rng.normal(size=10), rng.normal(size=(4, 10)),
                           rng.normal(size=5), rng.normal(size=(4, 5)))

    def test_post_period_length_mismatch_is_refused(self):
        """Treated and donor post-periods must describe the same periods."""
        from mlsynth.utils.bilevel.jackknife_plus import jackknife_plus
        rng = np.random.default_rng(4)
        with pytest.raises(MlsynthEstimationError, match="period"):
            jackknife_plus(rng.normal(size=10), rng.normal(size=(10, 4)),
                           rng.normal(size=5), rng.normal(size=(4, 3)))

    def test_no_post_periods_is_refused(self):
        from mlsynth.utils.bilevel.jackknife_plus import jackknife_plus
        rng = np.random.default_rng(2)
        with pytest.raises(MlsynthEstimationError, match="post-treatment"):
            jackknife_plus(rng.normal(size=10), rng.normal(size=(10, 4)),
                           np.array([]), np.zeros((4, 0)))


class TestVanillaSCSurface:
    """Reachable, and correct, through the public estimator.

    The tests above drive the helper directly. These check the wiring: that the
    option is accepted under both spellings, that it lands in the standardized
    inference slot, and that it refuses the configuration it cannot serve.
    """

    @staticmethod
    def _panel():
        d = pd.read_csv(_DATA)
        d = d[["fips", "year_qtr", "lngdpcapita"]].copy()
        d["treated"] = ((d.fips == TREATED_FIPS) & (d.year_qtr >= T_INT)).astype(int)
        return d

    @staticmethod
    def _fit(**extra):
        from mlsynth import VanillaSC
        return VanillaSC({
            "df": TestVanillaSCSurface._panel(), "outcome": "lngdpcapita",
            "treat": "treated", "unitid": "fips", "time": "year_qtr",
            "display_graphs": False, "augment": "ridge", **extra}).fit()

    def test_average_bound_matches_augsynth(self):
        """The scalar slots carry the bound on the post-period average.

        augsynth reports that as its ``average_att`` row, and it is the number a
        user quotes -- not the mean of the per-period bounds, which is a
        different quantity.
        """
        g = pd.read_csv(_GOLD / "gold_jackknife_plus_conservative_false.csv")
        avg = g.iloc[-1]                      # the appended average row
        res = self._fit(inference="jackknife_plus")
        assert res.inference is not None
        assert res.inference.ci_lower == pytest.approx(avg.lb, abs=1e-6)
        assert res.inference.ci_upper == pytest.approx(avg.ub, abs=1e-6)

    def test_per_period_band_has_one_entry_per_post_period(self):
        res = self._fit(inference="jackknife_plus")
        det = res.inference.details
        assert len(det["pi_lower"]) == 16
        assert len(det["periods"]) == 16

    @pytest.mark.parametrize("spelling", ["jackknife_plus", "jackknife+",
                                          "jackknife-plus", "JACKKNIFE+"])
    def test_augsynths_own_spelling_is_accepted(self, spelling):
        """Somebody porting an R script should not have to guess our rename."""
        res = self._fit(inference=spelling)
        assert res.inference is not None
        assert "jackknife+" in res.inference.method

    def test_the_conservative_switch_reaches_the_helper(self):
        a = self._fit(inference="jackknife_plus")
        b = self._fit(inference="jackknife_plus", jackknife_conservative=True)
        assert b.inference.details["conservative"] is True
        assert abs(a.inference.ci_lower - b.inference.ci_lower) > 1e-6

    def test_it_refuses_an_unaugmented_fit_with_a_reason(self):
        """Without augmentation there is nothing to de-bias, so the refit is
        just the base SCM again and the procedure is not what augsynth runs."""
        from mlsynth import VanillaSC
        with pytest.raises(MlsynthEstimationError, match="augment='ridge'"):
            VanillaSC({
                "df": self._panel(), "outcome": "lngdpcapita", "treat": "treated",
                "unitid": "fips", "time": "year_qtr", "display_graphs": False,
                "inference": "jackknife_plus"}).fit()

    def test_an_unknown_inference_name_is_still_refused(self):
        """The alias table must not have opened a hole for typos."""
        from mlsynth import VanillaSC
        from mlsynth.exceptions import MlsynthConfigError
        from pydantic import ValidationError
        with pytest.raises((ValidationError, MlsynthConfigError)):
            VanillaSC({
                "df": self._panel(), "outcome": "lngdpcapita", "treat": "treated",
                "unitid": "fips", "time": "year_qtr", "display_graphs": False,
                "augment": "ridge", "inference": "jackknife_pluss"}).fit()
