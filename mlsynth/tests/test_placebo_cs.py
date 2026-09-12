"""Firpo-Possebom confidence sets: the inversion, and what it refuses.

Firpo, S. & Possebom, V. (2018), "Synthetic Control Method: Inference,
Sensitivity Analysis and Confidence Sets", *Journal of Causal Inference* 6(2),
20160026.

Two tests pin the inversion against the authors' own ``SCM.CS``.
:class:`TestAgainstTheAuthorsDriver` is the decisive one: it runs on the weights
the authors' own California script produces, from R ``Synth`` under their
predictor specification, so the whole construction is theirs and only the
inversion is ours. :class:`TestAgainstTheAuthorsR` repeats the comparison on
outcome-only simplex weights, which anyone can regenerate without R. Both runs,
their provenance and the reasoning behind sharing the weights are staged in
``benchmarks/reference/fp_confidence_sets/``; the gold files are read here
instead of transcribed, so the constants and the captured runs cannot drift
apart.

The rest of the file pins the pieces that gold cannot see on its own: that the
null is imposed across the whole panel and not only on the treated unit, that
the uniform case reproduces the ordinary placebo p-value mlsynth already
computes, and that the two ways the search can fail are reported instead of
returning a number.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.vanillasc_helpers.placebo_cs import (
    PlaceboConfidenceSet, confidence_set, effect_path, placebo_pvalue,
    sensitivity_sweep,
)

_REF = (pathlib.Path(__file__).resolve().parents[2]
        / "benchmarks" / "reference" / "fp_confidence_sets")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def prop99():
    """The staged California inputs: outcomes and placebo weights."""
    if not (_REF / "Ymat.csv").exists():  # pragma: no cover - bundle is committed
        pytest.skip("reference bundle not present")
    Y = np.loadtxt(_REF / "Ymat.csv", delimiter=",")
    W = np.loadtxt(_REF / "weightsmat.csv", delimiter=",")
    return Y, W, 2, 19          # 0-based California, T0


@pytest.fixture(scope="module")
def gold():
    import csv
    if not (_REF / "gold_bounds.csv").exists():  # pragma: no cover
        pytest.skip("reference bundle not present")
    with (_REF / "gold_bounds.csv").open() as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def prop99_authors():
    """California as the authors' own driver builds it, via R ``Synth``."""
    if not (_REF / "Ymat_authors.csv").exists():  # pragma: no cover - committed
        pytest.skip("reference bundle not present")
    Y = np.loadtxt(_REF / "Ymat_authors.csv", delimiter=",", skiprows=1)
    W = np.loadtxt(_REF / "weightsmat_authors.csv", delimiter=",", skiprows=1)
    return Y, W, 2, 19


@pytest.fixture(scope="module")
def gold_authors():
    import csv
    if not (_REF / "gold_bounds_authors.csv").exists():  # pragma: no cover
        pytest.skip("reference bundle not present")
    with (_REF / "gold_bounds_authors.csv").open() as fh:
        return list(csv.DictReader(fh))


def _toy(n_periods=10, n_units=4, pre=6, seed=0):
    """A small panel and a simplex weight matrix, for the unit-level checks."""
    rng = np.random.default_rng(seed)
    Y = rng.normal(100.0, 5.0, size=(n_periods, n_units))
    W = rng.random((n_units - 1, n_units))
    W /= W.sum(axis=0, keepdims=True)
    return Y, W, pre


# =========================================================================== #
# the effect classes
# =========================================================================== #
class TestEffectPath:
    """The one-parameter families the set is inverted over."""

    def test_constant_is_flat_after_treatment_and_zero_before(self):
        p = effect_path(2.5, n_periods=6, pre_periods=4, kind="constant")
        assert np.array_equal(p, [0, 0, 0, 0, 2.5, 2.5])

    def test_linear_grows_one_step_per_post_period(self):
        p = effect_path(2.0, n_periods=7, pre_periods=4, kind="linear")
        assert np.array_equal(p, [0, 0, 0, 0, 2.0, 4.0, 6.0])

    def test_zero_parameter_is_the_no_effect_null(self):
        for kind in ("constant", "linear"):
            assert not effect_path(0.0, 6, 4, kind).any()

    def test_unknown_class_is_refused(self):
        with pytest.raises(MlsynthEstimationError, match="constant.*linear"):
            effect_path(1.0, 6, 4, "quadratic")


# =========================================================================== #
# the p-value
# =========================================================================== #
class TestPlaceboPvalue:
    """What the rank statistic is computed over."""

    def test_zero_null_reproduces_the_ordinary_placebo_pvalue(self, prop99):
        """At a zero effect this is Abadie's test, which mlsynth already has.

        The ordinary in-space placebo p-value is the treated unit's rank among
        post/pre RMSPE ratios. Inverting at the zero null must agree with it, or
        the two inference modes on the same estimator disagree about the same
        hypothesis.
        """
        Y, W, t0, pre = prop99
        n_periods, n_units = Y.shape
        p = placebo_pvalue(Y, W, t0, pre, np.zeros(n_periods))

        ratios = np.empty(n_units)
        for j in range(n_units):
            donors = [k for k in range(n_units) if k != j]
            gaps = Y[:, j] - Y[:, donors] @ W[:, j]
            post = float(gaps[pre:] @ gaps[pre:]) / (n_periods - pre)
            pre_ = float(gaps[:pre] @ gaps[:pre]) / pre
            ratios[j] = post / pre_
        assert p == pytest.approx(float(np.mean(ratios >= ratios[t0])))

    def test_the_null_is_imposed_on_the_whole_panel(self, prop99):
        """A placebo unit's donor pool contains the treated unit, and moves.

        Under a non-zero null the treated unit's observed series is not its
        untreated one, so leaving it unadjusted inside every placebo's donor
        pool tests a different hypothesis. The correction shifts unit ``j``'s
        fitted counterfactual by exactly ``path * w`` where ``w`` is the weight
        ``j``'s synthetic control puts on the treated unit -- so it bites on
        precisely the units that borrow from the treated one, and the size of
        the shift is pinned, not merely non-zero.
        """
        Y, W, t0, pre = prop99
        n_periods, n_units = Y.shape
        path = effect_path(-3.0, n_periods, pre, "linear")

        borrowed = 0
        for j in range(n_units):
            if j == t0:
                continue
            donors = [k for k in range(n_units) if k != j]
            position = t0 - 1 if j < t0 else t0
            y1 = Y[:, j] + path
            naive_gap = y1 - Y[:, donors] @ W[:, j] - path
            corrected = Y[:, donors].copy()
            corrected[:, position] -= path
            full_gap = y1 - corrected @ W[:, j] - path
            # the whole of the correction, in closed form
            np.testing.assert_allclose(full_gap - naive_gap,
                                       path * W[position, j], atol=1e-12)
            borrowed += abs(W[position, j]) > 1e-6
        assert borrowed > 0, "fixture must contain a unit that borrows"

    def test_the_donor_pool_correction_need_not_move_the_pvalue(self, prop99):
        """Pinning the coarseness, so a future reader does not read it as a bug.

        The correction changes eight of the thirty-nine California statistics,
        by up to two orders of magnitude. The p-value is a rank comparison
        against the treated unit, so it only moves when one of those units
        crosses it -- which on this panel none does, at any candidate tried.
        The mechanism and the p-value are separate claims and only the first is
        asserted above.
        """
        Y, W, t0, pre = prop99
        for value in (-3.0, -1.0, 1.0, 5.0):
            path = effect_path(value, Y.shape[0], pre, "linear")
            assert placebo_pvalue(Y, W, t0, pre, path) == pytest.approx(
                placebo_pvalue(Y, W, t0, pre, path,
                               _impose_on_donor_pool=False))

    def test_uniform_weights_are_the_default(self, prop99):
        Y, W, t0, pre = prop99
        path = effect_path(-2.0, Y.shape[0], pre, "linear")
        assert placebo_pvalue(Y, W, t0, pre, path) == pytest.approx(
            placebo_pvalue(Y, W, t0, pre, path, phi=0.0, v=np.zeros(Y.shape[1])))

    def test_pvalue_is_a_probability(self, prop99):
        Y, W, t0, pre = prop99
        for value in (-10.0, -1.0, 0.0, 5.0):
            p = placebo_pvalue(Y, W, t0, pre,
                               effect_path(value, Y.shape[0], pre, "constant"))
            assert 0.0 < p <= 1.0

    def test_tilting_toward_the_treated_unit_raises_its_own_pvalue(self, prop99):
        """The sensitivity direction has the sign the mechanism claims.

        ``prob = softmax(phi * v)``: mass moved onto the treated unit can only
        add to a sum the treated unit always satisfies, so its p-value rises and
        rejection becomes harder.
        """
        Y, W, t0, pre = prop99
        v = np.zeros(Y.shape[1]); v[t0] = 1.0
        path = effect_path(-2.0, Y.shape[0], pre, "linear")
        base = placebo_pvalue(Y, W, t0, pre, path)
        tilted = placebo_pvalue(Y, W, t0, pre, path, phi=1.5, v=v)
        assert tilted > base


# =========================================================================== #
# the inversion, against the authors' R
# =========================================================================== #
class TestAgainstTheAuthorsDriver:
    """Cross-validation on the authors' own California construction.

    ``benchmarks/reference/fp_confidence_sets/reference_authors.R`` is Firpo and
    Possebom's ``california_beta_testing`` script: R ``Synth`` fits all 39
    placebo units under their predictor specification, and ``SCM.CS`` inverts
    the placebo test on the result. Everything here except the inversion is the
    authors' code, so a match leaves nothing of the procedure unpinned.
    """

    def test_every_captured_configuration_matches(self, prop99_authors,
                                                  gold_authors):
        Y, W, t0, pre = prop99_authors
        v_tr = np.zeros(Y.shape[1]); v_tr[t0] = 1.0
        checked = 0
        for row in gold_authors:
            phi = float(row["phi"])
            v = None if phi == 0.0 else v_tr
            if row["lower"] in ("NA", ""):
                with pytest.raises(MlsynthEstimationError):
                    confidence_set(Y, W, t0, pre, kind=row["kind"],
                                   alpha=4 / 39, precision=30, phi=phi, v=v)
                checked += 1
                continue
            cs = confidence_set(Y, W, t0, pre, kind=row["kind"], alpha=4 / 39,
                                precision=30, phi=phi, v=v)
            assert cs.lower == pytest.approx(float(row["lower"]), abs=1e-10)
            assert cs.upper == pytest.approx(float(row["upper"]), abs=1e-10)
            checked += 1
        assert checked == len(gold_authors) > 0

    def test_the_weights_are_a_simplex_for_every_placebo_unit(self,
                                                              prop99_authors):
        """Synth's output, as handed to the inversion: non-negative, sums to one.

        The tolerances are Synth's, not ours. Its ``BFGS`` solve returns weights
        up to about 8e-9 below zero and column sums up to about 2e-8 off one;
        the inversion takes them as given, so the test records the size of that
        slack instead of cleaning it up.
        """
        _, W, _, _ = prop99_authors
        assert W.shape == (38, 39)
        assert (W >= -1e-7).all()
        assert W.sum(axis=0) == pytest.approx(np.ones(39), abs=1e-6)

    def test_the_conclusion_survives_a_tilt_it_lost_on_outcome_only_weights(
            self, prop99_authors, gold_authors):
        """Zero stays outside the set at every tilt the search resolves.

        On the outcome-only weights of :class:`TestAgainstTheAuthorsR` the sign
        is lost at ``phi = 1``. Under the authors' specification it is not, so
        the sensitivity verdict depends on which weights the inversion is given.
        """
        resolved = [r for r in gold_authors
                    if r["kind"] == "linear" and r["lower"] not in ("NA", "")]
        assert len(resolved) == 3
        for row in resolved:
            assert float(row["upper"]) < 0.0


class TestAgainstTheAuthorsR:
    """Cross-validation on inputs both implementations were handed."""

    def test_every_captured_configuration_matches(self, prop99, gold):
        Y, W, t0, pre = prop99
        v_tr = np.zeros(Y.shape[1]); v_tr[t0] = 1.0
        checked = 0
        for row in gold:
            v = (None if row["vlab"] == "zero"
                 else v_tr if row["vlab"] == "treated" else 1.0 - v_tr)
            phi = float(row["phi"])
            if row["lower"] in ("NA", ""):
                with pytest.raises(MlsynthEstimationError):
                    confidence_set(Y, W, t0, pre, kind=row["type"],
                                   alpha=4 / 39, precision=30, phi=phi, v=v)
                checked += 1
                continue
            cs = confidence_set(Y, W, t0, pre, kind=row["type"], alpha=4 / 39,
                                precision=30, phi=phi, v=v)
            assert cs.lower == pytest.approx(float(row["lower"]), abs=1e-10)
            assert cs.upper == pytest.approx(float(row["upper"]), abs=1e-10)
            checked += 1
        assert checked == len(gold) > 0

    def test_result_is_the_frozen_container(self, prop99):
        Y, W, t0, pre = prop99
        cs = confidence_set(Y, W, t0, pre, kind="linear", alpha=4 / 39,
                            precision=12)
        assert isinstance(cs, PlaceboConfidenceSet)
        with pytest.raises(Exception):
            cs.lower = 0.0

    def test_the_set_brackets_the_point_estimate(self, prop99):
        """Inversion starts at the point estimate, which cannot be rejected."""
        Y, W, t0, pre = prop99
        cs = confidence_set(Y, W, t0, pre, kind="linear", alpha=4 / 39,
                            precision=12)
        assert cs.lower <= cs.point_estimate <= cs.upper

    def test_the_bounds_are_where_the_test_flips(self, prop99):
        """The set is ``{c : p(c) > alpha}``, so its edges must be crossings.

        This is what ``precision`` buys and the only thing that makes the
        reported interval the inverted test and not an interval near it: just
        inside each bound the null survives, just outside it is rejected.
        """
        Y, W, t0, pre = prop99
        alpha, precision = 4 / 39, 22
        cs = confidence_set(Y, W, t0, pre, kind="linear", alpha=alpha,
                            precision=precision)

        def p_at(value):
            path = effect_path(value, Y.shape[0], pre, "linear")
            return placebo_pvalue(Y, W, t0, pre, path)

        # the bisection resolves the bound to 2**-precision of the point
        # estimate, so step outside by a comfortable multiple of that
        eps = abs(cs.point_estimate) * 2.0 ** -(precision - 4)
        assert p_at(cs.upper - eps) > alpha
        assert p_at(cs.upper + eps) <= alpha
        assert p_at(cs.lower + eps) > alpha
        assert p_at(cs.lower - eps) <= alpha

    def test_a_coarser_search_gives_a_narrower_set(self, prop99):
        """The search walks out from the point estimate, so it converges from
        inside: an under-set ``precision`` reports a set that is too small, and
        therefore under-covers. On this panel the width rises monotonically
        from 3.607 at precision 4 to 3.718 at 30.
        """
        Y, W, t0, pre = prop99
        widths = []
        for precision in (4, 8, 12, 16, 22):
            cs = confidence_set(Y, W, t0, pre, kind="linear", alpha=4 / 39,
                                precision=precision)
            widths.append(cs.upper - cs.lower)
        assert widths == sorted(widths)
        assert widths[0] < widths[-1]

    def test_each_refinement_moves_less_than_its_own_step(self, prop99):
        """Successive halvings cannot move a bound by more than the step they
        take, which is what makes ``precision`` a resolution and not a knob."""
        Y, W, t0, pre = prop99
        previous = None
        for precision in range(8, 15):
            cs = confidence_set(Y, W, t0, pre, kind="linear", alpha=4 / 39,
                                precision=precision)
            if previous is not None:
                step = abs(cs.point_estimate) * 2.0 ** -(precision - 1)
                assert abs(cs.upper - previous.upper) <= step + 1e-12
                assert abs(cs.lower - previous.lower) <= step + 1e-12
            previous = cs

    def test_paths_agree_with_the_bounds(self, prop99):
        Y, W, t0, pre = prop99
        cs = confidence_set(Y, W, t0, pre, kind="constant", alpha=4 / 39,
                            precision=12)
        assert cs.lower_path[-1] == pytest.approx(cs.lower)
        assert cs.upper_path[-1] == pytest.approx(cs.upper)
        assert not cs.lower_path[:pre].any()


# =========================================================================== #
# derived scales: cumulative and average effect
# =========================================================================== #
class TestCumulativeAndAverage:
    """The same set, read on two scales the parameter is not on.

    Within a one-parameter family the cumulative post-treatment effect and the
    average per-period effect are strictly increasing functions of the
    parameter, so the confidence set maps over exactly: inverting a test and
    then reparametrising gives the same set as reparametrising and then
    inverting. Nothing is recomputed and no coverage is given up.
    """

    def test_the_constant_class_multiplies_by_the_post_period_count(self, prop99):
        Y, W, t0, pre = prop99
        n_post = Y.shape[0] - pre
        cs = confidence_set(Y, W, t0, pre, kind="constant", alpha=4 / 39,
                            precision=12)
        lo, hi = cs.cumulative
        assert lo == pytest.approx(cs.lower * n_post)
        assert hi == pytest.approx(cs.upper * n_post)

    def test_the_linear_class_multiplies_by_the_triangular_number(self, prop99):
        """A linear path sums to ``c * K(K+1)/2`` over its ``K`` post-periods."""
        Y, W, t0, pre = prop99
        n_post = Y.shape[0] - pre
        weight = n_post * (n_post + 1) / 2.0
        cs = confidence_set(Y, W, t0, pre, kind="linear", alpha=4 / 39,
                            precision=12)
        lo, hi = cs.cumulative
        assert lo == pytest.approx(cs.lower * weight)
        assert hi == pytest.approx(cs.upper * weight)

    def test_the_average_is_the_cumulative_over_the_post_periods(self, prop99):
        Y, W, t0, pre = prop99
        n_post = Y.shape[0] - pre
        for kind in ("constant", "linear"):
            cs = confidence_set(Y, W, t0, pre, kind=kind, alpha=4 / 39,
                                precision=12)
            clo, chi = cs.cumulative
            alo, ahi = cs.average
            assert alo == pytest.approx(clo / n_post)
            assert ahi == pytest.approx(chi / n_post)

    def test_the_bounds_stay_ordered_on_every_scale(self, prop99):
        """The maps are strictly increasing, so they cannot flip the interval."""
        Y, W, t0, pre = prop99
        for kind in ("constant", "linear"):
            cs = confidence_set(Y, W, t0, pre, kind=kind, alpha=4 / 39,
                                precision=12)
            assert cs.lower <= cs.upper
            assert cs.cumulative[0] <= cs.cumulative[1]
            assert cs.average[0] <= cs.average[1]

    def test_zero_is_inside_on_every_scale_or_on_none(self, prop99):
        """A scale change cannot turn a significant result insignificant."""
        Y, W, t0, pre = prop99
        for kind in ("constant", "linear"):
            cs = confidence_set(Y, W, t0, pre, kind=kind, alpha=4 / 39,
                                precision=12)
            inside = [cs.contains_zero,
                      cs.cumulative[0] <= 0.0 <= cs.cumulative[1],
                      cs.average[0] <= 0.0 <= cs.average[1]]
            assert len(set(inside)) == 1

    def test_the_point_estimate_is_inside_on_every_scale(self, prop99):
        Y, W, t0, pre = prop99
        n_post = Y.shape[0] - pre
        cs = confidence_set(Y, W, t0, pre, kind="constant", alpha=4 / 39,
                            precision=12)
        assert cs.cumulative[0] <= cs.point_estimate * n_post <= cs.cumulative[1]

    def test_the_cumulative_bound_is_the_path_it_plots(self, prop99):
        """The scale is not a second formula: it is the drawn path, summed."""
        Y, W, t0, pre = prop99
        for kind in ("constant", "linear"):
            cs = confidence_set(Y, W, t0, pre, kind=kind, alpha=4 / 39,
                                precision=12)
            assert cs.cumulative[0] == pytest.approx(float(cs.lower_path.sum()))
            assert cs.cumulative[1] == pytest.approx(float(cs.upper_path.sum()))


# =========================================================================== #
# sensitivity
# =========================================================================== #
class TestSensitivitySweep:
    """How far from uniform assignment the conclusion survives."""

    def test_sweep_reports_each_tilt_and_whether_zero_is_inside(self, prop99):
        Y, W, t0, pre = prop99
        v = np.zeros(Y.shape[1]); v[t0] = 1.0
        rows = sensitivity_sweep(Y, W, t0, pre, phis=(0.0, 0.5, 1.0), v=v,
                                 kind="linear", alpha=4 / 39, precision=30)
        assert [r.phi for r in rows] == [0.0, 0.5, 1.0]
        assert rows[0].contains_zero is False
        assert rows[-1].contains_zero is True       # California flips at phi=1

    def test_a_failed_tilt_is_recorded_not_raised(self, prop99):
        """A sweep is a diagnostic: one unbounded tilt must not kill the rest."""
        Y, W, t0, pre = prop99
        v = np.zeros(Y.shape[1]); v[t0] = 1.0
        rows = sensitivity_sweep(Y, W, t0, pre, phis=(0.0, 2.0), v=v,
                                 kind="linear", alpha=4 / 39, precision=30)
        assert rows[0].confidence_set is not None
        assert rows[1].confidence_set is None
        assert rows[1].reason


# =========================================================================== #
# refusals
# =========================================================================== #
class TestRefusals:
    """Every way the search can fail to produce a set, reported as one."""

    def test_an_empty_set_is_reported(self, prop99):
        """A level so lax the point estimate itself rejects has no set."""
        Y, W, t0, pre = prop99
        with pytest.raises(MlsynthEstimationError, match="empty"):
            confidence_set(Y, W, t0, pre, kind="linear", alpha=0.999,
                           precision=6)

    def test_an_unbounded_search_is_reported(self, prop99):
        """A level so strict nothing rejects walks off and must say so."""
        Y, W, t0, pre = prop99
        with pytest.raises(MlsynthEstimationError, match="bound"):
            confidence_set(Y, W, t0, pre, kind="linear", alpha=1e-9,
                           precision=6)

    def test_mismatched_weight_matrix_is_refused(self):
        Y, W, pre = _toy()
        with pytest.raises(MlsynthEstimationError, match="shape"):
            confidence_set(Y, W[:-1], 0, pre, kind="linear", precision=4)

    def test_treated_index_out_of_range_is_refused(self):
        Y, W, pre = _toy()
        with pytest.raises(MlsynthEstimationError, match="treated"):
            confidence_set(Y, W, 99, pre, kind="linear", precision=4)

    def test_no_post_periods_is_refused(self):
        Y, W, _ = _toy(n_periods=6)
        with pytest.raises(MlsynthEstimationError, match="post"):
            confidence_set(Y, W, 0, 6, kind="linear", precision=4)

    def test_single_unit_panel_is_refused(self):
        Y = np.arange(20.0).reshape(10, 2)
        W = np.ones((1, 2))
        with pytest.raises(MlsynthEstimationError, match="donor|unit"):
            confidence_set(Y[:, :1], np.zeros((0, 1)), 0, 6, precision=4)


# =========================================================================== #
# through the estimator
# =========================================================================== #
class TestVanillaSCIntegration:
    """``inference='placebo_cs'`` end to end on Proposition 99."""

    @staticmethod
    def _prop99_frame():
        import pandas as pd
        df = pd.read_csv(
            pathlib.Path(__file__).resolve().parents[2] / "basedata" / "P99data.csv")
        df["treat"] = ((df.state == "California") & (df.year >= 1989)).astype(int)
        return df

    @staticmethod
    def _fit(**extra):
        from mlsynth import VanillaSC
        cfg = dict(df=TestVanillaSCIntegration._prop99_frame(), outcome="cigsale",
                   treat="treat", unitid="state", time="year",
                   inference="placebo_cs", alpha=4 / 39,
                   placebo_cs_precision=12, display_graphs=False)
        cfg.update(extra)
        return VanillaSC(cfg).fit()

    def test_returns_a_confidence_set_that_excludes_zero(self):
        inf = self._fit(placebo_cs_class="linear").inference
        assert "Firpo-Possebom" in inf.method
        assert inf.ci_lower < inf.ci_upper < 0
        assert inf.details["contains_zero"] is False
        assert inf.confidence_level == pytest.approx(1 - 4 / 39)

    def test_both_effect_classes_run_and_bracket_their_point_estimate(self):
        for kind in ("constant", "linear"):
            d = self._fit(placebo_cs_class=kind).inference.details
            assert d["effect_class"] == kind
            assert d["lower_path"][0] == 0.0

    def test_the_sweep_reports_where_the_sign_is_lost(self):
        """The applied payoff: how far from uniform assignment the sign holds."""
        d = self._fit(placebo_cs_sweep=[0.0, 0.5, 1.0]).inference.details
        assert [r["phi"] for r in d["sensitivity"]] == [0.0, 0.5, 1.0]
        assert d["breakdown_phi"] == 1.0

    def test_it_reports_the_cumulative_and_average_scales(self):
        """A slope is hard to read; total packs not smoked is not."""
        inf = self._fit(placebo_cs_class="linear").inference
        d = inf.details
        k = d["n_post_periods"]
        assert k == 12                                  # 1989-2000
        weight = k * (k + 1) / 2.0
        assert d["cumulative_lower"] == pytest.approx(inf.ci_lower * weight)
        assert d["cumulative_upper"] == pytest.approx(inf.ci_upper * weight)
        assert d["att_lower"] == pytest.approx(d["cumulative_lower"] / k)
        assert d["att_upper"] == pytest.approx(d["cumulative_upper"] / k)
        assert d["att_lower"] < d["att_upper"] < 0.0

    def test_the_average_scale_brackets_the_reported_att(self):
        """The set is on the ATT's scale, so the two can be read together."""
        res = self._fit(placebo_cs_class="constant")
        d = res.inference.details
        assert d["att_lower"] <= float(res.effects.att) <= d["att_upper"]

    def test_an_unavailable_set_is_reported_and_warned_not_raised(self):
        """A level at which nothing rejects must not take the whole fit down."""
        with pytest.warns(UserWarning, match="bound|empty"):
            inf = self._fit(alpha=1e-9).inference
        assert inf.ci_lower is None
        assert "unavailable_reason" in inf.details

    def test_an_unknown_effect_class_is_refused_at_config_time(self):
        from mlsynth import VanillaSC
        with pytest.raises(Exception, match="placebo_cs_class|validation"):
            VanillaSC(dict(df=self._prop99_frame(), outcome="cigsale",
                           treat="treat", unitid="state", time="year",
                           inference="placebo_cs", placebo_cs_class="cubic",
                           display_graphs=False))

    def test_misspelled_inference_mode_is_refused(self):
        from mlsynth import VanillaSC
        with pytest.raises(Exception, match="not a recognized"):
            VanillaSC(dict(df=self._prop99_frame(), outcome="cigsale",
                           treat="treat", unitid="state", time="year",
                           inference="placebo_cset", display_graphs=False))
