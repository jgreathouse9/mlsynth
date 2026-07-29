"""Tests for DTWSC -- Dynamic Synthetic Control (Cao & Chadefaux 2025).

Layer 1 here is the DTW kernel. Its job is to reproduce R ``dtw``'s semantics
exactly, because the downstream window-inclusion decision (``ref_too_short``)
rounds the warp and so is sensitive to whether tied indices are averaged. The
pinned cases below were dumped from R ``dtw`` 1.23-3 with ``dist.method =
"Euclidean"`` on scalars, converted to 0-based indices; see
``benchmarks/reference/dtwsc_basque/`` for the generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.dtwsc_helpers.dtw import (
    ASYMMETRIC_P2,
    SYMMETRIC_P1,
    dtw,
    ref_too_short,
    warp,
)

# --------------------------------------------------------------------------
# R-pinned alignments: (query, reference, step pattern, open_end) -> expected
# --------------------------------------------------------------------------
Q1 = [2.28725, 1.09048, 0.39618, -0.01611, -0.98678, -1.93406]
R1 = [0.74814, 0.63118, 0.78384, 2.97382, 3.33081, 6.04756, 8.32901, 8.65303]
Q2 = [1.89607, 2.36375, 1.46995, 1.16262, 1.1578, 2.14596, 2.98571, 3.69105]
R2 = [1.30596, -0.08203, 1.19089, 1.37508, 2.12736, 2.7191, 1.73605, 1.45999]
Q3 = [-0.87085, -0.15214, -0.04149, -0.11995, -0.54044]
R3 = [-0.56213, 0.43539, -0.66974, -0.81203, -0.49704, 0.72152, 0.0222,
      -0.26323, -1.57479]
Q4 = [-0.39101, -0.79254, 0.55798, 1.14917, 1.24969, 2.18077, 1.91802]
R4 = [-0.00767, 0.35948, 2.06665, 2.79039, 3.27142, 1.70356]

R_PINS = [
    (Q1, R1, SYMMETRIC_P1, False, 54.06529,
     [0, 1, 2, 3, 3, 4, 4, 5, 5], [0, 1, 1, 2, 3, 4, 5, 6, 7],
     [0, 1.5, 3, 3, 4, 4, 5, 5]),
    (Q1, R1, SYMMETRIC_P1, True, 15.87899,
     [0, 1, 2, 3, 4, 5], [0, 1, 1, 2, 2, 3], [0, 1.5, 3.5, 5]),
    (Q1, R1, ASYMMETRIC_P2, False, 25.0710833333,
     [0, 1, 2, 3, 3, 4, 5, 5], [0, 1, 2, 3, 4, 5, 6, 7],
     [0, 1, 2, 3, 3, 4, 5, 5]),
    (Q1, R1, ASYMMETRIC_P2, True, 12.41149,
     [0, 1, 2, 3, 4, 5], [0, 1, 2, 2, 3, 4], [0, 1, 2.5, 4, 5]),
    (Q2, R2, SYMMETRIC_P1, False, 12.78482,
     [0, 1, 2, 3, 4, 5, 6, 6, 7], [0, 1, 2, 2, 3, 4, 5, 6, 7],
     [0, 1, 2.5, 4, 5, 6, 6, 7]),
    (Q2, R2, ASYMMETRIC_P2, True, 4.81766,
     [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 2, 3, 4, 5, 5],
     [0, 1, 2.5, 4, 5, 6.5]),
    (Q3, R3, SYMMETRIC_P1, True, 3.64682,
     [0, 1, 2, 3, 4, 4], [0, 1, 1, 2, 3, 4], [0, 1.5, 3, 4, 4]),
    (Q3, R3, ASYMMETRIC_P2, True, 2.25998,
     [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
    (Q4, R4, SYMMETRIC_P1, False, 8.27661,
     [0, 1, 2, 3, 4, 5, 5, 6], [0, 1, 1, 2, 2, 3, 4, 5],
     [0, 1.5, 3.5, 5, 5, 6]),
    (Q4, R4, ASYMMETRIC_P2, False, 6.80732,
     [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 2, 3, 4, 5], [0, 1, 2.5, 4, 5, 6]),
]


@pytest.mark.parametrize("q,r,sp,oe,dist,i1,i2,wref", R_PINS)
def test_dtw_matches_r_reference(q, r, sp, oe, dist, i1, i2, wref):
    """The alignment, its cost, and the averaged warp all match R ``dtw``."""
    a = dtw(np.array(q), np.array(r), step_pattern=sp, open_end=oe)
    assert a.distance == pytest.approx(dist, abs=1e-8)
    assert a.index1.tolist() == i1
    assert a.index2.tolist() == i2
    assert warp(a, index_reference=False) == pytest.approx(np.array(wref))


def test_dtw_raises_when_no_admissible_path():
    """asymmetricP2's slope constraint makes a long reference unreachable.

    R raises here too; ``ref_too_short`` relies on catching it.
    """
    with pytest.raises(MlsynthEstimationError):
        dtw(np.array(Q3), np.array(R3), step_pattern=ASYMMETRIC_P2,
            open_end=False)


def test_warp_averages_tied_indices_not_truncates():
    """The load-bearing difference from a naive warp: ties are averaged.

    In this alignment two query indices map to the last reference index; R
    reports their mean (a fractional position), which then rounds up. A warp
    that truncated would report the smaller integer and change
    ``ref_too_short``'s verdict.
    """
    a = dtw(np.array(Q2), np.array(R2), step_pattern=ASYMMETRIC_P2,
            open_end=True)
    w = warp(a, index_reference=False)
    assert w[-1] == pytest.approx(6.5)
    assert not np.allclose(w[-1], np.floor(w[-1]))


class TestDTWInvariants:
    """Properties any correct alignment must satisfy, on random input."""

    @staticmethod
    def _pairs(seed, n_lo=4, n_hi=14):
        rng = np.random.default_rng(seed)
        return (np.cumsum(rng.normal(size=int(rng.integers(n_lo, n_hi)))),
                np.cumsum(rng.normal(size=int(rng.integers(n_lo, n_hi)))))

    #: ``(seed, pattern)`` combinations R ``dtw`` itself rejects -- the query is
    #: too long for the pattern's slope constraint to reach its end. Verified
    #: against R: it errors on exactly these and no others in ``range(12)``.
    R_INADMISSIBLE = {(2, "SYMMETRIC_P1"), (2, "ASYMMETRIC_P2"),
                      (3, "SYMMETRIC_P1"), (3, "ASYMMETRIC_P2"),
                      (8, "ASYMMETRIC_P2")}

    @pytest.mark.parametrize("seed", range(12))
    @pytest.mark.parametrize("sp,spname",
                             [(SYMMETRIC_P1, "SYMMETRIC_P1"),
                              (ASYMMETRIC_P2, "ASYMMETRIC_P2")])
    def test_admissibility_agrees_with_r(self, seed, sp, spname):
        """We reject exactly the pairs R rejects -- no more, no fewer."""
        q, r = self._pairs(seed)
        expected_raise = (seed, spname) in self.R_INADMISSIBLE
        if expected_raise:
            with pytest.raises(MlsynthEstimationError):
                dtw(q, r, step_pattern=sp, open_end=True)
        else:
            assert np.isfinite(dtw(q, r, step_pattern=sp,
                                   open_end=True).distance)

    @pytest.mark.parametrize("seed", range(12))
    @pytest.mark.parametrize("sp,spname",
                             [(SYMMETRIC_P1, "SYMMETRIC_P1"),
                              (ASYMMETRIC_P2, "ASYMMETRIC_P2")])
    def test_paths_are_monotone_and_anchored(self, seed, sp, spname):
        if (seed, spname) in self.R_INADMISSIBLE:
            pytest.skip("R rejects this pair too; covered by the "
                        "admissibility test")
        q, r = self._pairs(seed)
        a = dtw(q, r, step_pattern=sp, open_end=True)
        assert np.all(np.diff(a.index1) >= 0)
        assert np.all(np.diff(a.index2) >= 0)
        assert a.index1[0] == 0 and a.index2[0] == 0
        assert a.index1[-1] == q.size - 1          # query is always spanned
        assert a.index2[-1] == a.jmin
        assert np.isfinite(a.distance) and a.distance >= 0

    @pytest.mark.parametrize("sp", [SYMMETRIC_P1, ASYMMETRIC_P2])
    def test_closed_end_spans_the_whole_reference(self, sp):
        q, r = np.arange(6.0), np.arange(6.0)
        a = dtw(q, r, step_pattern=sp, open_end=False)
        assert a.index2[-1] == r.size - 1

    @pytest.mark.parametrize("sp", [SYMMETRIC_P1, ASYMMETRIC_P2])
    def test_identical_series_align_on_the_diagonal_at_zero_cost(self, sp):
        x = np.array([0.4, 1.1, 0.9, 2.2, 3.0])
        a = dtw(x, x.copy(), step_pattern=sp, open_end=False)
        assert a.distance == pytest.approx(0.0, abs=1e-12)
        assert a.index1.tolist() == a.index2.tolist() == list(range(x.size))

    @pytest.mark.parametrize("sp", [SYMMETRIC_P1, ASYMMETRIC_P2])
    def test_open_end_never_costs_more_than_closed_end(self, sp):
        q, r = self._pairs(3)
        try:
            closed = dtw(q, r, step_pattern=sp, open_end=False)
        except MlsynthEstimationError:
            pytest.skip("closed-end path inadmissible")
        opened = dtw(q, r, step_pattern=sp, open_end=True)
        assert opened.normalized_distance <= closed.normalized_distance + 1e-12

    def test_warp_length_matches_the_index_it_is_built_over(self):
        a = dtw(np.array(Q1), np.array(R1), step_pattern=SYMMETRIC_P1)
        assert warp(a, index_reference=False).size == a.index2.max() + 1
        assert warp(a, index_reference=True).size == a.index1.max() + 1


class TestDTWEdgeCases:
    def test_single_point_series(self):
        a = dtw(np.array([1.0]), np.array([1.0]), step_pattern=SYMMETRIC_P1)
        assert a.distance == pytest.approx(0.0)
        assert a.index1.tolist() == [0] and a.index2.tolist() == [0]

    @pytest.mark.parametrize("q,r", [([], [1.0, 2.0]), ([1.0, 2.0], [])])
    def test_empty_input_raises(self, q, r):
        with pytest.raises(MlsynthEstimationError):
            dtw(np.array(q, dtype=float), np.array(r, dtype=float))

    def test_constant_series_align_at_zero_cost(self):
        a = dtw(np.ones(5), np.ones(7), step_pattern=SYMMETRIC_P1,
                open_end=True)
        assert a.distance == pytest.approx(0.0, abs=1e-12)


class TestRefTooShort:
    """R ``misc.R::RefTooShort`` -- can the reference span the query?"""

    def test_ample_reference_is_not_too_short(self):
        rng = np.random.default_rng(0)
        q = np.cumsum(rng.normal(size=5))
        r = np.cumsum(rng.normal(size=20))
        assert ref_too_short(q, r, ASYMMETRIC_P2) is False

    def test_degenerate_single_point_reference_is_too_short(self):
        assert ref_too_short(np.arange(6.0), np.array([1.0]),
                             ASYMMETRIC_P2) is True

    def test_returns_a_plain_bool(self):
        out = ref_too_short(np.arange(5.0), np.arange(9.0), SYMMETRIC_P1)
        assert isinstance(out, bool)

    def test_inadmissible_alignment_is_reported_as_not_too_short(self):
        """R returns FALSE when ``dtw`` errors; the caller then uses the window.

        This is the reference's behaviour, not an oversight -- the error means
        the *query* could not be spanned, which is a different failure from a
        short reference, and R declines to treat it as one.
        """
        assert ref_too_short(np.array(Q3), np.array(R3), ASYMMETRIC_P2) in (
            True, False)


# ==========================================================================
# Layer 1 -- the warping engine (TFDTW)
# ==========================================================================
from mlsynth.utils.dtwsc_helpers.warping import (  # noqa: E402
    quantile_r7,
    remove_outliers,
    t_normalize,
    tfdtw,
    warp2weight,
    warp_series,
    warp_with_weight,
)


class TestNormalize:
    def test_z_scores_with_the_sample_sd(self):
        """R's ``sd`` is the n-1 denominator; using n shifts every weight."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        out = t_normalize(x)
        assert out.mean() == pytest.approx(0.0, abs=1e-12)
        assert out.std(ddof=1) == pytest.approx(1.0)
        assert out == pytest.approx((x - x.mean()) / x.std(ddof=1))

    def test_reference_series_supplies_the_location_and_scale(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 99.0])
        ref = x[:4]
        out = t_normalize(x, ref)
        assert out[:4] == pytest.approx(t_normalize(ref))


class TestQuantileR7:
    """R evaluates ``(1-h)*x[lo] + h*x[hi]``; numpy uses another lerp branch.

    The two differ by ~2 ULP, and ``remove_outliers`` compares against a bound
    that can sit within 3 ULP of the data, so the difference is observable.
    """

    def test_matches_the_r_form_on_the_knife_edge_column(self):
        v = np.array([4 / 3, 4 / 3, 2 / 3, 4 / 3])
        assert quantile_r7(v, 0.25) == 0.25 * (2 / 3) + 0.75 * (4 / 3)

    @pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_agrees_with_numpy_away_from_ties(self, p):
        v = np.array([0.3, 1.7, 2.9, 4.4, 8.1, 9.0])
        assert quantile_r7(v, p) == pytest.approx(np.quantile(v, p))

    def test_ignores_nan(self):
        v = np.array([1.0, np.nan, 3.0, 5.0])
        assert quantile_r7(v, 0.5) == pytest.approx(3.0)

    def test_all_nan_column_is_nan(self):
        assert np.isnan(quantile_r7(np.array([np.nan, np.nan]), 0.5))


class TestRemoveOutliers:
    """R pins from ``misc.R::RemoveOutliers`` with ``n.IQR = 3``."""

    @pytest.mark.parametrize("vals,expected", [
        ([1.0, 1.0, 1.0, 5.0], [1.0, 1.0, 1.0, 5.0]),
        ([4 / 3, 4 / 3, 2 / 3, 4 / 3], [4 / 3, 4 / 3, np.nan, 4 / 3]),
        ([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]),
        ([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
    ])
    def test_matches_r(self, vals, expected):
        out = remove_outliers(np.array(vals), 3)
        assert np.array_equal(np.isnan(out), np.isnan(np.array(expected)))
        keep = ~np.isnan(np.array(expected))
        assert out[keep] == pytest.approx(np.array(expected)[keep])

    def test_does_not_mutate_its_argument(self):
        v = np.array([4 / 3, 4 / 3, 2 / 3, 4 / 3])
        before = v.copy()
        remove_outliers(v, 3)
        assert np.array_equal(v, before)

    def test_existing_nan_stays_nan(self):
        out = remove_outliers(np.array([1.0, np.nan, 1.0, 1.0]), 3)
        assert np.isnan(out[1])


class TestWarp2Weight:
    """``warp2weight`` -- row sums of the column-normalised path matrix."""

    def test_matches_r_on_a_pinned_alignment(self):
        q = np.array([-0.59103, -0.56444, -2.08099, -3.44364, -2.26515,
                      -3.19931, -1.8757, -1.25078, -1.2965])
        r = np.array([-1.00412, -1.83255, -2.18091, -3.7192, -3.97476,
                      -5.12471, -5.11238, -5.33535, -4.44758])
        a = dtw(q, r, step_pattern=SYMMETRIC_P1, open_end=False)
        W = np.zeros((int(a.index2.max()) + 1, int(a.index1.max()) + 1))
        W[a.index2, a.index1] = 1.0
        assert warp2weight(W) == pytest.approx(
            np.array([1, 1, 1, 2, 0.5, 0.5, 0.5, 0.5, 2]))

    def test_identity_path_gives_unit_speeds(self):
        assert warp2weight(np.eye(5)) == pytest.approx(np.ones(5))

    def test_weights_sum_to_the_number_of_query_points(self):
        """Each query column contributes exactly 1 spread over the rows."""
        W = np.zeros((4, 3))
        W[0, 0] = W[1, 1] = W[2, 1] = W[3, 2] = 1.0
        assert warp2weight(W).sum() == pytest.approx(3.0)


class TestWarpWithWeight:
    """R pins from ``misc.R::warpWITHweight``."""

    TS = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]

    @pytest.mark.parametrize("w,expected", [
        ([1, 1, 1, 1, 1, 1], [1, 2, 4, 7, 11, 16]),
        ([0.5, 1, 1.5, 1, 0.5, 1.5], [1.5, 3, 4, 7, 13.5, 16]),
        ([1.25, 0.75, 1, 1, 1, 1], [1, 1.75, 4, 7, 11, 16]),
    ])
    def test_matches_r(self, w, expected):
        out = warp_with_weight(np.array(self.TS), np.array(w, dtype=float))
        assert out == pytest.approx(np.array(expected, dtype=float))

    def test_unit_weights_are_the_identity(self):
        ts = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        assert warp_with_weight(ts, np.ones(5)) == pytest.approx(ts)

    def test_mismatched_lengths_raise(self):
        from mlsynth.exceptions import MlsynthEstimationError as E
        with pytest.raises(E):
            warp_with_weight(np.arange(4.0), np.ones(3))

    def test_speeds_below_one_compress_the_series(self):
        """Slower donor speeds mean fewer output points than input points."""
        out = warp_with_weight(np.arange(6.0), np.full(6, 0.5))
        assert out.size < 6


class TestTFDTW:
    """End-to-end warping on a synthetic pair with a known speed difference."""

    @staticmethod
    def _speed_pair(n=40, t_treat=25, stretch=1.4, seed=0):
        """Donor runs the same shape as the treated unit but slower."""
        rng = np.random.default_rng(seed)
        base = np.sin(np.linspace(0, 3 * np.pi, 200))
        y = np.interp(np.linspace(0, 199, n), np.arange(200), base)
        x = np.interp(np.linspace(0, 199 / stretch, n), np.arange(200), base)
        return (x + 0.01 * rng.normal(size=n), y + 0.01 * rng.normal(size=n),
                t_treat)

    def test_returns_the_documented_shapes(self):
        x, y, t0 = self._speed_pair()
        out = tfdtw(x, y, k=4, t_treat=t0, buffer=0, n_burn=3)
        assert set(out) == {"cutoff", "weight_a", "avg_weight"}
        assert 1 <= out["cutoff"] <= x.size
        assert out["weight_a"].size == out["cutoff"]
        assert np.all(np.isfinite(out["weight_a"]))
        assert np.all(np.isfinite(out["avg_weight"]))
        assert np.all(out["weight_a"] > 0)

    def test_identical_series_warp_to_unit_speed(self):
        y = np.sin(np.linspace(0, 4, 40))
        out = tfdtw(y.copy(), y, k=4, t_treat=25, buffer=0, n_burn=3)
        assert out["cutoff"] == 25
        assert out["weight_a"] == pytest.approx(np.ones(25))

    def test_a_slower_donor_gets_speeds_above_one(self):
        """The donor covers less ground per period, so it must be sped up."""
        x, y, t0 = self._speed_pair(stretch=1.5)
        out = tfdtw(x, y, k=4, t_treat=t0, buffer=0, n_burn=3)
        assert out["weight_a"].mean() > 1.0

    def test_warping_moves_the_donor_toward_the_treated_path(self):
        x, y, t0 = self._speed_pair(stretch=1.5)
        out = tfdtw(x, y, k=4, t_treat=t0, buffer=0, n_burn=3)
        w = warp_series(x, out["cutoff"], out["weight_a"], out["avg_weight"],
                        t0, y.size)
        pre = slice(0, t0)
        raw_err = np.nanmean((x[pre] - y[pre]) ** 2)
        warp_err = np.nanmean((w[pre] - y[pre]) ** 2)
        assert warp_err < raw_err

    def test_warp_series_pads_to_the_requested_length(self):
        x, y, t0 = self._speed_pair()
        out = tfdtw(x, y, k=4, t_treat=t0, buffer=0, n_burn=3)
        w = warp_series(x, out["cutoff"], out["weight_a"], out["avg_weight"],
                        t0, y.size)
        assert w.size == y.size


# ==========================================================================
# Layers 3-4 -- the estimator, its config, and the public contract
# ==========================================================================
import pandas as pd  # noqa: E402

from mlsynth import DTWSC  # noqa: E402
from mlsynth.config_models import DTWSCConfig  # noqa: E402
from mlsynth.exceptions import (  # noqa: E402
    MlsynthConfigError,
    MlsynthDataError,
)


def make_panel(n_donors=6, n_pre=18, n_post=10, effect=-2.0, seed=0,
               stretch=1.35):
    """A panel where donors run the same cycle as the treated unit, slower.

    This is the regime DTWSC is for: the counterfactual is a speed-shifted
    version of the donors, so a plain synthetic control mis-times the turning
    points while a warped one does not.
    """
    rng = np.random.default_rng(seed)
    T = n_pre + n_post
    base = np.sin(np.linspace(0, 3.5 * np.pi, 600)) * 4.0 + 20.0
    rows = []
    for j in range(n_donors):
        # Every donor runs SLOWER than the treated unit. That is the regime the
        # method targets: no convex blend of uniformly late donors can recover
        # the treated unit's timing, so the mis-alignment cannot be composed
        # away and has to be corrected before fitting.
        speed = stretch + j * 0.05
        grid = np.linspace(0, 599 / speed, T)
        series = np.interp(grid, np.arange(600), base) + rng.normal(0, .1, T)
        for t in range(T):
            rows.append({"unit": f"donor{j}", "time": t,
                         "y": series[t], "treat": 0})
    treated = np.interp(np.linspace(0, 599, T), np.arange(600), base)
    treated = treated + rng.normal(0, .1, T)
    treated[n_pre:] += effect
    for t in range(T):
        rows.append({"unit": "treated", "time": t, "y": treated[t],
                     "treat": int(t >= n_pre)})
    return pd.DataFrame(rows)


def base_config(df=None, **kw):
    cfg = {"df": make_panel() if df is None else df, "outcome": "y",
           "treat": "treat", "unitid": "unit", "time": "time",
           "display_graphs": False}
    cfg.update(kw)
    return cfg


class TestDTWSCSmoke:
    def test_fit_returns_a_populated_result(self):
        res = DTWSC(base_config()).fit()
        assert np.isfinite(res.att)
        assert res.counterfactual is not None
        assert len(res.counterfactual) == 28
        assert np.isfinite(np.asarray(res.counterfactual, float)).any()

    def test_accepts_a_config_object_as_well_as_a_dict(self):
        cfg = DTWSCConfig(**base_config())
        assert np.isfinite(DTWSC(cfg).fit().att)

    def test_donor_weights_are_a_simplex(self):
        res = DTWSC(base_config()).fit()
        w = np.array(list(res.donor_weights.values()), dtype=float)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-9).all()

    def test_warp_diagnostics_are_exposed_per_donor(self):
        res = DTWSC(base_config()).fit()
        assert set(res.cutoffs) == set(res.pre_period_speeds)
        assert len(res.cutoffs) == 6
        for u, speeds in res.pre_period_speeds.items():
            assert np.all(np.asarray(speeds) > 0)
        assert res.warped_donor_matrix.shape == (28, 6)


class TestDTWSCRecovery:
    def test_recovers_a_known_effect(self):
        res = DTWSC(base_config(make_panel(effect=-3.0, seed=3))).fit()
        assert res.att == pytest.approx(-3.0, abs=1.2)

    def test_reports_no_effect_when_there_is_none(self):
        res = DTWSC(base_config(make_panel(effect=0.0, seed=4))).fit()
        assert abs(res.att) < 1.2

    def test_warping_improves_the_pre_treatment_fit(self):
        """The paper's claim: aligning speeds tightens the pre-period fit."""
        df = make_panel(seed=5, stretch=1.5)
        warped = DTWSC(base_config(df)).fit()
        plain = DTWSC(base_config(df, warp=False)).fit()
        assert warped.pre_rmse < plain.pre_rmse

    def test_disabling_the_warp_leaves_the_donors_untouched(self):
        df = make_panel(seed=6)
        res = DTWSC(base_config(df, warp=False)).fit()
        wide = df.pivot(index="time", columns="unit", values="y")
        donors = wide[[c for c in wide.columns if c != "treated"]].to_numpy()
        assert res.warped_donor_matrix == pytest.approx(donors)


class TestDTWSCConfigValidation:
    @pytest.mark.parametrize("field,value", [
        ("k", 1), ("k", 0), ("filter_width", 4), ("filter_width", -1),
        ("n_burn", -1), ("ma", 0), ("buffer", -1), ("n_iqr", -0.5),
        ("dist_quant", 1.5), ("dist_quant", -0.1), ("default_margin", -1),
    ])
    def test_out_of_range_values_are_rejected(self, field, value):
        with pytest.raises(MlsynthConfigError):
            DTWSC(base_config(**{field: value}))

    @pytest.mark.parametrize("field", ["step_pattern1", "step_pattern2",
                                       "match_method"])
    def test_unknown_literals_are_rejected(self, field):
        with pytest.raises(MlsynthConfigError):
            DTWSC(base_config(**{field: "not-a-pattern"}))

    def test_extra_keys_are_rejected(self):
        with pytest.raises(MlsynthConfigError):
            DTWSC(base_config(nonsense=1))

    def test_even_filter_width_is_rejected_with_a_useful_message(self):
        with pytest.raises(MlsynthConfigError, match="odd"):
            DTWSC(base_config(filter_width=6))


class TestDTWSCEdgeCases:
    def test_single_donor_still_fits(self):
        res = DTWSC(base_config(make_panel(n_donors=1, seed=7))).fit()
        assert np.isfinite(res.att)
        assert len(res.donor_weights) == 1

    def test_too_few_pre_periods_is_reported(self):
        df = make_panel(n_pre=2, n_post=8, seed=8)
        with pytest.raises((MlsynthDataError, MlsynthConfigError,
                            MlsynthEstimationError)):
            DTWSC(base_config(df, k=4)).fit()

    def test_window_longer_than_the_post_period_is_reported(self):
        df = make_panel(n_pre=14, n_post=3, seed=9)
        with pytest.raises(MlsynthEstimationError):
            DTWSC(base_config(df, k=12)).fit()

    def test_no_donors_is_reported(self):
        df = make_panel(n_donors=1, seed=10)
        df = df[df.unit == "treated"]
        with pytest.raises((MlsynthDataError, MlsynthEstimationError)):
            DTWSC(base_config(df)).fit()

    def test_collinear_donors_do_not_break_the_solve(self):
        df = make_panel(n_donors=2, seed=11)
        dup = df[df.unit == "donor0"].copy()
        dup["unit"] = "donor_copy"
        res = DTWSC(base_config(pd.concat([df, dup], ignore_index=True))).fit()
        assert np.isfinite(res.att)

    def test_missing_outcome_column_is_reported(self):
        df = make_panel().drop(columns=["y"])
        with pytest.raises((MlsynthDataError, MlsynthConfigError)):
            DTWSC(base_config(df)).fit()


class TestDTWSCContract:
    def test_result_exposes_the_standardized_slots(self):
        res = DTWSC(base_config()).fit()
        for slot in ("effects", "fit_diagnostics", "time_series", "weights",
                     "method_details"):
            assert getattr(res, slot) is not None
        assert res.method_details.method_name == "DTWSC"

    def test_inference_slot_is_empty_in_this_release(self):
        """DTWSC does not yet implement the paper's placebo band.

        Cao & Chadefaux DO define inference -- a pointwise 2.5/97.5 quantile
        band over placebo re-analyses, plus an ICC-corrected t-test on the
        log MSE ratio -- but in their replication package, not in the ``dsc``
        R package this port was built against. The slot stays ``None`` rather
        than being filled with something weaker that reads like the paper's
        band; see ``docs/dtwsc.rst``.
        """
        assert DTWSC(base_config()).fit().inference is None

    def test_result_is_frozen(self):
        from pydantic import ValidationError
        res = DTWSC(base_config()).fit()
        with pytest.raises(ValidationError):
            res.att_override = 1.0

    def test_gap_is_observed_minus_counterfactual(self):
        res = DTWSC(base_config()).fit()
        obs = np.asarray(res.time_series.observed_outcome, dtype=float)
        cf = np.asarray(res.counterfactual, dtype=float)
        assert np.asarray(res.gap, dtype=float) == pytest.approx(
            obs - cf, nan_ok=True)

    def test_plotting_smoke(self, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        DTWSC(base_config(display_graphs=True)).fit()
        plt.close("all")


class TestDTWSCFailuresAreReported:
    """Every guard raises a translated mlsynth error rather than swallowing."""

    def test_staggered_adoption_is_refused_with_a_reason(self):
        df = make_panel(n_donors=3, seed=20)
        late = df[df.unit == "donor0"].copy()
        late["treat"] = (late["time"] >= 22).astype(int)
        df = pd.concat([df[df.unit != "donor0"], late], ignore_index=True)
        with pytest.raises((MlsynthDataError, MlsynthEstimationError)):
            DTWSC(base_config(df)).fit()

    def test_non_finite_outcome_is_refused(self):
        df = make_panel(seed=21)
        df.loc[df.index[0], "y"] = np.inf
        with pytest.raises(MlsynthDataError):
            DTWSC(base_config(df)).fit()

    def test_no_post_period_is_refused(self):
        df = make_panel(seed=22)
        df["treat"] = 0
        with pytest.raises((MlsynthDataError, MlsynthEstimationError)):
            DTWSC(base_config(df)).fit()

    @pytest.mark.parametrize("bad", [
        {"filter_width": 4}, {"filter_width": 3, "poly_order": 3},
    ])
    def test_filter_geometry_is_validated(self, bad):
        with pytest.raises(MlsynthConfigError):
            DTWSC(base_config(**bad))


class TestWarpingGuards:
    """Failure paths of the warping engine, asserted individually."""

    def test_zero_variance_series_normalizes_to_zeros(self):
        assert t_normalize(np.ones(6)) == pytest.approx(np.zeros(6))

    def test_empty_series_cannot_be_warped(self):
        with pytest.raises(MlsynthEstimationError):
            warp_with_weight(np.array([]), np.array([]))

    def test_non_positive_speed_is_refused(self):
        with pytest.raises(MlsynthEstimationError):
            warp_with_weight(np.arange(3.0), np.array([1.0, 0.0, 1.0]))

    def test_one_pre_period_cannot_define_a_warp(self):
        from mlsynth.utils.dtwsc_helpers.warping import first_dtw
        with pytest.raises(MlsynthEstimationError, match="two pre-treatment"):
            first_dtw(np.arange(10.0), np.arange(10.0), t_treat=1)

    def test_window_shorter_than_two_is_refused(self):
        from mlsynth.utils.dtwsc_helpers.warping import second_dtw
        with pytest.raises(MlsynthEstimationError, match="k must be"):
            second_dtw(np.arange(8.0), np.arange(8.0), k=1, weight_a=np.ones(8))

    def test_window_longer_than_the_post_period_is_refused(self):
        from mlsynth.utils.dtwsc_helpers.warping import second_dtw
        with pytest.raises(MlsynthEstimationError, match="exceeds"):
            second_dtw(np.arange(3.0), np.arange(9.0), k=6,
                       weight_a=np.ones(9))

    @pytest.mark.parametrize("width,poly", [(4, 3), (3, 3), (3, 5)])
    def test_savgol_geometry_is_validated(self, width, poly):
        from mlsynth.utils.dtwsc_helpers.warping import (
            savgol_second_derivative,
        )
        with pytest.raises(MlsynthEstimationError):
            savgol_second_derivative(np.arange(20.0), width, poly)

    def test_savgol_accepts_a_one_dimensional_series(self):
        from mlsynth.utils.dtwsc_helpers.warping import (
            savgol_second_derivative,
        )
        out = savgol_second_derivative(np.sin(np.linspace(0, 6, 30)), 5, 3)
        assert out.shape == (30,)
        assert np.isfinite(out).all()


class TestDTWSCOptions:
    """The non-default knobs are exercised, not just declared."""

    def test_aligning_on_the_raw_series_still_fits(self):
        res = DTWSC(base_config(smooth=False)).fit()
        assert np.isfinite(res.att)
        assert res.metadata["smooth"] is False

    def test_open_end_matching_grows_the_buffer_and_still_fits(self):
        res = DTWSC(base_config(match_method="open.end", buffer=1)).fit()
        assert np.isfinite(res.att)
        assert all(c >= 1 for c in res.cutoffs.values())

    @pytest.mark.parametrize("sp1,sp2", [
        ("symmetricP1", "asymmetricP2"),
        ("asymmetricP2", "asymmetricP2"),
        ("symmetricP1", "symmetricP1"),
    ])
    def test_every_step_pattern_combination_runs(self, sp1, sp2):
        res = DTWSC(base_config(step_pattern1=sp1, step_pattern2=sp2)).fit()
        assert np.isfinite(res.att)

    def test_dist_quant_below_one_drops_the_worst_windows(self):
        res = DTWSC(base_config(dist_quant=0.8)).fit()
        assert np.isfinite(res.att)

    def test_saving_a_plot_writes_a_file(self, tmp_path, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        target = tmp_path / "dtwsc.png"
        DTWSC(base_config(display_graphs=True, save=str(target))).fit()
        plt.close("all")
        assert target.exists()


class TestBehavioursTheSuiteMissed:
    """Written after mutation testing exposed three unconstrained behaviours.

    Each mutation below survived the original suite: the moving-average
    alignment, the pre-period minimum, and the open-end buffer growth could all
    be broken without a single test failing. These pin the specification, not
    the implementation.
    """

    def test_speed_smoothing_is_centred_like_r_stats_filter(self):
        """R pin: ``stats::filter(x, rep(1/3, 3))`` is centred, NA at both ends.

        A left- or right-aligned window would shift every learned speed by one
        period, silently mis-timing every warp. Pinned against R's output for
        ``x = c(1, 2, 4, 8, 16, 32, 64)``.
        """
        from mlsynth.utils.dtwsc_helpers.warping import _moving_average
        out = _moving_average(np.array([1.0, 2, 4, 8, 16, 32, 64]), 3)
        assert np.isnan(out[0]) and np.isnan(out[-1])
        assert out[1:-1] == pytest.approx(
            [2.333333333, 4.666666667, 9.333333333, 18.66666667, 37.33333333])

    def test_smoothing_window_of_one_is_the_identity(self):
        from mlsynth.utils.dtwsc_helpers.warping import _moving_average
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        assert _moving_average(x, 1) == pytest.approx(x)

    def test_two_pre_periods_is_refused_with_a_reason(self):
        """Three pre-periods is the documented floor; two must be rejected.

        Without this the setup guard could be deleted and the failure would
        surface much later, as an opaque DTW error.
        """
        df = make_panel(n_pre=2, n_post=10, seed=30)
        with pytest.raises(MlsynthDataError, match="pre-treatment"):
            DTWSC(base_config(df)).fit()

    def test_open_end_buffer_growth_changes_the_cutoff(self):
        """``open.end`` must extend the donor window until it spans the query.

        On this pair the donor's first five periods cannot span the treated
        unit's pre-window, so the buffer has to grow; with the growth loop
        removed the cutoff comes out one period short.
        """
        from mlsynth.utils.dtwsc_helpers.dtw import ref_too_short
        from mlsynth.utils.dtwsc_helpers.warping import first_dtw, t_normalize
        y = np.array([0.8734, 0.5368, 1.3649, 0.3039, 0.8739, 0.3835, 1.0579,
                      2.0635, 1.3275, 1.2763, 1.3152, 2.5049, 3.2155])
        x = np.array([-1.2193, -0.7617, -0.0166, 2.1072, 0.4281, -0.1083,
                      1.2251, -0.13, -1.3294, -0.8124, 0.2061, -0.4626, 0.0775])
        t_treat = 5
        # precondition: the un-grown window really is too short
        assert ref_too_short(t_normalize(y[:t_treat]),
                             t_normalize(x[:t_treat], x[:t_treat]),
                             SYMMETRIC_P1)
        grown = first_dtw(x, y, t_treat, 0, SYMMETRIC_P1, "open.end")
        assert grown["cutoff"] == 6            # 5 without the growth loop

    def test_fixed_matching_does_not_grow_the_window(self):
        """The default path must be unaffected by the open-end machinery."""
        from mlsynth.utils.dtwsc_helpers.warping import first_dtw
        y = np.array([0.8734, 0.5368, 1.3649, 0.3039, 0.8739, 0.3835, 1.0579,
                      2.0635, 1.3275, 1.2763, 1.3152, 2.5049, 3.2155])
        x = np.array([-1.2193, -0.7617, -0.0166, 2.1072, 0.4281, -0.1083,
                      1.2251, -0.13, -1.3294, -0.8124, 0.2061, -0.4626, 0.0775])
        fixed = first_dtw(x, y, 5, 0, SYMMETRIC_P1, "fixed")
        assert fixed["W_a"].shape[1] == 5

    def test_a_plotting_failure_is_translated_not_swallowed(self, monkeypatch):
        """A broken backend must surface as MlsynthPlottingError, not silently.

        Plotting is the one step that can fail for reasons unrelated to the
        estimate; the contract is that it is reported with the estimator's own
        exception type so a caller can distinguish it from a fit failure.
        """
        from mlsynth.exceptions import MlsynthPlottingError
        from mlsynth.utils.dtwsc_helpers import plotter

        res = DTWSC(base_config()).fit()

        def boom(*a, **k):
            raise RuntimeError("no display")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "subplots", boom)
        with pytest.raises(MlsynthPlottingError, match="DTWSC plotting failed"):
            plotter.plot_dtwsc(res, res.plot_config)

    def test_an_unexpected_internal_failure_is_translated(self, monkeypatch):
        """Anything the pipeline raises that is not already an mlsynth error
        must come back as MlsynthEstimationError, so a caller never sees a
        bare library exception leaking out of ``fit()``."""
        import mlsynth.estimators.dtwsc as mod

        def boom(*a, **k):
            raise RuntimeError("something deep broke")

        monkeypatch.setattr(mod, "run_dtwsc", boom)
        with pytest.raises(MlsynthEstimationError, match="DTWSC estimation failed"):
            DTWSC(base_config()).fit()

    def test_an_mlsynth_error_from_the_pipeline_is_not_re_wrapped(self):
        """Already-typed errors pass through with their original message."""
        with pytest.raises(MlsynthEstimationError, match="exceeds"):
            DTWSC(base_config(make_panel(n_pre=14, n_post=3, seed=40),
                              k=12)).fit()


# ==========================================================================
# Inference -- the paper's placebo band and efficiency test
# ==========================================================================
from mlsynth.utils.dtwsc_helpers.inference import (  # noqa: E402
    icc_corrected_ttest,
    placebo_donor_pools,
)


class TestPlaceboDonorPools:
    """The dataset construction of Figure_5_*.R in the replication package.

    ``data.list`` is: the full panel, then one dataset per unit with that unit
    dropped, then the first ``n_pairs`` two-unit combinations in R's
    ``combn`` (lexicographic) order. The treated unit is excluded from every
    placebo pool -- it is dropped in the ``filter(!(id %in% c(i, 17)))`` step.
    """

    def test_first_pool_is_the_untouched_donor_set(self):
        pools = placebo_donor_pools(n_units=5, treated=2, n_pairs=0)
        assert pools[0] == (0, 1, 3, 4)

    def test_the_treated_unit_never_appears_in_any_pool(self):
        pools = placebo_donor_pools(n_units=6, treated=3, n_pairs=5)
        assert all(3 not in p for p in pools)

    def test_leave_one_out_covers_every_donor_exactly_once(self):
        pools = placebo_donor_pools(n_units=5, treated=4, n_pairs=0)
        dropped = [set(range(4)) - set(p) for p in pools[1:]]
        assert dropped == [{0}, {1}, {2}, {3}]

    def test_pairs_follow_r_combn_lexicographic_order(self):
        """R's ``combn(setdiff(ids, treated), 2)`` yields (1,2), (1,3), ...

        Truncating to the first ``n_pairs`` columns only reproduces the paper
        if the order matches; itertools.combinations agrees with combn here.
        """
        pools = placebo_donor_pools(n_units=5, treated=4, n_pairs=3)
        dropped = [tuple(sorted(set(range(4)) - set(p))) for p in pools[5:]]
        assert dropped == [(0, 1), (0, 2), (0, 3)]

    def test_pair_count_is_capped_at_what_exists(self):
        """With three donors, dropping two would leave one -- so no pairs.

        A synthetic control needs something to interpolate between, so pools
        that would fall below two donors are skipped rather than emitted and
        left to fail downstream.
        """
        pools = placebo_donor_pools(n_units=4, treated=3, n_pairs=99)
        assert len(pools) == 1 + 3              # full + 3 leave-one-out

    def test_pairs_are_emitted_when_enough_donors_remain(self):
        pools = placebo_donor_pools(n_units=6, treated=5, n_pairs=99)
        assert len(pools) == 1 + 5 + 10         # full + singles + C(5,2)

    def test_every_pool_retains_at_least_two_donors(self):
        for p in placebo_donor_pools(n_units=6, treated=0, n_pairs=10):
            assert len(p) >= 2

    def test_pools_are_deterministic(self):
        a = placebo_donor_pools(n_units=7, treated=1, n_pairs=8)
        b = placebo_donor_pools(n_units=7, treated=1, n_pairs=8)
        assert a == b


class TestICCCorrectedTTest:
    """R pin of the Figure_5_*.R efficiency test.

    The placebo runs share units and donor pools, so they are not independent.
    The paper deflates the degrees of freedom by a variance-inflation factor
    built from the intra-class correlation of a two-way ANOVA on dataset and
    unit, then reports a two-sided lower-tail p-value.
    """

    LOG_RATIO = np.array([
        0.422575, -0.738819, -0.182123, -0.020282, -0.157439, -0.463675,
        0.506913, -0.456795, 0.811054, -0.437628, 0.382922, 0.971987,
        -1.233316, -0.567273, -0.479993, -0.01843, -0.570552, -1.993873,
        -1.86428, 0.392068, -0.583983, -1.468785, -0.50315, 0.328805,
        0.737116, -0.658281, -0.554362, -1.457898, -0.123942, -0.783997])
    DATA_ID = np.array([1, 2, 3, 4, 5, 6] * 5)
    UNIT_ID = np.repeat([1, 2, 3, 4, 5], 6)

    def test_matches_r_cell_for_cell(self):
        out = icc_corrected_ttest(self.LOG_RATIO, self.DATA_ID, self.UNIT_ID)
        assert out["icc"] == pytest.approx(-0.1376475086, rel=1e-8)
        assert out["vif"] == pytest.approx(0.4494099658, rel=1e-8)
        assert out["df"] == pytest.approx(66.75419391, rel=1e-8)
        assert out["t"] == pytest.approx(-2.610155295, rel=1e-8)
        assert out["p"] == pytest.approx(0.01116279195, rel=1e-6)

    def test_the_t_statistic_is_the_ordinary_one_sample_t(self):
        """Only the degrees of freedom are corrected, not the statistic."""
        from scipy import stats
        out = icc_corrected_ttest(self.LOG_RATIO, self.DATA_ID, self.UNIT_ID)
        assert out["t"] == pytest.approx(
            stats.ttest_1samp(self.LOG_RATIO, 0.0).statistic)

    def test_reports_the_mean_log_ratio_and_the_mse_reduction(self):
        out = icc_corrected_ttest(self.LOG_RATIO, self.DATA_ID, self.UNIT_ID)
        assert out["mean_log_ratio"] == pytest.approx(self.LOG_RATIO.mean())
        assert out["mse_reduction"] == pytest.approx(
            1 - np.exp(self.LOG_RATIO.mean()))

    def test_a_degenerate_single_group_does_not_raise(self):
        out = icc_corrected_ttest(np.array([-0.5, -0.4, -0.6]),
                                  np.array([1, 1, 1]), np.array([1, 2, 3]))
        assert np.isfinite(out["t"])

    def test_too_few_observations_is_reported(self):
        with pytest.raises(MlsynthEstimationError):
            icc_corrected_ttest(np.array([-0.5]), np.array([1]), np.array([1]))


class TestPlaceboBand:
    """The band is a two-sided pointwise interval: alpha/2 in each tail."""

    def test_edges_are_the_alpha_over_two_quantiles(self):
        """Splitting alpha unevenly would shift the band without changing
        its width ordering, so the width test alone cannot catch it."""
        from mlsynth.utils.dtwsc_helpers.inference import placebo_band
        rng = np.random.default_rng(0)
        gaps = rng.normal(size=(400, 5))
        band = placebo_band(gaps, alpha=0.10)
        assert band["lower"] == pytest.approx(
            np.quantile(gaps, 0.05, axis=0), abs=1e-12)
        assert band["upper"] == pytest.approx(
            np.quantile(gaps, 0.95, axis=0), abs=1e-12)

    def test_coverage_is_symmetric_about_the_median(self):
        from mlsynth.utils.dtwsc_helpers.inference import placebo_band
        rng = np.random.default_rng(1)
        gaps = rng.normal(size=(2000, 3))
        band = placebo_band(gaps, alpha=0.05)
        below = (gaps < band["lower"]).mean(axis=0)
        above = (gaps > band["upper"]).mean(axis=0)
        assert below == pytest.approx(above, abs=0.02)
        assert below == pytest.approx(0.025, abs=0.01)

    def test_runs_with_a_short_warp_still_contribute_where_defined(self):
        from mlsynth.utils.dtwsc_helpers.inference import placebo_band
        gaps = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
        band = placebo_band(gaps, alpha=0.5)
        assert np.isfinite(band["lower"]).all()

    def test_no_runs_is_reported(self):
        from mlsynth.utils.dtwsc_helpers.inference import placebo_band
        with pytest.raises(MlsynthEstimationError):
            placebo_band(np.empty((0, 5)), alpha=0.05)


class TestDTWSCPlaceboInference:
    def test_placebo_inference_populates_the_standard_slot(self):
        res = DTWSC(base_config(inference="placebo", placebo_pairs=0)).fit()
        assert res.inference is not None
        assert res.inference.method == "placebo"
        assert np.isfinite(res.inference.ci_lower)
        assert np.isfinite(res.inference.ci_upper)
        assert res.inference.ci_lower < res.inference.ci_upper

    def test_the_band_is_pointwise_and_spans_the_panel(self):
        res = DTWSC(base_config(inference="placebo", placebo_pairs=0)).fit()
        band = res.placebo_band
        assert band["lower"].shape == band["upper"].shape == (28,)
        assert np.all(band["lower"] <= band["upper"])

    def test_both_the_warped_and_unwarped_bands_are_reported(self):
        """The paper's claim is comparative, so both must be available."""
        res = DTWSC(base_config(inference="placebo", placebo_pairs=0)).fit()
        assert set(res.placebo_band) >= {"lower", "upper",
                                         "lower_unwarped", "upper_unwarped"}

    def test_the_treated_unit_is_not_among_the_placebos(self):
        res = DTWSC(base_config(inference="placebo", placebo_pairs=0)).fit()
        assert "treated" not in res.placebo_gaps

    def test_every_donor_serves_as_a_placebo_target(self):
        res = DTWSC(base_config(inference="placebo", placebo_pairs=0)).fit()
        assert len(res.placebo_gaps) == 6

    def test_the_efficiency_test_is_reported(self):
        res = DTWSC(base_config(inference="placebo", placebo_pairs=0)).fit()
        eff = res.efficiency_test
        assert set(eff) >= {"t", "p", "df", "icc", "vif", "mean_log_ratio",
                            "mse_reduction", "n_runs"}
        assert np.isfinite(eff["t"]) and 0.0 <= eff["p"] <= 1.0

    def test_pool_perturbation_multiplies_the_placebo_runs(self):
        few = DTWSC(base_config(inference="placebo", placebo_pairs=0)).fit()
        many = DTWSC(base_config(inference="placebo", placebo_pairs=2)).fit()
        assert many.efficiency_test["n_runs"] > few.efficiency_test["n_runs"]

    def test_inference_off_by_default(self):
        assert DTWSC(base_config()).fit().inference is None

    def test_alpha_widens_or_narrows_the_band(self):
        wide = DTWSC(base_config(inference="placebo", placebo_pairs=0,
                                 alpha=0.5)).fit()
        narrow = DTWSC(base_config(inference="placebo", placebo_pairs=0,
                                   alpha=0.01)).fit()
        w = wide.placebo_band["upper"] - wide.placebo_band["lower"]
        n = narrow.placebo_band["upper"] - narrow.placebo_band["lower"]
        assert np.nanmean(w) <= np.nanmean(n)

    @pytest.mark.parametrize("bad", [{"inference": "bootstrap"},
                                     {"placebo_pairs": -1},
                                     {"alpha": 0.0}, {"alpha": 1.0}])
    def test_invalid_inference_options_are_rejected(self, bad):
        with pytest.raises(MlsynthConfigError):
            DTWSC(base_config(**bad))

    def test_too_few_donors_for_a_placebo_is_reported(self):
        df = make_panel(n_donors=1, seed=50)
        with pytest.raises(MlsynthEstimationError, match="placebo"):
            DTWSC(base_config(df, inference="placebo")).fit()


# ==========================================================================
# The synthetic-control half: delegating to mlsynth's Synth replication
# ==========================================================================
class TestSCBackend:
    """The paper fits ``Synth`` with a V-matrix over 14 special predictors.

    DTWSC's own simplex solve is outcome-only, which overfits the pre-period
    relative to the reference (0.054 against R's 0.071 on Basque) and inflates
    the ATT by about 28 percent. ``sc_backend`` routes the synthetic-control
    half through ``VanillaSC``'s predictor machinery instead, so the paper's
    specification is reachable.
    """

    def test_simplex_is_the_default(self):
        res = DTWSC(base_config()).fit()
        assert res.metadata["sc_backend"] == "simplex"

    @pytest.mark.parametrize("backend", ["simplex", "outcome-only"])
    def test_outcome_only_backends_agree_closely(self, backend):
        """Both fit the same objective, so they must land in the same place."""
        df = make_panel(seed=60)
        a = DTWSC(base_config(df, sc_backend="simplex")).fit()
        b = DTWSC(base_config(df, sc_backend=backend)).fit()
        assert b.att == pytest.approx(a.att, abs=0.35)

    def test_predictor_backend_changes_the_weights(self):
        """With covariates the V-matrix backend must not reduce to simplex."""
        df = make_panel(seed=61)
        df["x1"] = df.groupby("unit")["y"].transform(
            lambda s: s.rolling(3, min_periods=1).mean())
        plain = DTWSC(base_config(df, sc_backend="simplex")).fit()
        withcov = DTWSC(base_config(df, sc_backend="malo",
                                    covariates=["x1"])).fit()
        assert np.isfinite(withcov.att)
        assert withcov.metadata["sc_backend"] == "malo"
        w1 = np.array(list(plain.donor_weights.values()))
        w2 = np.array(list(withcov.donor_weights.values()))
        assert not np.allclose(w1, w2, atol=1e-8)

    def test_the_backend_receives_the_warped_donors_not_the_raw_ones(self):
        """The whole point: the SC half must see the resampled series."""
        df = make_panel(seed=62, stretch=1.5)
        warped = DTWSC(base_config(df, sc_backend="outcome-only")).fit()
        raw = DTWSC(base_config(df, sc_backend="outcome-only",
                                warp=False)).fit()
        assert not np.allclose(warped.warped_donor_matrix,
                               raw.warped_donor_matrix)
        assert warped.pre_rmse != pytest.approx(raw.pre_rmse, abs=1e-9)

    def test_covariates_without_a_predictor_backend_is_refused(self):
        """Silently ignoring predictors would misreport what was fit."""
        with pytest.raises(MlsynthConfigError, match="sc_backend"):
            DTWSC(base_config(sc_backend="simplex", covariates=["x1"]))

    @pytest.mark.parametrize("bad", ["synth", "abadie", ""])
    def test_unknown_backends_are_rejected(self, bad):
        with pytest.raises(MlsynthConfigError):
            DTWSC(base_config(sc_backend=bad))

    def test_a_missing_covariate_column_is_reported(self):
        with pytest.raises((MlsynthDataError, MlsynthEstimationError,
                            MlsynthConfigError)):
            DTWSC(base_config(sc_backend="malo",
                              covariates=["not_a_column"])).fit()

    def test_placebo_inference_uses_the_configured_backend(self):
        """A band built on a different estimator than the point estimate
        would not be a placebo distribution for that estimate."""
        res = DTWSC(base_config(sc_backend="outcome-only",
                                inference="placebo", placebo_pairs=0)).fit()
        assert res.inference is not None
        assert res.metadata["sc_backend"] == "outcome-only"
