"""Ridge lambda cross-validation against augsynth's own CV curve.

The ridge penalty is selected by leave-one-pre-period-out CV and a 1-SE rule.
Both are cheap to get subtly wrong, and a wrong penalty is not an error -- it is
a plausible fit at the wrong amount of shrinkage, which then propagates into
every effect and every interval built on it.

Two panels, chosen because one of them cannot detect the defects this file pins:

``kansas``
    T0 = 89 pre-periods against J = 49 donors. The selected penalty here is
    insensitive to the defects below -- it comes out identical whether the CV
    loop runs 88 or 89 folds and whether the standard error uses the sample or
    the population denominator. A suite that tested only this panel would call
    the CV correct while it was not, which is what happened.

``song``
    T0 = 25 against J = 37, from Song et al. (2023). Its final pre-treatment
    period sits on the seasonal ramp into the heating season, so that fold's
    held-out error is about nine times the mean of the rest. Whether it counts
    as a fold at all therefore moves the entire curve, and the selected penalty
    with it -- 11.35 against augsynth's 28.52 before the fix.

Gold from a live augsynth 0.2.0 run at its pinned commit; see
``benchmarks/reference/ascm_ridge_cv/reference.R``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_GOLD = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
         / "ascm_ridge_cv")
_KANSAS = Path(__file__).resolve().parents[2] / "basedata" / "kansas_ascm.csv"


def _selected(slug):
    """``(lambda, T0, J)`` from the reference's provenance dump."""
    txt = (_GOLD / f"gold_selected_{slug}.txt").read_text().splitlines()
    got = dict(line.split(": ", 1) for line in txt if ": " in line)
    return float(got["lambda"]), int(got["T0"]), int(got["J"])


def _kansas_pre():
    d = pd.read_csv(_KANSAS)
    piv = d.pivot(index="fips", columns="year_qtr",
                  values="lngdpcapita").sort_index()
    times = np.array(sorted(d["year_qtr"].unique()))
    n_pre = int((times < 2012.25).sum())
    Y = piv.to_numpy()
    trt = piv.index.to_numpy() == 20
    return Y[trt][0][:n_pre], Y[~trt][:, :n_pre].T


def _song_pre():
    d = pd.read_csv(_GOLD / "song_pm25_cell.csv")
    d["date"] = pd.to_datetime(d["date"])
    piv = d.pivot(index="ID", columns="date", values="PM2.5wn").sort_index()
    n_pre = int((piv.columns < pd.Timestamp("2015-10-23")).sum())
    Y = piv.to_numpy()
    trt = piv.index.to_numpy() == "2+26 cities"
    return Y[trt][0][:n_pre], Y[~trt][:, :n_pre].T


PANELS = {"kansas": _kansas_pre, "song": _song_pre}


def _curve(slug):
    p = _GOLD / f"gold_curve_{slug}.csv"
    if not p.exists():  # pragma: no cover - the gold is committed
        pytest.skip(f"missing gold at {p}")
    return pd.read_csv(p).sort_values("lambda", ascending=False)


def _mlsynth_cv(slug):
    from mlsynth.utils.bilevel.ridge_augment import (
        build_matching, generate_lambdas, cross_validate, simplex_qp)
    y_pre, Y0_pre = PANELS[slug]()
    B, A = build_matching(y_pre, Y0_pre)
    lam = generate_lambdas(B)
    lams, mean, se = cross_validate(simplex_qp, B, A, lam, holdout_len=1)
    order = np.argsort(-lams)
    return lams[order], mean[order], se[order]


class TestPanelsDifferAsIntended:
    """The two panels must actually straddle the T0 vs J boundary."""

    def test_kansas_has_more_pre_periods_than_donors(self):
        _, t0, j = _selected("kansas")
        assert t0 > j

    def test_song_has_fewer_pre_periods_than_donors(self):
        _, t0, j = _selected("song")
        assert t0 < j

    def test_the_song_panels_final_fold_is_an_outlier(self):
        """The property that makes this panel diagnostic.

        If the last pre-period ever stops being an outlier, this file quietly
        loses its power to detect a fold-count defect, so the property is
        asserted rather than assumed.
        """
        from mlsynth.utils.bilevel.ridge_augment import (
            build_matching, generate_lambdas, simplex_qp, solve_ridge_path)
        y_pre, Y0_pre = _song_pre()
        B, A = build_matching(y_pre, Y0_pre)
        lam = generate_lambdas(B)
        m = B.shape[0]
        per_fold = []
        for i in range(m):
            keep = np.ones(m, dtype=bool)
            keep[i] = False
            W = simplex_qp(B[keep], A[keep])
            W_aug = W[None, :] + solve_ridge_path(A[keep], B[keep], W, lam)
            resid = A[~keep][None, :] - W_aug @ B[~keep].T
            per_fold.append(float(np.sum(resid ** 2)))
        per_fold = np.asarray(per_fold)
        assert per_fold[-1] > 5.0 * per_fold[:-1].mean()


class TestGridAndRule:
    """The parts that were already right, pinned so they stay right."""

    @pytest.mark.parametrize("slug", sorted(PANELS))
    def test_grid_matches_augsynth(self, slug):
        """``create_lambda_list``: 21 points for ``n_lambda=20``.

        R's ``seq(0:n_lambda) - 1`` yields exponents ``0..n_lambda``, so the grid
        has ``n_lambda + 1`` entries. Reading that as ``n_lambda`` would shift
        every point on the grid.
        """
        from mlsynth.utils.bilevel.ridge_augment import (build_matching,
                                                         generate_lambdas)
        y_pre, Y0_pre = PANELS[slug]()
        B, _ = build_matching(y_pre, Y0_pre)
        got = np.sort(generate_lambdas(B))
        exp = np.sort(_curve(slug)["lambda"].to_numpy())
        assert got.size == exp.size == 21
        np.testing.assert_allclose(got, exp, rtol=1e-12, atol=0)

    def test_one_se_rule_takes_the_largest_within_one_se(self):
        """augsynth ``choose_lambda``: ``max(lambdas[err <= min + se[argmin]])``.

        Pinned including the boundary, since ``<=`` rather than ``<`` decides
        which lambda wins when a curve is flat.
        """
        from mlsynth.utils.bilevel.ridge_augment import best_lambda
        lambdas = np.array([100.0, 10.0, 1.0, 0.1])
        errors = np.array([5.0, 3.0, 2.0, 2.5])
        se = np.array([1.0, 1.0, 1.0, 1.0])
        # min error 2.0 at lambda=1, +1 se -> threshold 3.0; 10.0 qualifies at
        # exactly the boundary and is larger, so it wins.
        assert best_lambda(lambdas, errors, se, min_1se=True) == 10.0
        assert best_lambda(lambdas, errors, se, min_1se=False) == 1.0


class TestCVCurve:
    """The CV loop itself -- where both defects lived."""

    @pytest.mark.parametrize("slug", sorted(PANELS))
    def test_error_curve_matches_augsynth(self, slug):
        """augsynth loops ``i in 1:(ncol(X_c) - holdout_length)``.

        With ``holdout_length = 1`` that is ``1..T0-1``: the final pre-period is
        never held out. mlsynth iterated ``T0`` folds, one more than the
        reference. On kansas that changed nothing detectable; on song it moved
        the curve by up to 22 and the selected penalty from 28.52 to 11.35.
        """
        lams, mean, se = _mlsynth_cv(slug)
        g = _curve(slug)
        np.testing.assert_allclose(mean, g["err"].to_numpy(), rtol=0, atol=1e-5)

    @pytest.mark.parametrize("slug", sorted(PANELS))
    def test_standard_error_matches_augsynth(self, slug):
        """``sd(x) / sqrt(length(x))`` -- R's ``sd`` is the SAMPLE sd.

        numpy's ``.std()`` defaults to the population denominator, so the SE came
        out too small by ``sqrt((n-1)/n)``. It feeds the 1-SE threshold directly,
        so it can change which penalty is chosen.
        """
        lams, mean, se = _mlsynth_cv(slug)
        g = _curve(slug)
        np.testing.assert_allclose(se, g["se"].to_numpy(), rtol=0, atol=1e-6)

    @pytest.mark.parametrize("slug", sorted(PANELS))
    def test_selected_penalty_matches_augsynth(self, slug):
        from mlsynth.utils.bilevel.ridge_augment import best_lambda
        lams, mean, se = _mlsynth_cv(slug)
        expected, _, _ = _selected(slug)
        assert best_lambda(lams, mean, se, min_1se=True) == pytest.approx(
            expected, rel=1e-9)

    @pytest.mark.parametrize("slug", sorted(PANELS))
    def test_end_to_end_penalty_through_ridge_augment_weights(self, slug):
        """The public path, not just the helper."""
        from mlsynth.utils.bilevel.ridge_augment import ridge_augment_weights
        y_pre, Y0_pre = PANELS[slug]()
        expected, _, _ = _selected(slug)
        got = ridge_augment_weights(y_pre, Y0_pre).lambda_
        assert got == pytest.approx(expected, rel=1e-9)

    def test_fold_count_is_one_short_of_the_pre_period_count(self):
        """Pinned directly, so the off-by-one cannot silently return."""
        from mlsynth.utils.bilevel.ridge_augment import _HoldoutSplitter
        B = np.arange(50, dtype=float).reshape(10, 5)
        A = np.arange(10, dtype=float)
        assert len(list(_HoldoutSplitter(B, A, holdout_len=1))) == 9
        assert len(list(_HoldoutSplitter(B, A, holdout_len=3))) == 7

    def test_the_final_pre_period_is_never_held_out(self):
        """The substance of the off-by-one, stated as behaviour."""
        from mlsynth.utils.bilevel.ridge_augment import _HoldoutSplitter
        B = np.arange(40, dtype=float).reshape(8, 5)
        A = np.arange(8, dtype=float)
        held = [float(a_v[0]) for _, _, _, a_v in
                _HoldoutSplitter(B, A, holdout_len=1)]
        assert held == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert 7.0 not in held
