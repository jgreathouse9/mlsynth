"""Robustness tests for SPILLSYNTH.

Asserts the source-level guards and warnings added beyond Pydantic's
config validation:

* hard errors (``MlsynthDataError`` / ``MlsynthEstimationError`` /
  ``ValueError``) for inputs the estimator cannot proceed on;
* ``RuntimeWarning`` for "ran but the answer is suspect" situations
  (ill-conditioned identification matrix, near-singular ISCM
  cross-weight system);
* a ``warnings_as_errors`` context manager that turns every warning into
  an error, used to confirm the happy path is warning-free.

These complement the Pydantic-level checks already exercised in
test_spillsynth.py / test_spillsynth_iscm.py / test_spillsynth_grossi.py.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import SPILLSYNTH
from mlsynth.exceptions import (
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.spillsynth_helpers import (
    prepare_spillsynth_inputs,
    run_cd,
    run_iscm,
)
from mlsynth.utils.spillsynth_helpers.cd.estimation import (
    build_M,
    sp_estimate,
    sp_estimate_weighted,
)
from mlsynth.utils.spillsynth_helpers.cd.pipeline import run_cd as _run_cd


@contextlib.contextmanager
def warnings_as_errors():
    """Treat every warning as an error inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# --------------------------------------------------------------------------- #
# Panel builders
# --------------------------------------------------------------------------- #
def _panel(*, N=8, T=40, T0=30, treatment=-3.0, spillover=1.5,
           spillover_idx=1, seed=0):
    rng = np.random.default_rng(seed)
    loadings = rng.uniform(0.5, 1.5, size=N)
    f = np.cumsum(rng.standard_normal(T)) * 0.4 + 0.05 * np.arange(T)
    intercept = rng.uniform(-1, 1, size=N)
    Y = intercept[:, None] + np.outer(loadings, f) + 0.1 * rng.standard_normal((N, T))
    Y[0, T0:] += treatment
    if spillover_idx is not None:
        Y[spillover_idx, T0:] += spillover
    D = np.zeros((N, T))
    D[0, T0:] = 1
    rows = [
        {"unit": f"u{i}", "year": t, "y": float(Y[i, t]), "treat": int(D[i, t])}
        for i in range(N) for t in range(T)
    ]
    return pd.DataFrame(rows)


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="treat", unitid="unit", time="year",
                display_graphs=False)
    base.update(kw)
    return base


def _inputs(df, **kw):
    base = dict(outcome="y", treat="treat", unitid="unit", time="year")
    base.update(kw)
    return prepare_spillsynth_inputs(df=df, **base)


# --------------------------------------------------------------------------- #
# Hard errors: singular / ill-posed identification matrix A' M A
# --------------------------------------------------------------------------- #
class TestAMASingular:
    def test_sp_estimate_singular_AMA_raises_clear_error(self):
        # A with two identical columns -> A' M A rank-deficient -> singular.
        N, T1 = 5, 3
        B = np.zeros((N, N))
        M = build_M(B)
        A = np.ones((N, 2))           # rank-1 A => singular A' M A
        with pytest.raises(MlsynthEstimationError, match="not identified"):
            sp_estimate(np.random.default_rng(0).standard_normal((N, T1)),
                        a=np.zeros(N), B=B, M=M, A=A)

    def test_sp_estimate_weighted_singular_raises(self):
        N, T1 = 5, 3
        B = np.zeros((N, N))
        A = np.ones((N, 2))
        W = np.eye(N)
        with pytest.raises(MlsynthEstimationError, match="not identified"):
            sp_estimate_weighted(np.zeros((N, T1)), a=np.zeros(N), B=B, A=A, W=W)

    def test_all_units_affected_warns_or_raises_through_estimator(self):
        # Declaring every control affected makes A' M A (near-)singular.
        # Per_unit A is square + full-rank so the solve completes but is
        # numerically degenerate: we must either warn (ill-conditioned) or
        # raise -- never silently return an O(1)-conditioned garbage answer.
        df = _panel(N=8)
        all_controls = [f"u{i}" for i in range(1, 8)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = SPILLSYNTH(_cfg(df, affected_units=all_controls)).fit()
            except (MlsynthEstimationError, MlsynthDataError):
                return
        # Ran: cond must be flagged huge and a RuntimeWarning emitted.
        assert res.cd.cond_AMA > 1e8
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)


# --------------------------------------------------------------------------- #
# Warnings: ill-conditioned A' M A (opt-in via warn=True at top-level)
# --------------------------------------------------------------------------- #
class TestAMAIllConditioned:
    def test_warns_when_AMA_ill_conditioned(self):
        # Build B so that (I - B) collapses two donor directions, leaving
        # A' M A nearly singular but invertible.
        N, T1 = 4, 2
        B = np.zeros((N, N))
        M = build_M(B)
        # Two near-parallel A columns => huge condition number, still invertible.
        A = np.zeros((N, 2))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 1] = 1e-9
        # Force ill-conditioning by scaling: make second column tiny.
        A[:, 1] *= 1e-7
        with pytest.warns(RuntimeWarning, match="ill-conditioned"):
            sp_estimate(np.zeros((N, T1)), a=np.zeros(N), B=B, M=M, A=A, warn=True)

    def test_no_warn_when_warn_false(self):
        N, T1 = 4, 2
        B = np.zeros((N, N))
        M = build_M(B)
        A = np.zeros((N, 2)); A[0, 0] = 1.0; A[1, 1] = 1e-12
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # warn=False (default) must stay silent even when ill-conditioned.
            sp_estimate(np.zeros((N, T1)), a=np.zeros(N), B=B, M=M, A=A)


# --------------------------------------------------------------------------- #
# ISCM cross-weight system Omega: singular -> error, near-singular -> warning
# --------------------------------------------------------------------------- #
class TestISCMOmega:
    def test_singular_omega_raises(self):
        # Treated + 2 affected, no clean control: each unit's SC is a convex
        # combination of the other two -> rows of the cross matrix sum to 1
        # -> Omega singular.
        rng = np.random.default_rng(5)
        N, T, T0 = 3, 40, 30
        Y = np.zeros((N, T))
        for i in range(N):
            Y[i] = 10 * i + np.cumsum(rng.standard_normal(T)) * 0.3
        Y[0, T0:] += -3.0
        Y[1, T0:] += 1.5
        rows = [
            {"unit": f"u{i}", "year": t, "y": float(Y[i, t]),
             "treat": int(i == 0 and t >= T0)}
            for i in range(N) for t in range(T)
        ]
        df = pd.DataFrame(rows)
        inp = _inputs(df, affected_units=["u1", "u2"])
        with pytest.raises(MlsynthEstimationError, match="Omega is singular"):
            run_iscm(inp)

    def test_near_singular_omega_warns(self, monkeypatch):
        # Drive the cross-weight system to a tiny-but-nonzero determinant so
        # the de-contamination is flagged as numerically unstable.
        import mlsynth.utils.spillsynth_helpers.iscm.pipeline as mod

        real = mod.build_omega

        def _near_singular(cross):
            om = real(cross)
            # Nudge the off-diagonals so |det| lands just below the warn
            # threshold (1e-6) without being exactly singular.
            if om.shape == (2, 2):
                om[0, 1] = -(1.0 - 1e-7)
                om[1, 0] = -(1.0 - 1e-7)
            return om

        monkeypatch.setattr(mod, "build_omega", _near_singular)
        inp = _inputs(_panel(), affected_units=["u1"])
        # p == 1 -> 2x2 omega; the nudge makes det ~ 2e-7 < 1e-6.
        with pytest.warns(RuntimeWarning, match="near-singular"):
            run_iscm(inp)


# --------------------------------------------------------------------------- #
# run_cd weighting guard (ValueError on bad weighting string)
# --------------------------------------------------------------------------- #
class TestRunCdWeighting:
    def test_bad_weighting_rejected(self):
        inp = _inputs(_panel(), affected_units=["u1"])
        with pytest.raises(ValueError, match="weighting must be"):
            run_cd(inp, weighting="bogus")


# --------------------------------------------------------------------------- #
# Estimator-boundary data validation (setup.py)
# --------------------------------------------------------------------------- #
class TestSetupBoundaries:
    def test_missing_column_raises(self):
        df = _panel()
        with pytest.raises(MlsynthDataError, match="required column"):
            _inputs(df, outcome="nope")

    def test_unknown_spillover_structure_raises(self):
        with pytest.raises(MlsynthDataError, match="unknown spillover_structure"):
            _inputs(_panel(), spillover_structure="bananas")

    def test_distance_decay_without_distances_raises(self):
        with pytest.raises(MlsynthDataError, match="requires"):
            _inputs(_panel(), spillover_structure="distance_decay")

    def test_homogeneous_without_affected_raises(self):
        with pytest.raises(MlsynthDataError, match="needs at least"):
            _inputs(_panel(), spillover_structure="homogeneous")

    def test_empty_after_dropping_nans_raises(self):
        df = _panel()
        df["y"] = np.nan
        with pytest.raises(MlsynthDataError, match="empty after dropping"):
            _inputs(df, affected_units=["u1"])

    def test_distance_decay_negative_distance_raises(self):
        with pytest.raises(MlsynthDataError, match="non-negative"):
            _inputs(_panel(), spillover_structure="distance_decay",
                    unit_distances={"u1": -1.0})

    def test_distance_decay_infinite_distance_raises(self):
        with pytest.raises(MlsynthDataError, match="finite"):
            _inputs(_panel(), spillover_structure="distance_decay",
                    unit_distances={"u1": np.inf})

    def test_distance_decay_unknown_label_raises(self):
        with pytest.raises(MlsynthDataError, match="not in the panel"):
            _inputs(_panel(), spillover_structure="distance_decay",
                    unit_distances={"ghost": 1.0})

    def test_distance_decay_non_dict_raises(self):
        # unit_distances must be a dict; a list trips the type guard.
        with pytest.raises(MlsynthDataError, match="must be a dict"):
            prepare_spillsynth_inputs(
                df=_panel(), outcome="y", treat="treat", unitid="unit",
                time="year", spillover_structure="distance_decay",
                unit_distances=[1.0, 2.0])

    def test_distance_decay_all_zero_weights_raises(self):
        # All declared distances effectively infinite (here, only ghosts
        # absent) -> every decay weight zero.
        with pytest.raises(MlsynthDataError, match="every declared decay weight"):
            _inputs(_panel(), spillover_structure="distance_decay",
                    unit_distances={"u1": np.float64(1e9)})


# --------------------------------------------------------------------------- #
# Happy path emits no warnings
# --------------------------------------------------------------------------- #
class TestHappyPathNoWarnings:
    def test_cd_happy_path_no_warning(self):
        with warnings_as_errors():
            res = SPILLSYNTH(_cfg(_panel(), affected_units=["u1"])).fit()
            assert np.isfinite(res.att)

    def test_cd_no_affected_no_warning(self):
        with warnings_as_errors():
            res = SPILLSYNTH(_cfg(_panel(spillover_idx=None))).fit()
            assert np.isfinite(res.att)
