"""What ``dataprep`` refuses, and that it says so.

``dataprep`` is the library's ingestion gate: invariant 2 routes every
estimator through it, and ``CLAUDE.md`` refuses Postel's law on its account
because a lenient validator turns a malformed panel into a number that looks
like an estimate. The refusals are therefore the part of the module that earns
the rest of it.

They were also the untested part. Coverage over the dataprep-focused suite left
seven ``raise MlsynthDataError`` statements unexecuted, and a mutation run
deleting five of them scored 0/5 killed -- every refusal could be removed and
the suite stayed green. Each test below pins one of those five, and the
``dataprep-refusals`` target in ``tools/mutation/targets.toml`` carries the
mutant that proves it bites.

Two of the seven are not tested here because they cannot be reached. ``logictreat``
derives ``treated_indices`` from ``np.any(matrix == 1, axis=0)``, so a unit in
that list has at least one treated period by construction and the two "has no
post-treatment period" safeguards cannot fire. An exhaustive search over every
binary panel up to 3x3, and every ternary panel with NaN up to 2x2, reaches
neither. They carry ``# pragma: no cover`` and state that reason.

What each test asserts is the translated type and the message, not the
traceback: the contract callers rely on is that a bad panel raises
``MlsynthDataError`` and names what was wrong.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.datautils import build_covariate_matrix, logictreat


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cov_panel(n_units=3, n_periods=4):
    """A balanced panel with one covariate, in dataprep's column convention."""
    rows = [
        {"unit": u, "time": t, "outcome": float(u * 10 + t), "x": float(u + t)}
        for u in range(n_units)
        for t in range(n_periods)
    ]
    return pd.DataFrame(rows)


def _unit_order(df):
    return list(pd.unique(df["unit"]))


# --------------------------------------------------------------------------- #
# logictreat: the treatment matrix itself
# --------------------------------------------------------------------------- #
class TestTreatmentMatrixRefusals:
    """Refusals about the shape of treatment, before any outcome is read."""

    def test_non_binary_treatment_is_refused(self):
        """A 2 in the treatment column is not a treatment status.

        Everything downstream counts ``== 1``, so an admitted 2 does not raise
        later -- it drops out of the treated mask and the unit becomes neither
        treated nor a clean donor. The refusal has to happen here or not at all.
        """
        matrix = np.array([[0], [2], [2]], dtype=float)
        with pytest.raises(MlsynthDataError,
                           match="Treatment indicator must be a binary variable"):
            logictreat(matrix)

    def test_negative_treatment_value_is_refused(self):
        """The same guard, from the other side of zero."""
        matrix = np.array([[0], [-1], [1]], dtype=float)
        with pytest.raises(MlsynthDataError,
                           match="Treatment indicator must be a binary variable"):
            logictreat(matrix)

    def test_nan_is_not_treated_as_a_non_binary_value(self):
        """NaN is missingness, not an invalid state, and is excluded by design.

        Pinned because the binary check is written to exclude NaN explicitly;
        a future tightening that forgets to would break panels that legitimately
        carry missing treatment cells.
        """
        matrix = np.array([[0.0], [np.nan], [1.0], [1.0]])
        out = logictreat(matrix)
        assert out["Num Treated Units"] == 1

    def test_unsustained_treatment_is_refused_for_a_cohort_unit(self):
        """Treatment that switches back off, in the multiple-treated branch.

        The single-treated branch has its own check and its own message, and
        that one was already tested. This is the cohort path: ``post_periods``
        is counted from the first 1 to the end of the panel, so a unit that
        reverts has untreated periods billed as treated.
        """
        matrix = np.array([
            [0, 0],
            [1, 1],
            [0, 1],   # unit 0 reverts
            [1, 1],
        ], dtype=float)
        with pytest.raises(MlsynthDataError,
                           match=r"Treatment is not sustained for unit 0"):
            logictreat(matrix)

    def test_sustained_cohort_treatment_is_admitted(self):
        """The companion: staggered adoption is not what the check refuses."""
        matrix = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 1],
        ], dtype=float)
        out = logictreat(matrix)
        assert out["Num Treated Units"] == 2
        assert np.array_equal(out["First Treat Periods"], np.array([1, 2]))


# --------------------------------------------------------------------------- #
# build_covariate_matrix: the covariate request
# --------------------------------------------------------------------------- #
class TestCovariateRefusals:
    """Refusals about what the caller asked for, before any aggregation."""

    def test_missing_covariate_is_refused_by_name(self):
        """A mistyped covariate names itself in the error.

        Without this the name reaches ``_wide_pivot`` and pandas raises its own
        ``KeyError``: a real failure, but untranslated, so a caller narrowing on
        ``MlsynthDataError`` sees an exception it does not handle.
        """
        df = _cov_panel()
        with pytest.raises(MlsynthDataError, match=r"covariate 'nope' not present"):
            build_covariate_matrix(df, "unit", "time", ["nope"], 2, _unit_order(df))

    def test_missing_covariate_is_refused_even_beside_a_present_one(self):
        """The scan does not stop at the first column that happens to exist."""
        df = _cov_panel()
        with pytest.raises(MlsynthDataError, match=r"covariate 'nope' not present"):
            build_covariate_matrix(df, "unit", "time", ["x", "nope"], 2,
                                   _unit_order(df))

    def test_pre_mean_over_an_empty_window_is_refused(self):
        """Zero pre-periods is an empty slice, not a mean of nothing.

        Admitted, ``pivot.iloc[:0, :].mean()`` is NaN for every unit, so the
        covariate matrix comes back all-NaN and the fit runs on it.
        """
        df = _cov_panel()
        with pytest.raises(MlsynthDataError, match="requires at least one"):
            build_covariate_matrix(df, "unit", "time", ["x"], 0, _unit_order(df))

    def test_negative_pre_periods_is_refused(self):
        """The same guard below zero, where the slice is also empty."""
        df = _cov_panel()
        with pytest.raises(MlsynthDataError, match="requires at least one"):
            build_covariate_matrix(df, "unit", "time", ["x"], -1, _unit_order(df))

    def test_unknown_aggregation_is_refused_and_names_itself(self):
        """An aggregation the module does not implement is not silently replaced.

        The failure mode this prevents is the quiet one: the caller asks for an
        estimand, is given a different one, and nothing in the result says so.
        """
        df = _cov_panel()
        with pytest.raises(MlsynthDataError,
                           match=r"Unknown covariate_aggregation: 'post_mean'"):
            build_covariate_matrix(df, "unit", "time", ["x"], 2, _unit_order(df),
                                   aggregation="post_mean")

    def test_pre_mean_remains_the_working_default(self):
        """The companion: the valid request still returns the aggregation."""
        df = _cov_panel()
        cov, names, means, scales = build_covariate_matrix(
            df, "unit", "time", ["x"], 2, _unit_order(df), normalize=False)
        assert names == ("x",)
        assert cov.shape == (3, 1)
        # unit u has x = u + t, so the mean over t in {0, 1} is u + 0.5
        assert np.allclose(cov[:, 0], [0.5, 1.5, 2.5])
