"""Property tests for ``dataprep``'s three opt-in return contracts.

``test_datautils_properties.py`` covers the default single-treated contract.
Three other paths return different dictionaries and had no property coverage:

* **cohort mode**, taken whenever more than one unit is treated. It returns
  ``cohorts`` keyed by first-treatment time instead of the flat ``y`` /
  ``donor_matrix`` pair, and its donor pool is the never-treated units --
  every treated unit is excluded from every cohort's donors, not just from its
  own. A cohort that borrowed from a later-treated unit would be borrowing
  from a contaminated series, and nothing in the flat contract's tests looks
  at that.
* **covariates**, which add a per-unit predictor matrix and split it into
  donor and treated blocks by position. The split is a slice, so it is correct
  only if the unit ordering it assumes holds -- ``donors + [treated]``.
* **marex**, the Abadie-Zhou stacked predictor matrix, whose contract is an
  algebraic relation between four returned arrays.

The properties asserted are the ones a slice or a reindex silently breaks:
partitioning, ordering, and the arithmetic tying the marex outputs together.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.datautils import dataprep

_VALUE = st.floats(min_value=-1e3, max_value=1e3,
                   allow_nan=False, allow_infinity=False, width=64)


@st.composite
def panels(draw, min_units: int = 3, max_units: int = 7,
           n_treated: int = 1, with_covariates: bool = True):
    """A balanced panel with ``n_treated`` absorbing-treated units.

    Treated units get distinct adoption dates where possible, so cohort mode
    sees more than one cohort.
    """
    n_units = draw(st.integers(min_value=max(min_units, n_treated + 1),
                               max_value=max_units))
    n_periods = draw(st.integers(min_value=4, max_value=14))
    starts = draw(st.lists(st.integers(min_value=1, max_value=n_periods - 1),
                           min_size=n_treated, max_size=n_treated))

    size = n_units * n_periods
    y = np.asarray(draw(st.lists(_VALUE, min_size=size, max_size=size)))
    y = y.reshape(n_units, n_periods)
    x1 = np.asarray(draw(st.lists(_VALUE, min_size=size, max_size=size)))
    x1 = x1.reshape(n_units, n_periods)

    rows = []
    for u in range(n_units):
        start = starts[u] if u < n_treated else None
        for t in range(n_periods):
            row = {"unit": f"u{u}", "time": 2000 + t, "y": float(y[u, t]),
                   "treated": int(start is not None and t >= start)}
            if with_covariates:
                row["x1"] = float(x1[u, t])
                row["x2"] = float(u)          # time-invariant by construction
            rows.append(row)
    return pd.DataFrame(rows), n_units, n_periods, starts


def _prep(df, **kw):
    return dataprep(df, "unit", "time", "y", "treated", **kw)


# ---------------------------------------------------------------------------
# Cohort mode
# ---------------------------------------------------------------------------

@given(panel=panels(n_treated=2))
@settings(max_examples=120)
def test_cohorts_partition_the_treated_units(panel):
    """Every treated unit belongs to exactly one cohort, keyed by its date."""
    df, _, _, starts = panel
    out = _prep(df)
    assert "cohorts" in out and "y" not in out

    assigned = [u for c in out["cohorts"].values() for u in c["treated_units"]]
    assert sorted(assigned) == sorted(set(assigned))            # no duplicates
    assert set(assigned) == {"u0", "u1"}
    assert len(out["cohorts"]) == len(set(starts))


@given(panel=panels(n_treated=2))
@settings(max_examples=120)
def test_no_cohort_borrows_from_a_treated_unit(panel):
    """The donor pool is the never-treated units, for every cohort.

    A cohort treated later is still a treated unit, so using it as a donor for
    an earlier cohort would contaminate the counterfactual with the later
    unit's own post-treatment path.
    """
    df, _, _, _ = panel
    out = _prep(df)
    treated = {u for c in out["cohorts"].values() for u in c["treated_units"]}

    for cohort in out["cohorts"].values():
        assert treated.isdisjoint(cohort["donor_names"])


@given(panel=panels(n_treated=2))
@settings(max_examples=120)
def test_every_cohort_partitions_the_panel_in_time(panel):
    df, _, n_periods, _ = panel
    out = _prep(df)
    for start_time, cohort in out["cohorts"].items():
        assert cohort["pre_periods"] + cohort["post_periods"] == cohort["total_periods"]
        assert cohort["total_periods"] == n_periods
        assert cohort["pre_periods"] == out["Ywide"].index.get_loc(start_time)


@given(panel=panels(n_treated=2))
@settings(max_examples=120)
def test_cohort_arrays_have_the_shapes_their_membership_implies(panel):
    df, _, n_periods, _ = panel
    out = _prep(df)
    for cohort in out["cohorts"].values():
        assert cohort["y"].shape == (n_periods, len(cohort["treated_units"]))
        assert cohort["donor_matrix"].shape == (n_periods, len(cohort["donor_names"]))


@given(panel=panels(n_treated=2), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=60)
def test_cohort_mode_is_invariant_to_row_order(panel, seed):
    df, _, _, _ = panel
    shuffled = df.sample(frac=1.0, random_state=seed % (2**31)).reset_index(drop=True)
    base, other = _prep(df), _prep(shuffled)

    assert set(other["cohorts"]) == set(base["cohorts"])
    for key, cohort in base["cohorts"].items():
        np.testing.assert_allclose(other["cohorts"][key]["y"], cohort["y"], atol=0)
        assert other["cohorts"][key]["treated_units"] == cohort["treated_units"]


# ---------------------------------------------------------------------------
# Covariates
# ---------------------------------------------------------------------------

@given(panel=panels())
@settings(max_examples=120)
def test_covariate_blocks_partition_the_covariate_matrix(panel):
    """``donor_covariate_matrix`` and ``treated_covariate_matrix`` are slices.

    The split is positional, so it is right only if the rows really are
    ordered donors-then-treated. Stacking them back has to give the whole
    matrix, in the same order.
    """
    df, n_units, _, _ = panel
    out = _prep(df, covariates=["x1", "x2"])

    cov = out["covariate_matrix"]
    assert cov.shape == (n_units, 2)
    assert out["covariate_names"] == ("x1", "x2")
    assert out["donor_covariate_matrix"].shape == (n_units - 1, 2)
    assert out["treated_covariate_matrix"].shape == (1, 2)

    np.testing.assert_allclose(
        np.vstack([out["donor_covariate_matrix"], out["treated_covariate_matrix"]]),
        cov, atol=0,
    )


@given(panel=panels())
@settings(max_examples=120)
def test_the_treated_covariate_row_is_the_treated_unit(panel):
    """Positional slicing must land on ``u0``, the treated unit.

    Checked against an unnormalized run, where each entry is the unit's own
    pre-treatment mean and can be recomputed from the input frame.
    """
    df, _, _, starts = panel
    out = _prep(df, covariates=["x1"], normalize_covariates=False)

    pre = df[(df["unit"] == "u0") & (df["time"] < 2000 + out["pre_periods"])]
    np.testing.assert_allclose(
        out["treated_covariate_matrix"][0, 0], pre["x1"].mean(), rtol=1e-9, atol=1e-9
    )


@given(panel=panels())
@settings(max_examples=120)
def test_normalization_standardizes_each_column_across_units(panel):
    """``normalize=True`` centers and scales per column, across units."""
    df, n_units, _, _ = panel
    out = _prep(df, covariates=["x1", "x2"], normalize_covariates=True)
    cov = out["covariate_matrix"]

    for j in range(cov.shape[1]):
        column = cov[:, j]
        assume(out["covariate_scales"][j] > 1e-8)     # a constant column cannot scale
        assert abs(float(column.mean())) < 1e-6


@given(panel=panels())
@settings(max_examples=120)
def test_normalization_is_a_recoverable_affine_map(panel):
    """The reported means and scales invert the normalization exactly.

    Without that, ``covariate_means`` / ``covariate_scales`` are decoration --
    a caller cannot get back to the units the covariate was measured in.
    """
    df, _, _, _ = panel
    raw = _prep(df, covariates=["x1", "x2"], normalize_covariates=False)
    normed = _prep(df, covariates=["x1", "x2"], normalize_covariates=True)

    scales = np.where(normed["covariate_scales"] > 1e-12,
                      normed["covariate_scales"], 1.0)
    recovered = normed["covariate_matrix"] * scales + normed["covariate_means"]
    np.testing.assert_allclose(recovered, raw["covariate_matrix"],
                               rtol=1e-6, atol=1e-6)


@given(panel=panels())
@settings(max_examples=100)
def test_a_time_invariant_covariate_collapses_to_its_constant(panel):
    """``x2`` is constant within each unit, so aggregation cannot move it."""
    df, n_units, _, _ = panel
    out = _prep(df, covariates=["x2"], normalize_covariates=False)
    expected = np.array([float(u) for u in range(1, n_units)] + [0.0])   # donors, then u0
    np.testing.assert_allclose(out["covariate_matrix"][:, 0], expected, atol=1e-9)


# ---------------------------------------------------------------------------
# marex
# ---------------------------------------------------------------------------

@given(panel=panels())
@settings(max_examples=120)
def test_marex_stacks_outcomes_over_covariates(panel):
    """``predictor_matrix`` is ``[Y_pre; cov']`` and the block sizes say where
    the seam is."""
    df, n_units, _, _ = panel
    out = _prep(df, covariates=["x1", "x2"], marex=True)

    t_pre, n_cov = out["predictor_block_sizes"]
    assert t_pre == out["pre_periods"] and n_cov == 2
    assert out["predictor_matrix"].shape == (t_pre + n_cov, n_units)

    np.testing.assert_allclose(out["predictor_matrix"][t_pre:],
                               out["covariate_matrix"].T, atol=0)


@given(panel=panels())
@settings(max_examples=120)
def test_marex_population_mean_is_the_weighted_predictor_matrix(panel):
    """``X_population_mean == predictor_matrix @ f_weights``, with ``f``
    uniform and summing to one. The relation is the design objective's target,
    so a mismatch is a wrong optimisation problem, not a wrong report."""
    df, n_units, _, _ = panel
    out = _prep(df, covariates=["x1"], marex=True)

    f = out["f_weights"]
    assert f.shape == (n_units,)
    assert float(f.sum()) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(f, f[0], atol=1e-12)             # uniform
    np.testing.assert_allclose(out["X_population_mean"],
                               out["predictor_matrix"] @ f, rtol=1e-9, atol=1e-9)


@given(panel=panels())
@settings(max_examples=120)
def test_marex_without_covariates_is_the_outcome_block_alone(panel):
    df, n_units, _, _ = panel
    out = _prep(df, marex=True)

    t_pre, n_cov = out["predictor_block_sizes"]
    assert n_cov == 0
    assert out["predictor_matrix"].shape == (t_pre, n_units)
    np.testing.assert_allclose(out["X_population_mean"],
                               out["predictor_matrix"] @ out["f_weights"],
                               rtol=1e-9, atol=1e-9)


@given(panel=panels())
@settings(max_examples=100)
def test_the_opt_ins_leave_the_default_contract_untouched(panel):
    """Asking for covariates or marex must not move ``y`` or the donors.

    The opt-ins add keys; they are not allowed to change the answer the
    estimator would otherwise have got.
    """
    df, _, _, _ = panel
    base = _prep(df)
    for kw in ({"covariates": ["x1", "x2"]}, {"marex": True},
               {"covariates": ["x1"], "marex": True}):
        other = _prep(df, **kw)
        np.testing.assert_allclose(other["y"], base["y"], atol=0)
        np.testing.assert_allclose(other["donor_matrix"], base["donor_matrix"], atol=0)
        assert other["pre_periods"] == base["pre_periods"]
        assert list(other["donor_names"]) == list(base["donor_names"])
