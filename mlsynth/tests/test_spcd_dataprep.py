"""SPCD ingestion through ``datautils.geoex_dataprep`` (#1138, part 1).

SPCD is a design estimator: it chooses which units to treat, so the panel has
no treatment column and ``dataprep`` cannot split it. ``geoex_dataprep`` is the
design-side sibling, as for SYNDES (#1142) and PANGEO (#1143): a balanced
``time x unit`` outcome matrix, refusing a duplicate or absent unit-period and
a missing or non-finite outcome, and an optional ``post_col`` whose post
periods must be one block at the end of the panel. The covariates come from
``build_covariate_matrix`` on ``balance``'s key scan, each unit's pre-period
mean, and SPCD z-scores them itself as before.

Two rules are changes. An infinite outcome used to reach the iteration matrix;
it is now refused. Post periods used to be accepted anywhere, and the plotter
and the synthetic paths stack ``Y_pre`` over ``Y_post`` and read the result in
time order, which puts the periods out of order when a post period sits between
pre periods; such a ``post_col`` is now refused.

``_reference`` is the old pivot, kept as the oracle the new path must match on
every panel both accept. The outcome matrices and every label must match it
exactly. The covariate block is a reduction, and the old and new paths reduce
over differently laid out arrays on some pandas versions, so it must match to
the summation bound in :func:`_covariate_tolerance`.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.fast_scm_helpers.structure import IndexSet
from mlsynth.utils.spcd_helpers import setup as SETUP
from mlsynth.utils.spcd_helpers.setup import prepare_spcd_inputs
from mlsynth.utils.spcd_helpers.structures import SPCDInputs

COLS = dict(outcome="y", unitid="unit", time="time")


def _reference(df, outcome, unitid, time, T0=None, post_col=None, covariates=None):
    """The pre-#1138 setup, verbatim: its pivots, split and covariate block."""
    if post_col is not None and post_col not in df.columns:
        raise MlsynthDataError(f"post_col '{post_col}' is not present in df.")
    ordered_df = df.sort_values([time, unitid]).copy()
    Ywide = ordered_df.pivot(index=time, columns=unitid, values=outcome).sort_index()
    if Ywide.isna().any().any():
        raise MlsynthDataError(
            "SPCD requires a complete balanced outcome matrix after pivoting.")
    time_labels = np.asarray(Ywide.index.to_list())
    unit_labels = np.asarray(Ywide.columns.to_list())
    Y_full = Ywide.to_numpy(dtype=float)
    if post_col is not None:
        post_by_time = (ordered_df[[time, post_col]].drop_duplicates(subset=[time])
                        .set_index(time).reindex(time_labels)[post_col])
        if post_by_time.isna().any():
            raise MlsynthDataError(
                "post_col must be defined for every time period in the panel.")
        post_mask = post_by_time.astype(bool).to_numpy()
        if post_mask.all():
            raise MlsynthConfigError("post_col marks every period as post-treatment.")
        pre_mask = ~post_mask
        Y_pre = Y_full[pre_mask, :]
        Y_post = Y_full[post_mask, :] if post_mask.any() else None
        pre_labels = time_labels[pre_mask]
        post_labels = time_labels[post_mask] if post_mask.any() else None
    else:
        if T0 is None:
            T0 = Y_full.shape[0]
        if T0 <= 0 or T0 > Y_full.shape[0]:
            raise MlsynthConfigError(
                f"T0 must be between 1 and the number of periods ({Y_full.shape[0]}).")
        Y_pre = Y_full[:T0, :]
        Y_post = Y_full[T0:, :] if T0 < Y_full.shape[0] else None
        pre_labels = time_labels[:T0]
        post_labels = time_labels[T0:] if T0 < Y_full.shape[0] else None
    if Y_pre.shape[0] < 2:
        raise MlsynthDataError("SPCD requires at least two pre-treatment periods.")
    if Y_pre.shape[1] < 2:
        raise MlsynthDataError("SPCD requires at least two units.")
    cov_matrix = None
    cov_names = None
    if covariates:
        cov_names = list(covariates)
        pre_label_set = set(np.asarray(pre_labels).tolist())
        columns = []
        for c in cov_names:
            if c not in ordered_df.columns:
                raise MlsynthDataError(f"covariate '{c}' is not present in df.")
            cwide = ordered_df.pivot(index=time, columns=unitid, values=c).sort_index()
            per_unit = cwide.loc[cwide.index.isin(pre_label_set), :].mean(axis=0)
            col = per_unit.reindex(unit_labels).to_numpy(dtype=float)
            if np.isnan(col).any():
                raise MlsynthDataError(
                    f"covariate '{c}' has missing values for some units in the "
                    f"pre-treatment period.")
            columns.append(col)
        X = np.column_stack(columns)
        cov_matrix = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-12)
    return SPCDInputs(
        Y_pre=Y_pre, Y_post=Y_post,
        unit_index=IndexSet.from_labels(unit_labels),
        time_index=IndexSet.from_labels(time_labels),
        pre_time_index=IndexSet.from_labels(pre_labels),
        post_time_index=IndexSet.from_labels(post_labels) if post_labels is not None else None,
        outcome=outcome, covariates=cov_matrix, covariate_names=cov_names)


def _panel(n_units=6, T=10, n_post=3, seed=0, names="int", shuffle=True, t0=2000):
    rng = np.random.default_rng(seed)
    share = rng.uniform(0.0, 1.0, size=n_units)
    rows = []
    for u in range(n_units):
        for t in range(T):
            rows.append(dict(unit=u if names == "int" else f"m{u:02d}",
                             time=t0 + t, y=rng.normal() + u + 0.1 * t,
                             post=int(t >= T - n_post), x=rng.normal() + 0.3 * u,
                             share=share[u]))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _labels(index_set):
    return None if index_set is None else np.asarray(index_set.labels)


EPS = np.finfo(float).eps


def _covariate_tolerance(df, ref, unitid="unit", time="time"):
    """Per-covariate bound on ``|z_new - z_old|``, from the summation rounding bound.

    Both paths z-score the same per-unit pre-period means, but each mean is a
    sum of ``T0`` values and the cross-unit mean and variance are sums of ``N``,
    and the two paths sum them in different orders: the old ``.loc`` selection
    is column-major on pandas 2.3, so ``DataFrame.mean`` sums each unit's column
    pairwise, while ``build_covariate_matrix``'s frame is row-major and is summed
    row by row (pandas 3 lays out both row-major). A sum of ``n`` terms of size
    at most ``M`` is within ``n * eps * M`` of the exact sum in either order, so
    the per-unit means, and the cross-unit centre and scale, of the two paths
    differ by at most ``delta = 2 * (T0 + N) * eps * M``. To first order
    ``z = (m - mean(m)) / s`` then moves by at most ``2 * delta / s * (1 + |z|)``
    for the centre and the scale; this doubles that for safety and adds
    ``8 * eps * (1 + |z|)`` for the z-score's own rounding.
    """
    pre = set(_labels(ref.pre_time_index).tolist())
    window = df[df[time].isin(pre)]
    T0, N = len(pre), ref.covariates.shape[0]
    tol = []
    for j, c in enumerate(ref.covariate_names):
        by_unit = window.groupby(unitid)[c]
        M = float(window[c].abs().max())
        s = float(by_unit.mean().std(ddof=0))
        delta = 2 * (T0 + N) * EPS * M
        z = float(np.max(np.abs(ref.covariates[:, j])))
        tol.append((4 * delta / (s + 1e-12) + 8 * EPS) * (1 + z))
    return np.asarray(tol)


def _assert_same(inputs, ref, df=None):
    assert isinstance(inputs, SPCDInputs)
    assert inputs.Y_pre.dtype == np.float64
    np.testing.assert_array_equal(inputs.Y_pre, ref.Y_pre)
    if ref.Y_post is None:
        assert inputs.Y_post is None and inputs.post_time_index is None
    else:
        np.testing.assert_array_equal(inputs.Y_post, ref.Y_post)
        np.testing.assert_array_equal(_labels(inputs.post_time_index),
                                      _labels(ref.post_time_index))
    for name in ("unit_index", "time_index", "pre_time_index"):
        got, want = _labels(getattr(inputs, name)), _labels(getattr(ref, name))
        np.testing.assert_array_equal(got, want)
        assert got.dtype == want.dtype, name
    assert inputs.outcome == ref.outcome
    assert inputs.covariate_names == ref.covariate_names
    if ref.covariates is None:
        assert inputs.covariates is None
    else:
        # The only reduction compared here: equal to the summation bound, not
        # bitwise (see _covariate_tolerance). Everything above is exact.
        assert df is not None, "a covariate comparison needs the panel for its bound"
        assert inputs.covariates.shape == ref.covariates.shape
        assert np.isfinite(inputs.covariates).all() == np.isfinite(ref.covariates).all()
        for j, atol in enumerate(_covariate_tolerance(df, ref)):
            np.testing.assert_allclose(inputs.covariates[:, j], ref.covariates[:, j],
                                       rtol=0, atol=atol)


def _spy(monkeypatch, name):
    calls = []
    real = getattr(SETUP, name)

    def spy(*a, **k):
        calls.append(k)
        return real(*a, **k)

    monkeypatch.setattr(SETUP, name, spy)
    return calls


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
class TestRouting:
    def test_the_outcome_goes_through_geoex_dataprep(self, monkeypatch):
        calls = _spy(monkeypatch, "geoex_dataprep")
        prepare_spcd_inputs(_panel(), **COLS, T0=6)
        assert [k.get("post_col") for k in calls] == [None]

    def test_post_col_is_validated_by_geoex_dataprep(self, monkeypatch):
        calls = _spy(monkeypatch, "geoex_dataprep")
        prepare_spcd_inputs(_panel(), **COLS, post_col="post")
        assert "post" in [k.get("post_col") for k in calls]

    def test_covariates_go_through_build_covariate_matrix(self, monkeypatch):
        calls = _spy(monkeypatch, "build_covariate_matrix")
        prepare_spcd_inputs(_panel(), **COLS, post_col="post", covariates=["x", "share"])
        assert len(calls) == 1
        # SPCD z-scores with its own scale (ddof=0), so dataprep's is off, and
        # the key scan from balance() is shared with the covariate pivots.
        assert calls[0]["normalize"] is False
        assert calls[0]["keys"] is not None

    def test_no_covariates_no_covariate_block(self, monkeypatch):
        calls = _spy(monkeypatch, "build_covariate_matrix")
        inputs = prepare_spcd_inputs(_panel(), **COLS, post_col="post")
        assert calls == [] and inputs.covariates is None

    def test_setup_does_not_pivot(self):
        # The setup module itself must not reshape the panel. A runtime patch of
        # DataFrame.pivot cannot test this: geoex_dataprep's _wide_pivot falls
        # back to df.pivot on pandas versions whose fast path would change the
        # array layout, so the patch fires inside geoex_dataprep there.
        tree = ast.parse(inspect.getsource(SETUP))
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert not called & {"pivot", "pivot_table", "unstack", "crosstab"}, called


# --------------------------------------------------------------------------- #
# smoke and equivalence
# --------------------------------------------------------------------------- #
class TestSameInputs:
    @pytest.mark.parametrize("kw,split", [
        (dict(), dict(post_col="post")),
        (dict(), dict(T0=6)),
        (dict(), dict()),                                   # design only
        (dict(n_post=0), dict(post_col="post")),            # post_col, no post
        (dict(names="str"), dict(post_col="post")),
        (dict(names="str", shuffle=False), dict(T0=10)),    # T0 at the last period
        (dict(shuffle=False, t0=-4), dict(T0=2)),           # two pre periods
        (dict(T=3, n_post=1), dict(post_col="post")),
        (dict(n_units=2), dict(post_col="post")),           # two units
    ])
    def test_inputs_match_the_old_pivot(self, kw, split):
        df = _panel(**kw)
        _assert_same(prepare_spcd_inputs(df, **COLS, **split),
                     _reference(df, **COLS, **split))

    @pytest.mark.parametrize("names", ["int", "str"])
    @pytest.mark.parametrize("split", [dict(post_col="post"), dict(T0=4), dict()])
    def test_covariates_match_the_old_pivot(self, names, split):
        df = _panel(names=names)
        cov = ["x", "share"]
        _assert_same(prepare_spcd_inputs(df, **COLS, **split, covariates=cov),
                     _reference(df, **COLS, **split, covariates=cov), df)

    @pytest.mark.parametrize("seed", range(4))
    def test_covariates_match_the_old_pivot_on_a_wide_panel(self, seed):
        # Nine pre periods and forty units: past eight terms numpy sums a
        # contiguous run pairwise, so here the two paths' summation orders differ
        # on pandas 2.3, in the per-unit means and in the cross-unit z-score.
        df = _panel(n_units=40, T=12, seed=seed)
        df["x"] = df["x"] * 1e3
        cov = ["x", "share"]
        ref = _reference(df, **COLS, post_col="post", covariates=cov)
        _assert_same(prepare_spcd_inputs(df, **COLS, post_col="post", covariates=cov),
                     ref, df)
        # The bound is a few hundred ulps, not a tolerance that hides a defect.
        assert _covariate_tolerance(df, ref).max() < 1e-11

    def test_a_covariate_gap_inside_the_pre_window_is_averaged_over(self):
        # One missing covariate value leaves the unit's pre-period mean defined.
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 2001), "x"] = np.nan
        _assert_same(prepare_spcd_inputs(df, **COLS, post_col="post", covariates=["x"]),
                     _reference(df, **COLS, post_col="post", covariates=["x"]), df)

    def test_extra_columns_are_ignored(self):
        df = _panel().assign(region="r", other=1.0)
        _assert_same(prepare_spcd_inputs(df, **COLS, post_col="post"),
                     _reference(df, **COLS, post_col="post"))

    def test_the_caller_frame_is_not_changed(self):
        df = _panel()
        before = df.copy()
        prepare_spcd_inputs(df, **COLS, post_col="post", covariates=["x"])
        pd.testing.assert_frame_equal(df, before)

    def test_fit_smoke(self):
        from mlsynth import SPCD
        res = SPCD(dict(df=_panel(n_units=8, T=20, n_post=4), **COLS, post_col="post",
                        covariates=["x"], enable_inference=False,
                        display_graph=False)).fit()
        assert np.isfinite(np.asarray(res.design.contrast_weights)).all()
        assert res.inputs.Y_pre.shape == (16, 8)

    def test_multi_arm_smoke(self):
        from mlsynth import SPCD
        df = _panel(n_units=10, T=20, n_post=4, names="str")
        df["arm"] = np.where(df.unit < "m05", "A", "B")
        res = SPCD(dict(df=df, **COLS, post_col="post", arm="arm",
                        enable_inference=False, display_graph=False)).fit()
        assert set(res.arm_designs) == {"A", "B"}
        for r in res.arm_designs.values():
            assert r.inputs.Y_pre.shape == (16, 5)


# --------------------------------------------------------------------------- #
# refusals now come from geoex_dataprep and balance
# --------------------------------------------------------------------------- #
class TestRefusals:
    def test_post_periods_between_pre_periods_are_refused(self):
        df = _panel(shuffle=False)
        df.loc[df.time == 2002, "post"] = 1      # a post period inside the pre window
        with pytest.raises(MlsynthConfigError, match="contiguous"):
            prepare_spcd_inputs(df, **COLS, post_col="post")

    def test_post_periods_at_the_start_are_refused(self):
        df = _panel(shuffle=False, n_post=0)
        df.loc[df.time <= 2001, "post"] = 1
        with pytest.raises(MlsynthConfigError, match="contiguous"):
            prepare_spcd_inputs(df, **COLS, post_col="post")

    def test_a_missing_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 1) & (df.time == 2003), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_spcd_inputs(df, **COLS, post_col="post")

    def test_a_missing_post_period_outcome_is_refused(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 1) & (df.time == 2009), "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_spcd_inputs(df, **COLS, post_col="post")

    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_a_non_finite_outcome_is_refused(self, value):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 2001), "y"] = value
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            prepare_spcd_inputs(df, **COLS, T0=6)

    def test_an_absent_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 3) & (df.time == 2004))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            prepare_spcd_inputs(df, **COLS, T0=6)

    def test_a_duplicate_unit_period_is_refused(self):
        df = _panel(shuffle=False)
        df = pd.concat([df, df.iloc[[3]]])
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_spcd_inputs(df, **COLS, T0=6)

    def test_a_duplicate_unit_period_with_covariates_is_refused(self):
        df = _panel(shuffle=False)
        df = pd.concat([df, df.iloc[[3]]])
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            prepare_spcd_inputs(df, **COLS, post_col="post", covariates=["x"])

    def test_a_missing_outcome_column_is_refused(self):
        with pytest.raises(MlsynthDataError, match="not found"):
            prepare_spcd_inputs(_panel().drop(columns="y"), **COLS, T0=6)

    def test_the_estimator_reports_the_refusal(self):
        from mlsynth import SPCD
        df = _panel(shuffle=False)
        df.loc[(df.unit == 2) & (df.time == 2001), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            SPCD(dict(df=df, **COLS, post_col="post", enable_inference=False,
                      display_graph=False)).fit()


# --------------------------------------------------------------------------- #
# SPCD's own requirements, kept on top
# --------------------------------------------------------------------------- #
class TestOwnChecks:
    def test_unknown_post_col(self):
        with pytest.raises(MlsynthDataError, match="post_col 'nope' is not present"):
            prepare_spcd_inputs(_panel(), **COLS, post_col="nope")

    def test_an_unknown_post_col_is_refused_before_the_panel_is_read(self, monkeypatch):
        calls = _spy(monkeypatch, "geoex_dataprep")
        df = _panel(shuffle=False)
        df.loc[df.index[0], "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="post_col 'nope' is not present"):
            prepare_spcd_inputs(df, **COLS, post_col="nope")
        assert calls == []

    def test_every_period_post_is_refused(self):
        df = _panel(n_post=10)
        with pytest.raises(MlsynthConfigError, match="every period"):
            prepare_spcd_inputs(df, **COLS, post_col="post")

    def test_post_col_undefined_for_a_period(self):
        df = _panel(shuffle=False).astype({"post": float})
        df.loc[df.time == 2003, "post"] = np.nan
        with pytest.raises(MlsynthDataError, match="every time period"):
            prepare_spcd_inputs(df, **COLS, post_col="post")

    @pytest.mark.parametrize("T0", [0, -1, 11])
    def test_T0_out_of_range(self, T0):
        with pytest.raises(MlsynthConfigError, match="T0 must be between"):
            prepare_spcd_inputs(_panel(), **COLS, T0=T0)

    def test_one_pre_period_is_too_few(self):
        with pytest.raises(MlsynthDataError, match="two pre-treatment"):
            prepare_spcd_inputs(_panel(), **COLS, T0=1)

    def test_one_pre_period_from_post_col_is_too_few(self):
        with pytest.raises(MlsynthDataError, match="two pre-treatment"):
            prepare_spcd_inputs(_panel(n_post=9), **COLS, post_col="post")

    def test_one_unit_is_too_few(self):
        with pytest.raises(MlsynthDataError, match="two units"):
            prepare_spcd_inputs(_panel(n_units=1), **COLS, T0=6)

    def test_unknown_covariate(self):
        with pytest.raises(MlsynthDataError, match="covariate 'nope' is not present in df"):
            prepare_spcd_inputs(_panel(), **COLS, post_col="post", covariates=["x", "nope"])

    def test_a_covariate_missing_over_a_units_pre_window(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.post == 0), "x"] = np.nan
        with pytest.raises(MlsynthDataError, match="covariate 'x' has missing values"):
            prepare_spcd_inputs(df, **COLS, post_col="post", covariates=["share", "x"])

    def test_a_covariate_missing_only_after_the_pre_window_is_accepted(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 3) & (df.post == 1), "x"] = np.nan
        inputs = prepare_spcd_inputs(df, **COLS, post_col="post", covariates=["x"])
        assert np.isfinite(inputs.covariates).all()

    def test_covariates_are_z_scored_across_units(self):
        inputs = prepare_spcd_inputs(_panel(), **COLS, post_col="post",
                                     covariates=["x", "share"])
        np.testing.assert_allclose(inputs.covariates.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(inputs.covariates.std(axis=0, ddof=0), 1.0, atol=1e-9)

    def test_a_constant_covariate_scores_zero(self):
        df = _panel().assign(c=3.0)
        inputs = prepare_spcd_inputs(df, **COLS, post_col="post", covariates=["c"])
        np.testing.assert_array_equal(inputs.covariates, 0.0)


# --------------------------------------------------------------------------- #
# properties over the domain, defects included
# --------------------------------------------------------------------------- #
@st.composite
def cases(draw):
    n_units = draw(st.integers(1, 7))
    T = draw(st.integers(1, 9))
    names = draw(st.sampled_from(["int", "str"]))
    seed = draw(st.integers(0, 10_000))
    df = _panel(n_units=n_units, T=T, n_post=0, seed=seed, names=names)
    how = draw(st.sampled_from(["post_col", "T0", "none"]))
    if how == "T0":
        split = dict(T0=draw(st.integers(0, T + 1)))
    elif how == "none":
        split = dict()
    else:
        split = dict(post_col="post")
        n_pre = draw(st.integers(0, T))
        df["post"] = (df["time"] >= 2000 + n_pre).astype(int)
    covariates = draw(st.sampled_from([None, ["x"], ["share", "x"]]))
    defects = ["none", "none", "nan", "inf", "dup", "cov_nan"]
    if n_units >= 2 and T >= 2:
        defects.append("gap")
    if how == "post_col" and T >= 2:
        defects.append("split")
    defect = draw(st.sampled_from(defects))
    i = draw(st.integers(0, len(df) - 1))
    if defect == "nan":
        df.iloc[i, df.columns.get_loc("y")] = np.nan
    elif defect == "inf":
        df.iloc[i, df.columns.get_loc("y")] = draw(st.sampled_from([np.inf, -np.inf]))
    elif defect == "gap":
        df = df.drop(df.index[i])
    elif defect == "dup":
        df = pd.concat([df, df.iloc[[i]]])
    elif defect == "cov_nan":
        # Not a refusal: the old setup averaged over it, and so must the new one.
        df.iloc[i, df.columns.get_loc("x")] = np.nan
    elif defect == "split":
        t = draw(st.integers(0, T - 2))          # a post period before a pre period
        df["post"] = 0
        df.loc[df.time == 2000 + t, "post"] = 1
    return df, split, covariates, defect


def _outcome(fn, *a, **k):
    try:
        return fn(*a, **k), None
    except (MlsynthDataError, MlsynthConfigError) as exc:
        return None, exc


@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_geoex_path_matches_the_old_pivot_or_refuses(case):
    df, split, covariates, defect = case
    kw = dict(**COLS, **split, covariates=covariates)
    if defect in ("nan", "inf", "gap", "dup"):
        with pytest.raises(MlsynthDataError):
            prepare_spcd_inputs(df, **kw)
        return
    if defect == "split":
        with pytest.raises(MlsynthConfigError, match="contiguous"):
            prepare_spcd_inputs(df, **kw)
        return
    # A clean panel: the old setup's answer, or the old setup's kind of refusal.
    ref, ref_exc = _outcome(_reference, df, **kw)
    if ref_exc is not None:
        with pytest.raises(type(ref_exc)):
            prepare_spcd_inputs(df, **kw)
        return
    inputs = prepare_spcd_inputs(df, **kw)
    _assert_same(inputs, ref, df)
    # The pre and post blocks are the panel's periods in time order.
    times = _labels(inputs.time_index)
    pre = _labels(inputs.pre_time_index)
    post = _labels(inputs.post_time_index)
    stacked = pre if post is None else np.concatenate([pre, post])
    np.testing.assert_array_equal(stacked, times)
    full = inputs.Y_pre if inputs.Y_post is None else np.vstack([inputs.Y_pre, inputs.Y_post])
    assert full.shape == (len(times), len(_labels(inputs.unit_index)))
    assert np.isfinite(full).all()
