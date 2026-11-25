"""``dataprep(missing_outcomes="keep")``: the opt-in for missing-data methods (#1146).

By default ``dataprep`` refuses any missing or non-finite outcome cell. Robust
synthetic control (Amjad, Shah and Shen, JMLR 19 (2018)), robust PCR (Agarwal,
Shah, Shen and Song, JASA 116 (2021)), SNN and MC-NNM are built for panels with
missing cells, so they opt in. With ``"keep"``:

* a NaN outcome in a row that is present stays NaN, and the panel carries an
  ``observed`` mask over ``Ywide``;
* an infinite outcome, an absent unit-period row (its treatment is unknown), a
  duplicate unit-period and a non-binary treatment are still refused;
* on a panel with no missing cell the output equals the default's, plus a mask
  of all ``True``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.datautils import balance, dataprep
from mlsynth.utils.prepared import CohortPanel, SinglePanel

ARGS = ("unit", "time", "y", "d")


def _panel(n_units=6, T=10, adopt=None, seed=0, names="int", shuffle=True):
    adopt = {0: 6} if adopt is None else adopt
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(T):
            rows.append(dict(unit=u if names == "int" else f"u{u:02d}", time=2000 + t,
                             y=rng.normal() + u, d=int(u in adopt and t >= adopt[u])))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _plant(df, cells):
    out = df.copy()
    for u, t in cells:
        out.loc[(out.unit == u) & (out.time == t), "y"] = np.nan
    return out


def _same_panel(a, b):
    """Every view key except ``observed`` agrees, NaN-aware."""
    for k in a:
        if k == "observed":
            continue
        x, y = a[k], b[k]
        if isinstance(x, pd.DataFrame):
            pd.testing.assert_frame_equal(x, y)
        elif isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y)
        elif isinstance(x, dict):
            assert list(x) == list(y)
            for c in x:
                _same_panel(x[c], y[c])
        else:
            assert (x == y) if not hasattr(x, "equals") else x.equals(y), k


# --------------------------------------------------------------------------- #
# the default is unchanged
# --------------------------------------------------------------------------- #
class TestDefault:
    def test_the_default_still_refuses_a_nan(self):
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            dataprep(_plant(_panel(), [(2, 2001)]), *ARGS)

    def test_refuse_is_the_default(self):
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            dataprep(_plant(_panel(), [(2, 2001)]), *ARGS, missing_outcomes="refuse")

    def test_the_default_panel_has_no_mask(self):
        p = dataprep(_panel(), *ARGS)
        assert p.observed is None
        assert "observed" not in p

    def test_an_unknown_value_is_refused(self):
        with pytest.raises(MlsynthDataError, match="missing_outcomes"):
            dataprep(_panel(), *ARGS, missing_outcomes="impute")


# --------------------------------------------------------------------------- #
# keep: what it keeps
# --------------------------------------------------------------------------- #
class TestKeep:
    @pytest.mark.parametrize("adopt", [{0: 6}, {0: 5, 1: 7}])
    def test_a_complete_panel_is_unchanged_with_an_all_true_mask(self, adopt):
        df = _panel(adopt=adopt)
        a = dataprep(df, *ARGS)
        b = dataprep(df, *ARGS, missing_outcomes="keep")
        assert type(a) is type(b)
        _same_panel(a, b)
        assert b.observed.dtype == bool and b.observed.all()
        assert b.observed.shape == b.Ywide.shape

    def test_a_nan_stays_where_it_was_single(self):
        df = _plant(_panel(), [(2, 2001), (0, 2003), (5, 2008)])
        p = dataprep(df, *ARGS, missing_outcomes="keep")
        assert isinstance(p, SinglePanel)
        missing = {(u, t) for u, t in [(2, 2001), (0, 2003), (5, 2008)]}
        for i, t in enumerate(p.Ywide.index):
            for j, u in enumerate(p.Ywide.columns):
                assert p.observed[i, j] == ((u, t) not in missing)
                assert np.isnan(p.Ywide.iat[i, j]) == ((u, t) in missing)
        # the treated vector and donor matrix carry the same NaNs
        assert np.isnan(p.y[3])
        assert np.isnan(p.donor_matrix).sum() == 2

    def test_a_nan_stays_where_it_was_cohort(self):
        df = _plant(_panel(adopt={0: 5, 1: 7}), [(1, 2002), (4, 2009)])
        p = dataprep(df, *ARGS, missing_outcomes="keep")
        assert isinstance(p, CohortPanel)
        assert (~p.observed).sum() == 2
        assert np.isnan(p.Ywide.loc[2002, 1]) and np.isnan(p.Ywide.loc[2009, 4])
        assert "observed" in p and p["observed"] is p.observed

    def test_the_mask_follows_ywide_with_string_ids(self):
        df = _plant(_panel(names="str"), [("u03", 2004)])
        p = dataprep(df, *ARGS, missing_outcomes="keep")
        j = list(p.Ywide.columns).index("u03")
        i = list(p.Ywide.index).index(2004)
        assert not p.observed[i, j] and (~p.observed).sum() == 1

    def test_supplied_keys_give_the_same_panel(self):
        df = _plant(_panel(), [(3, 2005)])
        a = dataprep(df, *ARGS, missing_outcomes="keep")
        b = dataprep(df, *ARGS, missing_outcomes="keep", keys=balance(df, "unit", "time"))
        _same_panel(a, b)
        np.testing.assert_array_equal(a.observed, b.observed)


# --------------------------------------------------------------------------- #
# keep: what it still refuses
# --------------------------------------------------------------------------- #
class TestKeepStillRefuses:
    def test_an_infinite_outcome(self):
        df = _panel(shuffle=False)
        df.loc[(df.unit == 1) & (df.time == 2002), "y"] = np.inf
        with pytest.raises(MlsynthDataError, match="infinite"):
            dataprep(df, *ARGS, missing_outcomes="keep")

    def test_an_infinite_outcome_beside_a_nan(self):
        df = _plant(_panel(shuffle=False), [(3, 2004)])
        df.loc[(df.unit == 1) & (df.time == 2002), "y"] = -np.inf
        with pytest.raises(MlsynthDataError, match="infinite"):
            dataprep(df, *ARGS, missing_outcomes="keep")

    def test_an_absent_unit_period_row(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 4) & (df.time == 2003))]
        with pytest.raises(MlsynthDataError):
            dataprep(df, *ARGS, missing_outcomes="keep")

    def test_a_duplicate_unit_period_is_balances_to_refuse(self):
        # dataprep never refused duplicates itself (without a key scan pandas
        # raises); callers run balance() first and hand its scan over, and a
        # kept NaN does not change that.
        df = _plant(_panel(shuffle=False), [(1, 2001)])
        with pytest.raises(MlsynthDataError, match="Duplicate"):
            balance(pd.concat([df, df.iloc[[3]]]), "unit", "time")

    def test_keep_with_the_marex_predictor_matrix(self):
        # marex stacks pre-period outcomes into a predictor matrix; a kept NaN
        # would pass into it, so the combination is refused.
        with pytest.raises(MlsynthDataError, match="marex"):
            dataprep(_panel(), *ARGS, missing_outcomes="keep", marex=True)

    def test_a_non_binary_treatment(self):
        df = _panel(shuffle=False)
        df.loc[df.d == 1, "d"] = 2
        with pytest.raises(MlsynthDataError):
            dataprep(df, *ARGS, missing_outcomes="keep")


# --------------------------------------------------------------------------- #
# the panel types check the mask
# --------------------------------------------------------------------------- #
class TestMaskConsistency:
    def test_a_mask_of_the_wrong_shape_is_refused(self):
        p = dataprep(_panel(), *ARGS, missing_outcomes="keep")
        fields = {k: getattr(p, k) for k in p.__dataclass_fields__}
        fields["observed"] = p.observed[:, :-1]
        with pytest.raises(MlsynthDataError, match="observed has shape"):
            SinglePanel(**fields)

    def test_a_mask_that_disagrees_with_ywide_is_refused(self):
        p = dataprep(_plant(_panel(), [(2, 2001)]), *ARGS, missing_outcomes="keep")
        fields = {k: getattr(p, k) for k in p.__dataclass_fields__}
        fields["observed"] = np.ones_like(p.observed)
        with pytest.raises(MlsynthDataError, match="disagrees with the NaN cells"):
            SinglePanel(**fields)

    def test_a_non_boolean_mask_is_refused(self):
        p = dataprep(_panel(adopt={0: 5, 1: 7}), *ARGS, missing_outcomes="keep")
        fields = {k: getattr(p, k) for k in p.__dataclass_fields__}
        fields["observed"] = p.observed.astype(int)
        with pytest.raises(MlsynthDataError, match="boolean array"):
            CohortPanel(**fields)


# --------------------------------------------------------------------------- #
# property over the domain: random missingness, defects included
# --------------------------------------------------------------------------- #
@st.composite
def cases(draw):
    n_units = draw(st.integers(3, 7))
    T = draw(st.integers(3, 9))
    n_treated = draw(st.integers(1, n_units - 1))
    adopt = {u: draw(st.integers(1, T - 1)) for u in range(n_treated)}
    df = _panel(n_units=n_units, T=T, adopt=adopt, seed=draw(st.integers(0, 9999)),
                names=draw(st.sampled_from(["int", "str"])))
    holes = draw(st.lists(st.integers(0, len(df) - 1), max_size=len(df) // 2, unique=True))
    df.iloc[holes, df.columns.get_loc("y")] = np.nan
    defect = draw(st.sampled_from(["none", "none", "none", "inf", "gap"]))
    i = draw(st.integers(0, len(df) - 1))
    if defect == "inf":
        df.iloc[i, df.columns.get_loc("y")] = np.inf
    elif defect == "gap":
        df = df.drop(df.index[i])
    return df, len(holes), defect


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
def test_keep_preserves_exactly_the_missing_cells(case):
    df, n_holes, defect = case
    if defect != "none":
        with pytest.raises(MlsynthDataError):
            dataprep(df, *ARGS, missing_outcomes="keep")
        return
    # the default refuses iff a cell is missing
    if n_holes:
        with pytest.raises(MlsynthDataError, match="Missing outcome values"):
            dataprep(df, *ARGS)
    p = dataprep(df, *ARGS, missing_outcomes="keep")
    ref = df.pivot(index="time", columns="unit", values="y").loc[p.Ywide.index, p.Ywide.columns]
    np.testing.assert_array_equal(p.Ywide.to_numpy(), ref.to_numpy())
    np.testing.assert_array_equal(p.observed, ~np.isnan(ref.to_numpy()))
    assert int((~p.observed).sum()) == n_holes
