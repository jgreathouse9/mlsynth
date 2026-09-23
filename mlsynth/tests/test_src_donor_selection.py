"""SRC's donor pool is chosen in one pass over the panel, not one pass per unit.

``prepare_src_inputs`` selects donors as "every unit that is never treated".
Expressed as a per-unit ``df.loc[df[unitid] == u, treat].sum()`` inside a
comprehension that is a full scan of the panel for each unit, so the work grows
as units x rows: on Prop 99 it was 8.8 ms of a 12.5 ms fit, and on an 819-unit
panel it was a full second. A single ``groupby`` answers the same question once.

These tests pin the selection *semantics* -- which units, in which order -- so
the rewrite cannot change the donor pool, and count the panel scans so the
quadratic form cannot come back unnoticed.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.src_helpers.setup import prepare_src_inputs


def _panel(n_units=6, n_time=10, treated="u0", t0=6):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n_units):
        u = f"u{i}"
        base = rng.normal(100, 5)
        for t in range(n_time):
            treat = int(u == treated and t >= t0)
            rows.append((u, t, base + t + 2.0 * treat, treat))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "treat"])


KW = dict(outcome="y", treat="treat", unitid="unit", time="time")


class TestWhichUnitsBecomeDonors:
    def test_every_never_treated_unit_is_a_donor(self):
        out = prepare_src_inputs(_panel(), **KW)
        assert list(out.donor_labels) == ["u1", "u2", "u3", "u4", "u5"]

    def test_the_treated_unit_is_excluded(self):
        out = prepare_src_inputs(_panel(), **KW)
        assert "u0" not in out.donor_labels

    def test_donor_order_follows_first_appearance_in_the_frame(self):
        # The donor columns are read positionally downstream, so the order is
        # part of the contract, not an implementation detail.
        df = _panel()
        shuffled = pd.concat([df[df.unit == u] for u in
                              ["u3", "u0", "u5", "u1", "u4", "u2"]])
        out = prepare_src_inputs(shuffled, **KW)
        assert list(out.donor_labels) == ["u3", "u5", "u1", "u4", "u2"]

    def test_a_second_treated_unit_is_rejected_before_donor_selection(self):
        # Any unit with treat == 1 anywhere counts as treated, so the two-treated
        # panel never reaches the donor pool. This is why the per-unit scan the
        # rewrite removes could only ever have excluded the treated unit itself.
        df = _panel()
        df.loc[(df.unit == "u3") & (df.time >= 6), "treat"] = 1
        with pytest.raises(MlsynthDataError, match="single treated unit"):
            prepare_src_inputs(df, **KW)

    def test_a_unit_with_a_nonzero_treat_code_other_than_one_is_not_a_donor(self):
        # The one case the scan decided on its own: `treat == 1` does not flag
        # this unit as treated, but its column is not all zeros either. It stays
        # out of the donor pool, and the rewrite must keep it out.
        df = _panel()
        df.loc[(df.unit == "u2") & (df.time >= 6), "treat"] = 2
        out = prepare_src_inputs(df, **KW)
        assert "u2" not in out.donor_labels
        assert list(out.donor_labels) == ["u1", "u3", "u4", "u5"]

    def test_a_panel_with_no_untreated_units_raises(self):
        df = _panel(n_units=2)
        df.loc[df.unit == "u1", "treat"] = 3          # nonzero, not treated-coded
        with pytest.raises(MlsynthDataError, match="donor pool is empty"):
            prepare_src_inputs(df, **KW)


class TestThePanelIsScannedOnce:
    """One pass over the panel, not one per unit.

    Timing this would measure the machine; what the rewrite actually changes is
    the *number of full-panel comparisons*, so that is what is counted. The old
    comprehension built a length-``n_rows`` boolean mask once per unit, so the
    count grew with the donor pool. The replacement builds a fixed handful
    regardless of how many units the panel holds.
    """

    @staticmethod
    def _full_length_comparisons(df, monkeypatch):
        seen = []
        original = pd.Series.__eq__

        def counting_eq(self, other):
            seen.append(len(self))
            return original(self, other)

        monkeypatch.setattr(pd.Series, "__eq__", counting_eq, raising=False)
        prepare_src_inputs(df, **KW)
        monkeypatch.undo()
        return sum(1 for n in seen if n == len(df))

    @pytest.mark.parametrize("n_units", [8, 40, 160])
    def test_the_count_does_not_grow_with_the_donor_pool(self, n_units,
                                                         monkeypatch):
        df = _panel(n_units=n_units, n_time=12, t0=8)
        scans = self._full_length_comparisons(df, monkeypatch)
        assert scans <= 4, (
            f"{n_units} units caused {scans} full-panel comparisons; donor "
            "selection is scanning the panel once per unit again")
