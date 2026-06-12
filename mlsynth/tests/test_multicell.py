"""Multi-cell GeoLift: dataprep, per-cell analysis, cross-cell winner, estimator."""

import numpy as np
import pandas as pd
import pytest

from mlsynth import MULTICELLGEOLIFT
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.geolift_helpers.multicell import (
    analyze_multicell, multicell_dataprep, MultiCellResults,
)


def _panel(T=34, pre=26, effA=8.0, effB=0.0, seed=0):
    """Long panel: 6 controls, cells A (effect effA) and B (effect effB)."""
    rng = np.random.default_rng(seed)
    trend = np.arange(T) * 0.3
    controls = {f"c{i}": trend + 5 * i + rng.normal(scale=0.4, size=T) for i in range(6)}
    C = np.array(list(controls.values()))                       # (6, T)
    def cell_unit(w, eff, s):
        y = w @ C + rng.normal(scale=0.3, size=T)               # synth from controls
        y[pre:] += eff                                          # post effect
        return y
    treated = {
        "a0": cell_unit(np.array([.4, .3, .3, 0, 0, 0]), effA, 1),
        "a1": cell_unit(np.array([.3, .3, .2, .2, 0, 0]), effA, 2),
        "b0": cell_unit(np.array([0, 0, .3, .3, .4, 0]), effB, 3),
        "b1": cell_unit(np.array([0, 0, 0, .3, .3, .4]), effB, 4),
    }
    cells = {**{u: "A" for u in ("a0", "a1")}, **{u: "B" for u in ("b0", "b1")},
             **{u: "" for u in controls}}
    rows = []
    for unit, y in {**controls, **treated}.items():
        for t in range(T):
            rows.append({"location": unit, "time": t, "Y": y[t],
                         "cell": cells[unit], "post": int(t >= pre)})
    return pd.DataFrame(rows), pre


# === dataprep ===

def test_dataprep_resolves_cells_control_and_split():
    df, pre = _panel()
    prep = multicell_dataprep(df, "location", "time", "Y",
                              cell_column_name="cell", post_col="post")
    assert set(prep["cell_map"]) == {"A", "B"}
    assert set(prep["cell_map"]["A"]) == {"a0", "a1"}
    assert set(prep["control_units"]) == {f"c{i}" for i in range(6)}
    assert prep["pre_periods"] == pre


def test_dataprep_explicit_control_label():
    df, _ = _panel()
    df["cell"] = df["cell"].replace("", "ctrl")             # explicit control label
    prep = multicell_dataprep(df, "location", "time", "Y",
                              cell_column_name="cell", post_col="post",
                              control_label="ctrl")
    assert len(prep["control_units"]) == 6 and set(prep["cell_map"]) == {"A", "B"}


def test_dataprep_cell_varies_within_unit_raises():
    df, _ = _panel()
    df.loc[(df["location"] == "a0") & (df["time"] == 0), "cell"] = "B"
    with pytest.raises(MlsynthDataError, match="constant per unit"):
        multicell_dataprep(df, "location", "time", "Y",
                           cell_column_name="cell", post_col="post")


def test_dataprep_no_control_raises():
    df, _ = _panel()
    df["cell"] = df["cell"].replace("", "A")                # everyone is treated
    with pytest.raises(MlsynthDataError, match="no control"):
        multicell_dataprep(df, "location", "time", "Y",
                           cell_column_name="cell", post_col="post")


def test_dataprep_missing_column_raises():
    df, _ = _panel()
    with pytest.raises(MlsynthConfigError, match="nope"):
        multicell_dataprep(df, "location", "time", "Y",
                           cell_column_name="nope", post_col="post")


# === analyze ===

def test_analyze_per_cell_reports_and_donor_exclusion():
    df, pre = _panel(effA=8.0, effB=0.0)
    prep = multicell_dataprep(df, "location", "time", "Y",
                              cell_column_name="cell", post_col="post")
    res = analyze_multicell(prep["Ywide"], prep["cell_map"], prep["control_units"],
                            prep["pre_periods"], ns=200, seed=0)
    assert isinstance(res, MultiCellResults)
    assert set(res.cells) == {"A", "B"}
    # the OTHER cell's markets are never donors (only controls can be)
    for label, other in [("A", {"b0", "b1"}), ("B", {"a0", "a1"})]:
        donors = set(res.cells[label].weights.donor_weights)
        assert donors.isdisjoint(other)
        assert donors <= {f"c{i}" for i in range(6)}


def test_analyze_detects_effect_cell_not_null_cell():
    df, pre = _panel(effA=10.0, effB=0.0)
    prep = multicell_dataprep(df, "location", "time", "Y",
                              cell_column_name="cell", post_col="post")
    res = analyze_multicell(prep["Ywide"], prep["cell_map"], prep["control_units"],
                            prep["pre_periods"], ns=400, seed=0)
    assert res.cells["A"].inference.p_value < 0.10            # real effect detected
    assert res.cells["B"].inference.p_value > 0.10            # null cell not rejected
    assert res.cells["A"].effects.att > res.cells["B"].effects.att


def test_analyze_winner_when_intervals_separate():
    df, pre = _panel(effA=12.0, effB=0.0)
    prep = multicell_dataprep(df, "location", "time", "Y",
                              cell_column_name="cell", post_col="post")
    res = analyze_multicell(prep["Ywide"], prep["cell_map"], prep["control_units"],
                            prep["pre_periods"], ns=400, seed=0)
    assert len(res.comparison) == 1
    row = res.comparison[0]
    assert {row["cell_a"], row["cell_b"]} == {"A", "B"}
    # huge A effect, null B -> A's ATT CI should sit above B's -> A wins
    assert res.winner == "A"


# === estimator end-to-end ===

def test_estimator_end_to_end():
    df, _ = _panel(effA=10.0, effB=0.0)
    res = MULTICELLGEOLIFT({
        "df": df, "outcome": "Y", "unitid": "location", "time": "date" if False else "time",
        "cell_column_name": "cell", "post_col": "post", "ns": 300,
        "display_graphs": False,
    }).fit()
    assert isinstance(res, MultiCellResults)
    assert set(res.cells) == {"A", "B"}
    assert res.report is not None                            # DesignResult.report set


def test_single_cell_reduces_to_single_realize():
    """One cell is *identical* to the single-cell realize: same treated set, same
    (all-other-units) donor pool, so the same fit, ATT, p, and weights."""
    from mlsynth.utils.geolift_helpers.marketselect.realize import realize_design

    df, pre = _panel(effA=8.0)
    df = df.copy()
    df.loc[df["cell"] == "B", "cell"] = ""                  # B -> control: one cell left
    prep = multicell_dataprep(df, "location", "time", "Y",
                              cell_column_name="cell", post_col="post")
    assert set(prep["cell_map"]) == {"A"}
    mc = analyze_multicell(prep["Ywide"], prep["cell_map"], prep["control_units"],
                           prep["pre_periods"], how="mean", ns=200, seed=0)
    single = realize_design(prep["Ywide"], frozenset(prep["cell_map"]["A"]),
                            prep["pre_periods"], how="mean", fixed_effects=True,
                            ns=200, seed=0)
    a = mc.cells["A"]
    assert a.effects.att == pytest.approx(single.effects.att)
    assert a.inference.p_value == single.inference.p_value
    assert a.weights.donor_weights == single.weights.donor_weights
    assert mc.winner is None and mc.comparison == []        # nothing to compare


def test_estimator_bad_config_raises():
    df, _ = _panel()
    with pytest.raises(MlsynthConfigError):
        MULTICELLGEOLIFT({"df": df, "outcome": "Y", "unitid": "location",
                          "time": "time", "cell_column_name": "cell",
                          "post_col": "post", "how": "bogus"}).fit()
