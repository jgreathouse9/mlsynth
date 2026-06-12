import pytest
import numpy as np
import pandas as pd

from mlsynth.utils.geolift_helpers.marketselect.helpers.batch import run_simulations
from mlsynth.exceptions import MlsynthConfigError


def _wide_panel(T=20, seed=3):
    rng = np.random.default_rng(seed)
    base = np.arange(T) * 0.3
    cols = {u: base + rng.normal(scale=1.0, size=T) + i for i, u in enumerate("ABCDE")}
    Ywide = pd.DataFrame(cols, index=pd.Index(range(T), name="time"))
    Ywide.columns.name = "unit"
    return Ywide


def test_run_simulations_smoke_and_columns():
    Y = _wide_panel()
    cands = [frozenset(["A", "B"]), frozenset(["C", "D"])]
    df = run_simulations(Y, cands, durations=[4], lookback_window=2,
                         effect_sizes=[0.0, 0.5], ns=20, seed=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2 * 1 * 2 * 2                  # cands x durs x sims x es
    assert list(df.columns) == ["candidate", "duration", "sim", "effect_size",
                                "p_value", "att", "scaled_l2", "pre_rmspe"]
    assert ((df["p_value"] >= 0) & (df["p_value"] <= 1)).all()


def test_run_simulations_tags_every_candidate_sim_duration():
    Y = _wide_panel()
    cands = [frozenset(["A", "B"]), frozenset(["C", "D"])]
    df = run_simulations(Y, cands, [4], 3, [0.0], ns=15, seed=0)
    assert set(df["candidate"].unique()) == set(cands)
    assert set(df["sim"].unique()) == {1, 2, 3}
    assert set(df["duration"].unique()) == {4}


def test_run_simulations_grid_product_count():
    Y = _wide_panel()
    df = run_simulations(Y, [frozenset(["A", "B"])], durations=[3, 5],
                         lookback_window=2, effect_sizes=[0.0, 0.5, 1.0],
                         ns=15, seed=0)
    assert len(df) == 1 * 2 * 2 * 3


def test_run_simulations_output_is_aggregatable():
    """Long-form output drops straight into a vectorized power computation."""
    Y = _wide_panel()
    cands = [frozenset(["A", "B"]), frozenset(["C", "D"])]
    df = run_simulations(Y, cands, [4], 3, [0.0, 0.5], ns=15, seed=0)
    power = (df.assign(detected=df["p_value"] < 0.1)
               .groupby(["candidate", "duration", "effect_size"])["detected"].mean())
    assert len(power) == 4                            # 2 cands x 1 dur x 2 es
    assert ((power >= 0) & (power <= 1)).all()


def test_run_simulations_mean_aggregation_runs():
    Y = _wide_panel()
    df = run_simulations(Y, [frozenset(["A", "B"])], [4], 1, [0.0],
                         how="mean", ns=15, seed=0)
    assert len(df) == 1


def test_run_simulations_bad_lookback_raises():
    Y = _wide_panel()
    with pytest.raises(MlsynthConfigError, match="lookback_window"):
        run_simulations(Y, [frozenset(["A", "B"])], [4], 0, [0.0], ns=10)


def test_run_simulations_off_start_propagates():
    Y = _wide_panel(T=6)
    with pytest.raises(MlsynthConfigError, match="runs off the start"):
        run_simulations(Y, [frozenset(["A", "B"])], durations=[6],
                        lookback_window=1, effect_sizes=[0.0], ns=10)


def test_run_simulations_no_candidates_returns_empty_framed():
    Y = _wide_panel()
    df = run_simulations(Y, [], durations=[4], lookback_window=2,
                         effect_sizes=[0.0], ns=10, seed=0)
    assert df.empty
    assert list(df.columns) == ["candidate", "duration", "sim", "effect_size",
                                "p_value", "att", "scaled_l2", "pre_rmspe"]
