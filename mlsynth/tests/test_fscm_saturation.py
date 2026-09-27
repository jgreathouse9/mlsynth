"""Where the forward scan saturates, and what the selection path does there.

Past the point where the in-sample score stops improving, every remaining
candidate scores identically -- 32 of 38 on Proposition 99, to the last digit --
and each gives a unique optimum with the added donor at exactly zero. So all of
those models are the same model, and which donor greedy adds is decided by the
order candidates happen to be evaluated in.

The out-of-sample score does not follow. The rolling CV refits on shorter
windows, where saturation has not happened and the added donor is not rejected,
so ``test_rmspe`` moves past the saturation point: 2.816, 2.893, 2.895 at sizes
6, 7, 8 on Proposition 99 while the in-sample sum of squares is pinned. That
curve is what ``optimal_size`` is the argmin of, so a size chosen past
saturation is chosen from numbers that depend on a tie-break and not on the
panel.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.fscm_helpers.estimation import _forward_select, scan_candidates
from mlsynth.utils.weights import solve_weights


def _inputs(name):
    import pandas as pd
    from pathlib import Path
    from mlsynth.utils.fscm_helpers.setup import prepare_fscm_inputs
    root = Path(__file__).resolve().parents[2]
    if name == "prop99":
        df = pd.read_csv(root / "basedata" / "P99data.csv")
        u, t, o, tr, yr = "state", "year", "cigsale", "California", 1989
    else:
        df = pd.read_csv(root / "basedata" / "basque_data.csv")
        df = df[df["regionname"] != "Spain (Espana)"]
        u, t, o = "regionname", "year", "gdpcap"
        tr, yr = "Basque Country (Pais Vasco)", 1975
    df["treat"] = ((df[u] == tr) & (df[t] >= yr)).astype(int)
    return prepare_fscm_inputs(df, unitid=u, time=t, outcome=o, treat="treat")


def _run(inputs):
    origins = np.arange(max(2, inputs.T0 // 2), inputs.T0)
    return _forward_select(inputs, origins, inputs.n_donors, Pt=None, Pd=None, v=None)


@pytest.fixture(scope="module")
def prop99():
    return _inputs("prop99")


@pytest.fixture(scope="module")
def basque():
    return _inputs("basque")


# --------------------------------------------------------------------------
# What a tie is, and what it is not
# --------------------------------------------------------------------------
def test_a_tied_candidate_gives_a_unique_optimum_with_the_donor_at_zero(prop99):
    """A tie across candidate sets is not a continuum inside one. Each set has
    exactly one minimiser; it simply puts nothing on the donor just added."""
    X = prop99.Y[:prop99.T0]
    y = prop99.y[:prop99.T0]
    sel, rem = [], list(range(prop99.n_donors))
    for _ in range(6):
        j, _, _ = scan_candidates(X, y, sel, rem)
        sel.append(j); rem.remove(j)

    for j in rem[:5]:
        sol = solve_weights(X[:, sel + [j]], y)
        assert sol.unique is True
        assert sol.free_directions.shape[1] == 0
        assert float(np.array(sol.weights)[-1]) == 0.0


# --------------------------------------------------------------------------
# The path records where it stopped being determined
# --------------------------------------------------------------------------
@pytest.mark.parametrize("panel,expected", [("prop99", 6), ("basque", 3)])
def test_the_path_reports_where_the_scan_saturated(request, panel, expected):
    _, path = _run(request.getfixturevalue(panel))
    assert path.saturated_at == expected


@pytest.mark.parametrize("panel", ["prop99", "basque"])
def test_the_path_stops_at_saturation(request, panel):
    inputs = request.getfixturevalue(panel)
    _, path = _run(inputs)
    assert len(path.order) == path.saturated_at
    assert len(path.train_rmspe) == path.saturated_at
    assert len(path.test_rmspe) == path.saturated_at
    assert path.sizes.tolist() == list(range(1, path.saturated_at + 1))


@pytest.mark.parametrize("panel", ["prop99", "basque"])
def test_the_in_sample_path_is_strictly_improving_where_it_is_kept(request, panel):
    """Every step retained bought something; that is what stopping means."""
    _, path = _run(request.getfixturevalue(panel))
    train = np.asarray(path.train_rmspe, float)
    assert np.all(np.diff(train) < -1e-12)


# --------------------------------------------------------------------------
# Truncating does not move the answer on either canonical panel
# --------------------------------------------------------------------------
@pytest.mark.parametrize("panel,size,donors", [
    ("prop99", 3, {"Montana", "Nevada", "Utah"}),
    ("basque", 2, {"Cataluna", "Madrid (Comunidad De)"}),
])
def test_the_chosen_size_and_set_are_unchanged(request, panel, size, donors):
    inputs = request.getfixturevalue(panel)
    selected, path = _run(inputs)
    assert path.optimal_size == size
    assert {str(inputs.donor_labels[j]) for j in selected} == donors


@pytest.mark.parametrize("panel", ["prop99", "basque"])
def test_the_chosen_size_lies_inside_the_determined_region(request, panel):
    """The condition under which truncating cannot have changed anything."""
    _, path = _run(request.getfixturevalue(panel))
    assert path.optimal_size <= path.saturated_at


# --------------------------------------------------------------------------
# And when it does not, the caller is told
# --------------------------------------------------------------------------
def test_a_size_chosen_at_the_saturation_boundary_is_warned_about(prop99, monkeypatch):
    """Forced by making the CV curve fall monotonically, so its argmin is the
    last size kept. The set there is the one the tie-break happened to reach."""
    import mlsynth.utils.fscm_helpers.estimation as estimation

    calls = {"n": 0}

    def descending(*args, **kwargs):
        calls["n"] += 1
        return 10.0 - calls["n"]

    monkeypatch.setattr(estimation, "_rolling_origin_rmspe", descending)
    with pytest.warns(UserWarning, match="saturat"):
        _, path = _run(prop99)
    assert path.optimal_size == path.saturated_at
