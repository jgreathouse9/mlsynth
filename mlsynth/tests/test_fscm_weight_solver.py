"""FSCM's trajectory-mode weights on the shared weight solver.

FSCM has two weight paths. The predictor path already runs the exact active set
-- ``bilevel/stages.py`` moved it off the FISTA primitive with a comment saying the
latter under-converges on long pre-periods and misses the Malo et al. (2024)
corner optimum on Proposition 99. The trajectory path was left on FISTA, and it
carries the same defect: it refits on expanding windows ``Y[:t]``, so every
rolling origin is a long pre-period fit and the resulting RMSPE is the criterion
forward selection ranks donors by.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.simplex import simplex_lstsq
from mlsynth.utils.weights import (
    WeightConstraint,
    WeightObjective,
    kkt_residual,
    solve_weights,
)


def _panel(name):
    import pandas as pd
    from pathlib import Path
    from mlsynth.utils.datautils import dataprep
    root = Path(__file__).resolve().parents[2]
    if name == "prop99":
        df = pd.read_csv(root / "basedata" / "P99data.csv")
        unit, time, out, treated, yr = "state", "year", "cigsale", "California", 1989
    else:
        df = pd.read_csv(root / "basedata" / "basque_data.csv")
        df = df[df["regionname"] != "Spain (Espana)"]
        unit, time, out = "regionname", "year", "gdpcap"
        treated, yr = "Basque Country (Pais Vasco)", 1975
    df["treat"] = ((df[unit] == treated) & (df[time] >= yr)).astype(int)
    p = dataprep(df, unit, time, out, "treat")
    return (np.asarray(p["donor_matrix"], float), np.asarray(p["y"], float).ravel(),
            int(p["pre_periods"]), list(p["donor_names"]))


@pytest.fixture(scope="module")
def prop99():
    return _panel("prop99")


@pytest.fixture(scope="module")
def basque():
    return _panel("basque")


def _ssr(B, A, w):
    return float(np.sum((A - B @ w) ** 2))


# --------------------------------------------------------------------------
# The solve itself
# --------------------------------------------------------------------------
@pytest.mark.parametrize("panel", ["prop99", "basque"])
def test_the_trajectory_fit_is_certified_optimal(request, panel):
    Y, y, T0, _ = request.getfixturevalue(panel)
    from mlsynth.utils.fscm_helpers.estimation import _fit_weights
    from mlsynth.utils.fscm_helpers.structures import FSCMInputs
    sol = solve_weights(Y[:T0], y[:T0])
    assert sol.status == "optimal" and sol.kkt_residual < 1e-9


@pytest.mark.parametrize("panel", ["prop99", "basque"])
def test_the_exact_fit_beats_the_fista_primitive(request, panel):
    """The defect, measured. FISTA stops short on both canonical panels."""
    Y, y, T0, _ = request.getfixturevalue(panel)
    B, A = Y[:T0], y[:T0]
    exact = np.array(solve_weights(B, A).weights)
    fista = simplex_lstsq(B, A)
    assert _ssr(B, A, exact) < _ssr(B, A, fista)
    con, obj = WeightConstraint(), WeightObjective()
    assert kkt_residual(B, A, exact, 0.0, con, obj) < 1e-9
    assert kkt_residual(B, A, fista, 0.0, con, obj) > 1e-6


@pytest.mark.parametrize("window", [10, 15, 19])
def test_every_rolling_origin_is_affected_not_just_the_full_window(prop99, window):
    """`_rolling_origin_rmspe` refits on `Y[:t]` at each origin, so the error
    is not confined to one fit -- it is in every term of the criterion."""
    Y, y, _, _ = prop99
    B, A = Y[:window], y[:window]
    exact = np.array(solve_weights(B, A).weights)
    assert _ssr(B, A, exact) < _ssr(B, A, simplex_lstsq(B, A))


# --------------------------------------------------------------------------
# Constraint set and support
# --------------------------------------------------------------------------
@pytest.mark.parametrize("panel", ["prop99", "basque"])
def test_the_weights_are_on_the_simplex_with_exact_zeros(request, panel):
    Y, y, T0, _ = request.getfixturevalue(panel)
    sol = solve_weights(Y[:T0], y[:T0])
    assert sol.weights.min() >= 0.0
    assert sol.weights.sum() == pytest.approx(1.0, abs=1e-9)
    off = sol.weights[sol.weights < 1e-8]
    assert off.size > 0 and np.all(off == 0.0)


def test_both_fscm_paths_now_run_the_same_solver(basque):
    """The predictor path already used the exact active set. After this change
    the trajectory path reaches the same minimiser, so the two modes no longer
    disagree about what the lower-level problem's answer is."""
    from mlsynth.utils.bilevel.ridge_augment import simplex_qp
    Y, y, T0, _ = basque
    assert np.array(solve_weights(Y[:T0], y[:T0]).weights) == pytest.approx(
        simplex_qp(Y[:T0], y[:T0]), abs=1e-9
    )


# --------------------------------------------------------------------------
# Wired through the estimator
# --------------------------------------------------------------------------
def _fit(panel):
    import pandas as pd
    from pathlib import Path
    from mlsynth import FSCM
    from mlsynth.utils.fscm_helpers.config import FSCMConfig
    root = Path(__file__).resolve().parents[2]
    if panel == "prop99":
        df = pd.read_csv(root / "basedata" / "P99data.csv")
        unit, time, out, treated, yr = "state", "year", "cigsale", "California", 1989
    else:
        df = pd.read_csv(root / "basedata" / "basque_data.csv")
        df = df[df["regionname"] != "Spain (Espana)"]
        unit, time, out = "regionname", "year", "gdpcap"
        treated, yr = "Basque Country (Pais Vasco)", 1975
    df["treat"] = ((df[unit] == treated) & (df[time] >= yr)).astype(int)
    return FSCM(FSCMConfig(df=df, unitid=unit, time=time, outcome=out,
                           treat="treat", display_graphs=False)).fit()


@pytest.mark.parametrize("panel", ["prop99", "basque"])
def test_the_selected_weights_are_the_exact_optimum_on_the_selected_pool(request, panel):
    """Whatever donors forward selection lands on, the weights on them are the
    certified minimiser over that pool and not a point near it."""
    Y, y, T0, names = request.getfixturevalue(panel)
    res = _fit(panel)
    carried = {k: v for k, v in res.weights.donor_weights.items() if v > 1e-9}
    idx = [names.index(k) for k in carried]
    w_reported = np.array([carried[names[j]] for j in idx])

    sol = solve_weights(Y[:T0][:, idx], y[:T0])
    assert w_reported == pytest.approx(np.array(sol.weights), abs=1e-6)
    assert sol.kkt_residual < 1e-9


@pytest.mark.parametrize("panel", ["prop99", "basque"])
def test_the_solver_label_names_the_shared_layer(request, panel):
    """The label is what tells a reader which of the two paths ran, and it
    named the primitive that is no longer used."""
    meta = _fit(panel).metadata
    assert meta["matching_mode"] == "trajectory"
    assert meta["solver"] == "simplex:active-set"


# --------------------------------------------------------------------------
# End to end. The final answer barely moves, and the reason is instructive:
# forward selection hands the solver a two- or three-donor pool, and FISTA
# converges essentially exactly on one that small. The large gaps above are on
# the full pool, which FSCM only touches while sweeping candidates.
# --------------------------------------------------------------------------
def test_forward_selection_picks_the_same_donors_as_before_the_migration(prop99, basque):
    """The selection criterion changed value; it did not change its argmin on
    either canonical panel."""
    assert set(k for k, v in _fit("prop99").weights.donor_weights.items() if v > 1e-9) == {
        "Montana", "Nevada", "Utah"}
    assert set(k for k, v in _fit("basque").weights.donor_weights.items() if v > 1e-9) == {
        "Cataluna", "Madrid (Comunidad De)"}


def test_prop99_att_moves_to_the_certified_optimum():
    """-20.150227 before, -20.151918 after. The move is 0.0017 and it is the
    right direction: on the selected pool the new weights attain a pre-period
    SSR of 73.944458 against FISTA's 73.944477, with a KKT residual below 1e-9
    where FISTA's is not. The pin is the certified minimiser, not a re-fit of
    whatever the new code happened to print."""
    assert float(_fit("prop99").effects.att) == pytest.approx(-20.151918, abs=1e-5)


def test_basque_att_does_not_move_at_all():
    """Two donors, and FISTA was already within 1.1e-10 of the optimum there."""
    assert float(_fit("basque").effects.att) == pytest.approx(-0.701495, abs=1e-6)


def test_each_rolling_origin_seeds_the_next_with_a_feasible_point(monkeypatch):
    """The warm start is a seed and not solver state, so it has to be a point
    of the simplex: the active set discards an infeasible one and every origin
    silently pays for a cold solve. That costs only work, so nothing asserting
    on weights can see it -- this asserts on the seed.
    """
    import mlsynth.utils.fscm_helpers.estimation as estimation

    seeds = []
    real = estimation.solve_weights

    def spy(B, A, *args, warm_start=None, **kwargs):
        seeds.append(None if warm_start is None else np.asarray(warm_start, float))
        return real(B, A, *args, warm_start=warm_start, **kwargs)

    monkeypatch.setattr(estimation, "solve_weights", spy)
    _fit("basque")

    carried = [s for s in seeds if s is not None]
    assert carried, "no rolling origin ever seeded the next"
    for s in carried:
        assert s.min() >= 0.0
        assert s.sum() == pytest.approx(1.0, abs=1e-9)
