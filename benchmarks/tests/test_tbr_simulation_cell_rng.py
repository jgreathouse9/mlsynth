"""A cell of the section 5.2 study belongs to itself.

``run`` used one generator for the whole grid, so a cell's draws depended on
which cells preceded it and a cell requested alone disagreed with the same cell
inside the full grid. The published numbers in ``results/simulation.txt`` come
from one full-grid run and are internally consistent; the fault is that a
subset re-run does not reproduce them, and the disagreement reads as a finding.

The harness beside this one was built with the same fault and the measurement
is recorded there: a 13-hit swing on 200 replications for one cell.
"""
import numpy as np
import pytest

from benchmarks.studies.tbr_geo import simulation as sim


def test_a_cell_answers_the_same_alone_and_inside_a_larger_grid():
    alone = sim.run(reps=5, pres=(20,), seed=29)
    inside = sim.run(reps=5, pres=(10, 20), seed=29)
    match = [r for r in inside if r["n_pre"] == 20]
    assert len(match) == len(alone) and alone
    for a, b in zip(alone, match):
        assert (a["rho"], a["c"]) == (b["rho"], b["c"])
        assert a["cov90"] == b["cov90"]
        assert a["cov50"] == b["cov50"]
        assert a["median"] == b["median"]


def test_different_cells_draw_from_different_streams():
    first = sim.cell_rng(11, 0.5, 0.25, 20).normal(size=5)
    for other in (sim.cell_rng(11, 0.8, 0.25, 20), sim.cell_rng(11, 0.5, 0.5, 20),
                  sim.cell_rng(11, 0.5, 0.25, 40), sim.cell_rng(12, 0.5, 0.25, 20)):
        assert not np.allclose(first, other.normal(size=5))


def test_a_cell_stream_is_the_same_one_every_time_it_is_asked_for():
    a = sim.cell_rng(11, 0.5, 0.25, 20).normal(size=5)
    b = sim.cell_rng(11, 0.5, 0.25, 20).normal(size=5)
    assert np.allclose(a, b)


def test_every_cell_derives_its_stream_from_its_own_identity(monkeypatch):
    """The stream has to come from ``cell_rng``, not from a re-seed in place."""
    seen = []
    real = sim.cell_rng

    def spy(seed, rho, c, n_pre):
        seen.append((seed, rho, c, n_pre))
        return real(seed, rho, c, n_pre)

    monkeypatch.setattr(sim, "cell_rng", spy)
    sim.run(reps=2, pres=(20, 40), seed=29)

    assert len(seen) == len(sim.RHOS) * len(sim.CS) * 2
    assert len(set(seen)) == len(seen)
    assert all(s == (29, r, c, n) for s, (r, c, n) in
               zip(seen, [(r, c, n) for r in sim.RHOS for c in sim.CS
                          for n in (20, 40)]))
