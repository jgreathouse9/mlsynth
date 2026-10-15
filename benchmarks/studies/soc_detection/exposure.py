"""Sideways blast radius: which other estimators reach SCIP the way MAREX does.

The collapse needs two things: a quadratic atom (``sum_squares`` and its
relatives) and cvxpy's SCIP interface, which writes each one as the tied
rotated-cone identity. A model written directly in PySCIPOpt, or a cvxpy
model with no quadratic atom, cannot produce it.

Read off the source, that leaves SYNDES: its formulation builds cvxpy
quadratic atoms and solves them on SCIP. PANGEO's MIP has no quadratic atom
and PDA's HCW writes PySCIPOpt directly.

The source reading was half right. SYNDES's default for ``two_way_global``
is ``backend='exact'``, which searches treated sets directly and solves no
mixed-integer program, so the default path never reaches SCIP: wrapping
``cvxpy.Problem.solve`` records no call at all. Only the MIP path does -- an
explicit ``backend='mip'``, or the other two modes, which force it. That path
is measured here: ``cvxpy.Problem.solve`` is wrapped so the SCIP model SYNDES
builds can be captured without touching SYNDES, and each captured model gets
the check ``causes.py`` gives MAREX -- negative squares as written, negative
squares after presolve, and the SOC handler's participation.

    python exposure.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import cvxpy as cp
import numpy as np
import pandas as pd
from pyscipopt import Model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instruments import handler_rows, negative_squares, presolved_rows

from mlsynth import SYNDES
from mlsynth.config_models import SYNDESConfig


def panel(n_units: int = 10, T: int = 14, n_post: int = 4, seed: int = 0):
    """The fixture ``mlsynth/tests/test_syndes.py`` uses."""
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((T, n_units)) * 0.4
    Y += np.linspace(0, 1, T)[:, None]
    Y += rng.standard_normal(n_units)
    return pd.DataFrame([{"unit": j, "time": t, "y": float(Y[t, j]),
                          "post": int(t >= T - n_post)}
                         for j in range(n_units) for t in range(T)])


def captured_scip_models(fit) -> list:
    models, original = [], cp.Problem.solve

    def spy(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        extra = getattr(self.solver_stats, "extra_stats", None)
        if isinstance(extra, dict) and "model" in extra:
            models.append(extra["model"])
        return out

    cp.Problem.solve = spy
    try:
        fit()
    finally:
        cp.Problem.solve = original
    return models


def check(model) -> dict:
    with tempfile.NamedTemporaryFile("w+", suffix=".cip", delete=False) as tf:
        path = tf.name
    try:
        model.writeProblem(path, trans=False, verbose=False)
        written = [l for l in open(path) if "[nonlinear]" in l]
        fresh = Model()
        fresh.hideOutput()
        fresh.readProblem(path)
        fresh.presolve()
        after = presolved_rows(fresh)
        fresh.optimize()
        soc = handler_rows(fresh)["soc"]
    finally:
        os.unlink(path)
    return dict(rows=len(written),
                neg_squares_written=sum(negative_squares(r) for r in written),
                neg_squares_presolved=sum(negative_squares(r) for r in after),
                soc_participations=soc[1] if soc else 0)


def main() -> None:
    df = panel()
    runs = [("two_way_global", {}),                  # default: exact backend
            ("two_way_global", {"backend": "mip"}),
            ("one_way_global", {}), ("per_unit", {})]
    for mode, extra in runs:
        cfg = SYNDESConfig(df=df, outcome="y", unitid="unit", time="time",
                           K=2, mode=mode, post_col="post", **extra)
        models = captured_scip_models(lambda: SYNDES(cfg).fit())
        label = f"mode={mode} backend={cfg.backend}"
        print(f"SYNDES {label}: {len(models)} SCIP solve(s) captured")
        for i, model in enumerate(models[:2]):
            print(f"  solve {i}: {check(model)}")


if __name__ == "__main__":
    main()
