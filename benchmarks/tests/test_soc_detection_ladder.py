"""Why SCIP's second-order-cone handler declines MAREX's cones, as tests.

MAREX hands its objective to SCIP through cvxpy, which writes each
``sum_squares(r) <= x`` as the Lorentz cone

    sum_i s_i^2 + u^2 <= tau^2,   s_i = 2 r_i,   u = 1 - x,   tau = 1 + x.

SCIP's ``nlhdlr_soc`` exists for that shape and carries the highest detect
priority of any nonlinear handler, and on MAREX's program it takes nothing.
The root-cause analysis in ``benchmarks/studies/soc_detection`` found two
causes, both necessary, and these tests are its ladder.

* Rung 1 -- the presolved row. What reaches the handlers is not the row cvxpy
  wrote: ``-4 + 4u + sum s_i^2 <= 0``, with no negative square left for the
  handler to anchor on.
* Rung 2 -- two mechanisms, each necessary. ``u`` and ``tau`` are affine in the
  same variable, so presolve substitutes one for the other (aggregation), and
  the simplifier expands the square of the resulting sum (rule POW7,
  ``expr/pow/expandmaxexponent``), so ``u^2 - tau^2`` cancels to ``-4u + 4``.
  Removing either restores the negative square and the handler takes the row.
* Rung 3 -- the statement over the domain: a cone whose right side shares a
  variable with one of its left components never survives default presolve,
  at any size, and one that does not share it always does. And the optimum is
  the same under every configuration, so the failure reaches the relaxation
  SCIP builds and nothing a MAREX result reports.
* Rung 4 -- the contracts the study ran without. Its instruments were never
  tested against known answers and three of them were wrong; its feature
  bisection encoded a standard cone where cvxpy emits a rotated one, so it had
  no power over the cause it was looking for.

Levels: smoke, unit invariants, edge, failure, property.
"""
from __future__ import annotations

import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pyscipopt import Model

from benchmarks.studies.soc_detection.causes import CIP, CONFIGS
from benchmarks.studies.soc_detection.instruments import (
    handler_rows,
    negative_squares,
    parse_statistics,
    presolved_rows,
)
from benchmarks.studies.soc_detection.minimal import build, observe

HERE = os.path.dirname(CIP)
STATS_FIXTURE = os.path.join(HERE, "stats_marex_J12.txt")
CORRECTIONS = ("no_aggregation", "no_expansion")


def _marex(params: dict) -> tuple[list[str], int, float]:
    """Presolved rows, SOC participations and optimum for the written model."""
    model = Model()
    model.hideOutput()
    model.readProblem(CIP)
    for key, val in params.items():
        model.setParam(key, val)
    model.presolve()
    rows = presolved_rows(model)
    model.optimize()
    soc = handler_rows(model)["soc"]
    return rows, (soc[1] if soc else 0), model.getObjVal()


# ------------------------------------------------------------- rung 1: smoke
def test_marex_cones_reach_the_handlers_without_a_negative_square():
    rows, soc, _ = _marex(CONFIGS["baseline"])
    assert len(rows) == 2, "the standard design emits one cone per fit term"
    assert [negative_squares(r) for r in rows] == [0, 0]
    assert all("+4*<" in r for r in rows), (
        "the cancelled pair should leave a linear term in the cone variable")
    assert soc == 0


# -------------------------------------------- rung 2: two necessary causes
@pytest.mark.parametrize("correction", CORRECTIONS)
def test_either_correction_alone_restores_marex_cones(correction):
    rows, soc, _ = _marex(CONFIGS[correction])
    assert [negative_squares(r) for r in rows] == [1, 1]
    assert soc == 2


def test_the_minimal_model_reproduces_the_failure():
    """Six squares, three continuous weights, no integers, no cvxpy."""
    obs = observe(build(tied=True, params=CONFIGS["baseline"]))
    assert obs["neg_squares"] == 0
    assert obs["soc"] == 0


def test_untying_the_cone_removes_the_failure():
    """The one-feature twin: ``u`` from its own variable instead of ``x``.

    This is the contract the feature bisection ran without. A probe that is
    meant to stand for MAREX has to reproduce MAREX's outcome first; this one
    differs from the reproducer in exactly the feature that matters, and it is
    the shape every case of the bisection had.
    """
    obs = observe(build(tied=False, params=CONFIGS["baseline"]))
    assert obs["neg_squares"] == 1
    assert obs["soc"] == 1


@pytest.mark.parametrize("correction", CORRECTIONS)
def test_either_correction_alone_restores_the_minimal_model(correction):
    obs = observe(build(tied=True, params=CONFIGS[correction]))
    assert obs["neg_squares"] == 1
    assert obs["soc"] == 1


# ------------------------------------------------ rung 3: over the domain
SIZES = dict(n=st.integers(min_value=1, max_value=20),
             k=st.integers(min_value=1, max_value=4),
             seed=st.integers(min_value=0, max_value=10_000))


@given(**SIZES)
@settings(max_examples=25, deadline=None)
def test_a_tied_cone_never_survives_default_presolve(n, k, seed):
    obs = observe(build(n=n, k=k, tied=True, seed=seed,
                        params=CONFIGS["baseline"]))
    assert obs["neg_squares"] == 0
    assert obs["soc"] == 0


@given(correction=st.sampled_from(CORRECTIONS), **SIZES)
@settings(max_examples=25, deadline=None)
def test_either_correction_restores_any_tied_cone(correction, n, k, seed):
    obs = observe(build(n=n, k=k, tied=True, seed=seed,
                        params=CONFIGS[correction]))
    assert obs["neg_squares"] == 1
    assert obs["soc"] == 1


@given(**SIZES)
@settings(max_examples=25, deadline=None)
def test_an_untied_cone_always_survives(n, k, seed):
    obs = observe(build(n=n, k=k, tied=False, seed=seed,
                        params=CONFIGS["baseline"]))
    assert obs["neg_squares"] == 1
    assert obs["soc"] == 1


@given(**SIZES)
@settings(max_examples=15, deadline=None)
def test_the_optimum_does_not_depend_on_the_configuration(n, k, seed):
    """Blast radius: the failure is in the relaxation, never in the answer."""
    values = [observe(build(n=n, k=k, tied=True, seed=seed,
                            params=params))["objective"]
              for params in CONFIGS.values()]
    ref = values[0]
    assert all(abs(v - ref) <= 1e-5 * (1.0 + abs(ref)) for v in values), values


def test_marex_optimum_does_not_depend_on_the_configuration():
    values = [_marex(params)[2] for params in CONFIGS.values()]
    assert max(values) - min(values) <= 1e-5 * (1.0 + abs(values[0])), values


# --------------------------------------------- rung 4: the instruments
def test_the_real_statistics_file_parses_to_the_measured_values():
    """A captured SCIP 10.0.2 run on MAREX at J = 12.

    It holds three rows named ``nonlinear`` in three different tables, and
    only the one in the Constraints table carries the cut counts, so a parser
    that takes the first match, or counts columns from the wrong table, fails
    here.
    """
    stats = parse_statistics(open(STATS_FIXTURE).read())
    assert stats["soc"] == (0, 0)
    assert stats["default"] == (76, 532)
    assert stats["nonlinear_cuts"] == 4627
    assert stats["nonlinear_applied"] == 4304


def test_statistics_with_a_plus_suffix_on_the_count_still_parse():
    text = (
        "Constraints        :     Number  MaxNumber  #Separate #Propagate"
        "    #EnfoLP    #EnfoRelax  #EnfoPS    #Check   #ResProp    Cutoffs"
        "    DomReds       Cuts    Applied      Conss   Children\n"
        "  linear           :         39+        43         22        592"
        "          2          0          0         35          0          1"
        "         74          0          0          0          0\n"
        "  nonlinear        :          2+         2        216        591"
        "         46          0          0         28          0          0"
        "       1430       4627       4304          0          0\n"
        "Nlhdlrs            :    Detects  DetectAll DetectTime\n"
        "  soc              :          3          7       0.00\n"
    )
    stats = parse_statistics(text)
    assert stats["nonlinear_cuts"] == 4627
    assert stats["nonlinear_applied"] == 4304
    assert stats["soc"] == (3, 7)
    assert stats["default"] is None


@pytest.mark.parametrize("row, expected", [
    ("[nonlinear] <c>: -4+4*<a>+(<b>)^2+(<c>)^2 <= 0;", 0),
    ("[nonlinear] <c>: -(<a>)^2+(<b>)^2+(<c>)^2 <= 0;", 1),
    ("[nonlinear] <c>: -((2-<a>))^2+(<a>)^2+(<b>)^2 <= 0;", 1),
    ("[nonlinear] <c>: (<a>)^2-((2+<b>))^2 <= 0;", 1),
    ("[nonlinear] <c>: -3.5*(<a>)^2+(<b>)^2 <= 0;", 1),
    ("[nonlinear] <c>: -(<b>)^2+2e-03*(<c>)^2 <= 0;", 1),
    ("[nonlinear] <c>: 1-(<t>)^2+(<s>)^2-2*<y>+(<y>)^2 <= 0;", 1),
    ("[nonlinear] <c>: (<a>)^2-<b>*<c> <= 0;", 0),
    ("[nonlinear] <c>: -(<a>)^2-(<b>)^2 <= 0;", 2),
    # the row hypothesis found: presolve flipped it to >=, and the cone's one
    # negative square is the one written with a plus
    ("[nonlinear] <c7>: -28.2636316654556+(<t_tau>)^2+36.9590047725869*<t_s2>"
     "-14.7434067614696*(<t_s2>)^2+2*<t_y>-(<t_y>)^2 >= -0;", 1),
    ("[nonlinear] <c>: -1 <= (<a>)^2-(<b>)^2 <= 4;", 1),
    ("[nonlinear] <c>: (<a>)^2-(<b>)^2 == 0;", 1),
])
def test_the_negative_square_counter_on_known_rows(row, expected):
    assert negative_squares(row) == expected


def test_an_empty_or_linear_row_has_no_negative_square():
    assert negative_squares("[nonlinear] <c>: 2*<a>-<b> <= 0;") == 0
    assert negative_squares("") == 0
