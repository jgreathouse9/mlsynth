"""The SYNDES two-way branch and bound's work, pinned on fixed panels (#841).

Above its candidate limit the two-way exact backend searches treated sets by a
certified branch and bound (:mod:`mlsynth.utils.syndes_helpers.branch`). This
case runs that search on fixed panels and pins how much work it does: the nodes
it bounds, the lifts it computes, the nodes it discards, the designs it prices
and the work in the cost model's seconds, beside the design it returns and
whether it certified it.

It is a regression gate, not a replication. Its job is to fail when a change
makes the search do more work for the same answer, or turns it back into
pricing every design, which nothing else in the suite would notice: the tests
check the answer, and the answer is the same at any cost. The cost model reads
counts and never a clock, so the case has no timing noise. The counts still rest
on floating point: the lift is an iterative dual solve, and where it stops
depends on the BLAS kernel the processor selects, so one instance's counts
differ between processors (see the tolerances).

The instances
-------------
Three kinds, each small enough to run in seconds:

* the default costs at 30 to 32 units, where the search bounds, lifts and
  prices as it does in production: GeoLift's first 30 markets with 8 treated,
  its first 32 with 10, a panel with no structure, and a four-factor panel on
  which the bounds prune nothing and the search prices every design;
* costs that make every node a bound and a lift (``price_seconds=1.0``) at 20
  units, which exercises the lift, its certificates and the pruning on a
  search small enough to run in about ten seconds in all;
* ``K=None`` at 26 units under a small ``work_limit``, which exercises the
  per-size search, the swap-search cutoff, best first and the mirror rule
  (#838): the sizes up to 13 are searched, and at 13 only the designs that
  treat unit 0. Before the rule both searches stopped at the limit with their
  bounds under the design; with it both close within the limit.

The panels are the generators of ``benchmarks/studies/syndes_global_bnb``, the
study the search came from, so a reader can trace every instance to the panels
its design was chosen on.
"""
from __future__ import annotations

import warnings

import numpy as np

#: Costs that make every node a bound and a lift; they change where the search
#: spends time, never its answer.
_EAGER = dict(price_seconds=1.0, bound_seconds=lambda n: 0.0,
              lift_seconds=lambda n: 0.0)


def _panels():
    from benchmarks.studies.syndes_global_bnb.quality import common_shock, factor, geolift
    return [
        ("geolift30_k8", geolift(30), 8, {}),
        ("geolift32_k10", geolift(32), 10, {}),
        ("noise30_k8", common_shock(30, seed=0), 8, {}),
        ("factor30_k8", factor(30, seed=0), 8, {}),
        ("geolift20_k8_eager", geolift(20), 8, _EAGER),
        ("noise20_k8_eager", common_shock(20, seed=0), 8, _EAGER),
        ("factor20_k8_eager", factor(20, seed=0), 8, _EAGER),
    ]


def run() -> dict:
    from benchmarks.studies.syndes_global_bnb.quality import factor, geolift
    from mlsynth.utils.syndes_helpers.branch import branch_and_bound
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter, solve_synthetic_design_exact
    from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, Y, K, costs in _panels():
            problem = TwoWayProblem.from_panel(Y)
            engine = BoundEngine.from_problem(problem)
            o = branch_and_bound(problem, engine, K, SubsetFilter(), **costs)
            enumerated = solve_synthetic_design_exact(Y=Y, K=K)
            out[f"{name}_nodes"] = float(o.nodes)
            out[f"{name}_lifts"] = float(o.lifts)
            out[f"{name}_pruned"] = float(o.pruned)
            out[f"{name}_priced"] = float(o.priced)
            out[f"{name}_work"] = float(o.work)
            out[f"{name}_certified"] = float(o.certified)
            # the search's design is the enumeration's: the answer this case
            # pins the work of
            out[f"{name}_matches_enumeration"] = float(
                tuple(int(i) for i in o.treated)
                == tuple(int(i) for i in enumerated.selected_unit_indices))
        for name, Y in (("factor26_free", factor(26, seed=0)),
                        ("geolift26_free", geolift(26))):
            d = solve_synthetic_design_exact(Y=Y, K=None, candidate_limit=100_000,
                                             work_limit=5.0)
            r = d.raw_results
            out[f"{name}_value"] = float(d.objective_value)
            out[f"{name}_lower"] = float(r["lower_bound"])
            out[f"{name}_size"] = float(len(d.selected_unit_indices))
            out[f"{name}_nodes"] = float(r["n_nodes"])
            out[f"{name}_lifts"] = float(r["n_lifts"])
    return out


def _band(value: float, rel: float) -> tuple:
    return (value, abs(value) * rel)


def _span(lo: float, hi: float, rel: float) -> tuple:
    """A band holding every value from ``lo`` to ``hi``, widened by ``rel`` of
    ``hi``: for a count measured at two values on two kinds of processor."""
    return ((lo + hi) / 2.0, (hi - lo) / 2.0 + abs(hi) * rel)


# Tolerances.
#
# Measured under benchmarks/constraints.txt (Python 3.11, numpy 2.4.6, scipy
# 1.17.1), the versions the pull-request suite installs, and checked against
# Python 3.13 with numpy 2.5.3 and scipy 1.18.1:
#
# * the designs, the certified flags, the sets priced and the sizes chosen
#   agreed exactly across the two, so they are pinned exactly (one instance's
#   sets priced aside, below);
# * the nodes bounded differed by one on two instances (76 against 75; 81
#   against 80), and the lifts by none, so node counts carry a 5% band;
# * the modelled work differed by up to 6%, from L-BFGS-B taking different
#   numbers of evaluations in the lift under the two scipy versions, so work
#   carries a 15% band;
# * the bound at the limit differed by 7e-6 relative, so it carried 1e-4.
#
# The mirror rule (#838) moved the K=None instances: both searches now close
# within the limit, so their bound is the design's value and carries the
# value's 1e-9. Under the two versions the nodes were 67 and 67 (factor) and
# 69 and 67 (GeoLift), and the lifts 9 and 9, and 7 and 8, inside the bands.
#
# The lift's scale moved four instances' work and no design or certified flag.
# The lift once divided a node's parts by the largest of them, the fixed units'
# constant included; the constant grows with the fixed units, so the free part
# shrank against the regularisation schedule, and deep nodes stopped at the
# trust region where the relaxation was tighter. Scaled by the free part alone,
# more nodes prune: geolift30_k8 prices 407,330 designs against 1,479,291 and
# bounds 35 nodes against 51; geolift32_k10 prices 8,008 against 221,442 and
# bounds 56 against 129; factor20_k8_eager bounds 119 against 131, and
# factor26_free 68 against 67. Both kernel families below gave the same counts
# on all four.
#
# The bands are loose against version noise and tight against what the case
# exists to catch: a change that multiplies the work, or sends a search that
# bounds back to pricing every design, moves these by far more than 15%.
# Pinned values are the constraints run's.
#
# The same versions give two sets of counts, by processor. OpenBLAS picks its
# kernels by CPU, and measured under each kernel a GitHub runner can select
# (OPENBLAS_CORETYPE = Haswell, Zen, SkylakeX, Cooperlake, SapphireRapids) the
# counts fall into two families: the AVX2 kernels (Haswell, Zen; AMD runners
# select Zen) and the AVX-512 ones, whose counts are the pins. Every design and
# certified flag agrees. geolift20_k8_eager is the one instance that leaves its
# bands: its root lift, an L-BFGS-B solve of the regularised dual, stops after
# 553 evaluations under AVX2 and 647 under AVX-512, at bounds 1.3e-5 apart
# relative. Both bounds are valid, the children start from different
# multipliers, and the AVX2 search bounds 46 nodes, lifts 39, discards 22 and
# prices 3 where the AVX-512 one bounds 42, lifts 33, discards 21 and prices 1.
# That instance's bands span both families. Elsewhere the families agree on
# every count and differ by at most 1.9% in work, inside the bands.
#
# The bands are drawn for the runner pool. Under Sandybridge, an AVX kernel
# with no FMA that no GitHub runner selects, the case fails: at default costs
# geolift30_k8 lifts 6 nodes against 15 and prices 6.7 times as many designs,
# and noise30_k8 lifts 18 against 5 and prices 1.72 times as many.
# Every design and certified flag still agrees. Bands that wide would let a
# doubling of the work through on the processors the gate runs on.
EXPECTED = {
    # default costs
    "geolift30_k8_nodes": _band(35, 0.05),
    "geolift30_k8_lifts": _band(15, 0.15),
    "geolift30_k8_pruned": _band(7, 0.05),
    "geolift30_k8_priced": (407_330, 0.0),
    "geolift30_k8_work": _band(2.0127647309999985, 0.15),
    "geolift30_k8_certified": (1.0, 0.0),
    "geolift30_k8_matches_enumeration": (1.0, 0.0),
    "geolift32_k10_nodes": _band(56, 0.05),
    "geolift32_k10_lifts": _band(22, 0.15),
    "geolift32_k10_pruned": _band(27, 0.05),
    "geolift32_k10_priced": (8_008, 0.0),
    "geolift32_k10_work": _band(1.8161887709999993, 0.15),
    "geolift32_k10_certified": (1.0, 0.0),
    "geolift32_k10_matches_enumeration": (1.0, 0.0),
    "noise30_k8_nodes": _band(40, 0.05),
    "noise30_k8_lifts": _band(5, 0.15),
    "noise30_k8_pruned": _band(8, 0.05),
    "noise30_k8_priced": (185_942, 0.0),
    "noise30_k8_work": _band(0.8724044260000008, 0.15),
    "noise30_k8_certified": (1.0, 0.0),
    "noise30_k8_matches_enumeration": (1.0, 0.0),
    # the bounds prune nothing here: every one of C(30, 8) designs is priced
    "factor30_k8_nodes": _band(75, 0.05),
    "factor30_k8_lifts": _band(8, 0.15),
    "factor30_k8_pruned": (0.0, 0.0),
    "factor30_k8_priced": (5_852_925, 0.0),
    "factor30_k8_work": _band(5.045938000000001, 0.15),
    "factor30_k8_certified": (1.0, 0.0),
    "factor30_k8_matches_enumeration": (1.0, 0.0),
    # every node bounded and lifted
    # AVX-512 against AVX2 kernels (see above)
    "geolift20_k8_eager_nodes": _span(42, 46, 0.05),
    "geolift20_k8_eager_lifts": _span(33, 39, 0.05),
    "geolift20_k8_eager_pruned": _span(21, 22, 0.05),
    "geolift20_k8_eager_priced": _span(1, 3, 0.0),
    "geolift20_k8_eager_work": _span(0.986765189, 1.2337806850000004, 0.15),
    "geolift20_k8_eager_certified": (1.0, 0.0),
    "geolift20_k8_eager_matches_enumeration": (1.0, 0.0),
    "noise20_k8_eager_nodes": _band(30, 0.05),
    "noise20_k8_eager_lifts": _band(29, 0.05),
    "noise20_k8_eager_pruned": _band(12, 0.05),
    "noise20_k8_eager_priced": (7, 0.0),
    "noise20_k8_eager_work": _band(0.5640328329999994, 0.15),
    "noise20_k8_eager_certified": (1.0, 0.0),
    "noise20_k8_eager_matches_enumeration": (1.0, 0.0),
    "factor20_k8_eager_nodes": _band(119, 0.05),
    "factor20_k8_eager_lifts": _band(111, 0.05),
    "factor20_k8_eager_pruned": _band(56, 0.05),
    "factor20_k8_eager_priced": (8, 0.0),
    "factor20_k8_eager_work": _band(3.647998439999999, 0.15),
    "factor20_k8_eager_certified": (1.0, 0.0),
    "factor20_k8_eager_matches_enumeration": (1.0, 0.0),
    # K=None under work_limit=5, sizes up to 13 under the mirror rule (#838):
    # both close within the limit, so the bound is the value
    "factor26_free_value": _band(0.6019339302995382, 1e-9),
    "factor26_free_lower": _band(0.6019339302995382, 1e-9),
    "factor26_free_size": (13.0, 0.0),
    "factor26_free_nodes": _band(68, 0.05),
    "factor26_free_lifts": _band(8, 0.15),
    "geolift26_free_value": _band(143431.35581503826, 1e-9),
    "geolift26_free_lower": _band(143431.35581503826, 1e-9),
    "geolift26_free_size": (13.0, 0.0),
    "geolift26_free_nodes": _band(69, 0.05),
    "geolift26_free_lifts": _band(7, 0.15),
}
