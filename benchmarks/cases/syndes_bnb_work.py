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
counts and never a clock, so these numbers are the same on every machine with
the same library versions, and a case can pin them without timing noise.

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
  per-size search, the swap-search cutoff, best first and the bound reported
  at the limit.

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


# Tolerances.
#
# Measured under benchmarks/constraints.txt (Python 3.11, numpy 2.4.6, scipy
# 1.17.1), the versions the pull-request suite installs, and checked against
# Python 3.13 with numpy 2.5.3 and scipy 1.18.1:
#
# * the designs, the certified flags, the sets priced and the sizes chosen
#   agreed exactly across the two, so they are pinned exactly;
# * the nodes bounded differed by one on two instances (76 against 75; 81
#   against 80), and the lifts by none, so node counts carry a 5% band;
# * the modelled work differed by up to 6%, from L-BFGS-B taking different
#   numbers of evaluations in the lift under the two scipy versions, so work
#   carries a 15% band;
# * the bound at the limit differed by 7e-6 relative, so it carries 1e-4.
#
# The bands are loose against version noise and tight against what the case
# exists to catch: a change that multiplies the work, or sends a search that
# bounds back to pricing every design, moves these by far more than 15%.
# Pinned values are the constraints run's.
EXPECTED = {
    # default costs
    "geolift30_k8_nodes": _band(51, 0.05),
    "geolift30_k8_lifts": _band(17, 0.15),
    "geolift30_k8_pruned": _band(5, 0.05),
    "geolift30_k8_priced": (1_479_291, 0.0),
    "geolift30_k8_work": _band(2.747280407999999, 0.15),
    "geolift30_k8_certified": (1.0, 0.0),
    "geolift30_k8_matches_enumeration": (1.0, 0.0),
    "geolift32_k10_nodes": _band(129, 0.05),
    "geolift32_k10_lifts": _band(20, 0.15),
    "geolift32_k10_pruned": _band(42, 0.05),
    "geolift32_k10_priced": (221_442, 0.0),
    "geolift32_k10_work": _band(2.0878994169999934, 0.15),
    "geolift32_k10_certified": (1.0, 0.0),
    "geolift32_k10_matches_enumeration": (1.0, 0.0),
    "noise30_k8_nodes": _band(40, 0.05),
    "noise30_k8_lifts": _band(5, 0.15),
    "noise30_k8_pruned": _band(8, 0.05),
    "noise30_k8_priced": (185_942, 0.0),
    "noise30_k8_work": _band(0.8733259620000007, 0.15),
    "noise30_k8_certified": (1.0, 0.0),
    "noise30_k8_matches_enumeration": (1.0, 0.0),
    # the bounds prune nothing here: every one of C(30, 8) designs is priced
    "factor30_k8_nodes": _band(75, 0.05),
    "factor30_k8_lifts": _band(8, 0.15),
    "factor30_k8_pruned": (0.0, 0.0),
    "factor30_k8_priced": (5_852_925, 0.0),
    "factor30_k8_work": _band(4.976353873000002, 0.15),
    "factor30_k8_certified": (1.0, 0.0),
    "factor30_k8_matches_enumeration": (1.0, 0.0),
    # every node bounded and lifted
    "geolift20_k8_eager_nodes": _band(44, 0.05),
    "geolift20_k8_eager_lifts": _band(35, 0.05),
    "geolift20_k8_eager_pruned": _band(22, 0.05),
    "geolift20_k8_eager_priced": (1, 0.0),
    "geolift20_k8_eager_work": _band(1.0318913750000003, 0.15),
    "geolift20_k8_eager_certified": (1.0, 0.0),
    "geolift20_k8_eager_matches_enumeration": (1.0, 0.0),
    "noise20_k8_eager_nodes": _band(30, 0.05),
    "noise20_k8_eager_lifts": _band(29, 0.05),
    "noise20_k8_eager_pruned": _band(12, 0.05),
    "noise20_k8_eager_priced": (7, 0.0),
    "noise20_k8_eager_work": _band(0.6024846729999994, 0.15),
    "noise20_k8_eager_certified": (1.0, 0.0),
    "noise20_k8_eager_matches_enumeration": (1.0, 0.0),
    "factor20_k8_eager_nodes": _band(131, 0.05),
    "factor20_k8_eager_lifts": _band(123, 0.05),
    "factor20_k8_eager_pruned": _band(62, 0.05),
    "factor20_k8_eager_priced": (8, 0.0),
    "factor20_k8_eager_work": _band(3.397196551999998, 0.15),
    "factor20_k8_eager_certified": (1.0, 0.0),
    "factor20_k8_eager_matches_enumeration": (1.0, 0.0),
    # K=None under work_limit=5: uncertified, and the bound at the limit pinned
    "factor26_free_value": _band(0.6019339302995382, 1e-9),
    "factor26_free_lower": _band(0.6006047207364451, 1e-4),
    "factor26_free_size": (13.0, 0.0),
    "factor26_free_nodes": _band(82, 0.05),
    "factor26_free_lifts": _band(11, 0.15),
    "geolift26_free_value": _band(143431.35581503826, 1e-9),
    "geolift26_free_lower": _band(143171.5590682762, 1e-4),
    "geolift26_free_size": (13.0, 0.0),
    "geolift26_free_nodes": _band(80, 0.05),
    "geolift26_free_lifts": _band(9, 0.15),
}
