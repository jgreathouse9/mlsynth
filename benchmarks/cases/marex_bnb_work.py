"""MAREX's branch and bound's work, pinned on fixed panels (#841).

Above twenty million treated sets in a cluster, MAREX's exact backend searches
the separable designs by a certified branch and bound
(:mod:`mlsynth.utils.marex_helpers.branch`), whose node bound is the persp+N
semidefinite relaxation. This case runs the search on fixed panels and pins how
much work it does: the bounds it computes and the sets it prices, beside the
design it returns and its certificate.

It is a regression gate, not a replication. It fails when a change makes the
search compute more bounds or price more sets for the same answer, which the
tests, which check the answer, would not notice.

The instances
-------------
* Walmart's 45 stores with six treated, the panel and configuration of
  ``benchmarks/cases/marex_walmart.py`` as the search sees them, under the
  default costs: the search computes two bounds and prices the rest, the
  ``docs/marex.rst`` table's row.
* GeoLift's first 20 markets with five treated and its first 24 with six, with
  pricing made expensive (``price_seconds=1.0``) so the search bounds every
  node it can. A bound costs a second or more at forty markets, so these keep
  the case to seconds while still exercising the bound, its certificate and the
  split.

The panels are those of ``benchmarks/studies/marex_bnb``, the study the bound
came from. On the GeoLift subsets the design is checked against the
enumeration, which is cheap at these sizes.
"""
from __future__ import annotations

import warnings

#: Pricing made expensive, so the search bounds every node it can.
_BOUND_EVERYTHING = dict(price_seconds=1.0)


def run() -> dict:
    from benchmarks.studies.marex_bnb.panels import geolift, walmart
    from mlsynth.utils.marex_helpers.branch import branch_and_bound
    from mlsynth.utils.marex_helpers.exact import search_cluster
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        B, _ = geolift()
        for name, n, m in (("geolift20_m5", 20, 5), ("geolift24_m6", 24, 6)):
            Bn = B[:, :n]
            An = Bn.mean(axis=1)
            res = branch_and_bound(Bn, An, [m], filt=SubsetFilter(), **_BOUND_EVERYTHING)
            best = search_cluster(Bn, An, [m], filt=SubsetFilter())[0]
            d, c = res.designs[0], res.certificate
            out[f"{name}_bounds"] = float(res.lifts)
            out[f"{name}_priced"] = float(res.priced)
            out[f"{name}_matches_enumeration"] = float(d.treated == best.treated)
            out[f"{name}_relative_gap"] = float(c.gap) / float(c.upper)
            out[f"{name}_unresolved"] = float(len(c.unresolved))
        Bw, Aw = walmart()
        res = branch_and_bound(Bw, Aw, [6], filt=SubsetFilter())
        d, c = res.designs[0], res.certificate
        out["walmart_m6_bounds"] = float(res.lifts)
        out["walmart_m6_priced"] = float(res.priced)
        out["walmart_m6_value"] = float(d.value)
        out["walmart_m6_is_the_known_optimum"] = float(d.treated == (1, 2, 11, 26, 31, 39))
        out["walmart_m6_relative_gap"] = float(c.gap) / float(c.upper)
        out["walmart_m6_unresolved"] = float(len(c.unresolved))
    return out


# Tolerances.
#
# Measured under benchmarks/constraints.txt (Python 3.11, numpy 2.4.6, scipy
# 1.17.1, clarabel 0.11.1) and checked against Python 3.13 with numpy 2.5.3 and
# scipy 1.18.1: every count, design and value agreed exactly, so the counts are
# pinned exactly. A change in the bounds computed or the sets priced is what
# this case exists to report, and no version noise was measured to excuse one.
#
# The certificates' relative gaps were 1e-13 to 1e-12 on these panels; 1e-9
# leaves room for a differently built solver while staying far below the
# spacing of distinct designs. The Walmart value is the docs table's optimum,
# to the relative 1e-9 the docs report it at.
EXPECTED = {
    "geolift20_m5_bounds": (16.0, 0.0),
    "geolift20_m5_priced": (1.0, 0.0),
    "geolift20_m5_matches_enumeration": (1.0, 0.0),
    "geolift20_m5_relative_gap": (0.0, 1e-9),
    "geolift20_m5_unresolved": (0.0, 0.0),
    "geolift24_m6_bounds": (14.0, 0.0),
    "geolift24_m6_priced": (1.0, 0.0),
    "geolift24_m6_matches_enumeration": (1.0, 0.0),
    "geolift24_m6_relative_gap": (0.0, 1e-9),
    "geolift24_m6_unresolved": (0.0, 0.0),
    "walmart_m6_bounds": (2.0, 0.0),
    "walmart_m6_priced": (1_086_008.0, 0.0),
    "walmart_m6_value": (0.03101918869178227, 0.03101918869178227 * 1e-9),
    "walmart_m6_is_the_known_optimum": (1.0, 0.0),
    "walmart_m6_relative_gap": (0.0, 1e-9),
    "walmart_m6_unresolved": (0.0, 0.0),
}
