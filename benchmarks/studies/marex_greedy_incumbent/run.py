"""Runner for the greedy-incumbent study: one JSON line per search setting.

From the repository root::

    python -m benchmarks.studies.marex_greedy_incumbent.run enumeration section5 geolift
    python -m benchmarks.studies.marex_greedy_incumbent.run scip section5 2,3,4,5,6,7,8,9
    python -m benchmarks.studies.marex_greedy_incumbent.run seedcheck section5 geolift walmart
    python -m benchmarks.studies.marex_greedy_incumbent.run bnb section5 walmart
    python -m benchmarks.studies.marex_greedy_incumbent.run table

``enumeration`` writes ``results/enumeration_<panel>.jsonl``: for every treated
count ``m = 1..17`` and for the unconstrained design (``m`` from 1 to 17,
``all``), the optimum by brute force, the greedy incumbent and its certified
bound, and the library's search four ways: as it is, seeded with the greedy
incumbent, seeded with the optimum, and with the two-sided bound. ``scip``
writes ``results/scip_<panel>.jsonl``: the MIQP backend cold, warm-started at
the greedy incumbent and warm-started at the optimum, each warm start
completed by SCIP however few variables it fixes, and records whether SCIP used
it; a fourth argument ``library`` keeps the library's SCIP settings instead and
writes ``results/scip_library_<panel>.jsonl``. ``seedcheck`` writes ``results/seedcheck.jsonl``: whether SCIP uses the
warm start at the library's own settings. ``bnb`` writes
``results/bnb.jsonl``: the certified branch and bound run directly at 4, 9 and
11 treated, beside the enumeration. ``table`` prints the tables of the README
from those files.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
N_MARKETS = 18


def _bounds_below(B, A, sizes, value, two_sided):
    """Candidates whose bound is under ``value`` plus the search's margin, priced
    in the search's blocks and at its rigour."""
    import mlsynth.utils.marex_helpers.exact as ex
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter

    from .search import LIBRARY_BOUND, LIBRARY_FLOOR, two_sided_bound

    n = B.shape[1]
    rigorous = ex.count_candidates(n, sizes, SubsetFilter()) <= ex.RIGOROUS_PRICING_LIMIT
    cut = value + ex.PRUNE_MARGIN * (1.0 + abs(value))
    total = 0
    for block in ex._chunks(n, sizes, SubsetFilter()):
        if two_sided:
            lb = two_sided_bound(B, A, block, rigorous=rigorous)
        else:
            lb = LIBRARY_BOUND(B, A, block, rigorous=rigorous) + LIBRARY_FLOOR(B, A)
        total += int((lb < cut).sum())
    return total


def _first_incumbent(B, A, sizes, g, weighted):
    """The objective of the first admissible set in the library's bound order:
    the incumbent the search starts from."""
    import mlsynth.utils.marex_helpers.exact as ex
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter

    n = B.shape[1]
    full = (1 << n) - 1
    rigorous = ex.count_candidates(n, sizes, SubsetFilter()) <= ex.RIGOROUS_PRICING_LIMIT
    best = (np.inf, np.inf)
    for block in ex._chunks(n, sizes, SubsetFilter()):
        lb = ex.treated_fit_bound(B, A, block, rigorous=rigorous)
        masks = (1 << block).sum(axis=1)
        ok = weighted[masks]
        if ok.any():
            i = int(np.flatnonzero(ok)[np.argmin(lb[ok])])
            best = min(best, (float(lb[i]), float(g[masks[i]] + g[full ^ masks[i]])))
    return best[1]


def enumeration(name: str) -> list:
    """Every treated count and the unconstrained design on one panel."""
    from .brute import optima, treated_fits
    from .greedy import as_design, bracket, incumbent
    from .panels import PANELS
    from .search import run_search

    B, A = PANELS[name]()
    n = B.shape[1]
    g, weighted = treated_fits(name, B, A)
    best = optima(g, weighted, n)
    rows = []
    settings = [(m, [m]) for m in range(1, n)] + [("all", list(range(1, n)))]
    for label, sizes in settings:
        value, S = min(best[m] for m in sizes)
        optimum = as_design(B, A, S)
        clock = time.perf_counter()
        design, forward_value, evaluated = incumbent(B, A, sizes)
        greedy_seconds = time.perf_counter() - clock
        row = {"panel": name, "m": label,
               "candidates": sum(math.comb(n, m) for m in sizes),
               "optimum": value, "optimum_set": list(S),
            "first": _first_incumbent(B, A, sizes, g, weighted) / value,
            "forward": forward_value / value, "evaluated": evaluated,
            "greedy_seconds": greedy_seconds}
        if design is not None:
            upper = bracket(B, A, design)[1]
            row.update(greedy=design.value / value, greedy_set=list(design.treated),
                       certified_upper=float(upper) / value,
                       valid=bool(upper >= bracket(B, A, optimum)[0]))
        else:                                             # pragma: no cover
            row.update(greedy=None, greedy_set=None, certified_upper=None, valid=None)
        runs = {"library": run_search(B, A, sizes),
                "greedy_seed": (run_search(B, A, sizes, seed=design)
                                if design is not None else None),
                "optimum_seed": run_search(B, A, sizes, seed=optimum),
                "two_sided": run_search(B, A, sizes, two_sided=True)}
        for key, run in runs.items():
            if run is not None and not abs(run["value"] - value) <= 1e-9 * (1 + value):
                raise AssertionError(f"{name} {label} {key}: {run['value']} != {value}")
        row.update(runs)
        row["bound_below"] = _bounds_below(B, A, sizes, value, two_sided=False)
        row["two_sided_below"] = _bounds_below(B, A, sizes, value, two_sided=True)
        rows.append(row)
        print(json.dumps(row), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"enumeration_{name}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return rows


#: SCIP drops a partial solution when more than this share of the variables is
#: left unknown (``heuristics/completesol/maxunknownrate``, default 0.85).
#: MAREX's warm start fixes only the selection, so on panels with many fit
#: periods its share unknown is past the default and SCIP ignores the start.
ACCEPT_ANY_START = {"heuristics/completesol/maxunknownrate": 1.0}


def _scip_stats():
    """Wrap cvxpy's SCIP solve to record, for every solve, its node count and
    whether SCIP completed the warm start (its ``completesol`` heuristic)."""
    import re
    import tempfile

    from cvxpy.reductions.solvers.conic_solvers import scip_conif

    stats = []
    original = scip_conif.SCIP._solve

    def solve(self, model, *args, **kwargs):
        out = original(self, model, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scip.stats"
            model.writeStatistics(str(path))
            text = path.read_text()
        row = re.search(r"^\s*completesol\s*:\s*\S+\s+\S+\s+(\d+)\s+(\d+)", text, re.M)
        first = re.search(r"First Solution.*found by <([^>]*)>", text)
        stats.append({"nodes": int(model.getNNodes()), "status": model.getStatus(),
                      "seconds": float(model.getSolvingTime()),
                      "completesol_calls": int(row.group(1)) if row else 0,
                      "completesol_found": int(row.group(2)) if row else 0,
                      "first_found_by": first.group(1) if first else None})
        return out

    scip_conif.SCIP._solve = solve
    return stats


def _start_settings(accept_any: bool):
    """Pass :data:`ACCEPT_ANY_START` to every warm-started solve, or leave the
    library's settings."""
    from mlsynth.utils.marex_helpers import warmstart

    library = warmstart.solve_warmstarted
    if not accept_any:
        return library

    def solve(prob, z_var, warm_z, *, verbose=False, solver_opts=None):
        opts = dict(solver_opts or {})
        opts["scip_params"] = {**(opts.get("scip_params") or {}), **ACCEPT_ANY_START}
        return library(prob, z_var, warm_z, verbose=verbose, solver_opts=opts)

    warmstart.solve_warmstarted = solve
    return library


def scip(name: str, sizes: list, time_limit: float = 300.0,
         accept_any: bool = True) -> list:
    """The MIQP backend cold and warm-started, one treated count at a time.

    With ``accept_any`` the warm-started solves let SCIP complete any start
    (:data:`ACCEPT_ANY_START`), so the seed is used; each solve records whether
    it was.
    """
    import mlsynth.utils.marex_helpers.exact as ex
    from mlsynth.utils.marex_helpers.optimization import solve_design
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter

    from .greedy import incumbent
    from .panels import PANELS

    from mlsynth.utils.marex_helpers import warmstart

    B, A = PANELS[name]()
    n, T = B.shape[1], B.shape[0]
    stats = _scip_stats()
    library = _start_settings(accept_any)
    rows = []
    for m in sizes:
        clock = time.perf_counter()
        exact = ex.search_cluster(B, A, [m], filt=SubsetFilter())[0]
        exact_seconds = time.perf_counter() - clock
        design, _, evaluated = incumbent(B, A, [m])
        row = {"panel": name, "m": m, "optimum": exact.value,
               "exact_seconds": exact_seconds,
               "greedy": design.value / exact.value, "evaluated": evaluated}
        for key, warm in (("cold", None), ("greedy_seed", list(design.treated)),
                          ("optimum_seed", list(exact.treated))):
            stats.clear()
            clock = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = solve_design(B.T, T, np.zeros(n), m_eq=m, design="standard",
                                   warm_start=warm, time_limit=time_limit)
            treated = np.flatnonzero(np.asarray(raw["z_opt"])[:, 0] > 0.5)
            w = np.asarray(raw["w_opt"])[treated, 0]
            control = [j for j in range(n) if j not in treated]
            value = (ex.side_fit(B, A, list(treated))[0]
                     + ex.side_fit(B, A, control)[0])
            row[key] = {"seconds": time.perf_counter() - clock,
                        "nodes": sum(s["nodes"] for s in stats), "solves": len(stats),
                        "status": stats[-1]["status"] if stats else None,
                        "value": value / exact.value,
                        "weighted": bool(np.all(w > 1e-8)),
                        "seed_used": any(s["completesol_found"] for s in stats),
                        "first_found_by": stats[0]["first_found_by"] if stats else None}
        row["accept_any_start"] = accept_any
        rows.append(row)
        print(json.dumps(row), flush=True)
    warmstart.solve_warmstarted = library
    RESULTS.mkdir(exist_ok=True)
    stem = f"scip_{name}" if accept_any else f"scip_library_{name}"
    (RESULTS / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def seedcheck(names: list, m: int = 2) -> list:
    """Whether SCIP uses MAREX's warm start at the library's settings, with the
    optimum as the seed; when it does not, the unknown share SCIP reports."""
    import re
    import tempfile

    import mlsynth.utils.marex_helpers.exact as ex
    from mlsynth.utils.marex_helpers.optimization import solve_design
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter

    from .panels import PANELS

    stats = _scip_stats()
    rows = []
    for name in names:
        B, A = PANELS[name]()
        n, T = B.shape[1], B.shape[0]
        exact = ex.search_cluster(B, A, [m], filt=SubsetFilter())[0]
        stats.clear()
        # SCIP writes its log and its warnings from C, so the capture is of
        # both descriptors.
        sys.stdout.flush()
        sys.stderr.flush()
        saved = {fd: os.dup(fd) for fd in (1, 2)}
        with tempfile.TemporaryFile() as log:
            for fd in saved:
                os.dup2(log.fileno(), fd)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    solve_design(B.T, T, np.zeros(n), m_eq=m, design="standard",
                                 warm_start=list(exact.treated), time_limit=300.0,
                                 verbose=True)
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                for fd, keep in saved.items():
                    os.dup2(keep, fd)
                    os.close(keep)
            log.seek(0)
            text = log.read().decode(errors="replace")
        dropped = re.search(r"unknown rate is too large \(([0-9.]+) > ([0-9.]+)\)", text)
        row = {"panel": name, "m": m, "fit_periods": T,
               "completesol_calls": stats[-1]["completesol_calls"],
               "seed_used": stats[-1]["completesol_found"] > 0,
               "first_found_by": stats[-1]["first_found_by"],
               "unknown_rate": float(dropped.group(1)) if dropped else None,
               "unknown_rate_limit": float(dropped.group(2)) if dropped else None}
        rows.append(row)
        print(json.dumps(row), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "seedcheck.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def bnb(names: list, sizes=(4, 9, 11)) -> list:
    """The certified branch and bound at 18 markets, beside the enumeration.

    At this size its cost model prices the root set by set, so it makes no
    persp+N bound and fits what the enumeration fits.
    """
    from mlsynth.utils.marex_helpers.branch import branch_and_bound
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter

    from .panels import PANELS
    from .search import run_search

    rows = []
    for name in names:
        B, A = PANELS[name]()
        for m in sizes:
            clock = time.perf_counter()
            found = branch_and_bound(B, A, [m], filt=SubsetFilter())
            seconds = time.perf_counter() - clock
            enum = run_search(B, A, [m])
            row = {"panel": name, "m": m, "bounds": found.lifts, "priced": found.priced,
                   "fitted": found.certificate.fitted, "work": found.work,
                   "closed": found.closed, "value": found.designs[0].value,
                   "seconds": seconds, "enumeration_fitted": enum["fitted"],
                   "enumeration_value": enum["value"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "bnb.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def _rows(prefix: str) -> list:
    out = []
    for path in sorted(RESULTS.glob(f"{prefix}_*.jsonl")):
        out += [json.loads(line) for line in path.read_text().splitlines() if line]
    return out


def table() -> None:
    """The README's tables, from the result files."""
    from .panels import PANELS
    order = {name: i for i, name in enumerate(PANELS)}
    rows = sorted(_rows("enumeration"), key=lambda r: (order[r["panel"]], str(r["m"]).zfill(3)))
    print("| panel | m | sets | first / f* | greedy / f* | evaluated | walked | "
          "greedy seed | optimum seed | two-sided | seconds | two-sided seconds |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['panel']} | {r['m']} | {r['candidates']:,} | {r['first']:.2f} | "
              f"{r['greedy']:.2f} | {r['evaluated']} | {r['library']['walked']:,} | "
              f"{r['greedy_seed']['walked']:,} | {r['optimum_seed']['walked']:,} | "
              f"{r['two_sided']['walked']:,} | {r['library']['seconds']:.2f} | "
              f"{r['two_sided']['seconds']:.2f} |")
    for prefix in ("scip", "scip_library"):
        scip_rows = sorted((r for r in _rows(prefix)
                            if r.get("accept_any_start", True) == (prefix == "scip")),
                           key=lambda r: (order[r["panel"]], r["m"]))
        if not scip_rows:
            continue
        print()
        print("| panel | m | exact seconds | greedy / f* | cold nodes | greedy-seed nodes | "
              "optimum-seed nodes | seeds used | cold seconds | greedy-seed seconds | "
              "optimum-seed seconds |")
        print("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|")
        for r in scip_rows:
            used = "/".join("yes" if r[k].get("seed_used") else "no"
                            for k in ("greedy_seed", "optimum_seed"))
            print(f"| {r['panel']} | {r['m']} | {r['exact_seconds']:.3f} | {r['greedy']:.2f} | "
                  f"{r['cold']['nodes']:,} | {r['greedy_seed']['nodes']:,} | "
                  f"{r['optimum_seed']['nodes']:,} | {used} | {r['cold']['seconds']:.1f} | "
                  f"{r['greedy_seed']['seconds']:.1f} | {r['optimum_seed']['seconds']:.1f} |")


if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    if command == "enumeration":
        for panel in args:
            enumeration(panel)
    elif command == "scip":
        scip(args[0], [int(m) for m in args[1].split(",")],
             float(args[2]) if len(args) > 2 else 300.0,
             accept_any=not (len(args) > 3 and args[3] == "library"))
    elif command == "seedcheck":
        seedcheck(args)
    elif command == "bnb":
        bnb(args)
    elif command == "table":
        table()
    else:
        raise SystemExit(f"unknown command {command!r}")
