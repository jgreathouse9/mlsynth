"""The test beds every change to the TBR certified search is measured on first (#965).

A bed is a fixed list of treatment groups on simulated panels (brute.geo_panel: one
trend, unit sizes that differ, AR(1) noise) whose certificates branch when every node
is bounded (slopefree.PRICE_LIMIT = 0). Small panels of the same form stand in for the
large ones: the counts that decide a change (nodes, Clarabel solves, groups left open
at the depth limit) move the same way on them, in seconds, not hours, and a per-solve
cost table projects the time at 100 geos. Where a group has at most
``ENUMERATION_LIMIT`` candidate controls its design is also checked against the
enumerated optimum, so a change that certifies a wrong design fails the run.

    python -m benchmarks.studies.tbr_bnb.beds                     # every bed, against the baseline
    python -m benchmarks.studies.tbr_bnb.beds --bed s15_k3 --workers 4
    python -m benchmarks.studies.tbr_bnb.beds --record            # rewrite the baseline (on main)

Each group was kept from a scan of random treatment groups for branching on main at
a460c434-764f703c: s15_k3 from 600 groups of 15 units with 3 treated, s20_k3 from 320
of 20 with 3, and the others from the earlier scans (#1003). The baseline,
``results/beds_baseline.json``, records every group on main."""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from mlsynth.exceptions import MlsynthConfigError

from . import slopefree
from .brute import geo_panel, positive_optimum
from .treatment import _split

#: Groups with at most this many candidate controls are checked against enumeration
#: (2^17 - 1 designs, about a second a group).
ENUMERATION_LIMIT = 17

#: The RSS descent's restarts on the beds, as in the measurements of #1003 and #1004.
BED_RESTARTS = 5

BASELINE = Path(__file__).resolve().parent / "results" / "beds_baseline.json"


@dataclass(frozen=True)
class Bed:
    """``groups`` holds (seed, treated units, periods): the panel is
    geo_panel(seed, n=periods, p=units), the treated series the sum of those units."""

    name: str
    units: int
    treated: int
    groups: Tuple[Tuple[int, Tuple[int, ...], int], ...]

    def panel(self, i: int):
        seed, T, periods = self.groups[i]
        y, X, _ = _split(geo_panel(seed, n=periods, p=self.units), T)
        return y, X


@dataclass
class GroupRun:
    seed: int
    treated: Tuple[int, ...]
    periods: int
    controls: List[int]
    certified: bool
    nodes: int
    full: int                      # Clarabel solves of the full relaxation
    shor: int                      # solves of the Shor relaxation
    open: int                      # nodes left open at the depth limit
    rounded: int
    optimum: Optional[bool]        # the design is the enumerated optimum; None above ENUMERATION_LIMIT
    seconds: float


@dataclass
class BedRun:
    bed: str
    runs: List[GroupRun] = field(default_factory=list)

    def unsound(self) -> List[GroupRun]:
        """Groups certified at a design other than the enumerated optimum."""
        return [r for r in self.runs if r.certified and r.optimum is False]

    def totals(self) -> Dict[str, float]:
        rs = self.runs
        return dict(groups=len(rs), nodes=sum(r.nodes for r in rs), full=sum(r.full for r in rs),
                    shor=sum(r.shor for r in rs), open=sum(r.open for r in rs),
                    uncertified=sum(not r.certified for r in rs), rounded=sum(r.rounded for r in rs),
                    optimum_missed=sum(r.optimum is False for r in rs), unsound=len(self.unsound()),
                    seconds=round(sum(r.seconds for r in rs), 2))


@contextmanager
def counting():
    """Count the relaxations solved, by level, restoring the solvers on exit: through
    slopefree.solve, and on the SCS workspace through Workspaces.solve (a workspace
    solve that fails and goes to Clarabel counts twice, once a solver)."""
    from .workspace import Workspaces

    n = {"full": 0, "shor": 0}
    real, real_ws = slopefree.solve, Workspaces.solve

    def solve(c, G, lam, R, solver=None):
        n[R.level] += 1
        return real(c, G, lam, R, solver=solver)

    def solve_ws(self, c, G, lam, R, **k):
        n[R.level] += 1
        return real_ws(self, c, G, lam, R, **k)

    slopefree.solve, Workspaces.solve = solve, solve_ws
    try:
        yield n
    finally:
        slopefree.solve, Workspaces.solve = real, real_ws


@contextmanager
def every_node_bounded():
    """PRICE_LIMIT = 0, so the search bounds every node instead of enumerating small
    subtrees; restored on exit."""
    before = slopefree.PRICE_LIMIT
    slopefree.PRICE_LIMIT = 0
    try:
        yield
    finally:
        slopefree.PRICE_LIMIT = before


def _bed(name) -> Bed:
    if name not in BEDS:
        raise MlsynthConfigError(f"no bed {name!r}; the beds are {', '.join(sorted(BEDS))}.")
    return BEDS[name]


def run_group(name: str, i: int, **search) -> GroupRun:
    """Certify group ``i`` of bed ``name`` with every node bounded; ``search`` goes to
    certify_slope_free (restarts defaults to BED_RESTARTS)."""
    bed = _bed(name)
    seed, T, periods = bed.groups[i]
    y, X = bed.panel(i)
    search.setdefault("restarts", BED_RESTARTS)
    t0 = time.perf_counter()
    with counting() as n, every_node_bounded():
        out = slopefree.certify_slope_free(y, X, **search)
    seconds = time.perf_counter() - t0
    optimum = None
    if X.shape[1] <= ENUMERATION_LIMIT:
        best, C = positive_optimum(y, X)
        optimum = set(out.controls) == C and abs(float(out.incumbent) - best) <= 1e-9 * max(1.0, abs(best))
    return GroupRun(seed=seed, treated=tuple(T), periods=periods, controls=sorted(out.controls),
                    certified=out.certified, nodes=out.nodes, full=n["full"], shor=n["shor"], open=len(out.open),
                    rounded=out.rounded, optimum=optimum, seconds=round(seconds, 2))


def _one(job):
    name, i, search = job
    return run_group(name, i, **search)


def run_bed(name: str, groups: Optional[Sequence[int]] = None, workers: int = 1, **search) -> BedRun:
    """Every group of bed ``name`` (or those indexed by ``groups``), on ``workers``
    processes; the runs come back in the bed's order."""
    bed = _bed(name)
    if type(workers) is not int or workers < 1:
        raise MlsynthConfigError(f"workers is a number of processes, an int at least 1, not {workers!r}.")
    if groups is None:
        groups = range(len(bed.groups))
    elif isinstance(groups, (str, bytes)) or not all(type(i) is int and 0 <= i < len(bed.groups) for i in groups):
        raise MlsynthConfigError(f"each group is an index into bed {name!r}, 0 to {len(bed.groups) - 1}, "
                                 f"not {groups!r}.")
    jobs = [(name, i, search) for i in groups]
    if workers == 1 or len(jobs) < 2:
        runs = [_one(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            runs = pool.map(_one, jobs, chunksize=1)
    return BedRun(bed=name, runs=runs)


def compare(run: BedRun, baseline: dict) -> Dict[str, Tuple[int, int, int]]:
    """(baseline, now, change) for nodes, Clarabel and Shor solves, open nodes and
    uncertified groups, over the groups of ``run`` that the baseline holds."""
    rows = {(r["seed"], tuple(r["treated"]), r["periods"]): r for r in baseline.get(run.bed, {}).get("groups", [])}
    out = {}
    for key, get in (("nodes", lambda r: r["nodes"]), ("full", lambda r: r["full"]), ("shor", lambda r: r["shor"]),
                     ("open", lambda r: r["open"]), ("uncertified", lambda r: int(not r["certified"]))):
        pairs = [(get(rows[(r.seed, r.treated, r.periods)]), get(r.__dict__))
                 for r in run.runs if (r.seed, r.treated, r.periods) in rows]
        b, now = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
        out[key] = (b, now, now - b)
    return out


def _row(r: GroupRun) -> dict:
    d = dict(r.__dict__)
    d["treated"] = list(r.treated)
    return d


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.studies.tbr_bnb.beds", description=__doc__.split("\n\n")[0])
    ap.add_argument("--bed", nargs="+", default=None, help="beds to run (default: all)")
    ap.add_argument("--groups", nargs="+", type=int, default=None, help="group indices, with one bed")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--record", action="store_true", help="write the runs as the baseline")
    ap.add_argument("--solver", choices=slopefree.SOLVERS, default=slopefree.SOLVER,
                    help="the relaxations' solver (slopefree.solve)")
    a = ap.parse_args(argv)
    names = a.bed or sorted(BEDS)
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    recorded, bad = {}, 0
    for name in names:
        run = run_bed(name, groups=a.groups, workers=a.workers, solver=a.solver)
        t = run.totals()
        print(f"{name}: {t['groups']} groups, nodes {t['nodes']}, full {t['full']}, Shor {t['shor']}, "
              f"open nodes {t['open']}, uncertified {t['uncertified']}, rounded {t['rounded']}, "
              f"optimum missed {t['optimum_missed']}, {t['seconds']} s", flush=True)
        for key, (b, now, d) in compare(run, base).items():
            print(f"    {key:12s} baseline {b:6d}  now {now:6d}  change {d:+d}")
        for r in run.unsound():
            print(f"    UNSOUND: seed {r.seed} treated {list(r.treated)} certified {r.controls}, not the enumerated optimum")
        bad += len(run.unsound())
        recorded[name] = dict(groups=[_row(r) for r in run.runs], totals=t)
    if a.record:
        BASELINE.write_text(json.dumps(dict(recorded_on=_head(), **recorded), indent=1) + "\n")
    return 1 if bad else 0


def _head() -> str:  # pragma: no cover - records the commit, read only by --record
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        return "unknown"


_S15_K3 = (
    (1, (1, 11, 14), 20), (4, (1, 12, 13), 20), (120, (3, 4, 11), 20), (123, (2, 3, 14), 20),
    (124, (10, 11, 12), 20), (129, (11, 12, 13), 20), (136, (1, 4, 5), 20), (139, (5, 11, 13), 20),
    (150, (4, 5, 10), 20), (152, (0, 8, 13), 20), (155, (4, 9, 13), 20), (156, (1, 2, 14), 20),
    (157, (0, 2, 5), 20), (165, (6, 7, 13), 20), (177, (5, 6, 8), 20), (179, (2, 12, 13), 20),
    (15, (9, 10, 12), 20), (16, (1, 5, 7), 20), (18, (1, 3, 5), 20), (22, (1, 3, 9), 20),
    (28, (3, 8, 11), 20), (29, (3, 6, 9), 20), (181, (9, 11, 13), 20), (182, (0, 2, 10), 20),
    (182, (4, 6, 8), 20), (183, (1, 8, 12), 20), (189, (5, 9, 12), 20), (196, (1, 9, 13), 20),
    (197, (12, 13, 14), 20), (206, (6, 10, 12), 20), (208, (0, 2, 8), 20), (213, (2, 6, 9), 20),
    (225, (1, 3, 12), 20), (226, (9, 11, 12), 20), (228, (8, 9, 14), 20), (234, (4, 5, 11), 20),
    (243, (6, 9, 11), 20), (244, (4, 6, 9), 20), (245, (1, 10, 14), 20), (246, (3, 8, 9), 20),
    (249, (9, 13, 14), 20), (254, (1, 2, 9), 20), (258, (4, 6, 14), 20), (267, (0, 10, 14), 20),
    (275, (2, 12, 13), 20), (279, (1, 2, 14), 20), (281, (6, 12, 13), 20), (33, (5, 6, 11), 20),
    (34, (0, 3, 9), 20), (41, (1, 3, 12), 20), (43, (6, 9, 14), 20), (47, (1, 8, 10), 20),
    (60, (2, 4, 5), 20), (66, (0, 9, 10), 20), (70, (0, 4, 14), 20), (74, (8, 10, 14), 20),
    (81, (0, 6, 14), 20), (81, (5, 9, 11), 20), (84, (2, 8, 10), 20), (87, (2, 7, 10), 20),
    (107, (4, 6, 11), 20), (113, (1, 5, 9), 20),
)
_S20_K3 = (
    (0, (10, 12, 15), 20), (1, (2, 15, 18), 20), (3, (1, 3, 14), 20), (4, (13, 17, 19), 20),
    (13, (5, 14, 17), 20), (14, (2, 13, 15), 20), (17, (7, 10, 14), 20), (25, (3, 7, 10), 20),
    (27, (0, 4, 13), 20), (32, (5, 6, 19), 20), (37, (1, 14, 17), 20), (38, (4, 6, 9), 20),
    (127, (5, 17, 18), 20), (129, (3, 7, 9), 20), (137, (4, 8, 16), 20), (141, (0, 3, 11), 20),
    (142, (11, 12, 19), 20), (148, (1, 8, 12), 20), (154, (5, 11, 19), 20), (42, (1, 13, 14), 20),
    (42, (1, 13, 15), 20), (43, (8, 9, 12), 20), (45, (13, 16, 18), 20), (48, (0, 9, 13), 20),
    (50, (14, 18, 19), 20), (52, (0, 9, 15), 20), (58, (5, 9, 11), 20), (62, (4, 5, 14), 20),
    (65, (0, 3, 11), 20), (68, (12, 16, 19), 20), (69, (0, 11, 13), 20), (77, (4, 6, 16), 20),
    (80, (0, 13, 16), 20), (86, (3, 11, 17), 20), (88, (12, 13, 16), 20), (98, (1, 2, 5), 20),
    (101, (5, 13, 17), 20), (104, (6, 12, 15), 20), (107, (1, 12, 18), 20), (108, (0, 2, 7), 20),
    (109, (3, 6, 13), 20), (110, (10, 12, 13), 20), (115, (4, 11, 13), 20), (116, (15, 17, 19), 20),
    (119, (7, 11, 18), 20),
)
_S20_K4 = (
    (1, (4, 5, 16, 17), 20), (1, (4, 5, 16, 17), 40), (1, (4, 7, 10, 12), 20), (2, (1, 6, 11, 16), 20),
    (4, (10, 12, 16, 18), 20), (5, (0, 11, 14, 16), 40), (6, (3, 6, 8, 18), 40), (7, (2, 8, 14, 15), 20),
    (8, (0, 2, 6, 17), 20), (9, (0, 12, 16, 19), 20), (9, (0, 12, 16, 19), 40), (10, (2, 3, 9, 14), 20),
    (11, (2, 9, 14, 18), 20), (11, (2, 9, 15, 17), 20), (11, (2, 9, 15, 17), 40), (12, (2, 3, 11, 17), 20),
    (12, (3, 6, 9, 10), 40), (13, (15, 17, 18, 19), 20), (16, (0, 5, 9, 11), 40), (17, (1, 10, 11, 14), 40),
    (18, (1, 13, 17, 18), 20),
)
_S30_K10 = (
    (0, (0, 6, 10, 15, 16, 18, 19, 22, 23, 28), 20), (4, (1, 3, 7, 11, 20, 22, 23, 24, 26, 28), 20),
    (6, (0, 1, 2, 13, 17, 21, 22, 24, 27, 28), 20), (8, (3, 4, 5, 6, 10, 11, 20, 21, 26, 28), 20),
    (10, (4, 6, 13, 15, 16, 19, 21, 24, 25, 27), 20), (11, (0, 2, 4, 11, 14, 15, 18, 19, 21, 28), 20),
    (15, (2, 3, 4, 14, 17, 20, 21, 26, 27, 28), 20), (20, (2, 6, 8, 9, 12, 14, 15, 18, 23, 29), 20),
    (24, (0, 6, 7, 10, 15, 16, 17, 18, 20, 25), 20), (28, (2, 4, 6, 7, 10, 16, 18, 20, 21, 22), 20),
    (31, (1, 2, 10, 11, 13, 14, 17, 19, 20, 22), 20), (33, (1, 3, 5, 9, 10, 12, 24, 25, 28, 29), 20),
    (36, (3, 8, 9, 10, 11, 20, 23, 25, 26, 28), 20),
)
_S30_K4 = (
    (2, (3, 7, 8, 22), 20), (2, (2, 9, 17, 24), 20), (3, (2, 5, 7, 21), 20), (4, (15, 19, 25, 26), 20),
    (4, (2, 8, 12, 17), 20), (7, (17, 19, 25, 26), 20), (7, (1, 6, 8, 29), 20), (10, (4, 14, 23, 29), 20),
    (13, (23, 24, 25, 27), 20), (15, (1, 12, 16, 29), 20), (16, (0, 9, 13, 18), 20), (17, (3, 4, 20, 23), 20),
    (20, (6, 10, 11, 14), 20), (21, (8, 11, 18, 21), 20), (21, (2, 8, 18, 29), 20), (22, (5, 10, 19, 20), 20),
    (22, (0, 4, 13, 17), 20), (26, (14, 19, 21, 23), 20), (27, (0, 6, 9, 19), 20), (27, (5, 8, 12, 27), 20),
    (28, (0, 1, 4, 25), 20), (30, (2, 6, 12, 21), 20), (31, (2, 14, 25, 28), 20), (31, (9, 12, 19, 20), 20),
    (32, (8, 10, 18, 19), 20),
)

BEDS: Dict[str, Bed] = {
    "s15_k3": Bed("s15_k3", 15, 3, _S15_K3),     # 12 candidate controls, enumerated
    "s20_k3": Bed("s20_k3", 20, 3, _S20_K3),     # 17, enumerated
    "s20_k4": Bed("s20_k4", 20, 4, _S20_K4),     # 16, enumerated; 20 or 40 periods
    "s30_k10": Bed("s30_k10", 30, 10, _S30_K10),  # 20, more controls than enumeration allows
    "s30_k4": Bed("s30_k4", 30, 4, _S30_K4),     # 26, more controls than periods
}


if __name__ == "__main__":  # pragma: no cover - the command line
    raise SystemExit(main())
