"""Runner for the #839 study: one JSON line per search, appended under ``results/``.

From the repository root::

    python -m benchmarks.studies.bnb_branching.run optimum factor 40 8
    python -m benchmarks.studies.bnb_branching.run syndes factor 40 8 root,easy swap
    python -m benchmarks.studies.bnb_branching.run marex walmart 8 largest,pseudo on,off best

``optimum`` certifies the instance's optimum with the library's search and no
work limit, once, into ``results/optimum_<panel>_<N>_<K>.json``; the
``syndes`` rows read it to time the incumbent. Rules and seeds are comma
lists; a missing list of rules means every rule, of seeds the library's.
``marex`` rows take a size, rules, the guess (``on``, ``off``) and node orders,
and time the incumbent against the optimum the MAREX study certified
(``benchmarks/studies/marex_bnb/library_search.py``).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

#: Where the runner appends its rows.
RESULTS = Path(__file__).resolve().parent / "results"


def _panel(name: str, N: int):
    from benchmarks.studies.syndes_global_bnb.run import PANELS
    return PANELS[name](N)


def _optima() -> dict:
    """Every certified optimum, one file per instance so runs can share them."""
    table = {}
    for path in sorted(RESULTS.glob("optimum_*.json")):
        table.update(json.loads(path.read_text()))
    return table


def optimum(name: str, N: int, K: int) -> dict:
    """The library's search with no limit: the certified optimum and its cost."""
    from mlsynth.utils.syndes_helpers import branch
    from mlsynth.utils.syndes_helpers.exact import SubsetFilter
    from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

    problem = TwoWayProblem.from_panel(_panel(name, N))
    t0 = time.perf_counter()
    out = branch.branch_and_bound(problem, BoundEngine.from_problem(problem), K,
                                  SubsetFilter(), work_limit=1e9)
    return {"value": out.value, "certified": out.certified,
            "seconds": time.perf_counter() - t0, "work": out.work}


def syndes(name: str, N: int, K: int, rule: str, seeds: str) -> dict:
    from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

    from . import syndes_rules

    problem = TwoWayProblem.from_panel(_panel(name, N))
    known = _optima().get(f"{name}_{N}_{K}", {}).get("value")
    t0 = time.perf_counter()
    row = syndes_rules.search(problem, BoundEngine.from_problem(problem), K,
                              rule=rule, seeds=seeds, optimum=known)
    row.update(seconds=time.perf_counter() - t0, optimum=known)
    return row


def marex(name: str, m: int, rule: str, guess: bool, order: str) -> dict:
    from benchmarks.studies.marex_bnb.library_search import KNOWN
    from benchmarks.studies.marex_bnb.panels import geolift, walmart

    from . import marex_rules

    B, A = walmart() if name == "walmart" else geolift()
    known = KNOWN[(name, m)][0]
    t0 = time.perf_counter()
    row = marex_rules.search(B, A, m, rule=rule, guess=guess, order=order,
                             optimum=known)
    row.update(seconds=time.perf_counter() - t0, optimum=known)
    return row


def main(argv=None) -> int:
    from threadpoolctl import threadpool_limits

    args = list(sys.argv[1:] if argv is None else argv)
    kind, name = args[0], args[1]
    RESULTS.mkdir(exist_ok=True)
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "optimum":
            N, K = int(args[2]), int(args[3])
            key = f"{name}_{N}_{K}"
            (RESULTS / f"optimum_{key}.json").write_text(
                json.dumps({key: optimum(name, N, K)}, indent=1))
            return 0
        if kind == "syndes":
            from .syndes_rules import RULES, SEEDS
            N, K = int(args[2]), int(args[3])
            rules = args[4].split(",") if len(args) > 4 else list(RULES)
            seeds = args[5].split(",") if len(args) > 5 else ["swap"]
            for rule in rules:
                for s in seeds:
                    row = dict(syndes(name, N, K, rule, s), panel=name, N=N, K=K,
                               load=os.getloadavg()[0])
                    with open(RESULTS / "syndes.jsonl", "a") as fh:
                        fh.write(json.dumps(row) + "\n")
            return 0
        if kind == "marex":
            from .marex_rules import ORDERS, RULES
            m = int(args[2])
            rules = args[3].split(",") if len(args) > 3 else list(RULES)
            guesses = args[4].split(",") if len(args) > 4 else ["on"]
            orders = args[5].split(",") if len(args) > 5 else ["best"]
            for rule in rules:
                for g in guesses:
                    for order in orders:
                        row = dict(marex(name, m, rule, g == "on", order), panel=name,
                                   m=m, rule=rule, guess=g == "on", order=order,
                                   load=os.getloadavg()[0])
                        with open(RESULTS / "marex.jsonl", "a") as fh:
                            fh.write(json.dumps(row) + "\n")
            return 0
        raise SystemExit(f"unknown kind {kind!r}")


if __name__ == "__main__":  # pragma: no cover - the command line's entry; main is tested directly
    raise SystemExit(main())
