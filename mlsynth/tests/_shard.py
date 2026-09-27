"""Round-robin sharding for the test suite, so CI can run it in parallel jobs.

The suite is 7592 tests over 335 files and already runs under ``pytest -n auto``,
so the four cores of a hosted runner are spent before a job starts. The only
remaining axis is more jobs, which is what ``--num-shards N --shard i`` gives:
each job collects everything and runs its own disjoint slice.

Whole modules move together. Splitting at test granularity would balance the
shards more finely, but a module-scoped fixture is built once per shard holding
any of that module's tests, so a module split three ways builds its fixtures
three times. On this suite those fixtures are estimator fits measured in
seconds, so finer splitting buys balance and pays for it in repeated work.
Within a shard ``-n auto`` still distributes individual tests across workers, so
one slow file does not serialise its shard.

The split packs modules into shards longest-first, against measured durations
in ``_shard_durations.json``. Each module in turn goes to whichever shard is
currently cheapest, which is the LPT heuristic and is within 4/3 of the optimal
makespan.

It used to be the round-robin ``sorted(modules)[shard::num_shards]``. That is
blind to how long a module takes, and a quarter of the modules is not a quarter
of the work: shard 0 measured about twice shard 1 on every interpreter (#479),
so the cap had to clear the slowest shard while three runners sat idle, and a
healthy run was killed at 99% and reported as ``cancelled`` -- neither a pass
nor a failure. Sorting by name also stacks a slow family: ``test_geox*.py``
sort adjacently and are slow together, so whether they spread or pile up came
down to whether their positions happened to differ by ``num_shards``.

Balancing by time means shard sizes now differ freely in file count. That is
the point: one file that takes a minute is a fair share against sixty that take
a second.

A module the map has not seen is charged the median measured duration, not
zero. Charging zero would send every newly added test file to the same shard --
the map's blind spot becoming the imbalance it exists to remove -- and a new
file is likelier to be typical than free. Regenerate the map with
``python tools/measure_shard_durations.py``; it is committed so CI needs no
measurement pass, and it is advisory, so a stale entry costs balance and never
correctness.
"""

from __future__ import annotations

import json
import pathlib
import statistics
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _validate(shard: int, num_shards: int) -> None:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1; got {num_shards}.")
    if not 0 <= shard < num_shards:
        raise ValueError(
            f"shard must be in [0, {num_shards}); got {shard}.")


DURATIONS_PATH = pathlib.Path(__file__).with_name("_shard_durations.json")


def load_durations(path: "pathlib.Path | None" = None) -> Dict[str, float]:
    """The measured per-module seconds, or an empty map when absent.

    Absent is not an error: with no measurements every module is charged the
    same and the packing degrades to balancing file counts, which is where this
    started. A missing map costs balance, never correctness.
    """
    p = DURATIONS_PATH if path is None else pathlib.Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except (ValueError, OSError):            # pragma: no cover - unreadable map
        return {}                            # the map is advisory; fall back
    return {str(k): float(v) for k, v in raw.get("modules", {}).items()}


def select_shard_modules(module_paths: Iterable[str], shard: int,
                         num_shards: int,
                         durations: Optional[Mapping[str, float]] = None
                         ) -> List[str]:
    """The module paths this shard owns, packed to balance measured time.

    Parameters
    ----------
    module_paths : iterable of str
        Every collected module, in any order and with repeats (one per test is
        fine -- they collapse).
    shard : int
        0-based index of this shard.
    num_shards : int
        How many shards the suite is split into.
    durations : mapping of str to float, optional
        Seconds per module. Defaults to the committed map. A module missing
        from it is charged the median of those present, or 1.0 when the map is
        empty, so an unmeasured file is treated as typical and never as free.

    Returns
    -------
    list of str
        This shard's modules, sorted by name. Empty when this shard has
        nothing, which happens when there are more shards than modules and is a
        pass, not an error.

    Raises
    ------
    ValueError
        If ``num_shards < 1`` or ``shard`` is outside ``[0, num_shards)``.
    """
    _validate(shard, num_shards)
    modules = sorted(set(module_paths))
    if num_shards == 1:
        return modules

    known = load_durations() if durations is None else dict(durations)
    measured = [v for v in known.values() if v > 0]
    default = statistics.median(measured) if measured else 1.0

    # Longest first, name breaking ties, so the packing is a function of the
    # module set alone and not of the order pytest happened to collect it in.
    ordered = sorted(modules, key=lambda m: (-known.get(m, default), m))

    loads = [0.0] * num_shards
    bins: List[List[str]] = [[] for _ in range(num_shards)]
    for m in ordered:
        i = min(range(num_shards), key=lambda k: (loads[k], k))
        bins[i].append(m)
        loads[i] += known.get(m, default)
    return sorted(bins[shard])


def split_items(items: Sequence, shard: int, num_shards: int) -> Tuple[List, List]:
    """Partition collected pytest items into ``(selected, deselected)``.

    Collection order is preserved within each part, so a shard runs its tests in
    the order pytest collected them. With ``num_shards == 1`` everything is
    selected and nothing is deselected, which is the unsharded run.
    """
    _validate(shard, num_shards)
    if num_shards == 1:
        return list(items), []
    keep = set(select_shard_modules(
        (item.nodeid.split("::")[0] for item in items), shard, num_shards))
    selected, deselected = [], []
    for item in items:
        target = selected if item.nodeid.split("::")[0] in keep else deselected
        target.append(item)
    return selected, deselected


__all__ = ["select_shard_modules", "split_items", "load_durations"]


# --- pytest plugin ---------------------------------------------------------
#
# These hooks live here, next to the arithmetic they call, and are re-exported
# from ``conftest.py``. Keeping them in an importable module means the CI flags
# can be exercised by running pytest with ``-p _shard`` against a throwaway
# tree, which is how ``test_shard_selection.py`` checks the wiring: an option
# registered but never read would pass every unit test and shard nothing.


def pytest_addoption(parser):
    group = parser.getgroup("sharding")
    group.addoption(
        "--num-shards", type=int, default=1, dest="num_shards",
        help="split the suite into this many shards (round-robin over test "
             "modules), for parallel CI jobs")
    group.addoption(
        "--shard", type=int, default=0, dest="shard",
        help="0-based index of the shard to run (with --num-shards)")


def pytest_collection_modifyitems(config, items):
    """Keep this shard's modules and deselect the rest."""
    num_shards = config.getoption("num_shards")
    shard = config.getoption("shard")
    if num_shards == 1 and shard == 0:
        return
    try:
        selected, deselected = split_items(items, shard, num_shards)
    except ValueError as exc:
        # A bad --shard/--num-shards pair is a usage error. Reporting it as one
        # stops a typo from running an empty slice and reporting it as a pass.
        import pytest

        raise pytest.UsageError(str(exc)) from exc
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


__all__ += ["pytest_addoption", "pytest_collection_modifyitems"]
