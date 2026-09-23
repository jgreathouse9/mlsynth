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

The split is the round-robin ``sorted(modules)[shard::num_shards]``, the same
idiom ``benchmarks/run_benchmarks.py`` uses for benchmark cases. Sorting first
makes it independent of collection order, so the same flags select the same
tests on any machine. Round-robin over the sorted list also spreads a family of
related files -- ``test_geox*.py``, which sort adjacently and are slow together
-- across different shards instead of stacking them in one.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def _validate(shard: int, num_shards: int) -> None:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1; got {num_shards}.")
    if not 0 <= shard < num_shards:
        raise ValueError(
            f"shard must be in [0, {num_shards}); got {shard}.")


def select_shard_modules(module_paths: Iterable[str], shard: int,
                         num_shards: int) -> List[str]:
    """The module paths this shard owns.

    Parameters
    ----------
    module_paths : iterable of str
        Every collected module, in any order and with repeats (one per test is
        fine -- they collapse).
    shard : int
        0-based index of this shard.
    num_shards : int
        How many shards the suite is split into.

    Returns
    -------
    list of str
        The sorted slice ``sorted(set(module_paths))[shard::num_shards]``. Empty
        when this shard has nothing, which happens when there are more shards
        than modules and is a pass, not an error.

    Raises
    ------
    ValueError
        If ``num_shards < 1`` or ``shard`` is outside ``[0, num_shards)``.
    """
    _validate(shard, num_shards)
    return sorted(set(module_paths))[shard::num_shards]


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


__all__ = ["select_shard_modules", "split_items"]


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
