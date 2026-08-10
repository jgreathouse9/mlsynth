"""Unit tests for the test suite's own shard selection.

The suite runs across parallel CI jobs via ``pytest --num-shards N --shard i``.
These pin that the split is a genuine partition -- disjoint and exhaustive -- so
no test is dropped or double-run, and that it splits whole modules, which is
what keeps module-scoped fixtures from being rebuilt in every shard.

Mirrors ``benchmarks/tests/test_shard_selection.py``, which pins the same
round-robin idiom for the benchmark runner.
"""

from __future__ import annotations

import pytest

from _shard import select_shard_modules, split_items

MODULES = [f"mlsynth/tests/test_{c}.py" for c in "abcdefghij"]


class _FakeItem:
    """Enough of a pytest item for the split: a nodeid."""

    def __init__(self, module: str, name: str) -> None:
        self.nodeid = f"{module}::{name}"


def _items(modules, per_module=3):
    return [_FakeItem(m, f"test_{i}") for m in modules for i in range(per_module)]


# --- the module selection ------------------------------------------------


def test_one_shard_keeps_everything():
    assert select_shard_modules(MODULES, shard=0, num_shards=1) == sorted(MODULES)


def test_shard_is_a_round_robin_slice():
    got = select_shard_modules(MODULES, shard=1, num_shards=3)
    assert got == sorted(MODULES)[1::3]


def test_shards_partition_exactly():
    n = 4
    shards = [select_shard_modules(MODULES, shard=i, num_shards=n)
              for i in range(n)]
    flat = [m for s in shards for m in s]
    assert sorted(flat) == sorted(MODULES)
    assert len(flat) == len(set(flat)) == len(MODULES)


def test_selection_does_not_depend_on_input_order():
    """Sorted internally, so a different collection order gives the same split.

    Collection order varies with the filesystem and with ``-p no:randomly``
    style plugins; a shard that depended on it would run different tests on
    different machines under the same flags.
    """
    forward = select_shard_modules(MODULES, shard=2, num_shards=3)
    reverse = select_shard_modules(list(reversed(MODULES)), shard=2, num_shards=3)
    assert forward == reverse


def test_duplicates_collapse():
    """A module appears once however many of its tests were collected."""
    got = select_shard_modules(MODULES + MODULES, shard=0, num_shards=2)
    assert got == sorted(MODULES)[0::2]


def test_shard_sizes_differ_by_at_most_one():
    n = 3
    sizes = [len(select_shard_modules(MODULES, shard=i, num_shards=n))
             for i in range(n)]
    assert max(sizes) - min(sizes) <= 1


def test_more_shards_than_modules_leaves_some_empty_not_broken():
    """A shard with nothing to do is a pass, not an error.

    Over-sharding is a configuration mistake that should waste a runner, not
    fail the gate on a suite where every test passed.
    """
    n = len(MODULES) + 3
    shards = [select_shard_modules(MODULES, shard=i, num_shards=n)
              for i in range(n)]
    assert sum(len(s) for s in shards) == len(MODULES)
    assert [] in shards


def test_empty_input_is_empty_output():
    assert select_shard_modules([], shard=0, num_shards=4) == []


@pytest.mark.parametrize("shard, num_shards", [
    (0, 0), (0, -1), (-1, 2), (2, 2), (5, 2),
])
def test_invalid_shard_arguments_raise(shard, num_shards):
    with pytest.raises(ValueError):
        select_shard_modules(MODULES, shard=shard, num_shards=num_shards)


# --- splitting collected items -------------------------------------------


def test_split_keeps_whole_modules_together():
    """Every test in a file lands in the same shard.

    This is the point of splitting by module and not by test. A module-scoped
    fixture is built once per shard that holds any of its tests, so splitting a
    module across shards rebuilds its fixtures in each -- which is the cost this
    is meant to remove, not multiply.
    """
    items = _items(MODULES)
    n = 3
    for i in range(n):
        kept, _ = split_items(items, shard=i, num_shards=n)
        modules = {it.nodeid.split("::")[0] for it in kept}
        for module in modules:
            in_module = [it for it in items if it.nodeid.startswith(f"{module}::")]
            assert all(it in kept for it in in_module)


def test_split_partitions_the_items():
    items = _items(MODULES)
    n = 4
    kept = [split_items(items, shard=i, num_shards=n)[0] for i in range(n)]
    flat = [it for k in kept for it in k]
    assert len(flat) == len(items)
    assert {id(it) for it in flat} == {id(it) for it in items}


def test_split_reports_the_complement_as_deselected():
    items = _items(MODULES)
    kept, dropped = split_items(items, shard=0, num_shards=2)
    assert len(kept) + len(dropped) == len(items)
    assert not ({id(i) for i in kept} & {id(i) for i in dropped})


def test_one_shard_deselects_nothing():
    items = _items(MODULES)
    kept, dropped = split_items(items, shard=0, num_shards=1)
    assert kept == items
    assert dropped == []


def test_split_preserves_collection_order_within_a_shard():
    """Ordering is untouched, so a shard runs its tests in collection order."""
    items = _items(MODULES)
    kept, _ = split_items(items, shard=1, num_shards=3)
    assert kept == [it for it in items if it in kept]


# --- the pytest options, end to end --------------------------------------
#
# The unit tests above pin the arithmetic. These run pytest itself, because the
# failure mode that matters is the wiring: an option registered but never read
# would pass every test above and shard nothing, and CI would report four green
# jobs that each ran the whole suite.


def _collect(tmp_path, *args):
    """Collect a fixed two-module tree under the given flags."""
    import os
    import pathlib
    import subprocess
    import sys

    for name, count in (("test_alpha.py", 3), ("test_beta.py", 2)):
        (tmp_path / name).write_text(
            "".join(f"def test_{i}():\n    pass\n\n" for i in range(count)))
    env = dict(os.environ, PYTHONPATH=str(pathlib.Path(__file__).resolve().parent))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "--collect-only", "-q",
         "-p", "no:cacheprovider", "-p", "_shard", *args],
        capture_output=True, text=True, cwd=str(tmp_path), env=env)
    return [line for line in proc.stdout.splitlines() if "::" in line]


def test_the_options_are_registered(tmp_path):
    """`--num-shards`/`--shard` are accepted, and the default collects all."""
    assert len(_collect(tmp_path)) == 5
    assert len(_collect(tmp_path, "--num-shards", "1", "--shard", "0")) == 5


def test_sharding_actually_partitions_a_real_collection(tmp_path):
    first = _collect(tmp_path, "--num-shards", "2", "--shard", "0")
    second = _collect(tmp_path, "--num-shards", "2", "--shard", "1")
    assert sorted(first + second) == sorted(_collect(tmp_path))
    assert not (set(first) & set(second))
    assert first and second        # neither shard is empty on two modules


def test_a_shard_holds_whole_files(tmp_path):
    first = _collect(tmp_path, "--num-shards", "2", "--shard", "0")
    files = {line.split("::")[0] for line in first}
    assert len(files) == 1         # one of the two modules, all of it
    assert len(first) in (2, 3)


def test_the_repo_conftest_wires_the_plugin_in():
    """`_shard`'s hooks are reachable from the suite's own conftest.

    The tests above load `_shard` explicitly with `-p`, which proves the hooks
    work but not that the real suite has them. A CI job passing `--num-shards`
    against a conftest that never imported them fails on an unrecognized
    argument, so this pins the import that makes the flags exist.
    """
    import conftest

    import _shard

    assert conftest.pytest_addoption is _shard.pytest_addoption
    assert (conftest.pytest_collection_modifyitems
            is _shard.pytest_collection_modifyitems)
