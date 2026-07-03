"""Unit tests for the benchmark runner's case selection + sharding.

The daily benchmark ``suite`` job is sharded across parallel CI jobs via
``run_benchmarks.py --num-shards N --shard i``; these pin that the round-robin
split is a genuine partition (disjoint, exhaustive) so no case is dropped or
double-run, and that reference filtering still applies.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Import the driver module directly (it is a script, not an installed package).
_SPEC = importlib.util.spec_from_file_location(
    "run_benchmarks",
    Path(__file__).resolve().parents[1] / "run_benchmarks.py",
)
run_benchmarks = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_benchmarks)
select_cases = run_benchmarks.select_cases

ALL = [f"c{i}" for i in range(10)]
NEEDS = {"c2", "c7"}


def test_no_reference_filters_needs_reference():
    got = select_cases(ALL, NEEDS, with_reference=False, shard=0, num_shards=1)
    assert got == [c for c in ALL if c not in NEEDS]


def test_with_reference_keeps_all():
    got = select_cases(ALL, NEEDS, with_reference=True, shard=0, num_shards=1)
    assert got == ALL


def test_shards_partition_exactly():
    n = 4
    shards = [select_cases(ALL, NEEDS, with_reference=True, shard=i, num_shards=n)
              for i in range(n)]
    flat = [c for s in shards for c in s]
    # disjoint + exhaustive (a true partition, order-independent)
    assert sorted(flat) == sorted(ALL)
    assert len(flat) == len(set(flat)) == len(ALL)


def test_shard_is_round_robin_slice():
    got = select_cases(ALL, NEEDS, with_reference=True, shard=1, num_shards=3)
    assert got == ALL[1::3]


def test_sharding_applies_after_reference_filter():
    n = 3
    shards = [select_cases(ALL, NEEDS, with_reference=False, shard=i, num_shards=n)
              for i in range(n)]
    flat = [c for s in shards for c in s]
    kept = [c for c in ALL if c not in NEEDS]
    assert sorted(flat) == sorted(kept)  # needs-reference cases never appear
