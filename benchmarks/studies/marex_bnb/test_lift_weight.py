"""Checks of the #832 runner: the three prune rates return one answer, the
runner restores the library's rate, and it appends one row per variant."""
import json

import pytest

from mlsynth.utils.marex_helpers import branch

from . import lift_weight
from .panels import factor

#: A bound at five thousandths of a second and a set at a thousandth, so a
#: twelve-market panel is bounded.
COSTS = dict(price_seconds=1e-3, lift_seconds=lambda k: 5e-3)


@pytest.fixture(scope="module")
def rows():
    B, A = factor(12, 20, seed=3)
    return {v: lift_weight.run(B, A, 4, v, **COSTS) for v in lift_weight.VARIANTS}


def test_every_variant_returns_the_same_certified_design(rows):
    first = rows["weighted"]
    for row in rows.values():
        assert row["treated"] == first["treated"]
        assert row["value"] == pytest.approx(first["value"], rel=1e-12)
        assert row["lower"] <= row["value"] * (1 + 1e-12)
        assert row["value"] <= row["upper"] * (1 + 1e-12)


def test_the_unweighted_variant_lifts_at_least_as_often(rows):
    row = rows["unweighted"]
    assert 0 < row["lifts"] <= row["nodes"]
    assert rows["weighted"]["lifts"] <= row["lifts"]


def test_the_optimistic_rate_starts_at_one_and_falls_without_a_prune():
    rates = lift_weight._Optimistic()
    assert rates.rate(5) == 1.0
    rates.record(5, False)
    assert rates.rate(5) == pytest.approx((0 + 2 * 0.5) / (1 + 2))
    assert rates.rate(9) == pytest.approx(0.5)


def test_the_run_restores_the_librarys_rate():
    B, A = factor(10, 20, seed=4)
    lift_weight.run(B, A, 3, "unweighted", **COSTS)
    assert branch.PruneRates is lift_weight.VARIANTS["weighted"]


@pytest.mark.parametrize("name", ["walmart", "geolift", "factor", "noise"])
def test_the_panels(name, monkeypatch):
    import benchmarks.studies.marex_bnb.panels as panels
    monkeypatch.setattr(panels, "walmart", lambda: "walmart")
    monkeypatch.setattr(panels, "geolift", lambda: "geolift")
    got = lift_weight.panel(name)
    if name in ("walmart", "geolift"):
        assert got == name
    else:
        assert got[0].shape == (40, 40)


def test_the_runner_appends_a_row_per_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(lift_weight, "RESULTS", tmp_path)
    monkeypatch.setattr(lift_weight, "panel", lambda name: factor(10, 20, seed=5))
    monkeypatch.setattr(lift_weight, "run",
                        lambda B, A, m, v, **k: {"m": m, "variant": v})
    assert lift_weight.main(["factor", "3", "weighted", "unweighted"]) == 0
    lines = (tmp_path / "lift_weight.jsonl").read_text().splitlines()
    assert [json.loads(x)["variant"] for x in lines] == ["weighted", "unweighted"]
    assert lift_weight.main(["factor", "3"]) == 0
    lines = (tmp_path / "lift_weight.jsonl").read_text().splitlines()
    assert len(lines) == 2 + len(lift_weight.VARIANTS)
