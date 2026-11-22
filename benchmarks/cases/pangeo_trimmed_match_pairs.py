"""Cross-validation: PANGEO's pairs on the supergeo paper's objective against
Trimmed Match Design's pairing, run live on Google's example data.

Trimmed Match Design (Chen and Au, "Robust causal inference for incremental
return on ad spend with randomized paired geo experiments", Ann. Appl. Stat. 16
(2022) 1-20; Chen, Longfils and Remy, arXiv:2105.07060) pairs geos by sorting
them on their share of the pretest totals and pairing neighbours
(``TrimmedMatchGeoXDesign.create_geo_pairs`` in google/trimmed_match). PANGEO
with ``objective="supergeo_total"`` and ``max_supergeo_size=1`` minimises the
sum over pairs of the squared difference of the pair's totals (Chen,
Doudchenko, Jiang, Stein and Ying, arXiv:2301.12044) by an exact-cover MIP,
solved here through cvxpy (``mip_backend="cvxpy"``), the route as shipped. With
one matching metric, the response, the sorted neighbour pairing is the optimum
of that objective, so the two must give the same pairs.

The reference runs live, unmodified: ``benchmarks/reference/trimmed_match/
install.sh`` clones the package at a pinned commit, builds its own virtual
environment, compiles its C++ core with the system compiler (two stand-in
headers replace glog's CHECK and absl's optional, in place of its Bazel build)
and runs the package's own test suite there; ``run.sh pairs.py`` then executes
the package's pairing in that environment, on
``example_data_for_design.csv`` (174 geos,
364 days of 2020) with the design colab's windows: design 2020-01-01 to
2020-12-29, evaluation 2020-12-02 to 2020-12-29, cross validation on, so pairs
are formed on the 336 days before the evaluation window. PANGEO gets the same
364 days with ``frac_E = 336 / 364``. The case skips when the reference
environment has not been built.

Measured: all 87 pairs equal, at
the sorted pairing's cost (ratio 1.000000000); the MIP over 15,051 candidate
pairs solved to optimality in 0.57 to 0.66 s. With the package's default
metrics (spend share added with weight 0.01) the reference's pairing shares 81
of 87 pairs and costs 1.000668 times the optimum on the paper's objective.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_ROOT = Path(__file__).resolve().parents[2]
_REF = _ROOT / "benchmarks" / "reference" / "trimmed_match"
_CLONE = _ROOT / "benchmarks" / "reference" / ".cache" / "trimmed_match"
_EVAL_START = "2020-12-02"
_cache: dict = {}


def _reference(metrics: str) -> list:
    """The reference's pairing, from run.sh in its environment; skips when that environment is not built."""
    key = ("ref", metrics)
    if key not in _cache:
        out = subprocess.run(["bash", str(_REF / "run.sh"), "pairs.py", metrics],
                             capture_output=True, text=True)
        if out.returncode == 3:
            raise BenchmarkSkipped(out.stderr.strip())
        if out.returncode != 0:
            raise RuntimeError(f"trimmed_match reference failed: {out.stderr.strip()[-500:]}")
        _cache[key] = sorted(tuple(sorted(p)) for p in json.loads(out.stdout)["pairs"])
    return _cache[key]


def _panel():
    if "panel" not in _cache:
        df = pd.read_csv(_CLONE / "trimmed_match" / "example_datasets" / "example_data_for_design.csv",
                         parse_dates=["date"])
        df = df[df.date.between("2020-01-01", "2020-12-29")].copy()
        df["arm"] = "A"
        df["geo"] = df.geo.astype(str)
        _cache["panel"] = df
    return _cache["panel"]


def _pangeo():
    if "pangeo" not in _cache:
        from mlsynth import PANGEO
        df = _panel()
        dates = np.sort(df.date.unique())
        n_train = int((dates < np.datetime64(_EVAL_START)).sum())
        res = PANGEO(dict(df=df, outcome="response", arm="arm", unitid="geo", time="date",
                          objective="supergeo_total", max_supergeo_size=1, fast=False,
                          mip_backend="cvxpy", frac_E=n_train / len(dates),
                          compute_power=False, display_graphs=False)).fit()
        if res.metadata["n_estimation"] != n_train:  # pragma: no cover - defensive
            raise RuntimeError("PANGEO's estimation window is not the reference's training window")
        pairs = sorted(tuple(sorted((int(p.treatment[0]), int(p.control[0]))))
                       for p in res.arm_designs["A"].pairs)
        _cache["pangeo"] = (res, pairs)
    return _cache["pangeo"]


def _totals():
    df = _panel()
    z = df[df.date < _EVAL_START].groupby("geo").response.sum()
    return {int(g): float(v) for g, v in z.items()}


def _cost(pairs, z):
    return float(sum((z[a] - z[b]) ** 2 for a, b in pairs))


def _sorted_cost(z):
    s = np.sort(np.array(list(z.values())))
    return float(((s[1::2] - s[0::2]) ** 2).sum())


def run() -> dict:
    ref = _reference("response")                    # skips without the clone or environment
    ref_default = _reference("default")
    res, pairs = _pangeo()
    z = _totals()
    best = _sorted_cost(z)
    diag = res.metadata["solver_diagnostics"]["A"]
    return {
        "pairs_shared_with_reference": float(len(set(pairs) & set(ref)) / len(ref)),
        "pangeo_cost_over_sorted_pairing": _cost(pairs, z) / best,
        "reference_cost_over_sorted_pairing": _cost(ref, z) / best,
        "n_pairs": float(len(pairs)),
        "mip_optimal": float(diag["status"] == "optimal"),
        "mip_backend_cvxpy": float(diag["backend"] == "cvxpy"),
        # the package's default metrics, recorded as descriptors of the reference
        "default_metrics_pairs_shared": float(len(set(pairs) & set(ref_default))),
        "default_metrics_cost_over_sorted_pairing": _cost(ref_default, z) / best,
    }


def comparison() -> dict:
    """PANGEO (cvxpy) against google/trimmed_match's pairing, quantity by quantity."""
    ref = _reference("response")
    res, pairs = _pangeo()
    z = _totals()
    best = _sorted_cost(z)
    rows = [
        {"quantity": "pairs", "mlsynth": len(pairs), "reference": len(ref)},
        {"quantity": "pairs in common", "mlsynth": len(set(pairs) & set(ref)), "reference": len(ref)},
        {"quantity": "cost / sorted pairing", "mlsynth": round(_cost(pairs, z) / best, 9),
         "reference": round(_cost(ref, z) / best, 9)},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "PANGEO",
                         "config": {"objective": "supergeo_total", "max_supergeo_size": 1,
                                    "fast": False, "mip_backend": "cvxpy"}},
        "reference": {"impl": "google/trimmed_match TrimmedMatchGeoXDesign.create_geo_pairs "
                              "(live, own environment, C++ core compiled without Bazel)",
                      "version": "commit 0be855234ee3a589eaddc8b945e2a0d924ad8791"},
    }


# Cross-validation (scenario: full repo, live Python reference). The pairing is
# exact on both sides: the reference sorts, PANGEO's MIP stops at HiGHS's default
# relative gap (1e-4), which bounds its cost ratio. Skips without the reference.
EXPECTED = {
    "pairs_shared_with_reference": (1.0, 0.0),
    "pangeo_cost_over_sorted_pairing": (1.0, 1e-4),
    "reference_cost_over_sorted_pairing": (1.0, 1e-12),
    "n_pairs": (87.0, 0.0),
    "mip_optimal": (1.0, 0.0),
    "mip_backend_cvxpy": (1.0, 0.0),
    "default_metrics_pairs_shared": (81.0, 0.0),
    "default_metrics_cost_over_sorted_pairing": (1.000668, 1e-5),
}
