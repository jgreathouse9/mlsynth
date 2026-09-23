"""Regenerate ``results.json`` — every number the README reports.

    python benchmarks/reference/survsc_mc/dump_results.py

Runs the Section 4 grid under both weight modes and both readings of the
evaluation horizon, plus the rank-rule sweep that shows how much the
unstated ``r0`` rule decides. Deterministic: seeds are fixed in
``run_grid.py``.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from run_grid import DGPS, KS, PAPER, cell  # noqa: E402

OUT = pathlib.Path(__file__).parent / "results.json"


def grid(weight_mode: str, horizon: str) -> dict:
    out = {}
    for dgp in DGPS:
        for K in KS:
            c = cell(dgp, K, rank_rule="usvt", horizon_mode=horizon,
                     weight_mode=weight_mode)
            out[f"{dgp}_K{K}"] = {
                "ssc_mean": round(c["ssc"][0], 4),
                "ssc_sd": round(c["ssc"][1], 4),
                "oracle_mean": round(c["oracle"][0], 4),
                "oracle_sd": round(c["oracle"][1], 4),
                "share_monotone": round(c["monotone"][0], 2),
                "share_in_unit_range": round(c["in_unit_range"][0], 2),
                "median_horizon": round(c["horizon"][2], 3),
                "paper_ssc": list(PAPER[(dgp, K)]["ssc"]),
                "paper_oracle": list(PAPER[(dgp, K)]["oracle"]),
            }
    return out


def rank_sweep() -> dict:
    out = {}
    for rule in ("cumvar", "spectral", "usvt"):
        for dgp in DGPS:
            for K in KS:
                c = cell(dgp, K, rank_rule=rule, oracle=False)
                out[f"{rule}_{dgp}_K{K}"] = {
                    "ssc_mean": round(c["ssc"][0], 4),
                    "mean_rank": round(c["rank"][0], 1),
                    "paper_ssc_mean": PAPER[(dgp, K)]["ssc"][0],
                }
    return out


def main() -> None:
    payload = {
        "provenance": {
            "paper": "arXiv:2511.14133v1, Han & Shah, Synthetic Survival Control",
            "target": "Table 1 (sup-norm error, mean +/- sd over 20 replications)",
            "design": "Section 4",
            "replications": 20,
            "common_random_numbers_across_K": True,
        },
        "grid_pcr_pooled": grid("pcr", "pooled"),
        "grid_pcr_treated": grid("pcr", "treated"),
        "grid_simplex_pooled": grid("simplex", "pooled"),
        "rank_sweep": rank_sweep(),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
