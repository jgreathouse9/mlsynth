"""Regenerate ``results.json`` -- every number quoted in the README.

    python benchmarks/reference/lpca_kansas/dump_results.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_kansas import (  # noqa: E402
    N_MATCH_PERIODS, gap, kansas_matrices, lpca_arm, post_mean, sc_arm,
)

# Feng (2023) Section 6.1, the November 2023 arXiv version (v1).
PAPER_LPCA_GAP = -0.53
PAPER_SC_GAP = 0.19
PAPER_COUNT_BELOW = 9


def main() -> int:
    m = kansas_matrices()
    n_post = len(m["observed"]) - m["post_start"]
    observed_post = float(np.mean(m["observed"][m["post_start"]:]))

    lpca = lpca_arm(m)
    sc = sc_arm(m)
    lo = m["post_start"] - lpca["offset"]
    below = int(np.sum(lpca["path"][lo:lo + n_post] > m["observed"][m["post_start"]:]))

    out = {
        "panel": {
            "units": int(m["x"].shape[0]),
            "periods": int(m["levels"].shape[1]),
            "periods_differenced": int(m["x"].shape[1]),
            "pre_periods_differenced": int(m["post_start"]),
            "post_periods": int(n_post),
            "treatment_start": float(m["time"][m["post_start"]]),
            "observed_post_mean_growth": observed_post,
        },
        "lpca": {
            "K": lpca["K"], "match_periods": N_MATCH_PERIODS,
            "local_rank_selected": lpca["rank"],
            "neighbours": lpca["neighbours"],
            "post_mean": post_mean(lpca["path"], lpca["offset"], m),
            "gap": gap(lpca["path"], lpca["offset"], m),
            "pre_period_rmse": _pre_rmse(lpca["path"], lpca["offset"], m),
            "paper_gap": PAPER_LPCA_GAP,
            "post_periods_observed_below": below,
            "paper_post_periods_observed_below": PAPER_COUNT_BELOW,
        },
        "sc": {
            "gap_recentred": gap(sc["recentred"], sc["offset"], m),
            "gap_demeaned": gap(sc["demeaned"], sc["offset"], m),
            "paper_gap": PAPER_SC_GAP,
            "post_mean_recentred": post_mean(sc["recentred"], sc["offset"], m),
            "pre_period_rmse": float(np.sqrt(sc["pre_sse"] / m["post_start"])),
            "pre_period_rmse_shared_window": _pre_rmse(sc["recentred"],
                                                       sc["offset"], m),
            "col_mean_post_average": float(np.mean(m["col_mean"][m["post_start"]:])),
            "active_donors": {
                name: round(float(wt), 6)
                for name, wt in zip(sc["unit_names"], sc["weights"]) if wt > 1e-6
            },
        },
        "sensitivity": _sensitivity(m),
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")

    print(f"observed post-treatment mean growth   {observed_post:+.4f} pp\n")
    print(f"{'arm':38s} {'ported':>9s} {'paper v1':>9s}")
    for label, got, want in [
        ("LPCA gap (pp)", out["lpca"]["gap"], PAPER_LPCA_GAP),
        ("LPCA post periods observed below", below, PAPER_COUNT_BELOW),
        ("SC gap, v1 convention (pp)", out["sc"]["gap_demeaned"], PAPER_SC_GAP),
        ("SC gap, 2024 convention (pp)", out["sc"]["gap_recentred"], float("nan")),
    ]:
        print(f"{label:38s} {got:9.4f} {want:9.2f}")
    print(f"\nwrote {path}")
    return 0


def _pre_rmse(path, offset, m):
    """Pre-treatment fit over the window both arms can be scored on.

    Local PCA only predicts the periods held out from matching, so the arms are
    comparable on the intersection: differenced periods ``offset`` to the last
    untreated one.
    """
    lo, hi = offset, m["post_start"]
    resid = m["observed"][lo:hi] - path[: hi - lo]
    return float(np.sqrt(np.mean(resid ** 2)))


def _sensitivity(m):
    """How the LPCA answer moves with the two constants the paper fixes by fiat."""
    n = m["x"].shape[0]
    by_k, by_split, by_components = {}, {}, {}
    for K in (7, 10, 14, 20, 25, 30):
        arm = lpca_arm(m, n_neighbours=K)
        by_k[str(K)] = gap(arm["path"], arm["offset"], m)
    for split in (20, 30, 40, 50, 60, 70):
        arm = lpca_arm(m, n_match=split)
        by_split[str(split)] = gap(arm["path"], arm["offset"], m)
    for nc in (2, 3, 4, 5):
        arm = lpca_arm(m, n_components=nc)
        by_components[str(nc)] = {"gap": gap(arm["path"], arm["offset"], m),
                                  "rank": arm["rank"]}
    return {
        "default_K": int(round(n ** (2 / 3))),
        "gap_by_n_neighbours": by_k,
        "gap_by_match_periods": by_split,
        "gap_by_max_components": by_components,
    }


if __name__ == "__main__":
    raise SystemExit(main())
