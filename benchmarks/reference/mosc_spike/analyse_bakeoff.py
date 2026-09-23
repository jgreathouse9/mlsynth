"""Read the bake-off and answer the two questions the spike was opened to settle.

Question 1. Does the gamma-Poisson arm's advantage survive dropping the
undocumented lagged-outcome regressor? Upstream hardcodes
``include_previous_outcome=True`` in the script that produced Figure 8 and gives
the rSC baseline no equivalent term, so the published comparison confounds the
likelihood with the lag.

Question 2. Does the advantage hold at a short pre-period? The paper's own
Figure 8 shows the methods converging at 25 pre-periods and separating at 100,
and its identification argument is asymptotic in the number of pre-periods. Panels
in this library typically carry 12-30.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> dict:
    here = Path(__file__).parent
    frame = pd.read_csv(here / "bakeoff.csv")
    frame = frame[frame["relative_error"].notna()]

    print("Mean relative error over 30 post-periods, averaged across teams, "
          "mismatch levels,\neffect sizes and seeds. Lower is better.\n")

    print("Question 1 -- does the advantage survive dropping the lagged outcome?\n")
    lag_table = (
        frame.pivot_table(index="model", columns="include_previous_outcome",
                          values="relative_error", aggfunc="mean")
        .rename(columns={False: "without lag", True: "with lag (published)"})
    )
    print(lag_table.round(4).to_string(), "\n")

    def gap_margin(sub: pd.DataFrame) -> dict:
        wide = sub.pivot_table(index="model", values="relative_error", aggfunc="mean")
        gap = float(wide.loc["GAP", "relative_error"])
        return {
            "GAP": gap,
            "PPCA": float(wide.loc["PPCA", "relative_error"]),
            "rSC": float(wide.loc["rSC", "relative_error"]),
            "gap_beats_ppca": gap < float(wide.loc["PPCA", "relative_error"]),
            "gap_beats_rsc": gap < float(wide.loc["rSC", "relative_error"]),
        }

    with_lag = gap_margin(frame[frame["include_previous_outcome"]])
    without_lag = gap_margin(frame[~frame["include_previous_outcome"].astype(bool)])
    print(f"  with the lag (as published): GAP beats PPCA = {with_lag['gap_beats_ppca']}, "
          f"GAP beats rSC = {with_lag['gap_beats_rsc']}")
    print(f"  without the lag:             GAP beats PPCA = {without_lag['gap_beats_ppca']}, "
          f"GAP beats rSC = {without_lag['gap_beats_rsc']}\n")

    print("Question 2 -- does it hold at a short pre-period?\n")
    pre_table = frame.pivot_table(
        index=["n_pre", "include_previous_outcome"], columns="model",
        values="relative_error", aggfunc="mean",
    )
    print(pre_table.round(4).to_string(), "\n")

    print("By departure from the factor model (rho), without the lag:\n")
    mismatch_table = (
        frame[~frame["include_previous_outcome"].astype(bool)]
        .pivot_table(index=["mismatch", "n_pre"], columns="model",
                     values="relative_error", aggfunc="mean")
    )
    print(mismatch_table.round(4).to_string(), "\n")

    by_pre = {}
    for n_pre, sub in frame.groupby("n_pre"):
        by_pre[int(n_pre)] = {
            "with_lag": gap_margin(sub[sub["include_previous_outcome"]]),
            "without_lag": gap_margin(sub[~sub["include_previous_outcome"].astype(bool)]),
        }

    verdict = {
        "advantage_survives_dropping_lag": bool(
            without_lag["gap_beats_ppca"] and without_lag["gap_beats_rsc"]
        ),
        "advantage_holds_at_short_pre_period": bool(
            by_pre.get(25, {}).get("without_lag", {}).get("gap_beats_ppca", False)
            and by_pre.get(25, {}).get("without_lag", {}).get("gap_beats_rsc", False)
        ),
    }
    verdict["build"] = bool(
        verdict["advantage_survives_dropping_lag"]
        and verdict["advantage_holds_at_short_pre_period"]
    )

    print("Decision rule, fixed before the run:")
    print(f"  advantage survives dropping the lag     : {verdict['advantage_survives_dropping_lag']}")
    print(f"  advantage holds at 25 pre-periods       : {verdict['advantage_holds_at_short_pre_period']}")
    print(f"  => build                                : {verdict['build']}")

    out = {
        "overall_with_lag": with_lag,
        "overall_without_lag": without_lag,
        "by_pre_period": by_pre,
        "verdict": verdict,
    }
    (here / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    main()
