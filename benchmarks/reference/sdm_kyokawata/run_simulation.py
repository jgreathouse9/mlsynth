"""Regenerate ``simulation_results.json`` from the article's Section 4 design.

    python benchmarks/reference/sdm_kyokawata/run_simulation.py

Self-contained: the design is fully specified in the article, so this needs
nothing from the authors' replication package.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulation import J_GRID, N_REPS, run  # noqa: E402

HERE = Path(__file__).resolve().parent

# Article Figure 1(b), read off Section 4.3: SC rises in J, the proposal is flat.
PAPER = {5: dict(sc=4.4540, sdm_lo=1.7000, sdm_hi=1.9100),
         50: dict(sc=5.4300, sdm_lo=1.7000, sdm_hi=1.9100)}


def main():
    rows = run(seed=0)
    print(
        f"{'J':>4} {'SC':>9} {'MSCb':>9} {'MSCc':>9} "
        f"{'reachable':>10} {'needed':>7}  note"
    )
    for r in rows:
        note = "saturated (params >= obs)" if r["saturated"] else ""
        sc = "n/a" if r["rmspe_sc"] is None else f"{r['rmspe_sc']:9.4f}"
        print(
            f"{r['J']:>4} {sc:>9} {r['rmspe_mscb']:>9.4f} {r['rmspe_mscc']:>9.4f} "
            f"{r['max_reachable_loading']:>10.4f} {r['treated_loading']:>7.1f}  {note}"
        )

    print("\nArticle Figure 1(b), for comparison:")
    for J, p in PAPER.items():
        print(f"  J={J:<3} SC {p['sc']:.4f}   proposal {p['sdm_lo']}-{p['sdm_hi']}")
    print(
        "\nMSCb is plain non-negative least squares: no sequential updating, no\n"
        "convergence index, no article. It lands in the band the article reports\n"
        "for its own method, and MSCc beats it. The separation in Figure 1(b) is\n"
        "the adding-up constraint."
    )

    out = dict(design=dict(T0=20, n_reps=N_REPS, j_grid=list(J_GRID), seed=0),
               paper_reported=PAPER, rows=rows)
    (HERE / "simulation_results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {HERE / 'simulation_results.json'}")


if __name__ == "__main__":
    main()
