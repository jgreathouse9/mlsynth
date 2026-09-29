"""Driver: where the two hill climbs agree, and where they part company.

    MLSYNTH_MATCHED_MARKETS=/path/to/matched_markets \
        python benchmarks/studies/tbrmm_match/run.py

Three questions, in order, each answered by its own section:

1. Scoring. Given the same split, do the two engines return the same score?
   A disagreement here is an arithmetic fault in one of them.
2. The step. Given the same starting split and the same candidate set, do they
   take the same step? A disagreement here with section 1 clean is a difference
   between two heuristics.
3. The designs. What each engine recommends per treatment size, and which of the
   two recommendations the other engine scores higher.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.studies.tbrmm_match import candidates as cand   # noqa: E402
from benchmarks.studies.tbrmm_match import engine, reference     # noqa: E402

PANEL = Path(__file__).resolve().parents[3] / "basedata" / "geolift_test_data.csv"
N_TEST, K_MAX, N_PRETEST = 14, 4, 90
TOL = 1e-9


def _fmt(key: Sequence[float]) -> str:
    gates = "".join(str(int(v)) for v in key[:4])
    return f"[{gates}] corr={key[4]:.2f} inv={key[5]:.8f}"


def section_scoring(eng, wide, designs_ref) -> int:
    """Score every design either engine recommends, with both engines."""
    print("\n1. Scoring the same split with both engines")
    print("   " + "-" * 68)
    worst = 0.0
    for k, (treatment, control, _) in sorted(designs_ref.items()):
        ref_key = reference.score(eng, treatment, control)
        my_key = engine.score(wide, treatment, control, n_test=N_TEST)
        gap = abs(ref_key[-1] - my_key[-1]) / (abs(ref_key[-1]) or 1.0)
        worst = max(worst, gap)
        gates_match = ref_key[:-1] == my_key[:-1]
        print(f"   k={k}  reference {_fmt(ref_key)}")
        print(f"        mlsynth   {_fmt(my_key)}   gates_match={gates_match} "
              f"reldiff={gap:.2e}")
    print(f"   worst relative difference on the power term: {worst:.3e}")
    return 0 if worst < 1e-8 else 1


def section_step(eng, wide, start_treatment, start_control) -> int:
    """One augmentation step from a split both engines agree on."""
    print("\n2. One augmentation step from the same starting split")
    print("   " + "-" * 68)
    pool = reference.treatment_pool(eng)
    items = cand.augmentation_candidates(set(start_treatment), set(start_control), pool)
    print(f"   starting treatment {sorted(start_treatment)}, "
          f"|control|={len(start_control)}, {len(items)} candidates")

    ref_scored = cand.score_all(items, lambda t, c: reference.score(eng, t, c))
    my_scored = cand.score_all(
        items, lambda t, c: engine.score(wide, t, c, n_test=N_TEST))

    differing = cand.score_disagreements(ref_scored, my_scored, TOL)
    print(f"   candidates scored differently by the two engines: {len(differing)}")
    for geo, a, b, gap in differing[:5]:
        print(f"     {geo}: reference {_fmt(a)} | mlsynth {_fmt(b)} | {gap:.2e}")

    ref_pick, my_pick = cand.best(ref_scored), cand.best(my_scored)
    print(f"   reference would take: {ref_pick.candidate.geo}  {_fmt(ref_pick.key)}")
    print(f"   mlsynth would take  : {my_pick.candidate.geo}  {_fmt(my_pick.key)}")
    print(f"   same step: {ref_pick.candidate.geo == my_pick.candidate.geo}")
    return 0 if not differing else 1


def section_designs(eng, wide, designs_ref, designs_mine) -> None:
    """What each engine recommends, and which recommendation scores higher."""
    print("\n3. Recommended designs, and which one the objective prefers")
    print("   " + "-" * 68)
    for k in sorted(set(designs_ref) | set(designs_mine)):
        ref = designs_ref.get(k)
        mine = designs_mine.get(k)
        if ref is None or mine is None:
            print(f"   k={k}: reported by only one engine")
            continue
        same = ref[0] == mine[0] and ref[1] == mine[1]
        ref_on_ref = reference.score(eng, ref[0], ref[1])
        ref_on_mine = reference.score(eng, mine[0], mine[1])
        better = ("identical" if same else
                  "mlsynth" if ref_on_mine[-1] > ref_on_ref[-1] else "reference")
        print(f"   k={k}  same_design={same}   the reference's own score prefers: {better}")
        print(f"        reference trt={ref[0]} |ctl|={len(ref[1])} inv={ref_on_ref[-1]:.8f}")
        print(f"        mlsynth   trt={mine[0]} |ctl|={len(mine[1])} inv={ref_on_mine[-1]:.8f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(PANEL))
    args = parser.parse_args(argv)

    if not reference.available():
        print(f"reference not importable; set {reference.ENV_VAR} to a "
              f"google/matched_markets checkout")
        return 77

    panel = reference.long_panel(args.panel)
    eng = reference.engine(panel, n_test=N_TEST, k_max=K_MAX, n_pretest=N_PRETEST)
    wide = engine.scoring_window(panel, N_PRETEST)
    print(f"panel: {wide.shape[1]} geos over {wide.shape[0]} scored periods, "
          f"n_test={N_TEST}, K={K_MAX}")

    designs_ref = reference.designs(eng)
    designs_mine = engine.designs(panel, n_test=N_TEST, k_max=K_MAX,
                                  n_pretest=N_PRETEST)

    status = section_scoring(eng, wide, designs_ref)
    smallest = min(designs_ref)
    status |= section_step(eng, wide, designs_ref[smallest][0],
                           designs_ref[smallest][1])
    section_designs(eng, wide, designs_ref, designs_mine)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
