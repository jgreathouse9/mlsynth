"""Write one simulated Table 1 dataset as a long event table.

Seed ``s`` reproduces run ``s`` of ``scripts/replicate_table2.py`` exactly. The
output has one row per event (``path_id``, ``time`` in days, ``event`` label)
plus each path's window end ``T``, the format ``paths_from_frame`` reads with
``end_col="T", origin="zero"``. The channel-off ground truth goes to a JSON
file next to it.

    python scripts/export_simulation.py --seed 0 --out data/table1_seed0.csv.gz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as _P

import pandas as pd

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from gppmta import simulate  # noqa: E402
from gppmta.paper import ALPHA, DECAY, MU_ALL, N_PATHS, SPACE, T  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paths", type=int, default=N_PATHS)
    ap.add_argument("--out", type=_P, required=True)
    args = ap.parse_args(argv)
    sim = simulate(MU_ALL, ALPHA, DECAY, T, args.paths, SPACE.channel_types(), seed=args.seed)
    frame = pd.DataFrame(
        [(j, round(t, 6), SPACE.names[e], path.T)
         for j, path in enumerate(sim.paths) for t, e in zip(path.times, path.types)],
        columns=["path_id", "time", "event", "T"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    truth = {"seed": args.seed, "paths": args.paths, "conversions": sim.conversions,
             "ccc": sim.ccc, "ccc_shares": sim.ccc_shares}
    stem = args.out.name.split(".")[0]
    (args.out.parent / f"{stem}_truth.json").write_text(json.dumps(truth, indent=2) + "\n")


if __name__ == "__main__":
    main()
