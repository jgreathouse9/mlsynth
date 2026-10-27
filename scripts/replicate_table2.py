"""Reproduce Table 2 of Tao et al. (2023): channel shares on the Sec. 5.1 design.

Each run simulates 10,000 paths of 365 days from the Table 1 Hawkes process,
records the exact channel-off ground truth (CCC shares), fits the model with
cross-validated penalties, and scores every conversion by TRE and DRE. The
"oracle" rows score with the true parameters, isolating the scoring step from
estimation error.

    python scripts/replicate_table2.py --runs 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path as _P

import numpy as np
import pandas as pd

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from gppmta import GraphicalPointProcess, attribute, channel_shares, hellinger, kl_divergence, simulate  # noqa: E402
from gppmta.paper import ALPHA, DECAY, MU_ALL, N_PATHS, SPACE, T, TABLE2, TRUE_PARAMS  # noqa: E402

CHANNELS = ["display", "search"]


def one_run(seed: int, n_paths: int) -> dict:
    chan = SPACE.channel_types()
    sim = simulate(MU_ALL, ALPHA, DECAY, T, n_paths, chan, seed=seed)
    truth = np.array([sim.ccc_shares[ch] for ch in CHANNELS])
    model = GraphicalPointProcess(decay=DECAY, gamma="cv", seed=seed).fit(sim.paths, SPACE.p, SPACE.q)
    out = {"seed": seed, "conversions": sim.conversions, "truth_display": truth[0]}
    for label, params in (("fit", model.params_), ("oracle", TRUE_PARAMS)):
        table = attribute(params, sim.paths, chan)
        for kind in ("tre", "dre"):
            share = channel_shares(table, kind)[CHANNELS].to_numpy()
            out[f"{label}_{kind}_display"] = share[0]
            out[f"{label}_{kind}_kl"] = kl_divergence(truth, share)
            out[f"{label}_{kind}_hellinger"] = hellinger(truth, share)
    out["fit_max_alpha_err"] = float(np.abs(model.params_.alpha - TRUE_PARAMS.alpha).max())
    return out


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    def ms(col):
        return f"{runs[col].mean():.4f} ({runs[col].std(ddof=1):.4f})"

    rows = []
    rows.append(["Display share: truth (CCC)", ms("truth_display"), f"{TABLE2['truth']['display']:.4f}"])
    for kind in ("tre", "dre"):
        paper = TABLE2[kind]
        rows.append([f"{kind.upper()} display share", ms(f"fit_{kind}_display"), f"{paper['display']:.4f}"])
        rows.append([f"{kind.upper()} KL divergence", ms(f"fit_{kind}_kl"), f"{paper['kl']:.4f}"])
        rows.append([f"{kind.upper()} Hellinger", ms(f"fit_{kind}_hellinger"), f"{paper['hellinger']:.4f}"])
    for kind in ("tre", "dre"):
        rows.append([f"{kind.upper()} display share, true params", ms(f"oracle_{kind}_display"), ""])
    return pd.DataFrame(rows, columns=["quantity", "this prototype: mean (sd)", "paper Table 2"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--paths", type=int, default=N_PATHS)
    ap.add_argument("--out", type=_P, default=None, help="CSV of per-run results")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    runs = pd.DataFrame([one_run(seed, args.paths) for seed in range(args.runs)])
    if args.out:
        runs.to_csv(args.out, index=False)
    with pd.option_context("display.width", 120, "display.max_colwidth", 60):
        print(summarise(runs).to_string(index=False))
    print(f"\n{args.runs} runs x {args.paths} paths in {time.perf_counter() - t0:.0f}s; "
          f"max |alpha_hat - alpha| averaged {runs['fit_max_alpha_err'].mean():.4f}")


if __name__ == "__main__":
    main()
