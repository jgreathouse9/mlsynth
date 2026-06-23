"""Generate the Apple "go-dark" geo-experiment panel used in the JOSS paper.

A worked marketing-science framing of Abadie & Zhao's (2026) experimental
design, on their own baseline data-generating process (``generate_marex_sample``,
Section 5) -- no new DGP, only an applied narrative on top of it.

Story: Apple wants to measure the incremental sales revenue its paid media
generates. It runs a *go-dark* geo experiment -- switch paid media off in a
handful of whole DMAs (designated market areas) and compare each dark market to
a synthetic counterfactual that kept spending. MAREX chooses *which* of the 20
DMAs go dark so the dark group and the still-spending group both reproduce the
national market before the test.

We simulate that here. The selection solve (the experimental design) is run once,
offline, so the rendered paper can simply load the resulting panel and run MAREX
a single time -- recovering the same design and reading off the effect. The
``sales`` column is what Apple would actually observe (paid media on everywhere
pre-launch; off in the chosen DMAs post-launch); ``tau_true`` is the
simulation's known per-week average effect, kept so the paper can report
recovery error.

Run from the repo root::

    python tools/gen_apple_godark.py

Writes ``basedata/apple_godark_dmas.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlsynth import MAREX
from mlsynth.utils.marex_helpers.simulation import generate_marex_sample

# --- the DGP knobs: 20 DMAs, 20 pre-launch weeks, 10 post, treat 4--6 ---------
J, R, F, T, T0, T_POST = 20, 7, 11, 30, 20, 10
M_MIN, M_MAX = 4, 6
SEED = 7

OUT = Path(__file__).resolve().parents[1] / "basedata" / "apple_godark_dmas.csv"


def _panel(Y: np.ndarray) -> pd.DataFrame:
    """Tidy long frame from a (DMA x week) sales matrix."""
    J, T = Y.shape
    return pd.DataFrame(
        [{"dma": f"dma{j:02d}", "week": t, "sales": float(Y[j, t])}
         for j in range(J) for t in range(T)]
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    sample = generate_marex_sample(J=J, R=R, F=F, T=T, T0=T0, sigma=1.0, rng=rng)

    # The exact configuration the paper re-fits with, so the dark markets we
    # pre-commit here are the same ones the paper's single MAREX solve recovers
    # (inference=True holds out blank periods, which changes the selection).
    cfg = {
        "outcome": "sales", "unitid": "dma", "time": "week",
        "T0": T0, "T_post": T_POST, "m_min": M_MIN, "m_max": M_MAX,
        "design": "standard", "inference": True, "display_graph": False,
    }

    # Design phase: MAREX sees only the pre-launch sales (paid media on for
    # everyone, since nobody has gone dark yet) and chooses which DMAs to darken.
    design = MAREX({**cfg, "df": _panel(sample.Y_N)}).fit()
    w = design.globres.treated_weights_agg
    dark = np.where(w > 1e-8)[0]

    # Experiment: the chosen DMAs go dark (realize their paid-media-off sales)
    # from launch (week T0) on; everyone else keeps spending.
    realized = sample.Y_N.copy()
    realized[dark, T0:] = sample.Y_I[dark, T0:]

    # Known simulation truth: the per-week average effect of going dark.
    tau_true = np.zeros(T)
    tau_true[T0:] = sample.tau[T0:]

    df = _panel(realized)
    df["went_dark"] = df["dma"].isin([f"dma{j:02d}" for j in dark])
    df["tau_true"] = df["week"].map(dict(enumerate(tau_true)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows; {len(dark)} of {J} DMAs dark: "
          f"{sorted(dark.tolist())})")


if __name__ == "__main__":
    main()
