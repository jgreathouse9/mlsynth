"""Replay octave_crosscheck.m's cases through port.rpca_hqf and the library.

Reads crosscheck/*.mat written by the authors' RPCA_HQF.m under Octave and
reports, per case, max |X_port - X_octave| and max |X_library - X_octave|
(the library at the paper's settings, same U0).

Usage: python crosscheck.py CROSSCHECK_DIR [OUT.parquet]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import port  # noqa: E402
from run_tableone import _library  # noqa: E402

rows = []
for f in sorted(Path(sys.argv[1]).glob("*.mat")):
    d = loadmat(f)
    Mn, U0, X = d["M_noise"], d["U0"], d["X"]
    rak = U0.shape[1]
    Xp, _ = port.rpca_hqf(Mn, rak, 100, 3.0, 0.001, U0)
    Xl = _library(Mn, U0, rank=rak, ip=3.0, lam=0.001, max_iter=100).low_rank
    scale = np.abs(X).max()
    rows.append((f.stem, np.abs(Xp - X).max() / scale, np.abs(Xl - X).max() / scale,
                 float(d["rmse"].squeeze())))
    print(f"{f.stem:16s} port {rows[-1][1]:.1e}  library {rows[-1][2]:.1e}  octave RMSE {rows[-1][3]:.4f}")
if len(sys.argv) > 2:
    import pandas as pd
    pd.DataFrame(rows, columns=["case", "port_rel_maxabs", "library_rel_maxabs", "octave_rmse"]
                 ).to_parquet(sys.argv[2], index=False)
print(f"\nworst relative max-abs: port {max(r[1] for r in rows):.1e}, library {max(r[2] for r in rows):.1e}")
