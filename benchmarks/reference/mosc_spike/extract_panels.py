"""Extract the author's per-team COVID panels from Joshuashou/Synthetic-Control-Paper-Model.

The upstream repository ships each team's panel as a pair of ``torch.save``
tensors (``train_data.pt`` / ``test_data.pt``) plus a ``dates.csv``. The scripts
that consume them (``deconfound_and_plot.py``) instead read ``train_pivot.csv``
and ``test_pivot.csv``, which the repository does not ship, so the panels have to
be reassembled before anything upstream will run.

Reading them does not need torch. A ``.pt`` file is a zip whose ``data.pkl``
records the dtype and shape and whose ``data/0`` member is the raw storage, so
the array is recovered with ``zipfile`` and ``numpy.frombuffer``.

Column order matters: the upstream code asserts ``county_names[-1] ==
'Stadium_County'`` and treats that last column as the single treated unit
(``A[-1] = 1``). Column labels are not stored in the tensors, so they are
reconstructed as ``county_0 ... county_{N-2}`` plus ``Stadium_County``.
"""
from __future__ import annotations

import argparse
import io
import pickletools
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

_DTYPES = {"DoubleStorage": np.float64, "FloatStorage": np.float32, "LongStorage": np.int64}


def read_torch_tensor(path: Path) -> np.ndarray:
    """Read a single-tensor ``torch.save`` file into a numpy array."""
    with zipfile.ZipFile(path) as z:
        pkl_name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        prefix = pkl_name.rsplit("/", 1)[0]
        listing = io.StringIO()
        pickletools.dis(z.read(pkl_name), listing)
        text = listing.getvalue()

        storage = re.search(r"GLOBAL\s+'torch (\w+Storage)'", text)
        if storage is None or storage.group(1) not in _DTYPES:
            raise ValueError(f"unsupported storage in {path}")
        dtype = _DTYPES[storage.group(1)]

        ints = [int(m) for m in re.findall(r"BININT\d?\s+(\d+)", text)]
        # numel, storage_offset, then the shape tuple, then the stride tuple
        numel, offset, n_rows, n_cols = ints[0], ints[1], ints[2], ints[3]
        if offset != 0 or n_rows * n_cols != numel:
            raise ValueError(f"unexpected layout in {path}: {ints[:6]}")

        raw = z.read(f"{prefix}/data/0")
        return np.frombuffer(raw, dtype=dtype, count=numel).reshape(n_rows, n_cols)


def load_team(team_dir: Path) -> tuple[pd.DataFrame, int]:
    """Return the team's (dates x counties) panel and its intervention index."""
    train = read_torch_tensor(team_dir / "train_data.pt")
    test = read_torch_tensor(team_dir / "test_data.pt")
    if train.shape[1] != test.shape[1]:
        raise ValueError(f"{team_dir.name}: train/test county counts disagree")

    dates = pd.read_csv(team_dir / "dates.csv")["date"].to_numpy()
    total = np.vstack([train, test])
    if len(dates) != total.shape[0]:
        raise ValueError(
            f"{team_dir.name}: {len(dates)} dates against {total.shape[0]} rows"
        )

    n_counties = total.shape[1]
    columns = [f"county_{i}" for i in range(n_counties - 1)] + ["Stadium_County"]
    panel = pd.DataFrame(total, index=pd.Index(dates, name="date"), columns=columns)
    return panel, train.shape[0]


def to_long(panel: pd.DataFrame, intervention_t: int) -> pd.DataFrame:
    """Reshape to the long form dataprep ingests, with a 0/1 treatment column."""
    long = (
        panel.reset_index()
        .melt(id_vars="date", var_name="county", value_name="cases")
        .sort_values(["county", "date"], kind="stable")
        .reset_index(drop=True)
    )
    post_dates = set(panel.index[intervention_t:])
    long["stadium_open"] = (
        (long["county"] == "Stadium_County") & (long["date"].isin(post_dates))
    ).astype(int)
    return long


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True, help="clone of the author's repo")
    p.add_argument("--out", type=Path, required=True, help="directory to write parquet into")
    p.add_argument("--teams", nargs="+", default=["Indianapolis", "Baltimore"])
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for team in args.teams:
        panel, intervention_t = load_team(args.upstream / "dat" / team)
        long = to_long(panel, intervention_t)
        slug = team.lower().replace(" ", "_")
        long.to_parquet(args.out / f"{slug}.parquet", index=False)
        integral = bool(np.all(panel.to_numpy() == np.round(panel.to_numpy())))
        print(
            f"{team}: {panel.shape[0]} dates x {panel.shape[1]} counties, "
            f"pre={intervention_t}, post={panel.shape[0] - intervention_t}, "
            f"integer_counts={integral}, "
            f"range=[{panel.to_numpy().min():.0f}, {panel.to_numpy().max():.0f}]"
        )


if __name__ == "__main__":
    main()
