#!/usr/bin/env python3
"""Write the Dube panel as CSV for the R reference, which cannot read parquet.

Not committed: the CSV is 19 MB against the parquet's 2 MB, and it is a
byte-faithful re-encoding of a file the repository already carries.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2] / "basedata" / "dube_minwage.parquet"

if __name__ == "__main__":
    df = pd.read_parquet(SOURCE)
    df.to_csv(HERE / "dube_panel.csv", index=False)
    print(f"wrote {HERE / 'dube_panel.csv'} ({len(df):,} rows)")
