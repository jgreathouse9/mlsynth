"""Export the West Germany outcome and donor matrix the R scripts read.

This reproduces the authors' own prepare_realdata step: the treated series is
West Germany's GDP sorted by year, the donor matrix is every other country in
the order they first appear in the file, and the intervention is 1990. The donor
ordering matters, since the sampler consumes random draws in column order, so a
different ordering gives a different chain from the same seed.

    python scripts/export_inputs.py repgermany.dta

Writes y.csv (44), x.csv (44 x 16, year by donor) and donors.csv.
"""
import sys

import numpy as np
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python export_inputs.py <path to repgermany.dta>")
    d = pd.read_stata(sys.argv[1])

    wg = d[d.country == "West Germany"].sort_values("year")
    y = wg["gdp"].to_numpy(float)

    rest = d[d.country != "West Germany"]
    donors = list(dict.fromkeys(rest["country"]))          # file order, as pivot_wider gives
    X = np.column_stack([rest[rest.country == c].sort_values("year")["gdp"].to_numpy(float)
                         for c in donors])

    assert y.shape == (44,) and X.shape == (44, 16), (y.shape, X.shape)
    assert donors[:3] == ["USA", "UK", "Austria"], donors[:3]

    np.savetxt("y.csv", y, delimiter=",")
    np.savetxt("x.csv", X, delimiter=",")
    with open("donors.csv", "w") as fh:
        fh.write("\n".join(donors) + "\n")
    print(f"wrote y.csv {y.shape}, x.csv {X.shape}, donors.csv ({len(donors)} donors)")
