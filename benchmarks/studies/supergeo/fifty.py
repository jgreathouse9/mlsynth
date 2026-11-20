"""Branch, cut and price on 50 markets: subset-row cuts against the tuned rank-1 setting.

    python -m benchmarks.studies.supergeo.fifty INSTANCE ARM [TIME_LIMIT]

INSTANCE is ``urate`` (the 50-state, 40-month panel of the SYNDES benchmark,
``basedata/urate_cps.csv``, summed over months 34 to 37: 4 pretest months ending 3
before the panel's end, as the GeoLift subsets leave a test window) or ``lognormal`` (50 draws of exp(N(0, 1.3^2)), seed
2026, the generator of the N = 210 instance). ARM is ``sr3`` or ``managed``
(:func:`.bcp.managed`). Sizes are divided by the median gap between sorted sizes.
One JSON line at the end; progress every 30 s on stdout."""
import json, pathlib, sys
import numpy as np, pandas as pd
from . import bcp

DATA = pathlib.Path(__file__).resolve().parents[3] / "basedata"


def instance(name):
    if name == "urate":
        Y = pd.read_csv(DATA / "urate_cps.csv", header=None).to_numpy(float)
        assert Y.shape == (40, 50)
        pre = Y[-7:-3].sum(0)
    elif name == "lognormal":
        pre = np.exp(np.random.default_rng(2026).normal(0, 1.3, 50)) * 1000
    else:
        raise ValueError(name)
    assert len(pre) == 50
    return pre / np.median(np.diff(np.sort(pre)))


def main(name, arm, limit=1800.0):
    z = instance(name)
    kw = bcp.managed() if arm == "managed" else {}
    out = bcp.BCP(z, verbose=True, **kw).solve(time_limit=limit, heuristic_seconds=10.0)
    rec = dict(instance=name, arm=arm, status=out["status"], nodes=out["nodes"], t=round(out["t"], 1),
               root_bound=out["root_bound"], bound=out["bound"], opt=out["opt"],
               gap=(out["opt"] - out["bound"]) / out["opt"] if out["opt"] > 0 else 0.0,
               cuts=out["cuts"], max_pool=out.get("max_pool", 0), design=[list(map(int, g)) for g in out["design"]])
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0)
