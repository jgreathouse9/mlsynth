"""The cost-ruled certified search on GeoLift, against the exact backend.

For each treated count the line gives the design, the certified gap, the
lifts and their time, the sets enumerated and fitted, and the total, beside
the library's exact backend (search_cluster with its limit raised; times from
this study's RCA: 0.30 s, 3.94 s and 105.1 s for 4, 6 and 8 treated).

From the repository root::

    python -m benchmarks.studies.marex_bnb.run_hybrid 4 6 8
"""
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

from benchmarks.studies.marex_bnb.hybrid import search
from benchmarks.studies.marex_bnb.panels import geolift, walmart

EXACT = {4: (554319.7579, 0.30), 6: (297926.2374, 3.94), 8: (181943.3763, 105.1)}


def main(ms, kappa=1.0, bounder="treated", panel="geolift"):
    B, A = walmart() if panel == "walmart" else geolift()
    if panel == "walmart":
        global EXACT
        EXACT = {}
    for m in ms:
        res = search(B, A, m, kappa=kappa, log=print, bounder=bounder)
        st = res["stats"]
        value, secs = EXACT.get(m, (None, None))
        print(f"m={m} kappa={kappa} bounder={bounder}: set {res['set']}  hi {float(res['hi']):.10g}  gap "
              f"{float(res['gap'] / res['hi']):.2e}  open {res['open']}  unresolved "
              f"{res['unresolved']}  lifts {st['lifts']} ({st['lift_s']:.0f} s)  "
              f"subtrees enumerated {st['enumerated']} ({st['enumerated_sets']:,} sets, "
              f"{st['enumerate_s']:.0f} s)  fitted {st['fitted']}  price "
              f"{st['price_cost'] * 1e6:.2f} us/set  total {res['seconds']:.1f} s  |  exact "
              f"backend {value} in {secs} s", flush=True)
        if value is not None:
            assert abs(float(res["hi"]) - value) <= 1e-9 * value


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if "=" not in a]
    kap = [float(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("kappa=")]
    bnd = [a.split("=")[1] for a in sys.argv[1:] if a.startswith("bounder=")]
    pan = [a.split("=")[1] for a in sys.argv[1:] if a.startswith("panel=")]
    main([int(a) for a in args] or [4, 6, 8], kappa=kap[0] if kap else 1.0,
         bounder=bnd[0] if bnd else "treated", panel=pan[0] if pan else "geolift")
