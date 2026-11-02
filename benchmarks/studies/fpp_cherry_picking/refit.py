"""mlsynth's own V search on FPP's specifications (the ``refit`` layer).

    python refit.py germany [processes]      # two-step, mlsynth's mscmt engine
    python refit.py california [processes]   # PlaceboSpecificationStudy (PR #958)

Germany's two-step cross-validation (choose V on the training problem, reuse
it on re-measured predictors) is not expressible as a VanillaSC config, so the
training step calls mlsynth's bilevel engine directly on the same predictor
matrices the R run used. California's specifications are single-step and run
through SpecificationStudy unchanged, one study per unit.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import fpp
import replicate as rep

warnings.filterwarnings("ignore")


def _germany_task(args):
    name, j = args
    from mlsynth.utils.bilevel import BilevelSCM
    panel = rep.germany_panel()
    d = {x.name: x for x in fpp.germany_designs()}[name]
    Y, times = panel.Y, panel.times
    others = [k for k in range(len(panel.units)) if k != j]
    Xt = fpp.synth_scale(fpp.predictor_matrix(panel, d.train))
    ssr = np.isin(times, d.train_ssr)
    t = time.time()
    res = BilevelSCM("mscmt", seed=0).fit(Y[ssr, j], Y[np.ix_(ssr, others)],
                                          X1=Xt[:, j], X0=Xt[:, others])
    loss_v = float(np.mean((Y[ssr, j] - Y[np.ix_(ssr, others)] @ res.W) ** 2))
    Xm = fpp.synth_scale(fpp.predictor_matrix(panel, d.main))
    w = fpp.custom_v_weights(Xm, res.V, j, Y[np.isin(times, d.main_ssr)])
    return name, j, {"v": res.V.tolist(), "w": w.tolist(), "loss_v": loss_v,
                     "seconds": time.time() - t}


def germany(processes: int) -> dict:
    panel = rep.germany_panel()
    tasks = [(d.name, j) for d in fpp.germany_designs() for j in range(len(panel.units))]
    out = {}
    with ProcessPoolExecutor(processes) as pool:
        for name, j, fit in pool.map(_germany_task, tasks):
            out.setdefault(name, {})[j] = fit
    return out


def california(processes: int, chunk: str = None) -> dict:
    """``chunk="k/n"`` fits units ``k, k+n, ...`` only (merge with :func:`merge_parts`)."""
    from placebo_study import PlaceboSpecificationStudy
    panel = rep.california_panel()
    units = panel.units
    if chunk:
        k, n = map(int, chunk.split("/"))
        units = units[k::n]
    designs = fpp.california_designs()
    long = fpp.add_window_columns(panel.long, "stateid", "year", designs)
    first, rest = designs[0], designs[1:]           # a1, ADH (2010)'s own spec
    e = first.registry_entry()
    base = dict(outcome="cigsale", unitid="stateid", time="year", backend="mscmt",
                inference=False, display_graphs=False, covariates=e["covariates"],
                covariate_windows=e["covariate_windows"], fit_window=e["fit_window"])
    specs = [d.registry_entry() for d in rest]
    study = PlaceboSpecificationStudy(long, base, specs, range(1989, 2001),
                                      treated=3, treat_start=1989, units=units)
    t = time.time()
    res = study.fit(processes=processes)
    names = {"baseline": first.name} | {d.name: d.name for d in rest}
    gaps = {names[s]: {str(u): col.tolist() for u, col in zip(units, res.gaps(s).T)}
            for s in names}
    fails = res.failures()
    out = {"gaps": gaps, "seconds": time.time() - t,
           "failures": fails.to_dict("records") if len(fails) else []}
    if 3 in units:
        w = res.per_unit[3].weights
        out["weights_treated"] = {
            names[s]: dict(zip(w.loc[w.spec_id == s, "donor"].astype(str),
                               w.loc[w.spec_id == s, "weight"])) for s in names}
    return out


def merge_parts(n: int) -> dict:
    parts = [json.loads((rep.RESULTS / f"california_refit_{k}of{n}.json").read_text())
             for k in range(n)]
    out = {"gaps": {}, "seconds": sum(p["seconds"] for p in parts), "failures": []}
    for p in parts:
        for s, cols in p["gaps"].items():
            out["gaps"].setdefault(s, {}).update(cols)
        out["failures"] += p["failures"]
        out.update({k: v for k, v in p.items() if k == "weights_treated"})
    return out


if __name__ == "__main__":
    app = sys.argv[1]
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    rep.RESULTS.mkdir(exist_ok=True)
    if app == "california-merge":
        out, name = merge_parts(procs), "california_refit"
    elif app == "california" and len(sys.argv) > 3:
        k, n = sys.argv[3].split("/")
        out, name = california(procs, sys.argv[3]), f"california_refit_{k}of{n}"
    else:
        out, name = {"germany": germany, "california": california}[app](procs), f"{app}_refit"
    (rep.RESULTS / f"{name}.json").write_text(json.dumps(out))
    print(f"wrote results/{name}.json")
