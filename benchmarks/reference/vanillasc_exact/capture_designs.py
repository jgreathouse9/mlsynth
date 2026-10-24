#!/usr/bin/env python3
"""Capture the authors' answers on many designs, the oracle of the exact backend's unit tests.

    python benchmarks/reference/vanillasc_exact/capture_designs.py

Writes ``mlsynth/tests/fixtures/exact/designs_exterior.json`` and
``designs_interior.json``, each with a ``.provenance.json`` beside it, from the authors'
code at the commits ``authors.py`` pins:

- exterior: the replication package's ``solve_exterior_hullfree`` (diagonal class) and
  ``solve_exterior_pd`` on the paper's two applications, the leakage-free Proposition 99
  sample, and 24 seeded factor-model designs with the treated unit pushed outside the
  donor hull; on the applications, also the predictor weights ``predictor_weights``
  reports at full precision (``decimals=None``);
- interior: ``synthx.estimate``, which routes a treated unit inside the donor hull to
  ``interior_cellwalk.solve_interior_cellwalk`` (diagonal) and
  ``interior_pd.solve_interior_pd``, on the Proposition 99 donors that lie inside the
  other donors' hull, each taken as a placebo's treated unit, and on 24 seeded designs
  with the treated unit pulled inside the hull. A call that raises is recorded with its
  message, and a call on which ``estimate`` fell back to its inverse-variance baseline
  is recorded as such, so that neither is taken for an answer.

The designs come from ``mlsynth/tests/_exact_support.py``, so the tests and this capture
draw the same panels. The inputs are stored at full precision.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "mlsynth" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from _exact_support import random_instance, random_interior_instance  # noqa: E402

from benchmarks.reference.vanillasc_exact.authors import REPOS, fetch, load  # noqa: E402
from mlsynth.utils.bilevel.exact.hull import outside_hull  # noqa: E402

OUT = ROOT / "mlsynth" / "tests" / "fixtures" / "exact"
EXTERIOR_SIZES, EXTERIOR_PER_SIZE = [(3, 10), (4, 15), (5, 20)], 8
INTERIOR_SIZES, INTERIOR_PER_SIZE = [(2, 8), (3, 12), (4, 16), (5, 20)], 6


def _arr(a):
    return np.asarray(a, float).tolist()


def _seeded(make, sizes, per_size, seed, prefix):
    out = []
    for K, J in sizes:
        got = 0
        while got < per_size:
            inst = make(seed, K, J)
            seed += 1
            if inst is not None:
                out.append((f"{prefix}_K{K}_J{J}_seed{seed - 1}", inst))
                got += 1
    return out


def exterior_cases(ref, repo):
    dp, vfs, pw = ref["data_preparation"], ref["visible_facet_solver"], ref["predictor_weights"]
    cases = []
    ca = dp.build_california(str(repo / "data" / "smoking_data.csv"))
    ge = dp.build_germany(str(repo / "data" / "german_reunification.csv"))
    hold = list(dp.CA_HOLDOUT_IDX)
    apps = [("prop99", ca, ca["Z1"], ca["Z0"], list(dp.CA_PREDICTORS)),
            ("prop99_leakage_free", ca, ca["Z1"][hold], ca["Z0"][hold], None),
            ("germany", ge, ge["Z1"], ge["Z0"], list(dp.GER_PREDICTORS))]
    for name, d, Z1, Z0, names in apps:
        X1, X0 = d["X1"], d["X0"]
        ld, wd, _ = vfs.solve_exterior_hullfree(X0, X1, Z1, Z0)
        lp, wp, _ = vfs.solve_exterior_pd(X0, X1, Z1, Z0)
        case = dict(name=name, kind="application", X1=_arr(X1), X0=_arr(X0), Z1=_arr(Z1), Z0=_arr(Z0),
                    donors=d["donors"], diag=dict(loss=ld, w=_arr(wd)), pd=dict(loss=lp, w=_arr(wp)))
        if names is not None:
            dc = pw.diag_centers(X0, X1, wd, names, decimals=None)
            _, la = pw.compute_pd_logdet_and_ranges(X1, X0, wd, names, decimals=None, ranges=True)
            _, ac = pw.analytic_center(X1, X0, wd, names, decimals=None)
            case["V"] = dict(predictors=names,
                             diag_logdet=[dc["logdet"][n] for n in names],
                             diag_center=[dc["analytic_center"][n] for n in names],
                             diag_forced_zero=dc["forced_zero"],
                             pd_logdet_diag=[la["diag"][n] for n in names],
                             pd_logdet_min_eig=la["min_eig"],
                             pd_lo=[la["ranges"][n][0] for n in names],
                             pd_hi=[la["ranges"][n][1] for n in names],
                             pd_center_diag=[ac["diag"][n] for n in names])
        cases.append(case)
    for name, (X1, X0, Z1, Z0) in _seeded(random_instance, EXTERIOR_SIZES, EXTERIOR_PER_SIZE, 1000, "random"):
        ld, wd, _ = vfs.solve_exterior_hullfree(X0, X1, Z1, Z0)
        lp, wp, _ = vfs.solve_exterior_pd(X0, X1, Z1, Z0)
        cases.append(dict(name=name, kind="random", X1=_arr(X1), X0=_arr(X0), Z1=_arr(Z1), Z0=_arr(Z0),
                          diag=dict(loss=ld, w=_arr(wd)), pd=dict(loss=lp, w=_arr(wp))))
    return cases


def _estimate(estimate, X1, X0, Z1, Z0, V_class):
    """The authors' answer for one class, or the error it raised."""
    try:
        out = estimate(X1, X0, Z1, Z0, V_class=V_class)
    except Exception as exc:  # noqa: BLE001 - recorded, never taken for an answer
        return dict(error=f"{type(exc).__name__}: {exc}")
    info = out["info"]
    keep = {k: info[k] for k in ("cell", "n_attainable", "n_cells_walked", "n_sdp_calls") if k in info}
    if "cell" in keep:
        keep["cell"] = [int(j) for j in keep["cell"]]
    return dict(loss=float(out["loss"]), w=_arr(out["weights"]), regime=out["regime"],
                interior_slack=float(out["interior_slack"]),
                baseline_fallback=bool(info.get("baseline_fallback", False)), **keep)


def interior_cases(ref, repo):
    estimate = ref["synthx"].estimate
    ca = ref["data_preparation"].build_california(str(repo / "data" / "smoking_data.csv"))
    X0, Z0 = np.asarray(ca["X0"], float), np.asarray(ca["Z0"], float)
    family = []
    for j in range(X0.shape[1]):
        others = [k for k in range(X0.shape[1]) if k != j]
        if not outside_hull(X0[:, j], X0[:, others]):
            family.append(("placebo", f"prop99_placebo_{ca['donors'][j].replace(' ', '_')}",
                           (X0[:, j], X0[:, others], Z0[:, j], Z0[:, others]), [ca["donors"][k] for k in others]))
    for name, inst in _seeded(random_interior_instance, INTERIOR_SIZES, INTERIOR_PER_SIZE, 4000, "random_interior"):
        family.append(("random", name, inst, None))
    cases = []
    for kind, name, (X1, X0j, Z1, Z0j), donors in family:
        case = dict(name=name, kind=kind, X1=_arr(X1), X0=_arr(X0j), Z1=_arr(Z1), Z0=_arr(Z0j),
                    diag=_estimate(estimate, X1, X0j, Z1, Z0j, "diagonal"),
                    pd=_estimate(estimate, X1, X0j, Z1, Z0j, "pd"))
        if donors is not None:
            case["donors"] = donors
        cases.append(case)
        print(name, {c: case[c].get("loss", case[c].get("error")) for c in ("diag", "pd")}, flush=True)
    return cases


def _provenance(paths, repo, functions, data):
    import clarabel
    import cvxpy
    return dict(generated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                commits={k: v[1] for k, v in paths.items()},
                urls={k: url for k, (url, _) in REPOS.items()},
                functions=functions,
                data_sha256={p: hashlib.sha256((repo / "data" / p).read_bytes()).hexdigest() for p in data},
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                cvxpy=cvxpy.__version__, clarabel=clarabel.__version__, platform=platform.platform())


def main():
    paths = fetch()
    ref = load(paths)
    repo = paths["synth-AS-replication"][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exterior = exterior_cases(ref, repo)
        interior = interior_cases(ref, repo)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "designs_exterior.json").write_text(json.dumps(dict(cases=exterior)))
    (OUT / "designs_exterior.provenance.json").write_text(json.dumps(_provenance(
        paths, repo, ["visible_facet_solver.solve_exterior_hullfree", "visible_facet_solver.solve_exterior_pd",
                      "predictor_weights.diag_centers", "predictor_weights.compute_pd_logdet_and_ranges",
                      "predictor_weights.analytic_center", "data_preparation.build_california",
                      "data_preparation.build_germany"],
        ("smoking_data.csv", "german_reunification.csv")), indent=1))
    (OUT / "designs_interior.json").write_text(json.dumps(dict(cases=interior)))
    (OUT / "designs_interior.provenance.json").write_text(json.dumps(_provenance(
        paths, repo, ["synthx.estimate (sc_estimator.py)", "interior_cellwalk.solve_interior_cellwalk",
                      "interior_pd.solve_interior_pd", "data_preparation.build_california"],
        ("smoking_data.csv",)), indent=1))
    print(f"captured {len(exterior)} exterior and {len(interior)} interior cases")


if __name__ == "__main__":
    main()
