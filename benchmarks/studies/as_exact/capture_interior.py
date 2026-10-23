"""Capture the authors' interior solvers as the oracle for the interior case.

    python -m benchmarks.studies.as_exact.capture_interior    # -> results/reference_interior.json

The replication package has no interior case; the synthx package does, through
``estimate`` (``sc_estimator.py``), which standardises the predictors and routes an
interior treated unit to ``interior_cellwalk.solve_interior_cellwalk`` for diagonal V and
to ``interior_pd.solve_interior_pd`` for positive definite V (paper, Section 5). Two
families of instances:

  - placebo: the donors of Proposition 99 whose predictors lie inside the hull of the
    other donors, each taken as the treated unit the way an in-space placebo takes it,
    under the paper's validation sample. The leakage-free sample poses the same problem
    here: the three years it leaves out are the lagged outcomes among the predictors,
    which every exact match reproduces, so their residuals vanish in every cell;
  - random: seeded factor-model designs with the treated unit pulled toward the donors
    (``capture.random_interior_instance``), small enough that every set of K + 1 donors
    can also be checked by brute force.

A call that raises is recorded with its message, and a call on which ``estimate`` fell
back to its inverse-variance baseline is recorded as such, so that neither is taken for
an answer of the interior search.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import datetime          # noqa: E402
import hashlib           # noqa: E402
import json              # noqa: E402
import pathlib           # noqa: E402
import platform          # noqa: E402
import warnings          # noqa: E402

import numpy as np       # noqa: E402
import scipy             # noqa: E402

from . import reference  # noqa: E402
from .capture import random_interior_instance  # noqa: E402
from .hull import outside_hull  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RANDOM_SIZES = [(2, 8), (3, 12), (4, 16), (5, 20)]
RANDOM_PER_SIZE = 6


def _arr(a):
    return np.asarray(a, float).tolist()


def _solve(estimate, X1, X0, Z1, Z0, V_class):
    """The authors' answer for one class, or the error it raised."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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


def placebo_cases(dp, repo):
    """Proposition 99's interior donors as treated units, under the paper's validation sample."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ca = dp.build_california(str(repo / "data" / "smoking_data.csv"))
    X0, Z0 = np.asarray(ca["X0"], float), np.asarray(ca["Z0"], float)
    out = []
    for j in range(X0.shape[1]):
        others = [k for k in range(X0.shape[1]) if k != j]
        if not outside_hull(X0[:, j], X0[:, others]):
            out.append((f"prop99_placebo_{ca['donors'][j].replace(' ', '_')}",
                        (X0[:, j], X0[:, others], Z0[:, j], Z0[:, others]),
                        [ca["donors"][k] for k in others]))
    return out


def random_cases():
    out, seed = [], 4000
    for K, J in RANDOM_SIZES:
        got = 0
        while got < RANDOM_PER_SIZE:
            inst = random_interior_instance(seed, K, J)
            seed += 1
            if inst is not None:
                out.append((f"random_interior_K{K}_J{J}_seed{seed - 1}", inst, None))
                got += 1
    return out


def main():
    paths = reference.fetch()
    ref = reference.load(paths)
    estimate = ref["synthx"].estimate
    repo = paths["synth-AS-replication"][0]
    cases = []
    for kind, family in (("placebo", placebo_cases(ref["data_preparation"], repo)), ("random", random_cases())):
        for name, (X1, X0, Z1, Z0), donors in family:
            case = dict(name=name, kind=kind, X1=_arr(X1), X0=_arr(X0), Z1=_arr(Z1), Z0=_arr(Z0),
                        diag=_solve(estimate, X1, X0, Z1, Z0, "diagonal"),
                        pd=_solve(estimate, X1, X0, Z1, Z0, "pd"))
            if donors is not None:
                case["donors"] = donors
            cases.append(case)
            print(name, {c: {k: case[c][k] for k in ("loss", "regime", "n_attainable", "error") if k in case[c]}
                         for c in ("diag", "pd")}, flush=True)

    import cvxpy
    import clarabel
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    prov = dict(generated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                commits={k: v[1] for k, v in paths.items()},
                urls={k: reference.URL.format(k) for k in paths},
                functions=["synthx.estimate (sc_estimator.py)", "interior_cellwalk.solve_interior_cellwalk",
                           "interior_pd.solve_interior_pd", "data_preparation.build_california"],
                data_sha256={"smoking_data.csv": sha(repo / "data" / "smoking_data.csv")},
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                cvxpy=cvxpy.__version__, clarabel=clarabel.__version__, platform=platform.platform())
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "reference_interior.json").write_text(json.dumps(dict(cases=cases)))
    (HERE / "results" / "provenance_interior.json").write_text(json.dumps(prov, indent=1))
    print(f"captured {len(cases)} interior cases from {prov['commits']}")


if __name__ == "__main__":
    main()
