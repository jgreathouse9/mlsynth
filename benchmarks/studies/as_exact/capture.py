"""Capture the authors' code as the oracle: inputs and outputs at full precision.

    python -m benchmarks.studies.as_exact.capture

Runs the replication package at the pinned commit (reference.py) on its own data and
on seeded random exterior instances, and writes ``results/reference.json`` (inputs,
the replication's outputs, and their verbatim reprs) and ``results/provenance.json``.
The study's tests read this bundle, so they run offline; ``test_exact.py`` also has a
live test that reruns the authors' code when ``MLSYNTH_AS_EXACT_LIVE=1``.
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

HERE = pathlib.Path(__file__).resolve().parent


def random_instance(seed, K, J, L=12, factors=3):
    """A factor-model panel with the treated unit pushed outward; None if X1 is in the hull."""
    from .hull import outside_hull
    rng = np.random.default_rng(seed)
    lam = rng.normal(size=(J + 1, factors))
    lam[0] *= 1.3
    f = rng.normal(size=(L + 6, factors)).cumsum(axis=0) * 0.3 + rng.normal(size=factors)
    Y = lam @ f.T + 0.3 * rng.normal(size=(J + 1, L + 6))
    C = lam @ rng.normal(size=(factors, K - 2)) + 0.5 * rng.normal(size=(J + 1, K - 2))
    X = np.column_stack([C, Y[:, 1], Y[:, 4]]).T
    X = X / X.std(axis=1, ddof=1)[:, None]
    Z = Y[:, 6:].T
    X1, X0, Z1, Z0 = X[:, 0], X[:, 1:], Z[:, 0], Z[:, 1:]
    return (X1, X0, Z1, Z0) if outside_hull(X1, X0) else None


RANDOM_SIZES = [(3, 10), (4, 15), (5, 20)]
RANDOM_PER_SIZE = 8


def random_cases():
    out = []
    seed = 1000
    for K, J in RANDOM_SIZES:
        got = 0
        while got < RANDOM_PER_SIZE:
            inst = random_instance(seed, K, J)
            seed += 1
            if inst is not None:
                out.append((f"random_K{K}_J{J}_seed{seed - 1}", inst))
                got += 1
    return out


def _arr(a):
    return np.asarray(a, float).tolist()


def main():
    paths = reference.fetch()
    ref = reference.load(paths)
    dp, vfs, pw = ref["data_preparation"], ref["visible_facet_solver"], ref["predictor_weights"]
    repo = paths["synth-AS-replication"][0]
    cases = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        for name, (X1, X0, Z1, Z0) in random_cases():
            ld, wd, _ = vfs.solve_exterior_hullfree(X0, X1, Z1, Z0)
            lp, wp, _ = vfs.solve_exterior_pd(X0, X1, Z1, Z0)
            cases.append(dict(name=name, kind="random", X1=_arr(X1), X0=_arr(X0), Z1=_arr(Z1), Z0=_arr(Z0),
                              diag=dict(loss=ld, w=_arr(wd)), pd=dict(loss=lp, w=_arr(wp))))

    import cvxpy
    import clarabel
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    prov = dict(generated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                commits={k: v[1] for k, v in paths.items()},
                urls={k: reference.URL.format(k) for k in paths},
                functions=["visible_facet_solver.solve_exterior_hullfree", "visible_facet_solver.solve_exterior_pd",
                           "predictor_weights.diag_centers", "predictor_weights.compute_pd_logdet_and_ranges",
                           "predictor_weights.analytic_center", "data_preparation.build_california",
                           "data_preparation.build_germany"],
                data_sha256={p: sha(repo / "data" / p) for p in ("smoking_data.csv", "german_reunification.csv")},
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                cvxpy=cvxpy.__version__, clarabel=clarabel.__version__, platform=platform.platform())
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "reference.json").write_text(json.dumps(dict(cases=cases)))
    (HERE / "results" / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"captured {len(cases)} cases from {prov['commits']}")


if __name__ == "__main__":
    main()
