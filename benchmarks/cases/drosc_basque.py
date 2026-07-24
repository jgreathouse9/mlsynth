"""Golden cross-validation: DROSC vs the authors' own R ``DRoSC``, run live.

Koo & Guo (2026), *"Distributionally Robust Synthetic Control"* (arXiv:2511.02632),
target a worst-case treatment effect over the donor weights compatible with the
pre-treatment moments up to a robustness radius ``lambda``. This case validates
mlsynth's :class:`mlsynth.DROSC` against the authors' *own* code, **executed live
via ``Rscript``**: ``benchmarks/reference/drosc_basque/reference.R`` sources the
authors' unmodified ``src/helpers.R`` (cloned from github.com/taehyeonkoo/DRoSC)
and runs their ``sc()`` + ``DRoSC()`` (``limSolve::lsei``) on the shipped Basque
panel each time this case runs. It ``BenchmarkSkipped``s when ``Rscript`` /
``limSolve`` / the clone is unavailable, so a missing R toolchain never reds the
suite.

The panel is the Abadie-Gardeazabal Basque Country / ETA-terrorism study
(``basedata/basque_jasa.csv``): T0 = 15 pre-periods, N = 16 donor regions,
T1 = 28 post-periods. The deterministic worst-case point estimand is exact on
both sides -- mlsynth's cvxpy solve and the authors' ``lsei`` agree
value-for-value. As the robustness radius grows the effect shrinks from the
classical-SC neighbourhood toward zero (tau: -0.742 -> -0.256 -> 0 at
lambda = 0 / 0.03 / 0.06), and the lambda = 0 donor weights match by name
(Madrid 0.388, Baleares 0.274, Cataluna 0.203, Asturias 0.135). The perturbation
union CI is stochastic (seed-dependent) and is not pinned here; the deterministic
estimand is.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import warnings

from benchmarks.compare import BenchmarkSkipped

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_DATA = os.path.abspath(os.path.join(_ROOT, "basedata", "basque_jasa.csv"))
_REF_R = os.path.abspath(os.path.join(
    _ROOT, "benchmarks", "reference", "drosc_basque", "reference.R"))
_CACHE = os.path.abspath(os.path.join(_ROOT, "benchmarks", "reference", ".cache"))
_LAMBDAS = (0.0, 0.015, 0.03, 0.045, 0.06)
_ref_cache = {}


def _key(lam):
    return f"tau_lam{lam:.2f}" if lam in (0.0, 0.03, 0.06) else f"tau_lam{lam:g}"


def _reference_live() -> dict:
    """Run the authors' DRoSC live via ``Rscript`` (sourcing their helpers.R).

    Skips (``BenchmarkSkipped``) when ``Rscript`` / ``limSolve`` / the clone is
    unavailable. Returns ``{"values": {...tau...}, "weights": {region: w}}``.
    """
    if _ref_cache:
        return _ref_cache["ref"]
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise BenchmarkSkipped("Rscript not on PATH (install R + the limSolve package)")
    probe = subprocess.run([rscript, "-e", "suppressMessages(library(limSolve))"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        raise BenchmarkSkipped("R package 'limSolve' not installed")
    env = {**os.environ, "DROSC_CACHE": _CACHE}
    out = subprocess.run([rscript, _REF_R, _DATA], capture_output=True, text=True, env=env)
    if out.returncode != 0:
        raise BenchmarkSkipped(f"DRoSC reference failed: {out.stderr.strip()[-200:]}")
    try:
        ref = json.loads(out.stdout.strip())
    except json.JSONDecodeError as exc:
        raise BenchmarkSkipped(f"could not parse reference.R output: {exc}") from exc
    _ref_cache["ref"] = ref
    return ref


def _fit(lam):
    from mlsynth import DROSC
    import pandas as pd

    d = pd.read_csv(_DATA)
    d = d[d.regionname != "Spain (Espana)"].copy()
    treat_year = sorted(d.year.unique())[15]
    d["treat"] = ((d.regionname == "Basque Country (Pais Vasco)")
                  & (d.year >= treat_year)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DROSC({"df": d, "outcome": "gdpcap", "treat": "treat",
                      "unitid": "regionname", "time": "year",
                      "robustness_lambda": lam, "display_graphs": False}).fit()


def run() -> dict:
    ref = _reference_live()                          # skips if R/limSolve absent
    rv = ref["values"]
    rw = ref["weights"]
    res = {lam: _fit(lam) for lam in _LAMBDAS}
    tau = {lam: float(res[lam].effects.att) for lam in _LAMBDAS}
    w0 = res[0.0].weights.donor_weights
    by = lambda frag: next(v for n, v in w0.items() if frag in n)
    r_by = lambda frag: next(v for n, v in rw.items() if frag in n)

    tau_dev = max(abs(tau[lam] - rv[_key(lam)]) for lam in _LAMBDAS)
    weight_l1 = sum(abs(by(f) - r_by(f))
                    for f in ("Madrid", "Baleares", "Cataluna", "Asturias"))
    return {
        # DROSC reproduces the LIVE R worst-case estimand value-for-value
        "tau_max_abs_dev_vs_R": tau_dev,
        "classical_sc_att_dev_vs_R": abs(
            float(res[0.0].additional_outputs["classical_sc_att"]) - rv["tau_SC"]),
        "weight_l1_vs_R": weight_l1,
        "agrees_with_r": float(tau_dev < 1e-2 and weight_l1 < 1e-2),
        # mlsynth descriptors / regression guards
        "tau_lam0": tau[0.0],
        "tau_lam06": tau[0.06],
        "shrinks_to_zero": float(abs(tau[0.0]) > abs(tau[0.03]) > abs(tau[0.06])
                                 and abs(tau[0.06]) < 1e-2),
        "weights_sum_to_one": float(sum(w0.values())),
    }


def comparison() -> dict:
    """mlsynth DROSC vs the authors' R DRoSC (live via Rscript), quantity by quantity."""
    ref = _reference_live()
    rv, rw = ref["values"], ref["weights"]
    rows = []
    for lam in _LAMBDAS:
        rows.append({"quantity": f"tau (lambda={lam:g})",
                     "mlsynth": round(float(_fit(lam).effects.att), 4),
                     "reference": round(rv[_key(lam)], 4)})
    res0 = _fit(0.0)
    w0 = res0.weights.donor_weights
    rows.append({"quantity": "classical SC ATT",
                 "mlsynth": round(float(res0.additional_outputs["classical_sc_att"]), 4),
                 "reference": round(rv["tau_SC"], 4)})
    for frag in ("Madrid", "Baleares", "Cataluna", "Asturias"):
        rows.append({"quantity": f"w[{frag}] (lambda=0)",
                     "mlsynth": round(next(v for n, v in w0.items() if frag in n), 4),
                     "reference": round(next(v for n, v in rw.items() if frag in n), 4)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "DROSC", "config": {"backend": "lsei-band"}},
        "reference": {"impl": "authors' helpers.R sc() + DRoSC() (limSolve::lsei, live via Rscript)",
                      "version": "Koo & Guo (2026), arXiv:2511.02632"},
    }


# Cross-validation (scenario: full repo, live R). The deterministic worst-case
# estimand is exact on both sides (cvxpy vs lsei); tolerances cover solver noise
# on the shared optimum. Skips if Rscript / limSolve / the clone is absent.
EXPECTED = {
    "tau_max_abs_dev_vs_R": (0.0, 0.02),
    "classical_sc_att_dev_vs_R": (0.0, 0.01),
    "weight_l1_vs_R": (0.0, 0.03),
    "agrees_with_r": (1.0, 0.0),
    "tau_lam0": (-0.7424, 0.02),
    "tau_lam06": (0.0, 0.02),
    "shrinks_to_zero": (1.0, 0.0),
    "weights_sum_to_one": (1.0, 1e-4),
}
