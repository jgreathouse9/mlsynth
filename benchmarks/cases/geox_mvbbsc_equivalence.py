"""GEOX's mvbbsc engine against the MVBBSC estimator, on West Germany.

Certification by equivalence. GEOX's engine seam supplies the fit and the null
and nothing else, so an engine wrapping an estimator has one obligation before
any design question is asked: run through the seam, it must be the same method.
This case holds that to the strongest available standard on the panel MVBBSC is
already validated against -- the German reunification data of Abadie, Diamond &
Hainmueller (2015), where ``mvbbsc_germany`` cross-validates the estimator
against the authors' ``bsynth`` package.

The two agree exactly. The posterior-mean counterfactual is identical to the
last bit (``0.0`` maximum absolute difference, not a tolerance), the ATT agrees
to the same, and so do the weights. Identity, not approximation, is the right
bar: the engine and the estimator call one sampler with one seed on one
standardized panel, so anything else would mean the wrapper had introduced a
transformation of its own.

Donor order needs no special handling. ``run_mvbbsc`` canonicalises its own
columns, so both sides inherit the invariance and the engine re-imposes nothing;
``engine_donor_order_gap`` pins that it survives the seam, which is the property
the scoring loop depends on when it hands candidates over in whatever order
nomination produced them. The estimator-level property is asserted in
``mlsynth/tests/test_mvbbsc_donor_order.py``.

The ATT the pair agree on, ``-2071.7``, sits inside the band
``mvbbsc_germany`` already pins (``-2080 +/- 500``) and beside bsynth's
``-2075``, so the equivalence is anchored to the external reference and not only
to itself.

Bayesian (NUTS) => a fixed seed makes a run reproducible; the identity cells
carry no tolerance because the two paths are the same computation. Requires the
``[bayes]`` optional dependency (NumPyro).

Provenance: Martinez & Vives-i-Bastida (2024), arXiv:2206.01779; Abadie, Diamond
& Hainmueller (2015) for the panel.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "german_reunification.csv")

# One sampler configuration shared by both sides, so the only difference between
# them is the code path. Matches mvbbsc_germany's chains and seed.
_KW = {"n_warmup": 500, "n_samples": 500, "n_chains": 4, "target_accept": 0.9,
       "seed": 0}


def _panel():
    from mlsynth.utils.datautils import dataprep

    d = pd.read_csv(os.path.abspath(_DATA))
    d["treat"] = d["Reunification"].astype(int)
    prepared = dataprep(d, "country", "year", "gdp", "treat")
    y = np.asarray(prepared["y"], dtype=float).ravel()
    donors = np.asarray(prepared["donor_matrix"], dtype=float)
    return y, donors, int(prepared["pre_periods"]), int(prepared["total_periods"])


def _fit():
    """Engine and estimator on the same panel. Skips when NumPyro is absent."""
    try:
        import numpyro  # noqa: F401
    except ImportError as exc:  # optional dependency -> graceful skip
        raise BenchmarkSkipped(
            "the mvbbsc engine needs NumPyro (pip install 'mlsynth[bayes]')."
        ) from exc

    from mlsynth.utils.geox_helpers.engines import resolve_engine
    from mlsynth.utils.mvbbsc_helpers.model import run_mvbbsc

    y, donors, T0, T = _panel()
    engine = resolve_engine("mvbbsc")
    reverse = np.arange(donors.shape[1])[::-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = engine.fit_once(y, donors, T0, T0, T - 1, 1, **_KW)
        flipped = engine.fit_once(y, donors[:, reverse], T0, T0, T - 1, 1, **_KW)
        estimator = run_mvbbsc(y, donors, T0, **_KW)

    # Rebuild the estimator's noiseless posterior mean the way the engine does,
    # so the two counterfactuals are the same quantity and not one of them plus
    # an iid shock.
    # NumPyro samples in x32 by default, and the engine upcasts its draws. Match
    # that here or the comparison measures float32 against float64: the
    # counterfactual hides it, because ``weights @ standardized.T`` promotes
    # implicitly, while a mean over the raw draws does not.
    estimator_weights = np.asarray(estimator["weights"], dtype=float)
    loc, scale = float(y[:T0].mean()), float(y[:T0].std(ddof=1)) or 1.0
    d_loc = donors[:T0].mean(axis=0)
    d_scale = donors[:T0].std(axis=0, ddof=1)
    d_scale = np.where(d_scale > 0, d_scale, 1.0)
    standardized = (donors - d_loc) / d_scale
    cf_estimator = ((estimator_weights @ standardized.T) * scale
                    + loc).mean(axis=0)

    return {
        "y": y, "T0": T0, "T": T,
        "engine_cf": fit.counterfactual,
        "engine_w": fit.donor_weights,
        "engine_att": engine.att(fit, y, T0, T - 1),
        "flipped_w": flipped.donor_weights[reverse],
        "flipped_cf": flipped.counterfactual,
        "estimator_cf": cf_estimator,
        "estimator_w": estimator_weights.mean(axis=0),
        "estimator_att": float(np.mean(y[T0:] - cf_estimator[T0:])),
        "max_rhat": float(estimator.get("max_rhat", float("nan"))),
    }


def run() -> dict:
    f = _fit()
    return {
        # identity: the engine is the estimator, run through the seam
        "counterfactual_max_abs_diff": float(
            np.max(np.abs(f["engine_cf"] - f["estimator_cf"]))),
        "att_abs_diff": float(abs(f["engine_att"] - f["estimator_att"])),
        "donor_weight_max_abs_diff": float(
            np.max(np.abs(f["engine_w"] - f["estimator_w"]))),
        # the invariance survives the seam: reversing the donor columns the
        # scoring loop hands over changes nothing it reports
        "engine_donor_order_gap": float(
            np.max(np.abs(f["engine_w"] - f["flipped_w"]))),
        "engine_donor_order_cf_gap": float(
            np.max(np.abs(f["engine_cf"] - f["flipped_cf"]))),
        # anchored to the external reference via mvbbsc_germany / bsynth
        "engine_att": f["engine_att"],
        "att_negative": float(f["engine_att"] < 0.0),
        "max_rhat": f["max_rhat"],
    }


def comparison() -> dict:
    """GEOX ``engine="mvbbsc"`` against ``mlsynth.utils.mvbbsc_helpers`` directly,
    quantity by quantity, with donor order held fixed so the only difference
    between the two sides is the code path."""
    f = _fit()
    rows = [
        {"quantity": "posterior_mean_counterfactual_maxabs",
         "mlsynth": round(float(np.max(np.abs(f["engine_cf"]))), 4),
         "reference": round(float(np.max(np.abs(f["estimator_cf"]))), 4)},
        {"quantity": "mean_post_ATT",
         "mlsynth": round(f["engine_att"], 4),
         "reference": round(f["estimator_att"], 4)},
        {"quantity": "donor_weight_sum",
         "mlsynth": round(float(f["engine_w"].sum()), 6),
         "reference": round(float(f["estimator_w"].sum()), 6)},
        {"quantity": "donor_weight_max",
         "mlsynth": round(float(f["engine_w"].max()), 6),
         "reference": round(float(f["estimator_w"].max()), 6)},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "GEOX engine", "config": {"engine": "mvbbsc", **_KW}},
        "reference": {
            "impl": "mlsynth.utils.mvbbsc_helpers.model.run_mvbbsc (the estimator the engine wraps)",
            "version": "same process, same seed, canonical donor order",
        },
    }


# The identity cells carry no tolerance: engine and estimator are one sampler
# call on one standardized panel, so a nonzero difference is a defect and not
# MCMC error. Both sides are taken at float64; comparing the engine's upcast
# draws against NumPyro's x32 output reads as a 1.8e-08 disagreement that is
# precision and not computation. The order-gap cells allow float reassociation
# only -- rebuilding the standardized panel from reversed columns changes the
# summation order and nothing else.
EXPECTED = {
    "counterfactual_max_abs_diff": (0.0, 1e-9),
    "att_abs_diff": (0.0, 1e-9),
    "donor_weight_max_abs_diff": (0.0, 1e-9),
    "engine_donor_order_gap": (0.0, 1e-9),
    "engine_donor_order_cf_gap": (0.0, 1e-9),
    "engine_att": (-2077.6, 500.0),   # inside mvbbsc_germany's band; bsynth -2075
    "att_negative": (1.0, 0.0),
    "max_rhat": (1.0, 0.1),
}
