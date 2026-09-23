"""FSC Path A: mlsynth.FSC against Okano & Kurisu (2026), sections 6.1-6.3.

The companion case :mod:`benchmarks.cases.fsc_okano` pins a faithful port of the
authors' R code and reproduces every number they publish. This one asks the
other question: what does the estimator mlsynth actually ships do on the same
data?

It reproduces the fertility application exactly and diverges on the other two,
by amounts measured here. Both divergences are corrections rather than defects,
and pinning them means a future change to either would surface as a failure
rather than drift silently.

Section 6.1, fertility curves in :math:`L^2` -- exact
    0.1259 before augmentation and 0.0687 after, from the ordinary
    configuration with the penalty cross-validated rather than supplied.

Section 6.2, age-at-death distributions -- 0.2092 exact, 0.0630 against 0.0634
    The plain estimator, which involves no basis, reproduces. The augmented one
    differs because mlsynth computes the basis inner products over the whole
    argument grid where the reference code drops the first point. On fertility
    that choice is invisible (the fertility rate at age 12 is essentially zero
    everywhere, so the dropped coordinate carries nothing, and the weights agree
    to 7e-7); on mortality it is not, because the quantile function at p = 0.01
    is a real number. Using every point the data provides is the defensible
    reading of an L2 inner product.

Section 6.3, trade covariance matrices -- a different metric on purpose
    mlsynth applies the sqrt(2) off-diagonal scaling that makes the
    half-vectorisation a genuine Frobenius isometry, which the reference code
    does not, so its fit is a Frobenius norm and the published 39.3429 is a vech
    norm. They are not comparable. The like-for-like comparator is 51.9613 --
    the authors' own weights scored under Frobenius -- and mlsynth attains a
    lower value, which is what re-optimising under the right metric should do.

One further property is pinned here rather than in the unit tests, because it
needs real data to be convincing: the penalty is searched on a scale-free
parameterisation, as a multiple of the design's own Gram scale. A fixed absolute
interval -- what the authors' scripts use -- is four orders of magnitude too
coarse on their own mortality panel once the coefficients are a genuine L2 inner
product, and lands the augmented fit at 0.1688 instead of 0.0630. The relative
penalties selected on the three applications are recorded so that regression is
visible.

Data: ``basedata/okano_fsc_{fertility,mortality,service}.csv``; see
:mod:`benchmarks.reference.fsc_okano_data` for provenance.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# (outcome, argument, space, scale, value_bounds)
_SPECS = {
    "fertility": ("asfr", "age", "function", 1.0, None),
    "mortality": ("quantile", "prob", "distribution", 1e-3, (0.0, 0.110)),
    "service": ("cov", "pair", "matrix", 1.0, None),
}

# Okano & Kurisu (2026), sections 6.1-6.3: (FSC, augmented FSC).
_PAPER_FIT = {
    "fertility": (0.1259, 0.0687),
    "mortality": (0.2092, 0.0634),
    "service": (39.3429, 20.0639),
}

# The authors' own FSC weights scored under the Frobenius metric of Example 3,
# rather than under the vech norm their code uses. Computed in fsc_okano.
_REFERENCE_FROBENIUS_FIT = 51.9613


def _fit(name: str):
    from mlsynth import FSC

    outcome, argument, space, scale, bounds = _SPECS[name]
    df = pd.read_csv(os.path.join(_ROOT, "basedata", f"okano_fsc_{name}.csv"))
    if scale != 1.0:
        df = df.assign(**{outcome: df[outcome] * scale})
    config = {
        "df": df, "outcome": outcome, "argument": argument, "space": space,
        "treat": "treat", "unitid": "unit", "time": "time",
        "inference": "none", "display_graphs": False,
    }
    if bounds is not None:
        config["value_bounds"] = bounds
    return FSC(config).fit()


def run() -> dict:
    out: dict = {}
    results = {name: _fit(name) for name in _SPECS}

    for name, res in results.items():
        paper_fsc, paper_aug = _PAPER_FIT[name]
        out[f"{name}_fsc_fit"] = float(res.pre_treatment_fit_fsc)
        out[f"{name}_afsc_fit"] = float(res.pre_treatment_fit)
        out[f"{name}_lambda_relative"] = float(res.ridge_lambda_relative)
        # eq. (10) constrains the sum, not the sign
        out[f"{name}_weight_sum"] = float(res.augmented_weights.sum())
        # augmentation must not make the pre-treatment fit worse
        out[f"{name}_fit_improvement"] = float(
            res.pre_treatment_fit_fsc - res.pre_treatment_fit)

    # --- 6.1: the exact agreement -----------------------------------------
    fsc, aug = _PAPER_FIT["fertility"]
    out["fertility_fsc_diff"] = abs(out["fertility_fsc_fit"] - fsc)
    out["fertility_afsc_diff"] = abs(out["fertility_afsc_fit"] - aug)

    # --- 6.2: exact on the plain leg, measured gap on the augmented one ----
    out["mortality_fsc_diff"] = abs(out["mortality_fsc_fit"]
                                    - _PAPER_FIT["mortality"][0])
    out["mortality_afsc_rel_gap"] = abs(
        out["mortality_afsc_fit"] / _PAPER_FIT["mortality"][1] - 1.0)

    # --- 6.3: like-for-like under the metric Example 3 specifies -----------
    out["service_frobenius_headroom"] = (
        _REFERENCE_FROBENIUS_FIT - out["service_fsc_fit"])

    # --- structural checks on real data -----------------------------------
    fertility = results["fertility"]
    out["fertility_n_donors"] = float(
        len(fertility.weights.donor_weights))
    out["fertility_negative_weights"] = float(fertility.n_negative_weights)
    mortality = results["mortality"]
    # the Wasserstein projection has to leave a valid quantile function
    out["mortality_max_decrease"] = float(max(
        -np.diff(c.counterfactual).min() for c in mortality.curves))
    # and the PSD projection a valid covariance matrix
    from mlsynth.utils.fsc_helpers.project import vech_to_matrix
    out["service_min_eigenvalue"] = float(min(
        np.linalg.eigvalsh(vech_to_matrix(c.counterfactual, 9)).min()
        for c in results["service"].curves))
    return out


# Gate. The paper prints four decimals, so "agrees with the paper" is pinned at
# that precision. The rows that are *not* claims about the paper -- the measured
# divergences, the selected penalties, the structural checks -- are pinned at the
# values this estimator produces, as regression guards.
EXPECTED = {
    # 6.1, exact
    "fertility_fsc_diff": (0.0, 5e-5),
    "fertility_afsc_diff": (0.0, 5e-5),
    # 6.2, exact on the plain leg; the augmented gap is the dropped grid point
    "mortality_fsc_diff": (0.0, 5e-5),
    "mortality_afsc_rel_gap": (0.0063, 0.002),
    # 6.3, lower is better under the metric the paper's Example 3 specifies
    "service_frobenius_headroom": (0.1948, 0.05),
    # the fits themselves, as regression guards
    "fertility_fsc_fit": (0.1259, 5e-5),
    "fertility_afsc_fit": (0.0687, 5e-5),
    "mortality_fsc_fit": (0.2092, 5e-5),
    "mortality_afsc_fit": (0.0630, 1e-3),
    "service_fsc_fit": (51.7665, 1e-3),
    "service_afsc_fit": (21.7457, 1e-2),
    # eq. (10): the sum is constrained, the sign is not
    "fertility_weight_sum": (1.0, 1e-9),
    "mortality_weight_sum": (1.0, 1e-9),
    "service_weight_sum": (1.0, 1e-8),
    # augmentation never worsens the pre-treatment fit
    "fertility_fit_improvement": (0.0572, 1e-3),
    "mortality_fit_improvement": (0.1462, 1e-3),
    "service_fit_improvement": (30.02, 0.1),
    # the scale-free penalty lands in a comparable range on all three panels
    # despite outcomes spanning six orders of magnitude
    "fertility_lambda_relative": (0.2994, 1e-2),
    "mortality_lambda_relative": (1.788, 5e-2),
    "service_lambda_relative": (5.576e-05, 1e-5),
    # structural: the projections return valid objects
    "fertility_n_donors": (20.0, 0.5),
    "mortality_max_decrease": (0.0, 1e-12),
    "service_min_eigenvalue": (0.0, 1e-8),
}


def validation_rows() -> dict:
    """Row-level export for the validation dashboard."""
    got = run()
    rows = [
        {"quantity": "6.1 fertility: pre-treatment fit, FSC",
         "mlsynth": got["fertility_fsc_fit"], "reference": 0.1259},
        {"quantity": "6.1 fertility: pre-treatment fit, augmented FSC",
         "mlsynth": got["fertility_afsc_fit"], "reference": 0.0687},
        {"quantity": "6.2 mortality: pre-treatment fit, FSC",
         "mlsynth": got["mortality_fsc_fit"], "reference": 0.2092},
        {"quantity": "6.2 mortality: pre-treatment fit, augmented FSC "
                     "(full grid, not the reference's dropped point)",
         "mlsynth": got["mortality_afsc_fit"], "reference": 0.0634},
        {"quantity": "6.3 service: pre-treatment fit, FSC "
                     "(Frobenius isometry, vs the authors' weights scored "
                     "under the same metric)",
         "mlsynth": got["service_fsc_fit"],
         "reference": _REFERENCE_FROBENIUS_FIT},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {
            "estimator": "FSC",
            "config": {"space": {k: v[2] for k, v in _SPECS.items()},
                       "inference": "none",
                       "ridge_lambda": "cross-validated (scale-free)"},
        },
        "reference": {
            "impl": "Okano & Kurisu (2026) sections 6.1-6.3, cross-checked "
                    "against the authors' R code (github.com/RyoOkano21/FSC)",
            "version": "arXiv:2601.07539v1",
        },
    }
