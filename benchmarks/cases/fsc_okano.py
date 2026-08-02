"""FSC Path A: Okano & Kurisu (2026), all three empirical applications.

Okano, R. & Kurisu, D. (2026). *"Functional Synthetic Control Methods for Metric
Space-Valued Outcomes."* arXiv:2601.07539.
Replication package: https://github.com/RyoOkano21/FSC

This case pins the reference port in
:mod:`benchmarks.reference.fsc_okano_ref` -- a faithful transcription of the
authors' R code -- against every number the paper reports for its three
applications. It answers whether the method as its authors implemented it is
correctly understood here. :mod:`benchmarks.cases.fsc_estimator` asks the other
question, of what :class:`mlsynth.FSC` itself produces on the same data.

Reproduced exactly (four decimals, the precision the paper prints):

* section 6.1, East German abortion law, age-specific fertility curves in
  :math:`L^2([12, 55])`: pre-treatment fit 0.1259 (FSC) and 0.0687 (augmented),
  and all twenty augmented donor weights of Table 1 to three decimals;
* section 6.2, Soviet collapse, age-at-death quantile functions in the
  Wasserstein space: 0.2092 and 0.0634, and all seventeen augmented weights of
  Table 2;
* section 6.3, Brexit announcement, service-trade covariance matrices in
  :math:`\\mathrm{Sym}^+_9`: 39.3429 and 20.0639, and the three nonzero FSC
  weights of Table 3.

The cross-validated penalties of Remark 5 reproduce as well, checked at the spike
against the values the authors hard-code after running ``optimise``: mortality
lands on 5.889182 to eight digits and service on 0.001864 against a printed
0.00186. Fertility lands on 5.794844 rather than 5.759146 -- at a marginally
*lower* CV error, so this is Brent's bracket and not a porting error. The reason
it can differ at all is recorded as the ``lambda_cv_*`` rows: the CV objective is
nearly flat in the penalty. Over the whole range 5 to 7 it varies by 0.03 percent
while the pre-treatment fit moves by 0.0005, in its fourth decimal, so the
penalty is weakly identified and the estimate barely notices. The gate therefore
pins the flatness and the resulting estimates, not the argmin -- chasing the
authors' penalty to the digit would be pinning noise. The full ``select_lambda``
solve is left out of the gate because it costs a minute per application and
certifies nothing the flatness rows do not.

Four details of the reference code are load-bearing and are recorded here rather
than smoothed over, because a build has to decide what to do about each.

1. ``FSCM`` returns ``round(weight_scm, 4)``. On the service application the
   rounding is what separates the published 39.3429 from the exact QP's 39.3423;
   the other two applications are insensitive to it.
2. The pre-treatment fit is computed as ``sum(diffs[-1])`` -- dropping the first
   grid point -- for the FSC leg of all three applications, but the augmented leg
   is coded as ``sum(diffs)``. The published augmented figures are recovered
   under drop-first for fertility and mortality and under all-points for service,
   so the three reported pairs are not on one common norm. The gate below asks
   for each number under the convention that produced it and additionally records
   the service figure under both.
3. ``Rearrangement::rearrangement`` is not a sort (see the reference module).
4. The service outcome is the plain half-vectorisation of the covariance matrix,
   with no :math:`\\sqrt 2` on the off-diagonals. That is not the isometry for
   the Frobenius metric of the paper's Example 3 -- it understates off-diagonal
   discrepancies by a factor of two -- so 39.3429 is a vech norm, not a Frobenius
   norm. Recorded as ``service_frobenius_fsc_fit``: the same weights scored under
   the metric Example 3 actually specifies.

Data: ``basedata/okano_fsc_{fertility,mortality,service}.csv``, converted from the
authors' ``asfr.RData`` / ``aad.RData`` / ``service.RData`` with the ``rdata``
package (no R required). One row per (unit, time, argument): age 12-55 for
fertility, quantile level for mortality, and the 45 lower-triangle entries of the
9x9 covariance for service. Underlying sources are the Human Fertility Database,
the Human Mortality Database, and UN Trade and Development.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from benchmarks.reference import fsc_okano_ref as ref

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

N_BASIS = 50                      # K in sections 6.1-6.2

# Paper section 6 fits: (FSC, augmented FSC).
_PAPER_FIT = {
    "fertility": (0.1259, 0.0687),
    "mortality": (0.2092, 0.0634),
    "service": (39.3429, 20.0639),
}

# Paper Table 1: augmented FSC weights. Keyed by donor, because dataprep orders
# the donor pool alphabetically and the paper's tables do not.
_PAPER_TABLE1 = {
    "Austria": 0.526, "Belgium": 0.093, "Bulgaria": 0.382, "Canada": -0.090,
    "Switzerland": -0.026, "Czechia": 0.158, "Denmark": 0.036, "Spain": -0.059,
    "Finland": 0.004, "France": 0.099, "Hungary": -0.041, "Ireland": 0.075,
    "Italy": -0.144, "Japan": -0.063, "Netherlands": 0.033, "Portugal": -0.119,
    "Slovakia": 0.116, "Sweden": 0.077, "United Kingdom": 0.088,
    "United States": -0.144,
}
# Paper Table 2: augmented FSC weights.
_PAPER_TABLE2 = {
    "Austria": 0.046, "Belgium": -0.116, "Denmark": 0.282, "Finland": 0.461,
    "France": 1.258, "Germany": 0.034, "Iceland": 0.147, "Ireland": 0.134,
    "Italy": -0.068, "Luxembourg": 0.071, "Netherlands": -0.334,
    "Norway": -0.448, "Portugal": 0.898, "Spain": -0.943, "Sweden": -0.076,
    "Switzerland": -0.380, "United Kingdom": 0.034,
}
# Paper Table 3: the three donors the FSC gives nonzero weight.
_PAPER_TABLE3_FSC = {"France": 0.596, "Greece": 0.290, "United States": 0.115}

# The authors' scripts hard-code these after running optimise() (Remark 5).
_PAPER_LAMBDA = {"fertility": 5.759146, "mortality": 5.889182,
                 "service": 0.00186}
# The search intervals the authors pass to optimise(). Not used by the gate --
# recorded because a build has to pick a default penalty range, and these are
# the ones that produced the published estimates.
_LAMBDA_INTERVAL = {"fertility": (0.0, 10.0), "mortality": (0.0, 10.0),
                    "service": (0.0, 1.0)}


def _cube(name: str, argument: str, value: str, scale: float = 1.0):
    """Pivot the long panel to ``(N, T, M)`` with the treated unit first.

    Each grid slice goes through :func:`mlsynth.utils.datautils.dataprep`, which
    is what fixes the treated/donor split, the period ordering and the
    pre-period count; the slices are then stacked along the argument axis.
    """
    from mlsynth.utils.datautils import dataprep

    df = pd.read_csv(os.path.join(_ROOT, "basedata", f"okano_fsc_{name}.csv"))
    args = list(dict.fromkeys(df[argument]))
    slices, pre_periods, donors = [], None, None
    for a in args:
        prepped = dataprep(
            df[df[argument] == a].drop(columns=[argument]),
            unit_id_column_name="unit", time_period_column_name="time",
            outcome_column_name=value, treatment_indicator_column_name="treat",
        )
        slices.append(np.vstack([np.asarray(prepped["y"]).ravel(),
                                 np.asarray(prepped["donor_matrix"]).T]))
        pre_periods = prepped["pre_periods"]
        donors = list(prepped["donor_names"])
    return np.stack(slices, axis=-1) * scale, int(pre_periods), donors


def _grid(name: str) -> np.ndarray:
    """The argument grid each script passes to ``Bsplines``."""
    return {"fertility": np.linspace(0.0, 1.0, 44),
            "mortality": np.linspace(0.01, 0.99, 100)}[name]


def _basis(name: str) -> np.ndarray:
    return ref.cubic_bsplines(_grid(name), np.linspace(0.0, 1.0, N_BASIS - 2))


def _spec(name: str) -> dict:
    """Basis / grid-width keyword arguments for the augmented weights."""
    if name == "service":                       # Euclidean: FSCM_aug_covmat
        return {}
    grid = _grid(name)
    return {"basis": _basis(name), "grid_width": 1.0 / grid.size}


def run() -> dict:
    out: dict = {}
    cubes = {}

    for name, scale in (("fertility", 1.0), ("mortality", 1e-3),
                        ("service", 1.0)):
        Y, n_pre, donors = _cube(
            name,
            {"fertility": "age", "mortality": "prob", "service": "pair"}[name],
            {"fertility": "asfr", "mortality": "quantile",
             "service": "cov"}[name],
            scale=scale,
        )
        cubes[name] = (Y, n_pre, donors)
        spec = _spec(name)

        w = ref.fsc_weights(Y, n_pre)
        fit_fsc = ref.pre_treatment_fit(Y, n_pre, ref.synthetic(Y, w),
                                        drop_first_grid_point=True)

        w_aug = ref.augmented_fsc_weights(Y, n_pre, _PAPER_LAMBDA[name], **spec)
        synth = ref.synthetic(Y, w_aug)
        if name == "mortality":                 # project back onto Psi(M)
            grid = _grid(name)
            synth = np.array([ref.project_quantile(grid, s, 0.0, 110 / 1000)
                              for s in synth])
        # detail 2: each published figure under the convention that produced it
        drop = name != "service"
        fit_aug = ref.pre_treatment_fit(Y, n_pre, synth,
                                        drop_first_grid_point=drop)

        paper_fsc, paper_aug = _PAPER_FIT[name]
        out[f"{name}_fsc_fit"] = fit_fsc
        out[f"{name}_afsc_fit"] = fit_aug
        out[f"{name}_fsc_fit_diff"] = abs(fit_fsc - paper_fsc)
        out[f"{name}_afsc_fit_diff"] = abs(fit_aug - paper_aug)
        out[f"{name}_n_donors"] = float(len(donors))
        out[f"{name}_n_pre"] = float(n_pre)
        out[f"{name}_weights_sum"] = float(w_aug.sum())

    # --- published weight tables ------------------------------------------
    def _table_diff(name: str, paper: dict) -> float:
        Y, n_pre, donors = cubes[name]
        w = ref.augmented_fsc_weights(Y, n_pre, _PAPER_LAMBDA[name],
                                      **_spec(name))
        fitted = dict(zip(donors, w))
        assert set(fitted) == set(paper), set(fitted) ^ set(paper)
        return max(abs(round(fitted[d], 3) - v) for d, v in paper.items())

    out["table1_max_diff"] = _table_diff("fertility", _PAPER_TABLE1)
    out["table2_max_diff"] = _table_diff("mortality", _PAPER_TABLE2)

    Y, n_pre, donors = cubes["service"]
    w3 = ref.fsc_weights(Y, n_pre)
    fitted = {d: v for d, v in zip(donors, w3) if v > 1e-4}
    out["table3_max_diff"] = max(
        abs(round(fitted.get(d, 0.0), 3) - v)
        for d, v in _PAPER_TABLE3_FSC.items())
    out["table3_n_nonzero"] = float(len(fitted))

    # --- detail 1: the effect of FSCM's round(w, 4) ------------------------
    exact = ref.fsc_weights(Y, n_pre, round_to=None)
    out["service_fsc_fit_unrounded"] = ref.pre_treatment_fit(
        Y, n_pre, ref.synthetic(Y, exact), drop_first_grid_point=True)

    # --- detail 2: the service augmented fit under the other convention ----
    w_aug = ref.augmented_fsc_weights(Y, n_pre, _PAPER_LAMBDA["service"])
    out["service_afsc_fit_dropfirst"] = ref.pre_treatment_fit(
        Y, n_pre, ref.synthetic(Y, w_aug), drop_first_grid_point=True)

    # --- detail 4: the same weights under the Frobenius metric -------------
    # vech double-counts nothing, so restore the off-diagonals: ||A||_F^2 =
    # 2 ||vech(A)||^2 - ||diag(A)||^2.
    lo, hi = np.tril_indices(9)
    weight = np.where(lo == hi, 1.0, 2.0)
    gap = (Y[0, :n_pre] - ref.synthetic(Y, w3)[:n_pre]) ** 2
    out["service_frobenius_fsc_fit"] = float(np.sqrt((gap * weight).sum()))

    # --- Remark 5: how sharply is the penalty identified? ------------------
    # Three points spanning the authors' selected value. If the CV objective is
    # flat across them, the argmin is not a meaningful replication target and
    # the estimate is what has to agree.
    Y, n_pre, _ = cubes["fertility"]
    spec = _spec("fertility")
    lam0 = _PAPER_LAMBDA["fertility"]
    errors, fits = [], []
    for lam in (5.0, lam0, 7.0):
        errors.append(ref.cross_val(lam, Y, n_pre, **spec))
        w = ref.augmented_fsc_weights(Y, n_pre, lam, **spec)
        fits.append(ref.pre_treatment_fit(Y, n_pre, ref.synthetic(Y, w),
                                          drop_first_grid_point=True))
    out["lambda_cv_error_range"] = float(
        (max(errors) - min(errors)) / min(errors))
    out["lambda_cv_fit_range"] = float(max(fits) - min(fits))
    return out


# Gate. The paper prints its fits to four decimals and its weights to three, so
# the tolerances below are "agrees at the precision the paper reports". The
# *_diff rows are the claim of exact agreement; the raw fits are pinned as
# regression guards. The recorded-detail rows carry no paper counterpart and are
# pinned at the values this port produces.
EXPECTED = {
    # section 6.1-6.3 pre-treatment fits, each vs the published figure
    "fertility_fsc_fit_diff": (0.0, 5e-5),
    "fertility_afsc_fit_diff": (0.0, 5e-5),
    "mortality_fsc_fit_diff": (0.0, 5e-5),
    "mortality_afsc_fit_diff": (0.0, 5e-5),
    "service_fsc_fit_diff": (0.0, 5e-5),
    "service_afsc_fit_diff": (0.0, 5e-5),
    # published weight tables, at the printed three decimals
    "table1_max_diff": (0.0, 5e-4),
    "table2_max_diff": (0.0, 5e-4),
    "table3_max_diff": (0.0, 5e-4),
    "table3_n_nonzero": (3.0, 0.5),
    # panel shapes, so a data regression cannot hide behind a matching fit
    "fertility_n_donors": (20.0, 0.5),
    "fertility_n_pre": (16.0, 0.5),
    "mortality_n_donors": (17.0, 0.5),
    "mortality_n_pre": (21.0, 0.5),
    "service_n_donors": (22.0, 0.5),
    "service_n_pre": (29.0, 0.5),
    # eq. (9) leaves the simplex but keeps the sum-to-one constraint
    "fertility_weights_sum": (1.0, 1e-9),
    "mortality_weights_sum": (1.0, 1e-9),
    "service_weights_sum": (1.0, 1e-9),
    # recorded details (see the module docstring)
    "service_fsc_fit_unrounded": (39.3423, 1e-3),
    "service_afsc_fit_dropfirst": (19.7807, 1e-3),
    "service_frobenius_fsc_fit": (51.9613, 1e-3),
    # Remark 5: the penalty is weakly identified. Across lambda in [5, 7] the CV
    # objective varies by well under 0.1 percent while the fit moves by under
    # 0.001, which is why the gate above targets the estimate and not the argmin.
    "lambda_cv_error_range": (2.805e-4, 1e-5),
    "lambda_cv_fit_range": (5.4e-4, 1e-4),
}


def validation_rows() -> dict:
    """Row-level export for the validation dashboard."""
    got = run()
    rows = []
    for name in ("fertility", "mortality", "service"):
        paper_fsc, paper_aug = _PAPER_FIT[name]
        rows.append({"quantity": f"{name}: pre-treatment fit, FSC",
                     "mlsynth": got[f"{name}_fsc_fit"], "reference": paper_fsc})
        rows.append({"quantity": f"{name}: pre-treatment fit, augmented FSC",
                     "mlsynth": got[f"{name}_afsc_fit"], "reference": paper_aug})
    rows.append({"quantity": "Table 1 augmented weights, max |diff|",
                 "mlsynth": got["table1_max_diff"], "reference": 0.0})
    rows.append({"quantity": "Table 2 augmented weights, max |diff|",
                 "mlsynth": got["table2_max_diff"], "reference": 0.0})
    rows.append({"quantity": "Table 3 FSC weights, max |diff|",
                 "mlsynth": got["table3_max_diff"], "reference": 0.0})
    return {
        "rows": rows,
        "mlsynth_call": {
            "estimator": "(reference port: benchmarks.reference.fsc_okano_ref)",
            "config": {"n_basis": N_BASIS, "lambda": _PAPER_LAMBDA},
        },
        "reference": {
            "impl": "Okano & Kurisu (2026) published sections 6.1-6.3 and "
                    "Tables 1-3, cross-checked against the authors' R code "
                    "(github.com/RyoOkano21/FSC)",
            "version": "arXiv:2601.07539v1",
        },
    }
