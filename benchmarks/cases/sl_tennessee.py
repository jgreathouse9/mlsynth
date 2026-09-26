"""SL against an R implementation of Viviano and Bradic's own expert library.

Cross-validation, on their Medicaid-expansion panel: Tennessee against the six
southern states that did not expand, 100 quarters, experts trained on 1-30,
weights on 31-50, treatment at 51. The reference bundle is
``benchmarks/reference/sl_tennessee``; its ``reference.R`` transcribes
``generate_experts`` and ``Exp_algorithm`` from their ``libraries/library.R`` and
records every line where it departs from them.

The random forest is left out of the compared library on purpose. randomForest
and scikit-learn's ``RandomForestRegressor`` are different implementations, so its
path cannot agree cell for cell in any language pair, and including it would bound
the measured accuracy of the port by that gap instead of by the port. The other
three experts are algorithmically determined, and they are what this case pins.

This case is why two defects in the port are no longer in it. Neither was visible
from the paper, from the estimator's own tests, or from the simulation benchmark:

* the penalty grid. ``generate_experts`` passes ``cv.glmnet`` two explicit and
  different grids -- ``seq(exp(-10), exp(-1), 79)`` for the lasso expert and
  ``seq(exp(-10), exp(2), 79)`` for the factor expert -- and the port let
  scikit-learn derive its own. On this window the mean cross-validated error
  varies by a factor of only 1.064 across the whole grid, so the curve barely
  separates the null model from the six-donor one and whichever grid is searched
  decides the answer: the paper's puts the lasso expert's penalty at 0.368 keeping
  no donors, scikit-learn's default put it at 5.97e-05 keeping five.
* standardization. glmnet standardizes the design by default and the authors do
  not turn it off, and the divisor is the population standard deviation. Fitting
  unstandardized left the factor expert's path 2.1e-02 from the reference; with
  it, 7.0e-07. The sample standard deviation is not a substitute, at 6.8e-06.

Two of the path figures are bounded by the capture and not by the port. The
bundle prints at ten decimals, so no path comparison here can resolve below
5e-11, and ``did_path_max_abs_diff`` is that limit: the expert delegates to
``did_from_mean``, the difference-in-differences FDID reports beside its forward
fit, and against the authors' own line evaluated at full precision it agrees to
5.6e-17. ``lasso_path_max_abs_diff`` is the same limit met once instead of a
hundred times, since at the penalty their grid selects that expert keeps no
donors and its path is one constant.

One number here is a floor and not a target. The factor expert's penalty lands at
the grid's minimum, where the fit is nearly unregularized on a 30-by-6 design of
highly correlated state series, and the two solvers converge to different points
of a near-flat optimum: tightening scikit-learn's tolerance from 1e-04 to 1e-13
moves the disagreement from 7.0e-07 to 9.2e-07 and no further. So the ensemble
quantities agree to a few parts per million, and that is the solvers' own
non-uniqueness, not an error either side can remove.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference, reference_value

warnings.filterwarnings("ignore")

CASE = "sl_tennessee"
PANEL = "basedata/sl_tennessee_medcost.csv"
LIBRARY = ("lasso", "factor", "did")
TRAIN, T0 = 30, 50
SKIPS = (0, 4, 8, 12)


def _reference_paths() -> Dict[str, np.ndarray]:
    """The three expert paths, from the bundle's captured stdout.

    ``reference.json`` carries the scalars; the per-period paths are four-field
    rows that the bundle parser passes over, so they are read from
    ``reference.out``, which is the same committed artifact.
    """
    out = (Path(__file__).resolve().parents[1] / "reference" / CASE
           / "reference.out").read_text()
    paths: Dict[str, Dict[int, float]] = {}
    for line in out.splitlines():
        f = line.split("\t")
        if f[0] == "path" and len(f) == 4:
            paths.setdefault(f[1], {})[int(f[2])] = float(f[3])
    return {k: np.array([v[t] for t in sorted(v)]) for k, v in paths.items()}


def _panel() -> pd.DataFrame:
    return pd.read_csv(PANEL)


def _config(**kw) -> dict:
    base = dict(df=_panel(), outcome="medcost", treat="expansion",
                unitid="state", time="quarter", experts=list(LIBRARY),
                train_periods=TRAIN, n_boot=2000, block=3, seed=0,
                display_graphs=False)
    base.update(kw)
    return base


def run() -> dict:
    from mlsynth import SL
    from mlsynth.utils.sl_helpers.experts import build_experts
    from mlsynth.utils.sl_helpers.setup import prepare_sl_inputs

    ref = load_reference(CASE)["values"]
    ref_w = load_reference(CASE)["weights"]
    ref_paths = _reference_paths()

    inputs = prepare_sl_inputs(_panel(), unitid="state", time="quarter",
                               outcome="medcost", treat="expansion")
    lib = build_experts(inputs.Yco, inputs.y, slice(0, TRAIN), LIBRARY)
    path_gap = {
        name: float(np.max(np.abs(lib.predictions[:, j] - ref_paths[name])))
        for j, name in enumerate(lib.names)}

    res = SL(_config()).fit()
    f = res.fit
    rel = lambda a, b: float(abs(a - b) / abs(b))

    att_rel = max(
        rel(SL(_config(post_skip=m, n_boot=200)).fit().fit.att,
            ref[f"att_3_skip{m}"]) for m in SKIPS)

    return {
        # the experts, path by path over all 100 quarters
        "lasso_path_max_abs_diff": path_gap["lasso"],
        "factor_path_max_abs_diff": path_gap["factor"],
        "did_path_max_abs_diff": path_gap["did"],
        # the penalties the paper's grids select, which must be the same ones
        "lasso_lambda_abs_diff": abs(lib.details["lasso"]["alpha"]
                                     - ref["lasso_lambda"]),
        "lasso_n_selected_diff": abs(lib.details["lasso"]["n_selected"]
                                     - ref["lasso_n_selected"]),
        "factor_lambda_abs_diff": abs(lib.details["factor"]["alphas"][0]
                                      - ref["factor_lambda"]),
        # the weighting
        "eta_rel_diff": rel(f.eta, ref["eta"]),
        "weights_max_abs_diff": max(abs(f.weights[k] - ref_w[k]) for k in LIBRARY),
        "expert_ssr_max_rel_diff": max(rel(f.expert_ssr[k], ref[f"ssr_{k}"])
                                       for k in LIBRARY),
        # what a reader is shown
        "statistic_rel_diff": rel(f.test_statistic, ref["statistic_3"]),
        "att_max_rel_diff_over_horizons": att_rel,
    }


def comparison() -> List[dict]:
    """Side-by-side rows for the validation dashboard."""
    from mlsynth import SL

    ref = load_reference(CASE)["values"]
    ref_w = load_reference(CASE)["weights"]
    f = SL(_config()).fit().fit
    rows = [{"quantity": "eta", "mlsynth": f.eta, "reference": ref["eta"]},
            {"quantity": "test_statistic", "mlsynth": f.test_statistic,
             "reference": ref["statistic_3"]},
            {"quantity": "ATT", "mlsynth": f.att, "reference": ref["att_3"]}]
    rows += [{"quantity": f"weight[{k}]", "mlsynth": f.weights[k],
              "reference": ref_w[k]} for k in LIBRARY]
    rows += [{"quantity": f"in_window_SSR[{k}]", "mlsynth": f.expert_ssr[k],
              "reference": ref[f"ssr_{k}"]} for k in LIBRARY]
    rows.append({"quantity": "lasso_lambda",
                 "mlsynth": None, "reference": ref["lasso_lambda"]})
    return rows


EXPECTED = {
    # Two experts are closed-form or land on a grid endpoint, so they agree to
    # floating point and are pinned there: a tolerance any looser would let the
    # penalty grid or the standardization regress without failing.
    "lasso_path_max_abs_diff": (2.33e-12, 1e-9),
    "did_path_max_abs_diff": (4.88e-11, 1e-9),
    # The factor expert is the solver-limited one, for the reason in the module
    # docstring. Pinned at the measured value with room for a different BLAS, and
    # tight enough that losing standardization (2.1e-02) or the grid fails it.
    "factor_path_max_abs_diff": (7.03e-07, 5e-06),
    # The penalties themselves must be the same grid point, not merely close.
    # The tolerance is the bundle's own print precision: reference.R emits
    # %.10f, so a value near 0.368 is recorded to about 4e-11 and no comparison
    # through the bundle can be tighter than that.
    "lasso_lambda_abs_diff": (2.86e-11, 1e-9),
    "lasso_n_selected_diff": (0.0, 0.0),
    "factor_lambda_abs_diff": (2.98e-11, 1e-9),
    "eta_rel_diff": (8.24e-13, 1e-10),
    # The ensemble, at the few-parts-per-million floor the factor expert sets.
    "weights_max_abs_diff": (1.79e-06, 1e-05),
    "expert_ssr_max_rel_diff": (6.84e-06, 5e-05),
    "statistic_rel_diff": (7.42e-06, 5e-05),
    "att_max_rel_diff_over_horizons": (3.91e-06, 5e-05),
}
