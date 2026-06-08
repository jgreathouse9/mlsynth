"""Reference SpSyDiD estimator for cross-validation.

Drives the authors' own SDID weight machinery (``fit_time_weights`` /
``fit_unit_weights`` / ``join_weights`` from the pinned ``functions_ssdid``
clone) and reproduces the spatial WLS estimation step of
``State_Level_Simulations.ipynb`` (cell 16) in serenini/spatial_SDID: pure
controls + treated unit get SDID weights; indirectly-affected units are added
back with the mean treated-unit weight, and a weighted LSDV regression with
unit/time dummies plus ``interaction`` and ``spillover`` yields the direct ATT
and the spillover coefficient.

The weight functions are the authors' code (imported, not copied); the final
regression is a standard ``statsmodels`` WLS that we author here, mirroring the
notebook line-for-line.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference.clone_spsydid import import_functions_ssdid


def reference_ssdid(panel: pd.DataFrame, WD: float):
    """Return ``(att, aite, tau_s)`` from the authors' reference algorithm.

    Parameters
    ----------
    panel : DataFrame
        Long panel with columns ``ID, month, UR2, treatment (bool),
        after_treatment (bool), interaction, spillover``.
    WD : float
        Mean treated-neighbour exposure on the affected units (the scale that
        turns the spillover coefficient ``tau_s`` into the average indirect
        effect ``aite``).
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover - optional dep
        raise BenchmarkSkipped("statsmodels not installed") from exc

    fns = import_functions_ssdid()
    data = panel.copy()
    data["treatment"] = data["treatment"].astype(bool)
    data["after_treatment"] = data["after_treatment"].astype(bool)

    # Restrict donor pool to pure controls + treated unit (drop affected units).
    affected = data[(data["spillover"] > 0) & (~data["treatment"])]
    affected = data[data["ID"].isin(affected["ID"].unique())]
    data1 = data[~data["ID"].isin(affected["ID"].unique())]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tw = fns.fit_time_weights(data1, "UR2", "month", "ID",
                                  "treatment", "after_treatment")
        uw = fns.fit_unit_weights(data1, "UR2", "month", "ID",
                                  "treatment", "after_treatment")
        did_data = fns.join_weights(data1, uw, tw, "month", "ID",
                                    "treatment", "after_treatment")

    # Add the indirectly-affected units back with the mean treated-unit weight.
    affected = affected.assign(
        unit_weights=did_data[did_data["treatment"] == 1]["unit_weights"].mean())
    twc = (did_data.groupby("month").mean(numeric_only=True)["time_weights"]
           .to_frame().reset_index())
    affected = pd.merge(affected, twc, on="month")
    did2 = pd.concat([did_data, affected]).sort_values(["month", "ID"])
    did2["weights"] = np.round(did2["unit_weights"] * did2["time_weights"], 10)

    # Weighted LSDV: unit + time dummies, plus interaction and spillover.
    n = len(did2["ID"].unique())
    t = 36
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = pd.concat([pd.get_dummies(did2["ID"], prefix="ID"),
                         pd.get_dummies(did2["month"], prefix="month"), did2], axis=1)
        Y = new["UR2"]
        X = pd.concat([new[["interaction", "spillover"]],
                       new.iloc[:, 1:n + t - 1]], axis=1)
        X = sm.add_constant(X).astype(float)
        try:
            model = sm.WLS(Y, X, weights=did2["weights"] + 1e-5).fit()
        except Exception:  # pragma: no cover - matches the notebook's fallback
            model = sm.WLS(Y, X, weights=did2["weights"] + 1e-4).fit()

    att = float(model.params.iloc[1])      # interaction coefficient
    tau_s = float(model.params.iloc[2])    # spillover coefficient
    aite = tau_s * WD
    return att, aite, tau_s
