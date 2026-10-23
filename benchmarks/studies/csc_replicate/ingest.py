"""Can CSC ride ``dataprep``, or does it need its own ingestion?

CSC needs four things from a panel with many treated units sharing one
adoption date: the treated block ``Y1`` (n1 x T), the donor block ``Y0``
(n0 x T), the pre-period length, and one row of time-invariant covariates per
treated unit.

``dataprep`` on such a panel takes its cohort branch and returns
``cohorts[first_treat_period]`` carrying ``y`` (T x n1), ``donor_matrix``
(T x n0), ``treated_units``, ``donor_names`` and ``pre_periods``; passing
``covariates=[...]`` additionally returns ``covariate_matrix`` (N x M) aligned
to ``Ywide.columns``. That is the whole input list, so CSC needs no ingestion
module of its own -- only a slice of ``covariate_matrix`` by the treated units.

Run this file to see the returned keys and a round-trip check against the
simulator that produced the frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsynth.utils.datautils import dataprep


def prepare_csc_inputs(
    long_df: pd.DataFrame,
    *,
    unitid: str = "unit",
    time: str = "time",
    outcome: str = "y",
    treat: str = "treat",
    covariates: list[str],
) -> dict:
    """CSC's matrices, straight out of ``dataprep``."""
    prepped = dataprep(
        long_df,
        unitid,
        time,
        outcome,
        treat,
        covariates=covariates,
        normalize_covariates=False,
    )
    cohorts = prepped["cohorts"]
    if len(cohorts) != 1:
        raise ValueError(
            f"CSC assumes one adoption date; dataprep found {len(cohorts)} cohorts."
        )
    cohort = next(iter(cohorts.values()))

    T0 = int(cohort["pre_periods"])
    Y1 = np.asarray(cohort["y"], dtype=float).T  # (n1, T)
    Y0 = np.asarray(cohort["donor_matrix"], dtype=float).T  # (n0, T)

    units = list(prepped["Ywide"].columns)
    treated_pos = [units.index(u) for u in cohort["treated_units"]]
    X1 = np.asarray(prepped["covariate_matrix"], dtype=float)[treated_pos]

    return {
        "Y1_pre": Y1[:, :T0],
        "Y0_pre": Y0[:, :T0],
        "Y1_post": Y1[:, T0:],
        "Y0_post": Y0[:, T0:],
        "X1": X1,
        "treated_units": cohort["treated_units"],
        "donor_units": cohort["donor_names"],
        "pre_periods": T0,
        "covariate_names": prepped["covariate_names"],
    }


if __name__ == "__main__":
    from dgp import simulate, to_long

    panel = simulate(rng=np.random.default_rng(0))
    long_df = to_long(panel)

    # One-hot the categorical covariate in the long frame; dataprep collapses
    # each column to its per-unit value.
    categories = sorted(long_df["category"].unique())
    names = [f"cat{k}" for k in categories]
    for k, name in zip(categories, names):
        long_df[name] = (long_df["category"] == k).astype(float)

    probe = dataprep(long_df, "unit", "time", "y", "treat")
    print("dataprep on a many-treated panel returns:", sorted(probe))
    cohort = next(iter(probe["cohorts"].values()))
    print("  cohort keys:", sorted(cohort))
    print(
        "  y:", np.asarray(cohort["y"]).shape,
        " donor_matrix:", np.asarray(cohort["donor_matrix"]).shape,
        " pre_periods:", cohort["pre_periods"],
    )

    inputs = prepare_csc_inputs(long_df, covariates=names)
    tr, do = panel.treated_idx, panel.donor_idx
    T0 = panel.Y.shape[1] - 1
    print(
        "round-trip against the simulator:",
        {
            "Y1_pre": np.allclose(inputs["Y1_pre"], panel.Y[tr, :T0]),
            "Y0_pre": np.allclose(inputs["Y0_pre"], panel.Y[do, :T0]),
            "Y1_post": np.allclose(inputs["Y1_post"], panel.Y[tr, T0:]),
            "Y0_post": np.allclose(inputs["Y0_post"], panel.Y[do, T0:]),
            "X1": np.allclose(inputs["X1"], panel.X_dummies[tr]),
            "pre_periods": inputs["pre_periods"] == T0,
        },
    )
