"""Long-panel ingestion for BEAST (via :func:`mlsynth.utils.datautils.dataprep`)."""
from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import dataprep
from .structures import BEASTInputs


def prepare_beast_inputs(config) -> BEASTInputs:
    """Build :class:`BEASTInputs` from the config's long DataFrame.

    ``X`` is ``[constant, pre-treatment covariate means, lagged outcomes]`` per
    unit (covariates raw, as in the reference: the calibration's penalty loadings
    absorb scale). ``d`` is the single-treated indicator aligned to the outcome
    matrix's unit order.
    """
    prep = dataprep(
        config.df, config.unitid, config.time, config.outcome, config.treat,
        covariates=list(config.covariates), covariate_aggregation="pre_mean",
        normalize_covariates=False,
    )
    Ywide = prep["Ywide"]                       # (T, N), columns = unit names
    unit_names = list(Ywide.columns)
    treated_name = prep["treated_unit_name"]
    if treated_name not in unit_names:  # pragma: no cover - dataprep guarantees it
        raise MlsynthDataError("BEAST: treated unit not found in the panel.")
    treated_index = unit_names.index(treated_name)
    time_labels = np.asarray(Ywide.index)
    pre = int(prep["pre_periods"])
    Y = Ywide.to_numpy(dtype=float)             # (T, N)

    cov = np.asarray(prep["covariate_matrix"], dtype=float)   # (N, M) in unit order
    cov_names = list(prep["covariate_names"])
    if cov.shape[0] != len(unit_names):  # pragma: no cover - alignment guard
        raise MlsynthDataError("BEAST: covariate matrix is misaligned with the panel.")

    # Lagged outcomes: values of the outcome at the requested pre-period labels.
    lag_blocks, lag_names = [], []
    if config.outcome_lags:
        labels = list(time_labels)
        for lag in config.outcome_lags:
            if lag not in labels:
                raise MlsynthConfigError(
                    f"BEAST: outcome_lag {lag!r} is not a time label in the panel.")
            idx = labels.index(lag)
            if idx >= pre:
                raise MlsynthConfigError(
                    f"BEAST: outcome_lag {lag!r} is not in the pre-treatment period.")
            lag_blocks.append(Y[idx, :])
            lag_names.append(f"{config.outcome}_{lag}")
    lags = np.column_stack(lag_blocks) if lag_blocks else np.empty((len(unit_names), 0))

    X = np.column_stack([np.ones(len(unit_names)), cov, lags])
    if not np.all(np.isfinite(X)):
        raise MlsynthDataError("BEAST: covariate/lag design contains non-finite values.")
    d = np.array([1.0 if u == treated_name else 0.0 for u in unit_names])

    return BEASTInputs(
        X=X, d=d, Y=Y, y_treated=Y[:, treated_index],
        feature_names=tuple(cov_names + lag_names),
        pre=pre, time_labels=time_labels, treated_name=treated_name,
        unit_names=tuple(unit_names), treated_index=treated_index,
    )
