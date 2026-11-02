"""Rolling-origin predictions with fixed windows and complete fold diagnostics."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from copy import deepcopy
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from ...estimators.vanillasc import VanillaSC
from ...exceptions import MlsynthConfigError
from .numerics import rms
from .results import extract_paths


def normalize_validation_windows(
    windows: Iterable[Iterable[Any]] | None, pre: list[Any], limit: float | None
) -> list[list[Any]]:
    """Freeze disjoint observed windows with at least one earlier training period."""
    if limit is not None and (
        not isinstance(limit, Real)
        or isinstance(limit, bool)
        or not np.isfinite(limit)
        or limit < 0
    ):
        raise MlsynthConfigError("validation_max_rmse must be finite and nonnegative.")
    if windows is None:
        if limit is not None:
            raise MlsynthConfigError("validation_max_rmse requires validation_windows.")
        return []
    try:
        result = [sorted(window) for window in windows]
    except TypeError as exc:
        raise MlsynthConfigError(
            "validation_windows must contain iterable windows of comparable periods."
        ) from exc
    if not result:
        raise MlsynthConfigError("validation_windows must contain at least one window.")
    seen: set[Any] = set()
    for window in result:
        if (
            not window
            or len(set(window)) != len(window)
            or not set(window).issubset(pre)
            or window[0] <= pre[0]
            or seen.intersection(window)
        ):
            raise MlsynthConfigError(
                "validation_windows must be nonempty, unique, disjoint pre-treatment "
                "periods with earlier training observations."
            )
        seen.update(window)
    return sorted(result, key=lambda window: window[0])


def training_config(
    effective: dict[str, Any],
    fit_periods: list[Any],
    train: list[Any],
    cutoff: Any,
    treated: Any,
) -> dict[str, Any]:
    """Intersect all matching windows with training time on a copied fold panel.

    Empty windows remain empty tuples for the runner to record and reject.
    The fold's treatment indicator marks its forecasting boundary; the study's
    actual treatment assignment and final-fit configuration are unchanged.
    """
    cfg = deepcopy(effective)
    fitted = [t for t in fit_periods if t < cutoff]
    cfg["fit_window"] = (fitted[0], fitted[-1]) if fitted else ()
    cfg["fit_exclude"] = [t for t in (cfg.get("fit_exclude") or []) if t < cutoff]
    windows = cfg.get("covariate_windows") or {}
    clipped = {}
    for cov in cfg.get("covariates") or []:
        lo, hi = windows.get(cov, (train[0], train[-1]))
        periods = [t for t in train if lo <= t <= hi]
        clipped[cov] = (periods[0], periods[-1]) if periods else ()
    cfg["covariate_windows"] = clipped
    frame = cfg["df"]
    frame[cfg["treat"]] = (
        (frame[cfg["unitid"]] == treated) & (frame[cfg["time"]] >= cutoff)
    ).astype(int)
    cfg.update(inference=False, display_graphs=False, save=False)
    return cfg


def run_validation(
    registry: dict[str, dict[str, Any]],
    specifications: list[dict[str, Any]],
    windows: list[list[Any]],
    pre: list[Any],
    treated: Any,
    prepare: Callable[..., tuple[dict[str, Any], list[Any], list[Any]]],
) -> dict[str, Any]:
    """Return fold tables and native fits; failures never remove attempted folds."""
    rows, paths, weights, warning_rows = [], [], [], []
    configurations, fits = {}, {}
    for spec in specifications:
        name = spec["name"]
        configurations[name], fits[name] = {}, {}
        for number, window in enumerate(windows, 1):
            fold = f"fold_{number}"
            train = [t for t in pre if t < window[0]]
            periods = train + window
            cfg = deepcopy(registry[name])
            donors = deepcopy(cfg.get("donors"))
            row = dict(
                spec_id=name,
                fold_id=fold,
                rmse=np.nan,
                n_train=len(train),
                n_validation=len(window),
                status="failed",
                exception_type=None,
                message=None,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    effective, fit_periods, donors = prepare(spec, cfg, periods)
                    cfg = training_config(
                        effective, fit_periods, train, window[0], treated
                    )
                    if not cfg["fit_window"]:
                        raise MlsynthConfigError(
                            "The fitting window is empty before validation."
                        )
                    for cov, bounds in cfg["covariate_windows"].items():
                        if not bounds:
                            raise MlsynthConfigError(
                                f"Covariate window for {cov!r} is empty before validation."
                            )
                    model = VanillaSC(cfg).fit()
                    fits[name][fold] = model
                    solver_warnings = model.additional_outputs.get(
                        "solver_warnings", []
                    )
                    warning_rows.extend(
                        dict(spec_id=name, fold_id=fold, **w) for w in solver_warnings
                    )
                    extracted = extract_paths(model, name, periods)
                    predicted = extracted.paths.loc[
                        extracted.paths.time.isin(window)
                    ].copy()
                    predicted["fold_id"] = fold
                    donor_weights = extracted.weights.assign(fold_id=fold)
                    paths.extend(predicted.to_dict("records"))
                    weights.extend(donor_weights.to_dict("records"))
                    row.update(
                        rmse=rms(predicted.gap),
                        status="warning" if caught or solver_warnings else "ok",
                    )
                except Exception as exc:
                    row.update(exception_type=type(exc).__name__, message=str(exc))
                configurations[name][fold] = deepcopy(
                    {k: v for k, v in cfg.items() if k != "df"}
                )
                configurations[name][fold]["donors"] = deepcopy(donors)
                warning_rows.extend(
                    dict(
                        spec_id=name,
                        fold_id=fold,
                        warning_type=w.category.__name__,
                        message=str(w.message),
                    )
                    for w in caught
                )
            rows.append(row)
    return dict(
        validation=pd.DataFrame(
            rows,
            columns=[
                "spec_id",
                "fold_id",
                "rmse",
                "n_train",
                "n_validation",
                "status",
                "exception_type",
                "message",
            ],
        ),
        validation_paths=pd.DataFrame(
            paths,
            columns=["spec_id", "fold_id", "time", "observed", "synthetic", "gap"],
        ),
        validation_weights=pd.DataFrame(
            weights, columns=["spec_id", "fold_id", "donor", "weight"]
        ),
        validation_warnings=pd.DataFrame(
            warning_rows, columns=["spec_id", "fold_id", "warning_type", "message"]
        ),
        validation_registry=configurations,
        validation_fits=fits,
    )


def add_validation_summary(
    summary: pd.DataFrame, validation: dict[str, Any], limit: float | None
) -> pd.DataFrame:
    """Pool all held-out gaps; a failed fold makes the aggregate unavailable."""
    summary = summary.copy()
    scores, statuses = [], []
    for name in summary.spec_id:
        folds = validation["validation"].query("spec_id == @name")
        failed = (folds.status == "failed").any()
        gaps = validation["validation_paths"].query("spec_id == @name").gap
        scores.append(np.nan if failed else rms(gaps))
        statuses.append(
            "failed"
            if failed
            else "warning" if (folds.status == "warning").any() else "ok"
        )
    summary["validation_rmse"] = scores
    summary["validation_status"] = statuses
    if limit is not None:
        summary["validation_pass"] = summary.validation_rmse <= limit
    return summary
