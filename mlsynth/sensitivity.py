"""Prespecified research-design sensitivity, initially for classical VanillaSC."""

from __future__ import annotations

import warnings
from copy import deepcopy
from importlib.metadata import version
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from .estimators.vanillasc import VanillaSC
from .exceptions import MlsynthConfigError, MlsynthDataError

_DIMENSIONS = {"donors", "fit_window", "fit_exclude", "covariates", "covariate_windows"}


class SensitivityResult(BaseModel):
    """A collection of unchanged estimator results and tidy sensitivity tables.

    ``summary`` has one row per attempt; ``failures`` and ``warnings`` preserve
    unsuccessful fits and diagnostics. ``fits`` retains the complete native
    results, including predictor weights and solver diagnostics. ``registry``
    contains resolved configurations without repeated panel data; ``baseline``
    preserves the original configuration, including a copy of the panel.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    summary: pd.DataFrame
    paths: pd.DataFrame
    weights: pd.DataFrame
    failures: pd.DataFrame
    warnings: pd.DataFrame
    registry: dict[str, dict[str, Any]]
    baseline: dict[str, Any]
    metadata: dict[str, Any]
    fits: dict[str, Any]

    def dispersion(self) -> pd.Series:
        """Describe finite estimates, including warning-bearing fits.

        Sign shares have the number of finite estimates as their denominator.
        They describe registered designs and are not causal probabilities.
        """
        effects = self.summary.att.dropna()
        return pd.Series(
            {
                "attempted": len(self.summary),
                "estimated": len(effects),
                "failed": (self.summary.status == "failed").sum(),
                "with_warnings": (self.summary.status == "warning").sum(),
                "min": effects.min(),
                "median": effects.median(),
                "max": effects.max(),
                "std": effects.std(),
                "positive_share": (effects > 0).mean() if len(effects) else np.nan,
                "negative_share": (effects < 0).mean() if len(effects) else np.nan,
                "zero_share": (effects == 0).mean() if len(effects) else np.nan,
            }
        )

    def by_dimension(self, dimension: str) -> pd.DataFrame:
        """Group dispersion by one design choice, without attributing causality.

        Other dimensions can vary within each group. An unbalanced registry
        does not identify a variance contribution of the selected dimension.
        Failed designs remain in attempted/failed counts.
        """
        if dimension not in _DIMENSIONS:
            raise MlsynthConfigError(f"dimension must be one of {sorted(_DIMENSIONS)}")
        table = self.summary.copy()
        table["choice"] = [
            repr(self.registry[name].get(dimension)) for name in table.spec_id
        ]
        return (
            table.groupby("choice", sort=False, dropna=False)
            .agg(
                attempted=("spec_id", "size"),
                estimated=("att", "count"),
                failed=("status", lambda x: (x == "failed").sum()),
                min=("att", "min"),
                median=("att", "median"),
                max=("att", "max"),
            )
            .reset_index()
        )

    def plot(self):
        """Return a specification-curve Figure displaying every attempted design."""
        fig, ax = plt.subplots()
        ordered = self.summary.sort_values("att", kind="stable", na_position="last")
        for i, row in enumerate(ordered.itertuples()):
            if row.status == "failed":
                ax.text(
                    i,
                    0.02,
                    "failed",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    rotation=90,
                )
            else:
                ax.scatter(i, row.att, marker="x" if row.status == "warning" else "o")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xticks(range(len(ordered)), ordered.spec_id, rotation=90)
        ax.set_ylabel("ATT")
        ax.set_xlabel("Specification (x: fit with warnings; failed: no estimate)")
        return fig


class SpecificationStudy:
    """Freeze a design registry, then independently refit its specifications.

    Parameters
    ----------
    estimator : class
        Initially only ``VanillaSC`` is supported, without augmentation,
        bias correction, oracle weights or alternative weight constraints.
    base_config : dict
        Existing estimator arguments, including ``df``. ``donors`` optionally
        restricts the baseline pool and is removed before estimator construction.
        ``backend='auto'`` is resolved once from the baseline covariates.
    specifications : iterable of dict
        Unique nonempty ``name`` plus overrides for donors, covariates,
        covariate_windows, fit_window or fit_exclude. An unchanged baseline is
        inserted if absent. Invalid designs are retained as failed attempts.
    att_periods : iterable
        Fixed, unique observed post-treatment periods over which gaps are averaged.
    excluded_donors : iterable
        Known contaminated donors. A pool containing one fails. Units marked
        treated in the original panel cannot enter donor pools.

    Notes
    -----
    Exactly one unit must be treated, with a binary treatment indicator that
    switches permanently from zero to one. The runner infers the fixed treatment
    time before any donor subsetting. Declare designs before viewing effects.
    Precomputed covariate columns must themselves use only information available
    before treatment. Computational settings cannot vary in this initial API.
    """

    def __init__(
        self, estimator, base_config, specifications, att_periods, excluded_donors=()
    ):
        if estimator is not VanillaSC:
            raise MlsynthConfigError("Only VanillaSC is supported.")
        self._baseline = deepcopy(base_config)
        base = deepcopy(base_config)
        self._donors = base.pop("donors", None)
        self._config = estimator(base).config
        cfg = self._config
        if (
            cfg.augment is not None
            or cfg.bias_correct
            or cfg.oracle_weights is not None
            or cfg.w_constr is not None
            or cfg.backend == "penalized"
        ):
            raise MlsynthConfigError(
                "Sensitivity initially supports classical VanillaSC only."
            )
        frame = cfg.df
        if frame[[cfg.unitid, cfg.time, cfg.treat]].isna().any().any():
            raise MlsynthDataError(
                "Unit, time and treatment columns must not be missing."
            )
        if frame.duplicated([cfg.unitid, cfg.time]).any():
            raise MlsynthDataError("Unit/time pairs must be unique.")
        if not frame[cfg.treat].isin([0, 1]).all():
            raise MlsynthDataError("Treatment indicator must be binary.")
        treated = frame.loc[frame[cfg.treat] == 1, cfg.unitid].unique()
        if len(treated) != 1:
            raise MlsynthDataError("Exactly one treated unit is required.")
        self._treated = treated[0]
        treated_frame = frame.loc[frame[cfg.unitid] == self._treated].sort_values(
            cfg.time
        )
        self._treatment_time = treated_frame.loc[
            treated_frame[cfg.treat] == 1, cfg.time
        ].min()
        if not (
            treated_frame[cfg.treat].to_numpy()
            == (treated_frame[cfg.time] >= self._treatment_time).to_numpy()
        ).all():
            raise MlsynthDataError(
                "Treatment must be permanent after its first occurrence."
            )
        self._times = sorted(frame[cfg.time].unique())
        self._pre = [t for t in self._times if t < self._treatment_time]
        self._post = list(att_periods)
        if not self._pre or not self._post or len(set(self._post)) != len(self._post):
            raise MlsynthConfigError(
                "Nonempty pre-periods and unique att_periods are required."
            )
        if not set(self._post).issubset(self._times) or any(
            t < self._treatment_time for t in self._post
        ):
            raise MlsynthConfigError(
                "att_periods must be observed post-treatment periods."
            )
        if cfg.backend == "auto":
            self._config = cfg.model_copy(
                update={"backend": "mscmt" if cfg.covariates else "outcome-only"}
            )
        self._excluded = tuple(excluded_donors)
        self._specs = deepcopy(list(specifications))
        names = [s.get("name") for s in self._specs]
        if any(not isinstance(n, str) or not n for n in names) or len(
            set(names)
        ) != len(names):
            raise MlsynthConfigError(
                "Specification names must be unique nonempty strings."
            )
        if "baseline" not in names:
            self._specs.insert(0, {"name": "baseline"})

    def _prepare(self, design):
        cfg = self._config
        frame = cfg.df
        all_donors = set(frame[cfg.unitid].unique()) - {self._treated}
        donors = list(all_donors if design["donors"] is None else design["donors"])
        if not donors or len(set(donors)) != len(donors):
            raise MlsynthConfigError("Donors must be nonempty and unique.")
        if not set(donors).issubset(all_donors) or set(donors).intersection(
            self._excluded
        ):
            raise MlsynthConfigError(
                "Donor pool contains absent, treated or declared contaminated units."
            )
        selected = frame.loc[frame[cfg.unitid].isin([self._treated, *donors])].copy(
            deep=True
        )
        if (
            len(selected) != len(self._times) * (len(donors) + 1)
            or not np.isfinite(selected[cfg.outcome]).all()
        ):
            raise MlsynthDataError(
                "Every selected unit needs finite outcomes at all common trajectory periods."
            )
        fit_periods = list(self._pre)
        window = design.get("fit_window")
        if window is not None:
            if (
                len(window) != 2
                or window[0] > window[1]
                or window[1] >= self._treatment_time
            ):
                raise MlsynthConfigError(
                    "fit_window must be ordered and strictly pre-treatment."
                )
            fit_periods = [t for t in fit_periods if window[0] <= t <= window[1]]
        excluded = design.get("fit_exclude") or []
        if not set(excluded).issubset(self._pre):
            raise MlsynthConfigError(
                "fit_exclude must contain observed pre-treatment periods."
            )
        fit_periods = [t for t in fit_periods if t not in excluded]
        if not fit_periods:
            raise MlsynthConfigError("The fitting window is empty.")
        covariates = design.get("covariates") or []
        if covariates and design["backend"] == "outcome-only":
            raise MlsynthConfigError(
                "The fixed outcome-only backend cannot fit covariates."
            )
        for cov, window in (design.get("covariate_windows") or {}).items():
            if (
                cov not in covariates
                or len(window) != 2
                or window[0] > window[1]
                or window[1] >= self._treatment_time
                or not any(window[0] <= t <= window[1] for t in self._pre)
            ):
                raise MlsynthConfigError(
                    "covariate_windows must name fitting covariates and select pre-treatment periods."
                )
        effective = {k: deepcopy(v) for k, v in design.items() if k != "donors"}
        effective["df"] = selected.sort_values([cfg.unitid, cfg.time])
        return effective, fit_periods, donors

    def fit(self) -> SensitivityResult:
        """Fit all registered designs without filtering on fit quality or effect."""
        base = self._config.model_dump(exclude={"df"})
        base["donors"] = deepcopy(self._donors)
        registry = {
            s["name"]: {
                **deepcopy(base),
                **{k: deepcopy(v) for k, v in s.items() if k != "name"},
            }
            for s in self._specs
        }
        summaries, paths, weights, failures, warning_rows = [], [], [], [], []
        fits = {}
        for spec in self._specs:
            name = spec["name"]
            row = dict(
                spec_id=name,
                att=np.nan,
                pre_rmse=np.nan,
                fit_rmse=np.nan,
                n_donors=np.nan,
                status="failed",
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    if set(spec) - _DIMENSIONS - {"name"}:
                        raise MlsynthConfigError(
                            "Only design dimensions may vary; study and computational settings are fixed."
                        )
                    if name == "baseline" and len(spec) != 1:
                        raise MlsynthConfigError(
                            "The baseline cannot contain overrides."
                        )
                    effective, fit_periods, donors = self._prepare(registry[name])
                    row["n_donors"] = len(donors)
                    fitted = VanillaSC(effective).fit()
                    solver_warnings = fitted.additional_outputs.get(
                        "solver_warnings", []
                    )
                    warning_rows.extend(
                        dict(spec_id=name, **warning) for warning in solver_warnings
                    )
                    series = fitted.time_series
                    times = list(series.time_periods)
                    if times != self._times:
                        raise MlsynthDataError(
                            "Estimator changed the common trajectory periods."
                        )
                    observed = np.asarray(series.observed_outcome, dtype=float).reshape(
                        -1
                    )
                    synthetic = np.asarray(
                        series.counterfactual_outcome, dtype=float
                    ).reshape(-1)
                    donor_weights = fitted.weights.donor_weights
                    if (
                        observed.shape != (len(times),)
                        or synthetic.shape != observed.shape
                        or not np.isfinite(
                            np.r_[observed, synthetic, list(donor_weights.values())]
                        ).all()
                    ):
                        raise MlsynthDataError(
                            "Estimator returned missing or nonfinite paths/weights."
                        )
                    gap = pd.Series(observed - synthetic, index=times)
                    row.update(
                        att=float(gap.loc[self._post].mean()),
                        pre_rmse=float(np.sqrt(np.mean(gap.loc[self._pre] ** 2))),
                        fit_rmse=float(np.sqrt(np.mean(gap.loc[fit_periods] ** 2))),
                        status="warning" if caught or solver_warnings else "ok",
                    )
                    fits[name] = fitted
                    paths.extend(
                        dict(spec_id=name, time=t, observed=y, synthetic=c, gap=y - c)
                        for t, y, c in zip(times, observed, synthetic)
                    )
                    weights.extend(
                        dict(spec_id=name, donor=d, weight=w)
                        for d, w in donor_weights.items()
                    )
                except Exception as exc:
                    failures.append(
                        dict(
                            spec_id=name,
                            exception_type=type(exc).__name__,
                            message=str(exc),
                        )
                    )
                warning_rows.extend(
                    dict(
                        spec_id=name,
                        warning_type=w.category.__name__,
                        message=str(w.message),
                    )
                    for w in caught
                )
            summaries.append(row)
        return SensitivityResult(
            summary=pd.DataFrame(summaries),
            paths=pd.DataFrame(
                paths, columns=["spec_id", "time", "observed", "synthetic", "gap"]
            ),
            weights=pd.DataFrame(weights, columns=["spec_id", "donor", "weight"]),
            failures=pd.DataFrame(
                failures, columns=["spec_id", "exception_type", "message"]
            ),
            warnings=pd.DataFrame(
                warning_rows, columns=["spec_id", "warning_type", "message"]
            ),
            registry=registry,
            baseline=deepcopy(self._baseline),
            fits=fits,
            metadata=dict(
                treated_unit=self._treated,
                treatment_time=self._treatment_time,
                att_periods=list(self._post),
                pre_periods=list(self._pre),
                excluded_donors=self._excluded,
                backend=self._config.backend,
                estimator="mlsynth.VanillaSC",
                version=version("mlsynth"),
            ),
        )
