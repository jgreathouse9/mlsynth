"""Frozen dataclasses for the Forward Difference-in-Differences estimator.

FDID (Li 2023, *Frontiers: A Simple Forward Difference-in-Differences
Method*, Marketing Science) builds the control group for a single treated
unit by **forward selection**: it greedily adds the donor that most
improves pre-treatment fit (R^2 between the treated unit and the running
donor average), tracks the R^2 path, and keeps the subset that maximises
it. The synthetic control is the simple average of the selected donors,
with a difference-in-differences intercept.

Two estimates are always returned side by side:

* **FDID** -- the forward-selected difference-in-differences (best donor
  subset).
* **DID** -- the textbook two-way difference-in-differences using *all*
  donors (the average of every control unit). This is the natural
  benchmark the forward search improves upon.

Both carry Li (2023) analytical standard errors, or the
serial-correlation-robust alternative when the config asks for it. The three
layers below (inputs, per-method fit, top-level results) mirror the
CLUSTERSC / PROXIMAL container design used elsewhere in ``mlsynth``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import ConfigDict, Field as PydField, model_validator

from ...config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
from ..results_helpers import build_effect_submodels


# Public method names.
FDID = "FDID"
DID = "DID"


def _inference_label(method: str, lag: Optional[int]) -> str:
    """Human-readable name of the reported standard error."""
    if method == "hac":
        return f"HAC (pre-period autocovariances, lag {lag})"
    return "analytic (Li 2023)"


@dataclass(frozen=True)
class FDIDInputs:
    """Preprocessed panel data for the FDID pipeline.

    Parameters
    ----------
    y : np.ndarray
        Treated-unit outcome over all ``T`` periods, shape ``(T,)``.
    donor_matrix : np.ndarray
        Donor outcomes, shape ``(T, n_donors)``.
    pre_periods : int
        Number of pre-treatment periods ``T0``.
    post_periods : int
        Number of post-treatment periods ``T1 = T - T0``.
    T : int
        Total number of periods.
    donor_names : Sequence
        Length-``n_donors`` donor labels (column order of ``donor_matrix``).
    time_labels : np.ndarray
        Length-``T`` time labels.
    treated_unit_name : Any
        Identifier of the treated unit.
    verbose : bool
        Whether the forward-selection path is recorded step by step.
    prepped : dict
        The raw :func:`mlsynth.utils.datautils.dataprep` dictionary, kept
        so the plotter can reuse the prepared panel.
    """

    y: np.ndarray
    donor_matrix: np.ndarray
    pre_periods: int
    post_periods: int
    T: int
    donor_names: Sequence
    time_labels: np.ndarray
    treated_unit_name: Any
    verbose: bool = True
    prepped: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_donors(self) -> int:
        """Number of donor units."""
        return int(self.donor_matrix.shape[1])


@dataclass(frozen=True)
class FDIDMethodFit:
    """Single FDID/DID fit output.

    Parameters
    ----------
    name : str
        Method identifier (``"FDID"`` or ``"DID"``).
    counterfactual : np.ndarray
        Estimated counterfactual outcome path, shape ``(T,)``.
    gap : np.ndarray
        Observed treated minus counterfactual, shape ``(T,)``.
    att : float
        Mean post-treatment treatment effect.
    att_se : float
        Li (2023) analytical standard error of the ATT.
    att_percent : float
        ATT as a percentage of the post-period counterfactual mean.
    satt : float
        Standardised ATT (``att / se * sqrt(T1)``).
    pre_rmse : float
        Root-mean-squared pre-treatment fit error.
    r_squared : float
        Pre-treatment R^2 of the difference-in-differences fit.
    intercept : float
        Difference-in-differences intercept (treated minus donor
        pre-period mean).
    p_value : float
        Two-sided p-value for the ATT.
    ci : tuple of float
        ``(lower, upper)`` 95% confidence interval for the ATT.
    selected_indices : list of int
        Column indices of the donors retained (all donors for DID).
    selected_names : list
        Donor labels corresponding to ``selected_indices``.
    donor_weights : dict
        Mapping ``{donor_name: weight}`` (equal weights over the selected
        donors).
    r2_path : np.ndarray or None
        R^2 after each forward-selection step (FDID only; ``None`` for DID).
    intermediary : list or None
        Per-step diagnostics when ``verbose`` (FDID only).
    inference_method : str
        Which standard error ``att_se`` is -- ``"analytic"`` (Li 2023,
        Proposition 2.1) or ``"hac"``.
    lrvar_lag : int or None
        Truncation lag used by the HAC standard error; ``None`` under the
        analytic formula.
    metadata : dict
        Free-form per-method diagnostics.
    """

    name: str
    counterfactual: np.ndarray
    gap: np.ndarray
    att: float
    att_se: float
    att_percent: float
    satt: float
    pre_rmse: float
    r_squared: float
    intercept: float
    p_value: float
    ci: Tuple[float, float]
    selected_indices: List[int]
    selected_names: List[Any]
    donor_weights: Dict[Any, float]
    r2_path: Optional[np.ndarray] = None
    intermediary: Optional[list] = None
    inference_method: str = "analytic"
    lrvar_lag: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FDIDResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.FDID.fit`.

    An :class:`~mlsynth.config_models.EffectResult` (the observational
    report): in addition to the FDID-specific fields below, it exposes the
    standardized sub-models (``effects``, ``time_series``, ``weights``,
    ``inference``, ``fit_diagnostics``, ``method_details``) -- derived from
    the selected variant -- and the flat accessors ``att``/``att_ci``/
    ``counterfactual``/``gap``/``donor_weights``/``pre_rmse``.

    Parameters
    ----------
    inputs : FDIDInputs
        Preprocessed panel.
    fdid : FDIDMethodFit
        Forward-selected difference-in-differences fit (primary).
    did : FDIDMethodFit
        Textbook difference-in-differences using all donors.
    selected_variant : str
        Which fit is exposed via the convenience aliases ``att``,
        ``att_se``, ``counterfactual``, ``gap``, ``donor_weights`` --
        ``"FDID"`` or ``"DID"``. Defaults to ``"FDID"``.
    metadata : dict
        Free-form pipeline diagnostics.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: FDIDInputs
    fdid: FDIDMethodFit
    did: FDIDMethodFit
    selected_variant: str = FDID
    metadata: Dict[str, Any] = PydField(default_factory=dict)

    @model_validator(mode="after")
    def _populate_standard_submodels(self) -> "FDIDResults":
        """Derive the standardized EffectResult sub-models from the primary
        variant, so every FDID result exposes the common surface. Uses
        ``object.__setattr__`` because the model is frozen.
        """
        if self.effects is None:
            p = self._primary
            derived = build_effect_submodels(
                observed_outcome=np.asarray(self.inputs.y),
                counterfactual_outcome=np.asarray(p.counterfactual),
                n_pre_periods=int(self.inputs.pre_periods),
                n_post_periods=int(self.inputs.post_periods),
                time_periods=np.asarray(self.inputs.time_labels),
                weights=WeightsResults(
                    donor_weights={
                        str(k): float(v) for k, v in p.donor_weights.items()
                    }
                ),
                inference=InferenceResults(
                    p_value=float(p.p_value),
                    ci_lower=float(p.ci[0]),
                    ci_upper=float(p.ci[1]),
                    standard_error=float(p.att_se),
                    method=_inference_label(p.inference_method, p.lrvar_lag),
                ),
                method_name=p.name,
                is_recommended=True,
                att_std_err=float(p.att_se),
                # FDID computes its ATT / %ATT / pre-RMSE / R^2 analytically
                # (Li 2023); pin those authoritative values so the validated
                # numbers are unchanged while the helper unifies the mapping.
                effects_overrides={
                    "att": float(p.att),
                    "att_percent": float(p.att_percent),
                },
                fit_overrides={
                    "rmse_pre": float(p.pre_rmse),
                    "r_squared_pre": float(p.r_squared),
                },
            )
            for key, value in derived.items():
                object.__setattr__(self, key, value)
        return self

    @property
    def methods(self) -> Dict[str, FDIDMethodFit]:
        """``{method_name: fit}`` for both fits, FDID first."""
        return {FDID: self.fdid, DID: self.did}

    @property
    def _primary(self) -> FDIDMethodFit:
        return self.methods.get(self.selected_variant, self.fdid)

    @property
    def att(self) -> float:
        """ATT of the primary variant."""
        return self._primary.att

    @property
    def att_se(self) -> float:
        """ATT standard error of the primary variant."""
        return self._primary.att_se

    @property
    def counterfactual(self) -> np.ndarray:
        """Counterfactual of the primary variant."""
        return self._primary.counterfactual

    @property
    def gap(self) -> np.ndarray:
        """Gap of the primary variant."""
        return self._primary.gap

    @property
    def donor_weights(self) -> Dict[Any, float]:
        """Donor weights of the primary variant."""
        return self._primary.donor_weights

    @property
    def pre_rmse(self) -> float:
        """Pre-treatment RMSE of the primary variant."""
        return self._primary.pre_rmse

    def att_by_method(self) -> Dict[str, float]:
        """``{method: ATT}`` for both fits."""
        return {name: fit.att for name, fit in self.methods.items()}

    def se_by_method(self) -> Dict[str, float]:
        """``{method: ATT standard error}`` for both fits."""
        return {name: fit.att_se for name, fit in self.methods.items()}

    def ci_by_method(self) -> Dict[str, Tuple[float, float]]:
        """``{method: (lower, upper)}`` confidence intervals for both fits."""
        return {name: fit.ci for name, fit in self.methods.items()}


# ─── staggered adoption ───────────────────────────────────────────────────
#
# Li (2023) Web Appendix C extends Forward DID to several treated units by
# running the method per unit and averaging the results. The containers below
# carry that extension plus what it needs to be usable: an event clock, and a
# covariance that prices the dependence between treated units drawing on one
# donor pool.

@dataclass(frozen=True)
class FDIDStaggeredInputs:
    """Preprocessed panel for the staggered Forward DID pipeline.

    Parameters
    ----------
    treated_matrix : np.ndarray
        Treated-unit outcomes, shape ``(T, n_treated)``.
    treated_names : list
        Length-``n_treated`` treated-unit labels (column order).
    adoption_index : np.ndarray
        Integer position of each treated unit's first treated period.
    donor_matrix : np.ndarray
        Never-treated donor outcomes, shape ``(T, n_donors)``. One pool serves
        every cohort: units treated at any point are excluded, which is what
        keeps an adopting donor out of a treated unit's criterion as well as
        out of its counterfactual.
    donor_names : list
        Length-``n_donors`` donor labels (column order).
    time_labels : np.ndarray
        Length-``T`` time labels.
    T : int
        Total number of periods.
    verbose : bool
        Whether per-unit selection paths are recorded.
    """

    treated_matrix: np.ndarray
    treated_names: List[Any]
    adoption_index: np.ndarray
    donor_matrix: np.ndarray
    donor_names: List[Any]
    time_labels: np.ndarray
    T: int
    verbose: bool = True

    @property
    def n_treated(self) -> int:
        """Number of treated units."""
        return int(self.treated_matrix.shape[1])

    @property
    def n_donors(self) -> int:
        """Number of never-treated donor units."""
        return int(self.donor_matrix.shape[1])


@dataclass(frozen=True)
class FDIDUnitFit:
    """One treated unit's Forward DID fit within a staggered panel.

    Parameters
    ----------
    unit_name : Any
        Treated-unit label.
    adoption_index : int
        Integer position of the unit's first treated period.
    adoption_time : Any
        Time label of that period.
    selected_indices, selected_names : list
        Donors retained by forward selection, as column indices and labels.
    donor_weights : dict
        ``{donor_name: weight}``, equal over the selected donors.
    intercept : float
        Difference-in-differences intercept, the pre-window mean of the gap.
    observed, counterfactual, gap : np.ndarray
        Length-``T`` series, with ``gap = observed - counterfactual`` as in the
        single-treated fit. The gap is the parallel-trends residual: its
        post-treatment values are the ``ATT(g, h)`` estimates and its
        pre-window supplies the cross-unit covariance.
    pre_periods : int
        Length of the pre-window actually used, after any anticipation trim.
    att : float
        Mean effect over the reported horizons.
    att_by_horizon : dict
        ``{horizon: ATT(g, g + h)}``.
    pre_rmse, r_squared : float
        Pre-window fit of the selected donor average.
    selection_path : np.ndarray or None
        Criterion value after each forward-selection step.
    """

    unit_name: Any
    adoption_index: int
    adoption_time: Any
    selected_indices: List[int]
    selected_names: List[Any]
    donor_weights: Dict[Any, float]
    intercept: float
    observed: np.ndarray
    counterfactual: np.ndarray
    gap: np.ndarray
    pre_periods: int
    att: float
    att_by_horizon: Dict[int, float]
    pre_rmse: float
    r_squared: float
    selection_path: Optional[np.ndarray] = None


@dataclass(frozen=True)
class FDIDEventStudy:
    """Balanced event-time aggregation of the per-unit ``ATT(g, h)``.

    Parameters
    ----------
    horizons : np.ndarray
        Event times ``h = 0, ..., H``.
    att : np.ndarray
        Cohort-size-weighted average of ``ATT(g, g + h)`` at each horizon
        (Callaway and Sant'Anna 2021, equation 31).
    se, ci_lower, ci_upper : np.ndarray
        Standard error and interval at each horizon, from the joint covariance
        across treated units.
    n_units : np.ndarray
        How many treated units contribute at each horizon.
    """

    horizons: np.ndarray
    att: np.ndarray
    se: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    n_units: np.ndarray


class FDIDStaggeredResults(BaseEstimatorResults):
    """Container returned by :meth:`mlsynth.FDID.fit` on a staggered panel.

    An :class:`~mlsynth.config_models.EffectResult`. Forward DID over several
    treated units is an event-study estimator, so the standardized
    ``time_series`` is laid out over event-time horizons, not calendar time,
    as in ``SequentialSDID``: ``time_periods`` are the horizons and
    ``gap`` is the aggregated ``ATT(h)``. Per-unit detail stays in ``units``.

    This surface is experimental, and its intervals are anti-conservative from
    two directions. Forward selection minimises the pre-window residual
    variance and the variance is then estimated on that same window, so the
    estimate inherits a winner's curse; ``selection`` is the lever that
    mitigates it. And a persistent residual is only partly recoverable from a
    short pre-window, so ``overall_se`` still undercovers when the
    parallel-trends residual is strongly autocorrelated, even though this path
    prices autocovariances by default. Read ``event_study`` in preference to
    ``overall_att`` where the distinction matters: the per-horizon intervals
    average nothing and hold up far better.

    Parameters
    ----------
    inputs : FDIDStaggeredInputs
        Preprocessed panel.
    units : list of FDIDUnitFit
        One fit per treated unit, in column order.
    event_study : FDIDEventStudy
        Balanced event-time aggregation.
    overall_att, overall_se : float
        Average of the event-time path and its standard error.
    overall_ci : tuple of float
        ``(lower, upper)`` interval for ``overall_att``.
    selection : str
        How donors were chosen -- ``"unit"``, ``"pooled"`` or ``"partial"``.
    metadata : dict
        Free-form pipeline diagnostics.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: FDIDStaggeredInputs
    units: List[FDIDUnitFit]
    event_study: FDIDEventStudy
    overall_att: float
    overall_se: float
    overall_ci: Tuple[float, float]
    selection: str = "unit"
    metadata: Dict[str, Any] = PydField(default_factory=dict)

    @model_validator(mode="after")
    def _populate_standard_submodels(self) -> "FDIDStaggeredResults":
        """Derive the standardized sub-models over event time.

        The series layout follows ``SequentialSDID``: ``time_periods`` are the
        horizons, ``gap`` is the aggregated effect at each, and the
        counterfactual is the no-effect baseline. Uses ``object.__setattr__``
        because the model is frozen.
        """
        if self.effects is not None:
            return self
        es = self.event_study
        att = np.asarray(es.att, dtype=float)
        derived = {
            "effects": EffectsResults(att=float(self.overall_att)),
            "time_series": TimeSeriesResults(
                observed_outcome=att,
                counterfactual_outcome=np.zeros_like(att),
                estimated_gap=att,
                time_periods=np.asarray(es.horizons),
                intervention_time=0,
            ),
            "weights": WeightsResults(
                weights_at=["units"],
                summary_stats={
                    "constraint": "equal weights over the donors each treated "
                                  "unit selected"
                },
            ),
            "fit_diagnostics": FitDiagnosticsResults(
                rmse_pre=float(np.mean([u.pre_rmse for u in self.units]))
                if self.units else None,
            ),
            "inference": InferenceResults(
                standard_error=float(self.overall_se),
                ci_lower=float(self.overall_ci[0]),
                ci_upper=float(self.overall_ci[1]),
                method="joint covariance across treated units",
            ),
            "method_details": MethodDetailsResults(
                method_name=f"FDID (staggered, selection={self.selection})",
                is_recommended=True,
            ),
        }
        for key, value in derived.items():
            object.__setattr__(self, key, value)
        return self

    def plot(self, kind: str = "auto", *, ax: Any = None, **overrides: Any) -> Any:
        """Render the event study.

        A staggered fit has one estimate per event time and no single
        counterfactual path, so the base class's ``"counterfactual"`` and
        ``"gap"`` kinds do not apply and ``kind`` selects nothing: the chart is
        always effects against event time with the joint-covariance band.

        Parameters
        ----------
        kind : str, default "auto"
            Accepted for signature compatibility with
            :meth:`mlsynth.config_models.EffectResult.plot`.
        ax : matplotlib Axes, optional
            Draw into an existing axis.
        **overrides
            Per-call cosmetic overrides applied over the stored ``PlotConfig``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from ...config_models import PlotConfig
        from .plotter import plot_fdid_staggered

        ax = plot_fdid_staggered(self, ax=ax, **overrides)

        pc = self.plot_config or PlotConfig()
        if overrides:
            pc = pc.model_copy(update=overrides)
        if pc.save:
            fname = pc.save if isinstance(pc.save, str) else "fdid_event_study.png"
            ax.figure.savefig(fname, bbox_inches="tight")
        if pc.display:
            import matplotlib.pyplot as plt

            plt.show()
        return ax

    @property
    def att(self) -> float:
        """Average effect over the reported horizons."""
        return self.overall_att

    @property
    def att_se(self) -> float:
        """Standard error of :attr:`att`."""
        return self.overall_se

    def att_by_unit(self) -> Dict[Any, float]:
        """``{treated unit: mean effect over the reported horizons}``."""
        return {u.unit_name: u.att for u in self.units}

    def donors_by_unit(self) -> Dict[Any, List[Any]]:
        """``{treated unit: selected donor labels}``."""
        return {u.unit_name: list(u.selected_names) for u in self.units}


FDIDStaggeredResults.model_rebuild()
