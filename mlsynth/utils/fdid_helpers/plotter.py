"""Plotting wrappers for the Forward Difference-in-Differences estimator.

The single-treated fit draws observed against its two counterfactuals; a
staggered fit draws an event study, since its estimates live on an event clock
and a counterfactual path per treated unit is not one chart.
"""

from __future__ import annotations

import warnings
from typing import List, Union

from ...exceptions import MlsynthDataError, MlsynthPlottingError
from ..resultutils import plot_estimates
from .structures import FDIDResults, FDIDStaggeredResults


def plot_fdid(
    results: FDIDResults,
    *,
    time: str,
    unitid: str,
    outcome: str,
    treat: str,
    treated_color: str,
    counterfactual_color: Union[str, List[str]],
    save: Union[bool, dict],
) -> None:
    """Plot observed vs FDID and DID counterfactuals.

    Plotting failures are downgraded to warnings so a rendering problem
    never masks a successful estimation.
    """
    inputs = results.inputs
    treated_name = inputs.treated_unit_name
    try:
        plot_estimates(
            processed_data_dict=inputs.prepped,
            time_axis_label=time,
            unit_identifier_column_name=unitid,
            outcome_variable_label=outcome,
            treatment_name_label=treat,
            treated_unit_name=treated_name,
            observed_outcome_series=inputs.y,
            counterfactual_series_list=[
                results.fdid.counterfactual,
                results.did.counterfactual,
            ],
            estimation_method_name="FDID",
            counterfactual_names=[
                f"FDID {treated_name}",
                f"DID {treated_name}",
            ],
            treated_series_color=treated_color,
            save_plot_config=save,
            counterfactual_series_colors=counterfactual_color,
        )
    except (MlsynthPlottingError, MlsynthDataError) as e:
        warnings.warn(f"Plotting failed: {str(e)}", UserWarning)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"Unexpected plotting error: {str(e)}", UserWarning)


def plot_fdid_staggered(
    results: "FDIDStaggeredResults",
    *,
    ax: Union[object, None] = None,
    **overrides: object,
) -> object:
    """Event-study chart for a staggered Forward DID fit.

    Effects against event time with the joint-covariance band, a zero line and
    a marker at event time 0, drawn through the shared
    :meth:`mlsynth.utils.plotting.Plotter.event_study` archetype.

    Parameters
    ----------
    results : FDIDStaggeredResults
        A staggered fit.
    ax : matplotlib Axes, optional
        Draw into an existing axis (multi-panel composition).
    **overrides
        Per-call cosmetic overrides applied over the stored ``PlotConfig``
        (e.g. ``title=...``, ``counterfactual_color=...``).

    Returns
    -------
    matplotlib.axes.Axes
        The axis drawn into. Displaying and saving are the caller's; see
        :meth:`FDIDStaggeredResults.plot`, which honours the config's
        ``save`` and ``display``.
    """
    from ...config_models import PlotConfig
    from ..plotting import Plotter, mlsynth_style

    pc = results.plot_config or PlotConfig()
    if overrides:
        pc = pc.model_copy(update=overrides)

    es = results.event_study
    method = (results.method_details.method_name
              if results.method_details else "FDID")

    with mlsynth_style(pc.theme):
        plotter = Plotter.from_config(pc)
        return plotter.event_study(
            es.horizons,
            es.att,
            ci_lower=es.ci_lower,
            ci_upper=es.ci_upper,
            outcome=pc.ylabel or "Treatment effect",
            time=pc.xlabel or "Event time",
            title=pc.title or f"{method}: event study",
            ax=ax,
        )
