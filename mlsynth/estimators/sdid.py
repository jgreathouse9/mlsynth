"""Synthetic Difference-in-Differences (SDID) estimator with event-study output.

Implements:

    Arkhangelsky, D., Athey, S., Hirshberg, D., Imbens, G., & Wager, S. (2021).
    "Synthetic Difference-in-Differences." American Economic Review.

    Ciccia, D. (2024). "A Short Note on Event-Study Synthetic
    Difference-in-Differences Estimators." arXiv:2407.09565.

    Clarke, D., Pailanir, D., Athey, S., & Imbens, G. (2023). "Synthetic
    difference in differences estimation." arXiv preprint.

The estimator handles both the canonical single-treated-unit setup (e.g.
Proposition 99) and staggered-adoption designs with multiple cohorts.
Output is a typed :class:`mlsynth.utils.sdid_helpers.structures.SDIDResults`
object that exposes:

    * ``inference.att`` / ``inference.se`` / ``inference.ci`` / ``inference.p_value``
        the overall ATT and its placebo-based inference (Ciccia 2024 Eq. 7);
    * ``event_study.tau`` / ``event_study.se`` / ``event_study.ci`` / ``event_study.event_times``
        the pooled event-study estimator (Ciccia 2024 Eq. 6);
    * ``cohorts[a]`` for each adoption period ``a``: the cohort ATT
        ``tau_a^sdid`` (Eq. 2), the cohort-specific event-time effects
        ``tau_{a, ell}^sdid`` (Eq. 3), and the cohort's actual vs.
        bias-corrected synthetic control trajectories.
"""

from __future__ import annotations

from typing import Any, List, Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import SDIDConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.sdid_helpers.orchestration import run_sdid
from ..utils.sdid_helpers.plotter import plot_sdid
from ..utils.sdid_helpers.structures import SDIDResults


class SDID:
    """Synthetic Difference-in-Differences estimator with event-study output.

    Parameters
    ----------
    config : SDIDConfig or dict
        Configuration object. See :class:`mlsynth.config_models.SDIDConfig`.

    Returns
    -------
    SDIDResults
        Typed container with the overall ATT and placebo inference
        (:attr:`SDIDResults.inference`), the pooled event-study estimator
        (:attr:`SDIDResults.event_study`), and the per-cohort decomposition
        (:attr:`SDIDResults.cohorts`).

    Notes
    -----
    The estimator accepts either a single treatment date (the canonical SDID
    setup) or a staggered-adoption panel. ``dataprep`` distinguishes the two
    cases automatically.

    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D., Imbens, G., & Wager, S. (2021).
    "Synthetic Difference-in-Differences." *American Economic Review.*

    Ciccia, D. (2024). "A Short Note on Event-Study Synthetic
    Difference-in-Differences Estimators." arXiv:2407.09565.

    Examples
    --------
    >>> import pandas as pd                                                 # doctest: +SKIP
    >>> from mlsynth import SDID                                            # doctest: +SKIP
    >>> df = pd.read_csv(                                                   # doctest: +SKIP
    ...     "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
    ...     "refs/heads/main/basedata/smoking_data.csv"
    ... )
    >>> df["Proposition 99"] = df["Proposition 99"].astype(int)             # doctest: +SKIP
    >>> res = SDID({                                                        # doctest: +SKIP
    ...     "df": df, "outcome": "cigsale", "treat": "Proposition 99",
    ...     "unitid": "state", "time": "year", "B": 200,
    ...     "display_graphs": False,
    ... }).fit()
    >>> res.inference.att                                                   # doctest: +SKIP
    -14.485...
    """

    def __init__(self, config: Union[SDIDConfig, dict]) -> None:
        """Initialize SDID from an :class:`SDIDConfig` or compatible dict."""

        if isinstance(config, dict):
            try:
                config = SDIDConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid SDID configuration: {exc}") from exc

        self.config: SDIDConfig = config

        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.B: int = config.B
        self.seed: int = config.seed

        self.display_graphs: bool = config.display_graphs
        self.save: Any = config.save
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> SDIDResults:
        """Run the SDID pipeline and return the typed result container."""

        try:
            results = run_sdid(
                df=self.df,
                outcome=self.outcome,
                treat=self.treat,
                unitid=self.unitid,
                time=self.time,
                B=self.B,
                seed=self.seed,
            )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(f"SDID estimation failed: {exc}") from exc

        if self.display_graphs:
            try:
                plot_sdid(results)
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(f"SDID plotting failed: {exc}") from exc

        return results
