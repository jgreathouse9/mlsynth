"""GEOLIFT: GeoLift market-selection design (experimental design for geo-experiments)."""

from __future__ import annotations

from typing import Optional, Union

from pydantic import ValidationError

from ..config_models import BaseEstimatorResults
from ..exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from ..utils.datautils import geoex_dataprep
from ..utils.geolift_helpers.config import GeoLiftConfig
from ..utils.geolift_helpers.marketselect.orchestration import GEOLIFTResults, run_design
from ..utils.geolift_helpers.marketselect.realize import realize_design
from ..utils.geolift_helpers.marketselect.plotter import plot_design


class GEOLIFT:
    """GeoLift market-selection design.

    Chooses which markets to treat *before* an experiment by simulating power
    over the historical (pre-treatment) panel, then -- once outcomes are observed
    (a ``post_col`` leaving a pre/post split) -- realizes the chosen design into a
    standardized effect report with conformal inference.

    Parameters
    ----------
    config : GeoLiftConfig or dict
        Configuration. See
        :class:`mlsynth.utils.geolift_helpers.config.GeoLiftConfig`.

    Examples
    --------
    >>> from mlsynth import GEOLIFT
    >>> res = GEOLIFT({"df": panel, "outcome": "Y", "unitid": "location",
    ...                "time": "date", "treatment_size": 3, "durations": [14],
    ...                "effect_sizes": [0.0, 0.1, 0.2]}).fit()   # doctest: +SKIP
    >>> res.selected_units                                       # doctest: +SKIP
    """

    def __init__(self, config: Union[GeoLiftConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = GeoLiftConfig(**config)
            except ValidationError as exc:  # pragma: no cover - passthrough
                raise MlsynthConfigError(str(exc)) from exc
        if not isinstance(config, GeoLiftConfig):
            raise MlsynthConfigError(
                "config must be a GeoLiftConfig or a dict of its fields."
            )
        self.config = config
        self._result: Optional[GEOLIFTResults] = None

    def fit(self) -> GEOLIFTResults:
        """Run the market-selection design (design phase) and return the result."""
        try:
            self._result = run_design(self.config)
            return self._result
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise MlsynthEstimationError(f"GEOLIFT design failed: {exc}") from exc

    def realize(self) -> BaseEstimatorResults:
        """Realize the winning design on the full (pre+post) panel.

        Requires a ``post_col`` that leaves post-treatment periods. Populates and
        returns ``result.report`` (ATT, conformal intervals, joint p-value).
        """
        if self._result is None:
            self.fit()
        winner = self._result.search.winner
        if winner is None:
            raise MlsynthEstimationError("no winning design to realize.")
        pre = self._result.metadata["pre_periods"]
        full = geoex_dataprep(
            self.config.df, self.config.unitid, self.config.time, self.config.outcome
        )["Ywide"]
        if pre >= full.shape[0]:
            raise MlsynthConfigError(
                "no post-treatment periods to realize; set post_col to leave a post window."
            )
        try:
            report = realize_design(
                full, winner.candidate, pre,
                how=self.config.how, augment=self.config.augment,
                alpha=self.config.alpha, ns=self.config.ns, seed=self.config.seed,
            )
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise MlsynthEstimationError(f"GEOLIFT realize failed: {exc}") from exc
        self._result.report = report
        return report

    def plot(self, *, report: Optional[BaseEstimatorResults] = None, **kwargs):
        """Plot the recommended design (design phase, or realized post phase).

        Renders in the mlsynth house style. Pass a realized ``report`` (or call
        :meth:`realize` first) to draw the post phase with conformal intervals.
        """
        if self._result is None:
            self.fit()
        report = report if report is not None else self._result.report
        return plot_design(self._result, report=report, **kwargs)
