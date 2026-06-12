"""GEOLIFT: GeoLift market-selection design (experimental design for geo-experiments)."""

from __future__ import annotations

from typing import Optional, Union

from pydantic import ValidationError

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
        """Run the market-selection design and return the result.

        Behaviour is driven by the data and config, not by manual sequencing:

        * always runs the **design** (candidate nomination -> power -> MDE ->
          rank -> per-candidate synthetic controls);
        * when a ``post_col`` leaves a post-treatment window, **realizes** the
          winning design on the full panel under the hood, populating
          ``result.report`` (conformal effect report) -- the ``DesignResult``
          resolving to its ``EffectResult``;
        * when ``display_graphs`` is set, **plots** the recommended design
          (design phase, or the realized post phase).
        """
        try:
            result = run_design(self.config)
            winner = result.search.winner
            if winner is not None:
                # Resolve the design to its effect report iff post outcomes exist.
                full = geoex_dataprep(
                    self.config.df, self.config.unitid, self.config.time,
                    self.config.outcome,
                )["Ywide"]
                pre = result.metadata["pre_periods"]
                if pre < full.shape[0]:
                    result.report = realize_design(
                        full, winner.candidate, pre,
                        how=self.config.how, augment=self.config.augment,
                        alpha=self.config.alpha, ns=self.config.ns, seed=self.config.seed,
                        conformal_type=self.config.conformal_type,
                    )
                if self.config.display_graphs:
                    plot_design(result, report=result.report, show=True)
            self._result = result
            return result
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise MlsynthEstimationError(f"GEOLIFT failed: {exc}") from exc
