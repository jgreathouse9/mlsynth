"""SPILLCORRECT: correct known contaminated controls in a fitted design.

A post-estimation step for a design estimator (MAREX). Given the fitted
design, the realized panel and the control units known to carry a spillover
from the treatment, it rebuilds the contaminated controls from the clean ones,
recomputes the design's effect and inference on the corrected panel, and
reports the corrected estimate beside the uncorrected one and, with the
design's configuration, the design re-solved without the contaminated
controls. See :mod:`mlsynth.utils.spillcorrect_helpers`.
"""
from __future__ import annotations

from typing import Union

from pydantic import ValidationError

from ..exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from ..utils.spillcorrect_helpers.config import SPILLCORRECTConfig
from ..utils.spillcorrect_helpers.pipeline import run_spillcorrect
from ..utils.spillcorrect_helpers.structures import SpillCorrectResults


class SPILLCORRECT:
    """Correct known contaminated controls in a fitted design.

    Parameters
    ----------
    config : SPILLCORRECTConfig or dict
        See :class:`mlsynth.utils.spillcorrect_helpers.config.SPILLCORRECTConfig`.

    Examples
    --------
    >>> design = MAREX(design_cfg).fit()                        # doctest: +SKIP
    >>> res = SPILLCORRECT({"df": realized, "outcome": "sales",
    ...                     "unitid": "region", "time": "week",
    ...                     "design": design,
    ...                     "affected_units": ["r07"]}).fit()  # doctest: +SKIP
    >>> res.att, res.uncorrected.att                            # doctest: +SKIP
    """

    def __init__(self, config: Union[SPILLCORRECTConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SPILLCORRECTConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid SPILLCORRECT configuration: {exc}") from exc
        self.config: SPILLCORRECTConfig = config

    def fit(self) -> SpillCorrectResults:
        """Correct, recompute the design's inference and report the arms."""
        try:
            return run_spillcorrect(self.config)
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"SPILLCORRECT: estimation failed ({exc}).") from exc
