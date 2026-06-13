"""MULTICELLGEOLIFT: multi-cell GeoLift analysis (several treatment cells, one
shared control pool)."""

from __future__ import annotations

from typing import Optional, Union

from pydantic import ValidationError

from ..exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from ..utils.geolift_helpers.multicell import (
    MultiCellGeoLiftConfig,
    MultiCellResults,
    analyze_multicell,
    multicell_dataprep,
)


class MULTICELLGEOLIFT:
    """Multi-cell GeoLift: measure several treatment cells at once.

    Given a unit-level **cell-membership** column (``"A"``, ``"B"``, ... for
    treated geos; blank / ``control_label`` for controls) and a ``post_col``
    treatment window, each cell is measured against the **shared control** pool
    with the fixed-effect Augmented SCM + conformal inference (the other cells
    are excluded from each cell's donor pool), and the cells are compared
    (GeoLift's non-overlapping-CI winner rule).

    Parameters
    ----------
    config : MultiCellGeoLiftConfig or dict
        See
        :class:`mlsynth.utils.geolift_helpers.multicell.config.MultiCellGeoLiftConfig`.

    Examples
    --------
    >>> from mlsynth import MULTICELLGEOLIFT
    >>> res = MULTICELLGEOLIFT({"df": panel, "outcome": "Y", "unitid": "location",
    ...     "time": "date", "cell_column_name": "cell", "post_col": "post"}).fit()  # doctest: +SKIP
    >>> res.cells["A"].effects.att, res.winner                                      # doctest: +SKIP
    """

    def __init__(self, config: Union[MultiCellGeoLiftConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = MultiCellGeoLiftConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(str(exc)) from exc
        if not isinstance(config, MultiCellGeoLiftConfig):
            raise MlsynthConfigError(
                "config must be a MultiCellGeoLiftConfig or a dict of its fields."
            )
        self.config = config
        self._result: Optional[MultiCellResults] = None

    def fit(self) -> MultiCellResults:
        """Resolve the cells, measure each against the shared control, compare."""
        try:
            prep = multicell_dataprep(
                self.config.df, self.config.unitid, self.config.time,
                self.config.outcome, cell_column_name=self.config.cell_column_name,
                post_col=self.config.post_col, control_label=self.config.control_label,
            )
            result = analyze_multicell(
                prep["Ywide"], prep["cell_map"], prep["control_units"],
                prep["pre_periods"], how=self.config.how, augment=self.config.augment,
                fixed_effects=self.config.fixed_effects, alpha=self.config.alpha,
                ns=self.config.ns, seed=self.config.seed,
                conformal_type=self.config.conformal_type, cpic=self.config.cpic,
            )
            if self.config.display_graphs:
                from ..utils.geolift_helpers.multicell.plotter import plot_multicell
                plot_multicell(result, show=True)
            self._result = result
            return result
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise MlsynthEstimationError(f"MULTICELLGEOLIFT failed: {exc}") from exc
