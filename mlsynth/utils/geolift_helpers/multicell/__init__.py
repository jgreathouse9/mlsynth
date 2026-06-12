"""Multi-cell GeoLift analysis: per-cell effects against a shared control pool."""

from .analyze import analyze_multicell
from .config import MultiCellGeoLiftConfig
from .dataprep import multicell_dataprep
from .structures import MultiCellResults

__all__ = [
    "analyze_multicell",
    "multicell_dataprep",
    "MultiCellGeoLiftConfig",
    "MultiCellResults",
]
