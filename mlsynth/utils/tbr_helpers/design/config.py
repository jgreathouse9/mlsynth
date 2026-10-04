"""The searched mode reads the same config as the named one.

TBR and TBRMM were the analysis and design halves of one method and their
configs are now one class, :class:`~mlsynth.config_models.TBRConfig`: the
groups arrive either named by ``treatment_col``/``control_col`` or searched
for under ``max_treatment_size``. This module re-exports it so the design
helpers can import their config from beside themselves, as every other
helper package does.
"""

from __future__ import annotations

from ..config import TBRConfig

__all__ = ["TBRConfig"]
