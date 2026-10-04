"""Configuration for the TBRMM estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator

from ...config_models import BaseMAREXConfig
from ...exceptions import MlsynthDataError


from ..tbr_helpers.config import TBRConfig

#: The merged config. TBR and TBRMM were the analysis and design halves of one
#: method, and their configs are now one class: the groups arrive either named
#: by ``treatment_col``/``control_col`` or searched for under
#: ``max_treatment_size``. ``TBRMMConfig`` is the pre-merge name for it.
TBRMMConfig = TBRConfig
