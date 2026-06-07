"""VanillaSC: the standard synthetic control, on the bilevel engine.

The ordinary single-treated synthetic control method (Abadie & Gardeazabal
2003; Abadie, Diamond & Hainmueller 2010), implemented on mlsynth's
self-contained bilevel machinery:

* **No covariates** -> the well-posed convex problem: donor weights ``W``
  minimise the pre-treatment outcome fit on the simplex. Unique up to donor
  collinearity, deterministic, reproducible.
* **Covariates** -> the bilevel program (predictor weights ``V`` + donor
  weights ``W``), solved by a *reliable* backend: ``"mscmt"`` (global
  differential evolution, Becker-Kloessner 2018), ``"malo"`` (corner search,
  Malo et al. 2024), or ``"penalized"`` (unique/sparse, Abadie-L'Hour 2021).

Because predictor weights are generically non-identified, VanillaSC reports a
``v_agreement`` diagnostic (the gap between the two MSCMT canonical ``V``
choices): small means ``V`` is well identified, large means the predictor
weights -- and the donor weights they imply -- are fragile.
"""

from __future__ import annotations

from typing import Union

from ..config_models import BaseEstimatorResults
from ..utils.vanillasc_helpers.config import VanillaSCConfig
from ..exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from ..utils.vanillasc_helpers.pipeline import run_vanillasc

try:  # pydantic v2 / v1 compatibility for the error type
    from pydantic import ValidationError
except Exception:  # pragma: no cover
    ValidationError = Exception  # type: ignore


class VanillaSC:
    """Standard synthetic control estimator (bilevel engine).

    Parameters
    ----------
    config : VanillaSCConfig or dict
        Configuration. See
        :class:`mlsynth.utils.vanillasc_helpers.config.VanillaSCConfig`.

    Examples
    --------
    >>> from mlsynth.estimators.vanillasc import VanillaSC
    >>> cfg = {"df": panel, "outcome": "gdp", "treat": "treated",
    ...        "unitid": "country", "time": "year",
    ...        "covariates": ["trade", "infrate"], "backend": "mscmt"}
    >>> res = VanillaSC(cfg).fit()
    >>> res.effects.att                      # doctest: +SKIP
    """

    def __init__(self, config: Union[VanillaSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = VanillaSCConfig(**config)
            except ValidationError as exc:  # pragma: no cover - passthrough
                raise MlsynthConfigError(str(exc)) from exc
        if not isinstance(config, VanillaSCConfig):
            raise MlsynthConfigError(
                "config must be a VanillaSCConfig or a dict of its fields."
            )
        self.config = config

    def fit(self) -> BaseEstimatorResults:
        """Estimate the synthetic control and return standardized results."""
        try:
            return run_vanillasc(self.config)
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise MlsynthEstimationError(f"VanillaSC estimation failed: {exc}") from exc
