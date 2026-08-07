"""SDIDGEO: synthetic difference-in-differences geo experiment design.

SDIDGEO chooses which geographic markets to treat in a marketing geo experiment,
before the experiment runs. It scores every candidate test region by simulated
power: slide a pseudo-treatment window back through the observed history, inject
a known lift, and ask how often synthetic difference-in-differences would have
detected it. The region with the smallest reliably detectable effect wins.

The market-selection loop follows GeoLift (Meta's geo-experiment package):
anchor a candidate at each market and take its most correlated neighbours,
sweep effect sizes over backward placements, then rank by a composite of the
minimum detectable effect, the power at that effect, and how faithfully the
estimator recovers the injected lift.

The estimator underneath is synthetic difference-in-differences (Arkhangelsky,
Athey, Hirshberg, Imbens & Wager 2021), not the augmented synthetic control
GeoLift uses. SDID reweights donor markets *and* pre-periods, so it does not
demand that any single pre-period be matched exactly, and its unit weights carry
a closed-form ridge instead of a cross-validated one. Detection is a
normal-approximation test against the placebo standard error (the paper's
Algorithm 4, the only one of its three variance procedures defined for a single
treated series).

The output is a *design* -- which markets to treat, the donor weights the
analysis will use, and the power table behind the choice -- not a treatment
effect. The ``report`` slot stays ``None`` until the experiment has run.
"""

from ..utils.sdidgeo_helpers.config import SDIDGEOConfig
from ..utils.sdidgeo_helpers.orchestration import run_design
from ..utils.sdidgeo_helpers.structures import SDIDGEOResults


class SDIDGEO:
    """Synthetic difference-in-differences geo experiment design.

    Parameters
    ----------
    config : SDIDGEOConfig
        Panel, candidate-region constraints, simulation grid, and decision rule.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlsynth import SDIDGEO
    >>> from mlsynth.config_models import SDIDGEOConfig
    >>> df = pd.read_csv("basedata/geolift_test_data.csv")  # doctest: +SKIP
    >>> design = SDIDGEO(SDIDGEOConfig(          # doctest: +SKIP
    ...     df=df, unitid="location", time="date", outcome="Y",
    ...     treatment_size=2, durations=[14],
    ...     effect_sizes=[-0.1, 0.0, 0.1, 0.2], lookback_window=5,
    ... )).fit()
    >>> design.selected_units                     # doctest: +SKIP
    ['chicago', 'portland']
    """

    def __init__(self, config: SDIDGEOConfig) -> None:
        self.config = config

    def fit(self) -> SDIDGEOResults:
        """Run the design search and return the chosen test region.

        Returns
        -------
        SDIDGEOResults
            A :class:`~mlsynth.config_models.DesignResult` carrying
            ``selected_units``, ``assignment``, ``design_weights``, the ranked
            ``power`` table, and the full ``search`` detail.
        """
        return run_design(self.config)
