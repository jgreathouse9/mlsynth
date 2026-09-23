"""GEOX: geo experiment design by simulated power.

GEOX chooses which geographic markets to treat in a marketing geo experiment,
before the experiment runs. It scores every candidate test region by simulated
power: slide a pseudo-treatment window back through the observed history, inject
a known lift, and ask how often the estimator would have detected it. The region
with the smallest reliably detectable effect wins.

The market-selection loop follows GeoLift (Meta's geo-experiment package):
anchor a candidate at each market and take its most correlated neighbours,
sweep effect sizes over backward backtests, then rank by a composite of the
minimum detectable effect, the power at that effect, and how faithfully the
estimator recovers the injected lift.

Which estimator does the scoring is a setting. ``engine="sdid"`` (the default)
is synthetic difference-in-differences (Arkhangelsky, Athey, Hirshberg, Imbens &
Wager 2021), which reweights donor markets *and* pre-periods, so no single
pre-period has to be matched exactly and a level gap between the test region and
its donors is differenced out. ``engine="augsynth"`` is the ridge-augmented
synthetic control (Ben-Michael, Feller & Rothstein 2021) GeoLift itself scores
with. Everything around the fit -- nomination, backtests, injection, power, the
MDE, the rank, the constraints -- is the same either way.

``inference`` varies separately. Placebo reassignment (SDID's Algorithm 4) needs
only a donor-built counterfactual and runs on either engine; conformal
permutation runs on augsynth alone, because SDID's time weights exist to say
pre-periods are not exchangeable.

The output is a *design* -- which markets to treat, the donor weights the
analysis will use, and the power table behind the choice -- not a treatment
effect. The ``report`` slot stays ``None`` until the experiment has run.
"""

from ..utils.geox_helpers.config import GEOXConfig
from ..utils.geox_helpers.orchestration import run_design
from ..utils.geox_helpers.structures import GEOXResults


class GEOX:
    """Geo experiment design by simulated power.

    Parameters
    ----------
    config : GEOXConfig
        Panel, candidate-region constraints, scoring engine, simulation grid,
        and decision rule.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlsynth import GEOX
    >>> from mlsynth.config_models import GEOXConfig
    >>> df = pd.read_csv("basedata/geolift_test_data.csv")  # doctest: +SKIP
    >>> design = GEOX(GEOXConfig(          # doctest: +SKIP
    ...     df=df, unitid="location", time="date", outcome="Y",
    ...     treatment_size=2, durations=[14],
    ...     effect_sizes=[-0.1, 0.0, 0.1, 0.2], n_backtests=5,
    ... )).fit()
    >>> design.selected_units                     # doctest: +SKIP
    ['chicago', 'portland']
    """

    def __init__(self, config: GEOXConfig) -> None:
        self.config = config

    def fit(self) -> GEOXResults:
        """Run the design search and return the chosen test region.

        Returns
        -------
        GEOXResults
            A :class:`~mlsynth.config_models.DesignResult` carrying
            ``selected_units``, ``assignment``, ``design_weights``, the ranked
            ``power`` table, and the full ``search`` detail.
        """
        return run_design(self.config)
