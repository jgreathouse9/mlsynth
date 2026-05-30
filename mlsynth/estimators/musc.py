"""Modified Unbiased Synthetic Control (MUSC).

A thin, NumPy-first orchestration over :mod:`mlsynth.utils.musc_helpers`.
MUSC is the Bottmer, Imbens, Spiess & Warnick (2024 JBES) modification
of the Synthetic Control estimator. It adds a single linear restriction
to the canonical SC quadratic programme -- the column-sums-to-zero
condition on the weight matrix -- and that single change makes the
resulting ATT estimator **exactly unbiased under random assignment of
which unit is treated** (Lemma 1).

In addition to the unbiased point estimator the package ships:

* :func:`mlsynth.utils.musc_helpers.unbiased_variance` -- the
  closed-form Proposition 1 variance estimator (eq. 3.3 of the paper);
* :func:`mlsynth.utils.musc_helpers.randomization_ci` -- the exact
  randomization-based confidence interval of Section 3.5;
* an SC comparator under the same matrix-form parametrisation, so the
  effect of adding the column-balance restriction is directly visible
  on the result object.
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd

from ..config_models import MUSCConfig
from ..utils.datautils import balance
from ..utils.musc_helpers import (
    MUSCResults,
    derive_treatment,
    plot_musc,
    prepare_musc_inputs,
    run_musc,
)


class MUSC:
    """Modified Unbiased Synthetic Control estimator.

    Parameters
    ----------
    config : MUSCConfig or dict
        Validated configuration. Beyond the common fields (``df``,
        ``outcome``, ``treat``, ``unitid``, ``time``,
        ``display_graphs``, ``save``, colours), MUSC reads ``alpha``
        (significance level for the CIs), ``run_inference`` (toggle
        the Prop 1 variance + randomization CI), and ``solver``
        (cvxpy solver).

    References
    ----------
    Bottmer, L., Imbens, G. W., Spiess, J., & Warnick, M. (2024).
    A Design-Based Perspective on Synthetic Control Methods.
    Journal of Business & Economic Statistics, 42(2), 762-773.
    DOI: 10.1080/07350015.2023.2238788.
    """

    def __init__(self, config: Union[MUSCConfig, dict]) -> None:
        if isinstance(config, dict):
            config = MUSCConfig(**config)
        self.config = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str, dict] = config.save
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> MUSCResults:
        """Run the MUSC pipeline end to end.

        Returns
        -------
        MUSCResults
            Result container with two :class:`MUSCVariantFit` entries
            (``"SC"`` and ``"MUSC"``), the Prop 1 variance and CI in
            the :attr:`inference` attribute, and convenience aliases
            (``att``, ``att_ci``, ``counterfactual``, ``gap``,
            ``donor_weights``, ``pre_rmse``) forwarding to the MUSC
            variant.
        """
        balance(self.df, self.unitid, self.time)
        treated_unit, intervention_time = derive_treatment(
            self.df, self.unitid, self.time, self.treat
        )
        inputs = prepare_musc_inputs(
            self.df,
            unitid=self.unitid,
            time=self.time,
            outcome=self.outcome,
            treated_unit=treated_unit,
            intervention_time=intervention_time,
        )
        results = run_musc(
            inputs,
            alpha=self.config.alpha,
            run_inference=self.config.run_inference,
            solver=self.config.solver,
            verbose=False,
        )
        if self.display_graphs:
            plot_musc(
                results,
                outcome=self.outcome,
                time=self.time,
                treated_color=self.treated_color,
                counterfactual_color=(
                    self.counterfactual_color
                    if isinstance(self.counterfactual_color, str)
                    else self.counterfactual_color[0]
                ),
                save=self.save,
            )
        return results
