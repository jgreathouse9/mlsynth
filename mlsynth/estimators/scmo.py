"""Synthetic Control with Multiple Outcomes (SCMO).

A thin, NumPy-first orchestration over :mod:`mlsynth.utils.scmo_helpers`.
SCMO builds the synthetic control by matching the treated unit to donors on a
**matching matrix** assembled from one or more related outcomes/predictors
(optionally across several pre-treatment periods), rather than a single
outcome's long trajectory. Two weighting schemes from the literature are
supported, plus a model-average and the conventional baseline:

* ``concatenated`` -- Tian, Lee & Panchenko (2024): stack the standardized
  matching columns and solve one simplex SC.
* ``averaged`` -- Sun, Ben-Michael & Feller (2025): match the average of the
  standardized outcomes within each period (extra ``1/sqrt(K)`` bias gain).
* ``separate`` -- the conventional single-outcome SC baseline.
* ``MA`` -- convex model-average of the above, chosen by pre-treatment fit.

Matching is configured by a ``spec`` (``{"year": int|list, "vars": {...}}``);
when omitted it is built by stacking the primary outcome and ``addout`` over
the pre-treatment period. Inference defaults to the Abadie permutation
(placebo) test; conformal intervals are available via ``inference="conformal"``.
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd

from ..config_models import SCMOConfig
from ..utils.datautils import balance
from ..utils.scmo_helpers import (
    CONCATENATED,
    SCMOResults,
    assemble_scmo_results,
    build_spec,
    derive_treatment,
    plot_scmo,
    prepare_scmo_inputs,
    resolve_schemes,
    run_scmo,
)


class SCMO:
    """Synthetic Control with Multiple Outcomes estimator.

    Parameters
    ----------
    config : SCMOConfig or dict
        Validated configuration. Beyond the common fields (``df``, ``outcome``,
        ``treat``, ``unitid``, ``time``, ``display_graphs``, ``save``, colors),
        SCMO reads ``spec`` (matching specification), ``schemes`` /
        ``method``, ``demean``, ``inference``, ``addout`` and
        ``conformal_alpha``.

    References
    ----------
    Tian, W., Lee, S., & Panchenko, V. (2024). Synthetic Controls with Multiple
    Outcomes. arXiv:2304.02272.

    Sun, L., Ben-Michael, E., & Feller, A. (2025). Using Multiple Outcomes to
    Improve the Synthetic Control Method. Review of Economics and Statistics.
    """

    def __init__(self, config: Union[SCMOConfig, dict]) -> None:
        if isinstance(config, dict):
            config = SCMOConfig(**config)
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

    def fit(self) -> SCMOResults:
        """Build the matching matrix, fit the requested schemes, and return results.

        Returns
        -------
        SCMOResults
            Container of per-scheme
            :class:`~mlsynth.utils.scmo_helpers.structures.SCMOMethodFit`
            objects with convenience aliases (``att``, ``counterfactual``,
            ``gap``, ``donor_weights``) forwarding to the primary scheme.
        """
        balance(self.df, self.unitid, self.time)

        treated_unit, intervention_time, pre_years = derive_treatment(
            self.df, self.unitid, self.time, self.treat
        )
        spec = build_spec(self.config.spec, self.outcome, self.config.addout, pre_years)
        inputs = prepare_scmo_inputs(
            self.df, unitid=self.unitid, time=self.time, outcome=self.outcome,
            spec=spec, treated_unit=treated_unit, intervention_time=intervention_time,
        )

        schemes = resolve_schemes(self.config.schemes, self.config.method)
        fits = run_scmo(
            inputs, schemes,
            demean=self.config.demean,
            inference=self.config.inference,
            conformal_alpha=self.config.conformal_alpha,
        )
        results = assemble_scmo_results(
            inputs, fits, selected_variant=schemes[0] if schemes else CONCATENATED
        )

        if self.display_graphs:
            plot_scmo(
                results, outcome=self.outcome, time=self.time,
                treated_color=self.treated_color,
                counterfactual_color=self.counterfactual_color, save=self.save,
            )
        return results
