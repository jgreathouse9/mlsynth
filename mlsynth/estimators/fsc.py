"""FSC -- functional synthetic control for metric space-valued outcomes.

Okano, R. & Kurisu, D. (2026). *"Functional Synthetic Control Methods for Metric
Space-Valued Outcomes."* arXiv:2601.07539.

Ordinary synthetic control wants a number per unit per period. FSC is for the
case where what you observe is a whole object instead: an age-specific fertility
curve, an age-at-death distribution, a covariance matrix across product lines.
Collapsing such an object to a scalar -- a total, a mean, a Gini -- often
throws away the shape of the effect. The East German abortion
law in the paper's first application moves fertility down sharply at ages 20-30
and barely at all elsewhere; a total-fertility-rate synthetic control returns
that as one number.

How it works
------------

The obstacle is that these objects have no linear structure -- you cannot
average two networks by averaging their entries and expect a network -- so the
usual weighted-average counterfactual is not defined. FSC removes the obstacle
by carrying the objects into a Hilbert space through an isometry
:math:`\\Psi`, a map that preserves distances exactly. Inside that space
everything is linear again: the counterfactual is the weighted average of the
donor images, the donor weights solve the ordinary simplex problem on the
stacked pre-treatment objects, and the answer is carried back by
:math:`\\Psi^{-1}`.

Two consequences follow, and both are handled here. Because the image of
:math:`\\Psi` is convex, a convex combination of donors is always a valid object,
so the plain estimator needs no repair. The ridge augmentation of section 3.2 --
which corrects residual pre-treatment imbalance the simplex cannot, at the cost
of allowing negative weights -- can leave that image, so its output is projected
back onto it.

Spaces
------

``space`` selects the isometry and the projection, following the paper's
examples:

* ``"function"`` -- curves in :math:`L^2`. :math:`\\Psi` is the identity and the
  image is the whole space, so nothing is projected.
* ``"distribution"`` -- probability distributions under the 2-Wasserstein
  metric, supplied as quantile functions. The image is the set of increasing
  functions, so the projection is the increasing rearrangement.
* ``"matrix"`` -- symmetric positive semidefinite matrices under the Frobenius
  metric, supplied as the row-major lower triangle. The projection clips
  negative eigenvalues. Since :math:`\\mathcal H` is Euclidean here the spline
  expansion is skipped, and the off-diagonal coordinates are scaled by
  :math:`\\sqrt 2` so that the half-vectorisation really is an isometry.

Data
----

A long panel with one row per ``(unit, time, argument)``, where ``argument``
names the point at which each value is observed -- the age, the quantile level,
the matrix cell. Every cell must be observed on the same grid. Ingestion pivots
each argument slice through
:func:`mlsynth.utils.datautils.dataprep` and stacks them.

Related estimators
------------------

:class:`mlsynth.DSC` targets unconditional quantile effects from individual-level
microdata; reach for it when your rows are people rather than grid points.
:class:`mlsynth.COMPSC` handles compositions, which are the paper's Example 5.
The scalar special case of FSC is :class:`mlsynth.VanillaSC` with
``augment="ridge"`` -- the same ridge correction, one grid point wide.

Note the unrelated :class:`mlsynth.FSCM`, which is Cerulli's forward-selected
synthetic control.
"""

from __future__ import annotations

from typing import Union

from ..exceptions import (MlsynthConfigError, MlsynthDataError,
                          MlsynthEstimationError)
from ..utils.fsc_helpers.config import FSCConfig
from ..utils.fsc_helpers.pipeline import run_fsc
from ..utils.fsc_helpers.plotter import plot_fsc
from ..utils.fsc_helpers.setup import prepare_fsc_inputs
from ..utils.fsc_helpers.structures import FSCResults

try:  # pydantic v2 / v1 compatibility for the error type
    from pydantic import ValidationError
except Exception:  # pragma: no cover
    ValidationError = Exception  # type: ignore


class FSC:
    """Functional synthetic control estimator.

    Parameters
    ----------
    config : FSCConfig or dict
        Configuration. See
        :class:`mlsynth.utils.fsc_helpers.config.FSCConfig`.

    Examples
    --------
    >>> from mlsynth import FSC                                # doctest: +SKIP
    >>> res = FSC({                                            # doctest: +SKIP
    ...     "df": fertility, "outcome": "asfr", "treat": "treat",
    ...     "unitid": "unit", "time": "time", "argument": "age",
    ...     "space": "function",
    ... }).fit()
    >>> res.pre_treatment_fit                                  # doctest: +SKIP
    >>> res.curves[-1].effect                                  # doctest: +SKIP
    """

    def __init__(self, config: Union[FSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = FSCConfig(**config)
            except MlsynthConfigError:
                raise
            except (ValidationError, MlsynthDataError) as exc:
                raise MlsynthConfigError(
                    f"Invalid FSC configuration: {exc}") from exc
        if not isinstance(config, FSCConfig):
            raise MlsynthConfigError(
                "config must be an FSCConfig or a dict of its fields.")
        self.config: FSCConfig = config

    def fit(self) -> FSCResults:
        """Estimate the counterfactual objects and return standardized results."""
        inputs = prepare_fsc_inputs(
            df=self.config.df, outcome=self.config.outcome,
            treat=self.config.treat, unitid=self.config.unitid,
            time=self.config.time, argument=self.config.argument,
            numeric_grid=self.config.space != "matrix")
        results = run_fsc(inputs, self.config)

        plot = self.config.resolved_plot()
        if plot.display or plot.save:
            plot_fsc(results, argument_label=self.config.argument,
                     outcome_label=self.config.outcome, save=plot.save,
                     display=bool(plot.display))
        return results
