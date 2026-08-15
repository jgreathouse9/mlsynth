"""Configuration for the LPCA estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from ...config_models import BaseEstimatorConfig


class LPCAConfig(BaseEstimatorConfig):
    """Configuration for the LPCA estimator.

    Feng (2024), *"Optimal Estimation of Large-Dimensional Nonlinear Factor
    Models"* (arXiv:2311.07243). Imputes the treated unit's counterfactual
    under a factor structure that may be nonlinear: units are matched to their
    ``K`` nearest neighbours on one block of periods, and a truncated SVD of
    the neighbour submatrix on the complementary block supplies the fit.
    Inherits the standard ``df`` / ``outcome`` / ``treat`` / ``unitid`` /
    ``time`` interface.

    The estimator takes the outcome as given. Feng's Kansas application works
    with first-differenced log GDP per capita; transform the outcome before
    passing it in, so the returned counterfactual is always on the scale of the
    column named by ``outcome``.

    Parameters
    ----------
    n_neighbors : int, optional
        Neighbourhood size ``K``, counting the unit itself. Defaults to
        ``round(N ** (2/3))``, the paper's rule. Ties in the distance are all
        kept, so the realised neighbourhood can exceed this; the result reports
        both.
    match_periods : int, optional
        Number of leading periods reserved for nearest-neighbour matching.
        Defaults to ``round(T * match_fraction)``. Feng's Kansas application
        uses 40 of 104 quarters. Must leave at least one pre-treatment period
        in the remaining block.
    match_fraction : float
        Fraction of periods used for matching when ``match_periods`` is None.
        Default 0.5, matching the paper's simulations.
    distance : {"pseudo_max", "euclidean", "average"}
        Distance used for matching. The default separates units by their
        noise-free factor structure under heteroskedastic errors, which the
        Euclidean distance does not.
    max_components : int
        Maximum number of local components extracted (the paper's ``nlam``,
        default 3). The realised rank is chosen by a singular-value ratio rule
        and reported on the result.
    demean : bool
        Centre each period across units before matching, adding the period
        means back to the counterfactual. Default True.
    """

    n_neighbors: Optional[int] = Field(
        default=None, ge=2,
        description="Neighbourhood size K; defaults to round(N ** (2/3)).")
    match_periods: Optional[int] = Field(
        default=None, ge=1,
        description="Leading periods reserved for nearest-neighbour matching.")
    match_fraction: float = Field(
        default=0.5, gt=0.0, lt=1.0,
        description="Fraction of periods used for matching when "
                    "match_periods is None.")
    distance: Literal["pseudo_max", "euclidean", "average"] = Field(
        default="pseudo_max", description="Distance used for matching.")
    max_components: int = Field(
        default=3, ge=1,
        description="Maximum number of local components extracted (nlam).")
    demean: bool = Field(
        default=True,
        description="Centre each period across units before matching.")

    # Every constraint that does not need the panel is a field constraint
    # above. The rest -- whether the matching block leaves a pre-treatment
    # period, whether the neighbourhood fits the unit count -- depends on ``T``
    # and ``N``, so it is enforced in
    # :func:`~mlsynth.utils.lpca_helpers.setup.prepare_lpca_inputs` and
    # :func:`~mlsynth.utils.lpca_helpers.pipeline.run_lpca`, which raise
    # ``MlsynthDataError``.
