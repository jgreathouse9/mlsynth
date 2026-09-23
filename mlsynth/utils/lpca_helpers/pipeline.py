"""The LPCA causal pipeline: preprocess, match, fit locally, re-centre."""

from __future__ import annotations

import numpy as np

from ...config_models import WeightsResults
from ...exceptions import MlsynthDataError
from ..results_helpers import build_effect_submodels
from .core import local_pca_row, neighbour_index, pseudo_max_distance
from .structures import LPCAInputs, LPCAResults


def run_lpca(
    inputs: LPCAInputs,
    *,
    n_neighbours: int,
    max_components: int = 3,
    distance: str = "pseudo_max",
    demean: bool = True,
) -> LPCAResults:
    """Run local PCA and assemble :class:`LPCAResults`.

    Parameters
    ----------
    inputs : LPCAInputs
    n_neighbours : int
        Requested neighbourhood size ``K``, resolved by the caller.
    max_components : int
        Maximum local components extracted; the realised rank comes from the
        singular-value ratio rule.
    distance : str
        Matching distance, one of
        :data:`~mlsynth.utils.lpca_helpers.core.METRICS`.
    demean : bool
        Centre each period across units before matching. The period means are
        added back to the counterfactual, so the returned path is always on the
        scale of the outcome column.

    Raises
    ------
    MlsynthDataError
        If the requested neighbourhood exceeds the number of units, or exceeds
        what the local SVD can use.
    """
    if n_neighbours > inputs.N:
        raise MlsynthDataError(
            f"n_neighbors ({n_neighbours}) exceeds the {inputs.N} units in the "
            "panel; the neighbourhood counts the unit itself."
        )
    if n_neighbours < 2:
        raise MlsynthDataError(
            f"n_neighbors ({n_neighbours}) must be at least 2; a neighbourhood "
            "of one unit carries no local structure to decompose."
        )
    # The local SVD cannot extract more components than the neighbourhood has
    # units, so a narrow neighbourhood caps the request. The effective value is
    # reported in the result's metadata.
    components = min(max_components, n_neighbours - 1)

    match, T0, treated = inputs.match_periods, inputs.T0, inputs.treated_idx
    working = inputs.Y_masked.copy()
    if demean:
        period_means = np.nanmean(working, axis=0)
        working = working - period_means
    else:
        period_means = None
    # The missing cells sit at the centre of their period once centred, which
    # is the perturbation Theorem 6.1 bounds.
    working[np.isnan(working)] = 0.0

    distances = pseudo_max_distance(working[:, :match], metric=distance)
    keep = neighbour_index(distances, n_neighbours)
    block = working[:, match:]

    fitted = np.empty((treated.size, block.shape[1]), dtype=float)
    ranks, sizes, neighbours_by_unit, spectra = [], [], {}, []
    weights_by_unit, self_weights = {}, []
    for row, unit in enumerate(treated):
        indicator = keep[:, unit]
        members = np.flatnonzero(indicator)
        reconstructed, rank, weights = local_pca_row(
            block, indicator, int(unit), n_neighbours,
            n_components=components,
        )
        fitted[row] = reconstructed
        ranks.append(rank)
        sizes.append(int(indicator.sum()))
        names = [inputs.unit_names[j] for j in members]
        neighbours_by_unit[inputs.unit_names[unit]] = names
        weights_by_unit[inputs.unit_names[unit]] = {
            name: float(w) for name, w in zip(names, weights)
        }
        self_weights.append(float(weights[int(np.flatnonzero(members == unit)[0])]))
        spectra.append(
            np.linalg.svd(block[members], compute_uv=False)[:components]
        )

    if period_means is not None:
        fitted = fitted + period_means[match:]

    observed = inputs.Y[treated][:, match:].mean(axis=0)
    counterfactual = fitted.mean(axis=0)
    n_pre_reported = int(T0 - match)
    n_post = int(inputs.T - T0)
    att = float(np.mean(observed[n_pre_reported:] - counterfactual[n_pre_reported:]))

    first = inputs.unit_names[treated[0]]
    metadata = {
        "N": inputs.N, "T": inputs.T, "T0": T0,
        "match_periods": match,
        "pre_periods_reported": n_pre_reported,
        "post_periods": n_post,
        "n_treated": int(treated.size),
        "distance": distance,
        "demeaned": bool(demean),
        "max_components_effective": int(components),
        "rank_rule_active": bool(np.log(np.log(n_neighbours)) > 1.0),
        "self_weight": self_weights[0],
        "ranks_by_unit": {inputs.unit_names[u]: int(r)
                          for u, r in zip(treated, ranks)},
    }
    submodels = build_effect_submodels(
        observed_outcome=observed,
        counterfactual_outcome=counterfactual,
        n_pre_periods=n_pre_reported,
        n_post_periods=n_post,
        time_periods=np.asarray(inputs.time_labels[match:]),
        weights=WeightsResults(
            donor_weights=weights_by_unit[first],
            summary_stats={"neighbourhood_size": float(sizes[0]),
                           "self_weight": self_weights[0],
                           "weight_sum": float(sum(weights_by_unit[first].values()))},
        ),
        method_name="LPCA",
        effects_overrides={"att": att},
        intervention_time=(inputs.time_labels[T0] if T0 < inputs.T
                           else inputs.time_labels[-1]),
    )
    return LPCAResults(
        **submodels,
        inputs=inputs,
        counterfactual_matrix=fitted,
        neighbours=neighbours_by_unit[first],
        neighbours_by_unit=neighbours_by_unit,
        neighbour_weights=weights_by_unit[first],
        self_weight=self_weights[0],
        n_neighbours=int(n_neighbours),
        neighbourhood_size=sizes[0],
        local_rank=int(ranks[0]),
        singular_values=np.asarray(spectra[0], dtype=float),
        period_means=period_means,
        metadata=metadata,
    )
