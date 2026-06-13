"""End-to-end assembly for the GeoLift market-selection design.

``run_design`` wires the whole pipeline together:

    geoex_dataprep -> rank_markets_by_correlation -> generate_candidate_markets
    -> run_simulations -> compute_power -> compute_rank
    -> design_fit (per candidate) -> GEOLIFTResults (a DesignResult).
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from mlsynth.config_models import DesignResult
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.datautils import geoex_dataprep

from ..config import GeoLiftConfig
from .helpers.similarity import rank_markets_by_correlation
from .helpers.candidates import generate_candidate_markets
from .helpers.batch import run_simulations
from .helpers.aggregate import compute_power, compute_rank
from .helpers.constraints import (
    admissible_candidates,
    build_conflict_graph,
    conflict_neighbors,
    unit_attribute_map,
)
from .helpers.design import design_fit, CandidateDesign


@dataclass
class MarketSelectSearch:
    """The design search -- every candidate test region and the winner.

    Attributes
    ----------
    shortlist : pd.DataFrame
        Ranked table of candidate designs (one row per candidate, duration).
    candidates : list of CandidateDesign
        Every candidate's deployable design (weights / intercept / time series /
        fit diagnostics), with ``mde`` / ``power`` / ``rank`` stitched in.
    winner : CandidateDesign or None
        The top-ranked design (``None`` if no candidate cleared the power
        threshold).
    """

    shortlist: pd.DataFrame
    candidates: List[CandidateDesign]
    winner: Optional[CandidateDesign]


class GEOLIFTResults(DesignResult):
    """Top-level result of the GeoLift market-selection design (``run_design``).

    A :class:`~mlsynth.config_models.DesignResult` front door (``selected_units``
    / ``design_weights`` / ``power`` / ``metadata`` / ``report``) plus the grouped
    :class:`MarketSelectSearch` detail.
    """

    search: Optional[MarketSelectSearch] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


def run_design(config: GeoLiftConfig) -> GEOLIFTResults:
    """Run the full GeoLift market-selection design from a :class:`GeoLiftConfig`."""
    prep = geoex_dataprep(
        config.df, config.unitid, config.time, config.outcome, post_col=config.post_col
    )
    Ywide = prep["Ywide"]

    ranked = rank_markets_by_correlation(Ywide)
    candidates = generate_candidate_markets(
        ranked,
        config.treatment_size,
        to_be_treated=config.to_be_treated,
        not_to_be_treated=config.not_to_be_treated,
        run_stochastic=config.run_stochastic,
        stochastic_mode=config.stochastic_mode,
        rng=config.seed,
    )

    # Design constraints (geography / interference). The conflict graph drives
    # both the Stage-1 no-interference filter on candidates (treatment criterion)
    # and the Stage-2 spillover donor exclusion (control criterion). Absent any
    # constraint config the graph is empty and nothing changes.
    conflict = None
    if config.cluster_col is not None or config.adjacency is not None:
        cluster_map = (
            unit_attribute_map(config.df, config.unitid, config.cluster_col)
            if config.cluster_col is not None else None
        )
        conflict = build_conflict_graph(
            Ywide.columns, cluster_map=cluster_map, adjacency=config.adjacency,
            spillover_threshold=config.spillover_threshold,
        )
        candidates = admissible_candidates(candidates, conflict=conflict)
        if not candidates:
            raise MlsynthConfigError(
                "no candidate test region satisfies the design constraints "
                "(e.g. treatment_size exceeds the number of clusters, or the "
                "forced-in markets interfere). Relax the constraint or the "
                "treatment_size."
            )

    cube = run_simulations(
        Ywide, candidates, config.durations, config.lookback_window, config.effect_sizes,
        how=config.how, augment=config.augment, ns=config.ns, seed=config.seed,
        conformal_type=config.conformal_type, fixed_effects=config.fixed_effects,
        cpic=config.cpic, conflict=conflict,
    )
    power_table = compute_power(cube, alpha=config.alpha)
    shortlist = compute_rank(power_table, power_threshold=config.power_threshold,
                             budget=config.budget)

    # Per-candidate deployable design fit (full pre-period), with the same
    # spillover donor exclusion the scoring stage used.
    designs = {
        candidate: design_fit(
            Ywide, candidate, how=config.how, augment=config.augment,
            fixed_effects=config.fixed_effects,
            exclude=(conflict_neighbors(candidate, conflict) if conflict else None),
        )
        for candidate in candidates
    }

    # Stitch each candidate's best (lowest-rank) shortlist row into its design.
    best_row = {}
    for _, row in shortlist.iterrows():
        cand = row["candidate"]
        if cand not in best_row or row["rank"] < best_row[cand]["rank"]:
            best_row[cand] = {"rank": row["rank"], "mde": row["mde"], "power": row["power"]}
    for candidate, design in designs.items():
        match = best_row.get(candidate)
        if match is not None:
            design.rank = float(match["rank"])
            design.mde = float(match["mde"])
            design.power = float(match["power"])

    candidate_designs = list(designs.values())

    winner = None
    winner_units = None
    if not shortlist.empty:
        winner_candidate = shortlist.sort_values("rank").iloc[0]["candidate"]
        winner = designs[winner_candidate]
        winner_units = sorted(map(str, winner_candidate))

    search = MarketSelectSearch(
        shortlist=shortlist, candidates=candidate_designs, winner=winner
    )

    return GEOLIFTResults(
        report=None,  # realized once post-treatment outcomes are observed
        power=shortlist,
        selected_units=winner_units,
        assignment=({"treated": winner_units} if winner_units is not None else None),
        design_weights=(winner.weights if winner is not None else None),
        metadata={
            "n_candidates": len(candidate_designs),
            "treatment_size": config.treatment_size,
            "pre_periods": prep["pre_periods"],
            "post_col": prep["post_col"],
            "winner_mde": (float(winner.mde) if winner is not None and winner.mde is not None else None),
        },
        search=search,
    )
