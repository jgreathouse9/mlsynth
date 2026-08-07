"""End-to-end assembly for the SDIDGEO design.

``run_design`` wires the pipeline together:

    geoex_dataprep -> rank_markets_by_correlation -> generate_candidate_markets
    -> run_simulations -> compute_power -> compute_rank -> design_fit
    -> SDIDGEOResults
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ...config_models import WeightsResults
from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import geoex_dataprep
from .aggregate import compute_power, compute_rank
from .batch import run_simulations
from .candidates import generate_candidate_markets
from .config import SDIDGEOConfig
from .engine import sdid_fit_once
from .shaping import aggregate_treated, donor_matrix
from .similarity import rank_markets_by_correlation
from .structures import CandidateDesign, MarketSearch, SDIDGEOResults


def design_fit(Ywide: pd.DataFrame, candidate, n_pre: int,
               duration: int) -> CandidateDesign:
    """Deployable SDID design for one candidate, fit on the full pre-period.

    The scoring stage fits each lookback placement; this fits the design the
    experiment will actually deploy, with the pseudo-treatment window sitting at
    the end of the observed history.
    """
    treated = aggregate_treated(Ywide, candidate, how="mean").to_numpy()
    donors_df = donor_matrix(Ywide, candidate)
    donors = donors_df.to_numpy()
    start = n_pre
    end = min(n_pre + duration - 1, Ywide.shape[0] - 1)
    fit = sdid_fit_once(treated, donors, n_pre, start, end,
                        n_tr=len(candidate))
    # SDID carries two weight vectors, so both go into the standard container:
    # donors over markets, and lambda over the pre-period dates.
    weights = WeightsResults(
        donor_weights={str(name): float(w)
                       for name, w in zip(donors_df.columns, fit.omega)},
        time_weights={period: float(w)
                      for period, w in zip(Ywide.index[:n_pre], fit.lam)},
    )
    return CandidateDesign(
        units=sorted(map(str, candidate)),
        weights=weights,
        observed=treated,
        counterfactual=fit.counterfactual,
        n_pre=int(n_pre),
        pre_rmspe=fit.pre_rmspe,
        scaled_l2=fit.scaled_l2,
    )


def run_design(config: SDIDGEOConfig) -> SDIDGEOResults:
    """Run the full SDIDGEO market-selection design from a config."""
    prep = geoex_dataprep(config.df, config.unitid, config.time,
                          config.outcome, post_col=config.post_col)
    Ywide = prep["Ywide"]
    n_periods = Ywide.shape[0]
    units = list(Ywide.columns)

    sizes = config.treatment_sizes
    if max(sizes) >= len(units):
        raise MlsynthConfigError(
            f"every treatment_size must leave at least one donor market; the "
            f"panel has {len(units)} and the largest requested is {max(sizes)}.")

    longest = max(config.durations)
    if longest + config.lookback_window - 1 >= n_periods:
        raise MlsynthConfigError(
            f"the deepest lookback placement (duration {longest}, sim "
            f"{config.lookback_window}) leaves no pre-period in a panel of "
            f"{n_periods} periods. Shorten durations or lookback_window.")

    forced = list(config.to_be_treated or ())
    unknown = [u for u in forced + list(config.not_to_be_treated or ())
               if u not in units]
    if unknown:
        raise MlsynthDataError(f"markets not found in the panel: {unknown}.")

    ranked = rank_markets_by_correlation(Ywide)
    # One nomination pass per requested size, pooled into a single field. A
    # size-3 region competes with a size-2 one on the same ranking, which is
    # the point of scanning sizes together.
    candidates: List[frozenset] = []
    seen = set()
    for size in sizes:
        for candidate in generate_candidate_markets(
            ranked, size,
            to_be_treated=config.to_be_treated,
            not_to_be_treated=config.not_to_be_treated,
            run_stochastic=config.run_stochastic,
            stochastic_mode=config.stochastic_mode,
            rng=config.seed,
        ):
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    if not candidates:
        raise MlsynthConfigError(
            "no candidate test region satisfies the constraints; relax "
            "to_be_treated / not_to_be_treated or the treatment_size.")

    cube = run_simulations(
        Ywide, candidates, config.durations, config.lookback_window,
        config.effect_sizes, n_draws=config.n_draws, seed=config.seed,
        cpic=config.cpic, n_jobs=config.n_jobs,
    )
    power_table = compute_power(cube, alpha=config.alpha)
    shortlist = compute_rank(power_table,
                             power_threshold=config.power_threshold,
                             budget=config.budget)
    if not shortlist.empty:
        # Surface the region size next to its MDE, so a size scan is readable
        # without unpacking the candidate frozensets.
        shortlist.insert(1, "treatment_size",
                         shortlist["candidate"].map(len).astype(int))

    # Deployable fit per candidate, with the design window at the end of history.
    deploy_duration = max(config.durations)
    n_pre_deploy = n_periods - deploy_duration
    designs: Dict[frozenset, CandidateDesign] = {
        c: design_fit(Ywide, c, n_pre_deploy, deploy_duration)
        for c in candidates
    }

    # Stitch each candidate's best (lowest-rank) shortlist row into its design.
    best: Dict[frozenset, dict] = {}
    for _, row in shortlist.iterrows():
        cand = row["candidate"]
        if cand not in best or row["rank"] < best[cand]["rank"]:
            best[cand] = {"rank": row["rank"], "mde": row["mde"],
                          "power": row["power"]}
    for cand, design in designs.items():
        match = best.get(cand)
        if match is not None:
            design.rank = float(match["rank"])
            design.mde = float(match["mde"])
            design.power = float(match["power"])

    winner = None
    winner_units = None
    if not shortlist.empty:
        winning = shortlist.sort_values("rank").iloc[0]["candidate"]
        winner = designs[winning]
        winner_units = sorted(map(str, winning))

    search = MarketSearch(shortlist=shortlist, power_table=power_table,
                          candidates=list(designs.values()), winner=winner)

    return SDIDGEOResults(
        report=None,  # realized once post-treatment outcomes are observed
        power=shortlist,
        selected_units=winner_units,
        assignment=({"treated": winner_units}
                    if winner_units is not None else None),
        design_weights=(winner.weights if winner is not None else None),
        metadata={
            "n_candidates": len(designs),
            "treatment_size": config.treatment_size,
            "treatment_sizes": sizes,
            "pre_periods": prep["pre_periods"],
            "post_col": prep["post_col"],
            "n_draws": config.n_draws,
            "winner_mde": (float(winner.mde) if winner is not None
                           and winner.mde is not None else None),
        },
        search=search,
    )
