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
from .aggregate import compute_mde, compute_power, compute_rank
from .batch import run_simulations
from .candidates import generate_candidate_markets
from .config import SDIDGEOConfig
from .constraints import (
    admissible_candidates,
    build_conflict_graph,
    conflict_neighbors,
    eligible_by_size,
    unit_attribute_map,
)
from .feasibility import audit_sdidgeo_feasibility
from .engine import sdid_fit_once
from .shaping import aggregate_treated, donor_matrix
from .simulate import simulate_backtest
from .similarity import rank_markets_by_correlation
from .structures import CandidateDesign, MarketSearch, SDIDGEOResults


def design_fit(Ywide: pd.DataFrame, candidate, n_pre: int,
               duration: int, exclude=None) -> CandidateDesign:
    """Deployable SDID design for one candidate, fit on the full pre-period.

    The scoring stage fits each backtest; this fits the design the
    experiment will actually deploy, with the pseudo-treatment window sitting at
    the end of the observed history.

    ``exclude`` drops the candidate's conflict-neighbours from its donor pool
    (the spillover exclusion restriction).
    """
    treated = aggregate_treated(Ywide, candidate, how="mean").to_numpy()
    donors_df = donor_matrix(Ywide, candidate, exclude=exclude)
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



def planning_mde(Ywide: pd.DataFrame, candidate, config: SDIDGEOConfig,
                power_table: pd.DataFrame, exclude=None) -> Optional[float]:
    """The winner's MDE re-scored on backtests that did not choose it.

    ``compute_rank`` hands back the smallest MDE in the field, and the smallest
    of many noisy estimates is optimistic: the region most likely to be picked
    is the one whose estimate happened to come out low. Re-scoring the winner on
    backtests held back from the search removes that, because those backtests
    played no part in selecting it -- the same reason a region fixed in advance
    is calibrated at any backtest count.

    Backtests ``n_backtests + 1 .. n_backtests + n_validation_backtests``
    sit deeper in history, so their pseudo-treatment windows differ from every
    window the search used. Returns ``None`` when the panel cannot carry them.
    """
    if config.n_validation_backtests < 1:
        return None
    n_periods = Ywide.shape[0]
    longest = max(config.durations)
    deepest = config.n_backtests + config.n_validation_backtests
    if longest + deepest - 1 >= n_periods:
        return None  # the panel cannot carry the extra backtests

    treated = aggregate_treated(Ywide, candidate, how="mean").to_numpy()
    donors = donor_matrix(Ywide, candidate, exclude=exclude).to_numpy()
    rows: List[dict] = []
    for duration in config.durations:
        for sim in range(config.n_backtests + 1, deepest + 1):
            for row in simulate_backtest(
                treated, donors, n_periods, duration, sim, config.effect_sizes,
                n_draws=config.n_draws, n_tr=len(candidate), seed=config.seed,
            ):
                row["candidate"] = candidate
                rows.append(row)
    if not rows:  # pragma: no cover - guarded by the depth check above
        return None

    held = compute_power(pd.DataFrame(rows), alpha=config.alpha)
    mde_table = compute_mde(held, power_threshold=config.power_threshold)
    values = mde_table["mde"].dropna()
    if values.empty:
        return None  # nothing detectable on the held-back backtests
    # Across durations, the design deploys the one it ranked best; take the
    # smallest magnitude, matching how compute_rank reads a candidate's row.
    return float(values.loc[values.abs().idxmin()])


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
    if longest + config.n_backtests - 1 >= n_periods:
        raise MlsynthConfigError(
            f"the deepest backtest (duration {longest}, sim "
            f"{config.n_backtests}) leaves no pre-period in a panel of "
            f"{n_periods} periods. Shorten durations or n_backtests.")

    forced = list(config.to_be_treated or ())
    unknown = [u for u in forced + list(config.not_to_be_treated or ())
               if u not in units]
    if unknown:
        raise MlsynthDataError(f"markets not found in the panel: {unknown}.")

    # Size band -- a treated-eligibility filter on the nomination pool only.
    # Out-of-band markets stay available as donors.
    not_treated = set(config.not_to_be_treated or ())
    size_ineligible: set = set()
    if config.size_col is not None and (
        config.min_size is not None or config.max_size is not None
    ):
        size_map = unit_attribute_map(config.df, config.unitid, config.size_col)
        eligible = eligible_by_size(units, size_map, min_size=config.min_size,
                                    max_size=config.max_size)
        size_ineligible = set(units) - set(eligible)

    eligible_for_treatment = [u for u in units
                              if u not in (not_treated | size_ineligible)]
    if size_ineligible and len(eligible_for_treatment) < max(sizes):
        raise MlsynthConfigError(
            f"the size band leaves only {len(eligible_for_treatment)} market(s) "
            f"eligible for treatment, fewer than the largest treatment_size "
            f"({max(sizes)}). Widen the size band.")

    # Conflict graph (cluster_col + adjacency): the independent-set filter on
    # candidates, and the spillover donor exclusion applied per candidate.
    conflict = None
    if config.cluster_col is not None or config.adjacency is not None:
        cluster_map = (
            unit_attribute_map(config.df, config.unitid, config.cluster_col)
            if config.cluster_col is not None else None)
        conflict = build_conflict_graph(
            units, cluster_map=cluster_map, adjacency=config.adjacency,
            spillover_threshold=config.spillover_threshold)

    # Stratum quotas: a coverage filter on the nominated regions.
    stratum_map = None
    required_strata = None
    has_quota = (config.min_per_stratum is not None
                 or config.max_per_stratum is not None)
    if config.stratum_col is not None and has_quota:
        stratum_map = unit_attribute_map(config.df, config.unitid,
                                         config.stratum_col)
        if config.min_per_stratum is not None:
            required_strata = {stratum_map[u] for u in eligible_for_treatment
                               if u in stratum_map}

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
            not_to_be_treated=(sorted(not_treated | size_ineligible,
                                      key=str) or None),
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

    if conflict is not None or (stratum_map is not None and has_quota):
        candidates = admissible_candidates(
            candidates, conflict=conflict, stratum_map=stratum_map,
            min_per_stratum=config.min_per_stratum,
            max_per_stratum=config.max_per_stratum,
            required_strata=required_strata)
        if not candidates:
            # Report which constraint bound the search, itemised in
            # have-vs-need shape. Audit the smallest requested size: if that
            # one cannot be met, no larger one can either.
            audit_sdidgeo_feasibility(
                eligible_for_treatment, min(sizes), conflict=conflict,
                stratum_map=stratum_map,
                min_per_stratum=config.min_per_stratum,
                max_per_stratum=config.max_per_stratum,
                required_strata=required_strata)
            raise MlsynthConfigError(  # pragma: no cover - audit covers the common cases
                "no candidate test region satisfies the design constraints "
                "(treatment_size may exceed the clusters/strata available to "
                "cover, or the forced-in markets may interfere). Relax a "
                "constraint or the treatment_size.")

    # Each candidate's spillover exclusion: its conflict-neighbours may not
    # enter its own synthetic control.
    excluded: Dict[frozenset, frozenset] = {
        c: (conflict_neighbors(c, conflict) if conflict is not None
            else frozenset())
        for c in candidates
    }

    cube = run_simulations(
        Ywide, candidates, config.durations, config.n_backtests,
        config.effect_sizes, n_draws=config.n_draws, seed=config.seed,
        cpic=config.cpic, n_jobs=config.n_jobs, excluded=excluded,
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
        c: design_fit(Ywide, c, n_pre_deploy, deploy_duration,
                      exclude=excluded.get(c))
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
        # Scored only now, and only for the winner, so the backtests behind it
        # cannot have influenced which region was picked.
        winner.mde_planning = planning_mde(Ywide, winning, config, power_table,
                                           exclude=excluded.get(winning))

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
            # The smallest MDE in the field, so the region most likely to be
            # picked is the one whose estimate came out low by luck.
            "winner_mde_optimistic": (
                float(winner.mde) if winner is not None
                and winner.mde is not None else None),
            # The same region scored on backtests that did not select it. Plan
            # against this one; winner_mde is the optimistic end.
            "winner_mde_planning": (
                float(winner.mde_planning) if winner is not None
                and winner.mde_planning is not None else None),
            "n_validation_backtests": config.n_validation_backtests,
        },
        search=search,
    )
