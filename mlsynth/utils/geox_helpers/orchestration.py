"""End-to-end assembly for the GEOX design.

``run_design`` wires the pipeline together:

    geoex_dataprep -> rank_markets_by_correlation -> generate_candidate_markets
    -> run_simulations -> compute_power -> compute_rank -> design_fit
    -> GEOXResults
"""

from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

from ...config_models import WeightsResults
from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import geoex_dataprep
from .aggregate import (compute_accuracy, compute_exact_mde, compute_mde,
                        compute_power, compute_rank)
from .batch import run_simulations
from .candidates import generate_candidate_markets
from .config import GEOXConfig
from .constraints import (
    admissible_candidates,
    build_conflict_graph,
    conflict_neighbors,
    eligible_by_size,
    unit_attribute_map,
)
from .feasibility import audit_geox_feasibility
from .engines import resolve_engine
from .shaping import aggregate_treated, donor_matrix
from .simulate import simulate_backtest
from .realize import realize_design
from .similarity import rank_markets_by_correlation
from .structures import CandidateDesign, MarketSearch, GEOXResults


def design_fit(Ywide: pd.DataFrame, candidate, n_pre: int,
               duration: int, exclude=None, how: str = "mean",
               engine: str = "sdid",
               engine_kwargs: Optional[dict] = None) -> CandidateDesign:
    """Deployable SDID design for one candidate, fit on the full pre-period.

    The scoring stage fits each backtest; this fits the design the
    experiment will actually deploy, with the pseudo-treatment window sitting at
    the end of the observed history.

    ``exclude`` drops the candidate's conflict-neighbours from its donor pool
    (the spillover exclusion restriction).
    """
    # The fit runs on the mean whatever the reporting scale, so every candidate
    # is scored on the series the MDE describes.
    treated = aggregate_treated(Ywide, candidate, how="mean").to_numpy()
    donors_df = donor_matrix(Ywide, candidate, exclude=exclude)
    donors = donors_df.to_numpy()
    start = n_pre
    end = min(n_pre + duration - 1, Ywide.shape[0] - 1)
    fit = resolve_engine(engine).fit_once(treated, donors, n_pre, start, end,
                                          len(candidate),
                                          **(engine_kwargs or {}))
    # SDID carries two weight vectors, so both go into the standard container:
    # donors over markets, and lambda over the pre-period dates.
    weights = WeightsResults(
        donor_weights={str(name): float(w)
                       for name, w in zip(donors_df.columns, fit.donor_weights)},
        time_weights=({period: float(w)
                       for period, w in zip(Ywide.index[:n_pre],
                                            fit.time_weights)}
                      if fit.time_weights is not None else None),
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


class PlanningReadout(NamedTuple):
    """The winner's design scored on backtests that did not choose it.

    ``mde`` and ``rmse`` are read at the same duration, so the two numbers
    describe one experiment.
    """

    mde: Optional[float] = None
    rmse: Optional[float] = None


def planning_backtests(Ywide: pd.DataFrame, candidate, config: GEOXConfig,
                       exclude=None) -> pd.DataFrame:
    """Score one candidate on backtests held back from the search.

    ``compute_rank`` hands back the smallest MDE in the field, and the smallest
    of many noisy estimates is optimistic: the region most likely to be picked
    is the one whose estimate happened to come out low. Re-scoring the winner on
    backtests held back from the search removes that, because those backtests
    played no part in selecting it -- the same reason a region fixed in advance
    is calibrated at any backtest count. The same argument applies to accuracy,
    which is why one cube serves both readings.

    Backtests ``n_backtests + 1 .. n_backtests + n_validation_backtests``
    sit deeper in history, so their pseudo-treatment windows differ from every
    window the search used. Returns an empty frame when the panel cannot carry
    them.
    """
    if config.n_validation_backtests < 1:
        return pd.DataFrame()
    n_periods = Ywide.shape[0]
    longest = max(config.durations)
    deepest = config.n_backtests + config.n_validation_backtests
    if longest + deepest - 1 >= n_periods:
        return pd.DataFrame()  # the panel cannot carry the extra backtests

    treated = aggregate_treated(Ywide, candidate, how="mean").to_numpy()
    donors = donor_matrix(Ywide, candidate, exclude=exclude).to_numpy()
    # The validation backtests score on the same engine and settings the search
    # used, so the planning numbers are comparable with the ones they correct.
    ekw = engine_settings(config)
    rows: List[dict] = []
    for duration in config.durations:
        for sim in range(config.n_backtests + 1, deepest + 1):
            for row in simulate_backtest(
                treated, donors, n_periods, duration, sim, config.effect_sizes,
                n_draws=config.n_draws, n_tr=len(candidate), seed=config.seed,
                engine=config.engine, engine_kwargs=ekw,
            ):
                row["candidate"] = candidate
                rows.append(row)
    return pd.DataFrame(rows)


def planning_readout(cube: pd.DataFrame, config: GEOXConfig) -> PlanningReadout:
    """Read the MDE and the RMSE off the held-back backtests.

    Across durations the design deploys the one it ranked best, so the MDE is
    the smallest magnitude, matching how ``compute_rank`` reads a candidate's
    row. The RMSE is reported at that same duration. When nothing is detectable
    the MDE is absent and the RMSE falls back to the duration the design
    deploys at, which is the longest requested.
    """
    if cube.empty:
        return PlanningReadout()

    held = compute_power(cube, alpha=config.alpha)
    mde_table = compute_mde(held, power_threshold=config.power_threshold)
    accuracy = compute_accuracy(cube).set_index("duration")

    detectable = mde_table.dropna(subset=["mde"])
    if detectable.empty:
        mde = None  # nothing detectable on the held-back backtests
        duration = max(config.durations)
    else:
        best = detectable.loc[detectable["mde"].abs().idxmin()]
        mde = float(best["mde"])
        duration = best["duration"]

    rmse = (float(accuracy.loc[duration, "rmse"])
            if duration in accuracy.index else None)
    return PlanningReadout(mde=mde, rmse=rmse)


def engine_settings(config: GEOXConfig) -> dict:
    """The engine-specific settings bundle for ``config``.

    One bundle whichever engine runs: each engine reads what applies to it and
    ignores the rest, so the pipeline never branches on which is active.
    """
    kw: dict = {"inference": config.inference}
    if config.engine == "augsynth":
        kw.update(ns=config.ns, conformal_type=config.conformal_type,
                  fixed_effects=bool(config.fixed_effects),
                  augment=config.augment,
                  finite_sample=config.finite_sample_p)
    return kw


def run_design(config: GEOXConfig) -> GEOXResults:
    """Run the full GEOX market-selection design from a config."""
    ekw = engine_settings(config)
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
            audit_geox_feasibility(
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
        engine=config.engine, engine_kwargs=ekw, alpha=config.alpha,
    )
    power_table = compute_power(cube, alpha=config.alpha)
    # The effect the design actually detects, solved instead of read off the
    # grid. Reported beside mde, which the ranking and the GeoLift
    # cross-validation both depend on and which is left alone.
    exact = compute_exact_mde(cube, power_threshold=config.power_threshold)
    # How far the estimate lands from an injected truth, beside how small an
    # effect the design can detect. Reported, and no part of the rank: the
    # composite is what the GeoLift cross-validation pins.
    accuracy = compute_accuracy(cube)
    shortlist = compute_rank(power_table,
                             power_threshold=config.power_threshold,
                             budget=config.budget)
    if not shortlist.empty and not exact.empty:
        shortlist = shortlist.merge(exact, on=["candidate", "duration"],
                                    how="left")
    if not shortlist.empty and not accuracy.empty:
        shortlist = shortlist.merge(accuracy, on=["candidate", "duration"],
                                    how="left")
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
                      exclude=excluded.get(c), how=config.how,
                      engine=config.engine, engine_kwargs=ekw)
        for c in candidates
    }

    # Stitch each candidate's best (lowest-rank) shortlist row into its design.
    stitched = ("rank", "mde", "power", "bias", "error_sd", "rmse",
                "calibration_ratio")
    best: Dict[frozenset, dict] = {}
    for _, row in shortlist.iterrows():
        cand = row["candidate"]
        if cand not in best or row["rank"] < best[cand]["rank"]:
            best[cand] = {name: row.get(name) for name in stitched}
    for cand, design in designs.items():
        match = best.get(cand)
        if match is None:
            continue
        for name in stitched:
            value = match[name]
            setattr(design, name,
                    None if value is None or pd.isna(value) else float(value))

    winner = None
    winner_units = None
    if not shortlist.empty:
        winning = shortlist.sort_values("rank").iloc[0]["candidate"]
        winner = designs[winning]
        winner_units = sorted(map(str, winning))
        # Scored only now, and only for the winner, so the backtests behind it
        # cannot have influenced which region was picked.
        held = planning_backtests(Ywide, winning, config,
                                  exclude=excluded.get(winning))
        readout = planning_readout(held, config)
        winner.mde_planning = readout.mde
        winner.rmse_planning = readout.rmse

    search = MarketSearch(shortlist=shortlist, power_table=power_table,
                          candidates=list(designs.values()), winner=winner)

    # A genuine post period means the experiment has run, so the design can be
    # read out. geoex_dataprep truncates to the pre-period when post_col is set,
    # so the design reproduces identically on a pre-only or a full panel; the
    # readout needs the periods that truncation removed, and re-preps without it.
    report = None
    if winner_units is not None and config.post_col is not None:
        full = geoex_dataprep(config.df, config.unitid, config.time,
                              config.outcome)["Ywide"]
        if full.shape[0] > n_periods:
            report = realize_design(
                full, winning, n_periods, how=config.how,
                exclude=excluded.get(winning), alpha=config.alpha,
                n_draws=config.n_draws, seed=config.seed, cpic=config.cpic,
            engine=config.engine, engine_kwargs=ekw)

    return GEOXResults(
        report=report,
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
            # What the winner gets wrong, on the same held-back backtests.
            "winner_rmse_planning": (
                float(winner.rmse_planning) if winner is not None
                and winner.rmse_planning is not None else None),
            "n_validation_backtests": config.n_validation_backtests,
        },
        search=search,
    )
