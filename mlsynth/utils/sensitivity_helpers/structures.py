"""Typed collections of native fits and extracted sensitivity trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field

from ...exceptions import MlsynthConfigError
from ..vanillasc_helpers.structures import VanillaSCResults
from .inference import (
    donohue,
    dube_zipperer,
    fp_band,
    joint_pvalue,
    permutation_pvalue,
    ratio_statistic,
)
from .numerics import median, std
from .placebo import PlaceboStack, stack
from .plotter import plot_specification_curve

_DIMENSIONS = {"donors", "fit_window", "fit_exclude", "covariates", "covariate_windows"}
IDENTIFICATION_COLUMNS = ["spec_id", "att", "att_min", "att_max", "identified"]


class PlaceboRuns(BaseModel):
    """Every design refit with each unit of its pool treated in turn.

    The actually treated unit stays in the placebo units' pools as an ordinary
    donor. ``paths`` and ``weights`` hold one block per design and unit, the
    treated unit's own block being its main fit; ``failures`` and ``warnings``
    log placebo fits that raised or warned. A unit with a failed fit has no
    statistic and is left out of every ranking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    paths: pd.DataFrame
    weights: pd.DataFrame
    failures: pd.DataFrame
    warnings: pd.DataFrame


@dataclass(frozen=True)
class FittedPaths:
    """Aligned, finite trajectories and donor weights for one specification.

    ``paths`` has one row per common study period; ``weights`` has one row
    per donor. Both tables include ``spec_id`` for concatenating attempts.
    """

    paths: pd.DataFrame
    weights: pd.DataFrame


class SensitivityResult(BaseModel):
    """A collection of unchanged estimator results and tidy sensitivity tables.

    ``summary`` has one row per attempt; ``failures`` and ``warnings`` preserve
    unsuccessful fits and diagnostics. ``fits`` retains the complete native
    results, including predictor weights and solver diagnostics. ``registry``
    contains resolved configurations without repeated panel data; ``baseline``
    preserves the original configuration, including a copy of the panel.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    summary: pd.DataFrame
    paths: pd.DataFrame
    weights: pd.DataFrame
    failures: pd.DataFrame
    warnings: pd.DataFrame
    registry: dict[str, dict[str, Any]]
    baseline: dict[str, Any]
    metadata: dict[str, Any]
    fits: dict[str, VanillaSCResults]
    identification: pd.DataFrame = Field(
        default_factory=lambda: pd.DataFrame(columns=IDENTIFICATION_COLUMNS),
        description="Per fitted design: the ATT and its range over every donor-weight "
        "vector that fits the design's matched quantities as well as the solved one. "
        "identified is False when that range has positive width: the design does not "
        "pin the weights, and the effect, down.",
    )
    placebo: Optional[PlaceboRuns] = Field(
        default=None, description="Placebo runs; present when the study ran with placebo=True."
    )

    def dispersion(self) -> pd.Series:
        """Describe finite estimates, including warning-bearing fits.

        Sign shares have the number of finite estimates as their denominator.
        They describe registered designs and are not causal probabilities.
        Standard deviation is NaN when fewer than two estimates are finite or
        its value exceeds the floating-point range.
        """
        effects = self.summary.loc[np.isfinite(self.summary.att), "att"]
        return pd.Series(
            {
                "attempted": len(self.summary),
                "estimated": len(effects),
                "failed": (self.summary.status == "failed").sum(),
                "with_warnings": (self.summary.status == "warning").sum(),
                "min": effects.min(),
                "median": median(effects),
                "max": effects.max(),
                "std": std(effects),
                "positive_share": (effects > 0).mean() if len(effects) else np.nan,
                "negative_share": (effects < 0).mean() if len(effects) else np.nan,
                "zero_share": (effects == 0).mean() if len(effects) else np.nan,
            }
        )

    def by_dimension(self, dimension: str) -> pd.DataFrame:
        """Group dispersion by one design choice, without attributing causality.

        Other dimensions can vary within each group. An unbalanced registry
        does not identify a variance contribution of the selected dimension.
        Failed designs remain in attempted/failed counts.
        """
        if dimension not in _DIMENSIONS:
            raise MlsynthConfigError(f"dimension must be one of {sorted(_DIMENSIONS)}")
        table = self.summary.copy()
        table.loc[~np.isfinite(table.att), "att"] = np.nan
        table["choice"] = [
            repr(self.registry[name].get(dimension)) for name in table.spec_id
        ]
        return (
            table.groupby("choice", sort=False, dropna=False, observed=True)
            .agg(
                attempted=("spec_id", "size"),
                estimated=("att", "count"),
                failed=("status", lambda x: (x == "failed").sum()),
                min=("att", "min"),
                median=("att", lambda values: median(values.dropna())),
                max=("att", "max"),
            )
            .reset_index()
        )

    def plot(self) -> Figure:
        """Return a specification-curve Figure displaying every attempted design."""
        return plot_specification_curve(self.summary)

    # --- inference across designs (Ferman, Pinto and Possebom 2020) ---------

    def _runs(self) -> PlaceboRuns:
        if self.placebo is None:
            raise MlsynthConfigError(
                "This study ran without placebo runs; refit it with placebo=True."
            )
        return self.placebo

    def _designs(self, designs: Optional[Sequence[str]]) -> list:
        fitted = list(dict.fromkeys(self._runs().paths.spec_id))
        if designs is None:
            return fitted
        unknown = set(designs) - set(fitted)
        if unknown:
            raise MlsynthConfigError(f"No placebo runs for designs {sorted(unknown)}.")
        return list(designs)

    def _stack(self, designs: Sequence[str]) -> PlaceboStack:
        runs, base = self._runs(), self.baseline
        return stack(runs.paths, runs.weights, runs.failures, base["df"],
                     base["unitid"], base["time"], base["outcome"],
                     self.metadata["treated_unit"], designs)

    def _windows(self, s: PlaceboStack, pre_periods, post_periods):
        pre = self.metadata["pre_periods"] if pre_periods is None else pre_periods
        post = self.metadata["att_periods"] if post_periods is None else post_periods
        return s.mask(pre), s.mask(post)

    def placebo_test(self, pre_periods: Optional[Sequence[Any]] = None,
                     post_periods: Optional[Sequence[Any]] = None,
                     root: bool = False) -> pd.DataFrame:
        """One placebo test per design: the treated unit's post/pre statistic and its rank.

        The statistic is the post-period over the pre-period mean squared gap
        (``root`` takes its square root, which leaves the ranking unchanged);
        the periods default to the study's pre-treatment and ATT periods. The
        p-value is the share of the design's units whose statistic is at least
        the treated unit's, over the ``n_units`` that have one. Reading one
        design's p-value after looking at several overstates significance; the
        joint test is the one to report.
        """
        rows = []
        for spec in self._designs(None):
            s = self._stack([spec])
            pre, post = self._windows(s, pre_periods, post_periods)
            stat = ratio_statistic(s.gaps[0], pre, post, root=root)
            treated = stat[s.position]
            rows.append(dict(
                spec_id=spec, statistic=treated,
                p_value=permutation_pvalue(stat, s.position) if np.isfinite(treated) else np.nan,
                n_units=int(np.isfinite(stat).sum()),
            ))
        return pd.DataFrame(rows).set_index("spec_id")

    def joint_test(self, designs: Optional[Sequence[str]] = None,
                   pre_periods: Optional[Sequence[Any]] = None,
                   post_periods: Optional[Sequence[Any]] = None,
                   root: bool = False) -> pd.Series:
        """Ferman, Pinto and Possebom's test combining the designs.

        Each unit's statistic is averaged across ``designs`` (default: every
        design with placebo runs), and the treated unit's average is ranked
        among the units'. The designs must share one unit set.
        """
        s = self._stack(self._designs(designs))
        pre, post = self._windows(s, pre_periods, post_periods)
        stats = np.column_stack([ratio_statistic(G, pre, post, root=root) for G in s.gaps])
        n_units = int(np.isfinite(stats).any(axis=1).sum())
        return pd.Series({"p_value": joint_pvalue(stats, s.position),
                          "n_units": n_units, "n_designs": len(s.designs)})

    def confidence_band(self, designs: Optional[Sequence[str]] = None,
                        significance: float = 0.1, precision: int = 30,
                        initial_step: float = 1.0,
                        pre_periods: Optional[Sequence[Any]] = None,
                        post_periods: Optional[Sequence[Any]] = None) -> pd.DataFrame:
        """Firpo-Possebom band around the average effect across ``designs``.

        It holds the effect paths, the average effect shifted by a constant,
        that the joint test does not reject at ``significance`` (a null is
        rejected when its p-value is at most ``significance``; with ``n`` units
        the attainable levels are multiples of ``1/n``). ``initial_step`` sets
        the scale of the search and should be of the outcome's order. Columns
        ``lower`` and ``upper`` are missing before treatment.
        """
        s = self._stack(self._designs(designs))
        pre, post = self._windows(s, pre_periods, post_periods)
        rows = np.flatnonzero(pre | post)
        band = fp_band(s.outcomes[rows], s.weights, s.position, n_pre=int(pre.sum()),
                       significance=significance, precision=precision,
                       initial_step=initial_step)
        index = pd.Index([s.times[r] for r in rows], name="time")
        return pd.DataFrame({"average": band.average, "lower": band.lower,
                             "upper": band.upper}, index=index)

    def choose(self, rule: str, designs: Optional[Sequence[str]] = None,
               window: Optional[Sequence[Any]] = None,
               post_periods: Optional[Sequence[Any]] = None) -> pd.Series:
        """Rank designs by an objective rule; the first entry is the one chosen.

        ``"dube-zipperer"``: the placebo units' mean squared gap over the ATT
        periods (Dube and Zipperer 2015). ``"donohue"``: the treated unit's mean
        squared gap over ``window``, pre-treatment periods held out of the fit
        (Donohue, Aneja and Weber 2019).
        """
        if rule not in ("dube-zipperer", "donohue"):
            raise MlsynthConfigError("rule must be 'dube-zipperer' or 'donohue'.")
        if rule == "donohue" and window is None:
            raise MlsynthConfigError("The donohue rule needs a validation window.")
        values = {}
        for spec in self._designs(designs):
            s = self._stack([spec])
            if rule == "dube-zipperer":
                _, post = self._windows(s, None, post_periods)
                values[spec] = dube_zipperer(s.gaps, post, s.position)[0]
            else:
                values[spec] = donohue(s.gaps, s.mask(window), s.position)[0]
        return pd.Series(values, name=rule).sort_values()
