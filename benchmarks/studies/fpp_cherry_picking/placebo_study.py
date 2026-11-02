"""PR #958's SpecificationStudy, refit with every unit as the treated one.

FPP's inference needs, for each registered design, the post/pre statistic of
every unit in the panel, the treated unit included as an ordinary donor in the
placebo units' pools (ADH 2010: "shifting California to the donor pool"). The
PR's runner fits one treated unit. This wrapper moves the treatment indicator
to each unit in turn and runs one SpecificationStudy per unit, unchanged, so
every placebo fit goes through the PR's validation, failure log and fixed
backend. It returns the gaps as ``(T, N)`` arrays per design, the input every
function in :mod:`fpp` takes.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import mlsynth
import mlsynth.sensitivity as sensitivity
from mlsynth import VanillaSC
from mlsynth.sensitivity import SensitivityResult, SpecificationStudy

# SpecificationStudy.fit() records importlib.metadata.version("mlsynth"), which
# raises when mlsynth runs from a source checkout that was never pip-installed
# (this spike runs from the repository tree). mlsynth/__init__.py already falls
# back for __version__; the PR's module does not.
sensitivity.version = lambda _name: mlsynth.__version__


@dataclass
class PlaceboSpecificationResult:
    units: List
    times: np.ndarray
    treated: object
    per_unit: Dict[object, SensitivityResult]

    def gaps(self, spec: str) -> np.ndarray:
        """``(T, N)`` gap matrix for one design; NaN columns for failed fits."""
        G = np.full((len(self.times), len(self.units)), np.nan)
        for j, u in enumerate(self.units):
            paths = self.per_unit[u].paths
            g = paths.loc[paths.spec_id == spec].set_index("time")["gap"]
            if len(g):
                G[:, j] = g.reindex(self.times).to_numpy(float)
        return G

    def failures(self) -> pd.DataFrame:
        frames = [r.failures.assign(unit=u) for u, r in self.per_unit.items()
                  if len(r.failures)]
        return pd.concat(frames) if frames else pd.DataFrame()


def _fit_unit(args):
    unit, panel, base, specs, att_periods, unitid, time, treat_start = args
    df = panel.copy()
    df["placebo_treated"] = ((df[unitid] == unit) & (df[time] >= treat_start)).astype(int)
    cfg = dict(base, df=df, treat="placebo_treated")
    study = SpecificationStudy(VanillaSC, cfg, specs, att_periods)
    return unit, study.fit()


class PlaceboSpecificationStudy:
    """Run SpecificationStudy once per unit, each unit treated in turn.

    ``base_config`` is a SpecificationStudy base config without ``df`` and
    ``treat``; ``treat_start`` is the first treated period. The baseline design
    is whatever ``base_config`` describes, exactly as in SpecificationStudy.
    """

    def __init__(self, panel: pd.DataFrame, base_config: dict,
                 specifications: Sequence[dict], att_periods: Sequence,
                 treated, treat_start, units: Sequence = None):
        self.panel = panel
        self.base = dict(base_config)
        self.specs = [dict(s) for s in specifications]
        self.att_periods = list(att_periods)
        self.treated = treated
        self.treat_start = treat_start
        uid = self.base["unitid"]
        self.units = list(units) if units is not None else sorted(panel[uid].unique())

    def fit(self, processes: int = 1) -> PlaceboSpecificationResult:
        uid, tcol = self.base["unitid"], self.base["time"]
        jobs = [(u, self.panel, self.base, self.specs, self.att_periods, uid, tcol,
                 self.treat_start) for u in self.units]
        if processes > 1:
            with ProcessPoolExecutor(processes) as pool:
                done = dict(pool.map(_fit_unit, jobs))
        else:
            done = dict(map(_fit_unit, jobs))
        times = np.array(sorted(self.panel[tcol].unique()))
        return PlaceboSpecificationResult(self.units, times, self.treated, done)
