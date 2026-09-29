"""The mlsynth side, wrapped to the same shape as :mod:`reference`.

Both sides expose a scorer of the form ``(treatment, control) -> key`` and a
``designs`` mapping keyed by treatment size, so the comparison code never has to
know which implementation it is holding.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from mlsynth import TBRMM
from mlsynth.config_models import TBRMMConfig
from mlsynth.utils.tbrmm_helpers.objective import score_split

Key = Tuple[float, ...]


def scoring_window(panel: pd.DataFrame, n_pretest: int) -> pd.DataFrame:
    """The last ``n_pretest`` periods, wide, periods by geos.

    The reference truncates inside its constructor, so the same truncation is
    applied here; scoring a longer window would compare two different panels.
    """
    dates = sorted(panel["date"].unique())[-n_pretest:]
    wide = panel[panel["date"].isin(dates)].pivot(
        index="date", columns="geo", values="Y").sort_index()
    return wide[sorted(wide.columns)]


def long_window(panel: pd.DataFrame, n_pretest: int) -> pd.DataFrame:
    """The same truncation, left long, for feeding the estimator."""
    dates = sorted(panel["date"].unique())[-n_pretest:]
    return panel[panel["date"].isin(dates)]


def aggregate(wide: pd.DataFrame, geos: Sequence[str]) -> np.ndarray:
    """A group's aggregate series: the unweighted sum over its members."""
    return wide[sorted(geos)].sum(axis=1).to_numpy(dtype=float)


def score(wide: pd.DataFrame, treatment: Sequence[str], control: Sequence[str],
          *, n_test: int) -> Key:
    """mlsynth's score for one split, as a plain tuple."""
    key = score_split(aggregate(wide, treatment), aggregate(wide, control),
                      objective="reference", n_test=n_test).key
    return tuple(float(v) for v in key)


def designs(panel: pd.DataFrame, *, n_test: int, k_max: int,
            n_pretest: int) -> Dict[int, Tuple[List[str], List[str], Key]]:
    """One recommended design per treatment size, keyed by size."""
    result = TBRMM(TBRMMConfig(
        df=long_window(panel, n_pretest), unitid="geo", time="date",
        outcome="Y", max_treatment_size=k_max, n_test=n_test)).fit()
    out: Dict[int, Tuple[List[str], List[str], Key]] = {}
    for design in result.designs:
        detail = design.detail
        key = (float(detail["corr_test"]), float(detail["aa_test"]),
               float(detail["bb_test"]), float(detail["dw_test"]),
               round(float(detail["corr"]), 2), float(design.objective_value))
        out[design.k] = (sorted(design.treatment_units),
                         sorted(design.control_units), key)
    return out
