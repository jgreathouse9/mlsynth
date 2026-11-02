"""Ferman, Pinto & Possebom (2020), "Cherry Picking with Synthetic Controls".

Spike module: the paper's empirical procedure built on top of
:class:`mlsynth.sensitivity.SpecificationStudy` (PR #958). Nothing here is wired
into mlsynth. Three parts:

1. Specification registries for the two applications in the authors'
   replication package -- West Germany (Abadie, Diamond & Hainmueller 2015,
   main text Table 3) and California (ADH 2010, Appendix F Table F2) -- with
   each predictor written as ``(column, years)`` exactly as the authors'
   ``dataprep`` calls build it.
2. A port of the two Synth steps the authors use: predictor construction
   (``dataprep``: window means with missing values dropped, then each row scaled
   to unit variance across all units) and the donor weights for a given
   predictor-weight vector V (``synth(custom.v = ...)``), solved with mlsynth's
   own lower-level QP.
3. The inference FPP build from per-specification placebo runs: the post/pre
   ratio statistic and its permutation p-value, the joint test on the mean
   statistic across specifications, the pre-treatment fit index R-tilde
   (their eq. D.1), the Dube-Zipperer and Donohue-Aneja-Weber choice rules, and
   the Firpo-Possebom (2018) confidence band around the average effect.

Labels follow the authors' code. Their code and the paper disagree on what
"a" and "b" mean; :data:`GERMANY_PAPER_LABEL` and
:data:`CALIFORNIA_PAPER_LABEL` map code labels to the paper's tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from mlsynth.utils.bilevel.solver import lower_level_weights
from mlsynth.utils.bilevel.structure import BilevelProblem

Row = Tuple[str, Tuple[int, ...]]  # (column, years averaged over)


# ---------------------------------------------------------------------------
# Specification registries
# ---------------------------------------------------------------------------

def _years(a: int, b: int, step: int = 1) -> Tuple[int, ...]:
    return tuple(range(a, b + 1, step))


@dataclass(frozen=True)
class Design:
    """One specification: the predictor rows of each Synth step.

    ``train`` and ``train_ssr`` define the problem V is chosen on. ``main`` is
    the second step of ADH (2015)'s cross-validation (the same V, predictors
    re-measured); ``None`` for a single-step design, where the weights come from
    the ``train`` problem itself.
    """

    name: str
    train: Tuple[Row, ...]
    train_ssr: Tuple[int, ...]
    main: Optional[Tuple[Row, ...]] = None
    main_ssr: Optional[Tuple[int, ...]] = None
    covariates: bool = False

    def registry_entry(self) -> dict:
        """The design as a SpecificationStudy override (training problem).

        SpecificationStudy averages each covariate over one window per column,
        so every row becomes its own column named ``<column>__<first>_<last>``;
        :func:`add_window_columns` creates those columns in the panel.
        """
        names = [window_column(c, ys) for c, ys in self.train]
        return {
            "name": self.name,
            "covariates": names,
            "covariate_windows": {
                window_column(c, ys): (min(ys), max(ys)) for c, ys in self.train
            },
            "fit_window": (min(self.train_ssr), max(self.train_ssr)),
        }


def window_column(column: str, years: Sequence[int]) -> str:
    years = tuple(years)
    return f"{column}__{years[0]}_{years[-1]}" if len(years) > 1 else f"{column}__{years[0]}"


def add_window_columns(panel: pd.DataFrame, unit: str, time: str,
                       designs: Sequence[Design]) -> pd.DataFrame:
    """Copy each predictor column under every name a design's rows use.

    A row that averages over non-contiguous years (FPP's ``schooling`` over 1970
    and 1975) is pre-averaged into a column constant within unit, because a
    SpecificationStudy window is a contiguous ``(start, end)``.
    """
    out = panel.copy()
    for d in designs:
        for column, years in d.train:
            name = window_column(column, years)
            if name in out:
                continue
            contiguous = tuple(years) == _years(min(years), max(years))
            if contiguous:
                out[name] = out[column]
            else:
                means = (out[out[time].isin(years)].groupby(unit)[column].mean())
                out[name] = out[unit].map(means)
    return out


GERMANY_TREATED = 7          # ``index`` of West Germany in repgermany.dta
GERMANY_LAGS = {
    "1": _years(1971, 1980),
    "2": _years(1971, 1978),
    "3": _years(1971, 1975),
    "4": _years(1971, 1979, 2),
    "5": _years(1972, 1980, 2),
    "7": (1971, 1975, 1980),
}
_TRAIN, _VALID, _MAIN = _years(1971, 1980), _years(1981, 1990), _years(1981, 1990)


def germany_designs() -> List[Design]:
    """The 14 West Germany specifications of ``replication_germany.R``.

    Code ``a`` carries ADH (2015)'s five covariates, code ``b`` does not. The
    training step averages covariates over 1971-1980 and chooses V on
    1981-1990; the main step re-measures them over 1981-1990 (schooling over
    1980 and 1985, investment from the 1980 row), keeps the outcome lags at
    their 1971-1980 values, and reuses V.
    """
    cov_train = (("trade", _TRAIN), ("infrate", _TRAIN), ("industry", _TRAIN),
                 ("schooling", (1970, 1975)), ("invest70", (1980,)))
    cov_main = (("trade", _MAIN), ("infrate", _MAIN), ("industry", _MAIN),
                ("schooling", (1980, 1985)), ("invest80", (1980,)))
    designs = []
    for k in ("1", "2", "3", "4", "5", "6", "7"):
        for with_cov in (True, False):
            name = k + ("a" if with_cov else "b")
            if k == "6":
                # The outcome enters as an ordinary predictor averaged over the
                # step's own window, so it moves with the window.
                if with_cov:
                    train = (("gdp", _TRAIN), ("trade", _TRAIN), ("infrate", _TRAIN),
                             cov_train[2], cov_train[3], cov_train[4])
                    main = (("gdp", _MAIN), ("trade", _MAIN), ("infrate", _MAIN),
                            cov_main[2], cov_main[3], cov_main[4])
                else:
                    train, main = (("gdp", _TRAIN),), (("gdp", _MAIN),)
            else:
                lags = tuple(("gdp", (y,)) for y in GERMANY_LAGS[k])
                train = (cov_train + lags) if with_cov else lags
                main = (cov_main + lags) if with_cov else lags
            designs.append(Design(name, train, _VALID, main, _years(1960, 1989),
                                  covariates=with_cov))
    return designs


# Code label -> paper Table 3 label (the paper's "a" is "no covariates").
GERMANY_PAPER_LABEL = {f"{k}{c}": f"{k}{'b' if c == 'a' else 'a'}"
                       for k in "1234567" for c in "ab"}

CALIFORNIA_TREATED = "California"
CALIFORNIA_LAGS = {
    "a": (1975, 1980, 1988),                      # ADH (2010)'s own three lags
    "b": None,                                    # the 1970-1988 mean
    "c": _years(1970, 1988),
    "d": _years(1970, 1979),
    "e": _years(1970, 1984),
    "f": (1971, 1973, 1975, 1977, 1979, 1981, 1983, 1985, 1986, 1987),
    "g": _years(1970, 1988, 2),
}
# Code letter -> paper specification number; digit 1 carries covariates. "Odd"
# and "even" follow the calendar year in the authors' code (f: 1971, 1973, ...;
# g: 1970, 1972, ...), not the period index of the paper's formula; Table F2's
# p-values settle it (f gives the printed 4a/4b, g the printed 5a/5b).
_CA_NUMBER = {"a": "7", "b": "6", "c": "1", "d": "3", "e": "2", "f": "4", "g": "5"}
CALIFORNIA_PAPER_LABEL = {f"{l}{d}": _CA_NUMBER[l] + ("b" if d == "1" else "a")
                          for l in _CA_NUMBER for d in "12"}


def california_designs() -> List[Design]:
    """The 14 California specifications of ``replication_california.R``.

    Single-step: covariates averaged over 1980-1988 (``lnincome``, ``beer``,
    ``age15to24``, ``retprice``), V chosen on 1970-1988. Lag set ``f`` (odd
    years) is reproduced as the authors wrote it, with 1986 in it.
    """
    covs = tuple((c, _years(1980, 1988)) for c in
                 ("lnincome", "beer", "age15to24", "retprice"))
    designs = []
    for letter, lags in CALIFORNIA_LAGS.items():
        rows = ((("cigsale", _years(1970, 1988)),) if lags is None
                else tuple(("cigsale", (y,)) for y in lags))
        for digit in ("1", "2"):
            train = (covs + rows) if digit == "1" else rows
            designs.append(Design(letter + digit, train, _years(1970, 1988),
                                  covariates=digit == "1"))
    return designs


# ---------------------------------------------------------------------------
# Synth port
# ---------------------------------------------------------------------------

@dataclass
class Panel:
    """Wide outcome matrix and long panel for one application."""

    long: pd.DataFrame
    unit: str
    time: str
    outcome: str
    units: List           # unit labels in the authors' order
    times: np.ndarray     # sorted time labels

    @property
    def Y(self) -> np.ndarray:
        wide = self.long.pivot(index=self.time, columns=self.unit, values=self.outcome)
        return wide.loc[self.times, self.units].to_numpy(float)


def predictor_matrix(panel: Panel, rows: Sequence[Row]) -> np.ndarray:
    """``(K, N)`` window means in ``panel.units`` order, NaNs dropped (Synth)."""
    out = np.empty((len(rows), len(panel.units)))
    for k, (column, years) in enumerate(rows):
        sub = panel.long[panel.long[panel.time].isin(years)]
        out[k] = sub.groupby(panel.unit)[column].mean().reindex(panel.units).to_numpy(float)
    return out


def synth_scale(X: np.ndarray) -> np.ndarray:
    """Synth's ``divisor``: each predictor row over its SD across all units."""
    return X / X.std(axis=1, ddof=1, keepdims=True)


def custom_v_weights(Xs: np.ndarray, v: np.ndarray, pos: int,
                     Y_ssr: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Donor weights for unit ``pos`` given V: Synth's ``custom.v`` step.

    ``argmin_w (X1 - X0 w)' V (X1 - X0 w)`` on the simplex, solved by mlsynth's
    exact active-set QP. The outcome-fit term mlsynth adds as a tie-break is
    scaled to order one and weighted ``eps``, so it only chooses among weights
    with the same predictor loss. Donors are every other unit, in order.
    """
    others = [j for j in range(Xs.shape[1]) if j != pos]
    scale = max(1.0, float(np.abs(Y_ssr).max()))
    prob = BilevelProblem(y1_pre=Y_ssr[:, pos] / scale,
                          Y0_pre=Y_ssr[:, others] / scale,
                          X1=Xs[:, pos], X0=Xs[:, others])
    return lower_level_weights(prob, np.asarray(v, float), eps=eps)


def predictor_loss(Xs: np.ndarray, v: np.ndarray, pos: int, w: np.ndarray) -> float:
    others = [j for j in range(Xs.shape[1]) if j != pos]
    r = Xs[:, pos] - Xs[:, others] @ w
    v = np.asarray(v, float) / np.sum(v)
    return float(r @ (v * r))


def gaps_from_weights(Y: np.ndarray, weights: Dict[int, np.ndarray]) -> np.ndarray:
    """``(T, N)`` gaps: unit ``j``'s outcome minus its synthetic control."""
    T, N = Y.shape
    G = np.full((T, N), np.nan)
    for j, w in weights.items():
        others = [k for k in range(N) if k != j]
        G[:, j] = Y[:, j] - Y[:, others] @ w
    return G


# ---------------------------------------------------------------------------
# FPP inference on placebo runs
# ---------------------------------------------------------------------------

def ratio_statistic(G: np.ndarray, pre: np.ndarray, post: np.ndarray,
                    kind: str = "mse") -> np.ndarray:
    """Post/pre statistic per unit: ``"mse"`` (eq. 3) or ``"rmse"`` (its root)."""
    pre_ms = np.mean(G[pre] ** 2, axis=0)
    post_ms = np.mean(G[post] ** 2, axis=0)
    ratio = post_ms / pre_ms
    return np.sqrt(ratio) if kind == "rmse" else ratio


def placebo_pvalue(stat: np.ndarray, pos: int, denominator: str = "all") -> float:
    """R's ``rank(-stat)[pos] / N``: average ranks for ties, NaN ranked last.

    A unit without a statistic (a fit the script skipped or that failed) is
    ranked last. ``denominator="all"`` divides by every unit (the California
    script), ``"finite"`` by the units with a statistic (the Germany script's
    specification 7a, which skips Austria and divides by 16).
    """
    s = np.asarray(stat, float)
    ranks = np.empty(len(s))
    finite = np.isfinite(s)
    ranks[finite] = rankdata(-s[finite], method="average")
    ranks[~finite] = finite.sum() + rankdata(np.flatnonzero(~finite))
    n = len(s) if denominator == "all" else int(finite.sum())
    return float(ranks[pos] / n)


def joint_pvalue(stats: Dict[str, np.ndarray], specs: Sequence[str], pos: int) -> float:
    """Permutation p-value of the mean statistic across ``specs`` (rowMeans, NaN dropped)."""
    M = np.column_stack([stats[s] for s in specs])
    return placebo_pvalue(np.nanmean(M, axis=1), pos)


def r_tilde(G: np.ndarray, Y: np.ndarray, pre: np.ndarray, pos: int) -> float:
    """FPP eq. (D.1): one minus pre-period SSR over the treated unit's own variation."""
    y = Y[pre, pos]
    return float(1.0 - np.sum(G[pre, pos] ** 2) / np.sum((y - y.mean()) ** 2))


def dube_zipperer(G_by_spec: Dict[str, np.ndarray], post: np.ndarray, pos: int,
                  candidates: Sequence[str]) -> Dict[str, float]:
    """Mean post-period MSPE of the placebo units per specification (pick the min)."""
    out = {}
    for s in candidates:
        G = G_by_spec[s]
        placebos = [j for j in range(G.shape[1]) if j != pos and np.isfinite(G[:, j]).all()]
        out[s] = float(np.mean([np.mean(G[post, j] ** 2) for j in placebos]))
    return out


def donohue(G_by_spec: Dict[str, np.ndarray], window: np.ndarray, pos: int,
            candidates: Sequence[str]) -> Dict[str, float]:
    """The treated unit's MSPE over a validation window per specification."""
    return {s: float(np.mean(G_by_spec[s][window, pos] ** 2)) for s in candidates}


@dataclass
class Band:
    average: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    lb: float
    ub: float


def fp_band(Y: np.ndarray, weights_by_spec: Dict[str, Dict[int, np.ndarray]],
            average_specs: Sequence[str], test_specs: Sequence[str], pos: int,
            T0: int, significance: float, precision: int = 30) -> Band:
    """Firpo-Possebom (2018) band around the average effect, as ``cs_germany.R``.

    Nulls are the average post-treatment effect across ``average_specs`` shifted
    by a constant ``c``. Under each null the treated unit's outcome is netted of
    the hypothesised effect, in its own statistic and wherever it serves as a
    placebo's donor; weights stay fixed (they depend on pre-treatment data
    only). The statistic is the mean MSE ratio across ``test_specs``, rows
    ``[:T0]`` pre and ``[T0:]`` post. The bounds are found by the authors'
    step-halving search, ``precision + 1`` passes starting from ``c = 0``.
    """
    T, N = Y.shape
    postp = np.r_[np.zeros(T0), np.ones(T - T0)]
    own = {s: Y[:, pos] - Y[:, [k for k in range(N) if k != pos]] @ weights_by_spec[s][pos]
           for s in average_specs}
    avg = np.mean([own[s] for s in average_specs], axis=0)
    nh0 = np.r_[np.zeros(T0), avg[T0:]]

    def pvalue(c: float) -> float:
        nh = nh0 + c * postp
        Yn = Y.copy()
        Yn[:, pos] = Y[:, pos] - nh  # the treated unit's outcome under the null
        stat = np.zeros(N)
        for s in test_specs:
            for j in range(N):
                others = [k for k in range(N) if k != j]
                g = Yn[:, j] - Yn[:, others] @ weights_by_spec[s][j]
                stat[j] += np.mean(g[T0:] ** 2) / np.mean(g[:T0] ** 2)
        stat /= len(test_specs)
        r = rankdata(stat)
        return float(np.mean(r >= r[pos]))

    ub = lb = 0.0
    for power in range(precision + 1):
        step = 0.5 ** power
        while True:
            if pvalue(ub) <= significance:
                ub -= step
                break
            ub += step
        while True:
            if pvalue(lb) <= significance:
                lb += step
                break
            lb -= step
    return Band(avg, nh0 + lb * postp, nh0 + ub * postp, lb, ub)
