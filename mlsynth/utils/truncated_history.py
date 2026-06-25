"""Truncated History (TH) robustness diagnostic for synthetic-control estimators.

> Spoelstra, P., Stolp, T., Golsteyn, B.H.H., Cornelisz, I., & van Klaveren, C.
> (2025). "Truncated History Framework for Synthetic Control Approaches."
> *Economics Letters* 257, 112701.

TH re-estimates a synthetic-control effect on *truncated pre-treatment windows*
and reports how the ATT (and the pre-fit / inference) move with the pretreatment
horizon. It is a robustness check, not an estimator: it re-runs any mlsynth
estimator that returns :class:`~mlsynth.config_models.BaseEstimatorResults`.

Modes
-----
* ``"left"``   -- drop the EARLIEST pre-periods, one at a time (the paper's
  left-TH); the post-period is kept intact.
* ``"right"``  -- drop the LATEST pre-periods (the in-time placebo direction).
* ``"loo"``    -- leave one pre-period out.
* ``"l2o"``    -- leave two pre-periods out (all pairs).
* ``"random"`` -- drop a random subset of pre-periods, repeatedly (seeded).

Stable results across truncations support a causal reading; instability says a
point estimate is fragile and an interval is the more honest summary.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError

_MODES = ("left", "right", "loo", "l2o", "random")


class TruncatedHistoryWindow(BaseModel):
    """One truncated-window re-estimate."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(..., description="Human label for the window (e.g. '1972-1988' or 'drop 1975').")
    n_pre_periods: int = Field(..., description="Number of pre-treatment periods kept.")
    att: float = Field(..., description="ATT estimated on the truncated panel.")
    pre_mspe: Optional[float] = Field(None, description="Pre-treatment MSPE (rmse_pre^2), if the estimator reports it.")
    p_value: Optional[float] = Field(None, description="Inference p-value (e.g. in-space placebo), if present.")
    standard_error: Optional[float] = Field(None, description="ATT standard error, if present.")


class TruncatedHistoryResult(BaseModel):
    """The TH stability profile across truncated pre-treatment windows."""

    model_config = ConfigDict(frozen=True)

    mode: str
    att_full: float = Field(..., description="ATT on the full (untruncated) pre-period.")
    profile: List[TruncatedHistoryWindow]
    att_min: float
    att_max: float
    att_mean: float
    stable: bool = Field(..., description="True if the ATT is sign-consistent and its spread is small relative to its mean.")
    stability_note: str


def _pre_post(df: pd.DataFrame, time: str, treat: str) -> Tuple[List[Any], Any]:
    """Return ``(sorted pre-period time labels, treatment-start time)``."""
    treated = df[df[treat] == 1]
    if treated.empty:
        raise MlsynthDataError("Truncated History needs at least one treated row "
                               f"(no rows with {treat}==1).")
    treat_start = treated[time].min()
    pre = sorted(t for t in df[time].unique() if t < treat_start)
    if not pre:
        raise MlsynthDataError("Truncated History found no pre-treatment periods.")
    return pre, treat_start


def _windows(mode: str, pre: List[Any], min_pre: int, n_random: int,
             rng: np.random.Generator) -> List[Tuple[str, set]]:
    """Return a list of ``(label, dropped_pre_period_set)`` for the mode."""
    out: List[Tuple[str, set]] = []
    if mode == "left":
        # drop the earliest k periods, k = 1 .. len(pre)-min_pre
        for k in range(1, len(pre) - min_pre + 1):
            dropped = set(pre[:k])
            out.append((f"{pre[k]}-{pre[-1]}", dropped))
    elif mode == "right":
        for k in range(1, len(pre) - min_pre + 1):
            dropped = set(pre[-k:])
            out.append((f"{pre[0]}-{pre[-k - 1]}", dropped))
    elif mode == "loo":
        if len(pre) - 1 < min_pre:  # pragma: no cover - subsumed by the min_pre+1 guard in truncated_history()
            return out
        for t in pre:
            out.append((f"drop {t}", {t}))
    elif mode == "l2o":
        if len(pre) - 2 < min_pre:
            return out
        for a, b in itertools.combinations(pre, 2):
            out.append((f"drop {a},{b}", {a, b}))
    elif mode == "random":
        max_drop = len(pre) - min_pre
        seen = set()
        for _ in range(n_random):
            k = int(rng.integers(1, max_drop + 1))
            idx = tuple(sorted(rng.choice(len(pre), size=k, replace=False).tolist()))
            if idx in seen:
                continue
            seen.add(idx)
            dropped = {pre[i] for i in idx}
            out.append((f"drop {sorted(dropped)}", dropped))
    return out


def _extract(res) -> Tuple[float, Optional[float], Optional[float], Optional[float]]:
    """Pull (att, pre_mspe, p_value, standard_error) from an estimator result."""
    att = float(res.att)
    mspe = None
    if res.fit_diagnostics is not None and res.fit_diagnostics.rmse_pre is not None:
        mspe = float(res.fit_diagnostics.rmse_pre) ** 2
    p = se = None
    if res.inference is not None:
        if res.inference.p_value is not None:
            p = float(res.inference.p_value)
        if res.inference.standard_error is not None:
            se = float(res.inference.standard_error)
    return att, mspe, p, se


def truncated_history(
    estimator: Any,
    config: Dict[str, Any],
    *,
    mode: str = "left",
    min_pre: int = 2,
    n_random: int = 20,
    seed: int = 0,
    stability_tol: float = 0.25,
) -> TruncatedHistoryResult:
    """Run the Truncated History robustness check on an mlsynth estimator.

    Parameters
    ----------
    estimator
        An mlsynth estimator class (anything callable as ``estimator(config)``
        whose ``.fit()`` returns a ``BaseEstimatorResults``), e.g. ``VanillaSC``
        or ``SDID``.
    config
        The estimator's config dict, including ``df`` and the ``time`` / ``treat``
        / ``unitid`` / ``outcome`` keys. The panel is truncated per window and the
        estimator re-run; ``display_graphs`` is forced off.
    mode
        One of ``"left"``, ``"right"``, ``"loo"``, ``"l2o"``, ``"random"``.
    min_pre
        Minimum number of pre-treatment periods any truncated window must retain.
    n_random
        Number of random draws for ``mode="random"``.
    seed
        RNG seed for ``mode="random"``.
    stability_tol
        The ATT is flagged ``stable`` when it keeps its sign across all windows
        and its spread ``(max-min)`` is at most ``stability_tol`` times the mean
        magnitude.

    Returns
    -------
    TruncatedHistoryResult
        The full-sample ATT, the per-window profile, and a stability verdict.
    """
    if mode not in _MODES:
        raise MlsynthConfigError(f"mode must be one of {_MODES}; got '{mode}'.")
    if not isinstance(config, dict) or "df" not in config:
        raise MlsynthConfigError("config must be a dict containing the panel 'df'.")
    for key in ("time", "treat", "unitid", "outcome"):
        if key not in config:
            raise MlsynthConfigError(f"config is missing the '{key}' column name.")
    if min_pre < 1:
        raise MlsynthConfigError(f"min_pre must be >= 1; got {min_pre}.")

    df = config["df"]
    time, treat = config["time"], config["treat"]
    pre, _ = _pre_post(df, time, treat)
    if len(pre) < min_pre + 1:
        raise MlsynthConfigError(
            f"need more than min_pre={min_pre} pre-periods to truncate; "
            f"the panel has {len(pre)}.")

    def _fit(frame: pd.DataFrame):
        cfg = {**config, "df": frame, "display_graphs": False}
        try:
            return estimator(cfg).fit()
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:  # pragma: no cover - estimator-specific failure
            raise MlsynthEstimationError(
                f"Truncated History: estimator failed on a window: {exc}") from exc

    att_full = float(_fit(df).att)

    rng = np.random.default_rng(seed)
    windows = _windows(mode, pre, min_pre, n_random, rng)
    if not windows:
        raise MlsynthConfigError(
            f"mode='{mode}' with min_pre={min_pre} yields no truncated windows "
            f"for {len(pre)} pre-periods.")

    profile: List[TruncatedHistoryWindow] = []
    for label, dropped in windows:
        kept = df[~df[time].isin(dropped)]
        att, mspe, p, se = _extract(_fit(kept))
        profile.append(TruncatedHistoryWindow(
            label=label, n_pre_periods=len(pre) - len(dropped),
            att=att, pre_mspe=mspe, p_value=p, standard_error=se))

    atts = np.array([w.att for w in profile] + [att_full], dtype=float)
    a_min, a_max, a_mean = float(atts.min()), float(atts.max()), float(atts.mean())
    sign_consistent = np.all(atts > 0) or np.all(atts < 0)
    spread = a_max - a_min
    tol_ok = spread <= stability_tol * abs(a_mean) if a_mean != 0 else spread == 0
    stable = bool(sign_consistent and tol_ok)
    note = (f"ATT in [{a_min:.3g}, {a_max:.3g}] across {len(profile)} {mode}-truncations "
            f"(spread {spread:.3g}, {spread / abs(a_mean) * 100:.0f}% of mean); "
            + ("sign-consistent and tight -> stable." if stable
               else "fragile -> prefer an interval estimate."
               if sign_consistent else "the ATT changes sign -> unstable."))

    return TruncatedHistoryResult(
        mode=mode, att_full=att_full, profile=profile,
        att_min=a_min, att_max=a_max, att_mean=a_mean, stable=stable, stability_note=note)
