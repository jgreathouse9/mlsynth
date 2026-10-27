"""Path-to-purchase data: event streams and the event-type space.

Event types are integers ``0..p-1`` ordered as in Tao et al. (2023), Sec. 2.3:
``0`` is conversion, ``0..q-1`` are customer-initiated types (modelled), and
``q..p-1`` are firm-initiated types (observed inputs, never modelled).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Path:
    """One customer's event stream ``{(t_i, e_i)}`` observed on ``[0, T]``."""

    times: np.ndarray
    types: np.ndarray
    T: float

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        types = np.asarray(self.types, dtype=int)
        if times.ndim != 1 or times.shape != types.shape:
            raise ValueError("times and types must be 1-D arrays of equal length")
        if times.size and (times[0] < 0 or times[-1] > self.T):
            raise ValueError(f"event times must lie in [0, T={self.T}]")
        if np.any(np.diff(times) <= 0):
            raise ValueError("event times must be strictly increasing (no ties)")
        if self.T <= 0:
            raise ValueError("T must be positive")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "types", types)

    def __len__(self) -> int:
        return self.times.size


@dataclass(frozen=True)
class EventSpace:
    """Names of the event types and the channel each touchpoint belongs to.

    Index layout: ``conversion`` -> 0, then ``customer`` in order, then ``firm``.
    """

    conversion: str
    customer: tuple[str, ...]
    firm: tuple[str, ...]
    channels: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = self.names
        if len(set(names)) != len(names):
            raise ValueError("event type names must be unique")
        known = set(names) - {self.conversion}
        for ch, members in self.channels.items():
            unknown = set(members) - known
            if unknown:
                raise ValueError(f"channel {ch!r} names unknown touchpoints {sorted(unknown)}")

    @property
    def names(self) -> tuple[str, ...]:
        return (self.conversion, *self.customer, *self.firm)

    @property
    def p(self) -> int:
        return len(self.names)

    @property
    def q(self) -> int:
        return 1 + len(self.customer)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def channel_types(self) -> dict[str, list[int]]:
        return {ch: [self.index(m) for m in members] for ch, members in self.channels.items()}


def paths_from_frame(
    df: pd.DataFrame,
    space: EventSpace,
    *,
    path_col: str = "path_id",
    time_col: str = "time",
    event_col: str = "event",
    end_col: str | None = None,
    origin: str = "first",
) -> tuple[list[Path], list]:
    """Build paths from a long frame with one row per event.

    ``origin="first"`` shifts each path so its first event is at ``t = 0`` (the
    paper's convention for the DASS+ data, Sec. 5.2); ``origin="zero"`` keeps
    the times as given. The window end ``T`` is ``end_col`` when supplied,
    else the last event time (again the DASS+ convention).

    Returns the paths and the path ids in the same order.
    """
    unknown = set(df[event_col].unique()) - set(space.names)
    if unknown:
        raise ValueError(f"events not in the event space: {sorted(map(str, unknown))}")
    if origin not in ("first", "zero"):
        raise ValueError("origin must be 'first' or 'zero'")
    code = {name: i for i, name in enumerate(space.names)}
    paths, ids = [], []
    for pid, g in df.sort_values([path_col, time_col]).groupby(path_col, sort=False):
        t = g[time_col].to_numpy(dtype=float)
        shift = t[0] if origin == "first" else 0.0
        t = t - shift
        T = float(g[end_col].iloc[0]) - shift if end_col is not None else float(t[-1])
        if T <= 0:  # single-event path observed at its own origin
            T = np.finfo(float).eps
        paths.append(Path(t, g[event_col].map(code).to_numpy(), T))
        ids.append(pid)
    return paths, ids


def validate_paths(paths: Sequence[Path], p: int) -> None:
    if len(paths) == 0:
        raise ValueError("need at least one path")
    for j, path in enumerate(paths):
        if path.types.size and (path.types.min() < 0 or path.types.max() >= p):
            raise ValueError(f"path {j} has event types outside 0..{p - 1}")
