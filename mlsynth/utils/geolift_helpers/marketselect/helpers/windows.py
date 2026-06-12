"""Lookback-window arithmetic for GeoLift market-selection power simulation.

The power analysis slides a fixed-length pseudo-treatment window back from the
end of the historical panel, one period at a time, and fits the SCM once per
placement. These dumb helpers translate a ``(n_periods, duration, sim)`` triple
into the pre/post split for a *single* such placement — nothing more, so they
stay trivial to reason about, debug, and test.

Faithful to GeoLift's ``treatment_start_time = max_time - tp - sim + 2``,
expressed here in 0-indexed positions over the time axis:

- ``sim`` is the 1-indexed lookback placement (``sim = 1`` is flush with the
  end of the panel; higher ``sim`` slides the window back one period each).
- ``duration`` is the length of the pseudo-treatment (effect) block.
"""

from typing import Tuple

import numpy as np

from mlsynth.exceptions import MlsynthConfigError


def _check_positive_int(name: str, value: object) -> None:
    """Reject anything that is not an integer >= 1 (booleans included)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise MlsynthConfigError(
            f"{name} must be a positive integer; got {value!r}."
        )


def lookback_pre_periods(n_periods: int, duration: int, sim: int) -> int:
    """Number of pre-treatment periods for lookback placement ``sim``.

    Equivalently, the 0-indexed position of the *first* pseudo-treatment period
    (everything before it is pre-period). Faithful to GeoLift's
    ``max_time - tp - sim + 2``, in 0-indexed counts:

        n_pre = n_periods - duration - sim + 1

    Raises
    ------
    MlsynthConfigError
        If any argument is not a positive integer, or if the placement runs off
        the start of the panel (fewer than one pre-period remains).
    """
    _check_positive_int("n_periods", n_periods)
    _check_positive_int("duration", duration)
    _check_positive_int("sim", sim)

    n_pre = int(n_periods) - int(duration) - int(sim) + 1
    if n_pre < 1:
        raise MlsynthConfigError(
            "Lookback placement runs off the start of the panel: "
            f"n_periods={n_periods}, duration={duration}, sim={sim} leaves "
            f"{n_pre} pre-period(s); need >= 1."
        )
    return n_pre


def lookback_treatment_window(
    n_periods: int, duration: int, sim: int
) -> Tuple[int, int]:
    """0-indexed ``(start, end)`` inclusive positions of the pseudo-treatment block.

    The block has length ``duration`` and ends ``sim - 1`` periods before the
    final period (``sim = 1`` ends exactly at the last period, ``n_periods - 1``).
    Shares the same validation/guards as :func:`lookback_pre_periods`.
    """
    start = lookback_pre_periods(n_periods, duration, sim)
    end = start + int(duration) - 1
    return start, end
