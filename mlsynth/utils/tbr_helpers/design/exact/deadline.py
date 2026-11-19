"""The time limit of the certified search (#965, v1).

certify_treatment turns its ``time_limit`` into a deadline on one clock, and every
stage that can stop early reads it: the floors between blocks, the pass between
waves, the certifications as each group is admitted and each node of a size's branch
and bound as it is entered, and the retries. The clock is CLOCK_MONOTONIC, which
Linux keeps system-wide, so a deadline taken in the parent means the same instant in
a worker forked from it. ``clock`` is read at each call, so a test can replace it."""
from __future__ import annotations

import time
from typing import Optional

#: The clock a deadline is read on.
clock = time.monotonic


def at(limit: Optional[float]) -> Optional[float]:
    """The deadline ``limit`` seconds from now, None when there is no limit."""
    return None if limit is None else clock() + limit


def passed(deadline: Optional[float]) -> bool:
    """Whether the deadline has passed; never, when there is none."""
    return deadline is not None and clock() >= deadline
