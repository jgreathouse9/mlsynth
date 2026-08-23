"""Reading a captured golden-steps fixture, and comparing against it honestly.

A golden fixture is decimal text written by a reference implementation, so two
things bound what a test can assert against it, and neither is a free choice.

The first is the fixture's own precision. A value printed to eight decimals says
nothing about the ninth, so a comparison tighter than one unit in its last
recorded place is a comparison against rounding. :func:`ulp_of` reads that bound
off each token, which is why the capture scripts emit every compared quantity
twice: the readable ``%.8f`` line, and a ``hi:`` line at ``%.17g``. Where the
``hi:`` line exists it wins, so improving a capture tightens its tests without
anyone editing a constant.

The second is the arithmetic. mlsynth's least-squares steps run through numpy's
``lstsq`` (LAPACK ``gelsd``) and the R references through ``lm`` (LINPACK
``dqrls``); on the SBC panels the two differ by 1.7e-14 of each series' scale on
the German reunification data and 2.6e-14 on the Hong Kong handover, uniformly
across coefficients, cycles and forecasts in both. Two panels of different
length, scale and donor pool landing at the same order is what says this is the
kernels and not the method. They do not agree bit for bit, so exact equality is
not available at any capture precision. :data:`CROSS_IMPL_FLOOR` sets the
comparison at 1e-12 of scale -- thirty-nine times the larger of the two measured
differences, which leaves room for a different BLAS while still failing on a
defect three orders below what a hand-set 1e-5 band could see.

Shared by ``test_sbc_reference.py`` (German reunification) and
``test_sbc_hongkong_reference.py`` (the handover), which pin the same estimator
against the same authors' functions on two panels.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

CROSS_IMPL_FLOOR = 1e-12
MIN_SIGNIFICANT_DIGITS = 8


def ulp_of(token: str) -> float:
    """One unit in ``token``'s last recorded place.

    ``"610.16314485"`` gives 1e-8; ``"1.2661625755613e+06"`` gives 1e-7.
    """
    mantissa, _, exponent = token.strip().lower().partition("e")
    decimals = len(mantissa.partition(".")[2])
    return 10.0 ** (int(exponent or 0) - decimals)


def load(path: Path) -> dict:
    """Parse a fixture into ``{key: (values, per-value tolerances)}``.

    Skips the test module that calls it when the fixture is absent, which is how
    an optional capture stays optional without a silent pass.
    """
    if not Path(path).exists():
        pytest.skip(f"golden fixture missing: {path}")
    parsed: dict = {}
    for line in Path(path).read_text().splitlines():
        if "\t" not in line:
            continue
        key, val = line.split("\t", 1)
        tokens = [t for t in val.split(",") if t.strip()]
        values = np.array([float(t) for t in tokens])
        tols = np.array([ulp_of(t) for t in tokens])
        if key.startswith("hi:"):
            parsed[key[3:]] = (values, tols)
        elif key not in parsed:
            parsed[key] = (values, tols)
    return parsed


def scalar(golden: dict, key: str) -> float:
    return float(golden[key][0][0])


def assert_reproduces(ours, golden: dict, key: str) -> None:
    """mlsynth matches the reference on ``key``, at whichever bound is real."""
    ref, tol = golden[key]
    tol = np.maximum(tol, CROSS_IMPL_FLOOR * np.max(np.abs(ref)))
    ours = np.asarray(ours, dtype=float)[: len(ref)]
    delta = np.abs(ours - ref)
    worst = int(np.argmax(delta - tol))
    assert np.all(delta <= tol), (
        f"{key}[{worst}]: mlsynth {ours[worst]!r} vs R {ref[worst]!r}; "
        f"delta {delta[worst]:.3e} exceeds {tol[worst]:.1e}"
    )


def assert_resolves_enough_digits(golden: dict, keys) -> None:
    """The capture resolves enough digits for the comparisons made against it.

    Every compared value is asserted at one unit in its last recorded place, so
    a capture written with fewer digits would loosen every comparison while all
    of them still passed. This fails first instead.
    """
    for key in keys:
        values, tols = golden[key]
        digits = np.array([
            math.floor(math.log10(abs(v))) + 1 - math.floor(math.log10(t))
            for v, t in zip(values, tols) if v != 0.0
        ])
        assert digits.min() >= MIN_SIGNIFICANT_DIGITS, (
            f"{key} resolves only {digits.min()} significant digits; "
            f"comparisons against it cannot be tighter than that"
        )
