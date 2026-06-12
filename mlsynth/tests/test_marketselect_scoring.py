import pytest
import numpy as np

from mlsynth.utils.geolift_helpers.marketselect.helpers.windows import (
    lookback_pre_periods,
    lookback_treatment_window,
)
from mlsynth.exceptions import MlsynthConfigError


# === lookback_pre_periods ===

def test_pre_periods_smoke():
    assert lookback_pre_periods(100, 4, 1) == 96


def test_pre_periods_matches_geolift_formula():
    """Faithful to GeoLift's max_time - tp - sim + 2 (in 0-indexed counts)."""
    for n, d, s in [(100, 4, 1), (100, 4, 2), (50, 7, 3), (20, 1, 5)]:
        assert lookback_pre_periods(n, d, s) == n - d - s + 1


def test_pre_periods_shrinks_one_per_sim():
    base = lookback_pre_periods(100, 4, 1)
    assert lookback_pre_periods(100, 4, 2) == base - 1
    assert lookback_pre_periods(100, 4, 3) == base - 2


def test_pre_periods_minimal_one():
    assert lookback_pre_periods(5, 4, 1) == 1            # 5 - 4 - 1 + 1


def test_pre_periods_runs_off_start_raises():
    with pytest.raises(MlsynthConfigError, match="runs off the start"):
        lookback_pre_periods(4, 4, 1)                    # 4 - 4 - 1 + 1 = 0
    with pytest.raises(MlsynthConfigError, match="runs off the start"):
        lookback_pre_periods(5, 4, 2)                    # 5 - 4 - 2 + 1 = 0


@pytest.mark.parametrize("bad", [0, -1])
def test_pre_periods_nonpositive_args_raise(bad):
    with pytest.raises(MlsynthConfigError, match="positive integer"):
        lookback_pre_periods(bad, 4, 1)
    with pytest.raises(MlsynthConfigError, match="positive integer"):
        lookback_pre_periods(100, bad, 1)
    with pytest.raises(MlsynthConfigError, match="positive integer"):
        lookback_pre_periods(100, 4, bad)


def test_pre_periods_rejects_bool():
    with pytest.raises(MlsynthConfigError, match="positive integer"):
        lookback_pre_periods(True, 4, 1)


# === lookback_treatment_window ===

def test_treatment_window_smoke():
    assert lookback_treatment_window(100, 4, 1) == (96, 99)


def test_treatment_window_slides_back_one_per_sim():
    assert lookback_treatment_window(100, 4, 1) == (96, 99)
    assert lookback_treatment_window(100, 4, 2) == (95, 98)
    assert lookback_treatment_window(100, 4, 3) == (94, 97)


def test_treatment_window_length_equals_duration():
    for n, d, s in [(100, 4, 1), (50, 7, 3), (30, 2, 5), (12, 1, 1)]:
        start, end = lookback_treatment_window(n, d, s)
        assert end - start + 1 == d


def test_treatment_window_sim1_ends_at_last_period():
    n = 40
    _, end = lookback_treatment_window(n, 6, 1)
    assert end == n - 1                                  # 0-indexed last period


def test_treatment_window_inherits_off_start_guard():
    with pytest.raises(MlsynthConfigError, match="runs off the start"):
        lookback_treatment_window(5, 4, 2)
