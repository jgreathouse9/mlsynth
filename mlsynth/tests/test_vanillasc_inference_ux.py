"""Discoverability / UX guards for ``VanillaSC(inference=...)``.

A user should never be surprised by *what* inference they got. Two failure
modes are pinned here:

* an unknown / misspelled ``inference`` value (e.g. ``"scpii"``) must raise a
  clear ``MlsynthConfigError`` listing the valid options -- not silently return
  no inference;
* a *valid* inference mode that cannot be computed on the given panel (e.g.
  ``"placebo"`` with a single donor) must emit a warning and return an
  ``InferenceResults`` whose ``method`` says it was not computed and why -- not a
  silent ``None`` and not a mid-analysis crash.

Valid, computable modes must keep working unchanged (regression guard).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError


def _panel(n_units: int, n_periods: int = 12, t0: int = 8, seed: int = 0) -> pd.DataFrame:
    """A small balanced panel: unit ``u0`` is treated from period ``t0``."""
    rng = np.random.default_rng(seed)
    donor_base = rng.normal(10.0, 1.0, size=max(n_units - 1, 1))
    loads = rng.dirichlet(np.ones(max(n_units - 1, 1)))
    rows = []
    for t in range(n_periods):
        common = 0.4 * t + rng.normal(0.0, 0.2)
        donors = donor_base + common + rng.normal(0.0, 0.15, size=donor_base.size)
        treated = float(loads @ donors) + (4.0 if t >= t0 else 0.0)
        rows.append({"unit": "u0", "time": t, "y": treated,
                     "treat": int(t >= t0)})
        for j, dv in enumerate(donors):
            rows.append({"unit": f"d{j}", "time": t, "y": float(dv), "treat": 0})
    return pd.DataFrame(rows)


_CFG = dict(outcome="y", treat="treat", unitid="unit", time="time",
            backend="outcome-only", display_graphs=False)


@pytest.mark.parametrize("bad", ["scpii", "conformall", "bogus", "SCPI_", "placebos"])
def test_unknown_inference_value_raises(bad):
    df = _panel(6)
    with pytest.raises(MlsynthConfigError):
        VanillaSC({"df": df, "inference": bad, **_CFG}).fit()


def test_unknown_inference_message_lists_valid_options():
    df = _panel(6)
    with pytest.raises(MlsynthConfigError) as exc:
        VanillaSC({"df": df, "inference": "scpii", **_CFG}).fit()
    msg = str(exc.value).lower()
    assert "scpi" in msg and "conformal" in msg and "placebo" in msg


def test_uncomputable_placebo_warns_and_explains():
    # One donor -> in-space placebo needs >= 2 donors, so it cannot run.
    df = _panel(2)
    with pytest.warns(UserWarning, match="not computed"):
        res = VanillaSC({"df": df, "inference": "placebo", **_CFG}).fit()
    # never a silent None: an explanatory object survives, and it says why.
    assert res.inference is not None
    assert res.inference.method is not None
    assert "not computed" in res.inference.method.lower()
    assert res.inference.p_value is None


def test_uncomputable_lto_warns_and_explains():
    # Two donors -> leave-two-out needs >= 3 donors.
    df = _panel(3)
    with pytest.warns(UserWarning, match="not computed"):
        res = VanillaSC({"df": df, "inference": "lto", **_CFG}).fit()
    assert res.inference is not None and "not computed" in res.inference.method.lower()


def test_valid_inference_modes_still_produce_output():
    df = _panel(8)
    # bands modes
    for mode in ("scpi", "conformal"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")           # no spurious "not computed" warning
            res = VanillaSC({"df": df, "inference": mode, **_CFG}).fit()
        assert res.inference is not None
        assert res.time_series.counterfactual_lower is not None
        assert "not computed" not in (res.inference.method or "").lower()
    # placebo with enough donors: a real p-value, no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = VanillaSC({"df": df, "inference": "placebo", **_CFG}).fit()
    assert res.inference is not None and res.inference.p_value is not None


def test_inference_false_disables_cleanly():
    df = _panel(6)
    res = VanillaSC({"df": df, "inference": False, **_CFG}).fit()
    assert res.inference is None            # explicit opt-out stays None
