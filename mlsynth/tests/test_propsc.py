"""Tests for PROPSC (compositional common-weights SC/SDID).

The reproduction tests are a cross-validation against the authors' R package
``propsdid`` (``lstoetze/propsdid``): the ``_R_ORACLE`` constants below were
produced by running that package on the *same* deterministic panel this module
builds (regeneration script:
``benchmarks/R/propsdid_reference.R`` via ``benchmarks/cases/propsc_spain.py``;
the tiny panel is reproduced in ``benchmarks/R/propsdid_tiny.R``). ``PROPSC.fit``
matches the package cell-by-cell (ATTs, jackknife SEs, unit and time weights) to
machine precision. A live-R re-validation on the paper's Spain data lives in the
benchmark suite.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mlsynth import PROPSC
from mlsynth.config_models import PROPSCConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError

OUTCOMES = ["p1", "p2", "p3"]

# propsdid @ lstoetze/propsdid, run on the panel built by ``_tiny_panel`` below.
_R_ORACLE = {
    "sdid": {
        "att": [0.194932301402, -0.0981704015921, -0.0967618998097],
        "se": [0.0358967880242, 0.0145514184622, 0.0464014531723],
        "omega": [0.13037921406, 0.224798052413, 0.240176057602,
                  0.143888247804, 0.13037921406, 0.13037921406],
        "lambda": [0.500220939783, 0.127902580633, 0.249757673503, 0.122118806081],
    },
    "sc": {
        "att": [0.208980422014, -0.107179580275, -0.101800841739],
        "se": [0.281722858687, 0.0697998288765, 0.215051008372],
        "omega": [0, 0.392567452866, 0.529631271746, 0.0778012753885, 0, 0],
        "lambda": [0, 0, 0, 0],
    },
    "did": {
        "att": [0.187677439418, -0.0932406726291, -0.0944367667892],
        "se": [0.0170357107321, 0.0131325144725, 0.0173251928322],
        "omega": [1 / 6] * 6,
        "lambda": [0.25, 0.25, 0.25, 0.25],
    },
}


def _tiny_panel() -> pd.DataFrame:
    """Deterministic, RNG-free compositional panel (N=8, T=6, K=3, N0=6, T0=4).

    Identical to the panel the R oracle was run on: a heterogeneous latent field
    mapped through a softmax so each unit-time row of ``p1, p2, p3`` sums to 1,
    with a +1.0 latent bump on ``p1`` for the two treated units post period 4.
    """
    N, T, K, N0, T0 = 8, 6, 3, 6, 4
    recs = []
    for i in range(1, N + 1):
        for t in range(1, T + 1):
            lat = [
                math.sin(1.3 * i + 0.9 * k) + 0.8 * math.cos(0.7 * t + 0.5 * k)
                + 0.25 * i - 0.18 * t * k + 0.12 * ((i * 7 + t * 3 + k * 11) % 5)
                for k in range(1, K + 1)
            ]
            treated = i > N0
            if treated and t > T0:
                lat[0] += 1.0
            m = max(lat)
            ex = [math.exp(x - m) for x in lat]
            s = sum(ex)
            rec = {"unit": f"u{i:02d}", "time": t,
                   "treat": 1 if (treated and t > T0) else 0}
            for k in range(K):
                rec[f"p{k + 1}"] = ex[k] / s
            recs.append(rec)
    return pd.DataFrame(recs)


def _fit(method="sdid", **over):
    cfg = {"df": _tiny_panel(), "outcomes": OUTCOMES, "treat": "treat",
           "unitid": "unit", "time": "time", "method": method,
           "display_graphs": False}
    cfg.update(over)
    return PROPSC(cfg).fit()


# --------------------------------------------------------------------------- #
# Reproduction / cross-validation vs the R package (the DoD gate)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["sdid", "sc", "did"])
def test_matches_r_propsdid_cellwise(method):
    res = _fit(method)
    ora = _R_ORACLE[method]
    np.testing.assert_allclose(res.att_vector, ora["att"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(res.se_vector, ora["se"], atol=1e-9, rtol=1e-6)
    dw = np.array([res.donor_weights[f"u{j:02d}"] for j in range(1, 7)])
    np.testing.assert_allclose(dw, ora["omega"], atol=1e-9, rtol=0)
    if method != "sc":
        np.testing.assert_allclose(res.time_weights, ora["lambda"], atol=1e-9, rtol=0)


@pytest.mark.parametrize("method", ["sdid", "sc", "did"])
def test_atts_sum_to_zero(method):
    res = _fit(method)
    assert abs(res.sum_constraint) < 1e-10
    assert abs(sum(p.att for p in res.proportions)) < 1e-10


# --------------------------------------------------------------------------- #
# Smoke / structure
# --------------------------------------------------------------------------- #
def test_smoke_shapes_and_flat_accessors():
    res = _fit("sdid")
    assert len(res.proportions) == 3
    # flat accessors resolve to the target (first) proportion
    assert res.att == pytest.approx(res.proportions[0].att)
    assert res.counterfactual.shape == res.gap.shape
    assert res.counterfactual.ndim == 1
    lo, hi = res.att_ci
    assert lo < res.att < hi
    assert np.isfinite(res.pre_rmse)
    # common weights: every proportion carries the identical donor map
    for p in res.proportions[1:]:
        assert p.donor_weights == res.proportions[0].donor_weights


def test_target_selects_flat_accessors():
    res = _fit("sdid", target="p2")
    assert res.target == "p2"
    assert res.att == pytest.approx(res.att_vector[1])


def test_inference_none_skips_se():
    res = _fit("sdid", inference="none")
    assert np.all(np.isnan(res.se_vector))
    assert res.inference is None


def test_donor_weights_are_simplex():
    res = _fit("sdid")
    w = np.array(list(res.donor_weights.values()))
    assert w.min() >= -1e-9
    assert w.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Config validation / error paths
# --------------------------------------------------------------------------- #
def test_config_requires_two_outcomes():
    with pytest.raises(Exception):
        PROPSCConfig(df=_tiny_panel(), outcomes=["p1"], treat="treat",
                     unitid="unit", time="time")


def test_config_unknown_outcome_column():
    with pytest.raises(MlsynthConfigError):
        PROPSCConfig(df=_tiny_panel(), outcomes=["p1", "nope"], treat="treat",
                     unitid="unit", time="time")


def test_config_bad_target():
    with pytest.raises(MlsynthConfigError):
        PROPSCConfig(df=_tiny_panel(), outcomes=OUTCOMES, target="zzz",
                     treat="treat", unitid="unit", time="time")


def test_config_forbids_extra():
    with pytest.raises(Exception):
        PROPSCConfig(df=_tiny_panel(), outcomes=OUTCOMES, treat="treat",
                     unitid="unit", time="time", bogus=1)


def test_no_treatment_variation_raises():
    df = _tiny_panel()
    df["treat"] = 0
    with pytest.raises(MlsynthDataError):
        PROPSC({"df": df, "outcomes": OUTCOMES, "treat": "treat",
                "unitid": "unit", "time": "time"}).fit()


def test_default_outcome_filled_from_outcomes():
    cfg = PROPSCConfig(df=_tiny_panel(), outcomes=OUTCOMES, treat="treat",
                       unitid="unit", time="time")
    assert cfg.outcome == "p1"


# --------------------------------------------------------------------------- #
# Plotting smoke
# --------------------------------------------------------------------------- #
def test_plotting_smoke(monkeypatch, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    res = _fit("sdid", display_graphs=True, save=str(tmp_path / "p.png"))
    assert len(res.proportions) == 3
    assert (tmp_path / "p.png").exists()


# --------------------------------------------------------------------------- #
# Edge cases / additional failure paths
# --------------------------------------------------------------------------- #
def test_plotting_grid_and_dict_save(monkeypatch, tmp_path):
    """K=4 exercises the multi-row grid, empty-panel cleanup, and dict-save."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    rng = np.random.default_rng(2)
    recs = []
    for i in range(1, 7):
        for t in range(1, 6):
            lat = rng.standard_normal(4) + 0.3 * i + 0.1 * t * np.arange(4)
            if i > 4 and t > 3:
                lat[0] += 1.0
            ex = np.exp(lat - lat.max())
            p = ex / ex.sum()
            rec = {"unit": f"u{i}", "time": t, "treat": int(i > 4 and t > 3)}
            for k in range(4):
                rec[f"q{k + 1}"] = float(p[k])
            recs.append(rec)
    df = pd.DataFrame(recs)
    res = PROPSC({"df": df, "outcomes": ["q1", "q2", "q3", "q4"], "treat": "treat",
                  "unitid": "unit", "time": "time", "display_graphs": True,
                  "save": str(tmp_path / "g.png")}).fit()
    assert (tmp_path / "g.png").exists()
    assert len(res.proportions) == 4


def test_config_duplicate_outcomes():
    with pytest.raises(MlsynthConfigError):
        PROPSCConfig(df=_tiny_panel(), outcomes=["p1", "p1"], treat="treat",
                     unitid="unit", time="time")


def test_pipeline_rejects_unknown_method():
    from mlsynth.utils.propsc_helpers import estimate_common_weights
    Y = _tiny_panel()  # not used directly
    arr = np.zeros((8, 6, 3))
    with pytest.raises(ValueError):
        estimate_common_weights(arr, 6, 4, method="bogus")


def test_sc_weight_fw_2d_matrix_path():
    """The 2-D (single-outcome) solver branch still returns a simplex vector."""
    from mlsynth.utils.propsc_helpers import sc_weight_fw
    rng = np.random.default_rng(0)
    Y2d = rng.standard_normal((5, 4))  # 4 donors, T0=3, target col last
    w = sc_weight_fw(Y2d, zeta=0.1, intercept=True, max_iter=200)
    assert w.shape == (3,)
    assert w.min() >= -1e-9 and w.sum() == pytest.approx(1.0)


def test_single_treated_unit_has_nan_se():
    """Fixed-weights jackknife is undefined with one treated unit -> NaN SE."""
    df = _tiny_panel()
    # keep only one treated unit (u08); make u07 a control (never treated)
    df.loc[df.unit == "u07", "treat"] = 0
    res = PROPSC({"df": df, "outcomes": OUTCOMES, "treat": "treat",
                  "unitid": "unit", "time": "time"}).fit()
    assert np.all(np.isnan(res.se_vector))
    assert res.inference is None


def test_treatment_at_t0_raises():
    df = _tiny_panel()
    df.loc[df.unit.isin(["u07", "u08"]), "treat"] = 1  # treated from t=1
    with pytest.raises(MlsynthDataError):
        PROPSC({"df": df, "outcomes": OUTCOMES, "treat": "treat",
                "unitid": "unit", "time": "time"}).fit()


def test_non_binary_treatment_raises():
    df = _tiny_panel()
    df.loc[(df.unit == "u08") & (df.time == 6), "treat"] = 2
    with pytest.raises(MlsynthDataError):
        PROPSC({"df": df, "outcomes": OUTCOMES, "treat": "treat",
                "unitid": "unit", "time": "time"}).fit()


def test_non_simultaneous_adoption_raises():
    df = _tiny_panel()
    # u07 adopts at t=6 while u08 adopts at t=5 -> staggered
    df["treat"] = 0
    df.loc[(df.unit == "u08") & (df.time >= 5), "treat"] = 1
    df.loc[(df.unit == "u07") & (df.time >= 6), "treat"] = 1
    with pytest.raises(MlsynthDataError):
        PROPSC({"df": df, "outcomes": OUTCOMES, "treat": "treat",
                "unitid": "unit", "time": "time"}).fit()


def test_missing_outcome_value_raises():
    df = _tiny_panel()
    df.loc[(df.unit == "u01") & (df.time == 2), "p2"] = np.nan
    with pytest.raises(MlsynthDataError):
        PROPSC({"df": df, "outcomes": OUTCOMES, "treat": "treat",
                "unitid": "unit", "time": "time"}).fit()
