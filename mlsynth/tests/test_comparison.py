"""compare_estimators: fit several observational estimators and overlay them.

The design side has ``compare_methods`` (fit SYNDES/GEOLIFT/... on one panel and
compare); this is its observational twin. ``compare_estimators`` fits each
estimator you hand it (fully configured), reads each one's counterfactual and
canonical per-period band off the standardized contract, and returns the existing
:class:`~mlsynth.utils.counterfactual_compare.CounterfactualComparison` (summary
table + overlay plot). ``show_bands`` is a required choice.

Layered per ``agents/agents_tests.md``: the required toggle, a smoke over two
real estimators on one panel, the band-presence invariant (a method that carries
a canonical band contributes one; one that does not is a line only), and the two
failure paths (mixed panels, an estimator with no counterfactual).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC, FDID, compare_estimators, CounterfactualComparison
from mlsynth.exceptions import MlsynthConfigError


def _panel(seed: int = 0, n_donors: int = 6, T: int = 22, T0: int = 15,
           effect: float = -3.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, 3))
    L = rng.standard_normal((n_donors + 1, 3))
    Y = F @ L.T + 0.2 * rng.standard_normal((T, n_donors + 1))
    Y[T0:, 0] += effect
    return pd.DataFrame([
        {"unit": f"u{u}", "year": 2000 + t, "y": float(Y[t, u]),
         "tr": int(u == 0 and t >= T0)}
        for u in range(n_donors + 1) for t in range(T)])


def _base(df):
    return dict(df=df, outcome="y", treat="tr", unitid="unit", time="year",
               display_graphs=False)


def test_show_bands_is_required():
    df = _panel()
    with pytest.raises(TypeError):
        compare_estimators([VanillaSC(_base(df))])   # missing show_bands


@pytest.fixture(scope="module")
def compared():
    df = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compare_estimators(
            [VanillaSC({**_base(df), "inference": "scpi", "scpi_sims": 60}),
             FDID(_base(df))],
            show_bands=True)


def test_smoke_returns_comparison(compared):
    assert isinstance(compared, CounterfactualComparison)
    assert set(compared.summary.index) == {"VanillaSC", "FDID"}
    assert np.all(np.isfinite(compared.summary["att"].to_numpy(float)))
    assert compared.observed is not None


def test_band_present_only_where_the_method_carries_one(compared):
    curves = compared.curves
    vsc = curves[curves["method"] == "VanillaSC"]
    fdid = curves[curves["method"] == "FDID"]
    assert np.isfinite(vsc["lower"].to_numpy(float)).any()   # scpi band present
    assert np.isnan(fdid["lower"].to_numpy(float)).all()     # FDID: line only


def test_show_bands_false_drops_bands():
    df = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = compare_estimators(
            [VanillaSC({**_base(df), "inference": "scpi", "scpi_sims": 60})],
            show_bands=False)
    assert np.isnan(cmp.curves["lower"].to_numpy(float)).all()


def test_plot_smoke(compared):
    import matplotlib
    matplotlib.use("Agg")
    ax = compared.plot()
    assert ax is not None


# --------------------------------------------------------------------------
# failure paths
# --------------------------------------------------------------------------
def test_mixed_panels_rejected():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = VanillaSC(_base(_panel(seed=0)))
        b = VanillaSC(_base(_panel(seed=1)))       # different panel
        with pytest.raises(MlsynthConfigError):
            compare_estimators([a, b], show_bands=False)


def test_estimator_without_counterfactual_rejected():
    class _NoCf:
        def fit(self):
            class _R:
                time_series = None
            return _R()
    with pytest.raises(MlsynthConfigError):
        compare_estimators([_NoCf()], show_bands=False)
