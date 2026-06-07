"""Path-A replication of Li (2024) Forward DiD on the public Hong Kong panel.

Li's headline application uses a confidential retailer dataset, but the author
released a public companion replication (MATLAB/R) on the Hsiao, Ching & Wan
(2012) Hong Kong GDP panel. This asserts mlsynth's :class:`~mlsynth.FDID`
reproduces the released ATT / %ATT / pre-period R^2 / selected-control count
cell by cell (forward selection is deterministic, so tolerances absorb only the
estimator's 3-4 dp display rounding).

The durable, paper-facing version lives in
``benchmarks/cases/fdid_hongkong.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlsynth import FDID

_HK = Path(__file__).resolve().parents[2] / "basedata" / "HongKong.csv"


@pytest.fixture(scope="module")
def hk_fit():
    df = pd.read_csv(_HK)
    return FDID(
        {
            "df": df,
            "outcome": "GDP",
            "treat": "Integration",
            "unitid": "Country",
            "time": "Time",
            "display_graphs": False,
            "verbose": False,
        }
    ).fit()


def test_fdid_matches_released_output(hk_fit):
    """Forward DiD: selects 9 controls, ATT 0.0254 (53.84%), pre-R^2 0.843."""
    f = hk_fit.fdid
    assert len(f.selected_names) == 9
    assert f.att == pytest.approx(0.025405, abs=5e-4)
    assert f.att_percent == pytest.approx(53.843, abs=0.1)
    assert f.r_squared == pytest.approx(0.84278, abs=2e-3)


def test_did_matches_released_output(hk_fit):
    """Plain DiD (all 24 controls): ATT 0.0317 (77.62%), pre-R^2 0.505."""
    d = hk_fit.did
    assert d.att == pytest.approx(0.031721, abs=5e-4)
    assert d.att_percent == pytest.approx(77.62, abs=0.1)
    assert d.r_squared == pytest.approx(0.50465, abs=2e-3)
