"""Tests for the SSC estimator (Cao, Lu & Wu 2026).

Layered per agents/agents_tests.md:

* Layer 1 (numerical helpers): simplex SC weights; the treatment selector
  tensor and event-time indexing.
* Layer 2 (data utilities): staggered-panel ingestion, absorbing-treatment and
  pre-period guards.
* Layer 3 (estimator integration): recovery of an increasing event-time ATT on
  the paper's factor DGP; Andrews bands.
* Layer 4 (public API contracts): import, frozen results, config validation.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from mlsynth import SSC
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.ssc_helpers import (
    SSCInputs,
    SSCResults,
    prepare_ssc_inputs,
    simulate_ssc_panel,
    synthetic_control_batch,
)
from mlsynth.utils.ssc_helpers.estimation import build_treatment_structure


def _panel(n_units=20, n_never=4, T0=50, S=6, r=2, base=1.0, seed=1):
    return simulate_ssc_panel(n_units=n_units, n_never=n_never, T0=T0, S=S,
                              n_factors=r, base_effect=base, seed=seed)


# ----------------------------------------------------------------------
# Layer 1: numerical helpers
# ----------------------------------------------------------------------

class TestHelpers:
    def test_sc_weights_simplex_and_zero_diagonal(self):
        rng = np.random.default_rng(0)
        Y_pre = rng.standard_normal((6, 40))
        a, B = synthetic_control_batch(Y_pre)
        assert B.shape == (6, 6)
        assert np.allclose(np.diag(B), 0.0)            # no self-weight
        assert np.allclose(B.sum(axis=1), 1.0, atol=1e-4)   # rows on simplex
        assert (B >= -1e-8).all()                      # non-negative

    def test_treatment_structure_event_time(self):
        # 3 units, T0=2, S=2; unit 1 adopts at col 2, unit 2 at col 3.
        D = np.array([[0, 0, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 0, 1]])
        index, A = build_treatment_structure(D, T0=2)
        assert A.shape == (3, int(D[:, 2:].sum()), 2)
        # event times: unit1 at s=1 -> e=0, s=2 -> e=1; unit2 at s=2 -> e=0
        evt = {(int(r[0]), int(r[1])): int(r[2]) for r in index}
        assert evt[(1, 1)] == 0 and evt[(2, 1)] == 1   # unit 1 (index 1)
        assert evt[(2, 2)] == 0                         # unit 2 (index 2)


# ----------------------------------------------------------------------
# Layer 2: ingestion
# ----------------------------------------------------------------------

class TestIngestion:
    def test_shapes_and_staggering(self):
        df = _panel(n_units=12, n_never=3, T0=40, S=5)
        inp = prepare_ssc_inputs(df, "Y", "treated", "unit", "time")
        assert inp.N == 12 and inp.T0 == 40 and inp.S == 5
        assert inp.treated_idx.size == 9
        # staggered: more than one distinct adoption time
        adopts = set(inp.adoption[inp.treated_idx].tolist())
        assert len(adopts) > 1

    def test_nonabsorbing_rejected(self):
        df = _panel(n_units=8, n_never=2, T0=30, S=5)
        # turn one treated unit's last period back off
        u = sorted(df["unit"].unique())
        last = df["time"].max()
        treated_u = df[(df["treated"] == 1)]["unit"].iloc[0]
        df.loc[(df["unit"] == treated_u) & (df["time"] == last), "treated"] = 0
        with pytest.raises(MlsynthDataError):
            prepare_ssc_inputs(df, "Y", "treated", "unit", "time")

    def test_short_pre_period_point_only(self):
        # When T0 <= S there are no end-of-sample placebo windows: point
        # estimates are still produced, but the bands are NaN (as in the
        # paper's theft application).
        df = _panel(n_units=8, n_never=2, T0=6, S=8)
        res = SSC({"df": df, "outcome": "Y", "treat": "treated",
                   "unitid": "unit", "time": "time",
                   "inference": True, "display_graphs": False}).fit()
        assert np.isfinite(res.att)                 # point estimate exists
        assert np.isnan(res.att_band.lower)         # band undefined
        assert res.inference_detail.n_placebo == 0


# ----------------------------------------------------------------------
# Layer 3: estimator integration
# ----------------------------------------------------------------------

class TestIntegration:
    def test_recovers_increasing_event_study(self):
        df = _panel(n_units=20, n_never=4, T0=50, S=6, base=1.0, seed=1)
        res = SSC({"df": df, "outcome": "Y", "treat": "treated",
                   "unitid": "unit", "time": "time",
                   "inference": True, "display_graphs": False}).fit()
        assert isinstance(res, SSCResults)
        # overall effect is positive and significant
        assert res.att > 0
        assert res.att_band.significant
        assert res.att_band.p_value < 0.05
        # the effect grows with event time (true effect is 1 + e)
        events = sorted(res.event_att)
        late = np.mean([res.event_att[e] for e in events[-2:]])
        early = np.mean([res.event_att[e] for e in events[:2]])
        assert late > early
        # per-cell effect grid has the right footprint
        assert res.effects_matrix.shape == (20, 6)
        assert np.isfinite(res.effects_matrix).sum() == res.metadata["K"]

    def test_simulation_study_runs(self):
        # Path-B replication harness: returns event-time RMSE per (r, T0) cell.
        from mlsynth.utils.ssc_helpers.replication import (
            run_ssc_simulation, SSCSimConfig,
        )
        cfg = SSCSimConfig(n_units=12, n_never=3, S=5, n_factors=2,
                           T0_grid=[30], n_reps=3)
        out = run_ssc_simulation(cfg, seed=0, verbose=False)
        assert (2, 30) in out                      # keyed by (r, T0)
        rmse = out[(2, 30)]
        assert len(rmse) > 0
        assert all(v >= 0 for v in rmse.values())  # RMSEs are non-negative

    def test_guanajuato_empirical_replication(self):
        # Path-A: reproduce the paper's cartel-outcome estimates from the
        # shipped basedata files (fast outcomes only). Skip if data absent.
        import pathlib
        base = pathlib.Path(__file__).resolve().parents[2] / "basedata"
        crime = base / "guanajuato_crime_ssc.csv"
        cartel = base / "guanajuato_cartel_ssc.csv"
        ref = base / "guanajuato_ssc_reference.csv"
        if not (crime.exists() and cartel.exists() and ref.exists()):
            pytest.skip("Guanajuato basedata not present")
        from mlsynth.utils.ssc_helpers import replicate_guanajuato
        out = replicate_guanajuato(str(crime), str(cartel),
                                   outcomes=["war", "presence_strength"],
                                   verbose=False)
        r = pd.read_csv(ref, encoding="latin-1")
        errs = []
        for _, row in out.iterrows():
            rr = r[(r["outcome"] == row["outcome"]) &
                   (r["event time"] == row["event_time"])]
            if len(rr):
                errs.append(abs(row["att"] - rr["att estimate"].values[0]))
        assert max(errs) < 5e-3          # matches authors' reference

    def test_inference_off(self):
        df = _panel(n_units=12, n_never=3, T0=40, S=5)
        res = SSC({"df": df, "outcome": "Y", "treat": "treated",
                   "unitid": "unit", "time": "time",
                   "inference": False, "display_graphs": False}).fit()
        assert res.att_band is None and res.inference_detail is None
        assert res.event_bands == {}
        assert len(res.event_att) > 0          # point estimates still produced


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestAPI:
    def test_results_frozen(self):
        df = _panel(n_units=10, n_never=2, T0=30, S=5)
        res = SSC({"df": df, "outcome": "Y", "treat": "treated",
                   "unitid": "unit", "time": "time",
                   "display_graphs": False}).fit()
        # `att` is now a read-only accessor; mutate a real field to prove the
        # frozen pydantic model rejects assignment.
        with pytest.raises(Exception):
            res.tau = np.zeros_like(res.tau)

    def test_bad_config_raises(self):
        df = _panel(n_units=8)
        with pytest.raises(MlsynthConfigError):
            SSC({"df": df, "treat": "treated", "unitid": "unit", "time": "time"})

    def test_inputs_immutable(self):
        df = _panel(n_units=8, n_never=2, T0=30, S=5)
        inp = prepare_ssc_inputs(df, "Y", "treated", "unit", "time")
        assert isinstance(inp, SSCInputs)
        with pytest.raises(dataclasses.FrozenInstanceError):
            inp.T0 = 0
