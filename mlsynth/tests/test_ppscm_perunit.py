"""Per-unit (per-cohort) outputs of PPSCM alongside the pooled aggregate.

The partially-pooled estimator fits a separate synthetic control per treated unit
(or per cohort with ``time_cohort=True``) and averages them into the ATT, so the
unit-level estimates are the components of the pooled one. These tests pin that
``PPSCMResults.per_unit`` exposes those components and that they *reconstruct* the
reported aggregates -- so a manager asking for unit-level SC estimates and their
in-sample error gets the same fit as the pooled/aggregated report, not a re-run:

* per-unit in-sample error ``prefit_rmspe`` (``q_j``) satisfies
  ``sqrt(mean_j q_j^2) == design.ind_l2`` (the reported separate imbalance);
* per-unit event-study ``tau``, aggregated the estimator's way (n1-weighted per
  horizon), reproduces the pooled event study and ATT;
* keys align with ``donor_weights_by_cohort``; each fit carries its label,
  adoption time, member units, donor weights, and effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import PPSCM
from mlsynth.utils.ppscm_helpers.structures import PPSCMResults, PPSCMUnitFit


def _staggered_panel(*, seed=0, adoption_offsets=(10, 15, 20), N_donors=8, T=30,
                     true_effect=-3.0, noise=0.4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((T, 2))
    ld = rng.standard_normal((N_donors, 2)) * 0.5
    lt = ld.mean(axis=0)
    rec = []
    for j, Tj in enumerate(adoption_offsets):
        s = factors @ (lt + 0.1 * rng.standard_normal(2)) + rng.standard_normal(T) * noise
        s[Tj:] += true_effect
        for t in range(T):
            rec.append({"unit": f"treated_{j}", "year": 2000 + t,
                        "y": float(s[t]), "tr": int(t >= Tj)})
    for dd in range(N_donors):
        s = factors @ ld[dd] + rng.standard_normal(T) * noise
        for t in range(T):
            rec.append({"unit": f"d_{dd}", "year": 2000 + t, "y": float(s[t]), "tr": 0})
    return pd.DataFrame(rec)


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="tr", unitid="unit", time="year",
                display_graphs=False, run_inference=False)
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def fitted():
    return PPSCM(_cfg(_staggered_panel())).fit()


class TestPerUnitStructure:
    def test_per_unit_present_by_default(self, fitted):
        assert isinstance(fitted, PPSCMResults)
        assert isinstance(fitted.per_unit, dict) and len(fitted.per_unit) >= 1
        for v in fitted.per_unit.values():
            assert isinstance(v, PPSCMUnitFit)

    def test_keys_match_donor_weights(self, fitted):
        assert set(fitted.per_unit) == set(fitted.donor_weights_by_cohort)

    def test_fields_populated(self, fitted):
        for key, uf in fitted.per_unit.items():
            assert uf.label == key
            assert np.isfinite(uf.att)
            assert uf.prefit_rmspe >= 0.0
            assert uf.n_units >= 1
            assert len(uf.member_units) == uf.n_units
            assert uf.tau.shape == (fitted.design.n_leads,)
            # donor weights are a nonneg simplex (sum ~ 1)
            assert abs(sum(uf.donor_weights.values()) - 1.0) < 1e-4
            assert all(w >= -1e-9 for w in uf.donor_weights.values())

    def test_att_is_mean_of_own_event_study(self, fitted):
        for uf in fitted.per_unit.values():
            assert uf.att == pytest.approx(float(np.nanmean(uf.tau)), abs=1e-9)


class TestReconstructsAggregates:
    def test_prefit_rmspe_reconstructs_ind_l2(self, fitted):
        """sqrt(mean_j q_j^2) == the reported separate imbalance ind_l2."""
        q = np.array([uf.prefit_rmspe for uf in fitted.per_unit.values()])
        assert float(np.sqrt(np.mean(q ** 2))) == pytest.approx(
            fitted.design.ind_l2, abs=1e-8)

    def test_tau_reconstructs_pooled_event_study(self, fitted):
        """n1-weighted per-horizon average of the unit taus == pooled event study."""
        fits = list(fitted.per_unit.values())
        H = fitted.design.n_leads
        taus = np.vstack([uf.tau for uf in fits])           # (J, H)
        n1 = np.array([uf.n_units for uf in fits], dtype=float)
        denom = np.array([np.nansum(n1 * ~np.isnan(taus[:, h])) for h in range(H)])
        per_time = np.array([np.nansum(n1 * taus[:, h]) / denom[h] if denom[h] > 0 else np.nan
                             for h in range(H)])
        np.testing.assert_allclose(per_time, fitted.event_study.tau, atol=1e-8, equal_nan=True)
        assert float(np.nanmean(per_time)) == pytest.approx(fitted.effects.att, abs=1e-8)


class TestTimeCohort:
    def test_cohort_keyed_and_members_listed(self):
        # two units share an adoption time -> one cohort group of size 2
        df = _staggered_panel(adoption_offsets=(12, 12, 18))
        res = PPSCM(_cfg(df, time_cohort=True)).fit()
        sizes = sorted(uf.n_units for uf in res.per_unit.values())
        assert sizes == [1, 2]                       # cohort {t0,t1} and {t2}
        # every treated unit appears in exactly one cohort's member list
        members = [m for uf in res.per_unit.values() for m in uf.member_units]
        assert sorted(members) == ["treated_0", "treated_1", "treated_2"]
        # invariant still holds for cohorts
        q = np.array([uf.prefit_rmspe for uf in res.per_unit.values()])
        assert float(np.sqrt(np.mean(q ** 2))) == pytest.approx(res.design.ind_l2, abs=1e-8)
