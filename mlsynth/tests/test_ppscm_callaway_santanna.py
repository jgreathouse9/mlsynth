"""PPSCM reaches the Callaway-Sant'Anna estimator exactly, and says so honestly.

Ben-Michael, Feller & Rothstein (2022) p.369 note that with uniform donor
weights their intercept-shifted partially-pooled SCM "is equivalent to recent
proposals for DiD estimators that allow for treatment effect heterogeneity with
a fixed donor set per treatment time cohort (see Callaway & Sant'Anna, 2020; Sun
& Abraham, 2020)". Measured, that is not an approximation: three independent
implementations agree to 1e-14 once three conventions are aligned (#465).

The three, each separable and each isolated by measurement:

* donor weights -- CS/SA weight donors uniformly, PPSCM solves the SCM QP.
  Uniform is not reachable by sending ``lam`` up: the weights saturate at
  ``max|w - 1/J| = 3.47e-03`` and do not move between ``lam = 1e9`` and ``1e18``
  at any ``nu``, so it has to be imposed.
* base period -- ``fit_feff`` subtracts each unit's mean over its whole
  pre-adoption window (augsynth's convention); CS baselines on ``g-1``. The
  difference is a per-cohort level shift.
* donor eligibility -- augsynth admits units untreated through the cohort's
  estimation window, which is neither ``never_treated`` nor CS's
  ``not_yet_treated``. It coincides with ``never_treated`` only when every other
  cohort adopts inside the window, and that is exactly when the identity holds.

The identity tests pin the estimate at 1e-12, which needs no tolerance
negotiation because the true agreement is two orders tighter. The isolation
tests pin what each convention does on its own, so a regression names the
convention. The divergence test pins the regime where the three legitimately
disagree, as a documented difference and not a bug -- a benchmark panel with
adoptions spread across a long window lands there, so it must not be read as
breakage.

Inference is tested separately from estimation on purpose. Equal point
estimates do not make equal intervals: CS reports influence-function analytical
standard errors with an optional multiplier bootstrap, PPSCM reports jackknife,
and reporting one while naming the other is the failure #459 and #463 both
bottomed out at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import PPSCM
from mlsynth.exceptions import MlsynthConfigError

dd = pytest.importorskip("diff_diff", reason="CS reference implementation")
from diff_diff import CallawaySantAnna, SunAbraham  # noqa: E402


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def staggered(onsets, k=3, n_units=50, n_per=30, seed=0, effect="const"):
    """Never-treated donors plus cohorts adopting at ``onsets``.

    ``effect`` selects the treatment-effect structure, since an identity that
    only holds for a constant effect would be an artefact of the DGP.
    """
    rng = np.random.default_rng(seed)
    assign, u = {}, 0
    for g in onsets:
        for _ in range(k):
            assign[u] = g
            u += 1
    rows = []
    for unit in range(n_units):
        base = rng.normal(10, 2)
        walk = np.cumsum(rng.normal(0, 0.3, n_per))
        g = assign.get(unit)
        u_eff = rng.normal()
        for t in range(1, n_per + 1):
            on = g is not None and t >= g
            if not on:
                eff = 0.0
            elif effect == "const":
                eff = 2.0
            elif effect == "cohort":
                eff = 1.0 + onsets.index(g)
            elif effect == "dynamic":
                eff = 0.3 * (t - g + 1)
            elif effect == "unit":
                eff = 2.0 + u_eff
            else:                                    # pragma: no cover - typo guard
                raise ValueError(effect)
            rows.append((unit, t, base + walk[t - 1] + 0.5 * rng.normal() + eff,
                         int(on), g if g is not None else 0))
    return pd.DataFrame(rows, columns=["id", "time", "y", "d", "first_treat"])


CS_CFG = {"method": "callaway_santanna"}


def fit_cs_mode(df, **over):
    cfg = {"df": df[["id", "time", "y", "d"]], "outcome": "y", "treat": "d",
           "unitid": "id", "time": "time", "display_graphs": False,
           "run_inference": False, **CS_CFG, **over}
    return PPSCM(cfg).fit()


def reference_cs(df, L, control_group="never_treated"):
    r = CallawaySantAnna(control_group=control_group, estimation_method="dr",
                         n_bootstrap=0).fit(df, outcome="y", unit="id",
                                            time="time", first_treat="first_treat")
    gt = {(g, t): v["effect"] for (g, t), v in r.group_time_effects.items()}
    return float(np.mean([v for (g, t), v in gt.items() if 0 <= t - g < L])), r


def reference_sa(df, L):
    r = SunAbraham(control_group="never_treated").fit(
        df, outcome="y", unit="id", time="time", first_treat="first_treat")
    es = r.event_study_effects
    return float(np.mean([(v["effect"] if isinstance(v, dict) else v)
                          for l, v in sorted(es.items()) if 0 <= l < L])), r


def leads_of(res):
    return len(next(iter(res.per_unit.values())).tau)


# shapes where every non-focal cohort adopts inside the focal cohort's window,
# so the donor pools coincide and the identity is exact
ALIGNED = [("one-cohort", (12,), 3), ("two-cohort", (10, 14), 3),
           ("three-cohort", (10, 14, 18), 3), ("three-cohort-small", (10, 14, 18), 2),
           ("three-cohort-large", (10, 14, 18), 4)]
EFFECTS = ["const", "cohort", "dynamic", "unit"]


# =========================================================================== #
# 1. the configuration surface -- starts red
# =========================================================================== #
class TestConfigSurface:
    def test_preset_sets_all_three_conventions(self):
        df = staggered((10, 14))
        res = fit_cs_mode(df)
        d = res.method_details.parameters_used
        assert d.get("donor_weights") == "uniform"
        assert d.get("base_period") == "pre_treatment"
        assert d.get("donor_pool") == "never_treated"

    @pytest.mark.parametrize("field,value", [
        ("donor_weights", "uniform"), ("donor_weights", "scm"),
        ("base_period", "all_pre"), ("base_period", "pre_treatment"),
        ("donor_pool", "window"), ("donor_pool", "never_treated"),
        ("donor_pool", "not_yet_treated")])
    def test_each_convention_is_settable(self, field, value):
        df = staggered((10, 14))
        res = fit_cs_mode(df, **{field: value})
        assert res.method_details.parameters_used[field] == value

    @pytest.mark.parametrize("field,bad", [
        ("donor_weights", "equal"), ("base_period", "g-1"),
        ("donor_pool", "not_yet"), ("method", "callaway")])
    def test_an_unknown_convention_is_refused(self, field, bad):
        df = staggered((10, 14))
        with pytest.raises((MlsynthConfigError, ValueError)):
            fit_cs_mode(df, **{field: bad})

    def test_the_defaults_are_augsynth(self):
        """The preset is opt-in; plain PPSCM keeps multisynth's conventions."""
        df = staggered((10, 14))
        cfg = {"df": df[["id", "time", "y", "d"]], "outcome": "y", "treat": "d",
               "unitid": "id", "time": "time", "display_graphs": False,
               "run_inference": False}
        d = PPSCM(cfg).fit().method_details.parameters_used
        assert d.get("donor_weights") == "scm"
        assert d.get("base_period") == "all_pre"
        assert d.get("donor_pool") == "window"


# =========================================================================== #
# 2. the identity -- the whole point
# =========================================================================== #
class TestIdentityWithCallawaySantAnna:
    @pytest.mark.parametrize("name,onsets,k", ALIGNED)
    def test_matches_callaway_santanna(self, name, onsets, k):
        df = staggered(onsets, k=k)
        res = fit_cs_mode(df)
        ref, _ = reference_cs(df, leads_of(res))
        assert float(res.att) == pytest.approx(ref, abs=1e-12), name

    @pytest.mark.parametrize("name,onsets,k", ALIGNED)
    def test_matches_sun_abraham(self, name, onsets, k):
        df = staggered(onsets, k=k)
        res = fit_cs_mode(df)
        ref, _ = reference_sa(df, leads_of(res))
        assert float(res.att) == pytest.approx(ref, abs=1e-11), name

    @pytest.mark.parametrize("effect", EFFECTS)
    def test_identity_survives_effect_heterogeneity(self, effect):
        """A DGP-specific identity would be worthless; vary the effect."""
        df = staggered((10, 14, 18), effect=effect)
        res = fit_cs_mode(df)
        ref, _ = reference_cs(df, leads_of(res))
        assert float(res.att) == pytest.approx(ref, abs=1e-12), effect

    @pytest.mark.parametrize("seed", range(4))
    def test_identity_holds_across_seeds(self, seed):
        df = staggered((10, 14, 18), seed=seed)
        res = fit_cs_mode(df)
        ref, _ = reference_cs(df, leads_of(res))
        assert float(res.att) == pytest.approx(ref, abs=1e-12)

    def test_event_study_path_matches_lead_for_lead(self):
        """The ATT is one number; the path is where an estimator can hide."""
        df = staggered((10, 14, 18))
        res = fit_cs_mode(df)
        L = leads_of(res)
        _, ref = reference_cs(df, L)
        gt = {(g, t): v["effect"] for (g, t), v in ref.group_time_effects.items()}
        for l in range(L):
            ours = float(np.nanmean([f.tau[l] for f in res.per_unit.values()]))
            theirs = float(np.mean([v for (g, t), v in gt.items() if t - g == l]))
            assert ours == pytest.approx(theirs, abs=1e-12), f"lead {l}"


# =========================================================================== #
# 3. each convention, isolated
# =========================================================================== #
class TestConventionsIsolated:
    def test_scm_weights_move_the_answer(self):
        df = staggered((10, 14, 18))
        cs = float(fit_cs_mode(df).att)
        scm = float(fit_cs_mode(df, donor_weights="scm").att)
        assert abs(scm - cs) > 1e-6, "SCM weights should not reproduce uniform"

    def test_the_baseline_is_a_per_cohort_level_shift(self):
        """all-pre-lags vs g-1: the event-study *shape* must not move."""
        df = staggered((10, 14, 18))
        a = fit_cs_mode(df)
        b = fit_cs_mode(df, base_period="all_pre")
        L = leads_of(a)
        diffs = [float(np.nanmean([f.tau[l] for f in b.per_unit.values()])
                       - np.nanmean([f.tau[l] for f in a.per_unit.values()]))
                 for l in range(L)]
        assert max(diffs) - min(diffs) < 1e-9, f"not a level shift: {diffs}"
        assert abs(diffs[0]) > 1e-6, "expected a non-zero shift"

    def test_donor_pool_only_matters_when_a_cohort_survives_the_window(self):
        df_aligned = staggered((10, 14, 18))
        a1 = float(fit_cs_mode(df_aligned).att)
        a2 = float(fit_cs_mode(df_aligned, donor_pool="window").att)
        assert a1 == pytest.approx(a2, abs=1e-12), (
            "pools coincide here, so the setting must not matter")


# =========================================================================== #
# 4. the regime where they legitimately diverge
# =========================================================================== #
class TestDocumentedDivergence:
    def test_window_pool_differs_when_a_later_cohort_survives(self):
        """Four cohorts over a long window: augsynth admits the last cohort as a
        donor for the first, CS does not. Measured at ~1.2e-02 -- pinned as a
        documented difference so it is not read as breakage."""
        df = staggered((8, 12, 16, 20))
        cs_mode = float(fit_cs_mode(df).att)
        window = float(fit_cs_mode(df, donor_pool="window").att)
        res = fit_cs_mode(df)
        ref, _ = reference_cs(df, leads_of(res))
        assert cs_mode == pytest.approx(ref, abs=1e-12), (
            "never_treated must still match CS exactly")
        assert abs(window - ref) > 1e-4, (
            "augsynth's window pool is expected to differ here")


# =========================================================================== #
# 5. inference -- separate from estimation on purpose
# =========================================================================== #
class TestInference:
    @pytest.mark.parametrize("name,onsets,k", ALIGNED[:3])
    def test_per_cell_standard_errors_match(self, name, onsets, k):
        df = staggered(onsets, k=k)
        res = fit_cs_mode(df, run_inference=True)
        _, ref = reference_cs(df, leads_of(res))
        ours = res.inference_detail.group_time_se
        for (g, t), v in ref.group_time_effects.items():
            if (g, t) in ours and np.isfinite(v["se"]):
                assert ours[(g, t)] == pytest.approx(v["se"], rel=1e-9), (g, t)

    def test_aggregated_standard_error_matches(self):
        """Includes the weight-influence term; without it the SE is wrong by
        the amount R's did::aggte corrects for."""
        df = staggered((10, 14, 18))
        res = fit_cs_mode(df, run_inference=True)
        _, ref = reference_cs(df, leads_of(res))
        assert float(res.inference.standard_error) == pytest.approx(
            float(ref.se), rel=1e-8)

    def test_confidence_interval_matches(self):
        df = staggered((10, 14, 18))
        res = fit_cs_mode(df, run_inference=True)
        _, ref = reference_cs(df, leads_of(res))
        lo, hi = ref.conf_int
        assert float(res.inference.ci_lower) == pytest.approx(lo, rel=1e-8)
        assert float(res.inference.ci_upper) == pytest.approx(hi, rel=1e-8)

    def test_multiplier_bootstrap_bands_are_wider_than_pointwise(self):
        """Uniform bands cover the whole path simultaneously, so they cannot be
        narrower than the pointwise interval."""
        df = staggered((10, 14, 18))
        res = fit_cs_mode(df, run_inference=True, n_bootstrap=200, cband=True)
        assert (res.inference.ci_upper - res.inference.ci_lower) > 0
        band = res.inference_detail.uniform_band
        point = res.inference_detail.pointwise_band
        assert all(b[1] - b[0] >= p[1] - p[0] - 1e-12
                   for b, p in zip(band, point))

    def test_inference_is_not_silently_jackknife(self):
        """The preset must report which inference it ran; naming CS while
        reporting PPSCM's jackknife is the failure #459/#463 bottomed out at."""
        df = staggered((10, 14))
        res = fit_cs_mode(df, run_inference=True)
        assert res.method_details.parameters_used.get("inference_method") == \
            "influence_function"
