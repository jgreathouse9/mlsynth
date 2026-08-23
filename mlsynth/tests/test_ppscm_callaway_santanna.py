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

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from scipy.stats import norm

from mlsynth import PPSCM
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.ppscm_helpers.cs_inference import influence_function_inference
from mlsynth.utils.ppscm_helpers.engine import Conventions, run_multisynth
from mlsynth.utils.ppscm_helpers.setup import prepare_ppscm_inputs

try:                                             # corroboration, not the oracle
    from diff_diff import CallawaySantAnna, SunAbraham
    HAVE_DIFF_DIFF = True
except ImportError:                              # pragma: no cover - CI without it
    HAVE_DIFF_DIFF = False

needs_diff_diff = pytest.mark.skipif(
    not HAVE_DIFF_DIFF, reason="diff-diff not installed")

# The vendored transcription of ``diff-diff`` 3.9.0 at ``d9cd475``, which is the
# oracle for inference. Loaded by path because ``benchmarks/`` is not a package.
_REF_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "benchmarks" / "reference" / "ppscm_cs" / "reference.py")
_spec = importlib.util.spec_from_file_location("_ppscm_cs_reference", _REF_PATH)
REF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REF)


# --------------------------------------------------------------------------- #
# the oracle: Callaway-Sant'Anna's group-time ATT, transcribed
# --------------------------------------------------------------------------- #
def cs_group_time(df, outcome="y", unit="id", time="time", first_treat="first_treat"):
    """``ATT(g, t)`` with a never-treated comparison group and no covariates.

    With those two restrictions the estimator is a two-period, two-group DiD per
    cell, against the base period ``g - 1``::

        ATT(g,t) = mean_{i in G_g}(Y_it - Y_{i,g-1})
                 - mean_{i in C} (Y_it - Y_{i,g-1})

    Transcribed instead of imported so the identity this module exists to prove
    is enforced wherever the suite runs, and not only where ``diff-diff`` is
    installed. The imported package is compared against separately, which is
    what keeps this transcription honest.
    """
    wide = df.pivot(index=unit, columns=time, values=outcome)
    g_of = df.groupby(unit)[first_treat].first()
    never = g_of.index[g_of == 0]
    out = {}
    for g in sorted(x for x in g_of.unique() if x != 0):
        treated = g_of.index[g_of == g]
        for t in wide.columns:
            if t < g or (g - 1) not in wide.columns:
                continue
            d_tr = (wide.loc[treated, t] - wide.loc[treated, g - 1]).mean()
            d_co = (wide.loc[never, t] - wide.loc[never, g - 1]).mean()
            out[(g, t)] = float(d_tr - d_co)
    return out


def oracle_att(df, L):
    """The oracle's ATT over leads ``0 .. L-1``, aggregated as PPSCM aggregates."""
    gt = cs_group_time(df)
    return float(np.mean([v for (g, t), v in gt.items() if 0 <= t - g < L]))


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def staggered(onsets, k=3, n_units=50, n_per=30, seed=0, effect="const",
              sizes=None):
    """Never-treated donors plus cohorts adopting at ``onsets``.

    ``effect`` selects the treatment-effect structure, since an identity that
    only holds for a constant effect would be an artefact of the DGP. ``sizes``
    gives the cohorts different memberships, which is what separates
    Callaway-Sant'Anna's size-weighted aggregation from their unweighted one --
    equal cohorts make the two coincide and hide the distinction.
    """
    rng = np.random.default_rng(seed)
    sizes = [k] * len(onsets) if sizes is None else list(sizes)
    assign, u = {}, 0
    for g, size in zip(onsets, sizes):
        for _ in range(size):
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


# --------------------------------------------------------------------------- #
# the vendored reference, as an inference oracle
# --------------------------------------------------------------------------- #
def ref_parts(df):
    eff, ses, inf = REF.group_time_att(df, "y", "id", "time", "first_treat")
    return eff, ses, inf, df.groupby("id")["first_treat"].first().to_numpy()


def cells_of(df, L):
    """The ``(g, t)`` cells PPSCM aggregates: leads ``0 .. L-1`` of each cohort."""
    eff, _, _, _ = ref_parts(df)
    return [(g, t) for (g, t) in eff if 0 <= t - g < L]


def ref_aggregate(df, L, *, size_weighted=False, wif=True):
    """The reference's aggregate over PPSCM's cells.

    ``size_weighted`` hands it the cohort-share weights PPSCM uses; its own
    default is an unweighted mean over cells, and the two agree only when the
    cohorts are the same size. ``wif=False`` deletes the weight-influence
    correction, which is how its absence is measured instead of assumed.
    """
    eff, _, inf, cohort = ref_parts(df)
    cells = cells_of(df, L)
    w = None
    if size_weighted:
        pg = np.array([(cohort == g).sum() / cohort.shape[0] for g, _ in cells])
        w = pg / pg.sum()
    if wif:
        est, se, _ = REF.aggregate(cells, eff, inf, cohort, weights=w)
        return est, se
    w = np.full(len(cells), 1.0 / len(cells)) if w is None else w
    psi = np.zeros(cohort.shape[0])
    for k, c in enumerate(cells):
        psi[inf[c]["treated_idx"]] += w[k] * inf[c]["treated_inf"]
        psi[inf[c]["control_idx"]] += w[k] * inf[c]["control_inf"]
    est = float(np.array([eff[c] for c in cells]) @ w)
    return est, float(np.sqrt((psi ** 2).sum()))


def _refit(df):
    """The raw ``run_multisynth`` dictionary, for testing the module directly."""
    inputs = prepare_ppscm_inputs(df[["id", "time", "y", "d"]], outcome="y",
                                  treat="d", unitid="id", time="time")
    T, d = inputs.Xy.shape[1], inputs.n_pre
    return run_multisynth(inputs.Xy, inputs.trt, d, T - d, d,
                          fixedeff=True, time_cohort=False,
                          conventions=Conventions(
                              donor_weights="uniform",
                              base_period="pre_treatment",
                              donor_pool="never_treated"))


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
    def test_matches_the_callaway_santanna_oracle(self, name, onsets, k):
        df = staggered(onsets, k=k)
        res = fit_cs_mode(df)
        assert float(res.att) == pytest.approx(oracle_att(df, leads_of(res)),
                                               abs=1e-12), name

    @needs_diff_diff
    @pytest.mark.parametrize("name,onsets,k", ALIGNED)
    def test_matches_callaway_santanna(self, name, onsets, k):
        """Corroboration against an independent implementation."""
        df = staggered(onsets, k=k)
        res = fit_cs_mode(df)
        ref, _ = reference_cs(df, leads_of(res))
        assert float(res.att) == pytest.approx(ref, abs=1e-12), name

    @needs_diff_diff
    @pytest.mark.parametrize("name,onsets,k", ALIGNED)
    def test_the_oracle_agrees_with_the_package(self, name, onsets, k):
        """If these ever disagree, the transcription is what to distrust."""
        df = staggered(onsets, k=k)
        L = leads_of(fit_cs_mode(df))
        ref, _ = reference_cs(df, L)
        assert oracle_att(df, L) == pytest.approx(ref, abs=1e-12), name

    @needs_diff_diff
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
        assert float(res.att) == pytest.approx(oracle_att(df, leads_of(res)),
                                               abs=1e-12), effect

    @pytest.mark.parametrize("seed", range(4))
    def test_identity_holds_across_seeds(self, seed):
        df = staggered((10, 14, 18), seed=seed)
        res = fit_cs_mode(df)
        assert float(res.att) == pytest.approx(oracle_att(df, leads_of(res)),
                                               abs=1e-12)

    def test_event_study_path_matches_lead_for_lead(self):
        """The ATT is one number; the path is where an estimator can hide."""
        df = staggered((10, 14, 18))
        res = fit_cs_mode(df)
        L = leads_of(res)
        gt = cs_group_time(df)
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
        ref = oracle_att(df, leads_of(res))
        assert cs_mode == pytest.approx(ref, abs=1e-12), (
            "never_treated must still match CS exactly")
        assert abs(window - ref) > 1e-4, (
            "augsynth's window pool is expected to differ here")


# =========================================================================== #
# 5. inference -- separate from estimation on purpose
# =========================================================================== #
class TestInference:
    """The interval has to be theirs too.

    Equal point estimates do not make equal intervals, and PPSCM's delete-one
    jackknife is not the Callaway-Sant'Anna standard error. The oracle here is
    the vendored transcription (``benchmarks/reference/ppscm_cs/reference.py``),
    so the claim is enforced wherever the suite runs; ``diff-diff`` corroborates
    it separately in ``test_ppscm_cs_reference.py``.
    """

    def test_per_cell_standard_errors_match(self):
        for name, onsets, k in ALIGNED[:3]:
            df = staggered(onsets, k=k)
            res = fit_cs_mode(df, run_inference=True)
            ours = res.inference_detail.group_time_se
            _, ses, _, _ = ref_parts(df)
            for cell in cells_of(df, leads_of(res)):
                assert ours[cell] == pytest.approx(ses[cell], rel=1e-12), (name, cell)

    def test_aggregated_standard_error_matches(self):
        """Includes the weight-influence term; without it the SE is wrong by
        the amount R's did::aggte corrects for."""
        df = staggered((10, 14, 18))
        res = fit_cs_mode(df, run_inference=True)
        _, se_ref = ref_aggregate(df, leads_of(res))
        assert float(res.inference.standard_error) == pytest.approx(se_ref, rel=1e-10)

    def test_confidence_interval_matches(self):
        df = staggered((10, 14, 18))
        res = fit_cs_mode(df, run_inference=True)
        est, se_ref = ref_aggregate(df, leads_of(res))
        z = float(norm.ppf(0.975))
        assert float(res.inference.ci_lower) == pytest.approx(est - z * se_ref, rel=1e-10)
        assert float(res.inference.ci_upper) == pytest.approx(est + z * se_ref, rel=1e-10)

    @pytest.mark.parametrize("sizes", [(2, 5, 3), (1, 6, 3), (4, 1, 2)])
    def test_unequal_cohorts_use_the_size_weighted_aggregation(self, sizes):
        """PPSCM averages cohorts by size at each horizon, which is
        Callaway-Sant'Anna's dynamic aggregation and not their simple one.

        Equal cohorts hide the difference, so the two aggregations are separated
        here: handed the same cell weights, the reference reproduces PPSCM's
        estimate and standard error exactly at every cohort split.
        """
        df = staggered((10, 14, 18), sizes=sizes)
        res = fit_cs_mode(df, run_inference=True)
        est, se_ref = ref_aggregate(df, leads_of(res), size_weighted=True)
        assert float(res.att) == pytest.approx(est, abs=1e-12), sizes
        assert float(res.inference.standard_error) == pytest.approx(se_ref, rel=1e-10)

    def test_multiplier_bootstrap_bands_are_wider_than_pointwise(self):
        """Uniform bands cover the whole path simultaneously, so they cannot be
        narrower than the pointwise interval."""
        df = staggered((10, 14, 18))
        res = fit_cs_mode(df, run_inference=True, n_boot=2000, cband=True)
        assert (res.inference.ci_upper - res.inference.ci_lower) > 0
        band = res.inference_detail.uniform_band
        point = res.inference_detail.pointwise_band
        assert all(b[1] - b[0] >= p[1] - p[0] - 1e-12
                   for b, p in zip(band, point))
        assert res.inference_detail.critical_value > float(norm.ppf(0.975))

    def test_inference_is_not_silently_jackknife(self):
        """The preset must report which inference it ran; naming CS while
        reporting PPSCM's jackknife is the failure #459/#463 bottomed out at."""
        df = staggered((10, 14))
        res = fit_cs_mode(df, run_inference=True)
        assert res.method_details.parameters_used.get("inference_method") == \
            "influence_function"
        assert res.inference.method == "influence_function"

    def test_the_weight_influence_term_is_carried(self):
        """Dropping it leaves a standard error that is finite, plausible and
        too small, so its absence is measured and not assumed."""
        df = staggered((10, 14, 18))
        res = fit_cs_mode(df, run_inference=True)
        psi = res.inference_detail.influence
        H = psi.shape[1]
        naive = ref_aggregate(df, H, wif=False)[1]
        assert float(res.inference.standard_error) != pytest.approx(naive, rel=1e-6)

    def test_the_influence_function_reproduces_every_reported_standard_error(self):
        """``se`` is a functional of ``influence`` and not a parallel number."""
        df = staggered((10, 14, 18))
        res = fit_cs_mode(df, run_inference=True)
        psi = res.inference_detail.influence
        assert psi.shape == (50, len(res.event_study.tau))
        for h, s in enumerate(res.event_study.se):
            assert s == pytest.approx(float(np.sqrt((psi[:, h] ** 2).sum())), rel=1e-12)
        agg = psi.mean(axis=1)
        assert float(res.inference.standard_error) == pytest.approx(
            float(np.sqrt((agg ** 2).sum())), rel=1e-12)

    def test_the_comparison_group_enters_with_the_opposite_sign(self):
        """What makes a cell a difference in differences and not a sum.

        A never-treated unit whose outcome rose more than its group's average
        pulls the estimate down, so its contribution to the influence function
        is negative. No standard error can see this. The treated and control
        blocks sit on disjoint units, each cell's control influence sums to
        zero, and the weight-influence term is a single constant across
        never-treated units, so flipping the entire control block changes
        ``sum_i psi_i^2`` by ``4c * sum_i A_i = 0`` -- every reported number
        stays bit-identical while the object they are computed from is wrong.
        Measured: the influence matrix moves by 1.36e-01 and its norm by 0.

        ``influence`` is public and documented as what a caller re-aggregates,
        so its sign convention is part of the contract and not an internal
        detail.
        """
        df = staggered((12,), k=3)          # one cohort, so the wif term is 0
        res = fit_cs_mode(df, run_inference=True)
        psi = res.inference_detail.influence
        wide = df.pivot(index="id", columns="time", values="y")
        g_of = df.groupby("id")["first_treat"].first().reindex(wide.index)
        never = np.flatnonzero((g_of == 0).to_numpy())
        d_co = (wide.iloc[never][12].to_numpy() - wide.iloc[never][11].to_numpy())
        want = -(d_co - d_co.mean()) / never.size
        assert np.allclose(psi[never, 0], want, atol=1e-12), (
            "the comparison group's influence contributions have the wrong sign "
            "or the wrong scale")

    def test_the_influence_function_is_mean_zero(self):
        """A mean-zero influence function is what makes the sum of squares a
        variance; a non-zero mean is a location error masquerading as one."""
        df = staggered((10, 14, 18))
        res = fit_cs_mode(df, run_inference=True)
        psi = res.inference_detail.influence
        for h in range(psi.shape[1]):
            assert float(psi[:, h].sum()) == pytest.approx(0.0, abs=1e-12), h

    def test_the_cells_rebuild_the_event_study(self):
        """``group_time_att`` is the aggregate's parts, so it must recombine."""
        df = staggered((10, 14, 18), sizes=(2, 5, 3))
        res = fit_cs_mode(df, run_inference=True)
        gta = res.inference_detail.group_time_att
        n1 = {}
        for f in res.per_unit.values():
            n1[f.adoption_time] = n1.get(f.adoption_time, 0) + f.n_units
        for h, tau in enumerate(res.event_study.tau):
            cs = [(g, g + h) for g in sorted(n1) if (g, g + h) in gta]
            S = sum(n1[g] for g, _ in cs)
            rebuilt = sum(n1[g] * gta[(g, t)] for g, t in cs) / S
            assert rebuilt == pytest.approx(float(tau), abs=1e-12), h

    def test_the_cumulative_band_runs_off_the_influence_functions(self):
        df = staggered((10, 14, 18))
        res = fit_cs_mode(df, run_inference=True, cumulative_band=True, n_boot=500)
        band = res.inference_detail.cumulative
        assert band.method == "influence_function"
        assert band.n_replicates == 500
        assert np.all(band.upper > band.lower)


# =========================================================================== #
# 6. the influence-function mode refuses what it cannot compute
# =========================================================================== #
class TestInferenceRestrictions:
    """The formula is for uniform weights, a ``g-1`` baseline and a
    never-treated pool. Outside those it is wrong and not approximate, so the
    configuration refuses it instead of returning a number."""

    @pytest.mark.parametrize("field,value", [
        ("donor_weights", "scm"), ("base_period", "all_pre"),
        ("donor_pool", "window"), ("donor_pool", "not_yet_treated"),
        ("fixedeff", False)])
    def test_a_broken_convention_is_refused(self, field, value):
        df = staggered((10, 14))
        with pytest.raises(MlsynthConfigError, match="Callaway-Sant'Anna"):
            fit_cs_mode(df, run_inference=True,
                        inference_method="influence_function", **{field: value})

    def test_the_message_names_the_convention_that_is_wrong(self):
        df = staggered((10, 14))
        with pytest.raises(MlsynthConfigError, match="donor_weights='scm'"):
            fit_cs_mode(df, run_inference=True,
                        inference_method="influence_function", donor_weights="scm")

    def test_the_preset_selects_it_only_when_the_conventions_survive(self):
        """Both halves, because either alone is satisfied by doing nothing.

        Left to itself the preset must select the Callaway-Sant'Anna interval;
        asked for alongside augsynth's donor pool it must not, since that is a
        coherent question whose answer is the jackknife and not a configuration
        error about a field the caller never set.
        """
        df = staggered((10, 14))
        assert fit_cs_mode(df, run_inference=True).inference.method == \
            "influence_function"
        assert fit_cs_mode(df, run_inference=True,
                           donor_pool="window").inference.method == "jackknife"

    def test_an_unknown_inference_method_is_refused(self):
        df = staggered((10, 14))
        with pytest.raises(MlsynthConfigError, match="influence_function"):
            fit_cs_mode(df, run_inference=True, inference_method="analytic")

    def test_cband_needs_the_influence_function(self):
        """Matched on ``influence_function`` and not on ``cband``: an unknown
        field produces a pydantic message that carries the field's own name, so
        matching that alone passes against a build where ``cband`` does not
        exist."""
        df = staggered((10, 14))
        with pytest.raises(MlsynthConfigError, match="influence_function"):
            fit_cs_mode(df, run_inference=True, inference_method="jackknife",
                        cband=True)

    def test_cband_needs_inference_on(self):
        df = staggered((10, 14))
        with pytest.raises(MlsynthConfigError, match="run_inference"):
            fit_cs_mode(df, run_inference=False, cband=True)

    def test_a_panel_with_no_never_treated_unit_is_caught_at_ingestion(self):
        """The comparison group is missing before any inference is chosen.

        Not a test of the influence function: ``prepare_ppscm_inputs`` refuses
        this panel, and the layer that refuses it is the claim. The inference
        module's own empty-pool guard is reached by calling it directly
        (:meth:`TestInfluenceModuleEdges.test_a_cohort_with_no_donor_is_reported_not_divided_by`).
        """
        df = staggered((10, 14), k=25)          # 50 units, none left untreated
        with pytest.raises(MlsynthDataError, match="never-treated"):
            fit_cs_mode(df, run_inference=True)

    def test_the_other_methods_leave_the_cell_fields_empty(self):
        """A jackknife produces no per-cell quantities, and must not appear to."""
        df = staggered((10, 14))
        res = fit_cs_mode(df, run_inference=True, donor_pool="window")
        d = res.inference_detail
        assert d.group_time_att is None and d.group_time_se is None
        assert d.pointwise_band is None and d.uniform_band is None
        assert d.influence is None and d.critical_value is None


# =========================================================================== #
# 7. properties -- what must hold on any panel, not just the six above
# =========================================================================== #
@settings(deadline=None, max_examples=25,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(onsets=st.lists(st.integers(8, 20), min_size=1, max_size=3,
                       unique=True).map(sorted),
       sizes=st.lists(st.integers(1, 5), min_size=1, max_size=3),
       seed=st.integers(0, 2 ** 16))
def test_the_influence_function_matches_the_reference_on_any_panel(
        onsets, sizes, seed):
    """Estimate, standard error and every cell, against the transcription.

    The parametrized cases above are six panels someone chose. This is the same
    claim on panels nobody chose: cohorts of unequal size, adoptions at
    arbitrary spacings, and as few as one treated unit.
    """
    sizes = (sizes * len(onsets))[:len(onsets)]
    df = staggered(tuple(onsets), sizes=sizes, seed=seed)
    res = fit_cs_mode(df, run_inference=True)
    L = leads_of(res)
    _, ses, _, _ = ref_parts(df)
    for cell in cells_of(df, L):
        assert res.inference_detail.group_time_se[cell] == pytest.approx(
            ses[cell], rel=1e-10), cell
    est, se_ref = ref_aggregate(df, L, size_weighted=True)
    assert float(res.att) == pytest.approx(est, abs=1e-10)
    assert float(res.inference.standard_error) == pytest.approx(se_ref, rel=1e-9)


@settings(deadline=None, max_examples=25,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(onsets=st.lists(st.integers(8, 20), min_size=1, max_size=3,
                       unique=True).map(sorted),
       seed=st.integers(0, 2 ** 16))
def test_the_standard_error_is_positive_and_the_interval_brackets_the_estimate(
        onsets, seed):
    df = staggered(tuple(onsets), seed=seed)
    res = fit_cs_mode(df, run_inference=True)
    se = float(res.inference.standard_error)
    assert np.isfinite(se) and se > 0
    assert res.inference.ci_lower < res.att < res.inference.ci_upper
    assert np.all(np.isfinite(res.event_study.se))
    assert np.all(res.event_study.se > 0)


@settings(deadline=None, max_examples=20,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(shift=st.floats(-50.0, 50.0), scale=st.floats(0.2, 5.0))
def test_the_standard_error_scales_with_the_outcome_and_ignores_its_level(
        shift, scale):
    """A standard error is in the outcome's units: rescaling the outcome
    rescales it exactly, and shifting the outcome leaves it alone."""
    df = staggered((10, 14, 18))
    base = fit_cs_mode(df, run_inference=True)
    moved = df.copy()
    moved["y"] = shift + scale * moved["y"]
    got = fit_cs_mode(moved, run_inference=True)
    assert float(got.inference.standard_error) == pytest.approx(
        scale * float(base.inference.standard_error), rel=1e-9)
    assert float(got.att) == pytest.approx(scale * float(base.att), abs=1e-9)


@settings(deadline=None, max_examples=15,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(alpha=st.floats(0.005, 0.5), seed=st.integers(0, 2 ** 16))
def test_a_tighter_level_gives_a_wider_interval_around_the_same_estimate(
        alpha, seed):
    df = staggered((10, 14, 18), seed=seed)
    wide = fit_cs_mode(df, run_inference=True, alpha=alpha)
    tight = fit_cs_mode(df, run_inference=True, alpha=alpha / 2.0)
    assert float(tight.att) == pytest.approx(float(wide.att), abs=1e-12)
    assert (tight.inference.ci_upper - tight.inference.ci_lower) > \
        (wide.inference.ci_upper - wide.inference.ci_lower)


@settings(deadline=None, max_examples=15,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(0, 2 ** 16))
def test_relabelling_the_units_does_not_move_the_answer(seed):
    """The influence function is indexed by unit; a permutation of the unit
    labels must permute it and change nothing else."""
    df = staggered((10, 14, 18))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(df["id"].nunique())
    shuffled = df.copy()
    shuffled["id"] = shuffled["id"].map(lambda u: int(perm[u]))
    a = fit_cs_mode(df, run_inference=True)
    b = fit_cs_mode(shuffled.sort_values(["id", "time"]), run_inference=True)
    assert float(b.att) == pytest.approx(float(a.att), abs=1e-10)
    assert float(b.inference.standard_error) == pytest.approx(
        float(a.inference.standard_error), rel=1e-10)


# =========================================================================== #
# 8. the module's own edges
# =========================================================================== #
class TestInfluenceModuleEdges:
    def test_a_cohort_with_no_donor_is_reported_not_divided_by(self):
        df = staggered((10, 14))
        res = fit_cs_mode(df, run_inference=False)
        fit = _refit(df)
        fit["donors"] = {g: np.asarray([], dtype=int) for g in fit["groups"]}
        with pytest.raises(MlsynthDataError, match="comparison group"):
            influence_function_inference(
                fit, np.arange(1, 31), alpha=0.05, att_full=float(res.att),
                per_time_full=res.event_study.tau)

    def test_a_panel_with_no_estimable_cell_is_reported(self):
        df = staggered((10, 14))
        res = fit_cs_mode(df, run_inference=False)
        fit = _refit(df)
        # Push every adoption past the last observed column.
        fit["adopt_of"] = {g: 10_000 for g in fit["groups"]}
        fit["res"] = {10_000: next(iter(fit["res"].values()))}
        with pytest.raises(MlsynthDataError, match="no estimable"):
            influence_function_inference(
                fit, np.arange(1, 20_001), alpha=0.05, att_full=float(res.att),
                per_time_full=res.event_study.tau)

    def test_one_treated_unit_carries_no_weight_influence_term(self):
        """With a single cohort the group share is not estimated against
        anything, so the correction is exactly zero and the aggregate standard
        error is the plain weighted one."""
        df = staggered((12,), k=1)
        res = fit_cs_mode(df, run_inference=True)
        psi = res.inference_detail.influence
        _, ses, inf, _ = ref_parts(df)
        for h in range(psi.shape[1]):
            cell = (12, 12 + h)
            assert res.inference_detail.group_time_se[cell] == pytest.approx(
                ses[cell], rel=1e-12)
            assert float(np.sqrt((psi[:, h] ** 2).sum())) == pytest.approx(
                ses[cell], rel=1e-12)

    def test_time_cohort_collapses_to_the_same_cells(self):
        """A Callaway-Sant'Anna cell is indexed by adoption time, so pooling
        units into cohorts up front must not change the inference."""
        df = staggered((10, 14, 18))
        a = fit_cs_mode(df, run_inference=True)
        b = fit_cs_mode(df, run_inference=True, time_cohort=True)
        assert float(b.att) == pytest.approx(float(a.att), abs=1e-12)
        assert float(b.inference.standard_error) == pytest.approx(
            float(a.inference.standard_error), rel=1e-12)
        assert b.inference_detail.group_time_se.keys() == \
            a.inference_detail.group_time_se.keys()

    def test_the_multipliers_are_mammens_and_not_rademacher(self):
        """Mammen's two-point law has mean 0, variance 1 and third moment 1.

        The third moment is the whole reason ``did::mboot`` uses it over
        Rademacher signs, and it is the only one of the three that tells them
        apart -- so a band tabulated from signs looks entirely reasonable and
        has lost the refinement it was drawn for.
        """
        from mlsynth.utils.ppscm_helpers.cs_inference import _KAPPA, _PROB
        vals = np.array([1.0 - _KAPPA, _KAPPA])
        probs = np.array([_PROB, 1.0 - _PROB])
        assert float(probs @ vals) == pytest.approx(0.0, abs=1e-12)
        assert float(probs @ vals ** 2) == pytest.approx(1.0, abs=1e-12)
        assert float(probs @ vals ** 3) == pytest.approx(1.0, abs=1e-12)

    def test_a_horizon_no_cohort_reaches_comes_back_empty(self):
        """A horizon past the end of the panel has no cell, and it reports
        ``NaN`` -- not a zero effect with a zero standard error, which is what
        a silently skipped horizon looks like once it reaches an event-study
        plot.

        Reachable only by asking the module directly. PPSCM's ceiling on
        ``n_leads`` is the longest post window (#481), which is by construction
        the last horizon any cohort observes, so no configuration produces an
        empty one.

        The arithmetic: cohorts adopt at columns 9 and 13 of a 30-column panel,
        so the earlier one still reaches horizon 20 and only ``h >= 21`` is
        empty. ``_refit`` poses the balanced window of 17, so 25 horizons
        leaves exactly four with no cell -- 21 would leave none, and every
        horizon would come back populated.
        """
        df = staggered((10, 14))
        res = fit_cs_mode(df, run_inference=False)
        fit = _refit(df)
        assert fit["n_leads"] == 17
        H = 25
        fit["n_leads"] = H
        pt = np.concatenate([res.event_study.tau, np.full(H - 17, np.nan)])
        out = influence_function_inference(
            fit, np.arange(1, 31), alpha=0.05, att_full=float(res.att),
            per_time_full=pt)
        assert np.all(np.isfinite(out.per_time_se[:-4]))
        assert np.all(np.isnan(out.per_time_se[-4:]))
        assert np.all(np.isnan(out.influence[:, -4:]))
        assert np.isfinite(out.se) and out.se > 0

    def test_the_cells_are_keyed_by_public_time_labels(self):
        """``(2014, 2017)`` and not ``(9, 12)``: a reader must not have to know
        the column positions the estimator works in."""
        df = staggered((10, 14))
        df = df.assign(time=df["time"] + 2000)
        res = fit_cs_mode(df, run_inference=True)
        keys = res.inference_detail.group_time_se.keys()
        assert (2010, 2010) in keys and (2014, 2014) in keys
        assert all(g >= 2010 and t >= g for g, t in keys)
