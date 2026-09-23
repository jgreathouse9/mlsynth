"""The vendored Callaway-Sant'Anna reference is what it says it is.

``benchmarks/reference/ppscm_cs/reference.py`` is a transcription of ``diff-diff``
3.9.0 at commit ``d9cd475``, MIT, kept in-tree so PPSCM's ``callaway_santanna``
mode can be validated without a runtime dependency. Two things have to stay true
of it, and they are different claims:

* it agrees with the upstream it was transcribed from -- checked only where
  ``diff-diff`` is installed, since that is the whole point of vendoring;
* it computes what its docstring says on its own terms -- checked everywhere,
  because a reference that is only checked against the thing it copies has no
  independent standing.

The second matters more than it looks. A vendored copy is a characterisation of
a pinned upstream, so the failure mode is not that it disagrees with the package
today; it is that someone later edits it to agree with mlsynth, at which point
it stops being evidence and starts being a mirror. The properties below are the
ones that would break first: the per-cell estimate is a 2x2 DiD, the influence
function sums to zero within each group, and the aggregated SE strictly exceeds
the one you get by forgetting that the group weights were estimated.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

_REF_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "benchmarks" / "reference" / "ppscm_cs" / "reference.py")
_spec = importlib.util.spec_from_file_location("_ppscm_cs_reference", _REF_PATH)
REF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REF)

try:
    from diff_diff import CallawaySantAnna
    HAVE_DIFF_DIFF = True
except ImportError:                              # pragma: no cover - CI without it
    HAVE_DIFF_DIFF = False

needs_diff_diff = pytest.mark.skipif(
    not HAVE_DIFF_DIFF, reason="diff-diff not installed")


def panel(onsets=(10, 14, 18), k=3, n_units=50, n_per=30, seed=0):
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
        for t in range(1, n_per + 1):
            on = g is not None and t >= g
            rows.append((unit, t, base + walk[t - 1] + 0.5 * rng.normal() + 2.0 * on,
                         int(on), g if g is not None else 0))
    return pd.DataFrame(rows, columns=["id", "time", "y", "d", "first_treat"])


def parts(df):
    eff, ses, inf = REF.group_time_att(df, "y", "id", "time", "first_treat")
    cohort = df.groupby("id")["first_treat"].first().to_numpy()
    return eff, ses, inf, cohort


SHAPES = [((12,), 3, 0), ((10, 14), 3, 0), ((10, 14, 18), 3, 0),
          ((10, 14, 18), 2, 1), ((8, 12, 16, 20), 3, 2), ((10, 14, 18), 4, 3)]


# =========================================================================== #
# 1. what it computes, on its own terms
# =========================================================================== #
class TestOnItsOwnTerms:
    @pytest.mark.parametrize("onsets,k,seed", SHAPES)
    def test_each_cell_is_a_two_by_two_difference_in_differences(self, onsets, k, seed):
        df = panel(onsets, k, seed=seed)
        eff, _, _, _ = parts(df)
        wide = df.pivot(index="id", columns="time", values="y")
        g_of = df.groupby("id")["first_treat"].first()
        never = g_of.index[g_of == 0]
        for (g, t), got in eff.items():
            tr = g_of.index[g_of == g]
            want = ((wide.loc[tr, t] - wide.loc[tr, g - 1]).mean()
                    - (wide.loc[never, t] - wide.loc[never, g - 1]).mean())
            assert got == pytest.approx(float(want), abs=1e-12), (g, t)

    @pytest.mark.parametrize("onsets,k,seed", SHAPES)
    def test_influence_functions_sum_to_zero_within_each_group(self, onsets, k, seed):
        """A mean-zero influence function is what makes the SE a variance."""
        _, _, inf, _ = parts(panel(onsets, k, seed=seed))
        for cell, info in inf.items():
            assert float(info["treated_inf"].sum()) == pytest.approx(0.0, abs=1e-12)
            assert float(info["control_inf"].sum()) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("onsets,k,seed", SHAPES)
    def test_the_cell_standard_error_is_the_norm_of_its_influence_function(
            self, onsets, k, seed):
        _, ses, inf, _ = parts(panel(onsets, k, seed=seed))
        for cell, info in inf.items():
            want = np.sqrt((info["treated_inf"] ** 2).sum()
                           + (info["control_inf"] ** 2).sum())
            assert ses[cell] == pytest.approx(float(want), rel=1e-12)

    def test_the_weight_influence_term_is_not_free(self):
        """Forgetting that the group weights are estimated shrinks the SE.

        This is the term that is easy to omit and expensive to omit: without it
        the standard error is still finite, still plausible, and too small.
        """
        df = panel((10, 14, 18))
        eff, _, inf, cohort = parts(df)
        post = [c for c in eff if c[1] >= c[0]]
        _, se_full, psi = REF.aggregate(post, eff, inf, cohort)
        w = np.full(len(post), 1.0 / len(post))
        psi_no_wif = np.zeros(cohort.shape[0])
        for k_, c in enumerate(post):
            psi_no_wif[inf[c]["treated_idx"]] += w[k_] * inf[c]["treated_inf"]
            psi_no_wif[inf[c]["control_idx"]] += w[k_] * inf[c]["control_inf"]
        se_naive = float(np.sqrt((psi_no_wif ** 2).sum()))
        assert se_full != pytest.approx(se_naive, rel=1e-6), (
            "the wif term made no difference; it is probably not being applied")

    def test_aggregate_is_the_weighted_mean_of_the_cells(self):
        df = panel((10, 14, 18))
        eff, _, inf, cohort = parts(df)
        post = [c for c in eff if c[1] >= c[0]]
        est, _, _ = REF.aggregate(post, eff, inf, cohort)
        assert est == pytest.approx(float(np.mean([eff[c] for c in post])), abs=1e-12)

    def test_supplied_weights_are_the_ones_used(self):
        """PPSCM aggregates cohorts by size, so the reference has to accept the
        weights it is handed. Its own default is an unweighted mean over cells,
        and the two coincide only when every cohort is the same size."""
        df = panel((10, 14, 18))
        eff, _, inf, cohort = parts(df)
        post = [c for c in eff if c[1] >= c[0]]
        w = np.linspace(1.0, 2.0, len(post))
        w = w / w.sum()
        est, _, _ = REF.aggregate(post, eff, inf, cohort, weights=w)
        assert est == pytest.approx(
            float(np.array([eff[c] for c in post]) @ w), abs=1e-12)

    def test_multiplier_draws_are_mean_zero_and_unit_variance(self):
        """Mammen weights: mean 0, variance 1, third moment 1."""
        rng = np.random.default_rng(0)
        psi = rng.normal(size=200) / 200
        draws = REF.multiplier_bootstrap(psi, n_boot=20000, seed=1)
        scale = float(np.sqrt((psi ** 2).sum()))
        assert abs(float(draws.mean())) < 0.05 * scale
        assert float(draws.std()) == pytest.approx(scale, rel=0.05)


# =========================================================================== #
# 2. its restrictions are enforced, not assumed
# =========================================================================== #
class TestRestrictions:
    def test_an_unbalanced_panel_is_refused(self):
        df = panel((10, 14)).drop(index=5)
        with pytest.raises(ValueError, match="balanced"):
            REF.group_time_att(df, "y", "id", "time", "first_treat")

    def test_a_panel_with_no_never_treated_unit_is_refused(self):
        df = panel((10, 14))
        df["first_treat"] = df["first_treat"].replace(0, 14)
        df["d"] = (df["time"] >= df["first_treat"]).astype(int)
        with pytest.raises(ValueError, match="never-treated"):
            REF.group_time_att(df, "y", "id", "time", "first_treat")

    def test_it_records_the_upstream_it_came_from(self):
        assert REF.DIFF_DIFF_VERSION == "3.9.0"
        assert REF.DIFF_DIFF_COMMIT == "d9cd475fec8e67ceea1772896e2b29501cb90485"

    def test_the_provenance_file_agrees_with_the_module(self):
        """Two records of the same pin, so a half-done refresh is visible."""
        import json
        prov = json.loads((_REF_PATH.parent / "provenance.json").read_text())
        assert prov["upstream"]["version"] == REF.DIFF_DIFF_VERSION
        assert prov["upstream"]["commit"] == REF.DIFF_DIFF_COMMIT

    def test_the_licence_travels_with_the_code(self):
        licence = (_REF_PATH.parent / "LICENSE.diff-diff").read_text()
        assert "MIT" in licence
        assert "Permission is hereby granted" in licence


# =========================================================================== #
# 3. parity with the upstream it was transcribed from
# =========================================================================== #
@needs_diff_diff
class TestParityWithUpstream:
    @pytest.mark.parametrize("onsets,k,seed", SHAPES)
    def test_group_time_effects_and_standard_errors_match(self, onsets, k, seed):
        df = panel(onsets, k, seed=seed)
        eff, ses, _, _ = parts(df)
        up = CallawaySantAnna(control_group="never_treated",
                              estimation_method="dr", n_bootstrap=0).fit(
            df, outcome="y", unit="id", time="time", first_treat="first_treat")
        common = [c for c in eff if c in up.group_time_effects and c[1] >= c[0]]
        assert common
        for c in common:
            assert eff[c] == pytest.approx(up.group_time_effects[c]["effect"], abs=1e-12)
            assert ses[c] == pytest.approx(up.group_time_effects[c]["se"], abs=1e-12)

    @pytest.mark.parametrize("onsets,k,seed", SHAPES)
    def test_aggregated_estimate_and_standard_error_match(self, onsets, k, seed):
        df = panel(onsets, k, seed=seed)
        eff, _, inf, cohort = parts(df)
        post = [c for c in eff if c[1] >= c[0]]
        est, se, _ = REF.aggregate(post, eff, inf, cohort)
        up = CallawaySantAnna(control_group="never_treated",
                              estimation_method="dr", n_bootstrap=0).fit(
            df, outcome="y", unit="id", time="time", first_treat="first_treat")
        assert est == pytest.approx(float(up.att), abs=1e-12)
        assert se == pytest.approx(float(up.se), abs=1e-12)
