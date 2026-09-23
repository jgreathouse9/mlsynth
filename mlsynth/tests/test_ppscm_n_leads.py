"""``n_leads`` can be raised, and its ceiling is the longest post window.

Two quantities were conflated (#481). augsynth keeps them apart
(``multisynth``, ``R/multisynth_class.R:127-134``):

* the default, ``ncol(wide$y)``, is the post window of the *last*-treated
  cohort --- the shortest one, so every cohort reaches every horizon and the
  event study is rectangular;
* the ceiling, ``max_j(post periods for unit j)``, is the *longest* post
  window, which is how far the panel runs past the first adoption.

mlsynth used the first for both, so an explicit ``n_leads`` above the default
was discarded and horizons the panel plainly observes were unreachable. The
cost was concrete: the Ronczewski (2026) cannabis-alcohol pipeline calls
``multisynth(..., n_leads = 6)`` against a default of 2, and mlsynth could not
express the call, so it could not reproduce either headline number. With the
ceiling lifted it reproduces the Callaway-Sant'Anna aggregate to 2e-17 and the
augsynth aggregate to 8e-08.

Raising ``n_leads`` makes the event study ragged: a cohort adopting late has no
cell at a horizon the early cohorts still reach. Ragged is the point, and the
tests below fix that the missing cells report ``NaN`` --- a zero effect with a
zero standard error is what a silently skipped horizon looks like on a plot.

The panel underneath is chosen, not measured: with adoptions at ``t = 12`` and
``t = 16`` in twenty periods, the default is 5 and the ceiling is 9. A panel
whose two quantities coincide would pass every test here without exercising the
distinction, so the single-cohort case is tested separately and named for what
it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import PPSCM
from mlsynth.utils.ppscm_helpers.engine import eligible_donors

DEFAULT_LEADS = 5      # T - d: the t = 16 cohort's post window
CEILING_LEADS = 9      # T - first adoption: the t = 12 cohort's post window


def panel(n_units=40, n_per=20, onsets=(12, 16), k=2, seed=0, effect="dynamic"):
    """A staggered panel whose cohorts observe post windows of unequal length.

    ``effect`` is dynamic by default so a horizon carries a different number
    from its neighbours; under a constant effect an off-by-one in the window
    would move nothing and the tests would pass on the wrong cell set.
    """
    rng = np.random.default_rng(seed)
    assign = {u: g for i, g in enumerate(onsets) for u in range(i * k, (i + 1) * k)}
    rows = []
    for u in range(n_units):
        base, walk = rng.normal(10, 2), np.cumsum(rng.normal(0, 0.3, n_per))
        g = assign.get(u)
        for t in range(1, n_per + 1):
            on = g is not None and t >= g
            eff = 0.0 if not on else (0.4 * (t - g + 1) if effect == "dynamic"
                                      else 2.0)
            rows.append((u, t, base + walk[t - 1] + 0.5 * rng.normal() + eff,
                         int(on), g if g is not None else 0))
    return pd.DataFrame(rows, columns=["id", "time", "y", "d", "first_treat"])


def fit(df, **over):
    return PPSCM({"df": df[["id", "time", "y", "d"]], "outcome": "y",
                  "treat": "d", "unitid": "id", "time": "time",
                  "display_graphs": False, "run_inference": False,
                  **over}).fit()


def leads_of(res):
    return len(next(iter(res.per_unit.values())).tau)


def stack_of(res):
    """Per-unit effect paths as a ``units x horizons`` array."""
    return np.vstack([np.asarray(v.tau, float) for v in res.per_unit.values()])


def post_window_of(df, unit):
    """How many post periods that unit actually observes."""
    g = int(df.loc[df.id == unit, "first_treat"].iloc[0])
    return int((df.loc[df.id == unit, "time"] >= g).sum())


# --------------------------------------------------------------------------- #
# the ceiling
# --------------------------------------------------------------------------- #
class TestTheCeilingIsTheLongestPostWindow:
    def test_the_default_is_the_last_cohorts_post_window(self):
        """augsynth's default, unchanged: the window every cohort reaches."""
        assert leads_of(fit(panel())) == DEFAULT_LEADS

    def test_raising_n_leads_reaches_the_early_cohorts_horizons(self):
        assert leads_of(fit(panel(), n_leads=CEILING_LEADS)) == CEILING_LEADS

    def test_a_value_between_the_default_and_the_ceiling_is_honoured(self):
        """The ceiling is not a second default: intermediate values stand."""
        assert leads_of(fit(panel(), n_leads=7)) == 7

    def test_the_ceiling_still_binds_past_the_longest_window(self):
        """Lifting the ceiling is not removing it. Nothing observes horizon
        99, so the request is cut to what the earliest cohort reaches."""
        assert leads_of(fit(panel(), n_leads=99)) == CEILING_LEADS

    def test_lowering_n_leads_below_the_default_still_truncates(self):
        assert leads_of(fit(panel(), n_leads=2)) == 2

    def test_a_single_cohort_has_no_room_between_the_two(self):
        """With one adoption time the default and the ceiling coincide, so
        this panel cannot tell the fix from the bug. It is here to pin that
        the ceiling is still computed, not that it moved."""
        df = panel(onsets=(14,), k=2)
        assert leads_of(fit(df)) == 7
        assert leads_of(fit(df, n_leads=99)) == 7

    def test_the_ceiling_tracks_the_first_adoption_and_not_the_last(self):
        """Move only the early cohort and the ceiling moves with it, while the
        default --- fixed by the late cohort --- does not."""
        early = fit(panel(onsets=(10, 16)), n_leads=99)
        late = fit(panel(onsets=(14, 16)), n_leads=99)
        assert leads_of(early) == 11
        assert leads_of(late) == 7
        assert leads_of(fit(panel(onsets=(10, 16)))) == DEFAULT_LEADS
        assert leads_of(fit(panel(onsets=(14, 16)))) == DEFAULT_LEADS


# --------------------------------------------------------------------------- #
# what the extra horizons hold
# --------------------------------------------------------------------------- #
class TestTheRaisedHorizonsAreRagged:
    def test_each_unit_reports_its_own_window_and_nan_past_it(self):
        df = panel()
        res = fit(df, n_leads=CEILING_LEADS)
        # Without this the slices below are empty under the clamp and every
        # assertion holds vacuously.
        assert leads_of(res) == CEILING_LEADS
        for unit, uf in res.per_unit.items():
            tau = np.asarray(uf.tau, float)
            want = post_window_of(df, int(unit))
            assert np.isfinite(tau[:want]).all(), unit
            assert np.isnan(tau[want:]).all(), unit

    def test_the_late_cohort_is_absent_past_its_window_and_not_zero(self):
        """A skipped horizon and a zero effect are different claims, and a
        zero would be read off an event-study plot as the second."""
        df = panel()
        res = fit(df, n_leads=CEILING_LEADS)
        stack = stack_of(res)
        late = [i for i, u in enumerate(res.per_unit)
                if post_window_of(df, int(u)) == DEFAULT_LEADS]
        assert late, "the fixture must contain a short-window cohort"
        tail = stack[np.array(late), DEFAULT_LEADS:]
        assert tail.size == len(late) * (CEILING_LEADS - DEFAULT_LEADS)
        assert np.isnan(tail).all()

    def test_every_horizon_up_to_the_ceiling_has_at_least_one_cell(self):
        stack = stack_of(fit(panel(), n_leads=CEILING_LEADS))
        assert stack.shape[1] == CEILING_LEADS
        assert np.isfinite(stack).any(axis=0).all()

    def test_the_early_horizons_are_unmoved_by_asking_for_more(self):
        """Raising the window adds cells; it must not re-estimate the ones
        already there. Donor pools are held fixed by ``never_treated`` so the
        comparison isolates the window."""
        df = panel()
        short = fit(df, donor_pool="never_treated")
        long = fit(df, n_leads=CEILING_LEADS, donor_pool="never_treated")
        assert leads_of(short) == DEFAULT_LEADS
        assert leads_of(long) == CEILING_LEADS
        np.testing.assert_allclose(stack_of(long)[:, :DEFAULT_LEADS],
                                   stack_of(short), rtol=0, atol=1e-10)


class TestTheDefaultPathIsUnmoved:
    def test_asking_for_the_default_explicitly_changes_nothing(self):
        df = panel()
        implicit, explicit = fit(df), fit(df, n_leads=DEFAULT_LEADS)
        assert leads_of(implicit) == leads_of(explicit) == DEFAULT_LEADS
        np.testing.assert_array_equal(stack_of(implicit), stack_of(explicit))

    def test_the_donor_window_still_narrows_as_n_leads_rises(self):
        """``donor_pool='window'`` admits units treated more than ``n_leads``
        periods later, so a longer window is a stricter pool. Raising the
        ceiling exposes that trade-off; it does not remove it.

        A guard on behaviour the clamp made unreachable, not a red test:
        ``eligible_donors`` is unchanged by #481 and this fixes what a caller
        now has to weigh when reaching for a longer window. Asserted against
        the function directly, since a panel wide enough to show the narrowing
        through ``design.max_donors`` needs a third cohort and a different
        default, and would then be testing two things at once.
        """
        trt = np.array([11.0, 15.0, 18.0, np.inf, np.inf])
        wide = eligible_donors(trt, 11, DEFAULT_LEADS, "window")
        tight = eligible_donors(trt, 11, CEILING_LEADS, "window")
        assert list(wide) == [2, 3, 4]        # trt > 16
        assert list(tight) == [3, 4]          # trt > 20

    def test_the_event_study_plot_draws_over_a_ragged_window(self, monkeypatch):
        """The docs point a caller at the longer window, and the first thing
        they will do with it is plot the event study."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        df = panel()
        res = PPSCM({"df": df[["id", "time", "y", "d"]], "outcome": "y",
                     "treat": "d", "unitid": "id", "time": "time",
                     "display_graphs": True, "run_inference": True,
                     "n_leads": CEILING_LEADS}).fit()
        assert leads_of(res) == CEILING_LEADS
        plt.close("all")


# --------------------------------------------------------------------------- #
# inference over a ragged event study
# --------------------------------------------------------------------------- #
class TestInferenceSurvivesRaggedHorizons:
    @pytest.mark.parametrize("method", ["jackknife", "bootstrap"])
    def test_the_standard_errors_follow_the_cells(self, method):
        df = panel()
        res = PPSCM({"df": df[["id", "time", "y", "d"]], "outcome": "y",
                     "treat": "d", "unitid": "id", "time": "time",
                     "display_graphs": False, "run_inference": True,
                     "inference_method": method,
                     "n_leads": CEILING_LEADS}).fit()
        assert leads_of(res) == CEILING_LEADS
        se = np.asarray(res.event_study.se, float)
        assert np.isfinite(se).all() and (se[np.isfinite(se)] >= 0).all()
        assert np.isfinite(res.inference.standard_error)

    def test_the_influence_function_covers_the_raised_window(self):
        """The Callaway-Sant'Anna interval is the one that goes with the
        Callaway-Sant'Anna point estimate, and the aggregation it reports is
        over the raised window."""
        df = panel()
        res = PPSCM({"df": df[["id", "time", "y", "d"]], "outcome": "y",
                     "treat": "d", "unitid": "id", "time": "time",
                     "display_graphs": False, "run_inference": True,
                     "method": "callaway_santanna",
                     "n_leads": CEILING_LEADS}).fit()
        assert leads_of(res) == CEILING_LEADS
        se = np.asarray(res.event_study.se, float)
        assert np.isfinite(se).all()
        assert np.isfinite(res.inference.standard_error)
        assert res.inference.standard_error > 0


# --------------------------------------------------------------------------- #
# the Callaway-Sant'Anna identity over the whole cell set
# --------------------------------------------------------------------------- #
def cs_group_time(df):
    """``ATT(g, t)`` against never-treated units, base period ``g - 1``."""
    wide = df.pivot(index="id", columns="time", values="y")
    g_of = df.groupby("id")["first_treat"].first()
    never = g_of.index[g_of == 0]
    out = {}
    for g in sorted(x for x in g_of.unique() if x):
        tr = g_of.index[g_of == g]
        for t in wide.columns:
            if t >= g and (g - 1) in wide.columns:
                out[(int(g), int(t))] = float(
                    (wide.loc[tr, t] - wide.loc[tr, g - 1]).mean()
                    - (wide.loc[never, t] - wide.loc[never, g - 1]).mean())
    return out


class TestTheIdentityHoldsOverTheWholeCellSet:
    """The clamp did not only shorten the window; it changed which estimand
    was reachable. Callaway-Sant'Anna's ``simple`` aggregate runs over every
    unit-post-period cell, so under a five-lead ceiling on a nine-lead panel
    it could not be computed at all."""

    def test_every_group_time_cell_matches_the_oracle(self):
        df = panel()
        res = fit(df, n_leads=CEILING_LEADS, method="callaway_santanna")
        gt = cs_group_time(df)
        g_of = df.groupby("id")["first_treat"].first()
        for g in sorted(x for x in g_of.unique() if x):
            units = [u for u in res.per_unit if g_of[int(u)] == g]
            for h in range(CEILING_LEADS):
                t = int(g) + h
                if (int(g), t) not in gt:
                    continue
                mine = np.mean([float(np.asarray(res.per_unit[u].tau)[h])
                                for u in units])
                assert mine == pytest.approx(gt[(int(g), t)], abs=1e-12)

    def test_the_per_unit_stack_averages_to_the_simple_aggregation(self):
        """``did::aggte(type = 'simple')`` weights every unit-post-period cell
        equally, which the ragged unit-by-horizon stack already is."""
        df = panel()
        res = fit(df, n_leads=CEILING_LEADS, method="callaway_santanna")
        gt = cs_group_time(df)
        sizes = df.groupby("first_treat")["id"].nunique()
        num = sum(v * sizes[g] for (g, _), v in gt.items())
        den = sum(sizes[g] for (g, _) in gt)
        assert float(np.nanmean(stack_of(res))) == pytest.approx(num / den,
                                                                 abs=1e-12)

    def test_the_att_is_the_dynamic_aggregation_over_the_raised_window(self):
        """``result.effects.att`` averages the event-study horizons, which is
        ``did::aggte(type = 'dynamic')``'s overall and not the simple one. The
        two coincide only when every cohort reaches every horizon, so the
        raised window is where they separate."""
        df = panel()
        res = fit(df, n_leads=CEILING_LEADS, method="callaway_santanna")
        stack = stack_of(res)
        assert float(res.effects.att) == pytest.approx(
            float(np.nanmean(np.nanmean(stack, axis=0))), abs=1e-12)
        assert float(res.effects.att) != pytest.approx(
            float(np.nanmean(stack)), abs=1e-6)
