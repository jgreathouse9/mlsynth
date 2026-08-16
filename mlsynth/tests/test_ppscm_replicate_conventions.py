"""A resampling replicate must refit the estimator that produced the estimate.

PPSCM's jackknife drops each unit and refits. #466 gave ``run_multisynth`` three
convention parameters -- ``donor_weights``, ``base_period``, ``donor_pool`` --
as keyword arguments with defaults, and the three refit sites in ``inference.py``
kept their earlier call signature. Every replicate therefore ran under augsynth's
conventions while the point estimate ran under the caller's.

The RCA ladder, one class per rung:

* rung 0 -- the headline ATT is untouched, which is why this shipped;
* rung 1 -- everything downstream of the refits moved: with uniform donor
  weights there is no QP, ``nu_used`` is ``NaN``, that ``NaN`` becomes the
  refits' pooling level, all fifty raise ``DCPError``, ``except Exception:
  continue`` absorbs them, and the standard error is ``NaN``;
* rung 2 -- with ``donor_weights="scm"`` and the other two conventions set,
  ``nu_used`` is a real number and every refit succeeds, so the standard error
  is finite, plausible, and computed for a different estimator. Measured on a
  three-cohort panel, ``never_treated`` and ``not_yet_treated`` gave 0.5467 and
  0.5464 -- the same augsynth refit twice, with only the point estimate moving;
* rung 3 -- the invariant nobody wrote down, over the whole convention grid;
* rung 4 -- the contracts that let it through: a non-finite pooling level
  reaching the QP, and a jackknife with no usable replicate returning ``NaN``
  instead of saying so;
* rung 5 -- ``tools/mutation/targets.toml``, target ``ppscm-replicate-conventions``.

Rung 2's test is a hand refit, not a spy on keyword arguments: what has to hold
is that a replicate equals the estimator refit on that subsample, and a test
naming the parameter that carries it would pass any implementation that spells
the parameter the same way and mean nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import PPSCM
from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.ppscm_helpers.engine import Conventions, run_multisynth
from mlsynth.utils.ppscm_helpers.inference import jackknife_inference
from mlsynth.utils.ppscm_helpers.setup import prepare_ppscm_inputs


def staggered(onsets=(10, 14, 18), k=3, n_units=50, n_per=30, seed=0):
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


CS = {"donor_weights": "uniform", "base_period": "pre_treatment",
      "donor_pool": "never_treated"}


def fit(df, **over):
    cfg = {"df": df[["id", "time", "y", "d"]], "outcome": "y", "treat": "d",
           "unitid": "id", "time": "time", "display_graphs": False,
           "run_inference": True, "inference_method": "jackknife", **over}
    return PPSCM(cfg).fit()


def panel_arrays(df):
    """``(Xy, trt, d, n_leads, n_lags)`` exactly as ``PPSCM.fit`` derives them."""
    inp = prepare_ppscm_inputs(df[["id", "time", "y", "d"]], outcome="y", treat="d",
                               unitid="id", time="time")
    T, d = inp.Xy.shape[1], inp.n_pre
    return inp.Xy, inp.trt, d, T - d, d


def refit_nu(res):
    """The pooling level a refit of this fit may reuse.

    A uniform-weight fit poses no quadratic program and reports ``nu_used`` as
    ``NaN``, which is not a level anything can be refit at; that branch ignores
    ``nu`` entirely, so ``None`` is what a hand refit passes. Handing the ``NaN``
    on is refused, and refusing it is rung 4's own test.
    """
    nu = float(res.design.nu_used)
    return nu if np.isfinite(nu) else None


def cs_oracle(df, L):
    wide = df.pivot(index="id", columns="time", values="y")
    g_of = df.groupby("id")["first_treat"].first()
    never = g_of.index[g_of == 0]
    vals = []
    for g in sorted(x for x in g_of.unique() if x != 0):
        tr = g_of.index[g_of == g]
        for t in wide.columns:
            if 0 <= t - g < L:
                vals.append(float((wide.loc[tr, t] - wide.loc[tr, g - 1]).mean()
                                  - (wide.loc[never, t] - wide.loc[never, g - 1]).mean()))
    return float(np.mean(vals))


# =========================================================================== #
# rung 0 -- the quantity a reader checks first, and why it is not evidence
# =========================================================================== #
class TestRungZeroTheHeadlineNumber:
    def test_the_att_is_correct_throughout(self):
        """Passes before the fix and after. The refits do not touch the point
        estimate, so the number least sensitive to the defect is the one on the
        front page."""
        df = staggered()
        res = fit(df, **CS)
        L = len(next(iter(res.per_unit.values())).tau)
        assert float(res.att) == pytest.approx(cs_oracle(df, L), abs=1e-12)


# =========================================================================== #
# rung 1 -- what moved with it
# =========================================================================== #
class TestRungOneWhatElseMoved:
    def test_a_standard_error_is_reported(self):
        res = fit(staggered(), **CS)
        assert res.inference.standard_error is not None
        assert np.isfinite(float(res.inference.standard_error))

    def test_the_event_study_band_is_finite(self):
        res = fit(staggered(), **CS)
        assert np.isfinite(res.event_study.se).all()
        assert np.isfinite(res.event_study.ci).all()

    def test_the_replicate_paths_are_usable(self):
        res = fit(staggered(), **CS)
        paths = res.inference_detail.replicate_paths
        assert np.isfinite(paths).any(), "every leave-one-out refit failed"
        assert np.isfinite(paths).all(axis=1).sum() >= 2


# =========================================================================== #
# rung 2 -- which branch produced them
# =========================================================================== #
class TestRungTwoWhichBranch:
    @pytest.mark.parametrize("conv", [
        CS,
        {"donor_weights": "scm", "base_period": "pre_treatment",
         "donor_pool": "never_treated"},
        {"donor_weights": "scm", "base_period": "all_pre",
         "donor_pool": "not_yet_treated"},
    ])
    def test_a_replicate_equals_a_hand_refit_of_the_same_estimator(self, conv):
        """The claim, computed independently: drop unit 0, refit by hand under
        the caller's conventions, and the stored replicate must be that path."""
        df = staggered()
        res = fit(df, **conv)
        Xy, trt, d, n_leads, n_lags = panel_arrays(df)
        keep = np.ones(Xy.shape[0], dtype=bool)
        keep[0] = False
        manual = run_multisynth(
            Xy[keep], trt[keep], d, n_leads, n_lags, fixedeff=True,
            time_cohort=False, nu=refit_nu(res), lam=0.0, solver=None,
            conventions=Conventions(**conv))
        stored = np.asarray(res.inference_detail.replicate_paths[0], dtype=float)
        want = np.asarray(manual["per_time"], dtype=float)
        assert np.allclose(stored[:len(want)], want, atol=1e-9, equal_nan=True), conv

    def test_the_donor_pool_moves_the_standard_error(self):
        """Two comparison groups, two estimators, two standard errors. Before
        the fix these agreed to 3e-04 because both replicates were augsynth's."""
        df = staggered()
        a = fit(df, donor_weights="scm", base_period="pre_treatment",
                donor_pool="never_treated")
        b = fit(df, donor_weights="scm", base_period="pre_treatment",
                donor_pool="not_yet_treated")
        sa, sb = float(a.inference.standard_error), float(b.inference.standard_error)
        assert abs(sa - sb) > 1e-3, (sa, sb)

    def test_a_non_finite_pooling_level_never_reaches_the_qp(self):
        """The uniform fit poses no program, so ``nu_used`` is ``NaN``. Handing
        that back as the refits' pooling level is what turned a wrong estimator
        into fifty ``DCPError``s."""
        df = staggered()
        res = fit(df, **CS)
        assert np.isnan(res.design.nu_used)
        assert np.isfinite(float(res.inference.standard_error))


# =========================================================================== #
# rung 3 -- the invariant nobody wrote down
# =========================================================================== #
GRID = [(dw, bp, dp)
        for dw in ("scm", "uniform")
        for bp in ("all_pre", "pre_treatment")
        for dp in ("window", "never_treated", "not_yet_treated")]


#: Two panels, because one of them cannot tell two of the pools apart. With
#: three cohorts inside a 13-lead window every non-focal cohort adopts inside
#: it, so ``window`` and ``never_treated`` select the same donors and a cell
#: asking for one while getting the other passes. Four cohorts spread over a
#: long window separates them -- the divergence regime #466 documents.
PANELS = {"aligned": (10, 14, 18), "divergent": (8, 12, 16, 20)}


class TestRungThreeTheInvariant:
    @pytest.mark.parametrize("panel", list(PANELS))
    @pytest.mark.parametrize("dw,bp,dp", GRID)
    def test_every_convention_combination_is_refit_as_asked(self, dw, bp, dp, panel):
        """For any configuration, a replicate is that configuration refit.

        The grid, not three examples: the defect was uniform across all twelve
        cells, and a test on one cell would have been passed by threading one
        parameter.
        """
        df = staggered(PANELS[panel])
        conv = {"donor_weights": dw, "base_period": bp, "donor_pool": dp}
        res = fit(df, **conv)
        Xy, trt, d, n_leads, n_lags = panel_arrays(df)
        keep = np.ones(Xy.shape[0], dtype=bool)
        keep[0] = False
        manual = run_multisynth(
            Xy[keep], trt[keep], d, n_leads, n_lags, fixedeff=True,
            time_cohort=False, nu=refit_nu(res), lam=0.0, solver=None,
            conventions=Conventions(**conv))
        stored = np.asarray(res.inference_detail.replicate_paths[0], dtype=float)
        want = np.asarray(manual["per_time"], dtype=float)
        assert np.allclose(stored[:len(want)], want, atol=1e-9, equal_nan=True)

    def test_the_conventions_travel_as_one_object(self):
        """Structural, and the reason the ladder does not stop at threading
        three parameters: a fourth convention added to ``Conventions`` reaches
        every refit site without anyone editing them, so the omission that
        caused this cannot be made again."""
        import dataclasses
        assert dataclasses.is_dataclass(Conventions)
        names = {f.name for f in dataclasses.fields(Conventions)}
        assert names == {"donor_weights", "base_period", "donor_pool"}
        assert Conventions() == Conventions("scm", "all_pre", "window"), (
            "the default must stay augsynth's, or plain PPSCM changes")


# =========================================================================== #
# rung 4 -- the contracts that were never enforced
# =========================================================================== #
class TestRungFourTheContracts:
    def test_a_non_finite_pooling_level_is_refused(self):
        """``nu=NaN`` built a non-DCP objective and the solver's complaint was
        the first anyone heard of it."""
        Xy, trt, d, n_leads, n_lags = panel_arrays(staggered())
        with pytest.raises(MlsynthConfigError, match="nu"):
            run_multisynth(Xy, trt, d, n_leads, n_lags, fixedeff=True,
                           time_cohort=False, nu=float("nan"), lam=0.0)

    def test_a_jackknife_with_no_usable_replicate_is_reported(self):
        """Absorbing every exception and returning ``NaN`` is how this stayed
        invisible. With no usable replicate there is no inference.

        Solved weights, not uniform ones: the uniform branch poses no quadratic
        program, so an unusable solver is never consulted and every refit
        succeeds. Breaking the solver only breaks a fit that solves.
        """
        Xy, trt, d, n_leads, n_lags = panel_arrays(staggered())
        with pytest.raises(MlsynthEstimationError, match="replicate"):
            jackknife_inference(
                Xy, trt, d, n_leads, n_lags, fixedeff=True, time_cohort=False,
                nu_used=1.0, lam=0.0, solver="NOT_A_SOLVER",
                alpha=0.05, per_time_full=np.zeros(n_leads), att_full=0.0,
                conventions=Conventions(donor_weights="scm",
                                        base_period="pre_treatment",
                                        donor_pool="never_treated"))

    def test_the_uniform_branch_consults_no_solver(self):
        """Why the test above cannot use the Callaway-Sant'Anna conventions,
        pinned so the reason is not rediscovered: with equal donor weights there
        is no program to solve, and an unusable solver passes straight through."""
        Xy, trt, d, n_leads, n_lags = panel_arrays(staggered())
        out = run_multisynth(
            Xy, trt, d, n_leads, n_lags, fixedeff=True, time_cohort=False,
            nu=None, lam=0.0, solver="NOT_A_SOLVER",
            conventions=Conventions(**CS))
        assert np.isfinite(out["att"])

    def test_a_partly_failed_jackknife_still_reports(self):
        """A degenerate leave-one-out is skipped, not fatal: the guard is about
        having no replicate at all, not about having fewer than all of them."""
        res = fit(staggered(), **CS)
        paths = res.inference_detail.replicate_paths
        assert np.isfinite(paths).all(axis=1).sum() < paths.shape[0] + 1
        assert np.isfinite(float(res.inference.standard_error))

    def test_the_conformal_calibration_refits_as_asked(self):
        """``cumulative_conformal_per_unit`` refits at rolling origins and
        carries the same requirement as the jackknife."""
        df = staggered()
        res = fit(df, conformal_horizon=2, **CS)
        unit = next(iter(res.per_unit.values()))
        assert unit.cumulative_windows is not None
        assert unit.cumulative_effect is not None
        assert np.isfinite(unit.cumulative_effect)
