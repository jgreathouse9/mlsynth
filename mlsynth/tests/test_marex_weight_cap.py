"""A ceiling on any single control market's weight, imposed at design time.

The contamination identity in ``mlsynth.utils.contamination`` says the error
an outside event in control market ``k`` puts on the estimate is ``-v_k`` times
the size of the event. The weight is the part the analyst controls, and only
before the experiment runs: once the design is locked, ``v_k`` is a constant
and the exposure is whatever the optimizer happened to choose.

``max_control_weight`` makes it a decision. Capping every control weight at
``c`` buys three guarantees that hold whatever the optimizer does with the
freedom left to it: no market can move the estimate by more than ``c`` times
its own shock, at least ``ceil(1/c)`` markets carry weight, and the effective
sample size of the control group is at least ``1/c`` (since ``sum(v^2) <=
c * sum(v) = c``). The price is fit, and these tests measure that too.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mlsynth import MAREX, contamination_report
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.marex_helpers.config import MAREXConfig

TOL = 1e-6


def panel(J=12, T=24, T0=18, seed=3, n_good=5, hi=4.0):
    """A panel on which an uncapped design concentrates.

    Five markets track the cluster's mean factor loading and carry almost no
    idiosyncratic noise; the other seven load differently and are noisy. The
    optimizer matches the cluster mean, so it puts most of the control weight
    on whichever of the five it did not have to treat -- the uncapped design
    reaches a single control weight near 0.49 here. Without that the cap tests
    would be vacuous, and ``test_the_cap_binds_...`` asserts it still holds.
    """
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    load = rng.uniform(0.4, 1.6, size=(2, J))
    load[:, :n_good] = load.mean(axis=1, keepdims=True)
    scale = np.full(J, hi)
    scale[:n_good] = 0.05
    Y = f @ load + rng.normal(size=(T, J)) * scale + 20.0
    df = pd.DataFrame({"unit": np.repeat(np.arange(J), T),
                       "time": np.tile(np.arange(T), J),
                       "y": Y.T.reshape(-1)})
    return df, T0


def fit(cap=None, relaxed=False, m_eq=3, design=None, lambda2=None, **kw):
    df, T0 = panel(**kw)
    cfg = dict(df=df, outcome="y", unitid="unit", time="time", T0=T0,
               program_type="MIQP", display_graph=False, inference=False,
               m_eq=m_eq, relaxed=relaxed)
    if cap is not None:
        cfg["max_control_weight"] = cap
    if design is not None:
        cfg["design"] = design
    if lambda2 is not None:
        cfg["lambda2"] = lambda2
    return MAREX(MAREXConfig(**cfg)).fit()


def control_weights(res):
    return np.asarray(list(
        res.design_weights.summary_stats["control_weights_agg"].values()),
        dtype=float)


def cluster_control_weights(res):
    return np.asarray(res.globres.control_weights_agg, dtype=float)


# --------------------------------------------------------------------- smoke
def test_a_capped_design_fits_and_respects_the_cap():
    res = fit(cap=0.2)
    v = control_weights(res)
    assert v.size >= 5
    assert v.max() <= 0.2 + TOL
    assert v.sum() == pytest.approx(1.0)


def test_the_cap_is_recorded_on_the_result():
    res = fit(cap=0.25)
    assert res.design_weights.summary_stats["max_control_weight"] == 0.25


def test_an_uncapped_design_records_no_cap():
    res = fit()
    assert res.design_weights.summary_stats["max_control_weight"] is None


# ----------------------------------------------------------------- invariants
def test_the_cap_binds_on_a_design_that_would_otherwise_concentrate():
    """The test is only meaningful if the uncapped design exceeds the cap."""
    loose = control_weights(fit())
    assert loose.max() > 0.4, (
        f"fixture no longer concentrates (max v = {loose.max():.3f}); the cap "
        "test would be vacuous")
    assert control_weights(fit(cap=0.2)).max() <= 0.2 + TOL


@pytest.mark.parametrize("cap", [0.5, 0.34, 0.25, 0.2])
def test_the_cap_bounds_the_effective_sample_size_from_below(cap):
    v = control_weights(fit(cap=cap))
    rep = contamination_report(v / v.sum(), market=int(np.argmax(v)))
    assert rep.max_weight <= cap + TOL
    assert rep.effective_sample_size >= 1.0 / cap - 1e-6
    assert rep.n_carrying_weight >= math.ceil(1.0 / cap - TOL)


def test_the_cap_bounds_what_any_single_shock_can_do():
    cap, shock = 0.2, -1.5
    v = control_weights(fit(cap=cap))
    v = v / v.sum()
    worst = max(abs(contamination_report(v, market=k, shock=shock).bias)
                for k in range(v.size))
    assert worst <= cap * abs(shock) + TOL


def test_the_cap_holds_on_every_cluster_column_not_only_the_aggregate():
    """The aggregate is a convex combination of the per-cluster columns.

    So a cap on each column implies the cap on the aggregate, and checking
    the aggregate alone would not distinguish the constraint being imposed
    per cluster from it holding by cancellation.
    """
    res = fit(cap=0.2)
    assert cluster_control_weights(res).max() <= 0.2 + TOL


def test_the_cap_costs_pre_period_fit():
    tight = fit(cap=0.2)
    loose = fit()
    assert tight.globres.synthetic_control is not None
    rmse_tight = max(c.rmse for c in tight.clusters.values())
    rmse_loose = max(c.rmse for c in loose.clusters.values())
    assert rmse_tight >= rmse_loose - 1e-8


# ---------------------------------------------------------------- edge cases
def test_a_cap_of_one_changes_nothing():
    """And changes nothing to the last bit, not merely to solver tolerance.

    ``v <= 1`` is implied by ``v >= 0`` and ``sum v == 1``, so the row is not
    added to the program at all. Added, it would leave the feasible set
    identical and still perturb the returned weights at the solver's own
    tolerance, which measured at 2.5e-5 on this panel.
    """
    capped = control_weights(fit(cap=1.0))
    loose = control_weights(fit())
    assert capped.size == loose.size
    np.testing.assert_allclose(np.sort(capped), np.sort(loose), atol=0,
                               rtol=0)


def test_the_tightest_feasible_cap_forces_uniform_control_weights():
    """With ``m_eq`` treated of ``J``, ``1/(J - m_eq)`` leaves one design."""
    J, m_eq = 12, 3
    res = fit(cap=1.0 / (J - m_eq), m_eq=m_eq, J=J)
    v = cluster_control_weights(res)
    nz = v[v > 1e-8]
    assert nz.size == J - m_eq
    np.testing.assert_allclose(nz, np.full(nz.size, 1.0 / (J - m_eq)),
                               atol=1e-6)


def test_the_relaxed_path_respects_the_cap_after_discretization():
    """The rounded control weights land under the cap on a penalized design.

    The relaxed program's rounding renormalizes the control weights over the
    units it did not treat, which lifts them by ``1 / sum`` of the mass that
    survived. On the standard design this never bites: the objective matches
    each cluster's own mean, the uniform vector attains it exactly and is
    feasible for continuous ``z``, so the relaxed optimum is uniform and
    rounding returns a uniform ``1 / n_controls``. Under a distance penalty on
    the control weights the relaxed optimum concentrates instead -- it sits
    against the cap at 0.2 here -- and rounding takes it to 0.3333, two thirds
    above the ceiling. That is the case the water-filling exists for.
    """
    res = fit(cap=0.2, relaxed=True, design="penalized", lambda2=1.0)
    assert cluster_control_weights(res).max() <= 0.2 + TOL
    assert control_weights(res).max() <= 0.2 + TOL


def test_the_rounded_weights_stay_off_the_treated_markets():
    """Water-filling redistributes within the control set, not over all units.

    Spilling outside it would put the treated markets on both sides of the
    comparison and leave the control weights summing to less than one over
    the controls they are supposed to cover.
    """
    res = fit(cap=0.2, relaxed=True, design="penalized", lambda2=1.0)
    v = cluster_control_weights(res)
    w = np.asarray(res.globres.treated_weights_agg, dtype=float)
    treated = w > 1e-8
    assert treated.any()
    assert np.all(v[treated] == 0.0)
    assert v[~treated].sum() == pytest.approx(1.0)


# ------------------------------------------------------------------ failures
@pytest.mark.parametrize(
    "bad", [0.0, -0.1, 1.5, float("nan"), float("inf"), float("-inf")])
def test_a_cap_outside_the_unit_interval_is_refused(bad):
    df, T0 = panel()
    with pytest.raises(MlsynthConfigError, match="max_control_weight"):
        MAREXConfig(df=df, outcome="y", unitid="unit", time="time", T0=T0,
                    program_type="MIQP", display_graph=False, inference=False,
                    m_eq=3, max_control_weight=bad)


def test_a_cap_too_tight_for_the_available_controls_is_refused():
    """Nine controls cannot hold a unit of weight under a ceiling of 0.1."""
    df, T0 = panel(J=12)
    with pytest.raises(MlsynthConfigError, match="at least"):
        MAREXConfig(df=df, outcome="y", unitid="unit", time="time", T0=T0,
                    program_type="MIQP", display_graph=False, inference=False,
                    m_eq=3, max_control_weight=0.1)


def test_the_refusal_names_the_smallest_feasible_cap():
    df, T0 = panel(J=12)
    with pytest.raises(MlsynthConfigError) as exc:
        MAREXConfig(df=df, outcome="y", unitid="unit", time="time", T0=T0,
                    program_type="MIQP", display_graph=False, inference=False,
                    m_eq=4, max_control_weight=0.05)
    assert "0.125" in str(exc.value)


def test_a_design_with_no_controls_left_is_refused():
    """Treating every market leaves nothing for a cap to apply to."""
    df, T0 = panel(J=12)
    with pytest.raises(MlsynthConfigError, match="at least one control market"):
        MAREXConfig(df=df, outcome="y", unitid="unit", time="time", T0=T0,
                    program_type="MIQP", display_graph=False, inference=False,
                    m_eq=12, max_control_weight=0.5)


def test_feasibility_is_judged_on_the_smallest_cluster():
    """A cap the big cluster can meet and the small one cannot is refused.

    Clusters of 10 and 4 with two treated in each leave 8 controls and 2. A
    cap of 0.2 is comfortable for the first and impossible for the second,
    and the design has to satisfy both.
    """
    df, T0 = panel(J=14)
    df = df.copy()
    df["region"] = np.where(df["unit"] < 10, 0, 1)
    with pytest.raises(MlsynthConfigError, match="1/2"):
        MAREXConfig(df=df, outcome="y", unitid="unit", time="time", T0=T0,
                    cluster="region", program_type="MIQP", display_graph=False,
                    inference=False, m_eq=2, max_control_weight=0.2)


def test_a_clustered_design_accepts_a_cap_the_smallest_cluster_can_meet():
    df, T0 = panel(J=14)
    df = df.copy()
    df["region"] = np.where(df["unit"] < 10, 0, 1)
    cfg = MAREXConfig(df=df, outcome="y", unitid="unit", time="time", T0=T0,
                      cluster="region", program_type="MIQP",
                      display_graph=False, inference=False, m_eq=2,
                      max_control_weight=0.5)
    assert cfg.max_control_weight == 0.5


def test_passing_the_cap_as_none_explicitly_is_the_same_as_omitting_it():
    """A caller assembling kwargs passes ``None`` for the option it skipped."""
    df, T0 = panel()
    cfg = MAREXConfig(df=df, outcome="y", unitid="unit", time="time", T0=T0,
                      program_type="MIQP", display_graph=False, inference=False,
                      m_eq=3, max_control_weight=None)
    assert cfg.max_control_weight is None
