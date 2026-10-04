"""SNN's anchor cross recovers SI's per-intervention donor pool, for d != 0.

``test_pcr_nesting`` establishes that SNN, SI and PCR-RSC coincide on the
synthetic-control cross, but it runs SI's control arm, where the donor pool is
every untreated unit -- which is exactly RSC's pool. That is the degenerate end
of the chain. SI's own contribution over RSC is that the pool is
target-specific: to estimate unit ``i``'s outcome under intervention ``d``, it
regresses ``i``'s pre-period onto ``I(d)``, the units that actually took ``d``,
and applies the weights to their outcomes under ``d``.

These tests cover that end. Flatten the potential-outcomes tensor so a column
is an ``(intervention, period)`` pair -- the reduction SI's Assumption 2
describes -- and apply its observation law: before treatment every unit is
under ``d = 0``, and after treatment column ``(d, t)`` is observed exactly for
``I(d)``. Then hand SNN the mask and nothing else.

The mask turns out to do the restricting by itself. For a target
``(i, (d, t))`` with ``t > T0`` and ``i`` not in ``I(d)``, the rows observed at
that column are precisely ``I(d)``; the columns observed in row ``i`` are the
pre-period control columns plus ``i``'s own arm's post columns, and those last
are blank for every ``I(d)`` row, so the search drops them. What remains is
``I(d)`` crossed with the pre-period -- SI's cross, discovered from the
observation pattern without SNN being told that interventions exist.

An equality that cannot fail proves nothing, so two controls carry equal
weight: the arms must disagree with each other, and substituting the status-quo
pool ``I(0)`` where ``I(d)`` is required must change the answer. Both are
asserted below.

Layered per agents/agents_tests.md:

* smoke -- the flattened mask yields a cross for a multi-arm target.
* unit invariants -- the cross is exactly ``I(d)`` x pre-period; the imputation
  equals SI's arm counterfactual; a noiseless low-rank tensor is recovered.
* edge -- an arm with exactly ``r`` members, and a target drawn from a third arm.
* failure -- a pool too small to span, and an entry that is observed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import SI
from mlsynth.utils.pcr import pcr_weights
from mlsynth.utils.snn_helpers.completion import _find_anchors

T0, T1 = 16, 6
RANK = 3
ARMS = (0, 1, 2)                      # status quo, plus two interventions


def _tensor_panel(n_per_arm=(14, 5, 7), r=RANK, t0=T0, t1=T1, noise=0.0, seed=4):
    """A tensor factor model with SI's observation law.

    ``Y_ti(d) = <u_t(d), v_i>``: unit factors shared across time and
    intervention (SI Assumption 2), intervention-specific time factors.
    Returns the pieces both engines need.
    """
    rng = np.random.default_rng(seed)
    n = sum(n_per_arm)
    T = t0 + t1
    V = rng.standard_normal((n, r))                       # unit factors
    U = {d: rng.standard_normal((T, r)) for d in ARMS}    # time-intervention
    truth = {d: U[d] @ V.T + rng.standard_normal((T, n)) * noise for d in ARMS}

    arm = np.repeat(np.arange(len(n_per_arm)), n_per_arm)
    pool = {d: np.where(arm == d)[0] for d in ARMS}

    # Observed outcome: control before t0, own arm after.
    obs = truth[0].copy()
    for d in ARMS:
        if d == 0:
            continue
        obs[t0:, pool[d]] = truth[d][t0:, pool[d]]

    # Flattened matrix: rows are units, columns are (d, t) pairs.
    cols = [(d, t) for d in ARMS for t in range(T)]
    M = np.full((n, len(cols)), np.nan)
    for c, (d, t) in enumerate(cols):
        if t < t0:
            if d == 0:
                M[:, c] = truth[0][t]
        else:
            M[pool[d], c] = truth[d][t, pool[d]]
    mask = (~np.isnan(M)).astype(int)
    pre_cols = [c for c, (d, t) in enumerate(cols) if d == 0 and t < t0]
    return dict(truth=truth, obs=obs, arm=arm, pool=pool, M=M, mask=mask,
                cols=cols, pre_cols=pre_cols, n=n, T=T, t0=t0, t1=t1)


def _long(p):
    """The observed panel in long form, with one indicator column per arm."""
    rows = []
    for i in range(p["n"]):
        for t in range(p["T"]):
            rows.append({"unit": f"u{i}", "t": t, "y": float(p["obs"][t, i])})
    df = pd.DataFrame(rows)
    for d in ARMS:
        names = {f"u{i}" for i in p["pool"][d]}
        df[f"arm{d}"] = df.unit.isin(names).astype(int)
    return df


def _si_arm(p, target: int, d: int, rank=RANK):
    """SI's counterfactual for ``target`` under arm ``d``."""
    df = _long(p)
    df["treat"] = ((df.unit == f"u{target}") & (df.t >= p["t0"])).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SI({"df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
                  "time": "t", "inters": [f"arm{d}"], "rank_method": "fixed",
                  "rank": rank, "bias_correct": False,
                  "display_graphs": False}).fit()
    return np.asarray(res.arms[f"arm{d}"].counterfactual, float)[-p["t1"]:]


def _snn_arm(p, target: int, d: int, rank=RANK):
    """SNN's imputations for ``target`` down arm ``d``, and whether the cross is I(d)."""
    out, cross_ok = [], True
    want = sorted(x for x in p["pool"][d].tolist() if x != target)
    for t in range(p["t0"], p["T"]):
        j = p["cols"].index((d, t))
        AR, AC = _find_anchors(p["mask"], target, j)
        cross_ok &= (sorted(AR.tolist()) == want
                     and sorted(AC.tolist()) == p["pre_cols"])
        beta = pcr_weights(p["M"][np.ix_(AR, AC)].T, p["M"][target, AC], rank)
        out.append(float(p["M"][AR, j] @ beta))
    return np.array(out), cross_ok


def _rel(a, b):
    scale = max(np.abs(a).mean(), np.abs(b).mean(), 1e-12)
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / scale)


# --------------------------------------------------------------------- smoke
def test_a_multi_arm_target_gets_a_cross():
    p = _tensor_panel()
    target = int(p["pool"][0][0])
    j = p["cols"].index((2, p["t0"]))
    AR, AC = _find_anchors(p["mask"], target, j)
    assert AR.size > 0 and AC.size > 0


# ----------------------------------------------------------- unit invariants
@pytest.mark.parametrize("d", [1, 2])
def test_the_cross_is_exactly_the_arms_own_donor_pool(d):
    """``I(d)`` x pre-period, from the mask alone -- SI's cross, undeclared."""
    p = _tensor_panel()
    for target in p["pool"][0][:4]:
        _, cross_ok = _snn_arm(p, int(target), d)
        assert cross_ok, (d, target)


@pytest.mark.parametrize("d", [1, 2])
def test_the_imputation_equals_si_on_that_arm(d):
    p = _tensor_panel(noise=0.05)
    for target in p["pool"][0][:3]:
        target = int(target)
        snn, cross_ok = _snn_arm(p, target, d)
        assert cross_ok
        assert _rel(snn, _si_arm(p, target, d)) < 1e-10, (d, target)


@pytest.mark.parametrize("d", [1, 2])
def test_a_noiseless_low_rank_tensor_is_recovered_exactly(d):
    """With the rank right and no noise, SNN returns the true ``Y_ti(d)``."""
    p = _tensor_panel(noise=0.0)
    target = int(p["pool"][0][0])
    snn, cross_ok = _snn_arm(p, target, d)
    assert cross_ok
    truth = p["truth"][d][p["t0"]:, target]
    assert _rel(snn, truth) < 1e-10


def test_the_anchor_rows_never_include_the_target_or_another_arm():
    """The pool is the arm's members, not every unit observed somewhere."""
    p = _tensor_panel()
    target = int(p["pool"][0][0])
    for d in (1, 2):
        j = p["cols"].index((d, p["t0"]))
        AR, _ = _find_anchors(p["mask"], target, j)
        assert target not in AR
        other = set(p["pool"][0]) | set(p["pool"][3 - d])
        assert not (set(AR.tolist()) & other)


# ------------------------------------------------- controls (falsifiability)
def test_the_two_arms_disagree():
    """If every arm gave the same answer, recovering ``I(d)`` would not matter."""
    p = _tensor_panel()
    target = int(p["pool"][0][0])
    a, _ = _snn_arm(p, target, 1)
    b, _ = _snn_arm(p, target, 2)
    assert _rel(a, b) > 1e-3


@pytest.mark.parametrize("d", [1, 2])
def test_the_arm_answer_differs_from_the_control_answer_rsc_would_give(d):
    """Arm ``d`` answers a question the control arm does not.

    RSC's estimand is unit ``i`` under control. At ``d != 0`` SI and SNN answer
    "unit ``i`` under ``d``", using a different pool and a different target
    column, and the two answers have to differ -- otherwise the per-arm pool
    would be decoration.

    Substituting ``I(0)`` directly for ``I(d)`` is not the control to run here,
    and the reason is the point: the status-quo units have no observed outcome
    under ``d``, so there is nothing to apply the weights to. The pool is not
    merely the wrong choice, it is unavailable, which is why the mask alone
    pins the cross.
    """
    p = _tensor_panel()
    target = int(p["pool"][0][1])
    arm, _ = _snn_arm(p, target, d)
    control, _ = _snn_arm(p, target, 0)
    assert _rel(arm, control) > 1e-3
    # And the substitution really is unavailable, not just inferior.
    j = p["cols"].index((d, p["t0"]))
    status_quo = [x for x in p["pool"][0].tolist() if x != target]
    assert np.isnan(p["M"][status_quo, j]).all()


# ----------------------------------------------------------------------- edge
@pytest.mark.parametrize("d", [1, 2])
def test_an_arm_with_exactly_rank_many_members_still_nests(d):
    """``|I(d)| = r``: the pool spans, with nothing to spare."""
    p = _tensor_panel(n_per_arm=(14, RANK, RANK))
    target = int(p["pool"][0][0])
    snn, cross_ok = _snn_arm(p, target, d)
    assert cross_ok
    assert _rel(snn, _si_arm(p, target, d)) < 1e-10


def test_a_target_taken_from_a_third_arm_still_nests():
    """The target need not be a status-quo unit: arm 1 asking about arm 2."""
    p = _tensor_panel()
    target = int(p["pool"][1][0])
    snn, cross_ok = _snn_arm(p, target, 2)
    assert cross_ok
    assert _rel(snn, _si_arm(p, target, 2)) < 1e-10


# -------------------------------------------------------------------- failure
def test_a_pool_smaller_than_the_rank_cannot_span_and_says_so():
    """``|I(d)| < r``: PCR truncates to the pool's rank and the fit degrades.

    The cross is still found -- the mask holds one -- but the pre-period fit
    carries error that a spanning pool does not, which is Assumption 4 failing
    and the diagnostic that reports it.
    """
    p = _tensor_panel(n_per_arm=(14, 1, 7), noise=0.0)
    target = int(p["pool"][0][0])
    S = p["M"][np.ix_(p["pool"][1], p["pre_cols"])]
    q = p["M"][target, p["pre_cols"]]
    beta = pcr_weights(S.T, q, RANK)
    resid = S.T @ beta - q
    assert float(resid @ resid) / float(q @ q) > 1e-6


def test_an_observed_entry_is_not_a_target():
    """A unit inside ``I(d)`` has its ``(d, t)`` entry observed, so nothing is imputed."""
    p = _tensor_panel()
    inside = int(p["pool"][2][0])
    j = p["cols"].index((2, p["t0"]))
    assert p["mask"][inside, j] == 1
    assert np.isfinite(p["M"][inside, j])
