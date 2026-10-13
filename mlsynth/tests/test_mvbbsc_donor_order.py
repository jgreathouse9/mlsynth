"""MVBBSC's answer must not depend on the order its donors arrive in.

Column order carries no information about the effect: the donor pool is a set,
and relabelling its members is a relabelling of the caller's bookkeeping. NUTS
does not see it that way. The sampler walks a parameter vector, so permuting the
simplex coordinates changes the trajectory, and two runs at one seed on one
panel disagree by however far that walk diverges.

Measured on the German reunification panel before this was addressed: permuting
the 16 donors moved the posterior-mean weights by 7.2e-3 and the mean post-1990
ATT by 6.03, which is 0.29% of the estimate. Small against that panel's pinned
MCMC tolerances, and still a number that changes because somebody sorted a
dataframe.

The fix is to canonicalise the donor columns inside the sampler and map the
draws back, so every permutation of one donor set presents NUTS with one input.
That makes the relation exact, which is why these assertions carry no tolerance.

The settings are parametrized, and that is not incidental. ``target_accept``
defaults to 0.8 and moves the ATT on this panel by 5.92, which is the same size
as the 6.03 the donor order moved it. Checking the property at one setting is
checking it where two effects of equal magnitude can cancel: the pre-fix natural
order at 0.9 and the post-fix canonical order at 0.8 agree to 0.1 by
coincidence, which is enough to make a working fix look inert and a broken one
look fine. The property has to hold at the default and not only where it was
first measured.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpyro", reason="MVBBSC needs mlsynth[bayes]")

from mlsynth.utils.mvbbsc_helpers.model import run_mvbbsc

_BASE = dict(n_warmup=200, n_samples=200, n_chains=2, seed=0)
# 0.8 is the config default; 0.9 is what the property was first measured at.
_ACCEPTS = (0.8, 0.9)


def _kw(target_accept):
    return dict(_BASE, target_accept=target_accept)


def _panel(n_periods=30, n_donors=6, n_pre=22, seed=5):
    rng = np.random.default_rng(seed)
    t = np.arange(n_periods)
    donors = (rng.uniform(80.0, 400.0, n_donors)[None, :]
              + (12.0 * np.sin(2 * np.pi * t / 6.0))[:, None]
              + rng.normal(0.0, 6.0, (n_periods, n_donors)))
    y = donors @ rng.dirichlet(np.ones(n_donors)) + rng.normal(0.0, 4.0, n_periods)
    return y, donors, n_pre


@pytest.mark.parametrize("target_accept", _ACCEPTS)
@pytest.mark.parametrize("key", [0, 1, 2])
def test_donor_permutation_leaves_the_posterior_unchanged(key, target_accept):
    y, donors, n_pre = _panel()
    perm = np.random.default_rng(key).permutation(donors.shape[1])
    base = run_mvbbsc(y, donors, n_pre, **_kw(target_accept))
    shuffled = run_mvbbsc(y, donors[:, perm], n_pre, **_kw(target_accept))
    # weights follow their donors, exactly
    np.testing.assert_allclose(shuffled["weights"].mean(axis=0),
                               base["weights"].mean(axis=0)[perm], atol=1e-12)
    # and the counterfactual, which does not depend on bookkeeping at all
    np.testing.assert_allclose(shuffled["counterfactual"].mean(axis=0),
                               base["counterfactual"].mean(axis=0), atol=1e-12)


@pytest.mark.parametrize("target_accept", _ACCEPTS)
def test_reversing_the_donor_columns_leaves_the_att_unchanged(target_accept):
    y, donors, n_pre = _panel()
    base = run_mvbbsc(y, donors, n_pre, **_kw(target_accept))
    reversed_ = run_mvbbsc(y, donors[:, ::-1], n_pre, **_kw(target_accept))
    att = lambda r: float(np.mean(y[n_pre:] - r["counterfactual"].mean(axis=0)[n_pre:]))
    assert att(reversed_) == pytest.approx(att(base), abs=1e-9)


def test_weights_still_describe_the_donors_they_are_returned_against():
    """Canonicalising must not silently hand back weights in the sampler's
    internal order: the fit has to reconstruct from the caller's columns."""
    y, donors, n_pre = _panel()
    out = run_mvbbsc(y, donors, n_pre, **_kw(0.8))
    w = out["weights"].mean(axis=0)
    loc, scale = float(y[:n_pre].mean()), float(y[:n_pre].std(ddof=1))
    d_loc = donors[:n_pre].mean(axis=0)
    d_scale = np.where(donors[:n_pre].std(axis=0, ddof=1) > 0,
                       donors[:n_pre].std(axis=0, ddof=1), 1.0)
    rebuilt = ((donors - d_loc) / d_scale @ w) * scale + loc
    # the noiseless mean rebuilt from the returned weights must track the
    # posterior-mean counterfactual, which carries only the iid shock on top
    got = out["counterfactual"].mean(axis=0)
    assert float(np.max(np.abs(rebuilt - got))) < 0.5 * float(np.std(y[:n_pre]))


def test_weights_remain_a_simplex():
    y, donors, n_pre = _panel()
    w = run_mvbbsc(y, donors, n_pre, **_kw(0.8))["weights"]
    assert np.all(w >= -1e-9)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)


def test_single_donor_is_unaffected():
    y, donors, n_pre = _panel(n_donors=1)
    out = run_mvbbsc(y, donors, n_pre, **_kw(0.8))
    np.testing.assert_allclose(out["weights"].mean(axis=0), [1.0], atol=1e-6)


@pytest.mark.parametrize("target_accept", _ACCEPTS)
def test_the_estimator_is_order_invariant_end_to_end(target_accept):
    """The property has to survive the estimator, not only the sampler helper.

    A user reorders rows in a dataframe, not columns in an array, so this is the
    form the sensitivity actually reached people in.
    """
    import pandas as pd

    from mlsynth import MVBBSC

    y, donors, n_pre = _panel()
    n_periods = donors.shape[0]
    rows = []
    for t in range(n_periods):
        rows.append({"unit": "treated", "t": t, "y": y[t],
                     "treat": int(t >= n_pre)})
        for j in range(donors.shape[1]):
            rows.append({"unit": f"d{j}", "t": t, "y": donors[t, j], "treat": 0})
    frame = pd.DataFrame(rows)
    cfg = dict(outcome="y", treat="treat", unitid="unit", time="t",
               n_warmup=200, n_samples=200, n_chains=2,
               target_accept=target_accept, seed=0, display_graphs=False)
    forward = MVBBSC({"df": frame, **cfg}).fit()
    # the same panel with the donor rows in the opposite order
    backward = MVBBSC({"df": frame.iloc[::-1].reset_index(drop=True), **cfg}).fit()
    assert float(backward.att) == pytest.approx(float(forward.att), abs=1e-9)
