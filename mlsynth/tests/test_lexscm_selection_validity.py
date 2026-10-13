"""Is LEXSCM's design ranking allowed to look at the blank window?

Vives-i-Bastida (2022), Section 4, states the constraint plainly: "we can't
decide the outcome pair on the basis of the fit in the blank period (otherwise
we would bias our statistical analysis). We can use the blank periods ... to
evaluate whether our model is likely to have over-fitted." LEXSCM ranks
candidate designs by ``mde_sd``, which is computed from the blank-window
residuals, and then builds its inference from those same residuals. On the face
of it that is the thing the paper forbids: the window is used to choose and
then to test, so the winner's residuals are a minimum over candidates and the
null built from them would be too narrow.

Measured, it is not. These tests pin the property that makes the practice sound
here, so that a later change to the search which *does* induce a winner's curse
fails instead of passing.

The detector is the ranking's out-of-sample payoff, not the shape of its
blank-window residuals. Comparing the winner's blank-to-post RMSE ratio against
an unranked control turns out to have no power at all: under a panel where the
candidates differ only by noise, and again with the balance gate removed so the
ranking mines noise over all 40 designs, that ratio moves *up* (0.91 and 0.99
against 0.88 unranked) instead of down, because selecting a design with a low
blank-window sigma also picks up part of its post-window sigma. That check was
written, measured, and dropped; the positive control below is what remains of
it.

The reason the curse does not bite is that the candidates differ in true
quality by more than they differ by noise. Stage 1 hands Stage 4 designs that
already satisfy the balance objective, and among those the blank-window MDE is
picking real differences in donor-fit stability, not sampling noise -- so the
improvement it finds on the blank window is delivered on the post window too.

What the blank window does understate is the post-window error, by roughly a
sixth on this configuration. That is a horizon effect and not a selection one:
it is present in the same degree for a design chosen without ever consulting
the blank window, because the blank periods sit adjacent to the estimation
window and the post periods sit beyond them. It is priced by the placebo-bias
and scale-uncertainty terms on ``res.power``, not by the ranking rule.

No client data is used: the panels are simulated from a known factor model with
deliberately heterogeneous unit noise, which is what gives the comparison
against an unranked control the power to fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mlsynth.estimators.lexscm as lexscm_module
from mlsynth import LEXSCM
from mlsynth.utils.fast_scm_helpers.lexselect import select_design as SHIPPED

N_UNITS, N_CAND, T, N_POST, M, TOP_K, N_PANELS = 24, 14, 44, 6, 3, 40, 30
N_PRE = T - N_POST
N_FIT = int(round(0.7 * N_PRE))          # LEXSCM's default frac_E


def _panel(seed: int) -> pd.DataFrame:
    """A three-factor panel whose units differ in noise scale by up to 7.5x."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(N_UNITS, 3))
    factors = np.cumsum(rng.normal(size=(T, 3)), axis=0)
    Y = (100.0 + rng.normal(0, 5, N_UNITS) + factors @ loadings.T
         + rng.normal(0, 1.0, (T, N_UNITS)) * rng.uniform(0.4, 3.0, N_UNITS))
    cand = sorted(rng.choice(N_UNITS, N_CAND, replace=False).tolist())
    return pd.DataFrame([
        {"unitid": f"u{j}", "time": t, "y": Y[t, j],
         "candidate": int(j in cand), "post": int(t >= N_PRE)}
        for j in range(N_UNITS) for t in range(T)])


def _unranked(designs, **kw):
    """A control that never consults a blank-window quantity.

    Stage 1 has already filtered to designs meeting the balance objective, so
    taking a fixed one of those isolates what Stage 4's ranking contributes.
    """
    rec = SHIPPED(designs, **kw)
    rec.winner = sorted(designs, key=lambda d: d.design_id)[len(designs) // 2]
    return rec


def _fit(df: pd.DataFrame, rule):
    lexscm_module.select_design = rule
    try:
        return LEXSCM({
            "df": df, "outcome": "y", "unitid": "unitid", "time": "time",
            "candidate_col": "candidate", "m": M, "post_col": "post",
            "n_post_grid": [N_POST], "top_K": TOP_K, "verbose": False,
            "display_graph": False,
        }).fit()
    finally:
        lexscm_module.select_design = SHIPPED


def _rms_gap(df: pd.DataFrame, res, lo: int, hi: int) -> float:
    """Root-mean-square synthetic-control gap over ``[lo, hi)``."""
    tw = {str(k): float(v)
          for k, v in res.design_weights.summary_stats["treated_weights"].items()}
    dw = {str(k): float(v) for k, v in res.design_weights.donor_weights.items()}
    Y = df.pivot(index="time", columns="unitid", values="y").sort_index()
    gap = (Y[list(tw)].to_numpy() @ np.fromiter(tw.values(), float)
           - Y[list(dw)].to_numpy() @ np.fromiter(dw.values(), float))
    return float(np.sqrt(np.mean(gap[lo:hi] ** 2)))


def _flat_panel(seed: int) -> pd.DataFrame:
    """As ``_panel``, but every unit shares one noise scale.

    The candidates are then equal in true quality, so anything the blank-window
    ranking prefers it prefers for noise.
    """
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(N_UNITS, 3))
    factors = np.cumsum(rng.normal(size=(T, 3)), axis=0)
    Y = (100.0 + rng.normal(0, 5, N_UNITS) + factors @ loadings.T
         + rng.normal(0, 1.0, (T, N_UNITS)))
    cand = sorted(rng.choice(N_UNITS, N_CAND, replace=False).tolist())
    return pd.DataFrame([
        {"unitid": f"u{j}", "time": t, "y": Y[t, j],
         "candidate": int(j in cand), "post": int(t >= N_PRE)}
        for j in range(N_UNITS) for t in range(T)])


def _both_arms(panel_fn) -> pd.DataFrame:
    """Both selection rules on the same panels, scored on both windows."""
    rows = []
    for seed in range(N_PANELS):
        df = panel_fn(seed)
        for name, rule in (("shipped", SHIPPED), ("unranked", _unranked)):
            res = _fit(df, rule)
            rows.append({
                "seed": seed, "rule": name,
                "blank": _rms_gap(df, res, N_FIT, N_PRE),
                "post": _rms_gap(df, res, N_PRE, T),
            })
    out = pd.DataFrame(rows)
    out["ratio"] = out["blank"] / out["post"]
    return out


@pytest.fixture(scope="module")
def flat_arms() -> pd.DataFrame:
    return _both_arms(_flat_panel)


@pytest.fixture(scope="module")
def arms() -> pd.DataFrame:
    """Both selection rules run on the same panels, scored on both windows."""
    return _both_arms(_panel)


def test_ranking_on_the_blank_window_buys_real_out_of_sample_quality(arms):
    """The improvement the ranking finds is delivered on periods it never saw.

    If the blank-window MDE were selecting noise, the winner would look better
    on the blank window and no better on the post window.
    """
    paired = arms.pivot(index="seed", columns="rule", values="post")
    ratio = float((paired["shipped"] / paired["unranked"]).median())
    assert ratio < 0.95, (
        f"ranking bought no post-window quality: median post-window RMSE ratio "
        f"{ratio:.3f} against the unranked control")


def test_ranking_buys_nothing_when_candidates_differ_only_by_noise(flat_arms):
    """The positive control: the test above can fail, and this is when.

    Same estimator, same ranking, on a panel whose units share one noise scale,
    so the candidates are equal in true quality and the blank-window MDE has
    only sampling noise to sort on. The ranking then buys nothing out of sample,
    which is the winner's curse showing itself. Without this arm the test above
    would be passing on a design that had no way to fail.
    """
    paired = flat_arms.pivot(index="seed", columns="rule", values="post")
    ratio = float((paired["shipped"] / paired["unranked"]).median())
    assert ratio > 0.95, (
        f"expected the ranking to buy nothing when there is nothing to find; "
        f"got a post-window RMSE ratio of {ratio:.3f}")


def test_the_blank_window_understates_the_post_window_for_every_rule(arms):
    """The gap that is real, and is a horizon effect and not a selection one.

    Blank periods sit adjacent to the estimation window and post periods sit
    beyond them, so the blank-window sigma runs optimistic whether or not the
    ranking consulted it. This is what ``res.power``'s placebo-bias and
    scale-uncertainty terms exist to price; it is not fixed by changing the
    ranking rule.
    """
    for rule in ("shipped", "unranked"):
        ratio = float(arms.loc[arms.rule == rule, "ratio"].median())
        assert 0.6 <= ratio <= 1.0, (
            f"{rule}: blank/post RMSE ratio {ratio:.3f} outside the measured band")
