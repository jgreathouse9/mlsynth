"""VanillaSC's conformal band against the procedure it reports p-values from.

The band and the p-values in ``res.inference.details`` are two readings of one
test, so they have to agree: an endpoint the band reports is a null the test
does not reject, and a null the test rejects is outside the band. Nothing tied
them together, and they disagreed -- the inversion kept ``p == alpha``, which is
a rejection.

Whether that is visible depends on the panel, which is why it survived. A
conformal p-value from a single-period inversion lives on the grid
``k / (T0 + 1)``, so the inclusive and strict rules differ only when ``alpha``
is one of those values. On the Swedish carbon tax panel ``T0 + 1 = 31`` and
``alpha = 0.1`` falls between ``3/31`` and ``4/31``, so the two rules agree
exactly and every existing case passes under either. On the authors' own JASA
application ``T0 + 1 = 20``, ``alpha = 0.1`` is attained exactly, and the
inclusive rule widens all six per-period intervals by 15 to 40 percent.

The panels here are built to attain the level, so the boundary is exercised
instead of avoided. The value-for-value comparison against ``scinference`` on
the authors' own data is the ``cwz_conformal`` benchmark case.
"""
import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.conformal import conformal_refit_gaps

ALPHA = 0.1
T0, T1 = 19, 3          # T0 + 1 = 20, so p can equal alpha = 0.1 exactly
J = 6


def _panel(seed: int = 11, effect: float = -1.5) -> pd.DataFrame:
    """A donor pool the treated unit sits inside, plus a post-period effect."""
    rng = np.random.default_rng(seed)
    T = T0 + T1
    trend = np.linspace(0.0, 4.0, T)[:, None]
    Y0 = 10.0 + trend + rng.normal(0.0, 0.6, size=(T, J)) + np.arange(J)[None, :]
    w = np.zeros(J)
    w[:3] = [0.5, 0.3, 0.2]
    y = Y0 @ w + rng.normal(0.0, 0.25, size=T)
    y[T0:] += effect
    M = np.column_stack([y, Y0])
    units = ["treated"] + [f"donor{j}" for j in range(J)]
    df = pd.DataFrame({
        "unit": np.repeat(units, T),
        "t": np.tile(np.arange(T), J + 1),
        "yv": M.T.ravel(),
    })
    df["treated"] = ((df.unit == "treated") & (df.t >= T0)).astype(int)
    return df


def _cfg(**kw) -> dict:
    return {"df": _panel(), "outcome": "yv", "treat": "treated", "unitid": "unit",
            "time": "t", "backend": "outcome-only", "inference": "conformal",
            "conformal_type": "block", "alpha": ALPHA, "display_graphs": False,
            **kw}


def _period_p(df: pd.DataFrame, k: int, tau0: float) -> float:
    """The inversion's own p-value at candidate ``tau0`` for post period ``k``.

    Recomputed from the public refit rule, not read off the result, so
    the assertion is against the procedure and not against the same array the
    band came from.
    """
    wide = df.pivot(index="t", columns="unit", values="yv")
    y = wide["treated"].to_numpy(dtype=float)
    Y0 = wide[[c for c in wide.columns if c != "treated"]].to_numpy(dtype=float)
    idx = list(range(T0)) + [T0 + k]
    y_fit = y[idx].copy()
    y_fit[-1] -= tau0
    u = conformal_refit_gaps(y_fit, Y0[idx], rule="sc")
    return float(np.mean(np.abs(u) >= np.abs(u[-1])))


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------

def test_smoke_conformal_band_runs_and_is_finite():
    res = VanillaSC(_cfg()).fit()
    d = res.inference.details
    assert np.isfinite(np.asarray(d["pi_lower"], float)).all()
    assert np.isfinite(np.asarray(d["pi_upper"], float)).all()
    assert (np.asarray(d["pi_lower"], float) <= np.asarray(d["pi_upper"], float)).all()
    assert d["refit"] == "sc"


# --------------------------------------------------------------------------
# the duality invariant -- the rung that failed
# --------------------------------------------------------------------------

def test_every_reported_endpoint_is_a_null_the_test_does_not_reject():
    grid = np.round(np.arange(-4.0, 2.0 + 1e-9, 0.05), 10)
    res = VanillaSC(_cfg(conformal_grid=grid)).fit()
    d = res.inference.details
    lo = np.asarray(d["pi_lower"], float)
    hi = np.asarray(d["pi_upper"], float)
    df = _panel()
    for k in range(T1):
        for bound in (lo[k], hi[k]):
            assert _period_p(df, k, bound) > ALPHA, (
                f"period {k}: reported endpoint {bound} has p="
                f"{_period_p(df, k, bound):.4f} <= alpha={ALPHA}, so the band "
                "keeps a null the test rejects"
            )


def test_a_candidate_whose_p_value_equals_alpha_is_outside_the_band():
    """The discriminating case: ties are rejections.

    With ``T0 + 1 = 20`` residuals the attainable p-values are multiples of
    ``0.05``, so candidates with ``p == 0.1`` exist on any reasonable grid.
    Every one of them must fall outside the reported interval.
    """
    grid = np.round(np.arange(-4.0, 2.0 + 1e-9, 0.05), 10)
    res = VanillaSC(_cfg(conformal_grid=grid)).fit()
    d = res.inference.details
    lo = np.asarray(d["pi_lower"], float)
    hi = np.asarray(d["pi_upper"], float)
    df = _panel()
    seen_a_tie = False
    for k in range(T1):
        ps = np.array([_period_p(df, k, t) for t in grid])
        ties = grid[np.isclose(ps, ALPHA)]
        seen_a_tie |= ties.size > 0
        for t in ties:
            assert not (lo[k] <= t <= hi[k]), (
                f"period {k}: candidate {t} has p == alpha exactly and is inside "
                f"the reported band [{lo[k]}, {hi[k]}]"
            )
    assert seen_a_tie, "panel does not exercise the boundary; the test is vacuous"


# --------------------------------------------------------------------------
# the caller-supplied grid
# --------------------------------------------------------------------------

def test_the_band_is_read_off_the_supplied_grid():
    grid = np.round(np.arange(-4.0, 2.0 + 1e-9, 0.05), 10)
    d = VanillaSC(_cfg(conformal_grid=grid)).fit().inference.details
    for bound in np.concatenate([np.asarray(d["pi_lower"], float),
                                 np.asarray(d["pi_upper"], float)]):
        assert np.isclose(grid, bound).any(), f"{bound} is not a point of the grid"


def test_refining_the_grid_grows_the_band_by_at_most_one_coarse_step():
    """Grid resolution is the band's resolution, and the direction is outward.

    The accepted set is a subset of the candidates swept, so a finer grid that
    contains the coarse one can only accept more: the endpoints move outward,
    never inward. That is the sense in which a coarse grid understates the
    interval, and the amount is bounded by one coarse step -- which is what
    makes a resolution-limited comparison against another implementation
    interpretable instead of merely approximate.
    """
    coarse = np.round(np.arange(-4.0, 2.0 + 1e-9, 0.20), 10)
    fine = np.round(np.arange(-4.0, 2.0 + 1e-9, 0.05), 10)
    assert np.isin(coarse, fine).all(), "the fine grid must refine the coarse one"
    dc = VanillaSC(_cfg(conformal_grid=coarse)).fit().inference.details
    dfine = VanillaSC(_cfg(conformal_grid=fine)).fit().inference.details
    lo_c, hi_c = np.asarray(dc["pi_lower"], float), np.asarray(dc["pi_upper"], float)
    lo_f, hi_f = np.asarray(dfine["pi_lower"], float), np.asarray(dfine["pi_upper"], float)
    assert (lo_f <= lo_c + 1e-9).all() and (hi_f >= hi_c - 1e-9).all()
    assert (lo_c - lo_f <= 0.20 + 1e-9).all() and (hi_f - hi_c <= 0.20 + 1e-9).all()


def test_the_grid_does_not_disturb_the_reported_no_effect_p_value():
    """Zero is tested whether or not the caller's grid contains it."""
    without_zero = np.round(np.arange(-4.0, -0.05 + 1e-9, 0.05), 10)
    assert not np.isclose(without_zero, 0.0).any()
    d_grid = VanillaSC(_cfg(conformal_grid=without_zero)).fit().inference.details
    d_auto = VanillaSC(_cfg()).fit().inference.details
    assert np.allclose(np.asarray(d_grid["period_p_value"], float),
                       np.asarray(d_auto["period_p_value"], float))


# --------------------------------------------------------------------------
# the permutation knobs
# --------------------------------------------------------------------------

def test_block_scheme_ignores_the_draw_count_because_it_enumerates():
    """``block`` is deterministic: its reference set is the T cyclic shifts."""
    a = VanillaSC(_cfg(conformal_n_perm=50)).fit().inference.details["joint_p_value"]
    b = VanillaSC(_cfg(conformal_n_perm=5000)).fit().inference.details["joint_p_value"]
    assert a == b
    assert a >= 1.0 / (T0 + T1) - 1e-12       # one shift is the observed path


def test_finite_sample_convention_cannot_report_zero():
    """``(1 + #)/(1 + n)`` is scinference's iid form and is bounded below.

    The plain mean returns exactly 0 when the observed statistic beats every
    draw, and zero is not a p-value: ``n`` draws cannot evidence more than
    ``1 / (n + 1)``.
    """
    cfg = _cfg(conformal_type="iid", conformal_n_perm=200)
    plain = VanillaSC({**cfg, "df": _panel(effect=-40.0)}).fit()
    fs = VanillaSC({**cfg, "df": _panel(effect=-40.0),
                    "conformal_finite_sample": True}).fit()
    p_plain = plain.inference.details["joint_p_value"]
    p_fs = fs.inference.details["joint_p_value"]
    assert p_plain == 0.0
    assert p_fs == pytest.approx(1.0 / 201.0)
    assert p_fs > p_plain


def test_the_knobs_are_reported_in_details():
    d = VanillaSC(_cfg(conformal_n_perm=321, conformal_finite_sample=True)).fit()
    det = d.inference.details
    assert det["n_perm"] == 321
    assert det["finite_sample"] is True
    assert det["conformal_type"] == "block"


def test_n_perm_defaults_to_scpi_sims():
    d = VanillaSC(_cfg(scpi_sims=137)).fit().inference.details
    assert d["n_perm"] == 137


# --------------------------------------------------------------------------
# failure -- reported, not swallowed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_a_bad_draw_count_is_rejected_at_config_time(bad):
    with pytest.raises(MlsynthConfigError, match="conformal_n_perm"):
        VanillaSC(_cfg(conformal_n_perm=bad))


@pytest.mark.parametrize(
    "bad, msg",
    [([], "at least one candidate"),
     ([0.0, np.nan], "finite"),
     ([np.inf], "finite")],
)
def test_a_bad_grid_is_rejected_at_config_time(bad, msg):
    with pytest.raises(MlsynthConfigError, match=msg):
        VanillaSC(_cfg(conformal_grid=bad))


def test_an_unknown_permutation_scheme_is_rejected_at_config_time():
    with pytest.raises(MlsynthConfigError):
        VanillaSC(_cfg(conformal_type="moving-block"))
