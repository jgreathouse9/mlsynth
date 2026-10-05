r"""Where the shared two-group regression stops, and what sits on the other side.

``mlsynth/utils/groupfit`` fits :math:`y_t = \alpha + \beta x_t` from the centred
sums of squares, in closed form. FDID and TBR call it. PANGEO does not, and the
reason is its model: ``_adid`` fits ``[1, YC, t]``, a three-column design with an
optional linear trend, by ``lstsq``, and prices its interval on a long-run
residual variance. ``group_sums`` produces ``(n, S_xx, S_xy, S_yy)``, the
sufficient statistics of a *simple* regression, so a third column has nowhere to
go.

PR #694 introduced the package and recorded, in its commit message, that FDID and
PANGEO would follow on their own branches. FDID's landed (#696). PANGEO's did
not, and a note in a commit message is read once, by whoever reviews that commit.
This file replaces the note with assertions.

Two claims are pinned here.

The models coincide at ``augment=True, trend=False``, which is PANGEO's design
with the trend switched off, and there the two implementations have to agree.
That is the drift guard #694 wanted: if PANGEO's ``lstsq`` path ever diverges
from the shared closed form where both are defined, this fails. Agreement was
measured at 9.7e-15 relative over 200 geo-scale panels before the tolerance below
was chosen.

The callers are exactly FDID and TBR. A fourth copy of the regression appearing,
or one of these two leaving, changes a set this file asserts.
"""
from __future__ import annotations

import ast
import pathlib
import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError
from mlsynth.utils.groupfit import (
    fit_on_sums,
    fit_two_group,
    group_sums,
    is_identified,
)
from mlsynth.utils.pangeo_helpers.effects import _adid, adid_counterfactual

SETTINGS = settings(max_examples=50, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])

#: Relative tolerance on the overlap. Measured worst case is 9.7e-15 over 200
#: panels; this leaves two orders of headroom for a different BLAS and still
#: fails on any drift that changes a reported digit.
OVERLAP_REL = 1e-12


def geo_panel(n_pre=40, n_post=8, level=4.0e4, drift=250.0, noise=200.0,
              alpha=120.0, beta=0.85, seed=0):
    """A control aggregate at geo scale, trending, with the treated series built
    from it so the slope being recovered is known."""
    rng = np.random.default_rng(seed)
    n = n_pre + n_post
    YC = level + drift * np.arange(n, dtype=float) + rng.normal(0, noise, n)
    YT = alpha + beta * YC + rng.normal(0, noise * 0.75, n)
    return YT, YC


# --------------------------------------------------------------------------- #
# the overlap: PANGEO with the trend off is groupfit's model
# --------------------------------------------------------------------------- #
def test_the_slope_agrees_where_the_models_coincide():
    """``augment=True, trend=False`` is one regressor, which both can express."""
    YT, YC = geo_panel()
    n_pre = 40
    pangeo = _adid(YT, YC, n_pre, YT.size - n_pre,
                   augment=True, trend=False, alpha=0.05)
    shared = fit_two_group(YT[:n_pre], YC[:n_pre])
    assert pangeo["scale"] == pytest.approx(shared.beta, rel=OVERLAP_REL)


def test_the_counterfactual_agrees_where_the_models_coincide():
    """The plotted line too, which pins the intercept as well as the slope."""
    YT, YC = geo_panel()
    n_pre = 40
    drawn = adid_counterfactual(YT, YC, n_pre, augment=True, trend=False)
    shared = fit_two_group(YT[:n_pre], YC[:n_pre])
    projected = shared.alpha + shared.beta * YC
    assert np.allclose(drawn, projected, rtol=OVERLAP_REL, atol=0.0)


@st.composite
def panels(draw):
    return geo_panel(
        n_pre=draw(st.integers(12, 60)),
        n_post=draw(st.integers(2, 12)),
        level=draw(st.floats(1.0e3, 1.0e6)),
        drift=draw(st.floats(-500.0, 500.0)),
        noise=draw(st.floats(50.0, 2000.0)),
        beta=draw(st.floats(0.2, 3.0)),
        seed=draw(st.integers(0, 2**31 - 1)),
    )


@SETTINGS
@given(panels())
def test_the_overlap_holds_over_the_panel_domain(case):
    """The agreement is a property of the two formulas, not of one fixture."""
    YT, YC = case
    n_pre = YT.size - 8 if YT.size > 20 else YT.size - 2
    assume(float(np.std(YC[:n_pre])) > 1e-6)
    pangeo = _adid(YT, YC, n_pre, YT.size - n_pre,
                   augment=True, trend=False, alpha=0.05)
    shared = fit_two_group(YT[:n_pre], YC[:n_pre])
    assert pangeo["scale"] == pytest.approx(shared.beta, rel=1e-9)


# --------------------------------------------------------------------------- #
# the boundary: with a trend there is nothing to agree with
# --------------------------------------------------------------------------- #
def test_a_trend_moves_the_slope_off_the_shared_fit():
    """With the trend on, PANGEO is fitting a different model.

    The assertion is that the two differ, which is what makes the trend a real
    third regressor and not a reparametrisation. A tolerance-free equality here
    would mean the trend column was doing nothing.
    """
    YT, YC = geo_panel()
    n_pre = 40
    trended = _adid(YT, YC, n_pre, YT.size - n_pre,
                    augment=True, trend=True, alpha=0.05)
    shared = fit_two_group(YT[:n_pre], YC[:n_pre])
    assert trended["scale"] != pytest.approx(shared.beta, rel=1e-6)


def test_the_shared_fit_takes_two_series_and_not_a_design_matrix():
    """``groupfit`` is the one-regressor case; a design matrix is not its input."""
    YT, YC = geo_panel()
    n_pre = 40
    X = np.column_stack([np.ones(n_pre), YC[:n_pre], np.arange(n_pre, dtype=float)])
    with pytest.raises(MlsynthDataError, match="window"):
        fit_two_group(YT[:n_pre], X)


# --------------------------------------------------------------------------- #
# the degenerate case, where the two disagree on purpose
# --------------------------------------------------------------------------- #
def test_a_constant_regressor_is_refused_here_and_absorbed_there():
    """``S_xx = 0`` leaves no identified slope, and the two answer differently.

    ``groupfit`` refuses and hands the policy back. PANGEO's ``lstsq`` returns a
    minimum-norm solution, which is arithmetically correct and is not an
    estimate of a slope. Both behaviours are deliberate; this pins them so a
    migration cannot change one by inheriting the other.
    """
    YT, _ = geo_panel()
    YC = np.full(YT.size, 4.0e4)
    n_pre = 40

    sums = group_sums(YT[:n_pre], YC[:n_pre])
    assert not is_identified(sums)
    with pytest.raises(MlsynthEstimationError):
        fit_on_sums(sums, YT[:n_pre], YC[:n_pre])

    absorbed = _adid(YT, YC, n_pre, YT.size - n_pre,
                     augment=True, trend=False, alpha=0.05)
    assert np.isfinite(absorbed["scale"])


# --------------------------------------------------------------------------- #
# the caller set, asserted instead of remembered
# --------------------------------------------------------------------------- #
#: Helper packages that import the shared regression. PANGEO is absent by the
#: design recorded in this file's docstring, not by oversight.
GROUPFIT_CALLERS = {"fdid_helpers", "tbr_helpers"}


def _packages_importing_groupfit() -> set:
    """Helper packages with a live import of ``groupfit``, read off the AST."""
    root = pathlib.Path(__file__).resolve().parents[1] / "utils"
    found = set()
    for path in root.rglob("*.py"):
        if "groupfit" in path.parts or "__pycache__" in path.parts:
            continue
        with warnings.catch_warnings():
            # Parsing every module surfaces escape-sequence warnings that belong
            # to those files and not to this check.
            warnings.simplefilter("ignore", DeprecationWarning)
            tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any("groupfit" in n for n in names):
                found.add(path.relative_to(root).parts[0])
    return found


def test_the_callers_are_the_ones_the_package_claims():
    """A fourth copy of the regression, or a caller leaving, fails here."""
    assert _packages_importing_groupfit() == GROUPFIT_CALLERS


def test_pangeo_is_not_a_caller():
    """Stated separately, because it is the claim this file exists to record."""
    assert "pangeo_helpers" not in _packages_importing_groupfit()
