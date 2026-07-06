"""Scaling robustness for SPOTSYNTH's frequentist simplex SC solver.

The paper (O'Riordan & Gilligan-Lee 2025, Algorithm 1 step 1) normalises the
target and donors so the procedure is invariant to the scale of the outcome.
``simplex_weights`` fed the raw outcomes to CLARABEL, so a large-magnitude panel
(e.g. Spotify monthly-listener counts, ~1e7) produced a badly-scaled QP the
solver could not solve. Scaling ``y`` and ``D`` by a common constant leaves the
simplex argmin unchanged (the objective scales by ``c^2``), so the fix conditions
the QP without altering a single weight.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.spotsynth_helpers import simplex_weights


def _panel(scale: float, seed: int = 0):
    """A small, well-posed SC panel scaled by ``scale``."""
    rng = np.random.default_rng(seed)
    T, J = 40, 5
    D = rng.normal(0.0, 1.0, size=(T, J))
    y = D @ np.array([0.4, 0.3, 0.2, 0.1, 0.0]) + rng.normal(0.0, 0.05, size=T)
    return y * scale, D * scale


class TestSimplexWeightsScaling:
    def test_large_magnitude_panel_solves(self):
        # A Spotify-magnitude panel (~1e7) that the raw CLARABEL solve chokes on.
        y, D = _panel(scale=1e7)
        w, cf = simplex_weights(y, D, T0=25)
        assert np.isfinite(w).all()
        assert np.isclose(w.sum(), 1.0, atol=1e-6)
        assert np.all(w >= -1e-9)
        assert cf.shape == (40,)

    def test_weights_are_scale_invariant(self):
        # The simplex argmin is invariant to a common rescaling of y and D, so
        # the weights at scale 1 and scale 1e6 must coincide.
        w1, _ = simplex_weights(*_panel(scale=1.0), T0=25)
        w2, _ = simplex_weights(*_panel(scale=1e6), T0=25)
        assert np.allclose(w1, w2, atol=1e-6)

    def test_matches_golden_on_wellscaled_panel(self):
        # On a panel the raw solver already handled, the scaled solver must
        # reproduce the pre-fix weights (captured from the original CLARABEL fit).
        golden = np.array([0.40154771137081924, 0.30283477708141837,
                           0.19677718892426052, 0.09884032262350192, 0.0])
        w, _ = simplex_weights(*_panel(scale=1.0), T0=25)
        assert np.allclose(w, golden, atol=1e-6)

    def test_counterfactual_is_on_the_raw_scale(self):
        # The returned counterfactual must be D_raw @ w, i.e. on the input scale,
        # not the internal solve scale.
        y, D = _panel(scale=1e7)
        w, cf = simplex_weights(y, D, T0=25)
        assert np.allclose(cf, D @ w, atol=1e-3)
