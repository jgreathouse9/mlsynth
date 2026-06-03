"""Coverage tests for shc_helpers.selection.stepwise_donor_selection."""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.shc_helpers import selection as sel_mod
from mlsynth.utils.shc_helpers.selection import stepwise_donor_selection


def test_returns_simplex_weights_and_paths():
    rng = np.random.default_rng(0)
    m, N = 8, 5
    L_full = rng.normal(size=(m, N))
    L_post = rng.normal(size=(4, N))
    # target is an exact convex combination of two donors -> recoverable
    ell = 0.5 * L_full[:, 0] + 0.5 * L_full[:, 1]

    out = stepwise_donor_selection(L_full, L_post, ell, m)

    assert set(out) == {"best_donors", "best_weights", "best_mse", "mse_path", "bic_path"}
    # weights sit on a simplex
    w = np.asarray(out["best_weights"], dtype=float)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= -1e-7)
    # exact recovery -> near-zero MSE
    assert out["best_mse"] < 1e-10
    # path lengths consistent and non-empty
    assert len(out["mse_path"]) >= 1
    assert len(out["best_donors"]) == len(out["best_weights"])
    # best_mse equals the minimum along the recorded path
    assert np.isclose(out["best_mse"], min(out["mse_path"]))


def test_bic_early_stop_branch():
    """With many donors and a noisy target, the BIC monotone-increase break fires."""
    rng = np.random.default_rng(1)
    m, N = 6, 8
    L_full = rng.normal(size=(m, N))
    L_post = rng.normal(size=(3, N))
    ell = rng.normal(size=m)  # unstructured -> adding donors stops helping
    out = stepwise_donor_selection(L_full, L_post, ell, m)
    # bic_path may be one longer than mse_path when the loop breaks after
    # appending the BIC but before recording the step.
    assert len(out["bic_path"]) >= len(out["mse_path"])
    assert len(out["bic_path"]) >= 1


def test_all_infeasible_breaks(monkeypatch):
    """If every QP returns None, the loop hits the `best_idx is None` break.

    The function then has an empty mse_list; argmin([]) raises -- this is the
    degenerate path. We assert the documented failure mode rather than a value.
    """
    rng = np.random.default_rng(2)
    m, N = 5, 3
    L_full = rng.normal(size=(m, N))
    L_post = rng.normal(size=(2, N))
    ell = rng.normal(size=m)

    monkeypatch.setattr(sel_mod, "solve_shc_qp", lambda *a, **k: (None, None))

    with pytest.raises(ValueError):
        stepwise_donor_selection(L_full, L_post, ell, m)


def test_skip_candidate_when_qp_returns_none(monkeypatch):
    """A donor whose QP is infeasible is skipped (the `if w is not None` guard)."""
    rng = np.random.default_rng(3)
    m, N = 6, 3
    L_full = rng.normal(size=(m, N))
    L_post = rng.normal(size=(2, N))
    ell = 0.5 * L_full[:, 1] + 0.5 * L_full[:, 2]

    real = sel_mod.solve_shc_qp
    call = {"i": 0}

    def flaky(L, ell_eval, *args, **kwargs):
        # Make the very first single-donor candidate infeasible, forcing the
        # selector to skip it and pick a feasible one instead.
        call["i"] += 1
        if call["i"] == 1:
            return (None, None)
        return real(L, ell_eval, *args, **kwargs)

    monkeypatch.setattr(sel_mod, "solve_shc_qp", flaky)
    out = stepwise_donor_selection(L_full, L_post, ell, m)
    assert np.isclose(np.asarray(out["best_weights"], dtype=float).sum(), 1.0)
