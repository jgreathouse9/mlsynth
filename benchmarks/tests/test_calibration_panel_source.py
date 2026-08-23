"""Where the calibration study's panel comes from, and how it says so.

The study was built against a proprietary panel read from ``MLSYNTH_CAL_PANEL``.
With the variable unset every real-data arm raised and skipped, so the study was
unrunnable for anyone without that file. It now falls back to the synthetic
generator, which inverts the default: the study runs everywhere, and the real
panel is an option.

That makes provenance the thing to guard. An arm must be able to say which panel
produced its numbers, so a results file can never read as real-data agreement
when a substitution happened -- the failure #473 was about.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest

from benchmarks.studies.cumulative_calibration import panel as P


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv("MLSYNTH_CAL_PANEL", raising=False)
    P._CACHE.clear()
    yield
    P._CACHE.clear()


# --------------------------------------------------------------------------- smoke
def test_the_study_loads_with_nothing_configured():
    mean, factors, loadings, residuals = P.load()
    assert np.isfinite(mean).all() and np.isfinite(factors).all()
    assert np.isfinite(loadings).all() and np.isfinite(residuals).all()


def test_a_draw_comes_back_with_nothing_configured():
    Y = P.draw(seed=0, J=20)
    assert Y.shape == (20, P.T)
    assert np.isfinite(Y).all()


# ------------------------------------------------------------------ unit invariants
def test_the_components_agree_on_their_shapes():
    mean, factors, loadings, residuals = P.load()
    n_units, t_full = mean.size, factors.shape[0]
    assert factors.shape == (t_full, P.K)
    assert loadings.shape == (n_units, P.K)
    assert residuals.shape == (t_full, n_units)


def test_the_source_says_synthetic_when_nothing_is_configured():
    assert P.source() == "synthetic"


def test_the_source_names_the_file_when_one_is_configured(tmp_path, monkeypatch):
    csv = tmp_path / "p.csv"
    csv.write_text("dma,start_date,total\n")
    monkeypatch.setenv("MLSYNTH_CAL_PANEL", str(csv))
    assert P.source() == str(csv)


def test_the_panel_is_long_enough_for_the_design():
    """The arms need T0 + H periods and J units; the fallback has to supply them."""
    mean, factors, _, _ = P.load()
    assert factors.shape[0] >= P.T
    assert mean.size >= 20


def test_a_draw_is_reproducible_from_its_seed():
    assert np.array_equal(P.draw(seed=3, J=8), P.draw(seed=3, J=8))


def test_different_seeds_draw_different_panels():
    assert not np.allclose(P.draw(seed=3, J=8), P.draw(seed=4, J=8))


def test_the_residuals_are_what_the_factors_left_behind():
    """The decomposition is exact, so it can be reassembled."""
    mean, factors, loadings, residuals = P.load()
    rebuilt = mean[None, :] + factors @ loadings.T + residuals
    assert np.isfinite(rebuilt).all()
    assert rebuilt.shape == residuals.shape


# ------------------------------------------------------------------------- edge
def test_a_draw_can_ask_for_one_unit():
    assert P.draw(seed=0, J=1).shape == (1, P.T)


def test_the_frame_carries_no_identifying_label():
    """Units are positional; nothing in the frame names a place or a date."""
    f = P.frame(P.draw(seed=0, J=3))
    assert set(f.columns) == {"unit", "time", "y"}
    assert set(f.unit.unique()) == {"d000", "d001", "d002"}
    assert f.time.min() == 0


# ---------------------------------------------------------------------- failure
def test_a_configured_path_that_does_not_exist_is_refused(monkeypatch):
    monkeypatch.setenv("MLSYNTH_CAL_PANEL", "/nonexistent/panel.csv")
    with pytest.raises(P.PanelUnavailable, match="missing file"):
        P.load()


def test_a_configured_path_is_never_silently_replaced(tmp_path, monkeypatch):
    """Pointing at an unreadable panel must raise, not fall back to synthetic."""
    csv = tmp_path / "empty.csv"
    csv.write_text("not,a,panel\n1,2,3\n")
    monkeypatch.setenv("MLSYNTH_CAL_PANEL", str(csv))
    with pytest.raises(Exception) as exc:
        P.load()
    assert not isinstance(exc.value, SystemExit)


# ------------------------------------------- the way the arms actually import it
def test_the_fallback_works_when_panel_is_imported_as_a_script_module():
    """The arms put the study directory on ``sys.path`` and ``import panel``.

    That makes ``panel`` a top-level module with no package around it, so a
    relative import of the generator fails there while passing under the package
    import this file uses. A subprocess is the only way to ask the question,
    since module identity is decided once per interpreter.
    """
    import subprocess
    import sys as _sys
    from pathlib import Path

    study = Path(__file__).resolve().parents[1] / "studies" / "cumulative_calibration"
    code = (
        "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
        "import panel as P\n"
        "Y = P.draw(seed=0, J=4)\n"
        "print(P.source(), Y.shape[0], Y.shape[1])\n"
        % (str(study.parents[2]), str(study))
    )
    env = {k: v for k, v in __import__("os").environ.items()
           if k != "MLSYNTH_CAL_PANEL"}
    r = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                       text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["synthetic", "4", str(P.T)]
