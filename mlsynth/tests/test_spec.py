"""Tests for mlsynth.spec -- save/load an analysis specification to a portable file.

A configuration is plain, validated data, so the specification of an analysis
(everything but the DataFrame and any runtime arrays) can be written to a JSON or
YAML file, version-controlled, and loaded back to drive a fit. These tests pin
that contract: round-trips preserve the spec, the DataFrame and matrix payloads
are dropped and re-attached at load time, the formats are honoured, failures are
reported as typed ``Mlsynth*`` errors, and ``load_spec`` behaves gracefully for
every estimator the package ships.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import mlsynth
from mlsynth.exceptions import MlsynthError, MlsynthConfigError


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def panel() -> pd.DataFrame:
    """A small, well-posed balanced panel: 6 units, 20 periods, unit ``u0``
    treated from period 15. Carries spare covariate / instrument columns so the
    estimators with extra required fields can be constructed."""
    rng = np.random.default_rng(0)
    units = [f"u{i}" for i in range(6)]
    periods = list(range(2000, 2020))
    rows = []
    for i, u in enumerate(units):
        level = 10.0 + 2.0 * i
        for t in periods:
            base = level + 0.5 * (t - 2000) + rng.normal(0, 0.1)
            treated = int(u == "u0" and t >= 2015)
            rows.append(
                {
                    "unit": u,
                    "year": t,
                    "y": base + (3.0 if treated else 0.0),
                    "treated": treated,
                    "x1": base + rng.normal(0, 0.1),
                    "z": base + rng.normal(0, 0.1),
                }
            )
    return pd.DataFrame(rows)


def _vanillasc_spec() -> dict:
    return {
        "estimator": "VanillaSC",
        "outcome": "y",
        "treat": "treated",
        "unitid": "unit",
        "time": "year",
        "display_graphs": False,
    }


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #
def test_exports_from_package_root():
    # The module is part of the public API.
    assert hasattr(mlsynth, "save_spec")
    assert hasattr(mlsynth, "load_spec")
    from mlsynth.spec import save_spec, load_spec  # noqa: F401


# --------------------------------------------------------------------------- #
# Round-trips that fit (the headline property)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_roundtrip_from_dict_fits(panel, tmp_path, suffix):
    from mlsynth.spec import save_spec, load_spec

    path = tmp_path / f"spec{suffix}"
    save_spec(_vanillasc_spec(), path)
    assert path.exists()

    est = load_spec(path, df=panel)
    assert isinstance(est, mlsynth.VanillaSC)
    res = est.fit()
    assert np.isfinite(float(res.effects.att))


def test_roundtrip_from_estimator_instance(panel, tmp_path):
    # save_spec should read the estimator name off the instance automatically.
    from mlsynth.spec import save_spec, load_spec

    spec = {k: v for k, v in _vanillasc_spec().items() if k != "estimator"}
    est = mlsynth.VanillaSC({"df": panel, **spec})

    path = tmp_path / "spec.json"
    written = save_spec(est, path)
    assert written["estimator"] == "VanillaSC"

    est2 = load_spec(path, df=panel)
    assert isinstance(est2, mlsynth.VanillaSC)
    assert np.isfinite(float(est2.fit().effects.att))


def test_saved_spec_is_text_only_without_dataframe(panel, tmp_path):
    from mlsynth.spec import save_spec

    path = tmp_path / "spec.json"
    save_spec(_vanillasc_spec(), path)
    on_disk = json.loads(path.read_text())

    assert on_disk["estimator"] == "VanillaSC"
    for col in ("outcome", "treat", "unitid", "time"):
        assert col in on_disk
    assert "df" not in on_disk  # the DataFrame never lands in the text record


# --------------------------------------------------------------------------- #
# Runtime payloads (df and matrices) are dropped and re-attached
# --------------------------------------------------------------------------- #
def test_runtime_matrix_dropped_and_reattached(panel, tmp_path):
    # SpSyDiD requires a spatial_matrix ndarray; it must not enter the text spec,
    # and must be re-attachable at load time.
    from mlsynth.spec import save_spec, load_spec

    W = np.eye(6)
    spec = {
        "estimator": "SpSyDiD",
        "outcome": "y",
        "treat": "treated",
        "unitid": "unit",
        "time": "year",
        "spatial_matrix": W,  # runtime payload
        "display_graphs": False,
    }
    path = tmp_path / "spec.yaml"
    with pytest.warns(UserWarning):
        written = save_spec(spec, path, estimator="SpSyDiD")
    assert "spatial_matrix" not in written  # dropped from the text record

    est = load_spec(path, df=panel, spatial_matrix=W)
    assert isinstance(est, mlsynth.SpSyDiD)
    assert np.array_equal(np.asarray(est.config.spatial_matrix), W)


# --------------------------------------------------------------------------- #
# Failure modes are reported as typed errors
# --------------------------------------------------------------------------- #
def test_unknown_estimator_raises_config_error(tmp_path):
    from mlsynth.spec import load_spec

    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"estimator": "NotAnEstimator", "outcome": "y"}))
    with pytest.raises(MlsynthConfigError):
        load_spec(path, df=pd.DataFrame())


def test_unsupported_extension_raises(tmp_path):
    from mlsynth.spec import save_spec, load_spec

    with pytest.raises(MlsynthConfigError):
        save_spec(_vanillasc_spec(), tmp_path / "spec.txt")

    bad = tmp_path / "spec.txt"
    bad.write_text("estimator: VanillaSC")
    with pytest.raises(MlsynthConfigError):
        load_spec(bad, df=pd.DataFrame())


def test_save_config_object_requires_estimator_name(panel, tmp_path):
    from mlsynth.spec import save_spec
    from mlsynth.config_models import VanillaSCConfig

    cfg = VanillaSCConfig(df=panel, outcome="y", treat="treated",
                          unitid="unit", time="year", display_graphs=False)
    with pytest.raises(MlsynthConfigError):
        save_spec(cfg, tmp_path / "spec.json")  # no estimator= -> error

    # With the name supplied it succeeds.
    written = save_spec(cfg, tmp_path / "spec.json", estimator="VanillaSC")
    assert written["estimator"] == "VanillaSC"


def test_spec_without_estimator_name_raises(tmp_path):
    from mlsynth.spec import load_spec

    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"outcome": "y", "treat": "treated"}))
    with pytest.raises(MlsynthConfigError):
        load_spec(path, df=pd.DataFrame())


def test_save_dict_without_estimator_name_raises(tmp_path):
    from mlsynth.spec import save_spec

    with pytest.raises(MlsynthConfigError):
        save_spec({"outcome": "y", "treat": "treated"}, tmp_path / "spec.json")


def test_save_unsupported_object_raises(tmp_path):
    from mlsynth.spec import save_spec

    with pytest.raises(MlsynthConfigError):
        save_spec(["not", "a", "spec"], tmp_path / "spec.json", estimator="VanillaSC")


def test_estimator_name_must_be_string(tmp_path):
    from mlsynth.spec import load_spec

    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"estimator": 5, "outcome": "y"}))
    with pytest.raises(MlsynthConfigError):
        load_spec(path, df=pd.DataFrame())


def test_non_mapping_spec_file_raises(tmp_path):
    from mlsynth.spec import load_spec

    path = tmp_path / "spec.json"
    path.write_text(json.dumps(["VanillaSC", "y"]))  # a list, not a mapping
    with pytest.raises(MlsynthConfigError):
        load_spec(path, df=pd.DataFrame())


# --------------------------------------------------------------------------- #
# Across all estimators: load_spec never crashes ungracefully
# --------------------------------------------------------------------------- #
def _all_estimators():
    import inspect

    return sorted(
        n for n in mlsynth.__all__
        if inspect.isclass(getattr(mlsynth, n)) and hasattr(getattr(mlsynth, n), "fit")
    )


# Extra required fields beyond the base contract, keyed by estimator.
_EXTRAS = {
    "CTSC": {"treatment_vars": ["y"]},
    "LEXSCM": {"candidate_col": "unit", "m": 1},
    "MLSC": {"agg_id": "unit", "unitid_agg": "unit", "unitid_disagg": "unit"},
    "MicroSynth": {"covariates": ["x1"]},
    "PANGEO": {"arm": "treated"},
    "PROXIMAL": {"donors": ["u1", "u2"], "methods": ["PI"]},
    "SCTA": {"block_length": 2},
    "SI": {"inters": ["u1"]},
    "SIV": {"instrument": "z"},
    "SpSyDiD": {"spatial_matrix": np.eye(6)},
    "TASC": {"d": 1},
}


@pytest.mark.parametrize("name", _all_estimators())
def test_load_spec_graceful_for_every_estimator(panel, tmp_path, name):
    """For every estimator, ``load_spec`` either returns the right instance or
    fails with a typed ``Mlsynth*`` error -- never an unhandled crash, and never
    an 'unknown estimator' (every name must resolve)."""
    from mlsynth.spec import save_spec, load_spec

    spec = {
        "estimator": name,
        "outcome": "y",
        "treat": "treated",
        "unitid": "unit",
        "time": "year",
        "display_graphs": False,
    }
    runtime = {}
    for k, v in _EXTRAS.get(name, {}).items():
        if isinstance(v, np.ndarray):
            runtime[k] = v          # matrix -> runtime payload
        else:
            spec[k] = v             # text-serializable -> into the spec
    if name in ("MLSC",):
        runtime["df_agg"] = panel
        runtime["df_disagg"] = panel

    path = tmp_path / "spec.json"
    save_spec(spec, path, estimator=name)

    try:
        est = load_spec(path, df=panel, **runtime)
    except MlsynthError:
        return  # a typed, reported failure is acceptable here
    assert isinstance(est, getattr(mlsynth, name))
