"""Serialize an analysis specification to a portable text file, and load it back.

A configuration in mlsynth is plain, validated data, so the *specification* of an
analysis -- every column name and method option, everything except the
``DataFrame`` (and any other runtime arrays such as adjacency or spatial
matrices) -- can be written to a JSON or YAML file, version-controlled, and
loaded back to drive a fit. This separates the durable record of *what* analysis
was run from the data it ran on.

The contract is two functions:

``save_spec(obj, path, *, estimator=None)``
    Write the text specification of ``obj`` (an estimator instance, a
    configuration object, or a plain ``dict``) to ``path``. The DataFrame and any
    non-text-serializable fields (numpy arrays, adjacency/spatial DataFrames) are
    omitted -- they are runtime payloads, attached at load time. The format is
    chosen from the file extension (``.json``, ``.yaml``, ``.yml``).

``load_spec(path, df=None, *, estimator=None, **runtime)``
    Read a specification file and return a ready-to-fit estimator instance. The
    estimator is resolved by name from the spec (or the ``estimator=`` override);
    ``df`` and any ``runtime`` payloads (e.g. ``adjacency=...``,
    ``spatial_matrix=...``) are attached before the estimator validates them.

Example
-------
>>> from mlsynth.spec import save_spec, load_spec
>>> save_spec(VanillaSC({"df": panel, "outcome": "y", "treat": "treated",
...                       "unitid": "unit", "time": "year"}), "study.yaml")
>>> est = load_spec("study.yaml", df=panel)   # ready-to-fit estimator
>>> res = est.fit()
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Union

import mlsynth
from .config_models import BaseEstimatorConfig
from .exceptions import MlsynthConfigError

__all__ = ["save_spec", "load_spec"]

_JSON_SUFFIXES = {".json"}
_YAML_SUFFIXES = {".yaml", ".yml"}


def _is_jsonable(value: Any) -> bool:
    """True if ``value`` can be written to JSON/YAML as plain text data."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _resolve_estimator(name: str) -> type:
    """Resolve an estimator class by its public name, or raise a typed error."""
    if not isinstance(name, str):
        raise MlsynthConfigError(
            f"estimator name must be a string, got {type(name).__name__}."
        )
    cls = getattr(mlsynth, name, None)
    if cls is None or not (isinstance(cls, type) and hasattr(cls, "fit")):
        raise MlsynthConfigError(
            f"Unknown estimator {name!r}: not a fit()-bearing class exported by mlsynth."
        )
    return cls


def _config_dict(obj: BaseEstimatorConfig) -> Dict[str, Any]:
    """Plain-dict view of a configuration, keeping only fields the user set.

    ``exclude_defaults`` keeps the required fields and any non-default options,
    so the written record stays minimal and readable.
    """
    return dict(obj.model_dump(exclude_defaults=True))


def save_spec(
    obj: Any,
    path: Union[str, Path],
    *,
    estimator: Union[str, None] = None,
) -> Dict[str, Any]:
    """Write the text specification of an analysis to ``path``.

    Parameters
    ----------
    obj : estimator instance, configuration object, or dict
        The thing to serialize. An estimator instance contributes its own name;
        a configuration object or dict needs ``estimator=`` (or, for a dict, an
        ``"estimator"`` key).
    path : str or Path
        Destination file; the suffix selects JSON (``.json``) or YAML
        (``.yaml`` / ``.yml``).
    estimator : str, optional
        Estimator name, required when ``obj`` is a bare configuration object and
        optional otherwise (overrides what is inferred).

    Returns
    -------
    dict
        The specification that was written (handy for inspection and tests).
    """
    path = Path(path)

    # Resolve the estimator name and the configuration mapping.
    config = getattr(obj, "config", None)
    if isinstance(config, BaseEstimatorConfig):  # an estimator instance
        name = estimator or type(obj).__name__
        data = _config_dict(config)
    elif isinstance(obj, BaseEstimatorConfig):  # a bare config object
        if estimator is None:
            raise MlsynthConfigError(
                "save_spec needs the estimator name for a configuration object; "
                "pass estimator='VanillaSC' (or similar)."
            )
        name = estimator
        data = _config_dict(obj)
    elif isinstance(obj, dict):  # a plain dict of fields
        data = dict(obj)
        name = estimator or data.pop("estimator", None)
        data.pop("estimator", None)
        if name is None:
            raise MlsynthConfigError(
                "save_spec needs an estimator name; pass estimator=... or include "
                "an 'estimator' key in the dict."
            )
    else:
        raise MlsynthConfigError(
            "save_spec expects an estimator instance, a configuration object, or a "
            f"dict of fields, got {type(obj).__name__}."
        )

    # Keep only text-serializable fields; drop df and any runtime payloads.
    spec: Dict[str, Any] = {"estimator": name}
    dropped_runtime = []
    for key, value in data.items():
        if key == "df":
            continue
        if _is_jsonable(value):
            spec[key] = value
        else:
            dropped_runtime.append(key)

    if dropped_runtime:
        warnings.warn(
            "save_spec did not write non-serializable field(s) "
            f"{sorted(dropped_runtime)}; pass them to load_spec at load time.",
            UserWarning,
            stacklevel=2,
        )

    _write(path, spec)
    return spec


def load_spec(
    path: Union[str, Path],
    df: Any = None,
    *,
    estimator: Union[str, None] = None,
    **runtime: Any,
) -> Any:
    """Read a specification file and return a ready-to-fit estimator instance.

    Parameters
    ----------
    path : str or Path
        Specification file (``.json`` / ``.yaml`` / ``.yml``).
    df : pandas.DataFrame, optional
        The panel to attach. Required by every estimator at fit time, but left
        out of the text record.
    estimator : str, optional
        Override the estimator named in the file.
    **runtime
        Any non-text payloads the estimator needs (e.g. ``adjacency=...``,
        ``spatial_matrix=...``, ``df_agg=...``), attached before validation.

    Returns
    -------
    estimator instance
        Constructed and validated; call ``.fit()`` on it.
    """
    spec = _read(Path(path))
    if not isinstance(spec, dict):
        raise MlsynthConfigError(
            f"specification file {Path(path).name!r} must contain a mapping of fields."
        )

    spec = dict(spec)
    name = estimator or spec.pop("estimator", None)
    spec.pop("estimator", None)
    if name is None:
        raise MlsynthConfigError(
            "specification does not name an estimator; add an 'estimator' key or "
            "pass estimator=... to load_spec."
        )

    cls = _resolve_estimator(name)

    config: Dict[str, Any] = dict(spec)
    if df is not None:
        config["df"] = df
    config.update(runtime)

    return cls(config)


# --------------------------------------------------------------------------- #
# Format I/O (extension-dispatched)
# --------------------------------------------------------------------------- #
def _write(path: Path, spec: Dict[str, Any]) -> None:
    suffix = path.suffix.lower()
    if suffix in _JSON_SUFFIXES:
        path.write_text(json.dumps(spec, indent=2) + "\n")
    elif suffix in _YAML_SUFFIXES:
        yaml = _import_yaml()
        path.write_text(yaml.safe_dump(spec, sort_keys=False))
    else:
        raise MlsynthConfigError(
            f"unsupported spec extension {suffix!r}; use .json, .yaml, or .yml."
        )


def _read(path: Path) -> Any:
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix in _JSON_SUFFIXES:
        return json.loads(text)
    if suffix in _YAML_SUFFIXES:
        yaml = _import_yaml()
        return yaml.safe_load(text)
    raise MlsynthConfigError(
        f"unsupported spec extension {suffix!r}; use .json, .yaml, or .yml."
    )


def _import_yaml():
    try:
        import yaml  # PyYAML
    except ImportError as exc:  # pragma: no cover - PyYAML is a declared dependency
        raise MlsynthConfigError(
            "YAML support requires PyYAML; install mlsynth's dependencies or use a "
            ".json spec file."
        ) from exc
    return yaml
