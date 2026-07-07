"""PROPSC cross-validation vs the authors' R package propsdid (Spain application).

Path A + cross-validation (scenario 3, full repo). Reproduces Table 2 (common
weights column) of Bogatyrev & Stoetzer (2026), "Estimating Treatment Effects on
Proportions with Synthetic Controls," *Political Analysis* -- the electoral
effect of Spain's "Just Transition Agreement" on the full vector of party vote
shares, estimated by common-weights synthetic DID so the six party effects sum
to zero. The published common-weights estimates (p.p.) are
PSOE 1.30, PP 0.98, PODEMOS 0.30, Citizens 0.94, VOX -3.43, Others -0.09.

The reference is the R package propsdid @ 9ec3f65 (lstoetze/propsdid); populate
its cache with ``benchmarks/R/install_propsdid.sh``. R does not run in CI: the
gate compares mlsynth's ``PROPSC.fit()`` against a frozen capture of the live-R
output by default, and re-runs the R reference cell-by-cell when
``PROPSDID_LIVE=1``. mlsynth matches the package to ~1e-12.

Data: ``basedata/spain_propsc.csv`` -- the municipal panel (525 units x 5
elections, 109 treated in 2019) exported from the article's Harvard Dataverse
archive (doi:10.7910/DVN/MPUEIC), party vote shares in percentage points with
VOX coded 0 before its 2013 founding (the authors' balanced-panel convention).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings

import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DATA = os.path.join(_ROOT, "basedata", "spain_propsc.csv")
_RSCRIPT_REF = os.path.join(_ROOT, "benchmarks", "R", "propsdid_spain.R")
_CACHE = os.path.join(_ROOT, "benchmarks", "reference", ".cache", "propsdid")
_REPO = "lstoetze/propsdid"
_SHA = "9ec3f65e754af3b915dd884aaed68f7595f527d9"

_PARTIES = ["psoe", "pp", "podem", "cs", "vox", "others"]

# Frozen capture of the live-R propsdid run on basedata/spain_propsc.csv
# (propsdid @ 9ec3f65). Source of record is the R package; re-capture with
# PROPSDID_LIVE=1 after bumping the pin. Keeps CI green without R.
_FROZEN_REFERENCE = {
    "att": {
        "psoe": 1.30103974063, "pp": 0.983574626422, "podem": 0.299811273594,
        "cs": 0.936542748308, "vox": -3.43402059429, "others": -0.0869477946619,
    },
    "se": {
        "psoe": 0.684018705305, "pp": 0.81564506257, "podem": 0.35699742827,
        "cs": 0.674037334513, "vox": 0.528276733735, "others": 0.34711051491,
    },
}


def _ensure_cache() -> None:
    """Fetch the pinned propsdid source into the cache if absent.

    Tries the codeload tarball first (works in most environments); falls back to
    a ``git clone`` + checkout of the pinned SHA where codeload is proxy-gated.
    """
    if os.path.exists(os.path.join(_CACHE, "R", "synthdid.R")):
        return
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)

    # 1. codeload tarball
    url = f"https://codeload.github.com/{_REPO}/tar.gz/{_SHA}"
    tgz = os.path.join(_ROOT, "propsdid_ref.tar.gz")
    got = subprocess.run(["curl", "-sfL", url, "-o", tgz]).returncode == 0
    if got and subprocess.run(
            ["tar", "xzf", tgz, "-C", os.path.dirname(_CACHE)]).returncode == 0:
        os.rename(os.path.join(os.path.dirname(_CACHE), f"propsdid-{_SHA}"), _CACHE)
        os.remove(tgz)
        return
    if os.path.exists(tgz):
        os.remove(tgz)

    # 2. git clone + checkout the pinned commit
    tmp = os.path.join(os.path.dirname(_CACHE), "propsdid_git")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    if subprocess.run(["git", "clone", "--quiet",
                       f"https://github.com/{_REPO}.git", tmp]).returncode != 0:
        raise BenchmarkSkipped("could not fetch propsdid source (offline?)")
    if subprocess.run(["git", "-C", tmp, "checkout", "--quiet", _SHA]).returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise BenchmarkSkipped(f"could not checkout propsdid @ {_SHA[:7]}")
    shutil.rmtree(os.path.join(tmp, ".git"), ignore_errors=True)
    if os.path.exists(_CACHE):
        shutil.rmtree(_CACHE)
    os.rename(tmp, _CACHE)


def _propsdid_reference() -> dict:
    """Run propsdid via Rscript and parse its per-party ATT / SE dump."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise BenchmarkSkipped("Rscript not on PATH (run benchmarks/R/install_propsdid.sh)")
    _ensure_cache()
    out = subprocess.run([rscript, _RSCRIPT_REF, _DATA],
                         capture_output=True, text=True, timeout=1800, cwd=_ROOT)
    if out.returncode != 0:
        raise BenchmarkSkipped(f"propsdid reference failed: {out.stderr.strip()[-200:]}")
    att, se = {}, {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        tag, val = parts
        if tag.startswith("ATT_"):
            att[tag[4:]] = float(val)
        elif tag.startswith("SE_"):
            se[tag[3:]] = float(val)
    if set(att) != set(_PARTIES):
        raise BenchmarkSkipped("could not parse propsdid reference output")
    return {"att": att, "se": se}


def _reference() -> dict:
    if os.environ.get("PROPSDID_LIVE") == "1":
        return _propsdid_reference()
    return _FROZEN_REFERENCE


def _fit_mlsynth():
    from mlsynth import PROPSC
    df = pd.read_csv(_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PROPSC({
            "df": df, "outcomes": _PARTIES, "treat": "coalXpost",
            "unitid": "munid", "time": "year", "method": "sdid",
            "display_graphs": False,
        }).fit()


def run() -> dict:
    ref = _reference()
    res = _fit_mlsynth()
    att = {p: float(a) for p, a in zip(_PARTIES, res.att_vector)}
    se = {p: float(s) for p, s in zip(_PARTIES, res.se_vector)}

    att_diff = max(abs(att[p] - ref["att"][p]) for p in _PARTIES)
    se_diff = max(abs(se[p] - ref["se"][p]) for p in _PARTIES)
    out = {f"att_{p}": att[p] for p in _PARTIES}
    out["att_max_diff_vs_R"] = att_diff
    out["se_max_diff_vs_R"] = se_diff
    out["sum_constraint"] = float(res.sum_constraint)
    return out


def comparison() -> dict:
    """Side-by-side mlsynth vs live-R export (skips if R absent)."""
    ref = _propsdid_reference()
    res = _fit_mlsynth()
    rows = []
    for i, p in enumerate(_PARTIES):
        rows.append({"quantity": f"ATT[{p}]", "mlsynth": float(res.att_vector[i]),
                     "reference": ref["att"][p]})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "PROPSC",
                         "config": {"method": "sdid", "outcomes": _PARTIES}},
        "reference": {"impl": "R package propsdid (via Rscript)",
                      "version": f"propsdid @ {_SHA[:7]}"},
    }


# Cross-validation -> tolerances are numerical, not display precision. The ATT
# gates also pin Table 2 (Path A) to the published 2-d.p. values.
EXPECTED = {
    "att_psoe": (1.30103974063, 1e-4),
    "att_pp": (0.983574626422, 1e-4),
    "att_podem": (0.299811273594, 1e-4),
    "att_cs": (0.936542748308, 1e-4),
    "att_vox": (-3.43402059429, 1e-4),
    "att_others": (-0.0869477946619, 1e-4),
    "att_max_diff_vs_R": (0.0, 1e-6),
    "se_max_diff_vs_R": (0.0, 1e-5),
    "sum_constraint": (0.0, 1e-8),
}
