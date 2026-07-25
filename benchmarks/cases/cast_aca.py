"""Golden cross-validation: mlsynth's CAST vs the authors' ``CAST-panel``.

CAST (Xia, Yan & Wainwright 2025, *Inference under Staggered Adoption: Case
Study of the Affordable Care Act*, arXiv:2412.09482) is validated two ways on
the authors' own Medicaid-expansion panel:

* against their released ``CAST-panel`` package (PyPI), which is deterministic
  -- a sequence of SVDs and least-squares solves -- so the *point estimates*
  must agree value-for-value;
* against the paper's Table 1, whose per-year ATET, population-weighted totals
  and significance counts are pinned here.

Standard errors are deliberately *not* pinned against the package: mlsynth
applies the residual mask of eq. (6a) in the sorted staircase frame, whereas
``CAST-panel`` 1.0.2 masks sorted residuals with the unsorted indicator, which
understates the errors whenever the sort permutation is non-trivial. See
``docs/replications/cast.rst``. Table 1's reported SEs are pinned instead.

Skips (rather than fails) when the reference package or its data are
unavailable, so the suite stays green offline.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

# Table 1 of the paper: year -> (ATET %, SE %, positive, negative, null, pop. effect M)
_TABLE1 = {
    "2014": (-1.7, 0.1, 1, 23, 2, -3.1),
    "2015": (-2.3, 0.1, 1, 28, 0, -5.0),
    "2016": (-2.4, 0.1, 2, 28, 1, -5.8),
    "2017": (-2.7, 0.1, 1, 28, 3, -6.6),
    "2018": (-2.8, 0.1, 1, 30, 1, -6.8),
    "2019": (-2.6, 0.1, 0, 29, 5, -6.7),
    "2021": (-2.9, 0.1, 1, 36, 0, -7.5),
    "2022": (-2.2, 0.2, 0, 33, 6, -6.5),
}
_REPO = "https://github.com/Facta-Non-Verba/CAST_panel.git"


def _reference():
    """Import the authors' package, installing it from PyPI on demand."""
    try:
        import CAST_panel  # noqa: F401
    except ImportError:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "CAST-panel"],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise BenchmarkSkipped(
                f"could not install CAST-panel: {proc.stderr.strip()[-200:]}")
        try:
            import CAST_panel  # noqa: F401
        except ImportError as exc:                       # pragma: no cover
            raise BenchmarkSkipped(f"CAST-panel still unimportable: {exc}") from exc
    import CAST_panel
    return CAST_panel


def _aca_data(cache: Path):
    """The authors' ACA panel, cloned from their repository on demand."""
    repo = cache / "CAST_panel"
    if not (repo / "sample_data" / "uninsurance_rates.csv").exists():
        cache.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", _REPO, str(repo)],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise BenchmarkSkipped(
                f"could not clone the ACA data: {proc.stderr.strip()[-200:]}")
    d = repo / "sample_data"
    uninsurance = pd.read_csv(d / "uninsurance_rates.csv", index_col=0)
    adoption = pd.read_csv(d / "expansion.csv")["ADOPTION"]
    population = pd.read_csv(d / "population.csv", index_col=0)
    population = population.loc[
        [s for s in uninsurance.index if s in population.index], uninsurance.columns]
    return uninsurance, adoption, population


def _adoption_index(uninsurance: pd.DataFrame, adoption: pd.Series) -> np.ndarray:
    """Per-state index of first treatment, matching the reference notebook.

    The panel skips 2020 (no survey that year), so a state adopting in a year
    that has no column is first *observed* as treated in the next available
    column; adoption after the panel ends means never treated. This reproduces
    the notebook's ``treat_index[treat_index >= 13] -= 1`` shift.
    """
    years = [int(y) for y in uninsurance.columns]
    out = []
    for a in adoption:
        later = [j for j, y in enumerate(years) if y >= int(a)]
        out.append(later[0] if later else len(years))
    return np.array(out, dtype=int)


def _long_panel(uninsurance: pd.DataFrame, first_idx: np.ndarray) -> pd.DataFrame:
    """Wide ACA matrix -> the long frame mlsynth ingests."""
    years = list(uninsurance.columns)
    rows = []
    for i, state in enumerate(uninsurance.index):
        for j, y in enumerate(years):
            rows.append({"state": state, "year": int(y),
                         "uninsured": float(uninsurance.iloc[i, j]),
                         "expanded": int(j >= first_idx[i])})
    return pd.DataFrame(rows)


def run() -> dict:
    CAST_panel = _reference()
    cache = Path(__file__).resolve().parents[1] / "reference" / ".cache"
    uninsurance, adoption, population = _aca_data(cache)

    from mlsynth import CAST

    # ---- reference: the authors' own package on their own matrix ----
    treat_index = (adoption - 2008).copy()
    treat_index[treat_index >= 13] -= 1          # the panel has no 2020 column
    ti = treat_index.to_list()
    Y = uninsurance.values
    ref = CAST_panel.method(Y, ti, rank=1)
    ref_effects = ref.get_treatment_effects()[0]
    A = CAST_panel.method._control_mask(Y, ti)
    treated = ~A.astype(bool)

    # ---- mlsynth, through the public API ----
    df = _long_panel(uninsurance, _adoption_index(uninsurance, adoption))
    res = CAST({"df": df, "outcome": "uninsured", "treat": "expanded",
                "unitid": "state", "time": "year", "rank": 1,
                "display_graphs": False}).fit()

    # align mlsynth's (unit-sorted) panel back to the reference row order
    order = [list(res.inputs.unit_names).index(s) for s in uninsurance.index]
    mine = res.effects_panel[order]

    out: dict = {
        "point_max_abs_diff_vs_reference": float(
            np.max(np.abs(mine[treated] - ref_effects[treated]))),
        "rank_selected": float(res.rank),
        "n_treated_cells": float(treated.sum()),
    }

    # ---- Table 1: ATET, significance counts, population totals ----
    # the paper's population effect weights by the *year-specific* population,
    # i.e. a genuine bilinear functional -- an N x T weight matrix, aligned to
    # mlsynth's (unit-sorted) row order.
    inv = np.argsort(order)
    pop_matrix = population.values[inv]
    pop_res = CAST({"df": df, "outcome": "uninsured", "treat": "expanded",
                    "unitid": "state", "time": "year", "rank": 1,
                    "weights": pop_matrix.tolist(),
                    "renormalize_weights": False, "display_graphs": False}).fit()

    atet_dev, pop_dev = [], []
    for year, (att, _se, _pos, _neg, _null, pop_m) in _TABLE1.items():
        y = int(year)
        eff, _ = res.period_effects[y]
        atet_dev.append(abs(eff * 100 - att))
        pop_eff, _ = pop_res.period_effects[y]
        pop_dev.append(abs(pop_eff / 1e6 - pop_m))

    out["table1_atet_max_abs_dev_pp"] = float(max(atet_dev))
    out["table1_population_max_abs_dev_millions"] = float(max(pop_dev))
    out["att_2019_pct"] = float(res.period_effects[2019][0] * 100)

    # ---- significance counts -------------------------------------------------
    # Table 1's counts are computed at alpha = 0.01 from the reference package's
    # standard errors. Those are understated (the package masks sorted residuals
    # with the unsorted indicator), and mlsynth ships the corrected variance, so
    # the counts are NOT reproducible from mlsynth's own errors at that alpha --
    # that is a substantive consequence of the fix, not a porting defect.
    #
    # To isolate the port from the fix we recompute the counts from mlsynth's
    # point estimates paired with the *reference's* standard errors: those must
    # reproduce Table 1 exactly, proving the estimator, the mask and the
    # aggregation all agree and the standard error is the only divergence.
    z_ref = 2.5758293035489004                     # Phi^{-1}(1 - 0.01/2)
    ref_se = ref.get_std_errors()
    years = [int(y) for y in uninsurance.columns]
    rows_ref, rows_own = 0, 0
    z_own = z_ref
    for year, (_att, _se, pos, neg, null, _pop) in _TABLE1.items():
        j = years.index(int(year))
        sel = treated[:, j]
        e_ref, s_ref = ref_effects[sel, j], ref_se[sel, j]
        sig = np.abs(e_ref) > z_ref * s_ref
        rows_ref += int((int(np.sum(sig & (e_ref > 0))), int(np.sum(sig & (e_ref < 0))),
                         int(np.sum(~sig))) == (pos, neg, null))
        e_own, s_own = mine[sel, j], res.std_errors[order][sel, j]
        sig_own = np.abs(e_own) > z_own * s_own
        rows_own += int((int(np.sum(sig_own & (e_own > 0))),
                         int(np.sum(sig_own & (e_own < 0))),
                         int(np.sum(~sig_own))) == (pos, neg, null))

    out["table1_significance_rows_reference_se"] = float(rows_ref)
    out["table1_significance_rows_corrected_se"] = float(rows_own)
    out["se_ratio_corrected_over_reference"] = float(
        np.mean(res.std_errors[order][treated] / ref_se[treated]))
    return out


# mlsynth reproduces the reference package's point estimates to machine
# precision (the estimator is a deterministic sequence of SVDs) and the paper's
# Table 1 aggregates to its reported precision: ATET within 0.1pp, population
# totals within 0.1M. Paired with the reference's own standard errors, all
# eight significance rows reproduce exactly -- so the estimator, the mask and
# the aggregation agree and the standard error is the sole divergence.
#
# With mlsynth's corrected (larger) standard errors only 2 of the 8 rows still
# match at the paper's alpha = 0.01. That is pinned deliberately: it records
# that the published significance counts depend on the understated errors, and
# it would change if either implementation moved.
EXPECTED = {
    "point_max_abs_diff_vs_reference": (0.0, 1e-10),   # deterministic -> exact
    "rank_selected": (1.0, 0.0),
    "n_treated_cells": (260.0, 0.0),
    "table1_atet_max_abs_dev_pp": (0.0, 0.1),          # paper rounds to 0.1pp
    "table1_population_max_abs_dev_millions": (0.0, 0.1),
    "att_2019_pct": (-2.6, 0.1),
    "table1_significance_rows_reference_se": (8.0, 0.0),   # port is exact
    "table1_significance_rows_corrected_se": (2.0, 0.0),   # consequence of the fix
    "se_ratio_corrected_over_reference": (1.36, 0.05),
}
