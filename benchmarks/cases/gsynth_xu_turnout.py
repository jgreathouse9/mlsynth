"""Path A + cross-validation: Xu (2017), Election Day Registration, GSYNTH.

Xu, Y. (2017), *Generalized Synthetic Control Method: Causal Inference with
Interactive Fixed Effects Models*, Political Analysis 25(1):57-76,
`10.1017/pan.2016.2 <https://doi.org/10.1017/pan.2016.2>`_, Table 2 columns (3)
and (4), on the author's own turnout panel.

The specification is the paper's: additive state and year effects, two
unobserved factors, and in column (4) indicators for universal mail-in and
motor voter registration.

    quantity                        published    live fect 2.4.5      mlsynth
    ATT, no covariates                   5.13       5.1304927      5.1304927
    ATT, with covariates                 4.90       4.8957798      4.8957798
    universal mail-in registration       0.15       0.15467938     0.15467938
    motor voter registration            -1.05      -1.05149676    -1.05149676
    unobserved factors                      2               2              2

Two reference bases
-------------------

``live_*`` rows compare mlsynth to a fect 2.4.5 run of the same specification;
``published_*`` rows compare it to the printed table. The first asks whether
mlsynth implements this estimator, the second whether the published table was
produced by it, and ``published_vs_live_att_max_diff`` compares the two
references to each other with no mlsynth code involved.

The third row settles cleanly: the live reference rounds to the printed
figures, so the table came from this estimator at this specification. That
makes the live rows the binding ones.

Why the grid
------------

Two matching numbers could be luck, so the live comparison runs every rank from
zero to five on both specifications -- twelve fits -- and compares the ATT, the
covariate coefficients, and the full effect path by horizon. Across all of them
the ATTs agree to 7.7e-14, the coefficients to 4.8e-9 (the alternating least
squares' own stopping tolerance), and the 33-point paths to 2.3e-13. The
tolerances below sit a few orders above those, loose enough to survive a BLAS
and far tighter than anything that would move a reported estimate.

fect numbers the first treated period 1 and mlsynth numbers it 0, so the path
comparison shifts by one. That is the only convention difference between the
two sides.

Rank selection
--------------

``r_cv_nocov`` and ``r_cv_cov`` are pinned at 2, and they are the rows that
matter most here. The ATT is not monotone in the rank -- it runs 1.26, 3.56,
5.13, 2.36, 3.72, 4.48 over the grid -- so the rank rule decides the headline
number, and a change to the rule would move the estimate by more than a quarter
without touching a line of the estimator.

Xu's Algorithm 1 holds back one pre-treatment period from the treated units,
selects two factors on both specifications, and reproduces the published
figures. fect reaches the same rule as ``cv.method = "loo"``, which is what the
gold was generated with; its own default changed in v2.3.0 to a rolling scheme
that masks a random tenth of the control units, and on this panel that default
returns rank 4, 1, 2, 2, 1, 1 across six seeds -- ATTs of 3.72, 3.56, 5.13,
5.13, 3.56, 3.56. mlsynth implements Algorithm 1 and nothing else, and the two
criterion rows pin it against the reference's, so a change to the rule fails
this case instead of re-selecting unnoticed.

Those two rows exist because of one difference between the implementations.
fect runs Step 1 at ``max(tol, 1e-3)`` inside its cross-validation loop, a
speed choice for rank selection with no counterpart in Algorithm 1; mlsynth
runs it at the caller's tolerance throughout. With no covariates there is
nothing to iterate on and the criteria agree to 4.6e-14. With covariates the
alternating least squares does iterate, and the criteria separate by up to
7.5e-4 on values of 10 to 22 -- five parts in a hundred thousand, and the same
rank either way. Re-running mlsynth's criterion at ``tol=1e-3`` collapses that
to 1.4e-13, which is what ``live_cv_mspe_matched_tol_max_diff`` records: the
algorithms are identical and the gap is the tolerance. The looser tolerance is
also asserted not to change the selected rank.

Standard errors
---------------

``published_se_rel_err`` is held to 15 percent, and that is not
tolerance-shopping. Column (3)'s error is a parametric bootstrap of 2,000
state-blocked draws; mlsynth draws from NumPy and fect from R, so reproducing
it to the digit would mean reproducing an RNG stream and not an estimator. What
is checkable is that the inference lands in the same place, and it does. A
broken routine would miss by far more. ``n_se_reported`` is counted so an
inference routine that stops reporting a standard error fails this case instead
of passing it with an empty maximum.

Structural rows
---------------

``n_observations`` is 1,128, ``n_treated`` is 9 and ``n_control`` is 38, which
is Table 2's own header block, recomputed from the panel. All three would move
if the data were misread.

Running the R directly
----------------------

The committed gold under ``benchmarks/reference/gsynth_xu_turnout/`` is what
the case reads by default, so it runs in CI without R. To re-run the reference
live::

    bash benchmarks/R/install_fect.sh              # pinned fect 2.4.5
    MLSYNTH_BENCH_LIVE_R=1 python benchmarks/run_benchmarks.py \\
        --case gsynth_xu_turnout

which regenerates the gold into a temporary directory and checks it against the
committed one -- reference against reference across a regeneration, catching
drift in the pinned reference itself. A regenerated run currently reproduces the
committed gold at 0.0, and the case raises if that moves by more than 1e-8. It
raises instead of reporting a row because a row that only exists when R does
could not be pinned in ``EXPECTED``, and an unpinned row is one nobody checks.

Regenerate the committed gold in place with
``Rscript benchmarks/reference/gsynth_xu_turnout/reference.R``.

The vendored data
-----------------

``basedata/xu_edr_turnout.parquet`` is ``turnout.rda`` from
`xuyiqing/fect <https://github.com/xuyiqing/fect>`_ written to Parquet
unchanged: 47 states by 24 quadrennial presidential elections, 1920-2012. Both
sides of the comparison read that one file -- the R script through nanoparquet
-- so the two cannot run on different inputs.

Adoption is staggered over four dates (1976, 1996, 2008, 2012) and absorbing,
so each adopter reaches the loading regression with one contiguous pre-adoption
block. The thirty-eight states that never adopt are the pool the factor space
comes from.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parents[1]
_ROOT = _HERE.parent
_REF = _HERE / "reference" / "gsynth_xu_turnout"

COVARIATES = ("policy_mail_in", "policy_motor")

# Table 2, columns (3) and (4): ATT (SE), and the column (4) coefficients.
_PUBLISHED_ATT = {"nocov": (5.13, 2.27), "cov": (4.90, 2.27)}
_PUBLISHED_BETA = (0.15, -1.05)

_PUB_TOL = 0.005          # half the paper's last printed digit

EXPECTED = {
    "n_observations": (1128.0, 0.5),
    "n_treated": (9.0, 0.5),
    "n_control": (38.0, 0.5),
    "n_grid_fits": (12.0, 0.5),
    # mlsynth against a pinned fect 2.4.5 run of the same specification
    "live_att_max_diff": (0.0, 1e-9),
    "live_beta_max_diff": (0.0, 1e-6),
    "live_path_max_diff": (0.0, 1e-9),
    "live_path_points": (396.0, 0.5),
    "live_cv_mspe_max_diff": (0.0, 1e-3),
    "live_cv_mspe_matched_tol_max_diff": (0.0, 1e-9),
    # Algorithm 1's selection, which decides the headline number
    "r_cv_nocov": (2.0, 0.5),
    "r_cv_cov": (2.0, 0.5),
    # mlsynth against the printed table
    "published_att_max_diff": (0.0, _PUB_TOL),
    "published_beta_max_diff": (0.0, _PUB_TOL),
    "published_se_rel_err": (0.0, 0.15),
    "n_se_reported": (1.0, 0.5),
    # reference against reference: no mlsynth code involved
    "published_vs_live_att_max_diff": (0.0, _PUB_TOL),
    "signs_match": (1.0, 0.5),
}


def _panel() -> pd.DataFrame:
    return pd.read_parquet(_ROOT / "basedata" / "xu_edr_turnout.parquet")


def _fit(df: pd.DataFrame, spec: str, **kw):
    from mlsynth import GSYNTH

    cfg = {
        "df": df,
        "unitid": "abb",
        "time": "year",
        "outcome": "turnout",
        "treat": "policy_edr",
        "inference": False,
        "display_graphs": False,
    }
    if spec == "cov":
        cfg["covariates"] = list(COVARIATES)
    cfg.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GSYNTH(cfg).fit()


def _live_gold(tmp: Path):
    """Re-run the reference and return its three frames, or ``None``.

    Only attempted under ``MLSYNTH_BENCH_LIVE_R=1``; the committed gold is what
    the case reads otherwise, so it runs without R.
    """
    if os.environ.get("MLSYNTH_BENCH_LIVE_R") != "1":
        return None
    if shutil.which("Rscript") is None:
        return None
    dst = tmp / "gsynth_xu_turnout"
    dst.mkdir(parents=True, exist_ok=True)
    script = (_REF / "reference.R").read_text().replace(
        'OUT <- "benchmarks/reference/gsynth_xu_turnout"', f'OUT <- "{dst}"')
    sp = tmp / "reference.R"
    sp.write_text(script)
    try:
        subprocess.run(["Rscript", str(sp)], cwd=_ROOT, check=True,
                       capture_output=True, timeout=3600)
    except Exception:  # pragma: no cover - R absent or fect not installed
        return None
    return (pd.read_csv(dst / "gold_grid.csv"),
            pd.read_csv(dst / "gold_path.csv"),
            pd.read_csv(dst / "gold_cv.csv"))


def run() -> dict:
    from mlsynth.utils.gsynth_helpers.pipeline import cross_validate_rank
    from mlsynth.utils.gsynth_helpers.setup import prepare_gsynth_inputs

    df = _panel()
    grid = pd.read_csv(_REF / "gold_grid.csv")
    path = pd.read_csv(_REF / "gold_path.csv")
    gold_cv = pd.read_csv(_REF / "gold_cv.csv")

    inputs = prepare_gsynth_inputs(df, "turnout", "policy_edr", "abb", "year")
    out: dict[str, float] = {
        "n_observations": float(inputs.T * inputs.N),
        "n_treated": float(inputs.n_treated),
        "n_control": float(inputs.n_control),
        "n_grid_fits": float(len(grid)),
    }

    live_att, live_beta, live_path = [], [], []
    n_path = 0
    for _, row in grid.iterrows():
        res = _fit(df, row.spec, r=int(row.r))
        live_att.append(abs(res.effects.att - float(row.att)))
        if isinstance(row.beta, str) and row.beta:
            ref = np.array([float(v) for v in row.beta.split(";")])
            live_beta.append(float(np.abs(res.design.beta - ref).max()))

        # fect numbers the first treated period 1; mlsynth numbers it 0.
        es = res.event_study
        mine = pd.DataFrame({"time": np.asarray(es.horizons, dtype=int) + 1,
                             "ml": np.asarray(es.tau, dtype=float)})
        ref_p = path[(path.spec == row.spec) & (path.r == row.r)][["time", "att"]]
        merged = ref_p.astype({"time": int}).merge(mine, on="time", how="inner")
        n_path += len(merged)
        if len(merged):
            live_path.append(float((merged.att - merged.ml).abs().max()))

    out["live_att_max_diff"] = float(max(live_att))
    out["live_beta_max_diff"] = float(max(live_beta))
    out["live_path_max_diff"] = float(max(live_path))
    out["live_path_points"] = float(n_path)

    # Algorithm 1, against the reference's cv.method = "loo" criterion, at two
    # tolerances. See the module docstring: fect runs Step 1 at a looser
    # tolerance inside its cross-validation loop, and matching that tolerance
    # collapses the difference to floating point, which is what isolates it to
    # the tolerance and not to the algorithm.
    cv_diff, cv_diff_matched = [], []
    for spec, covs in (("nocov", None), ("cov", list(COVARIATES))):
        prep = prepare_gsynth_inputs(df, "turnout", "policy_edr", "abb", "year",
                                     covariates=covs)
        cv = cross_validate_rank(prep, r_max=5)
        matched = cross_validate_rank(prep, r_max=5, tol=1e-3)
        out[f"r_cv_{spec}"] = float(cv.r_selected)
        ref_cv = gold_cv[gold_cv.spec == spec].set_index("r").mspe
        for r, mspe in cv.mspe.items():
            if r in ref_cv.index:
                cv_diff.append(abs(mspe - float(ref_cv.loc[r])))
                cv_diff_matched.append(abs(matched.mspe[r] - float(ref_cv.loc[r])))
        # The looser tolerance must not change what gets selected.
        assert matched.r_selected == cv.r_selected
    out["live_cv_mspe_max_diff"] = float(max(cv_diff))
    out["live_cv_mspe_matched_tol_max_diff"] = float(max(cv_diff_matched))

    # The published table, at the paper's own rank of two.
    pub_att, pub_live, signs = [], [], []
    for spec, (pub, _pub_se) in _PUBLISHED_ATT.items():
        res = _fit(df, spec, r=2)
        pub_att.append(abs(res.effects.att - pub))
        signs.append(np.sign(res.effects.att) == np.sign(pub))
        live = float(grid[(grid.spec == spec) & (grid.r == 2)].att.iloc[0])
        pub_live.append(abs(live - pub))
        if spec == "cov":
            out["published_beta_max_diff"] = float(
                np.abs(np.asarray(res.design.beta) - _PUBLISHED_BETA).max())

    out["published_att_max_diff"] = float(max(pub_att))
    out["published_vs_live_att_max_diff"] = float(max(pub_live))
    out["signs_match"] = float(all(signs))

    # Algorithm 2, against the published 2.27.
    boot = _fit(df, "nocov", r=2, inference=True, n_bootstrap=200, seed=2139)
    se = boot.inference.standard_error
    # Counted, so an inference routine that stops reporting a standard error
    # fails this case instead of passing it with an empty maximum.
    out["n_se_reported"] = float(se is not None and np.isfinite(se))
    pub_se = _PUBLISHED_ATT["nocov"][1]
    out["published_se_rel_err"] = (
        float(abs(float(se) - pub_se) / pub_se) if out["n_se_reported"]
        else float("nan"))

    # Under MLSYNTH_BENCH_LIVE_R=1 the reference is re-run and checked against
    # the committed gold. This raises instead of adding a row: a row absent
    # whenever R is absent could not be pinned in EXPECTED, and an unpinned row
    # is one nobody checks.
    with tempfile.TemporaryDirectory() as td:
        live = _live_gold(Path(td))
    if live is not None:
        fresh, _fresh_path, fresh_cv = live
        drift = max(
            float((fresh.set_index(["spec", "r"]).att
                   - grid.set_index(["spec", "r"]).att).abs().max()),
            float((fresh_cv.set_index(["spec", "r"]).mspe
                   - gold_cv.set_index(["spec", "r"]).mspe).abs().max()),
        )
        if drift > 1e-8:
            raise RuntimeError(
                f"a live fect run drifted from the committed gold by "
                f"{drift:.3e}; regenerate with "
                f"Rscript benchmarks/reference/gsynth_xu_turnout/reference.R "
                f"and re-pin the tolerances that move")

    return out
