"""Cross-validation: Lang et al. (2026), age verification laws, GSYNTH.

Lang, D., Listyg, B., Ross, B. V., Musquera, A. V., & Sanderson, Z. (2026),
*Age Verification and Public Adaptation: A Pre-Registered Synthetic Control
Multiverse*, Journal of Law and Empirical Analysis 3(1):23-50,
`10.1177/2755323X251415424 <https://doi.org/10.1177/2755323X251415424>`_.
Replication package: https://github.com/davidnathanlang/internet_regulation_synth_project

The paper estimates the effect of state age-verification laws on Google Trends
search interest for four terms -- a compliant platform, a non-compliant one, VPN
services, and pornography generally -- using ``gsynth`` 1.2.1 at
``force = 'none'``.

Why this panel
--------------

46 states x 149 weekly periods, balanced, no missing cells, 14 ever-treated
across staggered adoption dates, 32 never-treated, adoption absorbing. That is
six times the periods and three times the adoption dates of the Xu (2017)
turnout panel GSYNTH is otherwise validated on, at weekly frequency, across four
outcomes whose signal-to-noise differs by a factor of three. It exercises the
estimator where the turnout case cannot.

The reference
-------------

``gsynth`` 1.2.1, the version the paper used, and the last release before the
package became a shell over ``fect``. Installing current ``gsynth`` would
silently run a different codebase than the paper did.

The two generations agree here, which is why either could serve: at matched rank
and ``force``, gsynth 1.2.1 and fect 2.4.5 return the same numbers to fect's
print precision (pornhub -35.185 against -35.1850, xvideos 28.403 against
28.4034, vpn 11.435 against 11.4351, porn 3.018 against 3.0183). This case pins
the one the paper ran, so a disagreement is about mlsynth and not a version gap.

What is compared
----------------

Four outcomes x four ``force`` settings x ranks 0 to 5 -- 96 fits -- on three
quantities each: the overall ATT, the mean over the paper's estimand window, and
the pre-treatment mean absolute error over the balanced pre-window. All three
agree with the reference to 6.4e-14. Algorithm 1's criterion is compared over
the same 96 cells (1.3e-12) and its selected rank in all 16 outcome-by-force
combinations.

The criterion agrees far more tightly here than in ``gsynth_xu_turnout``, where
it separates by 7.5e-4. That case fits covariates, so its alternating least
squares iterates and fect's looser in-CV tolerance bites; this panel has none,
so there is nothing to iterate and the tolerance never applies.

The published table
-------------------

Table 2's four ATTs are carried as a secondary comparison at a much looser
tolerance, and the reason is a finding.

    outcome    published   at the CV rank    published MAE   mlsynth MAE
    xvideos        27.85            27.91             1.17          1.15
    porn            3.48             3.59             0.76          0.77
    vpn            13.33            13.18             2.29          2.34
    pornhub       -33.84           -33.14             0.70          0.78

Three of four land within 0.15 on a 0-100 Google Trends scale, and the
pre-treatment errors line up to 0.08 across the board, which is what confirms
the specification was identified correctly. Pornhub misses by 0.70, and the
cause is rank selection: the published figure sits between r=2 (-33.93) and r=3
(-33.48) while Algorithm 1 selects r=5 (-33.14), and every published value lies
inside its outcome's r=0..5 band. Re-running with ``se=TRUE``,
``inference="parametric"`` and ``seed=12345`` exactly as the authors scripted
still selects r=5.

Three other explanations were tested and ruled out. Data vintage: the paper's
window is byte-identical across the two committed vintages of the upstream data
that contain it. Package generation: the table above. Donor pool: the script's
five-state exclusion against none moves estimates by 0.1 to 0.5, wrong direction
and wrong size.

So ``published_att_max_diff`` is held to 0.8 -- loose enough to admit the
pornhub rank gap, tight enough that a broken estimator could not pass it, since
the four outcomes span -34 to +28. This is not Path A. Pinning the printed
values tightly would pin something neither implementation returns.

The estimand
------------

The paper's ATT is the mean over gsynth relative periods 0 to 12, which its
script computes as ``cumuEff(period = c(0, 12))$est.catt`` divided by 13. gsynth
numbers the last pre-treatment period 0 and the first treated period 1; mlsynth
numbers the first treated period 0. So the same thirteen cells are mlsynth's
horizons -1 through 11, and ``_paper_window`` below takes exactly those.

Running the R directly
----------------------

The committed gold under ``benchmarks/reference/gsynth_av_laws/`` is what the
case reads by default, so it runs in CI without R. To re-run the reference
live::

    bash benchmarks/R/install_gsynth_121.sh
    MLSYNTH_BENCH_LIVE_R=1 python benchmarks/run_benchmarks.py \\
        --case gsynth_av_laws

which regenerates the gold into a temporary directory and checks it against the
committed one. A regenerated run currently reproduces it at 0.0, and the case
raises if that moves by more than 1e-8, because a row that exists only when R
does could not be pinned in ``EXPECTED``.

Regenerate the committed gold in place with
``Rscript benchmarks/reference/gsynth_av_laws/reference.R``.
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
_REF = _HERE / "reference" / "gsynth_av_laws"

OUTCOMES = ("pornhub", "xvideos", "vpn", "porn")
FORCES = ("none", "unit", "time", "two-way")

# Table 2: ATT and the pre-treatment mean absolute error.
_PUBLISHED = {
    "pornhub": (-33.84, 0.70),
    "xvideos": (27.85, 1.17),
    "vpn": (13.33, 2.29),
    "porn": (3.48, 0.76),
}

EXPECTED = {
    "n_outcomes": (4.0, 0.5),
    "n_states": (46.0, 0.5),
    "n_periods": (149.0, 0.5),
    "n_treated": (14.0, 0.5),
    "n_control": (32.0, 0.5),
    "n_grid_fits": (96.0, 0.5),
    # mlsynth against a pinned gsynth 1.2.1 run of the authors' own call
    "live_att_max_diff": (0.0, 1e-9),
    "live_paper_att_max_diff": (0.0, 1e-9),
    "live_mae_max_diff": (0.0, 1e-9),
    "live_cv_mspe_max_diff": (0.0, 1e-8),
    "cv_rank_agreement": (16.0, 0.5),
    # mlsynth against the printed table; see the module docstring for the
    # tolerance, which admits a known rank gap on one of four outcomes
    "published_att_max_diff": (0.0, 0.8),
    "published_mae_max_diff": (0.0, 0.2),
    "signs_match": (1.0, 0.5),
}


def _panel(outcome: str) -> pd.DataFrame:
    df = pd.read_parquet(_ROOT / "basedata" / "lang_av_laws.parquet")
    d = df[df.outcome == outcome]
    return d[["state", "int_date", "hits", "post_treat"]].reset_index(drop=True)


def _fit(outcome: str, force: str, r: int):
    from mlsynth import GSYNTH

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GSYNTH({
            "df": _panel(outcome),
            "unitid": "state",
            "time": "int_date",
            "outcome": "hits",
            "treat": "post_treat",
            "r": int(r),
            "force": force,
            "inference": False,
            "display_graphs": False,
        }).fit()


def _paper_window(res) -> float:
    """The paper's estimand: gsynth periods 0..12, i.e. mlsynth horizons -1..11."""
    h = np.asarray(res.event_study.horizons)
    tau = np.asarray(res.event_study.tau)
    return float(tau[(h >= -1) & (h <= 11)].mean())


def _pre_mae(res) -> float:
    """Mean absolute pre-treatment effect of the treated-average path.

    Two conventions have to be matched, and both were found by disagreement.
    The paper's MAE column is computed on the averaged path, not per
    unit-period cell; the two differ by roughly the square root of the treated
    count, which is what made the columns look three times apart at first.

    And the average is taken over the balanced pre-window only. gsynth's
    relative-time index stops at the earliest horizon where every treated unit
    is still observed -- here 53 periods, since the first adopter has 54
    pre-periods -- while mlsynth's event study reports the full range, down to
    -145, where a single late adopter carries the average alone. Those thin
    horizons are noisier, so including them inflates the MAE by a third.
    ``n_units`` marks the balanced block: it equals the treated count exactly
    on the horizons gsynth reports.
    """
    h = np.asarray(res.event_study.horizons)
    tau = np.asarray(res.event_study.tau)
    n = np.asarray(res.event_study.n_units)
    return float(np.mean(np.abs(tau[(h < -1) & (n == n.max())])))


def _live_gold(tmp: Path):
    """Re-run the reference and return ``(grid, cv)``, or ``None``."""
    if os.environ.get("MLSYNTH_BENCH_LIVE_R") != "1":
        return None
    if shutil.which("Rscript") is None:
        return None
    dst = tmp / "gsynth_av_laws"
    dst.mkdir(parents=True, exist_ok=True)
    script = (_REF / "reference.R").read_text().replace(
        'OUT <- "benchmarks/reference/gsynth_av_laws"', f'OUT <- "{dst}"')
    sp = tmp / "reference.R"
    sp.write_text(script)
    try:
        subprocess.run(["Rscript", str(sp)], cwd=_ROOT, check=True,
                       capture_output=True, timeout=7200)
    except Exception:  # pragma: no cover - R absent or gsynth not installed
        return None
    return (pd.read_csv(dst / "gold_grid.csv"), pd.read_csv(dst / "gold_cv.csv"))


def run() -> dict:
    from mlsynth.utils.gsynth_helpers.pipeline import cross_validate_rank
    from mlsynth.utils.gsynth_helpers.setup import prepare_gsynth_inputs

    grid = pd.read_csv(_REF / "gold_grid.csv")
    gold_cv = pd.read_csv(_REF / "gold_cv.csv")

    inputs = prepare_gsynth_inputs(_panel("pornhub"), "hits", "post_treat",
                                   "state", "int_date")
    out: dict[str, float] = {
        "n_outcomes": float(len(OUTCOMES)),
        "n_states": float(inputs.N),
        "n_periods": float(inputs.T),
        "n_treated": float(inputs.n_treated),
        "n_control": float(inputs.n_control),
        "n_grid_fits": float(len(grid)),
    }

    # ---- the grid: every outcome, every force, every rank ----
    d_att, d_paper, d_mae = [], [], []
    for _, row in grid.iterrows():
        res = _fit(row.outcome, row.force, int(row.r))
        d_att.append(abs(res.att - float(row.att)))
        d_paper.append(abs(_paper_window(res) - float(row.att_paper)))
        d_mae.append(abs(_pre_mae(res) - float(row.mae_pre)))
    out["live_att_max_diff"] = float(max(d_att))
    out["live_paper_att_max_diff"] = float(max(d_paper))
    out["live_mae_max_diff"] = float(max(d_mae))

    # ---- Algorithm 1: the criterion, and what it selects ----
    cv_diff, agree = [], 0
    for (outcome, force), block in gold_cv.groupby(["outcome", "force"]):
        prep = prepare_gsynth_inputs(_panel(outcome), "hits", "post_treat",
                                     "state", "int_date")
        cv = cross_validate_rank(prep, r_max=5, force=force)
        ref = block.set_index("r").mspe
        for r, mspe in cv.mspe.items():
            if r in ref.index:
                cv_diff.append(abs(mspe - float(ref.loc[r])))
        agree += int(cv.r_selected == int(block.r_cv.iloc[0]))
    out["live_cv_mspe_max_diff"] = float(max(cv_diff))
    out["cv_rank_agreement"] = float(agree)

    # ---- the published table, at the rank Algorithm 1 selects ----
    pub_att, pub_mae, signs = [], [], []
    for outcome, (patt, pmae) in _PUBLISHED.items():
        block = gold_cv[(gold_cv.outcome == outcome) & (gold_cv.force == "none")]
        res = _fit(outcome, "none", int(block.r_cv.iloc[0]))
        att = _paper_window(res)
        pub_att.append(abs(att - patt))
        pub_mae.append(abs(_pre_mae(res) - pmae))
        signs.append(np.sign(att) == np.sign(patt))
    out["published_att_max_diff"] = float(max(pub_att))
    out["published_mae_max_diff"] = float(max(pub_mae))
    out["signs_match"] = float(all(signs))

    # Under MLSYNTH_BENCH_LIVE_R=1 the reference is re-run and checked against
    # the committed gold. This raises instead of adding a row: a row absent
    # whenever R is absent could not be pinned in EXPECTED.
    with tempfile.TemporaryDirectory() as td:
        live = _live_gold(Path(td))
    if live is not None:
        fresh_grid, fresh_cv = live
        key = ["outcome", "force", "r"]
        drift = max(
            float((fresh_grid.set_index(key).att
                   - grid.set_index(key).att).abs().max()),
            float((fresh_cv.set_index(key).mspe
                   - gold_cv.set_index(key).mspe).abs().max()),
        )
        if drift > 1e-8:
            raise RuntimeError(
                f"a live gsynth 1.2.1 run drifted from the committed gold by "
                f"{drift:.3e}; regenerate with "
                f"Rscript benchmarks/reference/gsynth_av_laws/reference.R "
                f"and re-pin the tolerances that move")

    return out
