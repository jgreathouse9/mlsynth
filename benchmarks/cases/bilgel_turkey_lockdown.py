"""Path A + cross-validation: Bilgel (2022), Covid-19 lockdowns in Turkey, PPSCM.

Bilgel, F. (2022), *Effects of Covid-19 lockdowns on social distancing in
Turkey*, Econometrics Journal 25(3):781-805,
`10.1093/ectj/utac016 <https://doi.org/10.1093/ectj/utac016>`_, Table 3 column 1,
from the author's own replication package.

The specification is one line per outcome in his
``augsynth_replication_final.R``::

    set.seed(1234); multisynth(retail~lockdown, id, eday, nu=0.5,
                               data=augsynth_retail_t0c1)

which is what :class:`~mlsynth.PPSCM` is a port of, so nothing needed
reimplementing: the case reads the author's frames and fits them.

Two reference bases
-------------------

The rows come in two families, against two different references, and keeping
them apart is the point.

``live_*`` rows compare mlsynth to a pinned augsynth 0.2.0 run of the author's
own call. ``published_*`` rows compare it to the printed table. The first asks
whether mlsynth implements this specification; the second asks whether the
published table was produced by it. A third row,
``published_vs_live_att_max_diff``, compares the two references to each other and
involves no mlsynth code at all.

Here the third row settles cleanly. Live augsynth reproduces every printed figure
to within rounding, so unlike the Song et al. case there is no drift between the
published artifact and the pinned package: the table came from this call at this
version. That makes the live rows the binding ones and the published rows a
statement about the paper.

    outcome       published    live augsynth      mlsynth
    retail          -25.08     -25.0818493     -25.0818010
    grocery         -53.10     -53.1003895     -53.1004403
    park            -33.45     -33.4503451     -33.4503250
    transit         -16.76     -16.7622384     -16.7622501
    workplace       -27.61     -27.6115815     -27.6114318
    residential      12.02      12.0190048      12.0189919

Tolerances
----------

Against the live reference the ATTs agree to 1.5e-4 and the post-period
trajectories -- 17 event times per outcome, 102 points -- to 4.1e-4. Both sit
above the 1e-6 this project reaches on ridge ASCM, and the gap is the QP: PPSCM
and augsynth reach the partially pooled optimum through different solvers at
different stopping tolerances. The rows admit 5e-4 and 1e-3, loose enough to
survive a solver and far tighter than anything that would move a reported
estimate.

Against the published table the tolerance is 0.005, half the last printed digit.
The paper prints two decimals, so that is the tightest claim its precision
supports, and a closer one would need the author's unrounded output.

Standard errors are held to 10 percent relative, and that is not
tolerance-shopping. Column 1's errors are a wild bootstrap; mlsynth draws its
multipliers from NumPy and augsynth from R, so reproducing them to the digit
would mean reproducing an RNG stream and not an estimator. What is checkable is
that they land in the same place, and they do -- 0.3 to 3.3 percent apart. A
broken inference routine would miss by far more.

Structural rows
---------------

``n_treated_residential`` is 24 against 31 everywhere else, which is the paper's
own footnote a ("Only residential mobility covers 24 provinces"), and
``n_post_periods`` is the 17 days Table 3 reports, recomputed from the panel.
Both would move if the data were misread.

Running the R directly
----------------------

The committed gold under ``benchmarks/reference/bilgel_turkey_lockdown/`` is what
the case reads by default, so it runs in CI without R. To re-run the reference
live::

    bash benchmarks/R/install_augsynth.sh          # pinned augsynth 0.2.0
    MLSYNTH_BENCH_LIVE_R=1 python benchmarks/run_benchmarks.py \\
        --case bilgel_turkey_lockdown

which regenerates the gold into a temporary directory and checks it against the
committed one -- reference against reference across a regeneration, catching
drift in the pinned reference itself. A regenerated run currently reproduces the
committed gold at 0.0, and the case raises if that moves by more than 1e-8. It
raises instead of reporting a row because a row that only exists when R does
could not be pinned in ``EXPECTED``, and an unpinned row is one nobody checks.

Regenerate the committed gold in place with
``Rscript benchmarks/reference/bilgel_turkey_lockdown/reference.R``.

The vendored data
-----------------

``basedata/bilgel_turkey_lockdown.parquet`` is the package's six
``augsynth_dataset_<outcome>_t0c1.rdata`` frames concatenated, with the outcome
column renamed to ``mobility`` and tagged by ``outcome``. Both sides of the
comparison read that one file -- the R script through nanoparquet -- so the two
cannot silently run on different inputs.

Selecting the frames needs care. The ``.rdata`` files are cumulative workspace
saves: the residential file carries all six, the transit file four. Loading one
and taking the first object returns a different outcome's panel, and the fit
succeeds on it. Each was selected by name.

Treatment reversal is handled upstream by the author. Sixteen provinces lift
lockdown before the window ends, and Table 3 records the action as
"post-lockdown concatenation", so the ``_t0c1`` frames reach the estimator in
absorbing form with a single adoption day. The estimator never sees the
on-and-off treatment Table 3 marks ASCM as unable to accommodate; the paper's
interactive-fixed-effects and matrix-completion columns cover that case.
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
_REF = _HERE / "reference" / "bilgel_turkey_lockdown"

OUTCOMES = ("retail", "grocery", "park", "transit", "workplace", "residential")

# Table 3, column 1: ATT (SE).
_PUBLISHED = {
    "retail": (-25.08, 5.49),
    "grocery": (-53.10, 10.43),
    "park": (-33.45, 7.69),
    "transit": (-16.76, 3.90),
    "workplace": (-27.61, 6.37),
    "residential": (12.02, 2.04),
}

_PUB_TOL = 0.005          # half the paper's last printed digit

EXPECTED = {
    "n_outcomes": (6.0, 0.5),
    "n_gold_rows": (6.0, 0.5),
    "n_treated_retail": (31.0, 0.5),
    "n_treated_residential": (24.0, 0.5),
    "n_post_periods": (17.0, 0.5),
    # mlsynth against a pinned augsynth 0.2.0 run of the author's call
    "live_att_max_diff": (0.0, 5e-4),
    "live_traj_max_diff": (0.0, 1e-3),
    "live_traj_points": (102.0, 0.5),
    "live_se_max_rel_err": (0.0, 0.10),
    "n_se_reported": (6.0, 0.5),
    # mlsynth against the printed table
    "published_att_max_diff": (0.0, _PUB_TOL),
    # reference against reference: no mlsynth code involved
    "published_vs_live_att_max_diff": (0.0, _PUB_TOL),
    "signs_match": (1.0, 0.5),
}


def _panel(outcome: str) -> pd.DataFrame:
    df = pd.read_parquet(_ROOT / "basedata" / "bilgel_turkey_lockdown.parquet")
    d = df[df.outcome == outcome].copy()
    return d.sort_values(["id", "eday"]).reset_index(drop=True)


def _fit(outcome: str):
    from mlsynth import PPSCM

    d = _panel(outcome)
    cfg = {
        "df": d,
        "unitid": "id",                     # the author keys on the numeric code
        "time": "eday",
        "outcome": "mobility",
        "treat": "lockdown",
        "nu": 0.5,                          # the paper's partial pooling
        "inference_method": "bootstrap",    # column 1's wild bootstrap
        "seed": 1234,
        "display_graphs": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PPSCM(cfg).fit(), d


def _live_gold(tmp: Path):
    """Re-run the reference and return its ``(scalars, trajectory)``, or ``None``.

    Only attempted under ``MLSYNTH_BENCH_LIVE_R=1``; the committed gold is what
    the case reads otherwise, so it runs without R.
    """
    if os.environ.get("MLSYNTH_BENCH_LIVE_R") != "1":
        return None
    if shutil.which("Rscript") is None:
        return None
    dst = tmp / "bilgel_turkey_lockdown"
    dst.mkdir(parents=True, exist_ok=True)
    script = (_REF / "reference.R").read_text().replace(
        'OUT <- "benchmarks/reference/bilgel_turkey_lockdown"',
        f'OUT <- "{dst}"')
    sp = tmp / "reference.R"
    sp.write_text(script)
    try:
        subprocess.run(["Rscript", str(sp)], cwd=_ROOT, check=True,
                       capture_output=True, timeout=3600)
    except Exception:  # pragma: no cover - R absent or augsynth not installed
        return None
    return (pd.read_csv(dst / "gold_live_augsynth.csv"),
            pd.read_csv(dst / "gold_live_trajectory.csv"))


def run() -> dict:
    gold = pd.read_csv(_REF / "gold_live_augsynth.csv")
    traj = pd.read_csv(_REF / "gold_live_trajectory.csv")

    out: dict[str, float] = {"n_outcomes": float(len(OUTCOMES)),
                             "n_gold_rows": float(len(gold))}

    live_att, live_traj, live_se = [], [], []
    pub_att, pub_live, signs = [], [], []
    n_traj = 0

    for name in OUTCOMES:
        res, d = _fit(name)
        g = gold[gold.outcome == name].iloc[0]
        pub, _pub_se = _PUBLISHED[name]

        att = float(res.effects.att)
        live_att.append(abs(att - float(g.att)))
        pub_att.append(abs(att - pub))
        pub_live.append(abs(float(g.att) - pub))
        signs.append(np.sign(att) == np.sign(pub))

        se = getattr(res.effects, "att_std_err", None)
        if se is None:
            se = getattr(getattr(res, "inference", None), "standard_error", None)
        if se is not None and np.isfinite(float(se)):
            live_se.append(abs(float(se) - float(g.se)) / abs(float(g.se)))

        es = res.event_study
        ml = pd.DataFrame({"time": np.asarray(es.horizons, dtype=int),
                           "ml": np.asarray(es.tau, dtype=float)})
        ref = traj[traj.outcome == name][["time", "estimate"]].astype({"time": int})
        merged = ref.merge(ml, on="time", how="inner")
        n_traj += len(merged)
        if len(merged):
            live_traj.append(float((merged.estimate - merged.ml).abs().max()))

        if name == "retail":
            out["n_treated_retail"] = float(d.groupby("id").lockdown.max().sum())
            adopt = int(d[d.lockdown == 1].eday.min())
            out["n_post_periods"] = float(int(d.eday.max()) - adopt + 1)
        if name == "residential":
            out["n_treated_residential"] = float(
                d.groupby("id").lockdown.max().sum())

    out["live_att_max_diff"] = float(max(live_att))
    out["live_traj_max_diff"] = float(max(live_traj)) if live_traj else float("nan")
    out["live_traj_points"] = float(n_traj)
    # Counted, so an inference routine that stops reporting a standard error
    # fails this case instead of passing it with an empty maximum.
    out["n_se_reported"] = float(len(live_se))
    out["live_se_max_rel_err"] = float(max(live_se)) if live_se else float("nan")
    out["published_att_max_diff"] = float(max(pub_att))
    out["published_vs_live_att_max_diff"] = float(max(pub_live))
    out["signs_match"] = float(all(signs))

    # Under MLSYNTH_BENCH_LIVE_R=1 the reference is re-run and checked against
    # the committed gold. This raises instead of adding a row: a row absent
    # whenever R is absent could not be pinned in EXPECTED, and an unpinned row
    # is one nobody checks.
    with tempfile.TemporaryDirectory() as td:
        live = _live_gold(Path(td))
    if live is not None:
        fresh = live[0].set_index("outcome").att
        committed = gold.set_index("outcome").att
        drift = float((fresh - committed).abs().max())
        if drift > 1e-8:
            raise RuntimeError(
                f"a live augsynth run drifted from the committed gold by "
                f"{drift:.3e}; regenerate with "
                f"Rscript benchmarks/reference/bilgel_turkey_lockdown/reference.R "
                f"and re-pin the tolerances that move")

    return out
