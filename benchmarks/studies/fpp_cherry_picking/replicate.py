"""Replicate Ferman, Pinto & Possebom (2020)'s empirical findings, layer by layer.

    python replicate.py germany        # Table 3, joint test, choice rules, Figure 2 band
    python replicate.py california     # Appendix F, Table F2

Each application is checked at four layers, so a disagreement can be placed:

* ``paper``  -- the authors' R code (Synth 1.1-5), run live here, against the
  numbers printed in the paper;
* ``port``   -- this module's inference, computed from the R run's own donor
  weights, against the R run's p-values (the inference layer in isolation);
* ``inner``  -- mlsynth's lower-level QP given the R run's V, against the R
  weights (predictor construction, scaling and the inner problem);
* ``refit``  -- mlsynth's own V search (``refit.py``), against the R run's
  validation loss (the outer problem).

The reference traces are ``benchmarks/reference/fpp_cherry_picking/*_trace.json``
(``R/export_trace.R``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import fpp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
REFERENCE = ROOT / "benchmarks" / "reference" / "fpp_cherry_picking"

# Numbers printed in the paper (paper labels: "a" = no covariates).
PAPER_TABLE3 = {"1a": 0.059, "1b": 0.059, "2a": 0.059, "2b": 0.118, "3a": 0.118,
                "3b": 0.059, "4a": 0.059, "4b": 0.059, "5a": 0.118, "5b": 0.059,
                "6a": 0.588, "6b": 0.059, "7a": 0.353, "7b": 0.059}
PAPER_GERMANY_JOINT = 0.059
PAPER_GERMANY_DZ = ("1a", 0.059)       # Dube-Zipperer pick within specs 1-5
PAPER_GERMANY_DAW = ("2b", 0.118)      # Donohue-Aneja-Weber pick within specs 1-5
PAPER_TABLE_F2 = {"1a": 0.077, "1b": 0.077, "2a": 0.077, "2b": 0.077, "3a": 0.051,
                  "3b": 0.026, "4a": 0.051, "4b": 0.026, "5a": 0.077, "5b": 0.077,
                  "6a": 0.077, "6b": 0.077, "7a": 0.077, "7b": 0.026}
PAPER_TABLE_F2_RTILDE = {"1a": 0.979, "1b": 0.979, "2a": 0.969, "2b": 0.974,
                         "3a": 0.978, "3b": 0.978, "5a": 0.979, "5b": 0.979,
                         "6a": 0.525, "6b": 0.828, "7a": 0.909, "7b": 0.975}


# ---------------------------------------------------------------------------
# Panels and traces
# ---------------------------------------------------------------------------

def germany_panel() -> fpp.Panel:
    d = pd.read_stata(ROOT / "basedata" / "repgermany.dta")
    d["index"] = d["index"].astype(int)
    d["year"] = d["year"].astype(int)
    units = list(dict.fromkeys(d["index"]))        # R: co <- unique(d$index)
    return fpp.Panel(d, "index", "year", "gdp", units, np.arange(1960, 2004))


def california_panel() -> fpp.Panel:
    d = pd.read_csv(ROOT / "basedata" / "abadie_shang_smoking.csv")
    d["year"] = d["year"].astype(int)
    states = sorted(d["state"].unique())          # R: factor levels -> stateid 1..39
    d["stateid"] = d["state"].map({s: i + 1 for i, s in enumerate(states)})
    return fpp.Panel(d, "stateid", "year", "cigsale", list(range(1, 40)),
                     np.arange(1970, 2001))


def load_trace(name: str) -> dict:
    return json.loads((REFERENCE / f"{name}_trace.json").read_text())


def parse_fits(calls, units, specs, two_step: bool):
    """Group the traced synth() calls by specification and treated unit.

    Two-step (Germany): calls alternate training/main per unit; V and the
    validation loss come from the training call, W from the main call.
    """
    pos_of = {u: i for i, u in enumerate(units)}
    per = (2 if two_step else 1) * len(units)
    if len({c.get("part") for c in calls}) == len(specs):
        # One capture per specification: a unit the script skips has no calls.
        blocks = [[c for c in calls if c["part"] == k + 1] for k in range(len(specs))]
    else:
        if len(calls) != per * len(specs):
            raise ValueError(f"expected {per * len(specs)} synth calls, got {len(calls)}")
        blocks = [calls[k * per:(k + 1) * per] for k in range(len(specs))]
    out = {}
    for spec, block in zip(specs, blocks):
        pairs = zip(block[0::2], block[1::2]) if two_step else ((c, c) for c in block)
        fits = {}
        for train, main in pairs:
            if int(train["treat"]) != int(main["treat"]):
                raise ValueError("training and main calls out of step")
            pos = pos_of[int(main["treat"])]
            controls = [pos_of[int(c)] for c in main["controls"]]
            if controls != [k for k in range(len(units)) if k != pos]:
                raise ValueError("donor order differs from the panel's unit order")
            failed = main.get("failed") or not main["w"]
            fits[pos] = dict(
                failed=bool(failed),
                v=np.atleast_1d(np.asarray(train["v"], float)),  # jsonlite unboxes length 1
                w=None if failed else np.atleast_1d(np.asarray(main["w"], float)),
                loss_v=train.get("loss_v"),
                loss_w=None if failed else main.get("loss_w"),
                w_train=np.atleast_1d(np.asarray(train["w"], float)) if train["w"] else None,
            )
        out[spec] = fits
    return out


def r_matrix(obj) -> np.ndarray:
    return np.asarray(obj["values"], dtype=float)


def weights_of(fits) -> dict:
    return {pos: f["w"] for pos, f in fits.items() if not f["failed"]}


# ---------------------------------------------------------------------------
# Germany
# ---------------------------------------------------------------------------

def germany() -> dict:
    panel, designs = germany_panel(), fpp.germany_designs()
    specs = [d.name for d in designs]
    Y, times = panel.Y, panel.times
    pos = panel.units.index(fpp.GERMANY_TREATED)
    trace = load_trace("germany")
    fits = parse_fits(trace["synth_calls"], panel.units, specs, two_step=True)
    R = trace["objects"]
    r_results = r_matrix(R["results"])           # 14 x 2: pvalue, rtilde
    r_rmspe = r_matrix(R["rmspe_matrix"])         # 17 x 14
    paper = fpp.GERMANY_PAPER_LABEL

    pre = (times >= 1960) & (times <= 1989)       # gaps[1:30, ]
    post = times >= 1990                          # gaps[31:44, ]
    out = {"session": trace["session"], "specs": {}}

    # paper: the live R run against the printed Table 3.
    # port: the same p-values recomputed here from the R run's weights.
    G = {s: fpp.gaps_from_weights(Y, weights_of(fits[s])) for s in specs}
    stat_rmse = {s: fpp.ratio_statistic(G[s], pre, post, "rmse") for s in specs}
    stat_mse = {s: fpp.ratio_statistic(G[s], pre, post, "mse") for s in specs}
    r_order = [pos] + [k for k in range(len(panel.units)) if k != pos]  # R: WG first
    for i, s in enumerate(specs):
        out["specs"][s] = {
            "paper_label": paper[s],
            "paper_p": PAPER_TABLE3[paper[s]],
            "r_p": float(r_results[i, 0]),
            "port_p": fpp.placebo_pvalue(stat_rmse[s], pos, "finite"),
            "r_rtilde": float(r_results[i, 1]),
            "port_rtilde": fpp.r_tilde(G[s], Y, pre, pos),
            "port_stat_max_abs_diff": float(np.nanmax(np.abs(stat_rmse[s][r_order] - r_rmspe[:, i]))),
            "att_1990_2003": float(np.mean(G[s][post, pos])),
        }

    in_15 = [s for s in specs if s[0] in "12345"]
    out["joint"] = {
        "paper": PAPER_GERMANY_JOINT,
        # The script's tail (rowMeans of rmspe_matrix, rank, /17) on R's own
        # statistics; the per-specification runs stop before that tail.
        "r_as_coded": fpp.placebo_pvalue(np.nanmean(r_rmspe, axis=1), 0),
        "port_as_coded": fpp.joint_pvalue(stat_rmse, specs, pos),
        "port_specs_1to5_mse": fpp.joint_pvalue(stat_mse, in_15, pos),
    }

    dz = fpp.dube_zipperer(G, post, pos, specs)
    daw = fpp.donohue(G, (times >= 1981) & (times <= 1989), pos, specs)
    dz15, daw15 = min(in_15, key=dz.get), min(in_15, key=daw.get)
    out["choice"] = {
        "dz_all_ranked": sorted(dz, key=dz.get)[:3],
        "dz_1to5": [dz15, paper[dz15], out["specs"][dz15]["port_p"]],
        "paper_dz": list(PAPER_GERMANY_DZ),
        "daw_1to5": [daw15, paper[daw15], out["specs"][daw15]["port_p"]],
        "paper_daw": list(PAPER_GERMANY_DAW),
    }

    # Figure 2: cs_germany.R averages a list in which specs 2 and 3 were
    # overwritten by spec 1's weights; "as described" uses specs 1-5 as written.
    W = {s: weights_of(fits[s]) for s in specs}
    as_coded = ["1a", "1b", "1a", "1b", "1a", "1b", "4a", "4b", "5a", "5b"]
    T0 = int(np.flatnonzero(times == 1990)[0]) + 1
    bands = {}
    for label, lst in (("as_coded", as_coded), ("as_described", in_15)):
        b = fpp.fp_band(Y, W, lst, lst, pos, T0, significance=2 / 17 + 0.001)
        bands[label] = {"lb": b.lb, "ub": b.ub,
                        "average_2003": float(b.average[-1]),
                        "band_2003": [float(b.lower[-1]), float(b.upper[-1])],
                        "average": b.average.tolist(), "lower": b.lower.tolist(),
                        "upper": b.upper.tolist()}
    out["band"] = bands

    # inner: mlsynth's QP given the R run's V, on this module's predictor port.
    Y_main_ssr = Y[(times >= 1960) & (times <= 1989)]
    inner = {}
    for d in designs:
        Xs = fpp.synth_scale(fpp.predictor_matrix(panel, d.main))
        w_m, rel_loss, w_diff = {}, [], []
        for j, f in fits[d.name].items():
            if f["failed"]:
                continue
            w = fpp.custom_v_weights(Xs, f["v"], j, Y_main_ssr)
            w_m[j] = w
            lr = fpp.predictor_loss(Xs, f["v"], j, f["w"])
            lm = fpp.predictor_loss(Xs, f["v"], j, w)
            rel_loss.append((lm - lr) / max(lr, 1e-12))
            w_diff.append(float(np.max(np.abs(w - f["w"]))))
        Gm = fpp.gaps_from_weights(Y, w_m)
        inner[d.name] = {
            "max_w_diff": max(w_diff),
            "median_w_diff": float(np.median(w_diff)),
            "max_rel_loss_gap": float(max(rel_loss)),     # >0: mlsynth worse
            "min_rel_loss_gap": float(min(rel_loss)),
            "p": fpp.placebo_pvalue(fpp.ratio_statistic(Gm, pre, post, "rmse"), pos,
                                   "finite"),
        }
    out["inner"] = inner

    # Seam check: the predictor port against R's dataprep for West Germany.
    preps = trace["treated_preps"]
    seam = []
    for d, (p_train, p_main) in zip(designs, zip(preps[0::2], preps[1::2])):
        for rows, p in ((d.train, p_train), (d.main, p_main)):
            X = fpp.predictor_matrix(panel, rows)
            donors = [panel.units.index(int(c)) for c in p["controls"]]
            seam.append(max(float(np.max(np.abs(X[:, pos] - np.ravel(r_matrix(p["X1"]))))),
                            float(np.max(np.abs(X[:, donors] - r_matrix(p["X0"]))))))
    out["seam_max_abs_diff_predictors"] = max(seam)
    return out


# ---------------------------------------------------------------------------
# California
# ---------------------------------------------------------------------------

def california() -> dict:
    panel, designs = california_panel(), fpp.california_designs()
    specs = [d.name for d in designs]
    Y, times = panel.Y, panel.times
    pos = panel.units.index(3)                     # californiaid <- 3
    trace = load_trace("california")
    fits = parse_fits(trace["synth_calls"], panel.units, specs, two_step=False)
    paper = fpp.CALIFORNIA_PAPER_LABEL

    pre = times <= 1988
    post = times >= 1989
    out = {"session": trace["session"], "specs": {}}
    G = {s: fpp.gaps_from_weights(Y, weights_of(fits[s])) for s in specs}
    stat = {s: fpp.ratio_statistic(G[s], pre, post, "mse") for s in specs}
    for s in specs:
        n_failed = sum(f["failed"] for f in fits[s].values())
        out["specs"][s] = {
            "paper_label": paper[s],
            "paper_p": PAPER_TABLE_F2[paper[s]],
            "paper_rtilde": PAPER_TABLE_F2_RTILDE.get(paper[s]),
            "r_p": fpp.placebo_pvalue(stat[s], pos),
            "r_rtilde": fpp.r_tilde(G[s], Y, pre, pos) if not fits[s][pos]["failed"] else None,
            "failed_fits": int(n_failed),
            "att_1989_2000": float(np.mean(G[s][post, pos])),
        }

    # Appendix F's combined test and choice rule: specifications 1-5 with a fit
    # as good as the original (R-tilde >= 0.975 in this run), and the
    # Dube-Zipperer rule over those plus the original 7b, as in the authors'
    # choice-criteria script.
    good = [s for s in specs if paper[s][0] in "12345"
            and out["specs"][s]["r_rtilde"] is not None
            and round(out["specs"][s]["r_rtilde"], 3) >= 0.975]
    dz = fpp.dube_zipperer(G, post, pos, good + ["a1"])
    pick = min(dz, key=dz.get)
    out["joint"] = {"paper": 0.077, "specs": [paper[s] for s in good],
                    "port": fpp.joint_pvalue(stat, good, pos)}
    out["choice"] = {"paper_dz": ["4b", 0.026],
                     "dz": [pick, paper[pick], out["specs"][pick]["r_p"]]}

    # inner: mlsynth's QP given the R run's V.
    Y_ssr = Y[pre]
    inner = {}
    for d in designs:
        Xs = fpp.synth_scale(fpp.predictor_matrix(panel, d.train))
        w_m, rel_loss, w_diff = {}, [], []
        for j, f in fits[d.name].items():
            v = f["v"] if f["v"].size == Xs.shape[0] else np.ones(Xs.shape[0])
            w = fpp.custom_v_weights(Xs, v, j, Y_ssr)
            w_m[j] = w
            if not f["failed"]:
                lr = fpp.predictor_loss(Xs, v, j, f["w"])
                lm = fpp.predictor_loss(Xs, v, j, w)
                rel_loss.append((lm - lr) / max(lr, 1e-12))
                w_diff.append(float(np.max(np.abs(w - f["w"]))))
        Gm = fpp.gaps_from_weights(Y, w_m)
        inner[d.name] = {
            "max_w_diff": max(w_diff), "median_w_diff": float(np.median(w_diff)),
            "max_rel_loss_gap": float(max(rel_loss)),
            "min_rel_loss_gap": float(min(rel_loss)),
            "p": fpp.placebo_pvalue(fpp.ratio_statistic(Gm, pre, post, "mse"), pos),
            "rtilde": fpp.r_tilde(Gm, Y, pre, pos),
        }
    out["inner"] = inner
    return out


def main(argv):
    RESULTS.mkdir(exist_ok=True)
    for app in argv or ["germany", "california"]:
        res = {"germany": germany, "california": california}[app]()
        (RESULTS / f"{app}.json").write_text(json.dumps(res, indent=1))
        print(f"wrote results/{app}.json")


if __name__ == "__main__":
    main(sys.argv[1:])
