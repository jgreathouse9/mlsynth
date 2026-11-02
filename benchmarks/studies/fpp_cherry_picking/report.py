"""Tables and figures from ``results/*.json`` (run ``replicate.py`` and ``refit.py`` first).

    python report.py germany
    python report.py california
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fpp
import replicate as rep


def _load(name):
    path = rep.RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _refit_weights(refit, specs):
    return {s: {int(j): np.asarray(f["w"]) for j, f in refit[s].items()} for s in specs}


def germany():
    res, refit = _load("germany"), _load("germany_refit")
    panel, designs = rep.germany_panel(), fpp.germany_designs()
    specs = [d.name for d in designs]
    Y, times = panel.Y, panel.times
    pos = panel.units.index(fpp.GERMANY_TREATED)
    pre, post = (times <= 1989), (times >= 1990)
    trace = rep.load_trace("germany")
    fits = rep.parse_fits(trace["synth_calls"], panel.units, specs, two_step=True)
    in_15 = [s for s in specs if s[0] in "12345"]
    lines = []
    out = {}
    if refit:
        W = _refit_weights(refit, specs)
        G = {s: fpp.gaps_from_weights(Y, W[s]) for s in specs}
        st = {s: fpp.ratio_statistic(G[s], pre, post, "rmse") for s in specs}
        st_mse = {s: fpp.ratio_statistic(G[s], pre, post, "mse") for s in specs}
        lv = {s: {j: (refit[s][str(j)]["loss_v"], f["loss_v"]) for j, f in fits[s].items()}
              for s in specs}
        out["refit_p"] = {s: fpp.placebo_pvalue(st[s], pos, "finite") for s in specs}
        out["refit_joint_as_coded"] = fpp.joint_pvalue(st, specs, pos)
        out["refit_joint_1to5_mse"] = fpp.joint_pvalue(st_mse, in_15, pos)
        dz = fpp.dube_zipperer(G, post, pos, in_15)
        daw = fpp.donohue(G, (times >= 1981) & (times <= 1989), pos, in_15)
        out["refit_dz"] = min(in_15, key=dz.get)
        out["refit_daw"] = min(in_15, key=daw.get)
        T0 = int(np.flatnonzero(times == 1990)[0]) + 1
        b = fpp.fp_band(Y, W, in_15, in_15, pos, T0, significance=2 / 17 + 0.001)
        out["refit_band"] = {"lb": b.lb, "ub": b.ub, "average_2003": float(b.average[-1]),
                             "band_2003": [float(b.lower[-1]), float(b.upper[-1])]}
        out["refit_att"] = {s: float(np.mean(G[s][post, pos])) for s in specs}
        out["loss_v_ratio_treated"] = {s: lv[s][pos][0] / lv[s][pos][1] for s in specs}
        out["loss_v_better_share"] = float(np.mean(
            [a <= b * (1 + 1e-6) for s in specs for a, b in lv[s].values()]))
        out["loss_v_median_ratio"] = float(np.median([a / b for s in specs for a, b in lv[s].values()]))
        # Is Synth's training W lower-level optimal for Synth's own V? Compare its
        # predictor loss with the exact minimum for that V (mlsynth's QP).
        rows = []
        for d in fpp.germany_designs():
            Xt = fpp.synth_scale(fpp.predictor_matrix(panel, d.train))
            Yv = Y[np.isin(times, d.train_ssr)]
            for j, f in fits[d.name].items():
                w_min = fpp.custom_v_weights(Xt, f["v"], j, Yv)
                l_r = fpp.predictor_loss(Xt, f["v"], j, f["w_train"])
                l_min = fpp.predictor_loss(Xt, f["v"], j, w_min)
                rows.append((d.name, j, l_r, l_min, *lv[d.name][j]))
        slack = [r for r in rows if r[2] > r[3] * 1.01 + 1e-12]
        r_better = [r for r in rows if r[5] < r[4] * (1 - 1e-6)]
        out["lower_level"] = {
            "fits": len(rows),
            "synth_not_lower_level_optimal": len(slack),
            "synth_better_validation": len(r_better),
            "synth_better_validation_and_not_optimal": sum(
                r[2] > r[3] * 1.01 + 1e-12 for r in r_better),
        }
    header = ("| code | paper | paper p | R live p | port p | inner p | refit p "
              "| R-tilde | ATT R | ATT refit | val. MSE refit/R |")
    lines += [header, "|" + "---|" * 11]
    for s in specs:
        r = res["specs"][s]
        row = [s, r["paper_label"], f"{r['paper_p']:.3f}", f"{r['r_p']:.3f}",
               f"{r['port_p']:.3f}", f"{res['inner'][s]['p']:.3f}"]
        if refit:
            row += [f"{out['refit_p'][s]:.3f}", f"{r['r_rtilde']:.3f}",
                    f"{r['att_1990_2003']:.0f}", f"{out['refit_att'][s]:.0f}",
                    f"{out['loss_v_ratio_treated'][s]:.2f}"]
        else:
            row += ["", f"{r['r_rtilde']:.3f}", f"{r['att_1990_2003']:.0f}", "", ""]
        lines.append("| " + " | ".join(row) + " |")
    print("\n".join(lines))
    print(json.dumps({k: v for k, v in out.items() if k not in ("refit_p", "refit_att",
                                                                "loss_v_ratio_treated")}, indent=1))
    _germany_figures(res, out, Y, times, pos, fits, specs, refit)
    (rep.RESULTS / "germany_report.json").write_text(json.dumps(out, indent=1))
    (rep.RESULTS / "germany_table.md").write_text("\n".join(lines) + "\n")


def _germany_figures(res, out, Y, times, pos, fits, specs, refit):
    fig, axes = plt.subplots(1, 2 if refit else 1, figsize=(11 if refit else 6, 4),
                             sharey=True, squeeze=False)
    panels = [("Synth 1.1-5 (the authors' code)",
               {s: rep.weights_of(fits[s]) for s in specs})]
    if refit:
        panels.append(("mlsynth V search, same predictors", _refit_weights(refit, specs)))
    for ax, (title, W) in zip(axes[0], panels):
        for s in specs:
            if s[0] in "12345":
                g = fpp.gaps_from_weights(Y, {pos: W[s][pos]})[:, pos]
                ax.plot(times, g, color="0.65", lw=1.2)
        g = fpp.gaps_from_weights(Y, {pos: W["6a"][pos]})[:, pos]
        ax.plot(times, g, color="black", lw=1.8, label="6a (ADH 2015)")
        ax.axvline(1990.5, ls=":", color="0.3")
        ax.axhline(0, ls=":", color="0.3")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("year")
    axes[0][0].set_ylabel("gap in per-capita GDP")
    axes[0][0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(rep.RESULTS / "germany_effects.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in (("as_coded", "0.55"), ("as_described", "tab:blue")):
        b = res["band"][label]
        post = times > 1990
        ax.fill_between(times[post], np.asarray(b["lower"])[post],
                        np.asarray(b["upper"])[post], color=color, alpha=0.25,
                        label=f"band, {label.replace('_', ' ')}")
        ax.plot(times, b["average"], color=color, lw=1.6)
    ax.axvline(1990.5, ls=":", color="0.3")
    ax.axhline(0, ls=":", color="0.3")
    ax.set_xlabel("year")
    ax.set_ylabel("average gap, specs 1-5")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(rep.RESULTS / "germany_band.png", dpi=130)
    plt.close(fig)


def california():
    res, refit = _load("california"), _load("california_refit")
    panel = rep.california_panel()
    specs = [d.name for d in fpp.california_designs()]
    Y, times = panel.Y, panel.times
    pos = panel.units.index(3)
    pre, post = times <= 1988, times >= 1989
    out = {}
    if refit:
        G = {s: np.column_stack([np.asarray(refit["gaps"][s][str(u)], float)
                                 for u in panel.units]) for s in specs}
        st = {s: fpp.ratio_statistic(G[s], pre, post, "mse") for s in specs}
        out["refit_p"] = {s: fpp.placebo_pvalue(st[s], pos) for s in specs}
        out["refit_rtilde"] = {s: fpp.r_tilde(G[s], Y, pre, pos) for s in specs}
        out["refit_att"] = {s: float(np.mean(G[s][post, pos])) for s in specs}
        out["refit_failures"] = len(refit["failures"])
    lines = ["| code | paper | paper p | R live p | inner p | refit p | paper R-tilde "
             "| R-tilde R | R-tilde refit | ATT R | ATT refit | failed R fits |",
             "|" + "---|" * 12]
    for s in specs:
        r = res["specs"][s]
        pr = r["paper_rtilde"]
        row = [s, r["paper_label"], f"{r['paper_p']:.3f}", f"{r['r_p']:.3f}",
               f"{res['inner'][s]['p']:.3f}",
               f"{out['refit_p'][s]:.3f}" if refit else "",
               f"{pr:.3f}" if pr is not None else "",
               f"{r['r_rtilde']:.3f}" if r["r_rtilde"] is not None else "fail",
               f"{out['refit_rtilde'][s]:.3f}" if refit else "",
               f"{r['att_1989_2000']:.1f}", f"{out['refit_att'][s]:.1f}" if refit else "",
               str(r["failed_fits"])]
        lines.append("| " + " | ".join(row) + " |")
    print("\n".join(lines))
    (rep.RESULTS / "california_report.json").write_text(json.dumps(out, indent=1))
    (rep.RESULTS / "california_table.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    {"germany": germany, "california": california}[sys.argv[1]]()
