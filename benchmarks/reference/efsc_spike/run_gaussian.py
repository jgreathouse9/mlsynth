"""Gaussian benchmarks: the paper's Table 1 (tilt) and Table 2 (structured).

Usage: python run_gaussian.py {table1|table2} REPS OUT.csv

Methods, all evaluated on the treated post-treatment block:

  pmf            EFSC, paper settings (C.5: N(0,1) prior, lr 0.1, 3000 epochs)
  pmf_default    EFSC, package defaults (N(0,2^2) prior, 2000 epochs) [table1]
  sc_means       simplex SC on the cell-mean matrix (paper's row 1)
  mle_sc         simplex SC on cell-MLE natural parameters ("EFSC-MLE")
  mle_mcnnm      mlsynth.MCNNM on cell-MLE natural parameters
  mom_mcnnm      mlsynth.MCNNM on cell means and log unbiased variances, the
                 baseline a practitioner would run; treated side from the same
                 unbiased moments
  dsc            mlsynth.DSC on the microdata (counterfactual quantile curves)
                 [table2]
  pmfcf_mletr    EFSC counterfactual paired with the raw cell-MLE treated side:
                 isolates how much of EFSC's gain is treated-side smoothing
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dgp  # noqa: E402
import fastvi  # noqa: E402
import methods as M  # noqa: E402

fastvi.install()
torch.set_num_threads(2)


def moments_eta(truth):
    mean, var0 = M.cell_stats(truth)
    m = truth["m"]
    var = np.where(m > 1, var0 * m / np.maximum(m - 1, 1), np.nan)
    var = np.where(np.isfinite(var), var, np.nanmedian(var))
    return mean, np.maximum(var, 1e-6)


def mom_mcnnm(truth):
    mean, var = moments_eta(truth)
    mu_c = M.mcnnm_complete(mean, truth)
    lv_c = M.mcnnm_complete(np.log(var), truth)
    s2c = np.exp(lv_c)
    cf = np.stack([mu_c / s2c, -1 / (2 * s2c)], axis=-1)
    tr = np.stack([mean / var, -1 / (2 * var)], axis=-1)
    return cf, tr


def table1(rep, tau):
    panel, truth = dgp.gaussian_panel(intervention="tilt", tau=tau, lam=55, seed=1000 + rep)
    tp = truth["treated_post"]
    s2_true = (-1 / (2 * truth["eta_ctrl"][..., 1]))
    mean, _ = M.cell_stats(truth)
    mle = M.cell_mle(truth, "gaussian")
    fits = {}
    fits["pmf"] = M.efsc_fits(panel, truth, rank=2, config=M.PAPER, seed=rep)
    fits["pmf_default"] = M.efsc_fits(panel, truth, rank=2, config=M.PACKAGE_DEFAULT, seed=rep)
    fits["mle_sc"] = (M.gaussian_component_complete(mle, truth, M.sc_complete), mle)
    fits["mle_mcnnm"] = (M.gaussian_component_complete(mle, truth, M.mcnnm_complete), mle)
    fits["mom_mcnnm"] = mom_mcnnm(truth)
    fits["pmfcf_mletr"] = (fits["pmf"][0], mle)
    rows = []
    for name, (cf, tr) in fits.items():
        mu_c = cf[..., 0] / (-2 * cf[..., 1])
        rows.append(dict(method=name, metric="outcome_mae",
                         value=np.abs((mean - mu_c) / s2_true - tau)[tp].mean()))
        rows.append(dict(method=name, metric="eta1_effect_mae",
                         value=np.abs(tr[..., 0] - cf[..., 0] - tau)[tp].mean()))
        rows.append(dict(method=name, metric="cf_eta1_mae",
                         value=np.abs(cf[..., 0] - truth["eta_ctrl"][..., 0])[tp].mean()))
    sc_mean = M.sc_complete(mean, truth)
    rows.append(dict(method="sc_means", metric="outcome_mae",
                     value=np.abs((mean - sc_mean) / s2_true - tau)[tp].mean()))
    for r in rows:
        r.update(table="table1", setting=f"tau={tau}", rep=rep)
    return rows


def table2(rep, m):
    panel, truth = dgp.gaussian_panel(intervention="structured", kappa=(0.6, 0.5),
                                      m_fixed=m, seed=2000 + rep)
    tp = truth["treated_post"]
    mle = M.cell_mle(truth, "gaussian")
    kl_true = M.gaussian_kl(truth["eta_treat"], truth["eta_ctrl"])
    fits = {}
    fits["pmf"] = M.efsc_fits(panel, truth, rank=2, config=M.PAPER, seed=rep)
    fits["mle_sc"] = (M.gaussian_component_complete(mle, truth, M.sc_complete), mle)
    fits["mle_mcnnm"] = (M.gaussian_component_complete(mle, truth, M.mcnnm_complete), mle)
    fits["mom_mcnnm"] = mom_mcnnm(truth)
    fits["dsc"] = (M.dsc_counterfactual(truth), mle)
    fits["pmfcf_mletr"] = (fits["pmf"][0], mle)
    rows = []
    for name, (cf, tr) in fits.items():
        rows.append(dict(method=name, metric="kl_mae",
                         value=np.abs(M.gaussian_kl(tr, cf) - kl_true)[tp].mean()))
        rows.append(dict(method=name, metric="cf_eta1_mae",
                         value=np.abs(cf[..., 0] - truth["eta_ctrl"][..., 0])[tp].mean()))
        mu_c = cf[..., 0] / (-2 * cf[..., 1])
        mu_ct = truth["eta_ctrl"][..., 0] / (-2 * truth["eta_ctrl"][..., 1])
        rows.append(dict(method=name, metric="cf_mean_mae", value=np.abs(mu_c - mu_ct)[tp].mean()))
    rows.append(dict(method="truth", metric="kl_mean", value=kl_true[tp].mean()))
    for r in rows:
        r.update(table="table2", setting=f"m={m}", rep=rep)
    return rows


def table2_swap(rep, m):
    """Swap counterfactual and treated sides between EFSC and the moments baseline.

    Same panels and seeds as ``table2``. If ``momcf_pmftr`` matches ``pmf``,
    EFSC's KL advantage is its treated-side fit, not its counterfactual.
    """
    panel, truth = dgp.gaussian_panel(intervention="structured", kappa=(0.6, 0.5),
                                      m_fixed=m, seed=2000 + rep)
    tp = truth["treated_post"]
    kl_true = M.gaussian_kl(truth["eta_treat"], truth["eta_ctrl"])
    pmf_cf, pmf_tr = M.efsc_fits(panel, truth, rank=2, config=M.PAPER, seed=rep)
    mom_cf, mom_tr = mom_mcnnm(truth)
    fits = {"pmf": (pmf_cf, pmf_tr), "mom_mcnnm": (mom_cf, mom_tr),
            "momcf_pmftr": (mom_cf, pmf_tr), "pmfcf_momtr": (pmf_cf, mom_tr)}
    rows = []
    for name, (cf, tr) in fits.items():
        rows.append(dict(method=name, metric="kl_mae",
                         value=np.abs(M.gaussian_kl(tr, cf) - kl_true)[tp].mean()))
        rows.append(dict(method=name, metric="tr_eta1_mae",
                         value=np.abs(tr[..., 0] - truth["eta_treat"][..., 0])[tp].mean()))
    for r in rows:
        r.update(table="table2_swap", setting=f"m={m}", rep=rep)
    return rows


if __name__ == "__main__":
    which, reps, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    grid = {"table1": (table1, [0.1, 0.5, 2.0]), "table2": (table2, [5, 25, 100]),
            "table2_swap": (table2_swap, [5, 25, 100])}[which]
    fn, settings = grid
    all_rows = []
    for rep in range(reps):
        for s in settings:
            t0 = time.perf_counter()
            all_rows += fn(rep, s)
            pd.DataFrame(all_rows).to_csv(out, index=False)
            print(f"{which} rep={rep} setting={s} {time.perf_counter() - t0:.0f}s", flush=True)
