"""Multinomial head-to-head: EFSC vs COMPSC vs logit completion.

The paper's Medicaid application is a five-category composition; mlsynth's
COMPSC is the estimator a user would reach for. Panel: N=40, T=12, 16 units
treated over the last 6 periods, rank 1 per logit, Medicaid-shaped base shares
(dgp.BASE_PROBS), intervention moving mass from "uninsured" to "Medicaid".

Methods (treated post block):
  efsc            package defaults (N(0,2^2) prior as in C.7.3, 2000 epochs)
  efsc_cf_obs_tr  EFSC counterfactual against the observed treated shares
  compsc          mlsynth.COMPSC per treated unit, default zero replacement 1e-6
  compsc_halfm    mlsynth.COMPSC with zero replacement 0.5/m
  logit_mcnnm     Haldane-corrected cell logits, mlsynth.MCNNM per logit

Usage: python run_multinomial.py REPS OUT.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import rel_entr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dgp  # noqa: E402
import fastvi  # noqa: E402
import methods as M  # noqa: E402

fastvi.install()
torch.set_num_threads(2)


def one(rep, m):
    panel, truth = dgp.multinomial_panel(m_fixed=m, seed=3000 + rep)
    tp = truth["treated_post"]
    observed = truth["counts"] / truth["counts"].sum(axis=-1, keepdims=True)
    cf_e, tr_e = M.efsc_fits(panel, truth, rank=1, config=M.PACKAGE_DEFAULT, seed=rep)
    logits = M.cell_mle(truth, "multinomial")
    mc = np.stack([M.mcnnm_complete(logits[..., c], truth) for c in range(logits.shape[-1])], -1)
    fits = {
        "efsc": (M.softmax_ref(cf_e), M.softmax_ref(tr_e)),
        "efsc_cf_obs_tr": (M.softmax_ref(cf_e), observed),
        "compsc": (M.compsc_counterfactual(truth), observed),
        "compsc_halfm": (M.compsc_counterfactual(truth, zero_replacement=0.5 / m), observed),
        "logit_mcnnm": (M.softmax_ref(mc), observed),
    }
    p_c, p_t = truth["p_ctrl"], truth["p_treat"]
    eff_true = (p_t - p_c)[tp]
    rows = []
    for name, (cf, tr) in fits.items():
        eff = (tr - cf)[tp]
        kl_hat = np.sum(rel_entr(tr, cf), axis=-1)[tp]
        kl_true = np.sum(p_t * np.log(p_t / p_c), axis=-1)[tp]
        rows += [
            dict(method=name, metric="cf_prob_mae", value=np.abs(cf - p_c)[tp].mean()),
            dict(method=name, metric="cell_effect_mae", value=np.abs(eff - eff_true).mean()),
            # Table 4's quantity: the block-average shift per category.
            dict(method=name, metric="avg_effect_abs_err",
                 value=np.abs(eff.mean(0) - eff_true.mean(0)).mean()),
            dict(method=name, metric="medicaid_avg_effect",
                 value=eff.mean(0)[1]),
            dict(method=name, metric="kl_mae", value=np.abs(kl_hat - kl_true).mean()),
        ]
    rows.append(dict(method="truth", metric="medicaid_avg_effect", value=eff_true.mean(0)[1]))
    rows.append(dict(method="truth", metric="zero_share_cells",
                     value=float((truth["counts"] == 0).any(-1).mean())))
    for r in rows:
        r.update(setting=f"m={m}", rep=rep)
    return rows


if __name__ == "__main__":
    reps, out = int(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for rep in range(reps):
        for m in [20, 100, 1000]:
            t0 = time.perf_counter()
            rows += one(rep, m)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"multinomial rep={rep} m={m} {time.perf_counter() - t0:.0f}s", flush=True)
