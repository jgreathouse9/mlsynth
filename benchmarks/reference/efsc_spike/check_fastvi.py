"""Confirm ``fastvi`` reproduces the upstream fit on a small panel."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import efsc.fitting  # noqa: E402
from efsc import InferenceConfig  # noqa: E402

import dgp  # noqa: E402
import fastvi  # noqa: E402

panel, truth = dgp.gaussian_panel(n=12, T=30, r=2, m_fixed=20, treated=3, post=6, seed=4)
mask = truth["counterfactual_mask"]
cfg = InferenceConfig(epochs=400, learning_rate=0.1, prior_scale=1.0)

original = efsc.fitting.make_negative_elbo
t0 = time.perf_counter()
a = efsc.fitting.fit_factorization(panel, mask, rank=2, config=cfg, seed=11)
t_up = time.perf_counter() - t0
fastvi.install()
t0 = time.perf_counter()
b = efsc.fitting.fit_factorization(panel, mask, rank=2, config=cfg, seed=11)
t_fast = time.perf_counter() - t0
efsc.fitting.make_negative_elbo = original

diff = float(torch.max(torch.abs(a.natural_parameters - b.natural_parameters)))
loss = float(np.max(np.abs(a.loss_history - b.loss_history)))
out = {"max_abs_eta_diff": diff, "max_abs_loss_diff": loss,
       "seconds_upstream": t_up, "seconds_fastvi": t_fast}
print(json.dumps(out, indent=2))
(HERE / "fastvi_check.json").write_text(json.dumps(out, indent=2) + "\n")
