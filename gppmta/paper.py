"""The Hawkes simulation design of Tao et al. (2023), Sec. 5.1 and Table 1."""

from __future__ import annotations

import numpy as np

from .fit import GPPParams
from .paths import EventSpace

SPACE = EventSpace(
    conversion="conversion",
    customer=("display_click", "search_impression", "search_click"),
    firm=("display_impression",),
    channels={
        "display": ("display_impression", "display_click"),
        "search": ("search_impression", "search_click"),
    },
)
CONV, DCLICK, SIMP, SCLICK, DIMP = range(5)

# Table 1: alpha[from, to].
ALPHA = np.zeros((5, 5))
ALPHA[DIMP, DCLICK] = 0.08
ALPHA[DIMP, SIMP] = 0.08
ALPHA[DIMP, CONV] = 0.01
ALPHA[DCLICK, CONV] = 0.08
ALPHA[SIMP, SCLICK] = 0.08
ALPHA[SIMP, CONV] = 0.02
ALPHA[SCLICK, CONV] = 0.1

# Baselines: search impression 0.02, conversion 1e-4, clicks 0; display
# impressions are the firm's Poisson(0.02) input.
MU_ALL = np.array([1e-4, 0.0, 0.02, 0.0, 0.02])
DECAY = 0.1  # psi(t) = exp(-t / 10) / 10
T = 365.0
N_PATHS = 10_000

TRUE_PARAMS = GPPParams(mu=MU_ALL[:4], alpha=ALPHA[:, :4], decay=DECAY)

# Table 2 of the paper (means over 100 runs).
TABLE2 = {
    "truth": {"display": 0.3799, "search": 0.6201},
    "tre": {"display": 0.3782, "search": 0.6218, "kl": 0.0002, "hellinger": 0.0064},
    "dre": {"display": 0.3491, "search": 0.6509, "kl": 0.0023, "hellinger": 0.0226},
}
