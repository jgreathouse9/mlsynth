# gppmta: graphical point process multi-touch attribution (prototype)

A Python prototype of Tao, Chen, Snyder, Kumar, Meisami and Xue (2023), "A
Graphical Point Process Framework for Understanding Removal Effects in
Multi-Touch Attribution" (arXiv:2302.06075, Management Science), including its
electronic companion. NumPy, SciPy and pandas only.

## What it does

Each customer's path to purchase is a stream of timestamped events. Firm-initiated
touches (emails sent, ad impressions) are taken as given. Customer-initiated
events (clicks, searches, conversion) follow a Hawkes-type model, eq. (1):

    lambda_e(t) = mu_e + sum_k alpha_{ke} sum_{t_l < t, e_l = k} psi_k(t - t_l),

with `psi_k(t) = beta_k exp(-beta_k t)`. A zero `alpha_{ke}` means no Granger
edge `k -> e`. Given a fitted model, each conversion's credit is split two ways:

- Direct removal effect (DRE, eq. 4): the share of the conversion intensity that
  the removed touches contribute directly.
- Total removal effect (TRE, Sec. 3.2): additionally removes the later
  customer-initiated events those touches would have triggered (e.g. a display
  impression that led to a search click).

## Layout

| module | contents |
| --- | --- |
| `paths.py` | `Path`, `EventSpace` (names, customer/firm split, channels), `paths_from_frame` |
| `stats.py` | exact sufficient statistics `V`, `b` of the least-squares contrast (EC.2) |
| `fit.py` | per-node solver, the paper's ADMM (Algorithm 3), CV for `gamma` and the kernel decay, `GraphicalPointProcess` |
| `attribution.py` | DRE, TRE (exact linear solve; Algorithm 2 backprop; Algorithm 1 thinning), per-conversion channel table, CAS shares, KL / Hellinger |
| `simulate.py` | branching-form Hawkes simulator with exact channel-off ground truth; time-rescaling goodness of fit |
| `paper.py` | the Sec. 5.1 design (Table 1) and the paper's Table 2 numbers |
| `scripts/` | `replicate_table2.py` (Table 2 reproduction), `export_simulation.py` (write a simulated dataset) |

## Usage

```python
import pandas as pd
from gppmta import EventSpace, GraphicalPointProcess, paths_from_frame, attribute, channel_shares
from gppmta.fit import cv_decay

space = EventSpace(
    conversion="purchase",
    customer=("search_click", "email_click"),
    firm=("email_sent", "display_impression"),
    channels={"email": ("email_sent", "email_click"),
              "search": ("search_click",),
              "display": ("display_impression",)},
)
# one row per event: path_id, time (e.g. days), event label
paths, ids = paths_from_frame(df, space)

decay, _ = cv_decay(paths, space.p, space.q, grid=[0.05, 0.1, 0.2, 0.5, 1.0])
model = GraphicalPointProcess(decay=decay, gamma="cv").fit(paths, space.p, space.q)
model.params_.edges                      # estimated Granger graph, (from, to) indices
table = attribute(model.params_, paths, space.channel_types(), ids)  # one row per conversion
channel_shares(table, "tre")             # channel shares of credit
```

Install the dependencies with `pip install -r requirements.txt`. Run the tests
with `python -m pytest tests -q`, and the Table 2 reproduction with
`python scripts/replicate_table2.py --runs 100`.

## Data

All data here are simulated from the Sec. 5.1 design; there is no real
customer data.

- `data/table1_seed0.csv.gz`: run 0 of the Table 2 reproduction, 10,000 paths
  of 365 days and 167,368 events, one row per event (`path_id`, `time` in
  days, `event`, window end `T`). Load it with
  `paths_from_frame(df, gppmta.paper.SPACE, end_col="T", origin="zero")`.
- `data/table1_seed0_truth.json`: that run's channel-off ground truth (CCC).
- `results/table2_runs.csv`: per-run results of all 100 runs;
  `results/table2_summary.txt` is the printed summary.

`python scripts/export_simulation.py --seed S --out data/table1_seedS.csv.gz`
regenerates run `S` exactly.

## Departures from the paper

1. Solver. On the non-negative orthant the L1 penalty is linear, so each node's
   problem (EC.3) is a bound-constrained QP. With `V = L L'` it is exactly a
   non-negative least-squares problem, solved to optimality by
   `scipy.optimize.nnls`. The paper's ADMM is in `fit.admm` for comparison.
   Its theta-step clips the unconstrained minimiser at zero, which is not the
   constrained minimiser when features are correlated, so it can stop at a
   point that is not the optimum (its residuals are exactly zero there). On the
   Table 1 design it lands within 0.3% of the optimum; on a two-variable example
   in `tests/test_fit.py` it is far off.
2. Tuning. The companion's `gamma_e` depends on unobservable constants
   (Theorem 3). Here `gamma_e` is chosen per node by K-fold CV of the held-out
   least-squares contrast, which costs one statistics pass per fold because the
   contrast is additive over paths. The kernel decay, which the paper treats as
   known, can be chosen the same way (`cv_decay`); on the Table 1 design it
   recovers the true value.
3. TRE. The thinning probabilities of Algorithm 1 are linear in the indicators
   of earlier removals, so `P(l in R°)` solves one triangular system and TRE is
   exact. Algorithm 2 computes the adjoint of the same system; the tests hold
   the linear solve, Algorithm 2 and Algorithm 1's Monte Carlo to one another.
4. Ground truth. The simulator uses the branching (cluster) form of the Hawkes
   process, so switching a channel off removes exactly its events and their
   descendants under the same draws. `CCC_z` is therefore exact per run, not a
   difference of two independent simulations.
5. Kernel. One exponential kernel per source type (the paper allows one per
   edge, with box and Gaussian examples). With per-source kernels `V` is shared
   by all nodes.

## Reproduction of Table 2 (Sec. 5.1)

100 runs of 10,000 paths x 365 days from the Table 1 design; penalties chosen
by 5-fold CV; kernel decay 0.1 as in the paper. Mean (sd) over runs; the paper
reports mean (sd) over 100 runs.

| quantity | this prototype | paper Table 2 |
| --- | --- | --- |
| display share, truth (CCC) | 0.3811 (0.0090) | 0.3799 |
| TRE display share | 0.3792 (0.0112) | 0.3782 (0.0104) |
| TRE KL divergence | 0.0002 (0.0002) | 0.0002 (0.0002) |
| TRE Hellinger distance | 0.0050 (0.0038) | 0.0064 (0.0042) |
| DRE display share | 0.3503 (0.0122) | 0.3491 (0.0112) |
| DRE KL divergence | 0.0022 (0.0013) | 0.0023 (0.0015) |
| DRE Hellinger distance | 0.0226 (0.0067) | 0.0226 (0.0083) |
| TRE display share, true parameters | 0.3802 (0.0069) | |
| DRE display share, true parameters | 0.3513 (0.0075) | |

Every figure is within a fraction of one run-to-run sd of the paper's. Max
absolute error of the fitted `alpha` averaged 0.0077 (true values 0.01-0.1).
Run time 444 s for all 100 runs on one core. Per-run results:
`results/table2_runs.csv`.

## Caveats

- The scores are removal effects under the fitted model. Firm-initiated
  touches are treated as exogenous, but targeting usually follows customer
  intent (retargeting, bidding on likely buyers), and the impression
  coefficients absorb that. A holdout or geo experiment remains the check on
  true incrementality.
- TRE is subadditive and over-allocates on chain-shaped graphs (Sec. 3.3):
  score a channel by removing all of its touches at once (as `attribute`
  does), not by summing touchpoint TREs.
- Kernel misspecification moves the answer. On one Table 1 dataset (truth
  0.370), the display TRE share is 0.369 / 0.380 / 0.374 / 0.354 / 0.335 at
  decays 0.025 / 0.05 / 0.1 (true) / 0.2 / 0.4.
- Under fitted parameters a conversion can have zero intensity (no fitted
  baseline and no fitted parents); its scores are NaN and `attribute` warns.
- Only the Sec. 5.1 design is reproducible. DASS+ (Sec. 5.2) is an unreleased
  modification of Google's simulator, and the Sec. 6 data are proprietary.
