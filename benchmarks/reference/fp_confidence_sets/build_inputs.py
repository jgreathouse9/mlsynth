"""Shared inputs for the Firpo-Possebom cross-validation.

``SCM.CS`` (the authors' ``function_SCM-CS_v07.R``) takes the outcome matrix and
a matrix of pre-computed placebo weights and does the rest in base R. So the
weights are an INPUT to the procedure being validated, not part of it: computing
them once here and handing the same two matrices to R and to the Python port
isolates the confidence-set inversion, which is the thing under test.

Ingestion goes through ``dataprep``. The weights are the outcome-only simplex
fit over the pre-period, which is deterministic -- the authors use Synth's full
ADH predictor specification instead, so the absolute bounds here will not equal
the paper's published set. That is fine and is stated in the report: this run
validates the inversion, not the weight estimator.

State order is alphabetical, matching the reference driver
(``california_beta_testing_2018-08-12.R`` relies on R factor-level order, under
which ``californiaid <- 3``).
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/mlsynth")
from mlsynth.utils.datautils import dataprep
from mlsynth.utils.bilevel.simplex import simplex_lstsq

OUT = "/tmp/claude-0/-home-user-mlsynth/bdb00eab-f103-5c36-bc1b-0069a19c5fb9/scratchpad/fp"
T0 = 19                                   # 1970-1988 inclusive

df = pd.read_csv("/home/user/mlsynth/basedata/P99data.csv")
df["treat"] = ((df.state == "California") & (df.year >= 1989)).astype(int)

prep = dataprep(df, "state", "year", "cigsale", "treat")
assert int(prep["pre_periods"]) == T0, prep["pre_periods"]

# dataprep's canonical wide outcome frame, columns re-ordered alphabetically so
# the column index matches the reference driver's stateid.
wide = prep["Ywide"].copy()
wide = wide.reindex(sorted(wide.columns), axis=1)
states = list(wide.columns)
Ymat = wide.to_numpy(float)               # (31, 39)
cal = states.index("California") + 1      # 1-based, for R
assert cal == 3, cal

# weightsmat: column j holds unit j's 38 donor weights, donors in ascending
# index order excluding j -- the layout `SCM.CS` indexes with setdiff(1:N, j).
N = len(states)
W = np.zeros((N - 1, N))
for j in range(N):
    donors = [k for k in range(N) if k != j]
    y = Ymat[:T0, j]
    X = Ymat[:T0, donors]
    W[:, j] = simplex_lstsq(X, y)

np.savetxt(f"{OUT}/Ymat.csv", Ymat, delimiter=",")
np.savetxt(f"{OUT}/weightsmat.csv", W, delimiter=",")
pd.Series(states).to_csv(f"{OUT}/states.csv", index=False, header=False)
print(f"Ymat {Ymat.shape}  weightsmat {W.shape}  T0={T0}  California={cal}")
print(f"weight column sums (should be 1): min {W.sum(0).min():.6f} max {W.sum(0).max():.6f}")
print(f"California donors with weight > 0.01:")
for k, s in enumerate([s for i, s in enumerate(states) if i != cal - 1]):
    if W[k, cal - 1] > 0.01:
        print(f"   {s:<16} {W[k, cal-1]:.4f}")
