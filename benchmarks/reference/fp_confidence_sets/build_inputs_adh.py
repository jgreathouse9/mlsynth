"""The authors' ADH predictor specification, for the published comparison.

``california_beta_testing_2018-08-12.R`` builds its weights with Synth under

    predictors            lnincome, beer, age15to24, retprice
    predictors.op         mean, over 1980-1988
    special.predictors    cigsale at 1975, 1980, 1988
    time.optimize.ssr     1970-1988

``beer`` is only observed from 1984, so a mean over 1980-1988 that drops
missing values is the 1984-1988 mean -- which is how ADH themselves enter it.

Each of the 39 units is fitted as pseudo-treated against the other 38, which is
what the confidence set consumes.
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/mlsynth")
from mlsynth.utils.datautils import dataprep
from mlsynth.utils.bilevel import BilevelProblem, solve_bilevel

OUT = "/tmp/claude-0/-home-user-mlsynth/bdb00eab-f103-5c36-bc1b-0069a19c5fb9/scratchpad/fp"
METHOD = sys.argv[1] if len(sys.argv) > 1 else "malo"
T0 = 19

df = pd.read_csv("/home/user/mlsynth/basedata/P99data.csv")
df["treat"] = ((df.state == "California") & (df.year >= 1989)).astype(int)
prep = dataprep(df, "state", "year", "cigsale", "treat")
wide = prep["Ywide"].reindex(sorted(prep["Ywide"].columns), axis=1)
states = list(wide.columns)
Y = wide.to_numpy(float)                                  # (31, 39)
years = np.asarray(prep["time_labels"]).astype(int)

def col(var, lo, hi):
    w = df[(df.year >= lo) & (df.year <= hi)].pivot(index="year", columns="state",
                                                    values=var)
    return np.nanmean(w.reindex(columns=states).to_numpy(float), axis=0)

X = np.vstack([
    col("lnincome", 1980, 1988), col("beer", 1980, 1988),
    col("age15to24", 1980, 1988), col("retprice", 1980, 1988),
    Y[years == 1975][0], Y[years == 1980][0], Y[years == 1988][0],
])                                                        # (7, 39)
names = ["lnincome", "beer", "age15to24", "retprice",
         "cigsale1975", "cigsale1980", "cigsale1988"]
assert np.isfinite(X).all()

N = len(states)
W = np.zeros((N - 1, N))
t0 = time.time()
for j in range(N):
    donors = [k for k in range(N) if k != j]
    prob = BilevelProblem(y1_pre=Y[:T0, j], Y0_pre=Y[:T0, donors],
                          X1=X[:, j], X0=X[:, donors], predictor_names=names)
    W[:, j] = np.asarray(solve_bilevel(prob, method=METHOD).W, float).ravel()
    if (j + 1) % 10 == 0:
        print(f"  {j+1}/{N}  {time.time()-t0:.0f}s", flush=True)

np.savetxt(f"{OUT}/weightsmat_adh_{METHOD}.csv", W, delimiter=",")
cal = states.index("California")
print(f"\n{METHOD}: California donors with weight > 0.01")
for k, s in enumerate([s for i, s in enumerate(states) if i != cal]):
    if W[k, cal] > 0.01:
        print(f"   {s:<16} {W[k, cal]:.4f}")
print("ADH published: Utah .334  Nevada .234  Montana .199  Colorado .164  Connecticut .069")
