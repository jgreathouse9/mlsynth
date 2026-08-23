r"""Cross-validation: mlsynth's conformal test against Facure's chapter.

The Chernozhukov-Wuthrich-Zhu test is short enough to write twice, which makes it
a good candidate for a value-for-value cross-check. Matheus Facure's *Causal
Inference for the Brave and True* implements it from scratch in
``Conformal-Inference-for-Synthetic-Control.ipynb``: impose a constant null over
the post window, refit a simplex synthetic control on the imputed series, roll
the residual path through all ``T`` cyclic shifts, score each shift's post block
with ``S_q = (mean |u|^q)^(1/q)``, and report the share of shifts scoring at
least as extreme as the observed one.

This case transcribes that notebook -- the cvxpy program and the p-value loop,
not a paraphrase -- and runs it beside
:func:`~mlsynth.utils.bilevel.ridge_inference.conformal_pvalue` on the same
panel. The two are expected to agree exactly, not approximately. mlsynth's
statistic is ``(sum |u|^q / sqrt(n))^(1/q)`` where the notebook's is the mean,
and mlsynth counts ``obs <= perm`` where the notebook counts
``stat >= stat[0]``; over a post block of fixed length both are monotone
re-expressions of the same quantity, so any disagreement is a defect and not a
convention.

Also pinned: the cumulative band that inverts this test
(:func:`~mlsynth.utils.conformal.cumulative_conformal_by_inversion`), by both of
its searches, because this panel is the reason the second one exists. At
``alpha = 0.1`` the accepted set here is two islands -- roughly ``[-122, -20]``
and ``[-10, +5]`` -- separated by a candidate whose p-value falls to the block
scheme's floor of ``1 / 31``. A search that brackets outward from the point
estimate and bisects stops at the first crossing near ``-19.6`` and never sees
the far island, so it excludes zero even though the zero-null p-value of 0.161
accepts it. The grid search scans and returns the range of every survivor, which
puts zero back inside and flags that the survivors had holes. Both are recorded:
the disagreement is a property of the two searches, not a defect in either.

Data: Abadie, Diamond & Hainmueller's California Proposition 99 panel,
``basedata/smoking_data.csv``, 38 donor states over 1970-2000, on the notebook's
own convention of a 1988 intervention and a 13-period window.

Provenance: github.com/matheusfacure/python-causality-handbook, chapter
"Conformal Inference for Synthetic Controls" (MIT licence, not vendored).
"""
from __future__ import annotations

_INTERVENTION = 1988          # the notebook's convention, not Abadie's 1989
_LAST_YEAR = 2000
_NULLS = (0.0, -10.0, -20.0, -30.0)


def _panel():
    import numpy as np
    import pandas as pd
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    csv = root / "basedata" / "smoking_data.csv"
    if not csv.exists():                                   # pragma: no cover
        from benchmarks.compare import BenchmarkSkipped
        raise BenchmarkSkipped(f"missing {csv}")
    d = pd.read_csv(csv)
    wide = d.pivot(index="state", columns="year", values="cigsale").T
    wide = wide.loc[wide.index <= _LAST_YEAR].sort_index()
    y = wide["California"].to_numpy(dtype=float)
    Y0 = wide.drop(columns=["California"]).to_numpy(dtype=float)
    pre = int((wide.index < _INTERVENTION).sum())
    return y, Y0, pre


def _notebook_weights(X, target):
    """The notebook's ``SyntheticControl.fit``: cvxpy, ``sum(w) == 1``, ``w >= 0``."""
    import cvxpy as cp
    import numpy as np

    w = cp.Variable(X.shape[1])
    cp.Problem(cp.Minimize(cp.sum_squares(X @ w - target)),
               [cp.sum(w) == 1, w >= 0]).solve(verbose=False)
    return np.asarray(w.value, dtype=float)


def _notebook_pvalue(y, Y0, pre, null, q=1):
    """The notebook's ``with_effect`` / ``residuals`` / ``p_value``, transcribed."""
    import numpy as np

    y_null = y.copy()
    y_null[pre:] -= null
    resid = y_null - Y0 @ _notebook_weights(Y0, y_null)
    post = np.zeros(resid.size, dtype=bool)
    post[pre:] = True
    stats = np.array([(np.abs(np.roll(resid, s)[post]) ** q).mean() ** (1.0 / q)
                      for s in range(resid.size)])
    return float(np.mean(stats >= stats[0]))


def run() -> dict:
    import numpy as np

    from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue
    from mlsynth.utils.conformal import cumulative_conformal_by_inversion

    y, Y0, pre = _panel()
    out = {"pre_periods": float(pre), "post_periods": float(y.size - pre),
           "n_donors": float(Y0.shape[1])}

    worst = 0.0
    for null in _NULLS:
        theirs = _notebook_pvalue(y, Y0, pre, null)
        shifted = y.copy()
        shifted[pre:] -= null
        ours = conformal_pvalue(shifted, Y0, pre, refit="sc",
                                conformal_type="block", q=1.0)
        worst = max(worst, abs(theirs - ours))
        out[f"pvalue_null_{int(abs(null))}"] = ours
    out["max_pvalue_gap_vs_notebook"] = worst

    kw = dict(alpha=0.1, refit="sc", conformal_type="block")
    bisect = cumulative_conformal_by_inversion(y, Y0, pre, **kw)
    out.update(bisect_point=bisect.point,
               bisect_per_period_lower=bisect.per_period_lower,
               bisect_per_period_upper=bisect.per_period_upper,
               bisect_zero_inside=float(bisect.lower <= 0.0 <= bisect.upper))

    grid = cumulative_conformal_by_inversion(
        y, Y0, pre, search="grid", grid=np.arange(-160.0, 40.5, 2.5), **kw)
    out.update(grid_lower=grid.lower, grid_upper=grid.upper,
               grid_per_period_lower=grid.per_period_lower,
               grid_per_period_upper=grid.per_period_upper,
               grid_has_gaps=float(bool(grid.has_interior_gaps)),
               grid_zero_inside=float(grid.lower <= 0.0 <= grid.upper))

    # The band is the accepted set, so the search that can see the whole set must
    # place zero on the side the zero-null p-value does.
    out["grid_zero_agreement"] = float(
        out["grid_zero_inside"] == float(out["pvalue_null_0"] > 0.1))
    return out


EXPECTED = {
    "pre_periods": (18.0, 0.0),
    "post_periods": (13.0, 0.0),
    "n_donors": (38.0, 0.0),
    # exact agreement: 1/31 granularity, so a tolerance below half a step
    "max_pvalue_gap_vs_notebook": (0.0, 1e-12),
    "pvalue_null_0": (0.161290, 0.0005),
    "pvalue_null_10": (0.161290, 0.0005),
    "pvalue_null_20": (0.129032, 0.0005),
    "pvalue_null_30": (0.451613, 0.0005),
    # the grid search sees both islands, so it agrees with the p-value it inverts
    "grid_zero_agreement": (1.0, 0.0),
    "grid_zero_inside": (1.0, 0.0),
    "grid_has_gaps": (1.0, 0.0),
    "grid_per_period_lower": (-122.5, 2.6),
    "grid_per_period_upper": (5.0, 2.6),
    # the bisecting search stops at the near crossing; recorded, not gated as a
    # defect, because finding a narrow set at any width is what it is for
    "bisect_zero_inside": (0.0, 0.0),
    "bisect_per_period_upper": (-19.64, 0.5),
    "bisect_point": (-240.48, 0.5),
}
