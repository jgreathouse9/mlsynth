"""Cross-validation: DSC against the published ``disco`` Stata Journal numbers.

The only fully external reference mlsynth's ``DSC`` has. Gunsilius & Van Dijcke's
Stata Journal article ships a replication repository
(`Davidvandijcke/disco_stata_journal <https://github.com/Davidvandijcke/disco_stata_journal>`_,
MIT) containing an anonymised employee-tenure panel and a log with printed
results -- donor weights and a quantile-effects table.

Why this case exists alongside ``dsc_dube``
-------------------------------------------

``dsc_dube`` pins mlsynth's own output: the DiSCo vignette is built with
``eval=FALSE`` and publishes no numbers, so its only external anchor is the
qualitative ``p > 0.05``. That is one bit of information, and a materially wrong
DSC could satisfy it.

The R package cannot supply more, because its weights are not deterministic:
``DiSCo_weights_reg`` draws the quadrature points with ``runif``, so the fitted
weights are a Monte Carlo estimate with ``O(M^-1/2)`` error. Measured on the Dube
panel it disagrees with *itself* across seeds by up to 0.119, where mlsynth sits
0.044 from any one of them -- so no tolerance against a single R run means
anything.

The Stata implementation is deterministic. ``disco_prob_grid``
(``src/disco_utils.mata``) builds an evenly spaced closed grid and the seed is
consumed only by the bootstrap. So its published weights are a real target, and
this case pins them.

What it took to match
---------------------

Reproducing the printed weights required three alignments, in descending order
of effect:

* the quadrature grid must be ``linspace(0, 1, M)`` with endpoints *included*,
  evaluating the sample minimum and maximum -- excluding them costs 0.0869;
* the simplex weights must come from an exact QP, not projected gradient
  (0.0047) -- FISTA matches the objective to 0.00 percent while its argmin
  differs, and the argmin is the reported quantity;
* the empirical quantile must be type 7, not type 1 (0.0026).

With all three, every published weight matches to the printed precision. See
issue #304 for the investigation, including the hypotheses that were wrong.

What the published numbers are, and are not
-------------------------------------------

The published run uses ``m(100)``, and at that resolution the answer is not
converged. 100 quadrature points are approximating quantile functions over cells
holding up to 71,438 observations, so the discretisation error is large and
visible:

    M          amazon weight
    100        0.2203   <- the published value
    1,000      0.1860
    5,000      0.1827
    20,000     0.1821

Both grid rules converge to about 0.182. The R package's random-draw rule gets
there too -- at M = 10,000 it averages 0.1826 across seeds -- so the apparent
"R disagrees with Stata" is mostly the two being compared at different effective
resolutions, not a difference of method.

This case therefore pins **two different things**, and conflating them would be
a mistake:

* ``published_weight_max_diff`` and the ``w_*`` rows reproduce the reference
  bit-for-bit *at the reference's own settings*. That is a real guarantee -- it
  is what says mlsynth implements this specification -- but 0.2203 is a
  quadrature artifact of ``m(100)``, not the estimand.
* ``amazon_converged`` and ``att_converged`` pin the M-converged answer, which
  is what the estimator is actually estimating. They are mlsynth's own output,
  so they are a regression pin rather than external agreement; their value is
  that they fail if the quadrature stops converging where it should.

A benchmark that pinned only the first would quietly assert that 0.2203 is the
right answer. A benchmark that pinned only the second would lose the exact
reproduction that validates the implementation. Hence both.

Scope
-----

The weights are deterministic and pinned exactly. The log's standard errors and
confidence intervals come from a 300-replication bootstrap under ``seed(12143)``
and are *not* pinned: reproducing them would require reproducing Stata's RNG
stream, which is not a property of the estimator.

The quantile-effects table is pinned as well. It is a genuinely independent
check on the same fitted object: the weights test the quadrature grid, quantile
convention and simplex solve, while the effects test the counterfactual
construction and the reporting grid. Stata evaluates them on the ``g()`` grid --
``g(10)`` in this example -- which is separate from the ``m()`` fitting grid;
at G = 100 or 1,000 the four effects are off by 2 to 3, so this is the exact
grid or nothing.

Provenance: ``basedata/disco_tenure.parquet``, converted from
``examples/tenure_anonymized.dta`` in that repository with every column verified
identical on round-trip. 32 firms x 3 periods, 1,027,406 individual observations.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "disco_tenure.parquet"

#: Verbatim from ``results/tenure_example.log``. Stata prints four decimals.
PUBLISHED_WEIGHTS = {
    "amazon": 0.2203,
    "autodesk": 0.1271,
    "cisco": 0.1066,
    "dell technologies": 0.0991,
    "slalom consulting": 0.0962,
}

#: Verbatim from the ``disco_estat summary`` table in the same log. The Effect
#: column is deterministic; the Std. Err. / CI columns are bootstrap output and
#: are deliberately not pinned.
PUBLISHED_EFFECTS = {
    (0.00, 0.25): -6.659,
    (0.25, 0.50): -21.443,
    (0.50, 0.75): -50.509,
    (0.75, 1.00): -54.751,
}

TARGET_ID = 2      # idtarget(2)
T0_PERIOD = 3      # t0(3): treatment in period 3, pre-periods {1, 2}
M_POINTS = 100     # m(100) -- the reference's setting, NOT a converged one
G_POINTS = 10      # g(10)  -- the reporting grid the effects table uses
M_CONVERGED = 5000  # where the weights have settled; M=20000 moves them 6e-4


def run() -> dict:
    from mlsynth import DSC

    if not _DATA.exists():                      # pragma: no cover - vendored
        return {k: float("nan") for k in EXPECTED}
    df = pd.read_parquet(_DATA)
    d = df.copy()
    d["treat"] = ((d.id_col == TARGET_ID) & (d.time_col >= T0_PERIOD)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = DSC({
            "df": d, "outcome": "y_col", "treat": "treat", "unitid": "id_col",
            "time": "time_col", "M": M_POINTS, "compute_inference": False,
            "display_graphs": False,
        }).fit()

    names = df.groupby("id_col").company_name.first()
    w = {names[int(k)]: float(v) for k, v in res.donor_weights.items()}

    # The published quantile-effects table, on Stata's g() reporting grid.
    from mlsynth.utils.dsc_helpers.quantiles import empirical_quantile
    from mlsynth.utils.dsc_helpers.setup import prepare_dsc_inputs
    inp = prepare_dsc_inputs(d, "y_col", "treat", "id_col", "time_col")
    order = inp.unit_names
    wvec = np.array([res.donor_weights[u] for u in order[1:]])
    grid = np.linspace(0.0, 1.0, G_POINTS)
    post = inp.time_labels[inp.T0]
    donors_q = np.column_stack([
        empirical_quantile(inp.cell_samples[(u, post)], grid) for u in order[1:]])
    qte = empirical_quantile(inp.cell_samples[(order[0], post)], grid) - donors_q @ wvec
    eff_diffs = []
    for (lo, hi), published in PUBLISHED_EFFECTS.items():
        band = (grid >= lo) & (grid <= hi)
        eff_diffs.append(abs(float(qte[band].mean()) - published))
    diffs = [abs(w[n] - v) for n, v in PUBLISHED_WEIGHTS.items() if n in w]
    ranked = sorted(w, key=w.get, reverse=True)[:5]

    out = {
        "n_donors": float(len(w)),
        "weights_sum_to_one": float(abs(sum(w.values()) - 1.0) < 1e-9),
        # The headline: distance from the printed table, so the row reads as
        # "how far apart are we" and cannot be quietly re-fitted.
        "published_weight_max_diff": float(max(diffs)) if diffs else float("nan"),
        # Ordering as well as values -- a permutation would satisfy the
        # per-donor distances while reporting a different donor ranking.
        "top5_order_matches": float(ranked == list(PUBLISHED_WEIGHTS)),
        # Second, independent quantity from the same fitted object.
        "published_effect_max_diff": float(max(eff_diffs)),
    }
    for name in PUBLISHED_WEIGHTS:
        key = "w_" + name.split()[0]
        out[key] = float(w.get(name, float("nan")))

    # The M-converged answer: what the estimator estimates, as opposed to what
    # the reference reports at its own m(100).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conv = DSC({
            "df": d, "outcome": "y_col", "treat": "treat", "unitid": "id_col",
            "time": "time_col", "M": M_CONVERGED, "compute_inference": False,
            "display_graphs": False,
        }).fit()
    cw = {names[int(k)]: float(v) for k, v in conv.donor_weights.items()}
    out["amazon_converged"] = float(cw["amazon"])
    out["att_converged"] = float(conv.att)
    return out


# Deterministic: the Stata grid is evenly spaced and the weight solve is an
# exact QP, so repeat runs are bit-identical and the reference has no RNG in the
# quantity being compared.
EXPECTED = {
    "n_donors": (31.0, 0.0),
    "weights_sum_to_one": (1.0, 0.0),
    # 5e-5 is the printed precision of the reference: Stata reports four
    # decimals, so this is as tight as the published table can be checked, and
    # tightening it further would be pinning rounding noise.
    "published_weight_max_diff": (0.0, 5e-5),
    "top5_order_matches": (1.0, 0.0),
    "w_amazon": (0.2203, 5e-5),
    "w_autodesk": (0.1271, 5e-5),
    "w_cisco": (0.1066, 5e-5),
    "w_dell": (0.0991, 5e-5),
    "w_slalom": (0.0962, 5e-5),
    # The second published quantity. Stata prints three decimals here, so 5e-4
    # is its printed resolution.
    "published_effect_max_diff": (0.0, 5e-4),
    # --- the estimand, not the reference's approximation of it.
    # mlsynth's own output (a regression pin, not external agreement). The
    # tolerance covers the residual drift from M=5000 to M=20000, which is
    # 6e-4 on the weight: if convergence degrades, this fails.
    "amazon_converged": (0.1827, 0.002),
    "att_converged": (-32.003, 0.05),
}
