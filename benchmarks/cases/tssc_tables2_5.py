r"""TSSC Step 1: does the test that picks the estimator have the right size?

Path B and cross-validation. Li, K. T. & Shankar, V. (2023), "A Two-Step
Synthetic Control Approach for Estimating Causal Effects of Marketing Events",
*Management Science*, Tables 2 to 5, against the authors' own
``TSSC_Tables2_5_Test_MSE.m`` run under GNU Octave.

What this covers that ``tssc_figure2`` does not
-----------------------------------------------

``tssc_figure2`` pins the paper's MSE comparison *given* a choice of estimator.
The choice itself is the two-step part: Step 1 tests two restrictions on the
MSC(c) fit -- that the donor weights sum to one, and that the intercept is zero
-- and routes the panel to SC, MSC(a), MSC(b) or MSC(c) on the answer. mlsynth
implements that test in ``tssc_helpers/selection.py``, and ``TSSCConfig``
documents ``m = T1`` as "the bootstrap special case the paper's simulations
validate". Those are these simulations, and nothing in the suite ran them.

A test with the wrong size routes panels to the wrong variant, and every number
downstream of the choice still looks reasonable. That is a different failure
from anything ``tssc_figure2`` can see.

The design
----------

Eleven units, eighty pre-periods, twenty post, no treatment effect. Outcomes
come from a three-factor model -- a nonlinear trend with an AR(1) component, an
ARMA(1,1) factor and an MA(2) factor -- with homogeneous loadings. Three data
generating processes decide which null is true:

========  ===================  ==========================================
DGP       shift                what it violates
========  ===================  ==========================================
1         none                 nothing; both restrictions hold
2         ``b0 = 0.1``         the treated unit's loadings, so the weights
                               no longer sum to one (:math:`H_{0a}`)
3         ``a0 = 0.5``         the treated unit's intercept
                               (:math:`H_{0b}`)
========  ===================  ==========================================

Each DGP violates exactly one restriction, so the sharpest thing to assert is
not a level but a pattern: under DGP2 the sum-to-one test should fire and the
intercept test should not, and under DGP3 the reverse. That is pinned with no
slack, and it is what would break if the two restriction matrices were ever
transposed.

The two shifts are not equally visible, and the case records the gap. Moving
the treated unit's loadings by ``b0 = 0.1`` pushes the weights far enough that
the sum-to-one test fires on every draw; shifting its intercept by
``a0 = 0.5`` is caught about two times in three. So a practitioner reading
Step 1 should not treat a passed intercept test as strong evidence at this
sample size, and the joint test, which rejects 0.92 of the time under DGP3,
is the more sensitive of the two routes to the same conclusion.

Size needs no published number
------------------------------

Tables 2 and 3 report rejection at nominal 0.50, 0.20, 0.10, 0.05 and 0.01.
Under DGP1 both nulls hold, so a correctly sized test rejects at exactly those
rates whatever the paper measured -- the target is the nominal level itself,
and the paper's own numbers are not needed for this half. That matters here
because the replication package's ReadMe says its numbers are not exactly
reproducible: the script reseeds from the clock on every replication and every
subsampling draw, so "the results will be similar to but not be exactly what is
in the paper".

Where the two implementations meet
----------------------------------

The Octave reference and mlsynth fit the *same panels*: the reference writes
every replication's panel out and the Python side reads them back, so the
estimator comparison carries no Monte Carlo error at all. The four constrained
fits agree value-for-value -- core Octave's ``qp`` against mlsynth's
CLARABEL solves.

The restriction statistics are a looser match, around 1e-3 relative, and the
reason is structural. Each is ``T1`` times the square of a deviation that is
near zero under the null, so it multiplies the gap between two solvers'
stopping tolerances by eighty and then squares what is left. The ATTs average
over twenty post-periods and are insensitive to the same gap. Both are pinned,
at their own scales.

The joint statistic is not compared at all. It is studentized by a variance
estimated from the subsampling draws, and no random stream crosses the language
boundary, so the two sides estimate different ``V_hat`` on the same panel by
construction. Its *rate* is compared; its value is not.

Provenance
----------

* Paper: Li & Shankar (2023), Management Science, Tables 2-5.
* Replication package ``MS-MKG-20-01498``, ``TSSC_Tables2_5_Test_MSE.m``,
  transcribed in ``benchmarks/octave/tssc_tables2_5.m``. Three MATLAB-only
  calls are replaced there and the header says why: ``lsqlin`` (optim package)
  by core ``qp``, ``rng('shuffle')`` by a seed, ``datasample`` by ``randi``.
* Skips when ``octave-cli`` is absent.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "benchmarks" / "octave" / "tssc_tables2_5.m"

T1, T2 = 80, 20
SIMS = 120          # the paper runs 500
SUBSAMPLES = 200    # the paper runs 400
M = 20              # subsample size, the paper's mmv
LEVELS = (0.50, 0.80, 0.90, 0.95, 0.99)
VARIANTS = ("SC", "MSCa", "MSCb", "MSCc")


def _require_octave():
    from benchmarks.compare import BenchmarkSkipped

    octave = shutil.which("octave-cli") or shutil.which("octave")
    if octave is None:
        raise BenchmarkSkipped("octave-cli not on PATH")
    return octave


def _reference(octave, outdir):
    from benchmarks.compare import BenchmarkSkipped

    proc = subprocess.run(
        [octave, "--no-gui", str(_SCRIPT), "--out", str(outdir),
         "--nr", str(SIMS), "--nb", str(SUBSAMPLES), "--t1", str(T1),
         "--t2", str(T2), "--m", str(M), "--dump", str(SIMS)],
        capture_output=True, text=True, cwd=str(_ROOT), timeout=7200)
    missing = [d for d in (1, 2, 3)
               if not (Path(outdir) / f"summary_dgp{d}.csv").exists()]
    if proc.returncode != 0 or missing:
        raise BenchmarkSkipped(
            f"octave reference failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-400:]}")
    return {d: (pd.read_csv(Path(outdir) / f"summary_dgp{d}.csv"),
                pd.read_csv(Path(outdir) / f"panels_dgp{d}.csv"))
            for d in (1, 2, 3)}


def _panel(frame, rep):
    wide = frame[frame.rep == rep].pivot(index="time", columns="unit", values="y")
    arr = wide.to_numpy()
    return arr[:, 0], arr[:, 1:]


def _variant_atts(y, donors):
    """mlsynth's four SC-class fits on one panel."""
    from mlsynth.utils.tssc_helpers.estimation import _features, _solve

    out = {}
    for method in VARIANTS:
        w = _solve(method, donors[:T1], y[:T1], T1, donors.shape[1])
        cf = _features(method, donors) @ w
        out[method] = float(np.mean(y[T1:] - cf[T1:]))
    return out


def _tests(y, donors, rng):
    """The three Step-1 statistics and their subsampling acceptance regions.

    ``select_method`` walks a decision tree and stops at the first test it
    accepts, so it cannot report all three on every panel. Tables 2 and 3 do,
    so the tests are run directly off one shared set of subsample refits --
    which is also what makes this affordable, since the refits dominate.
    """
    from mlsynth.utils.tssc_helpers.estimation import fit_mscc_beta
    from mlsynth.utils.tssc_helpers.selection import (
        _restriction_matrices, _run_test,
    )

    n_donors = donors.shape[1]
    beta = fit_mscc_beta(donors[:T1], y[:T1], T1, n_donors)
    betas = []
    for _ in range(SUBSAMPLES):
        idx = rng.integers(0, T1, size=M)
        b = fit_mscc_beta(donors[:T1][idx], y[:T1][idx], M, n_donors)
        if b is not None and np.all(np.isfinite(b)):
            betas.append(b)
    diff = np.asarray(betas) - beta

    out = {}
    for name, (R, q) in _restriction_matrices(len(beta)).items():
        studentize = name == "joint"
        per_level = {}
        for level in LEVELS:
            t = _run_test(name, R, q, beta, diff, T1, M,
                          alpha=1.0 - level, studentize=studentize)
            per_level[level] = bool(t.rejected)
            stat = t.statistic
        out[name] = (stat, per_level)
    return out


def _measure():
    octave = _require_octave()
    tmp = tempfile.mkdtemp(prefix="tssc_")
    try:
        ref = _reference(octave, tmp)
        rows = []
        for dgp, (summary, panels) in ref.items():
            rng = np.random.default_rng(20260921 + dgp)
            for i, oct_row in summary.iterrows():
                rep = i + 1
                y, donors = _panel(panels, rep)
                atts = _variant_atts(y, donors)
                tests = _tests(y, donors, rng)
                row = {"dgp": dgp, "rep": rep}
                for m in VARIANTS:
                    row[f"my_{m}"] = atts[m]
                    row[f"oct_{m}"] = float(oct_row[f"att_{m}"])
                row["my_H0a"] = tests["sum_to_one"][0]
                row["my_H0b"] = tests["zero_intercept"][0]
                row["oct_H0a"] = float(oct_row["test_H0a"])
                row["oct_H0b"] = float(oct_row["test_H0b"])
                for key, tag in (("joint", "J"), ("sum_to_one", "A"),
                                 ("zero_intercept", "B")):
                    for level in LEVELS:
                        lv = int(level * 100)
                        row[f"my_rej{tag}{lv}"] = tests[key][1][level]
                        row[f"oct_rej{tag}{lv}"] = 1.0 - float(oct_row[f"cov{tag}{lv}"])
                rows.append(row)
        return pd.DataFrame(rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run() -> dict:
    d = _measure()
    out = {"n_draws": float(len(d)), "n_per_dgp": float(len(d) // 3)}

    # The seam: same panels, same four fits.
    att_gap = max(float(np.abs(d[f"my_{m}"] - d[f"oct_{m}"]).max()) for m in VARIANTS)
    out["att_max_gap_vs_octave"] = att_gap
    # The restriction statistics, at their own scale: T1 times a squared
    # near-zero deviation amplifies the two solvers' stopping tolerances.
    rel = []
    for tag in ("H0a", "H0b"):
        mine, theirs = d[f"my_{tag}"].to_numpy(), d[f"oct_{tag}"].to_numpy()
        keep = np.abs(theirs) > 1e-12
        rel.append(float(np.max(np.abs(mine[keep] - theirs[keep])
                                / np.abs(theirs[keep]))))
    out["stat_max_rel_gap_vs_octave"] = round(float(max(rel)), 6)

    # Size: under DGP1 both nulls hold, so every rate is its nominal level.
    one = d[d.dgp == 1]
    size_gap = 0.0
    for tag in ("J", "A", "B"):
        for level in LEVELS:
            lv = int(level * 100)
            rate = float(one[f"my_rej{tag}{lv}"].mean())
            out[f"size_{tag}{lv}"] = round(rate, 3)
            size_gap = max(size_gap, abs(rate - (1.0 - level)))
    out["size_max_gap_to_nominal"] = round(size_gap, 3)

    # Power: each DGP violates exactly one restriction, and the matching test
    # is the one that should fire.
    for dgp, fires, quiet in ((2, "A", "B"), (3, "B", "A")):
        g = d[d.dgp == dgp]
        hit = float(g[f"my_rej{fires}95"].mean())
        miss = float(g[f"my_rej{quiet}95"].mean())
        out[f"power_dgp{dgp}_{fires}"] = round(hit, 3)
        out[f"power_dgp{dgp}_{quiet}"] = round(miss, 3)
        # The assertion is the contrast, not a level: the restriction that was
        # violated rejects far more often than the one that was not. A level
        # would be an assertion about how far each DGP moves, which the paper
        # chooses and this case does not test.
        out[f"dgp{dgp}_fires_on_the_right_test"] = float(hit - miss > 0.4)
    # The two shifts are not equally visible: b0 = 0.1 on the loadings moves
    # the weights far enough that the sum-to-one test always fires, while
    # a0 = 0.5 on the intercept is caught about two times in three.
    out["power_gap_dgp2_over_dgp3"] = round(
        float(d[d.dgp == 2].my_rejA95.mean() - d[d.dgp == 3].my_rejB95.mean()), 3)
    out["joint_power_dgp2"] = round(float(d[d.dgp == 2].my_rejJ95.mean()), 3)
    out["joint_power_dgp3"] = round(float(d[d.dgp == 3].my_rejJ95.mean()), 3)

    # Rates against the reference, which draws its own subsamples, so this is
    # a comparison of rates and not of draws.
    rate_gap = 0.0
    for dgp in (1, 2, 3):
        g = d[d.dgp == dgp]
        for tag in ("A", "B"):
            rate_gap = max(rate_gap, abs(float(g[f"my_rej{tag}95"].mean())
                                         - float(g[f"oct_rej{tag}95"].mean())))
    out["rate_max_gap_vs_octave"] = round(rate_gap, 3)

    # Tables 4 and 5, from the ATTs, relative to MSC(c). No treatment effect is
    # applied, so every ATT is estimation error.
    one = d[d.dgp == 1]
    mse = {m: float(np.mean(one[f"my_{m}"] ** 2)) for m in VARIANTS}
    var = {m: float(np.var(one[f"my_{m}"])) for m in VARIANTS}
    bias2 = {m: float(np.mean(one[f"my_{m}"]) ** 2) for m in VARIANTS}
    for m in VARIANTS:
        out[f"mse_ratio_{m}"] = round(mse[m] / mse["MSCc"], 3)
        out[f"bias2_over_var_{m}"] = round(bias2[m] / var[m], 4)
    # Under DGP1 the SC restrictions hold in population, so the most
    # constrained fit should be the most efficient -- Figure 2's claim, here as
    # a level.
    out["sc_beats_mscc_under_dgp1"] = float(mse["SC"] < mse["MSCc"])
    return out


# The reference and mlsynth fit identical panels, so ``att_max_gap_vs_octave``
# and ``stat_max_rel_gap_vs_octave`` carry no Monte Carlo component; they are
# two solvers on one problem. Everything with "size", "power" or "rate" in its
# name is a proportion over 120 draws, where a rate near 0.05 has a standard
# error of 0.02 and one near 0.5 has 0.046.
EXPECTED = {
    "n_draws": (360.0, 0.0),
    "n_per_dgp": (120.0, 0.0),
    # Core Octave's qp against mlsynth's CLARABEL solves, same panels.
    "att_max_gap_vs_octave": (0.0, 3e-4),
    "stat_max_rel_gap_vs_octave": (0.002, 0.02),
    # Size under DGP1 is the nominal level itself, no published number needed.
    "size_A95": (0.05, 0.06),
    "size_B95": (0.05, 0.06),
    "size_J95": (0.05, 0.06),
    "size_A90": (0.10, 0.08),
    "size_B90": (0.10, 0.08),
    "size_A50": (0.50, 0.15),
    "size_B50": (0.50, 0.15),
    "size_max_gap_to_nominal": (0.06, 0.10),
    # Each DGP violates one restriction; that test fires and the other does not.
    "power_dgp2_A": (0.98, 0.10),
    "power_dgp2_B": (0.05, 0.15),
    "dgp2_fires_on_the_right_test": (1.0, 0.0),
    "power_dgp3_B": (0.68, 0.18),
    "power_dgp3_A": (0.05, 0.15),
    "dgp3_fires_on_the_right_test": (1.0, 0.0),
    "power_gap_dgp2_over_dgp3": (0.32, 0.22),
    "joint_power_dgp2": (0.98, 0.12),
    "joint_power_dgp3": (0.90, 0.20),
    "rate_max_gap_vs_octave": (0.05, 0.12),
    # Tables 4 and 5 under DGP1.
    "mse_ratio_SC": (0.41, 0.25),
    "mse_ratio_MSCa": (0.49, 0.25),
    "mse_ratio_MSCb": (0.90, 0.25),
    "mse_ratio_MSCc": (1.0, 0.0),
    "sc_beats_mscc_under_dgp1": (1.0, 0.0),
}
