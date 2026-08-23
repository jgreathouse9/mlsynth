"""Per-step cross-validation of mlsynth's SBC against Shi-Xi-Xie's own R code.

These tests pin each step of mlsynth's Synthetic Business Cycle estimator to a
golden value captured from the authors' actual functions (Germany.R: the ``lsq``
Hamilton detrending, the ``trend_predict`` forecast, and ``Synth::synth``) run on
the authoritative ``basedata/repgermany.dta``. The golden block lives in
``benchmarks/reference/sbc_germany/golden_steps.txt`` (regenerate with
``Rscript benchmarks/reference/sbc_germany/golden_steps.R``); the tests read it,
so no R toolchain is needed at test time.

How tight the comparison is, and why
------------------------------------

Two things bound it, and both are measured instead of chosen.

The first is the fixture. It is decimal text, so a comparison against it can
never be tighter than the precision the text carries. ``golden_steps.R`` emits
every compared quantity twice -- the legacy eight-decimal form and a
full-precision ``hi:`` form at ``%.17g`` -- and each recorded number is compared
at one unit in its own last recorded place (:func:`_ulp_of`). The eight-decimal
form resolves 1e-8 on values of order 1e2 to 1e4, three orders coarser than the
difference below, so on its own it reports a 5e-9 agreement that is the decimal
grid of ``sprintf("%.8f")`` and not a measurement of anything.
:func:`test_golden_fixture_resolves_eight_significant_digits` keeps that floor
from getting coarser still.

The second is the two implementations. Against the full-precision capture they
differ by 1.7e-14 of each series' scale -- about 1e-11 in absolute terms, uniform
across the trend coefficients, both sets of cycles and the forecast. That is the
distance between R's ``lm`` QR (LINPACK ``dqrls``) and numpy's ``lstsq``
(LAPACK ``gelsd``) on the same design, so the steps agree as closely as two
least-squares kernels can be expected to, and exact equality is not available at
any capture precision. ``CROSS_IMPL_FLOOR`` sets the comparison at 1e-12 of
scale, fifty-eight times the measured difference and four orders tighter than
what the eight-decimal capture could support.

Findings this harness locks in (see docs/replications/sbc.rst):

* mlsynth's Hamilton detrending and trend forecast reproduce the authors'
  ``lsq`` / ``trend_predict`` to 1.7e-14 of the series scale -- the distance
  between the two least-squares kernels.
* The donor cycles are detrended on the full sample and the treated unit on the
  pre window, matching the authors' code.
* The only divergence is the synthetic-control solver. The cyclical simplex
  least-squares is strictly convex and well-conditioned (donor matrix full rank
  16/16, Gram condition number ~3.8e3), so its optimum is unique. mlsynth's
  FISTA solve is certified to be within 1.4e-6 (relative) of that optimum by its
  own convexity bound, and cvxpy's ECOS and CLARABEL land on the same point to
  2.7e-6 in the weights. The authors' ``Synth::synth`` (kernlab ``ipop``) is the
  outlier (SSE ~1.299e6, +2.6%), and tightening ipop's tolerances does not close
  the gap -- it converges to a suboptimal point. So mlsynth attains the verified
  global optimum and the reference solver does not; mlsynth is the more accurate
  implementation, not a divergent one.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.sbc_helpers.hamilton import fit_hamilton_filter
from mlsynth.utils.sbc_helpers.trend_forecast import forecast_treated_trend
from mlsynth.utils.bilevel.simplex import simplex_lstsq

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "benchmarks" / "reference" / "sbc_germany" / "golden_steps.txt"
DATA = ROOT / "basedata" / "german_reunification.csv"

H, P = 4, 2

# Keys compared elementwise against R. Each must resolve its values to at least
# this many significant digits, since no comparison can be tighter than the
# precision the fixture carries.
COMPARED_KEYS = ("treated_trend_coef", "treated_cycle_pre", "trend_forecast",
                 "donor_cycle_full:Netherlands", "donor_cycle_full:Greece",
                 "donor_cycle_full:Italy")
MIN_SIGNIFICANT_DIGITS = 8

# mlsynth against R on the same design, measured against the full-precision
# capture: 1.7e-14 of each series' scale (~1e-11 absolute), uniform across the
# trend coefficients, the treated and donor cycles and the forecast. It is the
# difference between R's lm QR (LINPACK dqrls) and numpy's lstsq (LAPACK gelsd),
# so no capture precision makes the two exactly equal. The comparison is set
# fifty-eight times above it, which leaves room for a different BLAS while still
# failing on a defect three orders smaller than the old 1e-5 band could see.
CROSS_IMPL_FLOOR = 1e-12


def _ulp_of(token: str) -> float:
    """One unit in ``token``'s last recorded place.

    ``"610.16314485"`` gives 1e-8; ``"1.2661625755613e+06"`` gives 1e-7. This is
    the tightest tolerance the recorded text can support: a value written to
    eight decimals says nothing about the ninth, so asserting past it would be
    asserting against rounding noise.
    """
    mantissa, _, exponent = token.strip().lower().partition("e")
    decimals = len(mantissa.partition(".")[2])
    return 10.0 ** (int(exponent or 0) - decimals)


def _load_golden() -> dict:
    """Parse the fixture into ``{key: (values, tolerances)}``.

    ``golden_steps.R`` emits every compared quantity twice: once in the legacy
    eight-decimal form and once under a ``hi:`` prefix at full double precision.
    The ``hi:`` capture wins where it exists, so regenerating the fixture
    tightens every test here without touching this file.
    """
    if not GOLDEN.exists():
        pytest.skip(f"golden fixture missing: {GOLDEN}")
    parsed: dict = {}
    for line in GOLDEN.read_text().splitlines():
        if "\t" not in line:
            continue
        key, val = line.split("\t", 1)
        tokens = [t for t in val.split(",") if t.strip()]
        values = np.array([float(t) for t in tokens])
        tols = np.array([_ulp_of(t) for t in tokens])
        if key.startswith("hi:"):
            parsed[key[3:]] = (values, tols)
        elif key not in parsed:
            parsed[key] = (values, tols)
    return parsed


def _values(golden: dict, key: str) -> np.ndarray:
    return golden[key][0]


def _scalar(golden: dict, key: str) -> float:
    return float(golden[key][0][0])


def _assert_reproduces(ours: np.ndarray, golden: dict, key: str) -> None:
    """mlsynth matches R on ``key``, at whichever bound is the real one.

    Either the fixture's own resolution or the distance between the two
    least-squares kernels, whichever is coarser -- asserting past the first
    would test rounding noise, and asserting past the second would test which
    LAPACK driver is installed.
    """
    ref, tol = golden[key]
    tol = np.maximum(tol, CROSS_IMPL_FLOOR * np.max(np.abs(ref)))
    ours = np.asarray(ours, dtype=float)[: len(ref)]
    delta = np.abs(ours - ref)
    worst = int(np.argmax(delta - tol))
    assert np.all(delta <= tol), (
        f"{key}[{worst}]: mlsynth {ours[worst]!r} vs R {ref[worst]!r}; "
        f"delta {delta[worst]:.3e} exceeds {tol[worst]:.1e}"
    )


@pytest.fixture(scope="module")
def golden() -> dict:
    return _load_golden()


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not DATA.exists():
        pytest.skip(f"data missing: {DATA}")
    return pd.read_csv(DATA).pivot(index="year", columns="country", values="gdp")


@pytest.fixture(scope="module")
def treated_fit(panel):
    years = list(panel.index)
    T0 = years.index(1990) + 1
    return fit_hamilton_filter(panel["West Germany"].to_numpy()[:T0], h=H, p=P), T0


@pytest.fixture(scope="module")
def cyclical(panel, treated_fit):
    """The cyclical program the authors solve: treated cycle and donor cycles
    over the effective pre-treatment window."""
    fit, _ = treated_fit
    idx = np.where(~np.isnan(fit.cycle_pre))[0]
    donors = [c for c in panel.columns if c != "West Germany"]
    c0 = np.column_stack(
        [fit_hamilton_filter(panel[c].to_numpy(), h=H, p=P).cycle_pre[idx]
         for c in donors]
    )
    return fit.cycle_pre[idx], c0, donors


def test_golden_fixture_resolves_eight_significant_digits(golden):
    """The fixture's own precision is the ceiling on every comparison here.

    Each compared value is asserted at one unit in its last recorded place, so a
    capture written with fewer digits would silently loosen every test in this
    file while all of them still passed. That is the failure this test exists to
    make impossible: it fails when the fixture stops resolving eight significant
    digits, before any comparison against it can go slack.

    ``synth_loose_*`` is exempt: those record the authors' solver output for the
    inequality below, and are never compared elementwise.
    """
    for key in COMPARED_KEYS:
        values, tols = golden[key]
        digits = np.array([
            math.floor(math.log10(abs(v))) + 1 - math.floor(math.log10(t))
            for v, t in zip(values, tols) if v != 0.0
        ])
        assert digits.min() >= MIN_SIGNIFICANT_DIGITS, (
            f"{key} resolves only {digits.min()} significant digits; "
            f"comparisons against it cannot be tighter than that"
        )


def test_hamilton_treated_trend_coefficients(golden, treated_fit):
    """Step 1: the treated unit's Hamilton AR coefficients match R's ``lsq``."""
    fit, _ = treated_fit
    _assert_reproduces(fit.coefficients, golden, "treated_trend_coef")


def test_hamilton_treated_cycle(golden, treated_fit):
    """Step 1: the treated pre-treatment cyclical component matches R's ``lsq``."""
    fit, _ = treated_fit
    _assert_reproduces(fit.cycle_pre[~np.isnan(fit.cycle_pre)],
                       golden, "treated_cycle_pre")


@pytest.mark.parametrize("donor", ["Netherlands", "Greece", "Italy"])
def test_hamilton_donor_cycle_full_sample(golden, panel, donor):
    """Step 1: donor cycles are detrended on the FULL sample, matching R."""
    cyc = fit_hamilton_filter(panel[donor].to_numpy(), h=H, p=P).cycle_pre
    _assert_reproduces(cyc[~np.isnan(cyc)], golden, f"donor_cycle_full:{donor}")


def test_trend_forecast(golden, panel, treated_fit):
    """Step 2: the post-treatment trend extrapolation matches R's ``trend_predict``."""
    fit, T0 = treated_fit
    y = panel["West Germany"].to_numpy()
    fc = forecast_treated_trend(y_target=y, treated_fit=fit, T0=T0, horizon=H)
    _assert_reproduces(fc, golden, "trend_forecast")


# mlsynth's attained cyclical sum of squares on this panel, at full precision.
# The estimator is deterministic, so this is a value, not a target: it is
# reproduced bit for bit across runs. The tolerance covers floating-point
# differences between platforms, six orders above the 3.5e-12 weight movement
# measured under 1e-12 relative jitter in the cyclical inputs, and twenty times
# tighter than the 1.02 shift a solver truncated to 200 iterations produces.
ATTAINED_CYCLICAL_SSE = 1266162.5755613
SSE_TOL = 0.05

# The authors' Synth::synth (kernlab ipop) lands at ~1.299e6 at any tolerance.
# The gap is the solver's, not the model's.
OPTIMALITY_CERTIFICATE = 1e-5


def test_simplex_solve_is_feasible(cyclical):
    """The returned weights are on the simplex, exactly where exactness is
    available: non-negative, and summing to one to floating-point precision."""
    c1, c0, _ = cyclical
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = simplex_lstsq(c0, c1)
    assert w.min() >= 0.0
    assert abs(w.sum() - 1.0) <= 1e-12


def test_simplex_solve_is_certified_optimal(cyclical):
    """Step 3: mlsynth's solve is certified optimal, not merely pinned.

    The cyclical program is convex, so linearising at the returned point gives a
    lower bound on every feasible point's objective:
    ``f(w) >= f(w*) + g'(w - w*)``, minimised over the simplex by putting all
    mass on the smallest gradient coordinate. The distance between the attained
    objective and that bound is a certificate -- it proves how close the solve is
    to the global optimum without trusting any recorded number. mlsynth certifies
    to 1.4e-6 relative here; the assertion allows 1e-5.
    """
    c1, c0, _ = cyclical
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = simplex_lstsq(c0, c1)
    attained = float(np.sum((c1 - c0 @ w) ** 2))
    gradient = 2.0 * (c0.T @ (c0 @ w - c1))
    lower_bound = attained + float(gradient.min() - gradient @ w)
    assert (attained - lower_bound) / attained <= OPTIMALITY_CERTIFICATE


def test_the_certificate_moves_when_the_solve_does_not_converge(cyclical):
    """The certificate is only worth asserting if a bad solve fails it.

    A solver stopped after five iterations is nowhere near the optimum, and the
    certificate has to say so -- otherwise the assertion above passes because
    the instrument is flat, not because the solve is good. This is the check the
    attained-objective pin cannot make on its own: the objective is quadratic at
    its minimum, so a badly under-converged point can sit close in objective
    while the gradient says plainly that it is not optimal.
    """
    c1, c0, _ = cyclical
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = simplex_lstsq(c0, c1, max_iter=5, tol=0.0)
    attained = float(np.sum((c1 - c0 @ w) ** 2))
    gradient = 2.0 * (c0.T @ (c0 @ w - c1))
    lower_bound = attained + float(gradient.min() - gradient @ w)
    gap = (attained - lower_bound) / attained
    assert gap > 100 * OPTIMALITY_CERTIFICATE


def test_simplex_solve_reproduces_its_attained_objective(cyclical):
    """Step 3: the attained objective is pinned at the precision it is written,
    so a solver that stops short of convergence fails here."""
    c1, c0, _ = cyclical
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = simplex_lstsq(c0, c1)
    attained = float(np.sum((c1 - c0 @ w) ** 2))
    assert abs(attained - ATTAINED_CYCLICAL_SSE) <= SSE_TOL


def test_simplex_solve_beats_the_authors_synth(golden, cyclical):
    """Step 3: mlsynth's objective is strictly lower than the authors' ipop solve
    on the same program -- the one place the two implementations differ."""
    c1, c0, _ = cyclical
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = simplex_lstsq(c0, c1)
    assert float(np.sum((c1 - c0 @ w) ** 2)) < _scalar(golden, "synth_loose_sse")


# mlsynth's deterministic SBC output on the German panel, at full precision.
# Same tolerance basis as ATTAINED_CYCLICAL_SSE: 1e-6 relative.
ATT_1991_1994 = -952.2015762823139
CYCLE_WEIGHTS = {"Greece": 0.43651006181125107,
                 "Netherlands": 0.3652535087084823,
                 "Italy": 0.1551787869337406}
RELATIVE_TOL = 1e-6


def test_sbc_estimator_reaches_optimum_att(panel):
    """End-to-end: the SBC estimator reproduces the verified-optimum fit, not the
    authors' under-converged Synth fit (ATT ~-1006, Netherlands-dominant).

    The paper reports this application in figures only, so there is no printed
    number to match; what the authors' code produces is pinned separately in
    ``benchmarks/reference/sbc_germany/``. These values are mlsynth's own, and
    the estimator is deterministic, so they are pinned as values.
    """
    from mlsynth import SBC

    d = pd.read_csv(DATA)
    d["treat"] = ((d["country"] == "West Germany") & (d["year"] >= 1991)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SBC({"df": d, "outcome": "gdp", "treat": "treat", "unitid": "country",
                   "time": "year", "h": H, "p": P, "weights_mode": "simplex",
                   "display_graphs": False}).fit()
    assert abs(res.att - ATT_1991_1994) <= abs(ATT_1991_1994) * RELATIVE_TOL
    w = res.weights_by_donor or {}
    for donor, expected in CYCLE_WEIGHTS.items():
        assert abs(w.get(donor, 0.0) - expected) <= expected * RELATIVE_TOL
    # The verified optimum concentrates on Greece, the Netherlands and Italy.
    assert w["Greece"] > w["Netherlands"] > w["Italy"] > 0.05
