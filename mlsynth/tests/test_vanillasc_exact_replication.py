"""VanillaSC(backend="exact").fit() on Proposition 99 and West Germany against the authors' run.

The reference is a live run of ``code/scripts/run_all.py`` of
lychemon/synth-AS-replication at commit 791a3865 (v1.0.0), captured in
``benchmarks/reference/vanillasc_exact/``: its own verification reports every paper
number reproduced, and the bundle records the 479 numbers it writes. The fits and the
comparison live in ``benchmarks/cases/vanillasc_exact.py``; here every metric of that
case must sit inside its tolerance, and the paper's printed numbers (the values
``verify_results.py`` checks) are asserted as printed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.cases.vanillasc_exact import CA_PREDICTORS, EXPECTED, GE_PREDICTORS, _fits, measure

BUNDLE = Path(__file__).resolve().parents[2] / "benchmarks" / "reference" / "vanillasc_exact"


@pytest.fixture(scope="module")
def fits():
    return _fits()


@pytest.fixture(scope="module")
def measured(fits):
    return measure(fits)


def test_the_reference_is_the_authors_verified_run():
    out = (BUNDLE / "reference.out").read_text()
    assert "791a3865fe3e9fd0b82a8121c0b338791b7ffa39" in out
    assert "RESULT: every reported number reproduced" in out


@pytest.mark.parametrize("metric", sorted(EXPECTED))
def test_vanillasc_agrees_with_run_all(measured, metric):
    want, tol = EXPECTED[metric]
    assert metric in measured
    assert abs(measured[metric] - want) <= tol, (metric, measured[metric])


def _printed(got, paper, tol):
    """verify_results.py's rule: the four decimals run_all.py writes, within tol of the printed value."""
    return abs(round(float(got), 4) - paper) <= tol + 1e-9


def _shares(res, spec, vec):
    return np.asarray(vec)[[list(res.exact.predictor_names).index(c) for c in spec]]


def _weights_ok(res, paper):
    w = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
    return all(_printed(w.get(d, 0.0), v, 5e-4) for d, v in paper.items())


def test_tables_1_and_5_and_d_as_printed(fits):
    """The donor weights and losses the paper prints, under verify_results.py's tolerances
    (5e-4 on numbers printed to three decimals, 5e-3 to two, 0.5 to none)."""
    assert _weights_ok(fits["california", "diag"], {"Colorado": 0.160, "Connecticut": 0.068, "Montana": 0.202,
                                                    "Nevada": 0.236, "Utah": 0.335})
    assert _weights_ok(fits["california", "pd"], {"Utah": 0.319, "Nevada": 0.248, "Montana": 0.302,
                                                  "Colorado": 0.028, "Connecticut": 0.102})
    assert _weights_ok(fits["germany", "diag"], {"Austria": 0.507, "Switzerland": 0.166, "Japan": 0.146,
                                                 "USA": 0.135, "Australia": 0.045})
    assert _weights_ok(fits["germany", "pd"], {"Austria": 0.191, "Switzerland": 0.301, "Japan": 0.206,
                                               "Netherlands": 0.133, "Norway": 0.161, "Belgium": 0.009})
    assert abs(fits["california", "diag"].exact.validation_loss - 58.46) <= 5e-3
    assert abs(fits["california", "pd"].exact.validation_loss - 54.70) <= 5e-3
    assert abs(fits["germany", "diag"].exact.validation_loss - 45804) <= 0.5
    assert abs(fits["germany", "pd"].exact.validation_loss - 20287) <= 0.5


def test_the_mean_gaps_as_printed(fits):
    assert abs(fits["california", "diag"].effects.att - (-19.0)) <= 0.05
    assert abs(fits["california", "pd"].effects.att - (-19.8)) <= 0.05
    assert abs(fits["germany", "diag"].effects.att - (-1391)) <= 0.5
    assert abs(fits["germany", "pd"].effects.att - (-1582)) <= 0.5


@pytest.mark.parametrize("case,spec,field,paper", [
    ("california", CA_PREDICTORS, "diagonal.logdet", [0.0, 1 / 3, 1 / 3, 0.0, 0.0, 0.0, 1 / 3]),
    ("california", CA_PREDICTORS, "pd.logdet_shares", [0.006, 0.143, 0.144, 0.145, 0.224, 0.147, 0.191]),
    ("california", CA_PREDICTORS, "pd.center_shares", [0.030, 0.026, 0.036, 0.034, 0.563, 0.031, 0.279]),
    ("germany", GE_PREDICTORS, "diagonal.center", [0.443, 0.134, 0.071, 0.002, 0.107, 0.243]),
    ("germany", GE_PREDICTORS, "pd.logdet_shares", [0.205, 0.173, 0.150, 0.084, 0.221, 0.167]),
    ("germany", GE_PREDICTORS, "pd.center_shares", [0.176, 0.102, 0.069, 0.125, 0.462, 0.067]),
])
def test_tables_2_4_and_6_as_printed(fits, case, spec, field, paper):
    """The paper's predictor weights. Its Table 4 prints 0.145 for beer, which is the authors' 0.144465
    rounded to four decimals and then to three; verify_results.py checks the four decimals."""
    res = fits[case, "diag"]
    obj, attr = field.split(".")
    got = _shares(res, spec, getattr(getattr(res.exact, obj), attr))
    assert all(_printed(g, p, 5e-4) for g, p in zip(got, paper))


def test_the_smallest_eigenvalue_and_income_range_as_printed(fits):
    ca = fits["california", "diag"]
    assert abs(1000 * ca.exact.pd.logdet_min_eig - 3.7) <= 0.05
    assert _printed(ca.exact.pd.range_hi[list(ca.exact.predictor_names).index("lnincome")], 0.038, 5e-4)


def test_table_3_as_printed(fits):
    fc = fits["california", "diag"].exact.face_counts
    assert list(zip(fc.attainable, fc.visible, fc.total)) == [
        (0, 25, 34), (6, 183, 416), (29, 620, 2271), (59, 1129, 6235),
        (69, 1143, 8983), (49, 607, 6489), (14, 132, 1854)]
