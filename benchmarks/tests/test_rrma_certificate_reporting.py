"""The RRMA study counts every solve whose certificate failed, and every fit its data
did not identify, and reports both.

``solver_pairfree.CertificateWarning`` names how many problems of a batch
failed the duality-gap certificate. Before this, a failure bumped a module
counter that no caller read, and ``one_rep`` silenced every warning, so the
study's tables could not show whether any of their numbers came from an
uncertified solve (Nocedal and Wright 2006, p. 370: a solver's outcomes are
explicit). ``UnidentifiedFitWarning`` does the same for fits whose ``beta`` the
data identify only in part. These tests pin the path from the warnings to the
results file.
"""
from __future__ import annotations

import warnings

import numpy as np

from benchmarks.studies.rrma_ablation import lp as LP
from benchmarks.studies.rrma_ablation import run as R
from benchmarks.studies.rrma_ablation import simulate as S
from benchmarks.studies.rrma_ablation import solver_pairfree as PF
from benchmarks.studies.rrma_ablation.solver_pairfree import CertificateWarning, UnidentifiedFitWarning


class TestCounting:
    def test_it_sums_the_problems_each_warning_names_and_silences_the_rest(self):
        def noisy():
            warnings.warn(CertificateWarning("three", 3))
            warnings.warn("unrelated", UserWarning)
            warnings.warn(CertificateWarning("two", 2))
            warnings.warn(UnidentifiedFitWarning("four", 4))
            return 7

        before = list(warnings.filters)
        with warnings.catch_warnings(record=True) as outside:
            warnings.simplefilter("always")
            assert S.counting(noisy) == (7, dict(uncertified=5, unidentified=4))
        assert outside == []
        assert warnings.filters == before

    def test_a_clean_call_counts_zero(self):
        assert S.counting(lambda x: x + 1, 1) == (2, dict(uncertified=0, unidentified=0))


class TestOneReplication:
    def test_a_real_replication_reports_zero(self):
        diag = S.one_rep("b", 60, 30, 25, 3)["diag"]
        assert diag["uncertified"] == 0 and diag["unidentified"] == 0

    def test_a_failed_certificate_reaches_the_diagnostics(self, monkeypatch):
        monkeypatch.setattr(PF, "certify", lambda y, X, h, beta, phi: np.zeros(len(y), bool))
        monkeypatch.setattr(LP, "_gap_ok", lambda *a, **k: False)
        assert S.one_rep("b", 60, 30, 25, 3)["diag"]["uncertified"] > 0


def _row(case, rep, uncertified):
    rng = np.random.default_rng(rep)
    return dict(case=case, rep=rep, err={a: float(rng.normal()) for a in S.ARMS},
                sec={"RRMA": 1.0}, diag=dict(d=5, ties=0, m_rrma=2, S_rrma=3, m_level=2, m_ls_avg=2,
                                             w_rrma=[1.0], b_fallbacks=0, tau_fallbacks=0,
                                             rank_one_mallows=True, strong_in_screen=4,
                                             uncertified=uncertified, unidentified=0))


class TestTheSummary:
    def test_the_case_summary_totals_the_count(self):
        rows = [_row("a", r, u) for r, u in enumerate([0, 2, 0, 1])]
        out = R.summarise_case(rows, "a", np.random.default_rng(0))
        assert out["diag"]["uncertified"] == 3
        assert out["diag"]["reps_with_uncertified"] == 2


class TestPathA:
    def test_the_donor_order_job_returns_its_counts(self):
        att, n = R._perm_job(0)
        assert np.isfinite(att) and n == dict(uncertified=0, unidentified=0)

    def test_the_full_selection_on_basque_runs_and_counts_what_it_could_not_identify(self):
        """Algorithm 1 at ``m = 6`` puts 10 donors in the last group, and GWCV fits
        it on windows of 7 to 10 pre-periods: ``beta`` is not identified there.
        With the NumPy solvers this raised an opaque error until ``fit`` took over."""
        (key, seed, m, att), n = R._seed_job(("regionno", 0))
        assert np.isfinite(att) and m in range(2, 8)
        assert n["uncertified"] == 0 and n["unidentified"] > 0


class TestTheTauOffReading:
    def test_it_weighs_on_the_simplex_with_unit_pair_weights(self):
        """``run._no_tau`` is the "tau off" reading of Path A: Section 2.4's
        programme with every pair weighted 1. It called a HiGHS helper the oracle
        no longer has, which would have stopped Path A partway through."""
        from benchmarks.studies.rrma_ablation import highs_reference as HR

        rng = np.random.default_rng(3)
        y = np.cumsum(rng.normal(size=12))
        Pm = y[:, None] + rng.normal(size=(12, 3))
        w, ok = R._no_tau(y, Pm)
        i, j = np.triu_indices(12, 1)
        ref = HR.simplex_wlad(y[i] - y[j], Pm[i] - Pm[j], np.ones(len(i)))
        obj = lambda b: np.sum(np.abs((y[i] - y[j]) - (Pm[i] - Pm[j]) @ b))
        assert ok and np.all(w >= 0) and abs(w.sum() - 1) <= 1e-12
        assert obj(w) <= obj(ref) * (1 + 1e-9)
