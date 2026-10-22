"""Tests for MAREX's exact-search diagnostics (#774).

Contract. The exact backend reports what its search did, as typed fields: a
fixed-size summary per cluster (:class:`ExactSearchDiagnostics`) on each
``MAREXClusterDesign``, and their merge on ``MAREXResults.search``. Every
admissible candidate gets exactly one fate:

    cleared_by_bound + cleared_by_coupling + cleared_by_treated_fit
    + without_control + cleared_by_fit_cutoff + fitted == candidates

and each count is measured where the fate is decided, never by subtraction, so
the identity is a check. The clearance histogram holds, for each candidate the
walk discarded on its bound, ``(bound - cutoff) / |cutoff|`` in fixed log bins;
candidates discarded outside the buffer are counted with a lower bound on their
clearance. Summaries merge by addition, minimum or maximum. A bound that fell
back to zero, and a certificate with unresolved sets, warn.

The levels:

- the histogram bins, merges and reads quantiles (unit, property);
- the accounting identity holds over panels, sizes, designs and buffer sizes,
  and the treated fits counted are the treated fits run (property, unit);
- the fields marked order-independent do not move with the buffer (property);
- fallbacks, coupled-fit numerics, timing and the branch and bound's counts
  (unit, edge);
- the estimator carries the summaries, merged over clusters (smoke);
- a fallback and an unresolved certificate warn, and a search that raises
  keeps its counts (failure).
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from mlsynth import MAREX
from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.marex_helpers import exact as exact_mod
from mlsynth.utils.marex_helpers.config import MAREXConfig
from mlsynth.utils.marex_helpers.diagnostics import (
    CLEARANCE_EDGES,
    ExactSearchDiagnostics,
    LogHistogram,
    MAREXSearchWarning,
    SearchCounts,
    SearchReport,
)
from mlsynth.utils.marex_helpers.exact import (
    COUPLED_TOL,
    PRUNE_MARGIN,
    count_candidates,
    search_cluster,
)
from mlsynth.utils.syndes_helpers.exact import SubsetFilter

SETTINGS = dict(max_examples=20, deadline=None,
                suppress_health_check=[HealthCheck.too_slow])

FATES = ("cleared_by_bound", "cleared_by_coupling", "cleared_by_treated_fit",
         "without_control", "cleared_by_fit_cutoff", "fitted")


def _cluster(n=8, T=12, seed=0, dupe=False, twins_at_mean=False):
    """A cluster's markets ``B`` (``T`` by ``n``) and its target ``A``.

    ``dupe`` makes market 1 a copy of market 0. ``twins_at_mean`` makes both
    the mean of the others, so the cluster's mean is theirs too and each twin
    alone is an optimal treated set, tied with the other."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    B = f @ rng.uniform(0.4, 1.6, size=(2, n)) + rng.normal(size=(T, n)) + 20.0
    if dupe:
        B[:, 1] = B[:, 0]
    if twins_at_mean:
        B[:, 0] = B[:, 1] = B[:, 2:].mean(axis=1)
    return B, B.mean(axis=1)


def _search(B, A, sizes, *, design="standard", **kw):
    n = B.shape[1]
    if design == "penalized":
        rng = np.random.default_rng(1)
        kw.setdefault("linear_treated", rng.uniform(0, 2, n))
        kw.setdefault("linear_control", rng.uniform(0, 2, n))
    elif design == "weakly":
        kw.setdefault("beta", 1.0)
    report = SearchReport()
    designs = search_cluster(B, A, sizes, filt=kw.pop("filt", SubsetFilter()),
                             report=report, **kw)
    return designs, report.freeze("enumeration")


def _fates(d: ExactSearchDiagnostics) -> int:
    return sum(getattr(d.counts, f) for f in FATES)


# ----------------------------------------------------------------------
# the histogram
# ----------------------------------------------------------------------


class TestLogHistogram:
    def test_bins_run_four_a_decade_from_1e_minus_12_to_1e6(self):
        assert len(CLEARANCE_EDGES) == 73
        assert CLEARANCE_EDGES[0] == pytest.approx(1e-12)
        assert CLEARANCE_EDGES[-1] == pytest.approx(1e6)
        assert len(LogHistogram.empty().counts) == 74

    def test_values_land_in_their_bins(self):
        h = LogHistogram.of(np.array([0.0, 5e-13, 1e-12, 1.5e-12, 0.5, 1e6, np.inf, np.nan]))
        c = h.counts
        assert c[0] == 2                     # under 1e-12
        assert c[1] == 2                     # [1e-12, 10**-11.75)
        assert c[-1] == 3                    # 1e6 and past, and non-finite
        assert h.total == 8

    def test_weights_count_many_at_once(self):
        h = LogHistogram.of(np.array([1e-3]), weights=np.array([7]))
        assert h.total == 7

    def test_merge_adds(self):
        a = LogHistogram.of(np.array([1e-3, 1e-1]))
        b = LogHistogram.of(np.array([1e-3]))
        assert a.merge(b).counts == tuple(x + y for x, y in zip(a.counts, b.counts))

    @given(st.lists(st.floats(0, 1e8), max_size=30), st.lists(st.floats(0, 1e8), max_size=30))
    @settings(**SETTINGS)
    def test_merging_two_equals_binning_their_union(self, a, b):
        left = LogHistogram.of(np.array(a)).merge(LogHistogram.of(np.array(b)))
        assert left == LogHistogram.of(np.array(a + b))

    def test_quantiles_read_the_bin_upper_edges(self):
        h = LogHistogram.of(np.array([2e-9] * 3 + [2e-2]))   # off the edges
        assert h.quantile(0.5) == pytest.approx(10 ** -8.5)
        assert h.quantile(1.0) == pytest.approx(10 ** -1.5)
        assert np.isnan(LogHistogram.empty().quantile(0.5))
        assert LogHistogram.of(np.array([np.inf])).quantile(0.5) == np.inf


class TestMerge:
    def test_counts_add_and_the_second_pass_ors(self):
        a = SearchCounts(candidates=3, fitted=1, second_pass=False)
        b = SearchCounts(candidates=4, fitted=2, second_pass=True)
        m = a.merge(b)
        assert (m.candidates, m.fitted) == (7, 3)
        assert m.second_pass is True and m.merge(b).second_pass is True

    def test_diagnostics_merge_field_by_field(self):
        B, A = _cluster(seed=0)
        _, d1 = _search(B, A, [2])
        _, d2 = _search(*_cluster(seed=1), [3])
        m = d1.merge(d2)
        assert m.counts.candidates == d1.counts.candidates + d2.counts.candidates
        assert m.clearance.total == d1.clearance.total + d2.clearance.total
        assert m.numerics.bound_fallbacks == (d1.numerics.bound_fallbacks
                                              + d2.numerics.bound_fallbacks)
        assert m.timing.pricing == pytest.approx(d1.timing.pricing + d2.timing.pricing)
        assert m.search == "enumeration"

    def test_two_kinds_of_search_merge_as_mixed(self):
        a = ExactSearchDiagnostics.empty("enumeration")
        b = ExactSearchDiagnostics.empty("branch_and_bound")
        assert a.merge(b).search == "mixed"


class TestReport:
    def test_a_discard_records_its_clearance_against_the_cutoff(self):
        r = SearchReport()
        r.cleared(np.array([2.0, 4.0]), cutoff=1.0)            # clearances 1 and 3
        d = r.freeze("enumeration")
        assert d.counts.cleared_by_bound == 2
        assert d.clearance == LogHistogram.of(np.array([1.0, 3.0]))

    def test_a_discard_outside_the_buffer_keeps_a_lower_bound(self):
        r = SearchReport()
        r.cleared_outside(5, least_bound=1.5, cutoff=1.0)
        r.cleared_outside(0, least_bound=1.01, cutoff=1.0)     # nothing discarded
        d = r.freeze("enumeration")
        assert d.counts.cleared_by_bound == d.counts.cleared_outside_buffer == 5
        assert d.outside_buffer_clearance == pytest.approx(0.5)
        assert d.clearance.total == 0

    def test_the_largest_gap_is_kept_not_added(self):
        a = SearchReport(); a.coupled({"newton_steps": 2, "sweeps": 1, "gap": 2e-9}, 1.0)
        b = SearchReport(); b.coupled({"newton_steps": 1, "sweeps": 0, "gap": 6e-9}, 1.0)
        m = a.freeze("enumeration").merge(b.freeze("enumeration"))
        assert m.numerics.coupled_max_gap == pytest.approx(3e-9)
        assert (m.numerics.coupled_fits, m.numerics.coupled_steps) == (2, 3)


# ----------------------------------------------------------------------
# the accounting
# ----------------------------------------------------------------------


class TestAccounting:
    @given(seed=st.integers(0, 10 ** 6), n=st.integers(4, 9),
           design=st.sampled_from(["standard", "penalized", "weakly"]),
           buffer=st.sampled_from([1, 2, 8, 512]), data=st.data())
    @settings(**SETTINGS)
    def test_every_candidate_has_exactly_one_fate(self, seed, n, design, buffer, data):
        m = data.draw(st.integers(1, min(3, n - 1)))
        B, A = _cluster(n=n, seed=seed)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(exact_mod, "BUFFER", buffer)
            _, d = _search(B, A, [m], design=design)
        assert d.counts.candidates == count_candidates(n, [m], SubsetFilter())
        assert _fates(d) == d.counts.candidates
        assert d.clearance.total + d.counts.cleared_outside_buffer == d.counts.cleared_by_bound
        assert d.counts.priced == d.counts.candidates * (2 if d.counts.second_pass else 1)

    def test_the_weakly_targeted_stages_are_counted(self):
        """On this panel the hull-distance bound and the coupled fit's cutoff
        both discard candidates, and the identity still holds."""
        _, d = _search(*_cluster(n=9, seed=0), [3], design="weakly")
        assert d.counts.cleared_by_coupling > 0 and d.counts.cleared_by_fit_cutoff > 0
        assert _fates(d) == d.counts.candidates

    @given(seed=st.integers(0, 10 ** 6), buffer=st.sampled_from([2, 4, 8, 16]))
    @settings(**SETTINGS)
    def test_the_coupling_bound_discards_each_set_once(self, seed, buffer):
        """A set the hull-distance bound discarded in the first pass is not
        screened again in the second, where the cutoff is no larger, so its one
        fate is counted once and the outside-buffer count is not short."""
        discarded = []
        survivors = exact_mod.CouplingBound.survivors

        def spy(self, sets, lb, cutoff):
            keep = survivors(self, sets, lb, cutoff)
            discarded.extend(tuple(S) for S, k in zip(sets, keep) if not k)
            return keep

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(exact_mod, "BUFFER", buffer)
            mp.setattr(exact_mod.CouplingBound, "survivors", spy)
            _, d = _search(*_cluster(n=9, seed=seed), [3], design="weakly")
        assert len(discarded) == len(set(discarded)) == d.counts.cleared_by_coupling
        assert _fates(d) == d.counts.candidates

    def test_a_treated_fit_that_clears_the_incumbent_is_counted(self):
        """A penalty loosens the closed-form bound, so some candidates walked on
        it are cleared by their treated fit alone; on this panel, four."""
        _, d = _search(*_cluster(n=10, seed=4), [4], design="penalized")
        assert d.counts.cleared_by_treated_fit == 4
        assert _fates(d) == d.counts.candidates

    def test_a_small_buffer_forces_the_second_pass(self, monkeypatch):
        monkeypatch.setattr(exact_mod, "BUFFER", 2)
        _, d = _search(*_cluster(n=9, seed=3), [3])
        assert d.counts.second_pass
        assert _fates(d) == d.counts.candidates

    def test_a_large_buffer_holds_every_candidate(self):
        _, d = _search(*_cluster(n=8, seed=3), [2])
        assert not d.counts.second_pass
        assert d.counts.cleared_outside_buffer == 0

    @given(seed=st.integers(0, 10 ** 6), n=st.integers(8, 9),
           design=st.sampled_from(["standard", "penalized", "weakly"]))
    @example(seed=2, n=8, design="standard")
    @settings(**SETTINGS)
    def test_the_treated_fits_counted_are_the_fits_run(self, seed, n, design):
        """Every candidate the walk considers runs one treated-side fit, and one
        control fit when it is fitted outside the weakly-targeted design, whose
        coupled fit fits both sides at once."""
        calls = []
        real = exact_mod.side_fit

        def spy(B, A, cols, linear=None):
            calls.append(len(cols))
            return real(B, A, cols, linear)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(exact_mod, "side_fit", spy)
            _, d = _search(*_cluster(n=n, seed=seed), [3], design=design)
        c = d.counts
        considered = (c.cleared_by_treated_fit + c.without_control
                      + c.cleared_by_fit_cutoff + c.fitted)
        controls = 0 if design == "weakly" else c.fitted
        assert len(calls) == considered + controls
        assert sum(1 for k in calls if k == 3) == considered

    def test_the_walk_records_its_discards(self):
        """84 candidates fit in the buffer, so every discard is binned."""
        _, d = _search(*_cluster(n=9, seed=5), [3])
        assert d.clearance.total == d.counts.cleared_by_bound > 0

    @given(seed=st.integers(0, 10 ** 6), buffer=st.sampled_from([2, 8, 512]),
           design=st.sampled_from(["standard", "penalized", "weakly"]))
    @example(seed=5, buffer=512, design="standard")
    @settings(**SETTINGS)
    def test_no_discard_is_closer_than_the_margin(self, seed, buffer, design):
        """A candidate is discarded only when its bound clears the incumbent by
        the prune margin, so no clearance falls below it, inside the buffer or
        outside it."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(exact_mod, "BUFFER", buffer)
            _, d = _search(*_cluster(n=9, seed=seed), [3], design=design)
        assert d.outside_buffer_clearance >= PRUNE_MARGIN
        below = np.searchsorted(CLEARANCE_EDGES, PRUNE_MARGIN, side="right")
        assert sum(d.clearance.counts[:below]) == 0

    @given(seed=st.integers(0, 10 ** 6), n=st.integers(5, 9))
    @settings(**SETTINGS)
    def test_the_order_independent_fields_do_not_move_with_the_buffer(self, seed, n):
        B, A = _cluster(n=n, seed=seed)
        out = []
        for buffer in (1, 4, 512):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(exact_mod, "BUFFER", buffer)
                designs, d = _search(B, A, [2])
            out.append((d.order_independent(), designs[0].treated))
        assert out[0] == out[1] == out[2]


# ----------------------------------------------------------------------
# numerics, timing and the branch and bound
# ----------------------------------------------------------------------


class TestNumerics:
    def test_a_duplicated_market_makes_its_bounds_fall_back(self):
        n, m = 7, 2
        B, A = _cluster(n=n, seed=4, dupe=True)
        _, d = _search(B, A, [m])
        # every pair holding both copies has a singular Gram, and pass 1 prices it once
        assert d.numerics.bound_fallbacks >= 1
        assert d.numerics.bound_fallbacks <= count_candidates(n, [m], SubsetFilter())

    def test_fallbacks_are_counted_once_whatever_the_buffer(self, monkeypatch):
        B, A = _cluster(n=8, seed=4, dupe=True)
        _, wide = _search(B, A, [3])
        monkeypatch.setattr(exact_mod, "BUFFER", 1)
        _, narrow = _search(B, A, [3])
        assert narrow.counts.second_pass and not wide.counts.second_pass
        assert narrow.numerics.bound_fallbacks == wide.numerics.bound_fallbacks > 0

    def test_a_clean_panel_has_no_fallback(self):
        _, d = _search(*_cluster(n=8, seed=0), [2])
        assert d.numerics.bound_fallbacks == 0

    def test_coupled_fits_report_their_steps_and_gap(self):
        _, d = _search(*_cluster(n=7, seed=1), [2], design="weakly")
        assert d.numerics.coupled_fits == d.counts.fitted
        assert d.numerics.coupled_steps >= 0 and d.numerics.coupled_sweeps >= 0
        assert 0.0 <= d.numerics.coupled_max_gap <= COUPLED_TOL

    def test_the_other_designs_run_no_coupled_fit(self):
        _, d = _search(*_cluster(n=7, seed=1), [2])
        assert d.numerics.coupled_fits == 0

    def test_time_is_counted_and_never_negative(self):
        _, d = _search(*_cluster(n=8, seed=0), [3])
        assert d.timing.pricing > 0.0 and d.timing.fitting >= 0.0
        assert d.timing.branch_and_bound == 0.0


def _frame(n=9, T=14, seed=0, regions=None, dupe=False, twins_at_mean=False):
    B, _ = _cluster(n=n, T=T, seed=seed, dupe=dupe, twins_at_mean=twins_at_mean)
    labels = [f"u{j}" for j in range(n)]
    df = pd.DataFrame({"unit": np.repeat(labels, T), "time": np.tile(np.arange(T), n),
                       "y": B.T.reshape(-1)})
    if regions is not None:
        df["region"] = np.repeat(np.repeat(np.arange(len(regions)), regions), T)
    return df


def _fit(df, **over):
    cfg = dict(df=df, outcome="y", unitid="unit", time="time", T0=10,
               display_graph=False, inference=False, backend="exact", m_eq=2)
    cfg.update(over)
    return MAREX(MAREXConfig(**cfg)).fit()


class TestBranchAndBound:
    def test_the_branch_and_bound_reports_its_bounds_and_sets(self, monkeypatch):
        from mlsynth.utils.marex_helpers import branch as branch_mod
        found = []
        real = branch_mod.branch_and_bound

        def spy(*args, **kwargs):
            found.append(real(*args, **kwargs))
            return found[-1]

        monkeypatch.setattr(exact_mod, "CANDIDATE_LIMIT", 1)
        monkeypatch.setattr(branch_mod, "branch_and_bound", spy)
        d = _fit(_frame(n=8, seed=0)).search
        (out,) = found
        assert d.search == "branch_and_bound"
        assert d.counts.candidates == count_candidates(8, [2], SubsetFilter())
        assert (d.counts.bnb_bounds, d.counts.bnb_priced, d.counts.fitted) == (
            out.lifts, out.priced, out.certificate.fitted)
        assert d.counts.bnb_bounds + d.counts.bnb_priced >= 1
        assert d.timing.branch_and_bound > 0.0
        assert d.timing.pricing == d.timing.fitting == 0.0


# ----------------------------------------------------------------------
# the estimator
# ----------------------------------------------------------------------


class TestOnTheResult:
    def test_each_cluster_and_the_whole_design_carry_a_summary(self):
        res = _fit(_frame(n=10, regions=[5, 5]), cluster="region")
        per = [c.search for c in res.clusters.values()]
        assert all(isinstance(d, ExactSearchDiagnostics) for d in per)
        assert res.search.counts.candidates == sum(d.counts.candidates for d in per)
        assert res.search.counts.fitted == sum(d.counts.fitted for d in per)

    def test_the_mip_backend_carries_none(self):
        res = _fit(_frame(n=7), backend="mip")
        assert res.search is None
        assert all(c.search is None for c in res.clusters.values())

    def test_a_single_cluster(self):
        res = _fit(_frame(n=6))
        assert _fates(res.search) == res.search.counts.candidates


# ----------------------------------------------------------------------
# failure: what a user should act on, and what a raise leaves behind
# ----------------------------------------------------------------------


class TestFailure:
    def test_a_bound_that_fell_back_warns(self):
        with pytest.warns(MAREXSearchWarning, match="fell back"):
            _fit(_frame(n=7, seed=4, dupe=True))

    def test_a_clean_search_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", MAREXSearchWarning)
            _fit(_frame(n=8, seed=0))

    def test_an_unresolved_certificate_warns(self):
        """Two copies of one market tie, so the certificate cannot resolve
        the design against its twin."""
        with pytest.warns(MAREXSearchWarning, match="unresolved"):
            _fit(_frame(n=7, seed=4, twins_at_mean=True), m_eq=1, certify=True)

    def test_a_search_that_raises_keeps_its_counts(self):
        """Every treated market excludes every control: nothing is admissible,
        the search raises, and the report still says why."""
        n = 4
        B, A = _cluster(n=n, seed=0)
        donors = tuple((i, j) for i, j in itertools.permutations(range(n), 2))
        report = SearchReport()
        with pytest.raises(MlsynthEstimationError):
            search_cluster(B, A, [1], filt=SubsetFilter(), donor_exclusion=donors,
                           report=report)
        d = report.freeze("enumeration")
        assert d.counts.without_control == d.counts.candidates == n
