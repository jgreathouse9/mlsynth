"""What SYNDES's certificate reports when its search gives no proof (#822).

The certificate's bound is the mode's exact search's, and there are two ways it
falls short. The search can raise: no admissible design, a fit that did not
converge, or a two-way problem singular past its limits. Then there is no bound,
and the certificate says why. Or the search can stop at its node or work limit.
Then the bound is valid and the design is not proven optimal.

The bound is a diagnostic, never a correctness requirement: a failed bound
degrades to "no bound", through the certificate's own ``None`` contract, instead
of ending the fit.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.syndes_helpers.certificate import (
    SYNDESCertificate,
    design_objective,
    syndes_certificate,
)
from mlsynth.utils.syndes_helpers.per_unit_exact import solve_per_unit_exact

TWO_WAY = "global_2way"


def _Y(N=12, T=8, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, 3))
    Lam = rng.standard_normal((N, 3))
    return (Lam @ F.T).T + 0.3 * rng.standard_normal((T, N))


@pytest.fixture
def broken(monkeypatch):
    """Every search the certificate runs raises."""
    import mlsynth.utils.syndes_helpers.certificate as cert

    def boom(*a, **k):
        raise MlsynthEstimationError("the search broke")
    monkeypatch.setattr(cert, "_search", boom)


class TestASearchThatRaises:

    @pytest.mark.parametrize("mode", ["global_equal_weights", TWO_WAY, "per_unit"])
    def test_fields_are_none_and_uncertified(self, broken, mode):
        c = syndes_certificate(_Y(), 3, mode, 1.0)
        assert isinstance(c, SYNDESCertificate)
        assert c.lower_bound is None and c.optimality_gap is None
        assert c.certified is False and c.incumbent == 1.0

    def test_the_note_carries_the_searchs_reason(self, broken):
        c = syndes_certificate(_Y(), 3, "per_unit", 1.0)
        assert "the search broke" in c.note and "no lower bound" in c.note

    def test_a_singular_two_way_problem_past_its_limit_says_what_to_do(self):
        """Forty units of rank two over four periods at lam = 1e-14: singular,
        and 8.5e8 designs of ten, past the candidate limit."""
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((4, 2)) @ rng.standard_normal((2, 40))
        c = syndes_certificate(Y, 10, TWO_WAY, 1.0, lam=1e-14)
        assert c.lower_bound is None and c.certified is False
        assert "ill-conditioned" in c.note
        assert "lam" in c.note and "pre-treatment" in c.note


class TestASearchStoppedAtItsLimit:

    def test_a_stopped_search_gives_a_valid_bound_uncertified(self):
        Y = _Y(N=12, seed=22)
        exact = solve_per_unit_exact(Y, 5)
        with pytest.warns(Warning):
            stopped = solve_per_unit_exact(Y, 5, work_limit=exact.raw_results["work"] / 5)
        inc = design_objective(Y, "per_unit", stopped, stopped.lambda_value)
        c = syndes_certificate(Y, 5, "per_unit", inc, lam=stopped.lambda_value,
                               search=stopped)
        assert c.certified is False and "stopped at its limit" in c.note
        assert c.lower_bound <= exact.objective_value * (1 + 1e-12)
        assert c.optimality_gap > 0.0

    def test_a_two_way_search_stopped_at_its_work_limit_is_valid_uncertified(self):
        """Past the candidate limit the two-way search branches and bounds; at
        its work limit its bound stays valid and the design uncertified."""
        from mlsynth.utils.syndes_helpers.exact import solve_synthetic_design_exact
        Y = _Y(N=20, T=30, seed=3)
        with pytest.warns(Warning, match="work limit"):
            past = solve_synthetic_design_exact(Y, 8, candidate_limit=1,
                                                work_limit=1e-12)
        exact = solve_synthetic_design_exact(Y, 8)
        inc = design_objective(Y, TWO_WAY, past, past.lambda_value)
        c = syndes_certificate(Y, 8, TWO_WAY, inc, lam=past.lambda_value,
                               search=past)
        assert c.certified is False and "did not certify" in c.note
        assert c.lower_bound <= exact.objective_value
