"""The regularisation can be fixed, so a fit can be reproduced and compared.

MC-NNM's answer is a function of one number: the singular-value threshold that
SOFT-IMPUTE shrinks against. mlsynth chose it by cross-validation and offered no
way to say otherwise, which costs two things (#484).

A published fit cannot be reproduced. Papers report the threshold they used, and
without a way to set it the estimator can only be run at whatever its own
cross-validation picks.

And a disagreement cannot be attributed. Against ``gsynth(estimator = "mc")`` on
the Ronczewski (2026) cannabis panel, mlsynth reports 0.01254 where gsynth
reports 0.00775. Two explanations fit -- a different threshold, or a different
estimator -- and with the threshold unpinnable neither can be ruled out. Fixing
the factor number is exactly what let GSYNTH's estimator be validated against
``gsynth`` to 4e-17 while its selection rule was set aside as its own question
(#483); MC-NNM had no equivalent handle.

The tuning is doing real work here, which is why this matters and not merely
tidies the interface. Refining the CV grid fivefold moves the estimate by 1e-05,
but changing the fold count moves it by 3e-04 and drops the recovered rank from
15 to 13.

``lam`` is an absolute threshold on the singular values of the demeaned outcome
matrix, so its scale is the outcome's. It is not comparable to a value reported
by another implementation on its own normalised grid, and the tests below fix
the scale by construction -- a threshold above the largest singular value must
annihilate the low-rank term -- instead of by comparison with a number from
somewhere else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import MCNNM
from mlsynth.exceptions import MlsynthConfigError


def panel(n_units=25, n_per=18, r_true=2, t0=12, n_treated=3, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n_per, r_true))
    L = rng.normal(size=(n_units, r_true))
    rows = []
    for u in range(n_units):
        for t in range(n_per):
            on = u < n_treated and t >= t0
            rows.append((u, t, float(F[t] @ L[u] + noise * rng.normal() + 2.0 * on),
                         int(on)))
    return pd.DataFrame(rows, columns=["id", "time", "y", "d"])


def fit(df, **over):
    return MCNNM({"df": df, "outcome": "y", "treat": "d", "unitid": "id",
                  "time": "time", "display_graphs": False, "inference": False,
                  **over}).fit()


class TestTheThresholdCanBeFixed:
    def test_a_given_lam_is_the_one_used(self):
        res = fit(panel(), lam=0.5)
        assert res.best_lambda == pytest.approx(0.5)

    def test_it_is_reported_as_the_callers_choice(self):
        assert fit(panel(), lam=0.5).lambda_source == "user"

    def test_cross_validation_stays_the_default(self):
        res = fit(panel())
        assert res.lambda_source == "cv"
        assert res.best_lambda > 0

    def test_two_fits_at_the_same_lam_agree_exactly(self):
        """No fold draw enters a fixed-threshold fit, so it is deterministic
        without reference to ``random_state``."""
        a = fit(panel(), lam=0.5, random_state=0)
        b = fit(panel(), lam=0.5, random_state=99)
        np.testing.assert_array_equal(a.counterfactual_matrix,
                                      b.counterfactual_matrix)
        assert float(a.effects.att) == float(b.effects.att)


class TestTheThresholdMeansWhatItSays:
    def test_a_larger_threshold_gives_a_lower_rank(self):
        """Monotone in the sense the estimator is defined by: shrinking harder
        keeps fewer singular values. Asserted as an ordering, not as counts,
        since the counts are a property of this panel."""
        ranks = [fit(panel(), lam=lam).rank for lam in (0.05, 0.5, 2.0)]
        assert ranks[0] >= ranks[1] >= ranks[2]

    def test_a_threshold_above_the_spectrum_annihilates_the_low_rank_term(self):
        """The scale is the outcome's, fixed here by construction and not by
        comparison with another implementation's normalised grid."""
        df = panel()
        wide = df.pivot(index="id", columns="time", values="y").to_numpy()
        smax = float(np.linalg.svd(wide - wide.mean(), compute_uv=False)[0])
        res = fit(df, lam=10.0 * smax)
        assert res.rank == 0
        assert np.allclose(res.L, 0.0, atol=1e-8)

    def test_a_threshold_of_zero_is_refused(self):
        """Zero is not "no regularisation" here, it is a degenerate fit.

        With no shrinkage SOFT-IMPUTE's threshold operator is the identity, so
        the low-rank term is fit on the observed cells and never propagates to
        the unobserved ones. The imputation is then the fixed effects alone.
        Measured on a panel whose true effect is 2.0::

            lam=0     rank=17  |L|max=3.465  att=1.797264522
            lam=0.01  rank=17  |L|max=3.469  att=1.997495412
            lam=50    rank= 0  |L|max=0.000  att=1.797264520

        Zero returns what annihilating the low-rank term returns, while
        reporting a rank-17 L. The CV grid runs from the spectral norm down to
        a thousandth of it and never reaches zero, so nothing could ask for
        this before ``lam`` existed.
        """
        with pytest.raises(MlsynthConfigError, match="greater than 0"):
            fit(panel(), lam=0.0)

    def test_the_cv_choice_can_be_replayed_by_fixing_it(self):
        """The point of the option: read the selected threshold off one fit and
        reproduce that fit exactly."""
        chosen = fit(panel())
        replayed = fit(panel(), lam=chosen.best_lambda)
        np.testing.assert_allclose(replayed.counterfactual_matrix,
                                   chosen.counterfactual_matrix,
                                   rtol=0, atol=1e-10)
        assert float(replayed.effects.att) == pytest.approx(
            float(chosen.effects.att), abs=1e-12)


class TestTheTuningIsRefusedOrHonouredExplicitly:
    """Every refusal here matches on its own message.

    ``extra="forbid"`` rejects an unknown ``lam`` with ``MlsynthConfigError``
    too, so a bare ``pytest.raises`` passes before the field exists and reports
    a missing feature as a working validator.
    """

    def test_a_negative_lam_is_refused(self):
        with pytest.raises(MlsynthConfigError, match="greater than 0"):
            fit(panel(), lam=-1.0)

    def test_giving_lam_and_a_grid_size_together_is_refused(self):
        """``n_lambda`` sizes a search that a fixed threshold does not run, so
        passing both states two incompatible intentions."""
        with pytest.raises(MlsynthConfigError, match="n_lambda"):
            fit(panel(), lam=0.5, n_lambda=80)

    def test_giving_lam_and_a_fold_count_together_is_refused(self):
        with pytest.raises(MlsynthConfigError, match="n_folds"):
            fit(panel(), lam=0.5, n_folds=10)

    def test_the_grid_size_is_still_free_when_lam_is_not_given(self):
        assert fit(panel(), n_lambda=20).lambda_source == "cv"


class TestInferenceFollowsTheFixedThreshold:
    def test_the_jackknife_refits_at_the_given_lam(self):
        """The jackknife holds the threshold fixed across its refits, and the
        threshold it holds must be the caller's when one was given."""
        res = MCNNM({"df": panel(), "outcome": "y", "treat": "d",
                     "unitid": "id", "time": "time", "display_graphs": False,
                     "inference": True, "lam": 0.5}).fit()
        assert res.best_lambda == pytest.approx(0.5)
        assert res.lambda_source == "user"
        assert np.isfinite(res.inference.standard_error)
