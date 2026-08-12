"""The autocovariance normalisation in the PDA long-run variance.

:func:`mlsynth.utils.pda_helpers.inference.hac_lrv` builds a Bartlett-kernel
long-run variance from sample autocovariances. There are two conventions for the
lag-``l`` autocovariance of a series of length ``n``, and they differ by a factor
of ``n / (n - l)``:

    gamma_l = (1/n)     sum_{t=1}^{n-l} z_t z_{t+l}      <- R's acf(), Newey-West
    gamma_l = (1/(n-l)) sum_{t=1}^{n-l} z_t z_{t+l}      <- the unbiased-looking one

The first is the standard one and the one this estimator has to use. Two reasons.

It is what the fixed-lag path claims to be. ``fs`` and ``hcw`` document their
``lrvar_lag`` branch as reproducing Shi and Huang's released ``fsPDA`` package,
whose long-run variance is built from ``acf(type = "covariance")`` -- and ``acf``
divides by ``n``. Under the other convention the reproduction is wrong in the
third decimal of the t-statistic, which is enough to move a borderline test.

It is also the convention that keeps the estimator a variance. Dividing by ``n``
makes the autocovariance sequence positive semi-definite, so the weighted sum is
a non-negative estimate of a non-negative quantity. Dividing by ``n - l`` gives
up that guarantee, and the clamp at zero then hides the failure instead of the
estimator not having one.

The values below come from R 4.3.3 and are reproduced by the script recorded in
the docstring of :func:`test_matches_acf_on_an_ar1_series`.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.pda_helpers.inference import hac_lrv

#: ``set.seed(11); arima.sim(model = list(ar = 0.6), n = 40, n.start = 1000)``
_AR1 = np.array([
    -0.49383220349874707, -0.66723203709181289, -0.82548544757059528,
    -0.054139746211601891, -0.70523420413579774, -0.0064463978301257385,
    -1.4016548531822202, -1.6878231520529341, -1.6387625924201252,
    -0.28424136711590853, -0.72686234969065344, 0.35720059388036263,
    0.35462672999192879, 2.7555250523704502, 0.95203404290548299,
    1.195959783188604, 0.32679019924177066, -1.6514155200964016,
    0.23872508715785068, 0.835141846419865, 0.99060788759723473,
    -0.29507739622423557, -1.6614760113371694, -1.9925718437261728,
    -1.3826909435362329, 0.0725965667617815, -1.0538542773461037,
    -1.0251488791794934, -0.51809487131179655, 1.8378231871232085,
    0.97058383528290526, 0.70143406506043449, 0.40637456372995001,
    0.1839654730556633, 0.70596320880148833, 2.5431686965743721,
    1.689862373907949, 1.6485553615346635, 0.35485651692637932,
    0.56389761355181323,
])


class TestTheConvention:
    def test_alternating_series_gets_the_acf_value(self):
        """``z = (1,-1,...)`` of length 6, hand-checkable.

        ``gamma_0 = 1`` and the five cross products are each ``-1``, so R's
        ``acf`` gives ``gamma_1 = -5/6`` and the Bartlett sum at ``L = 1`` is
        ``1 + 2 (1/2) (-5/6) = 1/6``. Dividing by ``n - l = 5`` instead gives
        ``gamma_1 = -1`` and a long-run variance of exactly zero -- a clean
        number that comes from the wrong denominator.
        """
        z = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        assert hac_lrv(z, lag=1, kernel="bartlett") == pytest.approx(
            1.0 / 6.0, rel=1e-12)

    @pytest.mark.parametrize("lag,expected", [
        (1, 2.1249099857258731),
        (2, 2.6732881038962653),
        (3, 2.9788289919615538),
    ])
    def test_matches_acf_on_an_ar1_series(self, lag, expected):
        """Against R, on a serially dependent series where the lags matter.

        Reference::

            set.seed(11)
            x <- as.vector(arima.sim(model = list(ar = 0.6), n = 40, n.start = 1000))
            for (h in c(1, 2, 3)) {
              g  <- as.vector(acf(x, lag.max = h, type = "covariance", plot = FALSE)$acf)
              wh <- 1 - (1:h) / (h + 1)
              cat(g[1] + 2 * sum(wh * g[-1]), "\\n")
            }
        """
        assert hac_lrv(_AR1, lag=lag, kernel="bartlett") == pytest.approx(
            expected, rel=1e-6)

    def test_the_gap_grows_with_the_lag(self):
        """The two conventions differ by ``n / (n - l)`` per lag, so a longer
        truncation makes the discrepancy larger, not smaller."""
        n = _AR1.size
        zc = _AR1 - _AR1.mean()

        def wrong(L):
            g0 = float(np.mean(zc * zc))
            return g0 + 2.0 * sum((1 - l / (L + 1.0)) * float(np.mean(zc[l:] * zc[:-l]))
                                  for l in range(1, L + 1))

        gaps = [abs(hac_lrv(_AR1, lag=L) - wrong(L)) / hac_lrv(_AR1, lag=L)
                for L in (1, 3, 6)]
        assert gaps == sorted(gaps)
        assert gaps[0] < 0.02 < gaps[-1]


class TestStillAVariance:
    def test_a_positive_series_stays_positive(self):
        rng = np.random.default_rng(3)
        z = rng.standard_normal(200)
        assert hac_lrv(z, lag=4) > 0

    def test_negative_raw_value_is_still_clamped(self):
        """The clamp stays; the alternating series under a uniform kernel still
        drives the raw sum below zero."""
        z = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        assert hac_lrv(z, lag=1, kernel="uniform") == 0.0

    @pytest.mark.parametrize("lag", [0, 1, 5])
    def test_a_constant_series_has_no_variance(self, lag):
        assert hac_lrv(np.full(20, 4.0), lag=lag) == pytest.approx(0.0, abs=1e-12)


class TestEdges:
    def test_empty_series(self):
        assert np.isnan(hac_lrv(np.array([])))

    def test_single_observation(self):
        assert hac_lrv(np.array([2.0]), lag=0) == pytest.approx(0.0, abs=1e-12)

    def test_lag_beyond_the_sample_is_truncated(self):
        """``min(L, n - 1)`` caps the loop; a lag past the sample cannot index
        past the end, and the weights still come from the requested ``L``."""
        z = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isfinite(hac_lrv(z, lag=99))
