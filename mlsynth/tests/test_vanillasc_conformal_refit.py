"""VanillaSC's test-inversion conformal mode, against the authors' own package.

``inference="conformal"`` inverts the sharp null of Chernozhukov, Wuthrich & Zhu
(2021). Their reference implementation is the ``scinference`` R package; with
``estimation_method = "sc"`` it refits a simplex synthetic control under each
candidate null. VanillaSC's point estimate is a simplex synthetic control too
(unless ``augment="ridge"`` is asked for), so its null refit has to be the same
control, and these tests pin that it is -- by reproducing ``scinference``'s
p-values on the Swedish carbon tax panel to six decimals.

The tests that matter most here are the power ones. A refit free to re-level the
treated series absorbs a post-period effect and spreads it over the pre-period
residuals the test calibrates against, and the p-value then stops responding to
the effect size entirely. That failure is invisible to a test that checks only
that the p-value is a number in ``[0, 1]``, which is why it is checked directly:
inject an effect, watch the p-value reach the floor of its reference set.

Reference: ``scinference`` @ ``567c688``, ``inference_method = "conformal"``,
``estimation_method = "sc"``, ``alpha = 0.1``, ``lsei_type = 1``. The run script
is ``benchmarks/reference/cwz_conformal/reference.R``.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.vanillasc_helpers.config import VanillaSCConfig

# scinference, moving-block scheme, on the panel built below.
SCINFERENCE_MB_NULL = 0.391304
SCINFERENCE_MB_FLOOR = 0.021739     # 1 / T, T = 46


@pytest.fixture(scope="module")
def carbontax():
    ct = pd.read_stata("basedata/carbontax_data.dta")
    ct["treat"] = ((ct.country == "Sweden") & (ct.year >= 1990)).astype(int)
    return ct


def _fit(df, *, effect=0.0, **kw):
    d = df.copy()
    if effect:
        d.loc[d.treat == 1, "CO2_transport_capita"] += effect
    cfg = dict(df=d, unitid="country", time="year",
               outcome="CO2_transport_capita", treat="treat",
               display_graphs=False, inference="conformal", seed=0)
    cfg.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(VanillaSCConfig(**cfg)).fit()


# --------------------------------------------------------------------------
# the permutation scheme has to be reachable
# --------------------------------------------------------------------------

def test_conformal_type_is_configurable(carbontax):
    """The moving-block scheme is the authors' default and must be selectable.

    Without this the config could only reach the i.i.d. scheme, which assumes
    exchangeable errors -- the assumption panel data usually breaks, and the one
    Chernozhukov, Wuthrich & Zhu introduce the block scheme to avoid.
    """
    res = _fit(carbontax, conformal_type="block")
    assert res.inference.details["conformal_type"] == "block"


def test_conformal_type_rejects_an_unknown_scheme(carbontax):
    with pytest.raises((MlsynthConfigError, ValueError)):
        _fit(carbontax, conformal_type="bootstrap")


# --------------------------------------------------------------------------
# the reference values
# --------------------------------------------------------------------------

def test_block_pvalue_matches_scinference_under_the_null(carbontax):
    res = _fit(carbontax, conformal_type="block")
    assert res.inference.p_value == pytest.approx(SCINFERENCE_MB_NULL, abs=5e-6)


@pytest.mark.parametrize("effect", [1.0, 5.0, 100.0])
def test_block_pvalue_reaches_the_floor_under_an_injected_effect(carbontax, effect):
    """Power, asserted as the authors' package reports it.

    ``scinference`` returns ``1 / T`` for every one of these injected effects --
    the smallest value the moving-block reference set can produce, since one of
    the ``T`` cyclic shifts is the observed path.
    """
    res = _fit(carbontax, conformal_type="block", effect=effect)
    assert res.inference.p_value == pytest.approx(SCINFERENCE_MB_FLOOR, abs=5e-6)


def test_iid_pvalue_responds_to_an_injected_effect(carbontax):
    """The other scheme has to move too, or the refit is absorbing the effect."""
    null = _fit(carbontax).inference.p_value
    shifted = _fit(carbontax, effect=100.0).inference.p_value
    assert null > 0.2
    assert shifted < 1e-9


# --------------------------------------------------------------------------
# the refit follows the estimator
# --------------------------------------------------------------------------

def test_ridge_augmented_fit_keeps_the_augmented_refit(carbontax):
    """``augment="ridge"`` is augsynth's ASCM, and its conformal mode goes with it.

    Reported rather than corrected: the refit a caller gets is the one their own
    estimator uses, and silently substituting a different one would make
    VanillaSC stop reproducing augsynth.
    """
    res = _fit(carbontax, augment="ridge", conformal_type="block")
    assert res.inference.details["refit"] == "ridge"
    assert _fit(carbontax, conformal_type="block").inference.details["refit"] == "sc"


def test_the_augmented_refit_agrees_here_because_its_penalty_is_large(carbontax):
    """On this panel the augmentation is inert, so the two rules coincide.

    Worth pinning because it is the reason the rule alone does not explain the
    disagreement with ``scinference``. Cross-validation on the pre-period picks
    a penalty near 700 for the carbon tax panel, which leaves the ridge
    correction negligible and the augmented fit sitting on the simplex. What
    unleashes the augmentation is re-selecting that penalty on a window that
    includes the post block -- see ``test_conformal_refit.py`` -- not the
    augmentation being available.
    """
    ridge = _fit(carbontax, augment="ridge", conformal_type="block")
    assert ridge.inference.details["lambda"] > 100.0
    assert ridge.inference.p_value == pytest.approx(SCINFERENCE_MB_NULL, abs=5e-6)


def test_covariates_without_ridge_are_refused_rather_than_dropped(carbontax):
    """A refit that quietly ignores the covariates would answer a different question."""
    df = carbontax.copy()
    df["gdp"] = df.groupby("country")["CO2_transport_capita"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    with pytest.raises(MlsynthConfigError, match="augment='ridge'"):
        _fit(df, covariates=["gdp"])


# --------------------------------------------------------------------------
# the band the p-value comes with
# --------------------------------------------------------------------------

def test_interval_brackets_the_point_effect(carbontax):
    res = _fit(carbontax, conformal_type="block")
    d = res.inference.details
    tau, lo, hi = d["tau"], d["pi_lower"], d["pi_upper"]
    finite = np.isfinite(lo) & np.isfinite(hi)
    assert finite.any()
    assert np.all(lo[finite] <= tau[finite] + 1e-9)
    assert np.all(tau[finite] <= hi[finite] + 1e-9)


def test_the_mode_is_deterministic_under_the_block_scheme(carbontax):
    """The block reference set is the ``T`` cyclic shifts, so no RNG enters it."""
    a = _fit(carbontax, conformal_type="block").inference.p_value
    b = _fit(carbontax, conformal_type="block", seed=99).inference.p_value
    assert a == b
