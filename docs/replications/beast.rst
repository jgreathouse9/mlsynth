.. _replication-beast:

BEAST -- Immunized Doubly-Robust Synthetic Control (Proposition 99)
===================================================================

:Estimator: :doc:`../beast` -- :class:`mlsynth.BEAST`
:Source: Bléhaut, D'Haultfœuille, L'Hour & Tsybakov (2021), "An alternative to
   synthetic control for models with many covariates under sparsity"
   (`arXiv:2005.12225 <https://arxiv.org/abs/2005.12225>`_), with the authors'
   R reference (``jeremylhour/alternative-synthetic-control-sparsity``).
:Replication type: Cross-validation -- a live run of the authors' own R code on
   the identical design matrix.
:Status: Done -- the balancing coefficients match to five decimals; the
   immunized ATT reproduces the R (mean :math:`-22.44`, matched to ~0.02 packs;
   :math:`-31.51` by 2000, to ~0.006), with a per-year path gap under ~0.15
   packs and the standard errors matched to ~0.05.
:Durable check: ``benchmarks/cases/beast_prop99.py`` (``beast_prop99``), run
   against the pinned R reference under
   ``benchmarks/reference/beast_prop99/``; plus ``mlsynth/tests/test_beast.py``.

The target
----------

The authors ship a Proposition 99 application with the paper
(``EX1_CaliforniaTobaccoConsumption.R``): a panel synthetic control on the
Abadie, Diamond & Hainmueller (2010) tobacco panel, where California raises its
cigarette tax in 1989 and the donors are the other US states. Rather than
fitting the outcome on a simplex, BEAST balances a covariate design -- four
economic predictors (``loginc``, ``p_cig``, ``pct15-24``, ``pc_beer``) plus
lagged cigarette sales (1975/1980/1988) -- by an ℓ₁-penalized exponential tilt,
then immunizes the effect with a WLS-Lasso outcome model. This is the paper's
``basic`` informative-covariate regime: a handful of covariates that truly drive
the selection.

Demonstrate-first port
----------------------

Before wiring anything into mlsynth, the four R routines were ported to NumPy and
validated cell by cell against a live run of the authors' code on mlsynth's exact
design matrix (so any residual gap is purely R's OWL-QN calibration versus the
port's proximal-gradient / FISTA solves, not a difference in inputs):

* ``CalibrationLasso.R`` :math:`\to` :func:`~mlsynth.utils.beast_helpers.estimator.calibration_lasso`
  -- the exponential-tilting balancing weights with iterated
  Belloni--Chernozhukov--Hansen penalty loadings;
* ``OrthogonalityReg.R`` / ``LassoFISTA.R`` :math:`\to`
  :func:`~mlsynth.utils.beast_helpers.estimator.orthogonality_reg` -- the
  immunizing WLS-Lasso outcome model;
* ``ImmunizedATT.R`` :math:`\to`
  :func:`~mlsynth.utils.beast_helpers.estimator.immunized_att` -- the
  doubly-robust ATT and its closed-form standard error.

The R reference installs with no CRAN access: ``lbfgs`` (which the authors' code
depends on) compiles from the GitHub CRAN mirror against the system R. The
captured reference (per-year ATT path and standard errors, the mean, and the
2000 endpoint) is vendored under ``benchmarks/reference/beast_prop99/`` with its
provenance and data checksum pinned.

Cell-by-cell agreement
----------------------

On the identical design matrix, the port reproduces the R:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22

   * - Quantity
     - mlsynth (NumPy)
     - R reference
   * - balancing coefficients :math:`\boldsymbol\beta`
     - agree to 5 decimals
     - --
   * - mean post-1989 ATT
     - :math:`-22.44`
     - :math:`-22.4443`
   * - ATT by 2000
     - :math:`-31.51`
     - :math:`-31.5129`
   * - ATT path (per year)
     - max abs diff :math:`\sim 0.15`
     - --
   * - standard-error path
     - max abs diff :math:`\sim 0.05`
     - --
   * - sum of balancing weights
     - :math:`1.0`
     - :math:`1.0`
   * - covariates selected
     - :math:`1`
     - :math:`1`

The balancing weights sum to one (a valid synthetic control), the ℓ₁
calibration selects a single informative covariate, and the top donor is Utah --
the classic Proposition 99 synthetic California. The residual per-year gap in
the ATT path (largest about ``0.15`` packs, averaging out to ``0.02`` on the
mean) is the calibration solver difference (OWL-QN vs proximal-gradient), not a
modelling difference.

The honest finding: BEAST's operating envelope
----------------------------------------------

The replication also mapped out where the method stops working. Run on an
over-rich covariate specification -- the SparseSC augmented California design,
with many more predictors approaching the number of donors -- the exponential
tilt degenerates in *both* implementations: the R balancing weights sum to
:math:`0` and the Python to :math:`1.72`, neither of which is a valid synthetic
control, and no penalty level recovers a sensible balance. There is no free
lunch in the high-dimensional regime the title advertises unless the sparsity
assumption actually holds.

That finding is what shaped the build. BEAST is deployed for the ``basic``
informative-covariate regime, and the estimator carries a balance-validity guard
(``balance_tol``) that raises :class:`~mlsynth.exceptions.MlsynthEstimationError`
when :math:`\lvert \sum_j W_j - 1\rvert` exceeds tolerance -- turning the
degenerate over-saturated case into an explicit error rather than a silently
meaningless answer. ``mlsynth/tests/test_beast.py`` pins that behaviour
(``test_oversaturated_regime_is_rejected``) alongside the R cross-validation of
the basic regime.

Running it
----------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case beast_prop99

The case cross-checks the immunized ATT path and standard errors against the
vendored R reference, and asserts the valid-balancing (:math:`\sum_j W_j = 1`)
and sparse-selection properties of the informative Proposition 99 regime.
