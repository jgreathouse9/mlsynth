.. _replication-dpsc:

DPSC -- Differentially Private Synthetic Control (Proposition 99)
=================================================================

:Estimator: :doc:`../dpsc` -- :class:`mlsynth.DPSC`
:Source: Rho, Cummings & Misra (2023), "Differentially Private Synthetic
   Control" (AISTATS 2023, PMLR v206), with the authors' Python reference
   (``srho1/dpsc``).
:Replication type: Cross-validation -- a live run of the authors' own
   ``PrivateSC`` on the identical panel under a shared random seed.
:Status: Done -- under the shared Mersenne-Twister stream mlsynth reproduces the
   authors' private coefficients and private counterfactual value-for-value, for
   both mechanisms: output perturbation to :math:`\sim 10^{-11}` and objective
   perturbation to :math:`\sim 10^{-11}` (mlsynth's closed-form objective solve
   versus the authors' cvxpy).
:Durable check: ``benchmarks/cases/dpsc_prop99.py`` (``dpsc_prop99``), which
   clones ``srho1/dpsc`` at a pinned commit on demand; plus
   ``mlsynth/tests/test_dpsc.py``.

The target
----------

The authors provide a Python implementation of two differentially private
synthetic-control mechanisms -- output perturbation (Algorithm 2) and objective
perturbation (Algorithm 3) -- both built on a ridge synthetic control and
privatised through differentially private empirical risk minimisation. Their
experiments run on synthetic random-walk panels; here the same code is exercised
on the Abadie, Diamond & Hainmueller (2010) Proposition 99 tobacco panel
(California treated from 1989, 38 donor states, :math:`T_0 = 19` pre-periods),
so the cross-check runs on a canonical, inspectable design.

Demonstrate-first port
----------------------

Before wiring anything into mlsynth, the two mechanisms were ported to NumPy and
validated cell by cell against the authors' code on the identical Proposition 99
matrices. Both mechanisms are randomised, so the validation pins the randomness:
``numpy.random`` seeds the authors' global generator and
``numpy.random.RandomState`` seeds mlsynth's -- the same Mersenne-Twister stream
-- and the two mechanisms consume their draws in the same order, so a shared
seed reproduces the private output exactly.

* the non-private ridge coefficients match ``sklearn``'s to machine precision
  (mlsynth solves the normal equations in closed form);
* output perturbation (Algorithm 2): the private coefficients and the private
  counterfactual match the authors' ``PrivateSC`` to :math:`\sim 10^{-11}`;
* objective perturbation (Algorithm 3): mlsynth solves the perturbed program in
  closed form (the perturbed normal equations) rather than through cvxpy, and
  the private weights and counterfactual still match to :math:`\sim 10^{-11}`.

The deterministic noise scales -- the coefficient sensitivity
:math:`\Delta = 4 T_0 \sqrt{8 + N_0} / \lambda`, the objective curvature term,
and the Stage-2 donor-noise scale -- match the authors' formulas exactly.

The honest finding
------------------

The replication also mapped the method's operating envelope on a real panel. The
private ATT is unbiased in expectation but its variance is large: on the 38-donor
Proposition 99 panel, output perturbation is unusable -- a single release carries
ATT noise of tens even at a negligible privacy level -- because the coefficient
noise inflates the weight norm and the Stage-2 term amplifies it. Objective
perturbation is the viable mechanism (the ridge term bounds the weights), and it
is the default; even so, meaningful privacy on a few-dozen-donor panel costs a
biased estimate with a wide band. Differential privacy is favourable when the
donor pool is large and the pre-period long, where the noise averages down --
which is exactly the clinical external-control and cross-party-measurement
setting the method is built for.

Reproducing
-----------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --only dpsc_prop99

runs the case against the pinned ``srho1/dpsc`` clone (skips cleanly if the
clone is unavailable).
