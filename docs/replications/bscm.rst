.. _replication-bscm:

BSCM — Bayesian Synthetic Control Methods (Kim, Lee & Gupta 2020)
=================================================================

:Estimator: :doc:`../bscm` — :class:`mlsynth.BSCM`
:Source: Kim, Sungjin, Lee, Clarence, and Gupta, Sachin (2020),
   *"Bayesian Synthetic Control Methods,"* Journal of Marketing Research
   57(5):831-852 [BSCM2020]_.
:Replication type: cross-validation against the authors' reference Stan
   implementation (``clarencejlee/bscm``), plus Path A — the Basque / ETA study.
:Status: verified — the counterfactual matches the reference to a correlation
   above 0.999 for both priors.

Why Basque, and not the paper's data
------------------------------------

The paper's empirical application measures the effect of a 2010 Washington soda
tax using proprietary state-level weekly Nielsen retail data, which is not
public. The method, however, is data-agnostic, so this replication validates on
the canonical public Basque Country panel (Abadie-Gardeazabal 2003, ``basedata``):
treated unit Basque Country, 16 donor regions, treatment in 1970. This is the
same panel the :doc:`smc` replication uses, which makes the two Bayesian /
frequentist relax-the-simplex methods directly comparable.

Validation strategy
-------------------

BSCM ships a reference Stan implementation (``clarencejlee/bscm``): two Stan
programs, ``Horseshoe_Publish.stan`` and ``Spike_Slab_Publish.stan``, that fit
the treated unit on the donor pool with an intercept and the respective
shrinkage prior, then produce the post-treatment counterfactual in the
``generated quantities`` block. Sampled with rstan (4 chains, 2000 iterations,
1000 warm-up), those programs are the ground truth. mlsynth reproduces them with
a pure-numpy Gibbs sampler -- the horseshoe via the Makalic-Schmidt (2016)
auxiliary-variable representation, the spike-and-slab via conjugate updates with
a marginalised inclusion draw -- so there is no Stan or probabilistic-programming
dependency in the library.

Cross-validation — the reference Stan on Basque
-----------------------------------------------

Fed the identical Basque matching problem, the numpy port reproduces the
reference posterior. Point estimates (treatment 1970):

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Quantity
     - Stan reference
     - mlsynth BSCM
     - counterfactual corr
   * - horseshoe ATT
     - :math:`-0.72`
     - :math:`\approx -0.67`
     - :math:`> 0.999`
   * - spike-slab ATT
     - :math:`-0.69`
     - :math:`\approx -0.76`
     - :math:`> 0.999`
   * - pre-treatment RMSE
     - :math:`\approx 0.004`
     - :math:`\approx 0.005`
     -
   * - negative weights
     - 6–7 of 16
     - 6–7 of 16
     -

Both methods say Basque GDP per capita ran about :math:`0.7`
(thousand-1986-USD) below its synthetic counterfactual after 1970, and both put
several donor weights below zero -- the unconstrained-extrapolation signature
that a simplex SCM cannot produce. The near-zero pre-treatment RMSE reflects the
"large p, small n" regime (16 donors, 15 pre-periods): the model interpolates
the pre-period, so the shrinkage prior rather than the pre-fit governs the
counterfactual.

The credible band. mlsynth implements the paper's exact horseshoe hierarchy,
including the coupling of the global shrinkage scale to the residual scale
(:math:`\tau \mid \sigma \sim \mathcal{C}^+(0, \sigma)`). Folding the scales
(:math:`\lambda_j = \tau \tilde{\lambda}_j`, :math:`\tau = \sigma \tilde{\tau}`)
shows this is the canonical :math:`\sigma^2`-scaled horseshoe regression
:math:`\beta_j \sim \mathcal{N}(0, \sigma^2 \tilde{\tau}^2 \tilde{\lambda}_j^2)`,
sampled with the Makalic-Schmidt (2016) auxiliary variables. With the coupling
in place the numpy ATT credible interval tracks the reference closely -- on the
Basque problem the interval width is within roughly ten to fifteen percent of the
Stan interval, versus a markedly wider band if the global scale is left
uncoupled. Any residual difference is sampler geometry, not model: the paper's
sampler is Stan's NUTS, which navigates the horseshoe's funnel-shaped posterior
differently from a block Gibbs, so the two target the same posterior without
agreeing on the band to the last digit. The spike-and-slab, whose posterior is
tamer, matches the reference band closely.

Path A — the Basque study
-------------------------

Run through the public estimator on ``basedata/basque_data.csv`` (treatment in
1970), BSCM reproduces the Abadie-Gardeazabal result with a posterior attached:

.. code-block:: python

   import pandas as pd
   from mlsynth import BSCM

   df = pd.read_csv("basedata/basque_data.csv")
   df["treat"] = ((df["regionname"] == "Basque Country (Pais Vasco)")
                  & (df["year"] >= 1970)).astype(int)
   res = BSCM({"df": df, "outcome": "gdpcap", "treat": "treat",
               "unitid": "regionname", "time": "year",
               "prior": "horseshoe", "seed": 2019,
               "display_graphs": False}).fit()

gives a post-1969 ATT of about :math:`-0.7` with a 95% credible interval, a
near-exact pre-treatment fit, and combined donor weights that mix positive and
negative signs. The divergence traces the familiar economic cost of ETA
terrorism. The durable case is ``benchmarks/cases/bscm_basque.py``.
