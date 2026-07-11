.. _replication-bfsc:

BFSC -- Bayesian Factor Synthetic Control (Pinkney 2021)
========================================================

:Estimator: :doc:`../bfsc` -- :class:`mlsynth.BFSC`
:Source: Pinkney, S. (2021), *"An Improved and Extended Bayesian Synthetic
   Control,"* arXiv:2103.16244 [BFSC2021]_.
:Replication type: cross-validation against the author's own reference (the
   paper's appendix Stan program) on two panels -- German reunification (captured
   Stan) and Proposition 99 (Stan run live via ``rstan``) -- plus Path A, the
   German reunification study.
:Status: verified -- the NumPyro posterior matches the Stan posterior
   cell-for-cell (to Monte-Carlo error).

Validation strategy
-------------------

Pinkney (2021) ships its model as a Stan program in the appendix. BFSC ports that
program to NumPyro (both are NUTS samplers), so the appendix Stan is the ground
truth. We transcribe it verbatim, run it via ``rstan`` on the German
reunification panel, and compare the posterior counterfactual of West Germany to
the NumPyro estimator on the identical data.

Cross-validation -- cell for cell
---------------------------------

Both samplers use 4 chains, 500 warm-up + 500 draws, ``adapt_delta`` 0.95, and
8 latent factors on ``basedata/german_reunification.csv`` (West Germany treated,
16 OECD donors, 1960--2003, reunification in 1990). The posterior counterfactual
of West Germany agrees:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Quantity
     - NumPyro vs Stan
   * - counterfactual, posterior mean (all 44 years)
     - max :math:`|\Delta|` = 0.48% of level; corr 0.999999
   * - 90% credible band endpoints
     - within ~140 on ~1000-wide bands
   * - :math:`\widehat{\sigma}` posterior mean
     - 0.0265 vs 0.0265 (4 decimals)

The residual is the Monte-Carlo difference between two independent NUTS runs, not
a specification gap. A factor model is rotation/sign non-identified, so the raw
loadings do not mix (Stan flags this too); the identified quantities -- the
counterfactual, the ATT, and :math:`\sigma` -- mix cleanly and match.

Path A -- the German reunification study
----------------------------------------

Run through the public estimator, BFSC reproduces the classical result:

.. code-block:: python

   import pandas as pd
   from mlsynth import BFSC

   df = pd.read_csv("basedata/german_reunification.csv")
   df["treat"] = df["Reunification"].astype(int)
   res = BFSC({"df": df, "outcome": "gdp", "treat": "treat",
               "unitid": "country", "time": "year",
               "n_factors": 8, "seed": 0, "display_graphs": False}).fit()

gives a mean post-1990 ATT near :math:`-1600` PPP-USD per capita -- the
Abadie-Gardeazabal reunification effect -- statistically indistinguishable from
traditional synthetic control through the 1990s and a larger effect after 2000,
exactly the paper's stated finding, now with a credible band that widens through
the post-period. The durable case is ``benchmarks/cases/bfsc_germany.py``.

Second cross-check -- Prop 99, live via rstan
---------------------------------------------

The German cross-check above compares NumPyro against captured Stan numbers. The
durable case ``benchmarks/cases/bfsc_prop99.py`` goes further: it compiles and
runs the appendix Stan program *live* through ``rstan`` (so the reference is
Stan's own compiled model, not a stored table) on the shipped California
Proposition 99 cigarette panel -- California treated in 1989, 38 donor states,
1970--2000 -- and feeds the identical panel to :class:`mlsynth.BFSC`. Both use
eight factors, ``adapt_delta`` 0.95, and the same seeded budget.

The two independent NUTS runs agree on the identified quantities. At 1000 warmup
/ 1000 draws the counterfactual paths correlate at :math:`0.9999` (worst-cell gap
:math:`\approx 1.5` packs on a :math:`\sim 95`-pack level), :math:`\widehat{\sigma}`
agrees to :math:`0.0006`, the pre-period RMSE to :math:`0.02` packs, and the mean
post-1989 ATT to :math:`\approx 1` pack (both near :math:`-16`); the longer 2000 /
2000 run tightens the ATT agreement to :math:`\approx 0.6` packs. Substantively
both recover the classical Proposition 99 finding -- a widening decline in
cigarette sales -- but with a credible band wide enough to reach zero, which the
textbook point-estimate synthetic control does not show.

Prop 99 is a harder panel for this model than reunification: the pre-period is
short (19 years) and California sits somewhat outside the donor cloud, so the
post-period imputation is less pinned and both samplers mix less cleanly (Stan
reports it too -- NA R-hat on the non-identified factor loadings, low ESS). The
residual ATT difference is therefore Monte-Carlo error between two under-mixed
runs of one model, and it shrinks as the sampling budget grows -- itself evidence
that the two implementations are the same model, not two specifications. The case
requires the ``[bayes]`` extra for mlsynth and ``rstan`` for the reference, and
skips gracefully when either is absent.

Why NumPyro, not pure numpy
---------------------------

The model's factors and loadings are bilinear and the horseshoe+ scales are
heavy-tailed, which makes a hand-written Gibbs sampler finicky (the conditional
covariances go ill-conditioned). NUTS handles that geometry, so BFSC uses NumPyro
behind the ``[bayes]`` optional dependency, following :doc:`../spotsynth`. A
pure-numpy Gibbs sampler is feasible -- every conditional is conjugate and the
MAP coordinate-ascent reproduces the fit -- but reaching NUTS parity is a
sampler-engineering project; :doc:`bscm` is the dependency-free Bayesian SC for
the donor-weighting model.
