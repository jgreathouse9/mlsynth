.. _replication-sdid-euets:

Synthetic DiD — the EU emissions trading system (Basaglia et al. 2024)
========================================================================

:Estimator: :doc:`../sdid` — synthetic difference-in-differences with
   time-varying covariates.
:Source: Basaglia, P., Grunau, J., & Drupp, M. A. (2024), *"The European Union
   Emissions Trading System might yield large co-benefits from pollution
   reduction,"* PNAS 121(28):e2319908121,
   `10.1073/pnas.2319908121 <https://doi.org/10.1073/pnas.2319908121>`_.
   Replication package: `ccs282/EU_ETS_Co_Benefits
   <https://github.com/ccs282/EU_ETS_Co_Benefits>`_.
:Replication type: Path A — the paper's synthetic difference-in-differences
   estimates on the authors' own data, against the five-decimal values in their
   committed Stata log.
:Status: All three pollutants reproduce to 3e-4 once the covariate projection
   uses the reference's own row rule. That qualifier is the finding this page is
   about.

What the paper estimates
------------------------

The EU emissions trading system caps carbon dioxide from the sectors it covers.
It says nothing about sulphur dioxide, fine particulates or nitrogen oxides —
but those come out of the same combustion, so if the cap reduces fuel burned it
should reduce them too, without ever naming them. The paper asks how large that
side effect has been.

The comparison is between sectors, not countries: within each of 25 EU
countries, the sectors the system regulates against the sectors it does not.
Both halves of a country share its macroeconomy, its recessions and its energy
prices, so the unregulated half is a plausible counterfactual for the regulated
one. Treatment starts in 2005, when the system's first trading period began.

The headline estimates come from a generalized synthetic control model. This
page covers the synthetic difference-in-differences robustness check that runs
beside it, in Stata.

Why only the SDID half
----------------------

The two halves of the paper run on different samples, and only one of them is a
panel mlsynth will currently accept.

The full panel is unbalanced. Estonia, Latvia, Lithuania and Slovenia enter in
1995, Slovakia in 1992, Hungary in 1991, and the United Kingdom leaves after
2019. The gaps are all at the ends — no country is missing an interior year — so
nothing can be interpolated. The generalized synthetic control specifications use
that panel as it stands; ``gsynth`` accepts it and estimates on 1,550 of the
1,600 cells.

The SDID do-file cannot, and says so:

.. code-block:: stata

   tsset id year //Panel is unbalanced --> Need to make it strongly balanced
                 //to run SDID estimator (Arkhangelsky et al., 2021)

so it drops the six late-entering countries and caps the sample at 2019, leaving
38 units by 30 years — the ``Observations 1140`` printed in the tables. That
sample is balanced, which is why this half replicates today and the other half
does not.

The covariate row rule
----------------------

The authors write ``covariates(log_gdp log_gdp_2, projected)`` and cite Kranz
(2022). Both of those are accurate and they do not describe the same estimator.

Kranz's recipe is to fit the covariate coefficients on the rows where treatment
is not in force, keep only those coefficients, subtract :math:`X\beta` from the
outcome across the whole panel, and hand the adjusted outcome to ordinary SDID.
In ``xsynthdid``'s ``R/adjust_y.R`` the rows are chosen by

.. code-block:: r

   x.rows = as.integer(panel[[treatment]]) == 0

which is every untreated observation, including the treated units' own
pre-treatment years. mlsynth's ``covariates={'adjust': ...}`` implements this and
is cross-validated against a live ``xsynthdid`` run at the seam — the fitted
coefficient and the adjusted outcome, element by element.

Stata's ``sdid`` selects different rows. Its ``projected()`` routine takes

.. code-block:: stata

   cdat = Y[selectindex(Y[,6 - NotYet]:==0), (1,2,4,8..K)]

and the data matrix is laid out as ``y, id, id, time, treat_post, treated,
tyear, covariates``. Column 6 is the *ever-treated unit* flag, not the treatment
indicator, so the default fits beta on never-treated units only. The ``_not_yet``
option moves the selection to column 5 and recovers Kranz's rule; the authors did
not pass it.

On this panel the two rules see different data. Never-treated units give 570
rows. Kranz's rule adds the 19 treated units' fifteen pre-2005 years, for 855.

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 16 21 21 21 21

   * - Pollutant
     - Stata ``sdid``
     - beta on never-treated
     - beta on all untreated
     - Published table
   * - SO₂
     - −0.20753
     - −0.20778
     - −0.24492
     - −0.208
   * - PM₂.₅
     - −0.32966
     - −0.32958
     - −0.27433
     - −0.330
   * - NOx
     - −0.12233
     - −0.12234
     - −0.11560
     - −0.122

Under the reference's own row rule mlsynth reproduces all three published
estimates to 3e-4 — 2.5e-4 at worst, on a quantity of about 0.2 to 0.33 log
points. Under Kranz's rule the distance is 0.007 to 0.055, and PM₂.₅ is where it
is largest.

So mlsynth's SDID core — the unit and time weight programs and the estimator
itself — agrees with Stata essentially exactly. The whole of the earlier
disagreement was the covariate convention, and on that question mlsynth follows
Kranz while the package citing Kranz does not. That makes it a difference
between two reference implementations, not a defect on either side; but it does
mean mlsynth currently has no single configuration that reproduces
``sdid, projected`` directly, and a paper written against Stata will need the
adjustment done by hand, as the case does.

The comparison the case gates on is therefore the reference's rule, with the
covariate step performed explicitly and the adjusted outcome handed to ``SDID``
with no covariates. That isolates the estimator from a convention the two
references do not share — the same separation :doc:`vanillasc` uses when it fixes
:math:`V` so its comparison measures the solver and not the predictor-weight
search. The size of the convention gap is pinned in its own rows, so a change in
either direction shows up.

What is not compared
--------------------

The bootstrap standard errors. The Stata call requests 800 replications under
its own random number generator with ``seed(1615)``; reproducing 0.12296 would
mean reproducing that stream, not the estimator.

The ``method(did)`` column of the same tables (SO₂ −0.34666, PM₂.₅ −0.49103,
NOx −0.27287). It is Stata's plain two-way estimator on the same adjusted
outcome, and mlsynth's SDID exposes no DiD mode, so there is nothing on this
side to compare against.

Reading the data
----------------

``basedata/euets_cobenefits.parquet`` is
``Stata_SDID/data_in/{so2,pm25,nox}_gscm_data.csv`` from the authors' repository,
concatenated with a ``pollutant`` column and written to Parquet unchanged — no
Stata or R is needed to rebuild it. Each pollutant is 50 units by 32 years
(1990–2021); the case applies the do-file's own filters to reach the balanced
38 by 30 estimation sample.

The five-decimal targets come from ``Stata_SDID/logs/EUETS_SDID.log``, which the
authors committed alongside the three-decimal tables.

Case
----

``benchmarks/cases/sdid_euets.py``.
