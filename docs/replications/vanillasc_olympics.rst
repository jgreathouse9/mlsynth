.. _replication-vanillasc-olympics:

Standard synthetic control — the Tokyo 2020 Olympics (Yoneoka et al. 2022)
===========================================================================

:Estimator: :doc:`../vanillasc` — the ordinary Abadie-Diamond-Hainmueller
   synthetic control, with in-space placebo inference.
:Source: Yoneoka, D., Eguchi, A., Fukumoto, K., Kawashima, T., Tanoue, Y.,
   Tabuchi, T., Miyata, H., Ghaznavi, C., Shibuya, K., & Nomura, S. (2022),
   *"Effect of the Tokyo 2020 Summer Olympic Games on COVID-19 incidence in
   Japan: a synthetic control approach,"* BMJ Open 12:e061444,
   `10.1136/bmjopen-2022-061444 <https://doi.org/10.1136/bmjopen-2022-061444>`_.
   Replication package: `kingqwert/R/Synthetic_Olympic
   <https://github.com/kingqwert/R/tree/master/Synthetic_Olympic>`_.
:Replication type: Path A — the paper's published estimates on the authors'
   own data — and cross-validation against a commit-pinned ``tidysynth`` 0.2.0
   run of the authors' own script.
:Status: The paper's cumulative figures reproduce to the digit and its
   p-value exactly. The donor weights do not reproduce, in either
   implementation, and that is the finding this page is mostly about.

What the paper estimates
------------------------

The Tokyo 2020 Summer Olympics were held from 23 July to 8 August 2021,
without spectators, in the middle of Japan's fifth COVID-19 wave and under a
state of emergency in the host city. The question is whether hosting them
raised case counts.

There is no comparison country that also hosted an Olympics during a pandemic,
so the paper builds one: a weighted average of 42 donor countries chosen to
track Japan's daily confirmed cases per million before the opening ceremony,
and then compared with Japan afterwards. This is the ordinary synthetic control
estimator, with 34 predictors — testing and vaccination rates, the stringency
index, six Google mobility series, weather, demography, health system capacity,
an electoral democracy index — and five lagged values of the outcome itself.

The reported answer is about 53,900 excess infections over the Olympic period,
with the observed daily count at the closing ceremony roughly double the
counterfactual.

Why this panel
--------------

:doc:`vanillasc` is otherwise validated on the three canonical annual studies —
Proposition 99, German reunification, the Basque Country — where the panels are
short, smooth, and the treatment is a single well-separated date. This one is
43 countries by 50 daily periods of a noisy epidemiological series during an
exponential phase, with a 34-predictor specification and a published placebo
test. It exercises the predictor machinery and the inference where the annual
studies cannot.

Two reference bases
-------------------

The authors committed their own outputs — donor weights, the balance table,
the placebo distribution and the Fisher p-values, at all three intervention
timings they report. That artifact is the Path A gold: it is what produced the
printed table, so comparing against it asks whether this replication has
identified the paper's estimand.

A live ``tidysynth`` 0.2.0 run of the authors' own script is the second base,
and it asks a different question — whether the specification is reproducible
today.

Recovering the estimand
-----------------------

The paper's cumulative figures are not defined in the text, and had to be
recovered from the script. They are sums over ``date2`` 53 to 69 inclusive —
23 July through 8 August, the opening day through the closing day, seventeen
days — converted from per-million to counts at a population of 126.05 million.

At that window the authors' committed output gives 143,072 observed against
89,210 counterfactual, +60.4%, which is the paper's sentence to the digit. A
window off by one period misses it, so the case pins the observed side
recomputed from the vendored panel: the estimand and the data are checked
together.

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * - Quantity
     - Paper
     - Reproduced
   * - Cumulative observed, Olympic period
     - 143,072
     - 143,072
   * - Cumulative counterfactual
     - 89,210
     - 89,210
   * - Excess
     - +60.4%
     - +60.4%
   * - Observed, closing day (per million)
     - 109.2
     - 109.210
   * - Counterfactual, closing day
     - 51.0
     - 50.975
   * - Fisher exact p
     - 0.023
     - 0.0233
   * - Donors carrying weight
     - Germany, Hong Kong, Italy, Thailand, South Korea
     - 0.547, 0.203, 0.182, 0.045, 0.018

One arithmetic inconsistency. The paper reports the closing-day observed as
109.2 per million and the counterfactual as 51.0, both of which reproduce, and
then calls the first 115.7% higher than the second. 109.2 / 51.0 is 114.2%. The
cumulative pair is exact, so this is the text and not the estimator.

The weights do not reproduce; the answer does
----------------------------------------------

Running the authors' own script through ``tidysynth`` 0.2.0 today does not
return their committed weights.

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Timing
     - max :math:`|W_{\text{authors}} - W_{\text{live}}|`
     - max synthetic-path gap
     - Fisher p
   * - −7 days
     - 0.178
     - 5.54
     - same
   * - main
     - 0.183
     - 2.79
     - same
   * - +7 days
     - 0.038
     - 0.43
     - same

In the main specification Germany moves from 0.547 to 0.730, Italy from 0.182
to 0.078, South Korea from 0.018 to zero. The observed series is identical to
0.0 across the comparison, so this is the optimizer and not the data.

What does not move is the conclusion. The Fisher exact p-value is identical in
all three timings, and the closing-day counterfactual shifts from 50.98 to
51.81 against an observed 109.21 — a 1.6% move in a quantity the paper reports
as roughly doubled.

This is the ordinary non-identification of the synthetic-control weight vector.
It is the reason the case gates the p-value and the estimand and records the
weights: pinning weights that the authors' own package no longer returns would
pin noise.

Where mlsynth lands
-------------------

VanillaSC returns the same Fisher p-value — 0.0233, Japan ranked first of 43 —
at a pre-treatment RMSPE of 0.139 against ``tidysynth``'s 0.838. That is a
six-fold better fit to the same pre-period, on the criterion the reference is
itself minimising, and the better fit gives a larger effect: a closing-day
counterfactual of 46.0, so +137% where the paper says +115.7%.

The paper's direction and significance therefore survive a better-fitting
synthetic control; its magnitude is a lower bound under one. The improvement is
recorded, not gated as a win — a bug that lowered the objective for the wrong
reason would pass such a gate too. What is gated is that mlsynth does not do
worse on the reference's own criterion, and that the inference agrees exactly.

The 34 predictors barely bind
------------------------------

The predictor-weight search puts 93.7% of its mass on the five lagged values of
the outcome. The authors' own balance table shows what that leaves for the rest:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Predictor
     - Japan
     - Synthetic Japan
   * - Human mobility (parks)
     - −1.3
     - 98.2
   * - COVID-19 vaccination
     - 18.7
     - 35.0
   * - Number of tests for COVID-19
     - 0.46
     - 3.06
   * - Human mobility (transit stations)
     - −19.6
     - −5.3

A specification that reads as covariate-rich behaves as an outcome-only one.
mlsynth's ``outcome-only`` backend, which ignores all 34 predictors, reproduces
the qualitative answer (+79% excess against +90% for the full spec), which is
the same statement from the other direction. The case pins the concentration
so that a change in the predictor machinery shows up.

The sensitivity analysis is weaker than the paper suggests
-----------------------------------------------------------

The paper shifts the intervention by ±7 days and reports the results
"consistent with our main findings". In sign and significance they are, in both
implementations and at every timing. In magnitude they are not.

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Timing
     - Excess, authors
     - Excess, mlsynth
     - Fisher p
   * - −7 days
     - +94.5%
     - +93.8%
     - 0.023
   * - main
     - +76.5%
     - +89.6%
     - 0.023
   * - +7 days
     - +69.9%
     - +31.1%
     - 0.047

The +7-day window already contains the surge, so the counterfactual chases it,
and how much of it the counterfactual catches depends on how tightly the
pre-period is fitted. The two implementations differ most there for exactly
that reason.

Reading the data
----------------

``basedata/yoneoka_olympics_covid.parquet`` is ``data/df.csv`` from the authors'
repository at commit ``bde42e2``, restricted to the paper's analysis window
(``date2 < 75``) and written to Parquet unchanged — no R is needed to rebuild
it. 43 countries by 50 daily periods, balanced, Japan plus the 42 donors the
paper lists.

``benchmarks/reference/vanillasc_olympics/reference.R`` reads this same file
through ``nanoparquet``, so the R and Python sides of the comparison cannot run
on different inputs, and ``authors/`` alongside it holds the authors' committed
CSVs copied unchanged.

Running the reference
---------------------

The committed gold under ``benchmarks/reference/vanillasc_olympics/`` is what
the case reads by default, so it runs in CI without R. To re-run it live::

   bash benchmarks/R/install_tidysynth_olympics.sh
   Rscript benchmarks/reference/vanillasc_olympics/reference.R

The authors' script asks for ``quadopt = "LowRankQP"``, which is not
``tidysynth``'s default, so the install script pins that package too — without
it the reference solves a different inner problem than the paper did.

Case
----

``benchmarks/cases/vanillasc_olympics.py``. The case runs the full placebo
inference, which is the paper's own inference and the tightest agreement
available here, so it takes about eight minutes.
