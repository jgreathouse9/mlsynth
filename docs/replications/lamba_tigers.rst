.. _replication-lamba-tigers:

Tiger reserves and avoided deforestation (Lamba et al. 2023)
============================================================

:Estimator: :doc:`../vanillasc` — :class:`mlsynth.VanillaSC`
:Source: Lamba, A., Teo, H. C., Sreekar, R., Zeng, Y., Carrasco, L. R., &
   Koh, L. P. (2023), *"Climate co-benefits of tiger conservation,"*
   Nature Ecology & Evolution 7:1104–1113,
   `doi:10.1038/s41559-023-02069-x
   <https://doi.org/10.1038/s41559-023-02069-x>`_.
:Replication type: cross-validation against the R package ``tidysynth``.
:Status: Seam verified exactly; the estimates deliberately not matched, for
   reasons this page sets out.
:Case: `benchmarks/cases/lamba_tigers.py
   <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/lamba_tigers.py>`_

What the paper does
-------------------

India designates certain protected areas as tiger reserves, which brings extra
funding, patrolling and monitoring. The paper asks whether that protection also
slowed deforestation, and converts the answer into avoided carbon emissions.

The design is one synthetic control per reserve. For a reserve designated in,
say, 2013, the outcome is cumulative forest loss in hectares from 2001 to 2020;
the donor pool is the protected areas in the same tiger-conservation zone that
were never designated; and the synthetic control is fit on the years before
designation using thirteen predictors — the outcome's own pre-period mean,
rainfall, population density, baseline forest area, the reserve's age,
elevation, slope, aspect, area, travel time to the nearest city, purchasing
power, above-ground biomass and road length. The gap in 2020 between the
reserve's observed forest loss and its synthetic counterpart is the avoided
loss. Significance comes from the usual placebo exercise: refit treating each
donor as if it had been designated, and rank the reserve's ratio of
post-treatment to pre-treatment prediction error among them.

Forty-five reserves qualify, adopting across nine different years, with donor
pools of twenty to forty-six that differ reserve by reserve. That shape is why
the case exists: no other synthetic-control benchmark in mlsynth has staggered
adoption with a donor pool that changes per treated unit.

The paper reproduces — on a superseded package
----------------------------------------------

The replication package is unusually complete: 184 KB holding the data, the
estimation script, the Google Earth Engine code that produced the covariates,
and a metadata file naming every source. Running it reproduces Table 1 essentially
exactly:

.. list-table:: Avoided forest loss (ha), published against the authors' script
   :header-rows: 1
   :widths: 44 18 18 12

   * - Reserve
     - Published
     - ``tidysynth`` 0.1.0
     - Difference
   * - Nawegaon–Nagzira
     - 2,645
     - 2,643.4
     - −1.6
   * - Similipal–Hadagarh
     - 1,570
     - 1,569.9
     - −0.1
   * - Udanti
     - 1,611
     - 1,608.0
     - −3.0
   * - Valmiki
     - 315
     - 317.5
     - +2.5
   * - Nagarjuna Sagar–Srisailam
     - 167
     - 187.1
     - +20.1
   * - Palamau
     - 143
     - 125.4
     - −17.6
   * - Satkosia Gorge
     - 39
     - 35.7
     - −3.3
   * - Bandipur
     - 25
     - 25.4
     - +0.4
   * - Biligiri Rangaswamy Temple
     - 22
     - 21.6
     - −0.4
   * - Sariska
     - 14
     - 14.2
     - +0.2
   * - Satpura
     - 9
     - 8.7
     - −0.3
   * - Total
     - 6,560
     - 6,556.7
     - −3.3

That is a clean result, and rarer than it should be. The catch is the version.
The authors' script calls ``grab_signficance`` — a misspelling ``tidysynth``
corrected on 2021-11-30 — so their analysis predates a 2022-08-31 commit whose
own message says time-ordering problems "resulted in downstream distortions of
the package output". The numbers above come from a build older than that fix.

Why mlsynth is not checked against those numbers
------------------------------------------------

Because reproducing them means reproducing a defect the package's own maintainer
has since corrected. The reference bundle therefore runs ``tidysynth`` 0.2.0,
the current release, and the table above is recorded here as history, not
pinned anywhere.

Running the current release surfaces two further problems, both documented in
the reference script because they bound what this replication can claim.

It ignores ``quadopt``. The authors pass ``quadopt = "LowRankQP"``, still
documented in the package's help. In 0.2.0 ``R/synth.R:174`` calls
``kernlab::ipop`` unconditionally inside the predictor-weight search, and the
final solve at line 303 sits inside an ``if (quadopt == "ipop")`` with no
``else`` branch, so the ``LowRankQP`` path assigns no weights at all. Version
0.1.0 had eight ``LowRankQP`` call sites; 0.2.0 has only the argument
validation, after a 2023-05-20 commit dropped the dependency. The reference
passes ``"ipop"`` explicitly, which is what the package does either way.

It cannot fit four of the fifteen reserves. ``ipop`` fails outright on Valmiki,
Biligiri Rangaswamy Temple, Sariska and Anamalai with a computationally singular
system, on panels the older build fitted without complaint. Those failures are
captured in ``failures.csv`` and counted in the case, because a reference
that hid them would overstate how much of the paper
the current package can still reproduce.

The estimates differ, and the reason is identification
------------------------------------------------------

Even setting versions aside, mlsynth and ``tidysynth`` do not agree on the
effects — and the interesting part is *where* they agree.

The predictor matrix agrees exactly. Both implementations reduce the panel to
thirteen rows by one column per unit, and mlsynth reproduces ``tidysynth``'s
matrix to nine significant figures across every donor of every reserve. That is
the seam, it is a deterministic function of the data, and the benchmark gates it
tightly. Whatever else differs, the panel does not.

The weights do not agree. Abadie's method chooses the predictor weights
:math:`V` by an outer search that minimises pre-treatment prediction error, and
that search is not identified when predictors are numerous relative to
pre-periods — thirteen against as few as six here. On the headline reserve
``tidysynth`` returns a :math:`V` placing 1.2 percent of the weight on the
outcome's own history and 58 percent on elevation, and lands on a pre-treatment
fit 3.6 times worse than mlsynth's. Across the eleven reserves the current
package can fit, mlsynth reaches the lower pre-treatment error on ten; the
exception is Similipal–Hadagarh, where ``tidysynth`` fits slightly better (334.5
against 364.4). Holding mlsynth to agreement would therefore hold it to the
worse solution on most of the panel, so the benchmark records the comparison and
gates mlsynth's own values instead. The count of ten is itself pinned, so a
change in either direction fails and has to be looked at.

Two further properties of mlsynth's solutions on this panel are pinned because
they are not what one would guess. The donor weights are interior — all
forty-four donors carry weight on the headline reserve, the smallest about
0.019, where a simplex least-squares fit usually zeroes most of them. And the
:math:`V` search is mildly seed-sensitive on short pre-periods: Udanti moves
about 21 ha (1.4 percent) across seeds, while Nawegaon–Nagzira and
Similipal–Hadagarh are stable to about 1e-5. The benchmark's tolerances are set
from that measured spread.

The estimand is the paper's avoided loss: the terminal-year gap, synthetic
minus observed in 2020. That is not
:attr:`~mlsynth.config_models.EffectsResults.att`, which averages observed minus
synthetic over every post-treatment year — on the headline reserve the two are
+2825 and −1491. An earlier draft of this case substituted one for the other and
produced entirely plausible numbers; the reference comparison is what caught it,
and a test now pins the distinction.

How much of the disagreement is real ambiguity, not one side being wrong
can be measured. Among all synthetic controls fitting the pre-period within two
percent of the best attainable, the avoided loss on the headline reserve can lie
anywhere in [2768, 2922] ha; widen to ten percent and the range is [2594, 3026].
mlsynth's 2,825 ha sits inside the tight band and the published 2,645 ha inside
the wider one, so both are defensible readings of the same data. The current
``tidysynth``'s 1,999 ha falls outside even a fifty-percent band, which is what
marks it as a poor fit, not a rival answer. That diagnostic is not yet a
library feature; it is tracked in
`issue #310 <https://github.com/jgreathouse9/mlsynth/issues/310>`_.

What to take from this
----------------------

The paper's conclusion is not in question. Every implementation tried here
agrees that the headline reserves averted substantial forest loss, that
Nawegaon–Nagzira averted the most, and that a minority of reserves lost more
forest than their counterfactuals. What is fragile is the third digit, and
anyone quoting a specific hectare figure should know it moves by hundreds of
hectares depending on which build of which package produced it.

For mlsynth's purposes the case pins two things: that the panel
construction is exactly right, and that the estimator handles a staggered,
per-reserve, zone-restricted design without incident.
