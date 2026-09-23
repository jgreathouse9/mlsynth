.. _replication-gsynth-av-laws:

Generalized synthetic control — age verification laws (Lang et al. 2026)
=========================================================================

:Estimator: :doc:`../gsynth` — generalized synthetic control, no additive
   effects, rank chosen by Xu's Algorithm 1.
:Source: Lang, D., Listyg, B., Ross, B. V., Musquera, A. V., & Sanderson, Z.
   (2026), *"Age Verification and Public Adaptation: A Pre-Registered Synthetic
   Control Multiverse,"* Journal of Law and Empirical Analysis 3(1):23-50,
   `10.1177/2755323X251415424 <https://doi.org/10.1177/2755323X251415424>`_.
   Replication package: `davidnathanlang/internet_regulation_synth_project
   <https://github.com/davidnathanlang/internet_regulation_synth_project>`_.
:Replication type: Cross-validation against a commit-pinned ``gsynth`` 1.2.1 —
   the version the authors ran — with the published Table 2 carried as a
   secondary, looser comparison.
:Status: Verified against the reference at machine precision over 96 fits, and
   on Algorithm 1's criterion and its selected rank in all sixteen
   outcome-by-force combinations. Three of the four published ATTs reproduce
   within 0.15; the fourth misses by 0.70 because the paper's rank is not the
   one the paper's own cross-validation selects.

What the paper estimates
------------------------

Between 2023 and 2024, fourteen US states began requiring age verification to
access pornographic websites. The stated aim is to keep minors off those sites.
The question the paper asks is what adults did in response, and it answers it
from Google Trends: weekly state-level search interest in four terms, each
standing for a different adaptation.

``pornhub`` is a platform that complied by blocking the state outright, so
searches for it measure the law biting. ``xvideos`` is a platform that did not
comply, so searches for it measure substitution toward a site still reachable.
``vpn`` measures circumvention — a virtual private network makes traffic appear
to originate elsewhere, defeating the check. ``porn`` is the generic term, and
measures whether the category-level demand moved at all.

The pre-registered finding is that the laws redirected traffic without reducing
it. Pornhub searches fall about 34 points on the 0–100 Trends scale, xvideos
rises about 28, VPN interest rises about 13, and generic pornography searches
move about 3.

Why this panel
--------------

:doc:`gsynth` — the estimator's own Xu (2017) replication — is a 47-state panel
of 24 quadrennial elections with four adoption dates. This one is 46 states by
149 weekly periods with ten adoption dates and fourteen adopters. Six times the
periods, weekly frequency instead of four-yearly, staggering that is genuinely
staggered, and four outcomes whose signal-to-noise ranges over a factor of
three.

That is the reason it is here. An estimator can look correct on a short panel
with a near-simultaneous treatment and still be wrong about how it aligns event
time, how it averages across cohorts, or how it holds out a period for
cross-validation. This panel exercises all three.

The reference
-------------

``gsynth`` 1.2.1 is the version the paper used, and the last release before the
package became a shell over ``fect``. Installing current ``gsynth`` would run a
different codebase than the paper did, so
``benchmarks/R/install_gsynth_121.sh`` pins the commit behind the ``1.2.1``
tag, along with pinned ``lfe`` and ``nanoparquet``.

The two generations agree here, so either could have served. At matched rank
and ``force``, gsynth 1.2.1 and fect 2.4.5 return the same numbers to fect's
print precision — pornhub −35.185 both ways, xvideos 28.403, vpn 11.435, porn
3.018. This case pins the one the authors ran, so a disagreement is about
mlsynth and not about a version gap.

The R side reads ``basedata/lang_av_laws.parquet`` through ``nanoparquet``, the
same file the Python side reads, so the two cannot run on different inputs.

Results
-------

Four outcomes by four ``force`` settings by ranks zero through five is 96 fits.
Each is compared on three quantities: the overall ATT, the mean over the
paper's estimand window, and the pre-treatment mean absolute error.

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * - Quantity
     - Fits compared
     - Max :math:`|` mlsynth − gsynth 1.2.1 :math:`|`
   * - Overall ATT
     - 96
     - 6.4e-14
   * - ATT over the paper's window
     - 96
     - 4.6e-14
   * - Pre-treatment MAE
     - 96
     - 2.8e-14
   * - Algorithm 1 criterion, by rank
     - 96
     - 1.3e-12
   * - Rank Algorithm 1 selects
     - 16
     - all agree

The criterion agrees far more tightly here than on the turnout panel, where the
two implementations separate by 7.5e-4. That case fits covariates, so its
alternating least squares iterates and ``fect``'s looser in-cross-validation
tolerance bites. This panel has no covariates, so there is nothing to iterate
and the tolerance never applies.

The published table
-------------------

Table 2's four ATTs are carried at a much looser tolerance, and the reason is a
finding.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Outcome
     - Published ATT
     - At the CV rank
     - Published MAE
     - mlsynth MAE
   * - ``xvideos``
     - 27.85
     - 27.91
     - 1.17
     - 1.15
   * - ``porn``
     - 3.48
     - 3.59
     - 0.76
     - 0.77
   * - ``vpn``
     - 13.33
     - 13.18
     - 2.29
     - 2.34
   * - ``pornhub``
     - −33.84
     - −33.14
     - 0.70
     - 0.78

Three of four land within 0.15 on a 0–100 scale, and the pre-treatment errors
line up to 0.08 across the board, which is what confirms the specification was
identified correctly — the MAE column depends on the donor pool, the rank, the
``force`` setting and the pre-window all at once, so four of them agreeing is
not something a differently specified fit produces.

Pornhub misses by 0.70, and the cause is rank selection. The published figure
sits between :math:`r = 2` (−33.93) and :math:`r = 3` (−33.48), while
Algorithm 1 selects :math:`r = 5` (−33.14). Every published value lies inside
its outcome's :math:`r = 0, \dots, 5` band, so the gap is a choice of rank and
not a different estimator.

Three other explanations were tested and ruled out:

* Data vintage. The upstream repository has three committed versions of the
  Trends pulls. The paper's analysis window is byte-identical across the two
  that contain it, so it does not matter which is checked out.
* Package generation. gsynth 1.2.1 and fect 2.4.5 agree to print precision, as
  above.
* Donor pool. The authors' ``03_preregistered_hypotheses.R`` drops five states
  (ND, MO, AZ, OH, GA). Running with and without that exclusion moves the
  estimates by 0.1 to 0.5 — wrong direction and wrong size to explain 0.70.

Re-running the authors' call verbatim, with ``se = TRUE``,
``inference = "parametric"`` and ``seed = 12345``, still selects :math:`r = 5`.
So ``published_att_max_diff`` is held to 0.8: loose enough to admit the pornhub
rank gap, tight enough that a broken estimator could not pass it, since the four
outcomes span −34 to +28. This is not Path A. Pinning the printed values tightly
would pin something neither implementation returns.

Two incidental findings
-----------------------

Reading the authors' script closely turned up two things that do not change the
published conclusions but do affect anyone re-running it.

``cumuEff(period = c(0, 12))$est.catt`` returns 65 rows, not 13, so the
downstream ``filter(rn + 1 == 13)`` does not select what its form suggests. And
at line 119, ``hyperparameter_search(keyword = keyword)`` lets the function's
own default for ``time_range`` shadow the value set at the top of the script.

Both were found by trying to reproduce the estimand exactly, which is the kind
of thing a cross-validation case surfaces and a headline-number comparison does
not.

Two index conventions
---------------------

The estimand is the mean over gsynth relative periods 0 through 12. gsynth
numbers the last pre-treatment period 0 and the first treated period 1; mlsynth
numbers the first treated period 0. The same thirteen cells are therefore
mlsynth's horizons −1 through 11.

The pre-treatment window differs too. gsynth's relative-time index is truncated
below at the earliest horizon where every treated unit is still observed — −53
here, the first adopter having 54 pre-periods — so its MAE is a balanced-window
average. mlsynth's event study reports the full range, down to −145, where a
single late adopter carries the average alone. Those thin horizons are noisier,
and including them inflates the MAE by a third. The case restricts to the
balanced block, which ``n_units`` marks exactly.

Reading the data
----------------

``basedata/lang_av_laws.parquet`` is ``data/{pornhub,xvideos,vpn,porn}.csv``
from the authors' repository at commit ``38ab54b``, restricted to the paper's
analysis window (``time == "2022-01-01 2024-10-31"``) and dropping the five
states the authors' own script drops. Each of the four frames is 46 × 149 and
balanced; fourteen states adopt across ten staggered dates and thirty-two never
do, and adoption is absorbing.

Running the reference
---------------------

The committed gold under ``benchmarks/reference/gsynth_av_laws/`` is what the
case reads by default, so it runs in CI without R. To re-run the reference
live::

   bash benchmarks/R/install_gsynth_121.sh
   MLSYNTH_BENCH_LIVE_R=1 python benchmarks/run_benchmarks.py \
       --case gsynth_av_laws

That regenerates the gold into a temporary directory and checks it against the
committed copy. A regenerated run currently reproduces it at 0.0, and the case
raises if that moves by more than 1e-8, because a row that exists only when R
does could not be pinned and an unpinned row is one nobody checks.

Case
----

``benchmarks/cases/gsynth_av_laws.py``. Every distance row is against the
reference, so a regression moves it and cannot be absorbed by re-fitting.
