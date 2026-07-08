.. _replication-propsc:

PROPSC — Treatment Effects on Proportions (Bogatyrev & Stoetzer 2026)
=====================================================================

:Estimator: :doc:`../propsc` — :class:`mlsynth.PROPSC`
:Source: Bogatyrev, K., and L. F. Stoetzer (2026), *"Estimating Treatment
   Effects on Proportions with Synthetic Controls,"* Political Analysis
   (doi:10.1017/pan.2026.10046).
:Reference code: the authors' R package ``propsdid``
   (`lstoetze/propsdid <https://github.com/lstoetze/propsdid>`_), a GPL fork of
   ``synthdid`` extending it to compositional outcomes.
:Replication type: **Path A** — the two published empirical applications (Spain
   and Poland) reproduced cell by cell — **and cross-validation** against the
   authors' R package to numerical precision.
:Status: **Fully verified** — both empirical tables and the R package matched.

Validation strategy
-------------------

The paper's contribution is a coherence property, not a headline point estimate:
common weights across the components of a composition make the estimated effects
sum to zero, which separate synthetic controls violate. The natural target is
therefore the *whole vector* of effects in each application, checked two ways.

First, cross-validation. Because the authors ship a runnable R package, the
strongest evidence is a value-by-value diff against it on the same panel.
``mlsynth``'s :class:`~mlsynth.PROPSC` is a faithful port of the package's
common-weights estimator (the ``synthdid`` Frank-Wolfe solver with the two-round
sparsify pass, stacked across the :math:`K` components), and reproduces the
package cell by cell — estimates, jackknife standard errors, unit weights, and
time weights — to roughly :math:`10^{-11}` (floating-point reordering).

Second, Path A. That same run reproduces the paper's published Table 2 (Spain)
and Table 3 (Poland) common-weights columns to the printed two decimals.

Path A — Spain "Just Transition" (Table 2)
------------------------------------------

Bogatyrev and Stoetzer re-examine Bolet, Green and González-Eguino (2024): the
electoral effect of a compensatory "Just Transition Agreement" in Spanish
coal-mining municipalities, estimated on the full vector of party vote shares
rather than one party at a time. The panel (``basedata/spain_propsc.csv`` — 525
municipalities over five elections 2008–2019, 109 treated in 2019, party shares
in percentage points with VOX coded zero before its 2013 founding) is exported
from the article's Harvard Dataverse archive (doi:10.7910/DVN/MPUEIC).

.. code-block:: python

   import pandas as pd
   from mlsynth import PROPSC

   df = pd.read_csv("basedata/spain_propsc.csv")
   parties = ["psoe", "pp", "podem", "cs", "vox", "others"]
   res = PROPSC({
       "df": df, "outcomes": parties, "treat": "coalXpost",
       "unitid": "munid", "time": "year", "method": "sdid",
   }).fit()

   for party, att, se in zip(parties, res.att_vector, res.se_vector):
       print(f"{party:8s} {att:+.2f} ({se:.2f})")
   print("sum:", round(res.sum_constraint, 12))

The synthetic-DID common-weights estimates reproduce Table 2 exactly:

=========  =====================  ===============
Party      PROPSC (SDID, common)  Paper Table 2
=========  =====================  ===============
PSOE       +1.30 (0.68)           +1.30 (0.68)
PP         +0.98 (0.82)           +0.98 (0.82)
PODEMOS    +0.30 (0.36)           +0.30 (0.36)
Citizens   +0.94 (0.67)           +0.94 (0.67)
VOX        −3.43 (0.53)           −3.43 (0.53)
Others     −0.09 (0.35)           −0.09 (0.35)
Sum        0                      0
=========  =====================  ===============

Substantively, modelling the full composition changes the reading of the
original study: the effect is concentrated in a decline for the far-right VOX
(the mainstream gains lose significance under common weights), and the six
effects sum to zero, whereas separate synthetic DID leaves a net of
+0.55 percentage points that the composition forbids.

Path A — Poland anti-LGBTQ resolutions (Table 3)
------------------------------------------------

The second application revisits Haas et al. (forthcoming): the effect of
municipal anti-LGBTQ resolutions before the 2019 Polish parliamentary election
on a three-way composition of the electorate — government turnout, opposition
turnout, and abstention. With the same call (``method="sdid"``, the three
outcomes as the composition), PROPSC reproduces Table 3's common-weights column:
government turnout −0.25 (0.20), opposition turnout −1.13 (0.17), abstention
+1.39 (0.14), summing to zero.

Cross-validation harness
------------------------

The durable check is ``benchmarks/cases/propsc_spain.py``. It fetches
``propsdid`` at the pinned commit (``benchmarks/R/install_propsdid.sh``), runs
the package on ``basedata/spain_propsc.csv`` via
``benchmarks/R/propsdid_spain.R``, and diffs ``PROPSC.fit()`` against the live-R
output cell by cell. R is not run in CI: the gate compares against a frozen
capture of the R output by default and re-runs the R reference when
``PROPSDID_LIVE=1`` is set. The measured discrepancy is at the
floating-point-reordering level (:math:`\sim 10^{-11}` on the effects and the
jackknife standard errors).

References
----------

Bogatyrev, K., and L. F. Stoetzer (2026). "Estimating Treatment Effects on
Proportions with Synthetic Controls." Political Analysis.

Bolet, D., F. Green, and M. González-Eguino (2024). "How to Get Coal Country to
Vote for Climate Policy: The Effect of a 'Just Transition Agreement' on Spanish
Election Results." American Political Science Review 118(3).

Arkhangelsky, D., S. Athey, D. A. Hirshberg, G. W. Imbens, and S. Wager (2021).
"Synthetic Difference-in-Differences." American Economic Review 111(12).
