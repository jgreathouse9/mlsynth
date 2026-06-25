Truncated History robustness check
==================================

.. currentmodule:: mlsynth

When to use this diagnostic
---------------------------

A synthetic control fits the treated unit over a chosen pre-treatment window. How
far back that window reaches is a modelling choice, and a credible effect should
not depend on it. The Truncated History (TH) framework of Spoelstra, Stolp,
Golsteyn, Cornelisz and van Klaveren (2025) makes that dependence visible: it
re-estimates the effect on truncated pre-treatment windows and profiles how the
ATT moves with the pretreatment horizon. A stable profile supports a causal
reading; an unstable one says a single point estimate is fragile and an interval
is the more honest summary.

Use it as a routine robustness check after any synthetic-control or
difference-in-differences fit -- it is the pretreatment-horizon analogue of the
in-space placebo and leave-one-out checks, and it is especially useful for
telling a genuine null apart from a meaningful effect when the two are hard to
distinguish.

How it works
------------

:func:`mlsynth.truncated_history` re-runs an estimator on truncated panels and
collects, per window, the ATT, the pre-treatment MSPE, and whatever inference the
estimator reports (an in-space placebo p-value, a standard error). It is
estimator-agnostic: it accepts any mlsynth estimator that returns the standard
result object, so the same call profiles ``VanillaSC``, ``SDID``, ``PDA`` and the
rest. The truncation modes are:

- ``"left"`` drops the earliest pre-periods one at a time -- the left-Truncated
  History check, which targets over-reliance on distant pretreatment data;
- ``"right"`` drops the latest pre-periods (the in-time placebo direction);
- ``"loo"`` and ``"l2o"`` leave one or two pre-periods out;
- ``"random"`` drops random subsets of pre-periods, repeatedly.

The returned :class:`mlsynth.TruncatedHistoryResult` carries the full-sample ATT,
the per-window ``profile``, the ATT interval across windows, and a heuristic
``stable`` verdict (the ATT keeps its sign and its spread stays small relative to
its mean).

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import truncated_history, SDID

   # California Proposition 99 (treated from 1989)
   df = pd.read_csv("basedata/P99data.csv").rename(columns={"cigsale": "y"})
   df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)
   cfg = {"df": df, "outcome": "y", "treat": "treat",
          "unitid": "state", "time": "year"}

   res = truncated_history(SDID, cfg, mode="left")
   print(res.att_full, res.stable)            # -15.6, True
   for w in res.profile[:4]:
       print(w.label, round(w.att, 1))        # 1971-1988 -16.3, 1972-1988 -16.8, ...
   print(res.stability_note)

Verification
------------

The left-TH profile reproduces Table 1 of Spoelstra et al. (2025) on California's
tobacco program. Run through mlsynth's ``SDID``, the ATTs match the paper to the
decimal across the reported windows (full sample :math:`-15.6`; 1971--1988
:math:`-16.3`; 1972--1988 :math:`-16.7`; 1974--1988 :math:`-17.2`), and the
profile is flagged stable -- the paper's finding that SDID is robust to the
pretreatment horizon. This is pinned in
``mlsynth/tests/test_truncated_history.py`` and as the durable benchmark case
``th_prop99``.

Core API
--------

.. autofunction:: truncated_history

.. autoclass:: TruncatedHistoryResult
   :members:

.. autoclass:: TruncatedHistoryWindow
   :members:
