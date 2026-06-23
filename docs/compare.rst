Comparing counterfactuals across methods
========================================

When to use
-----------

Every synthetic-control estimator in mlsynth produces the same observable: a
counterfactual path for the treated unit -- what the outcome would have looked
like without the intervention -- and, when the estimator carries inference, a
per-period prediction interval around that path. Once you have fit more than one
estimator on the same panel, the natural next question is how their
counterfactuals compare, both to each other and to what was actually observed.

:func:`~mlsynth.utils.counterfactual_compare.compare_counterfactuals` answers
that question in one call. It reads each method's counterfactual, its prediction
interval (when present), and the stored ``att`` and pre-treatment fit (``pre_rmse``)
off the standardized result object, lines them all up on a common time axis, and
hands back a small container you can either inspect as a table or draw as a
single overlay. It saves you the per-method plotting loop and the second loop
that digs the interval arrays out of ``inference.details`` by hand.

The helper is deliberately undemanding about its inputs. A method can be a fitted
:class:`~mlsynth.config_models.BaseEstimatorResults` (the common case), a plain
array of counterfactual values, or a small dictionary spec. The last two let you
compare results that do not expose the standardized surface -- for instance the
:doc:`spillsynth` dispatcher, whose spillover-adjusted counterfactual is
assembled per method -- alongside ones that do.

What it returns
---------------

:func:`~mlsynth.utils.counterfactual_compare.compare_counterfactuals` returns a
:class:`~mlsynth.utils.counterfactual_compare.CounterfactualComparison` with
three faces:

- ``curves`` -- a tidy long table, one row per method and period, with columns
  ``counterfactual``, ``lower`` and ``upper`` (the last two empty where a method
  has no interval at that period);
- ``summary`` -- one row per method, holding the stored ``att`` and ``pre_rmse``,
  read off the result rather than recomputed;
- ``observed`` -- the observed treated series, when one was supplied or could be
  taken from a standardized result;

and a ``plot`` method that overlays the observed series and every method's
counterfactual, drawing prediction intervals as per-period error bars where they
exist.

Example: comparing different estimators on one panel
----------------------------------------------------

The clearest use is to put several different estimators side by side. Here three
of them -- the classical synthetic control, a cluster-denoised variant, and the
synthetic difference-in-differences estimator -- are fit on Abadie, Diamond and
Hainmueller's California tobacco panel, then compared. Only the first carries a
prediction interval, and the helper simply omits intervals for the methods that
do not:

.. code:: python

    import pandas as pd
    from mlsynth import VanillaSC, CLUSTERSC, SDID
    from mlsynth.utils.counterfactual_compare import compare_counterfactuals

    url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
           "main/basedata/augmented_cali_long.csv")
    df = pd.read_csv(url)
    df = df[df["year"] <= 2000].copy()
    df["treated"] = ((df["state"] == "California")
                     & (df["year"] >= 1989)).astype(int)
    base = dict(df=df, outcome="cigsale", treat="treated", unitid="state",
                time="year", display_graphs=False)

    sc   = VanillaSC({**base, "inference": "scpi", "alpha": 0.10}).fit()
    clus = CLUSTERSC({**base}).fit()
    sdid = SDID({**base}).fit()

    cmp = compare_counterfactuals(
        {"VanillaSC": sc, "CLUSTERSC": clus, "SDID": sdid})

    print(cmp.summary.round(2))
    #                att  pre_rmse
    # method
    # VanillaSC   -19.51      1.66
    # CLUSTERSC   -21.39      1.50
    # SDID        -15.61     24.81

    cmp.plot()

The ``summary`` table reads the stored effect and fit straight off each result,
so the numbers match what each estimator reports on its own; the ``plot`` overlays
the three counterfactuals against observed cigarette sales, with the prediction
interval shown only for the method that produced one.

Example: two solvers of the same estimator, with prediction intervals
---------------------------------------------------------------------

The comparison need not be across estimators. A common use is to hold the
estimator fixed and vary one setting -- here the predictor-weight solver behind
:doc:`vanillasc` -- so the disagreement is attributable to that setting alone.
Both solvers carry the Cattaneo-Feng-Titiunik prediction interval, which the
helper draws as per-period error bars, dodged so the two methods' bars stay
legible:

.. code:: python

    import pandas as pd
    from mlsynth import VanillaSC
    from mlsynth.utils.counterfactual_compare import compare_counterfactuals

    url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
           "main/basedata/augmented_cali_long.csv")
    df = pd.read_csv(url)
    df = df[df["year"] <= 2000].copy()
    df["treated"] = ((df["state"] == "California")
                     & (df["year"] >= 1989)).astype(int)
    # Three lagged outcomes as predictors, so the covariate-matching solvers run.
    for L in (1975, 1980, 1988):
        df[f"cig{L}"] = df["state"].map(
            df[df["year"] == L].set_index("state")["cigsale"])
    base = dict(df=df, outcome="cigsale", treat="treated", unitid="state",
                time="year", covariates=["cig1975", "cig1980", "cig1988"],
                inference="scpi", alpha=0.10, display_graphs=False)

    mscmt = VanillaSC({**base, "backend": "mscmt", "seed": 42}).fit()
    malo  = VanillaSC({**base, "backend": "malo", "seed": 0}).fit()

    cmp = compare_counterfactuals({"MSCMT nested": mscmt, "Malo bilevel": malo})
    cmp.plot(dodge=0.4,
             colors={"MSCMT nested": "C0", "Malo bilevel": "C3"},
             styles={"MSCMT nested": "--", "Malo bilevel": "-."})

The ``colors`` and ``styles`` mappings name the per-method line appearance; the
``dodge`` argument offsets each method's error bars horizontally so overlapping
intervals do not collide.

Example: results outside the standardized contract
--------------------------------------------------

Some results -- the :doc:`spillsynth` dispatcher among them -- do not expose the
standardized ``time_series`` surface, because their counterfactual is built per
method. For those, pass the counterfactual yourself: either a plain array, or a
dictionary spec that may also carry a prediction interval (with ``periods`` when
the interval covers only part of the series), the effect, and a time axis. Mixed
inputs are allowed, so a hand-built counterfactual can sit next to a standardized
result in the same comparison:

.. code:: python

    import numpy as np
    from mlsynth.utils.counterfactual_compare import compare_counterfactuals

    years = np.arange(2000, 2010)
    cmp = compare_counterfactuals(
        {
            # a dict spec: counterfactual, a post-period interval, and the att
            "method A": {"counterfactual": np.linspace(10, 5, 10),
                         "att": -2.1, "time": years,
                         "periods": years[5:],
                         "lower": np.linspace(4, 3, 5),
                         "upper": np.linspace(6, 5, 5)},
            # a bare array: just the counterfactual, no interval
            "method B": np.linspace(10, 6, 10),
        },
        observed=np.linspace(10, 4, 10), time=years)

    print(cmp.summary)        # att for A, NaN for B; pre_rmse NaN for both
    cmp.plot()

This is the same path the spillover illustration in the paper uses: each method's
full counterfactual is assembled, then handed to ``compare_counterfactuals`` with
its effect, so one overlay and one effect table cover all of the methods at once.

Accepted inputs
---------------

Each value in the ``methods`` mapping is one of:

- a standardized result -- the counterfactual, time axis, observed series,
  ``att``, ``pre_rmse``, and any ``inference.details`` interval are read off it;
- a mapping with at least ``counterfactual``, and optionally ``time``, ``lower``,
  ``upper``, ``periods`` (to align a partial interval), ``att``, ``pre_rmse`` and
  ``observed``;
- a plain array, treated as the counterfactual with no interval.

A prediction interval is always two-sided: supplying only ``lower`` or only
``upper`` is an error, as is an interval whose length matches neither the curve
nor a supplied ``periods`` list. These are surfaced as
:class:`~mlsynth.exceptions.MlsynthConfigError` or
:class:`~mlsynth.exceptions.MlsynthEstimationError`, never silently dropped.

Core API
--------

.. automodule:: mlsynth.utils.counterfactual_compare
   :members:
   :show-inheritance:
