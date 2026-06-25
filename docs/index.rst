.. mlsynth documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mlsynth 1.0.0
========================

.. meta::
   :description: A Python toolbox of synthetic-control estimators for program
                 evaluation. Express your causal panel-data problem in a long
                 DataFrame, pick an estimator, get an ATT.
   :keywords: synthetic control, causal inference, program evaluation,
              difference-in-differences, panel data, ATT, Python.

.. raw:: html

      <script type="application/ld+json">
      {
         "@context" : "https://schema.org",
         "@type" : "WebSite",
         "name" : "mlsynth",
         "url" : "https://mlsynth.readthedocs.io/"
      }
      </script>

Synthetic control, for everyone.

mlsynth is an open-source Python toolbox of synthetic-control methods for
program evaluation. It implements the classical Abadie-Diamond-Hainmueller
estimator alongside a growing catalogue of modern variants -- Bayesian
spike-and-slab selection, state-space modelling, instrumental variants,
sequential difference-in-differences, matrix completion -- under a single
long-DataFrame API. Every estimator's documentation page includes a
*Verification* section that reproduces the original paper's reported numbers
where applicable.

For example, the following code reproduces Abadie, Diamond and Hainmueller's
Proposition 99 study and cross-checks it against a robust variant. It loads
the California tobacco panel shipped with the library, fits the classical
synthetic control on the canonical predictor set, then fits an
ordinary-least-squares principal-component-regression control on the same
outcomes, and prints both estimated effects:

.. code:: python

    import pandas as pd
    from mlsynth import VanillaSC, CLUSTERSC

    # California tobacco panel, 1970-2000: per-capita cigarette sales plus the
    # Abadie-Diamond-Hainmueller predictors.
    url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
           "main/basedata/augmented_cali_long.csv")
    df = pd.read_csv(url)
    df = df[df["year"] <= 2000].copy()
    df["treated"] = ((df["state"] == "California")
                     & (df["year"] >= 1989)).astype(int)

    # Three lagged-outcome predictors, as in ADH (2010), Table 2.
    for L in (1975, 1980, 1988):
        df[f"cig{L}"] = df["state"].map(
            df[df["year"] == L].set_index("state")["cigsale"])

    covariates = ["loginc", "p_cig", "pct15-24", "pc_beer",
                  "cig1975", "cig1980", "cig1988"]
    windows = {"loginc": (1980, 1988), "p_cig": (1980, 1988),
               "pct15-24": (1980, 1988), "pc_beer": (1984, 1988),
               "cig1975": (1975, 1975), "cig1980": (1980, 1980),
               "cig1988": (1988, 1988)}

    common = dict(df=df, outcome="cigsale", treat="treated",
                  unitid="state", time="year", display_graphs=False)

    # Classical synthetic control on the canonical predictor set.
    sc = VanillaSC({**common, "covariates": covariates,
                    "covariate_windows": windows,
                    "backend": "mscmt", "canonical_v": "min.loss.w",
                    "seed": 0}).fit()

    # Robust principal-component-regression control on the same outcomes.
    pcr = CLUSTERSC({**common, "method": "pcr",
                     "pcr_objective": "OLS"}).fit()

    print(f"VanillaSC ATT = {sc.att:+.2f} packs/yr")
    print(f"OLS-PCR   ATT = {pcr.att:+.2f} packs/yr")

prints::

    VanillaSC ATT = -18.98 packs/yr
    OLS-PCR   ATT = -21.39 packs/yr

The classical synthetic control reproduces ADH's headline estimate -- a
drop of about nineteen packs per capita -- and the principal-component
control, built on a different identifying assumption, agrees on a large
reduction. Fitting an estimator and then stress-testing the finding against
a second one is the everyday way mlsynth is used.

This short script is a representative example of what mlsynth can do. In
addition to classical SC, mlsynth also supports Bayesian variable selection
(:doc:`bvss`), staggered-adoption sequential difference-in-differences
(:doc:`seq_sdid`, :doc:`spsydid`), instrumental synthetic control
(:doc:`siv`), matrix completion under missingness (:doc:`mcnnm`), state-
space time-aware control (:doc:`tasc`), and clustered / robust high-
dimensional variants (:doc:`clustersc`, :doc:`mlsc`, :doc:`marex`).

For a guided tour of the estimator catalogue, start with the :doc:`about`
page. Browse the *Estimators* sidebar for the full list grouped by
methodology.

mlsynth builds on top of `numpy <https://numpy.org/>`_,
`pandas <https://pandas.pydata.org/>`_,
`scipy <https://scipy.org/>`_,
`scikit-learn <https://scikit-learn.org/>`_,
`cvxpy <https://www.cvxpy.org/>`_,
`pydantic <https://docs.pydantic.dev/>`_, and
`statsmodels <https://www.statsmodels.org/>`_; convex programs are routed
through cvxpy's solver stack.

Installation.

Install the latest version straight from GitHub:

.. code:: bash

   pip install -U git+https://github.com/jgreathouse9/mlsynth.git

mlsynth runs on Python 3.9 and later. The base install carries every core
dependency and runs every estimator except two that rely on heavier,
specialised backends. Those backends ship as optional *extras*, so you install
only the weight you use:

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Extra
     - Adds
     - Needed for
   * - ``design``
     - ``pyscipopt`` (the SCIP mixed-integer solver)
     - the experimental-design estimators :doc:`syndes` and :doc:`marex`, whose
       market-selection step is a mixed-integer quadratic program
   * - ``bayes``
     - ``numpyro`` (JAX-based MCMC)
     - :doc:`spotsynth`'s Bayesian synthetic-control mode
   * - ``all``
     - both of the above
     - the full feature set

Request an extra with bracket syntax, quoting the specifier so the shell does
not glob the brackets:

.. code:: bash

   # SCIP solver for SYNDES / MAREX
   pip install -U "mlsynth[design] @ git+https://github.com/jgreathouse9/mlsynth.git"

   # NumPyro for SPOTSYNTH's Bayesian mode
   pip install -U "mlsynth[bayes] @ git+https://github.com/jgreathouse9/mlsynth.git"

   # everything
   pip install -U "mlsynth[all] @ git+https://github.com/jgreathouse9/mlsynth.git"

Both extra backends are imported lazily, so ``import mlsynth`` and importing any
estimator class always succeed on the base install; the extra is consulted only
when you actually run the design optimiser (SYNDES / MAREX) or SPOTSYNTH's
Bayesian path, which otherwise raise a clear error naming the missing package.
``pyscipopt`` ships prebuilt wheels that bundle SCIP, so ``mlsynth[design]`` is
normally a plain install with no separate solver setup. The test suite is a
development artifact and is not shipped in the installed package -- clone the
repository to run ``pytest``.

Not sure which estimator to use? Walk the :doc:`choose` decision tree --
a sequence of identification and design questions that funnels you from
"what kind of problem do I have?" down to one or two methods, with the
catalogue grouped by family.

Community.

The mlsynth community spans economists, statisticians, and data scientists
who use synthetic-control methods for program evaluation across policy,
marketing, sports, and public health. We welcome you to join us!

* To share feature requests and bug reports, use the
  `issue tracker <https://github.com/jgreathouse9/mlsynth/issues>`_.
* To follow development, watch the
  `mlsynth repository <https://github.com/jgreathouse9/mlsynth>`_ on GitHub.

Development.

mlsynth is maintained by `Jared Greathouse
<https://jgreathouse9.github.io/>`_ (Georgia State University). The project
would not be possible without the kind efforts of and discussions with
`Jason Coupet <https://aysps.gsu.edu/profile/jason-coupet/>`_,
`Kathy Li <https://sites.utexas.edu/kathleenli/>`_,
`Mani Bayani <https://www.linkedin.com/in/mani-bayani>`_,
`Zhentao Shi <https://zhentaoshi.github.io/>`_, and
`Jaume Vives-i-Bastida <https://jvivesb.github.io/>`_, along with a growing
list of contributors.

News.

The verification campaign now covers thirty-two of the
thirty-six estimators in mlsynth -- each auditing its
implementation against its source paper, either by reproducing an
empirical Table value on the authors' own data ("Path A") or by
reproducing a Monte Carlo from the paper's simulation section
("Path B"), or against an authoritative reference implementation.
See the :doc:`replications` page for the full catalogue with
headline numbers.

.. toctree::
   :hidden:
   :caption: Get started

   about
   choose
   replications
   benchmarks
   references

.. toctree::
   :hidden:
   :caption: Observational: canonical workhorses

   vanillasc
   tssc
   fdid

.. toctree::
   :hidden:
   :caption: Observational: decomposition-first

   sbc
   hsc

.. toctree::
   :hidden:
   :caption: Observational: generalised estimand / treatment / unit

   scmo
   scta
   ctsc
   dsc
   si
   microsynth

.. toctree::
   :hidden:
   :caption: Observational: convex-hull relaxation

   iscm
   nsc


.. toctree::
   :hidden:
   :caption: Observational: No Donors

   shc

.. toctree::
   :hidden:
   :caption: Observational: high-dimensional donors

   bvss
   clustersc
   mlsc
   fscm
   masc
   msqrt
   sparse_sc
   pda
   rescm

.. toctree::
   :hidden:
   :caption: Observational: time-aware and factor models

   tasc
   fma
   dscar

.. toctree::
   :hidden:
   :caption: Observational: staggered adoption

   sdid
   seq_sdid
   ppscm
   ssc

.. toctree::
   :hidden:
   :caption: Observational: spillover-aware

   spsydid
   spillsynth
   spotsynth
   cmbsts

.. toctree::
   :hidden:
   :caption: Observational: missing data

   mcnnm
   snn
   rmsi

.. toctree::
   :hidden:
   :caption: Observational: endogenous treatment

   siv
   orthsc
   proximal

.. toctree::
   :hidden:
   :caption: Experimental design

   lexscm
   marex
   syndes
   pangeo
   geolift
   spcd
   musc
   rolldid

.. toctree::
   :hidden:
   :caption: Utilities and internals

   compare
   truncated_history
   data
   helpers

