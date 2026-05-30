.. mlsynth documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mlsynth 0.1.2
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

**Synthetic control, for everyone.**

mlsynth is an open-source Python toolbox of synthetic-control methods for
program evaluation. It implements the classical Abadie-Diamond-Hainmueller
estimator alongside a growing catalogue of modern variants -- Bayesian
spike-and-slab selection, state-space modelling, instrumental variants,
sequential difference-in-differences, matrix completion -- under a single
long-DataFrame API. Every estimator's documentation page includes a
*Verification* section that reproduces the original paper's reported numbers
where applicable.

For example, the following code replicates Abadie, Diamond and Hainmueller's
Proposition 99 study end-to-end. It loads the panel shipped with the
library, fits TSSC (which auto-selects between four SC-class variants based
on a pre-trends test), and prints the recommended ATT with a 95%
subsampling confidence interval:

.. code:: python

    import pandas as pd
    from mlsynth import TSSC

    # Long panel: 50 US states x 31 years of per-capita cigarette sales.
    url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
           "main/basedata/prop99_packsales.csv")
    df = pd.read_csv(url)
    df["treat"] = ((df["state"] == "California")
                    & (df["year"] >= 1989)).astype(int)

    res = TSSC({"df": df, "outcome": "cigsale", "unitid": "state",
                 "time": "year", "treat": "treat",
                 "display_graphs": False, "seed": 0}).fit()

    print(f"recommended: {res.recommended_method}")
    print(f"ATT = {res.att:+.2f} packs/yr  "
           f"(95% CI: {res.att_ci[0]:+.2f}, {res.att_ci[1]:+.2f})")

prints::

    recommended: SC
    ATT = -14.95 packs/yr  (95% CI: -16.06, -9.65)

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

**Not sure which estimator to use?** Head to :doc:`choose` for a guided
pick by treatment design, data shape, and the kind of inference you
need.

**Community.**

The mlsynth community spans economists, statisticians, and data scientists
who use synthetic-control methods for program evaluation across policy,
marketing, sports, and public health. We welcome you to join us!

* To share feature requests and bug reports, use the
  `issue tracker <https://github.com/jgreathouse9/mlsynth/issues>`_.
* To follow development, watch the
  `mlsynth repository <https://github.com/jgreathouse9/mlsynth>`_ on GitHub.

**Development.**

mlsynth is maintained by `Jared Greathouse
<https://jgreathouse9.github.io/>`_ (Georgia State University). The project
would not be possible without the kind efforts of and discussions with
`Jason Coupet <https://aysps.gsu.edu/profile/jason-coupet/>`_,
`Kathy Li <https://sites.utexas.edu/kathleenli/>`_,
`Mani Bayani <https://www.linkedin.com/in/mani-bayani>`_,
`Zhentao Shi <https://zhentaoshi.github.io/>`_, and
`Jaume Vives-i-Bastida <https://jvivesb.github.io/>`_, along with a growing
list of contributors.

**News.**

The verification campaign now covers twenty-nine of the
thirty-three estimators in mlsynth -- each auditing its
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
   references

.. toctree::
   :hidden:
   :caption: Observational: canonical workhorses

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
   :caption: Observational: high-dimensional donors

   bvss
   clustersc
   mlsc
   fscm
   sparse_sc
   pda
   rescm

.. toctree::
   :hidden:
   :caption: Observational: time-aware and factor models

   tasc
   fma

.. toctree::
   :hidden:
   :caption: Observational: staggered adoption

   sdid
   seq_sdid
   spsydid
   ppscm

.. toctree::
   :hidden:
   :caption: Observational: missing data

   mcnnm
   snn

.. toctree::
   :hidden:
   :caption: Observational: endogenous treatment

   siv
   proximal

.. toctree::
   :hidden:
   :caption: Experimental design

   lexscm
   marex
   syndes
   pangeo
   spcd
   musc

.. toctree::
   :hidden:
   :caption: Utilities and internals

   optutils
   opthelpers
   data
   selector
   spill
   helpers
   exp
   est

