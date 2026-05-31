About mlsynth
=============

.. currentmodule:: mlsynth

**Synthetic control, with batteries included.** mlsynth is a Python
package that gives applied researchers and data scientists access to
the modern synthetic-control literature under a single, unified API.
You provide a long DataFrame and a configuration dictionary; the
estimator returns a typed result object with the ATT, the
counterfactual, donor weights, fit statistics, and (where
applicable) confidence or credible intervals.

At the time of writing, mlsynth ships more than thirty estimators
spanning the full breadth of the synthetic-control literature, from
the canonical Abadie-Diamond-Hainmueller method through Bayesian
spike-and-slab variable selection, state-space models,
matrix completion, synthetic difference-in-differences for staggered
adoption, instrumental synthetic IV, synthetic-design methods for
prospective experiments, and more. Every estimator is implemented from its
original source paper and -- for the verified subset -- replicates
the paper's published numbers in a dedicated *Verification* section.

Design philosophy
-----------------

mlsynth is built around three principles.

**1. Long DataFrame in, ATT out.**

Every estimator consumes the same long-format panel: one row per unit
per time period, with at minimum a unit identifier, a time index, an
outcome column, and a binary treatment indicator. There is no
``Dataprep`` object to construct, no pivoting to wide form, no
special-cased input for each method. The same DataFrame that fits a
:doc:`tssc` will fit :doc:`masc`.

**2. One config dict, one ``.fit()`` call.**

Estimators take a single configuration dictionary and expose a single ``.fit()`` method that returns a
frozen, typed result. There are no separate ``compute_weights`` /
``compute_counterfactual`` / ``compute_inference`` steps for the user
to assemble -- the orchestration is the estimator's job.

**3. Every estimator is verified against its source.**

Most synthetic-control libraries ship "an implementation"; the user
trusts it does what the paper says. mlsynth's *Verification campaign*
holds every estimator to a stronger contract: each
:file:`docs/<estimator>.rst` page contains (or will contain) a
*Verification* section that reproduces one of the source paper's
reported numbers -- either by replicating an empirical result on the
authors' own dataset ("Path A") or by reproducing a Monte Carlo from
the paper's simulation section ("Path B").

The unified API in action
-------------------------

The same data, the same DataFrame, four different estimators -- with
only the keys in the configuration dictionary changing. The example
below moves between vanilla Robust SCM, its convex variant from Dennis
Shen's MIT master's thesis (`MIT DSpace
<https://dspace.mit.edu/bitstream/handle/1721.1/115743/1036986794-MIT.pdf?sequence=1&isAllowed=y>`_),
the clustered variant of `Rho et al. (2025)
<https://arxiv.org/pdf/2503.21629>`_, and `the Bayesian variant<https://jmlr.csail.mit.edu/papers/volume19/17-777/17-777.pdf>`_ -- all
via the same :class:`~mlsynth.CLUSTERSC` class:

.. code-block:: python

    import pandas as pd
    from mlsynth import CLUSTERSC

    url = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
           "main/basedata/basque_data.csv")
    data = pd.read_csv(url)
    base = {"df": data, "outcome": data.columns[3],
            "treat": data.columns[-1], "unitid": data.columns[1],
            "time": data.columns[2], "display_graphs": True}

    variants = [
        ("Vanilla RSC",   {**base, "method": "PCR", "objective": "OLS"}),
        ("Convex RSC",    {**base, "method": "PCR", "objective": "SIMPLEX"}),
        ("Clustered RSC", {**base, "method": "PCR", "objective": "OLS",
                            "cluster": True}),
        ("Bayesian RSC",  {**base, "method": "PCR", "Frequentist": False,
                            "cluster": True}),
    ]

    for name, cfg in variants:
        res = CLUSTERSC(cfg).fit()
        print(f"{name:15} ATT = {res.att:+.3f}")

Four estimators, one DataFrame, four lines of differences. The same
pattern recurs throughout the library, depending on the circumstance.

The verification campaign
-------------------------

mlsynth, as much as possible, reproduces against the source
paper. Each verified estimator's documentation page contains a
replication section showing that the implementation matches one of
the paper's headline numbers. We distinguish between **Path A** (empirical replication
on the original authors' dataset, matching their published
estimates) and **Path B** (Monte Carlo replication of the paper's
simulation section). Where both paths are feasible, both are run;
where the authors' data is not redistributable/easily accessible, Path B is used.

See :doc:`replications` for the catalogue of all current
replications, with headline numbers and per-family coverage status.

Installation
------------

mlsynth requires Python 3.9 or later and standard scientific
dependencies. The simplest install is from PyPI::

    pip install mlsynth

For the development version directly from GitHub::

    pip install -U git+https://github.com/jgreathouse9/mlsynth.git

Confirm the install::

    >>> import mlsynth
    >>> mlsynth.__version__

For an isolated environment, the standard
``python -m venv mlsynth_env && source mlsynth_env/bin/activate``
pattern works as expected.

For a fuller tour of which estimator fits which problem, see
:doc:`choose`.

Use cases
---------

mlsynth is a general-purpose synthetic-control toolkit. The
applications below are the ones the library has been used for most
heavily in practice.

**Observational causal inference.** Estimate the average treatment
effect of a policy, a regulatory change, a marketing intervention, or
a supply shock from a panel of already-observed outcomes. This is
the canonical comparative-case-study setting that synthetic control
was built for; mlsynth supplies more than two dozen estimators
covering low-dim, high-dim, staggered, and instrumental variants.

**Experimental design at the market level.** When randomising
individual units is infeasible -- as in geo-marketing experiments,
cluster-level public-health interventions, or market-level pricing
studies -- mlsynth supports the design-stage problem of choosing
which units to treat, before any intervention takes place. See
:doc:`syndes`, :doc:`marex`, :doc:`pangeo`, :doc:`spcd`, and
:doc:`lexscm`. This use case is described in detail in
`Synthetic Controls for Marketing Experiments
<https://jgreathouse9.github.io/docs/scexp.html>`_.

**High-dimensional donor pools.** When :math:`N \gg T_0` -- as
arises with commodity-category panels, large product portfolios, or
fine industry classifications, or we have many covariates to choose from, the classical SC quadratic program
loses its unique solution. Lasso-style alternatives sometimes over-select.
:doc:`bvss`, :doc:`clustersc`, :doc:`mlsc`, :doc:`rescm`,
:doc:`sparse_sc`, :doc:`fscm`, and :doc:`pda` each address this
regime with a different selection strategy.

**Special data structures.** Multiple outcomes (:doc:`scmo`),
continuous treatments (:doc:`ctsc`), full distributional ATTs
(:doc:`dsc`), multiple treatment arms (:doc:`si`), individual-level
units (:doc:`microsynth`), missing outcome cells (:doc:`mcnnm`,
:doc:`snn`), and strongly trending series (:doc:`tasc`, :doc:`sbc`,
:doc:`hsc`) each have estimators built for them.

Citation
--------

If you use mlsynth in academic work, please cite the library as:

.. code-block:: bibtex

   @software{mlsynth,
     author  = {Greathouse, Jared},
     title   = {{mlsynth}: A Python Toolbox of Synthetic-Control Methods
                for Program Evaluation},
     year    = {2025},
     version = {0.1.2},
     url     = {https://github.com/jgreathouse9/mlsynth},
   }

Please also cite the original source paper of any estimator you use
in your analysis. The reference for each estimator is listed at the
top of its documentation page; a consolidated bibliography is in
:doc:`references`.

Acknowledgments
---------------

mlsynth was developed by `Jared Greathouse
<https://jgreathouse9.github.io/>`_ at Georgia State University and
benefited from advice, code contributions, and methodological
discussions with `Jason Coupet
<https://aysps.gsu.edu/profile/jason-coupet/>`_, `Kathy Li
<https://sites.utexas.edu/kathleenli/>`_, `Mani Bayani
<https://www.linkedin.com/in/mani-bayani>`_, `Zhentao Shi
<https://zhentaoshi.github.io/>`_, and `Jaume Vives-i-Bastida
<https://jvivesb.github.io/>`_. The Robust PCA Synthetic Control
implementation in particular would not exist without Mani Bayani's
original code contribution.

Roadmap
-------

Current development priorities, in roughly the order they will land:

* **Verification coverage to 100%.** Extend the Path A / Path B
  campaign to every estimator in the library.
* **A unified result-object contract.** All estimators currently
  expose an ATT, a counterfactual path, and (where applicable) a CI,
  but the attribute names differ across families. A documented
  minimum contract (``result.att``, ``result.att_ci``,
  ``result.counterfactual``, ``result.weights``, ``result.pre_rmse``)
  is in design.
* **Staggered-adoption expansion.** Several estimators currently
  built for a single treated unit (the :math:`\ell_2` relaxation of
  Shi & Wang, the Factor Model Approach of Li & Sonnier) are in
  principle compatible with staggered adoption; exposing those
  extensions in the API is planned.
* **A comparison matrix.** A single table indexing which estimators
  produce CIs, which handle staggered adoption, which scale to
  large :math:`N`, which require covariates, etc.
* **Performance work.** Several per-unit synthetic-control fits
  currently route through cvxpy's compilation tax; replacing with
  closed-form simplex projection where possible.

If you would like to contribute to any of the above, see the GitHub
`issue tracker <https://github.com/jgreathouse9/mlsynth/issues>`_.
