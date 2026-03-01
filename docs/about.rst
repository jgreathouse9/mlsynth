About mlsynth
===========================

``mlsynth`` is a Python package that provides a suite of causal inference methods using machine learning techniques. Per the name, these methodologies come from the synthetic control family of methods. Deeveloped for comparative case studies, synthetic control methods were developed for sitations where we have one or a few exposed units to some treatment or intervention, and the researcher wishes to generate a counterfactual to understand how some metric would have evolved absent the intervention. ``mlsynth`` provides a simple way to do this. For example, consider the Basque Country example from `the original synthetic control paper <https://economics.mit.edu/sites/default/files/publications/The%20Economic%20Costs%20of%20Conflict.pdf>`_:


.. code-block:: python

   import pandas as pd
   from mlsynth import FDID

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/basque_data.csv"
   data = pd.read_csv(url)

   config = {
       "df": data,
       "outcome": data.columns[2],
       "treat": data.columns[-1],
       "unitid": data.columns[0],
       "time": data.columns[1],
       "display_graphs": True,
       "save": False,
       "counterfactual_color": ["red", "blue"]
   }

   results = FDID(config).fit()


This code estimates the causal impact of terrorist violence in the Basque Country in 1975 on the GDP per Capita, using the Forward Difference-in-Differences estimator.

GitHub Repo
-----------

https://github.com/jgreathouse9/mlsynth

Installation
~~~~~~~~~~~~~

To use ``mlsynth``, you'll need Python 3.8+ and some standard scientific packages. Follow these steps to get started.

1.   
   ``mlsynth`` is installed directly from my GitHub repository, so you need will need Git on your machine. You can check by running::

       $ git --version

   If Git is not installed, download it from `https://git-scm.com/downloads <https://git-scm.com/downloads>`_ and follow the instructions for your OS.

2.   
   It's best practice to install Python packages in an isolated environment. This way, the code lives only in the area you are currently working within. For example::

       $ python -m venv mlsynth_env
       $ source mlsynth_env/bin/activate   # Linux/macOS
       $ mlsynth_env\Scripts\activate      # Windows

3.   
   ``mlsynth`` relies on several Python packages, which are installed by default. However, if you wish to install them first, you can install them via pip::

       $ pip install pandas numpy matplotlib scipy scikit-learn cvxpy statsmodels

4.  
   Once dependencies are installed, run::

       $ pip install -U git+https://github.com/jgreathouse9/mlsynth.git

   This will download and install the latest version of ``mlsynth`` directly from the repository.

5.   
   Open a Python shell and try importing the package::

       >>> import mlsynth
       >>> mlsynth.__version__

Mission
-------

The first reason I created this library can be summarized from the CausalML `Charter <https://github.com/uber/causalml/blob/master/CHARTER.md>`_, as I am:

    committed to democratizing causal machine learning through accessible, innovative, and well-documented open-source tools that empower data scientists, researchers, and organizations.

However, there is a slightly more practical reason as well. Frequently in public policy, we are concerned with estimation of casual impacts of some intervention on an outcome we care about. The longstanding traditional workhorses in this field are Difference-in-Differences methodologies and synthetic control methods, and for good reason. Difference-in-Differences is very simple to compute. Numerous advances have been made for the methodology in recent years, both in terms of econometric theory and practical implementation ([ROTH20232218]_ , [chaisesurvey]_). Equally, synthetic control methods have also become very popular amongst economists and policy analysts ([ABADIE2010]_ , [Abadie2021]_), most likely for their interpretability and ease of use in modern statistical software such as Stata or R.

However, as influential as the base toolkit had become, some important problems persist with them. For Difference-in-Differences, frequently the parallel trends assumption is impractical in a variety of real-world applications [Costa2023]_. For synthetic control methods, it is now known that computational issues with standard solvers are a bigger problem than first realized ([BECKER20181]_ , [albalate2021decoupling]_, [malo2023computing]_). Furthermore, synthetic control methods are known to be intractable in high dimensional settings, noisy outcomes, missing data, or where we are unsure on the donors pool to use ([KINN2018]_, [microsynth]_, [scmdisagg]_, [Amjad2018]_, [Agarwal2021]_, [Bayani2021]_). As a result, many developments in the causal inference literature have commonly employed machine learning methods to tackle these issues. Of course, many exciting developments already exist on this front ([aersdid]_, [FECT2024]_).


Why develop ``mlsynth``
=======================

Particularly in the fields of public policy and economics, synthetic control methods have existed and been used for a long time by empirical researchers. However, the more recent developments in this literature have not been as widely adopted (certainly by public policy scholars) as they perhaps ought to be. Of course, there are notable exceptions for Difference-in-Differences, and SCMs such as augmented synthetic controls and the synthetic Difference-in-Differences method.

The reasons for this lack of adoption are varied, but a few key barriers stand out.

- Accessibility

Many of these new developments appear in journals or conferences that applied economists and public policy scholars may not frequent, such as *Journal of Machine Learning Research*, *Journal of Econometrics*, *Marketing Science*, and *Journal of Marketing Research*. As a result, many researchers may simply be unaware of them.

- Software

Another barrier is the software environment. Not a majority, but many of the methods implemented in ``mlsynth`` originally had implementations primarily in MATLAB. As Zhentao Shi writes `here <https://zhentaoshi.github.io/econ5170/intro.html>`_:

   "MATLAB [and to a lesser degree Stata] may still linger in some areas in engineering, but it will be a dinosaur fossil buried under the wonderland of big data."

Even when code exists for more common platforms like R, Stata, or Python, it often requires significant modification. Researchers frequently have to manually adjust aspects such as which unit is treated, the number of pre- and post-intervention periods, or tunable hyperparameters like the number of singular values. For example, the Two Step Synthetic Control Method [TSSC]_, Forward Difference-in-Differences [Li2024]_, Robust Synthetic Control [Amjad2018]_, and the Factor Model Approach [li2023statistical]_ have publicly available code, but are not user-friendly for these reasons. Robust PCA Synthetic Control by [Bayani2021]_ had no public implementation, and the code (provided by Mani Bayani) was written for both R and Python, meaning users would need two different software environments to run it.

Additionally, much of the code for modern SCM implementations requires extensive setup to run. For instance, the ``pysyncon`` library requires users to manually specify predictors, time periods, special predictors, units for treatment and control, and optimization windows. Even a single treatment effect can require dozens of lines of setup code:

.. code-block:: python

   import pandas as pd
   from pysyncon import Dataprep, RobustSynth
   df = pd.read_csv("../../data/basque.csv")

   dataprep = Dataprep(
       foo=df,
       predictors=[...],
       predictors_op="mean",
       time_predictors_prior=range(1964, 1970),
       special_predictors=[...],
       dependent="gdpcap",
       unit_variable="regionname",
       time_variable="year",
       treatment_identifier="Basque Country (Pais Vasco)",
       controls_identifier=[...],
       time_optimize_ssr=range(1960, 1970),
   )

   robust = RobustSynth()
   robust.fit(dataprep, lambda_=0.1, sv_count=2)

In contrast, ``mlsynth``*minimizes setup overhead while providing the same causal estimates. The same analysis can be performed in just a few lines:

.. code-block:: python

   import pandas as pd
   from mlsynth import CLUSTERSC

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/basque_data.csv"
   data = pd.read_csv(url)

   config = {
       "df": data,
       "outcome": data.columns[2],
       "treat": data.columns[-1],
       "unitid": data.columns[0],
       "time": data.columns[1],
       "display_graphs": True,
       "save": False,
       "method": "PCR",
       "objective": "OLS",
       "cluster": False,
       "counterfactual_color": ["red", "blue"]
   }

   results = CLUSTERSC(config).fit()

By requiring only a single long-form dataframe and a concise configuration dictionary, ``mlsynth`` allows applied researchers and data scientists to focus on analyzing results rather than wrestling with setup.

These barriers to entry highlight the need for a simple, robust, free, and well-documented framework that unifies modern SCM approaches under a single banner, enabling researchers and practitioners to more easily apply advanced causal inference methods.

A Unified and Extensible Syntax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another critical design feature of ``mlsynth`` is that many distinct estimators are implemented through a single, universal syntax. Rather than exposing users to separate classes, data preparation objects, or bespoke APIs for closely related methods, ``mlsynth`` allows researchers to move between methodological variants by changing only a small number of configuration options. For example, in Dennis Shen’s master’s thesis  
`(MIT DSpace link) <https://dspace.mit.edu/bitstream/handle/1721.1/115743/1036986794-MIT.pdf?sequence=1&isAllowed=y>`_,  a convex version of the Robust Synthetic Control method is proposed that did not ultimately appear in the JMLR publication. In ``mlsynth``, this convex RSC estimator can be implemented simply by changing the optimization objective:

.. code-block:: python

   import pandas as pd
   from mlsynth import CLUSTERSC

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/basque_data.csv"
   data = pd.read_csv(url)

   config = {
       "df": data,
       "outcome": data.columns[2],
       "treat": data.columns[-1],
       "unitid": data.columns[0],
       "time": data.columns[1],
       "display_graphs": True,
       "save": False,
       "method": "PCR",
       "objective": "SIMPLEX",
       "cluster": False,
       "counterfactual_color": ["red", "blue"]
   }

   results = CLUSTERSC(config).fit()

A closely related estimator, the *clustered* Robust Synthetic Control method, is described in  `Rho et al. (2025) <https://arxiv.org/pdf/2503.21629>`_.  This approach uses k-means clustering on the singular vectors of outcome trajectories to subset the donor pool before applying the same principal component regression method. In ``mlsynth``, this variant is enabled by toggling a single argument:

.. code-block:: python

   import pandas as pd
   from mlsynth import CLUSTERSC

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/basque_data.csv"
   data = pd.read_csv(url)

   config = {
       "df": data,
       "outcome": data.columns[2],
       "treat": data.columns[-1],
       "unitid": data.columns[0],
       "time": data.columns[1],
       "display_graphs": True,
       "save": False,
       "method": "PCR",
       "objective": "OLS",
       "cluster": True,
       "counterfactual_color": ["red", "blue"]
   }

   results = CLUSTERSC(config).fit()

Finally, ``mlsynth`` also supports a Bayesian version of the same underlying model. As with the other variants, no changes to the estimator class or data structure are required—only a change in the configuration:

.. code-block:: python

   import pandas as pd
   from mlsynth import CLUSTERSC

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/basque_data.csv"
   data = pd.read_csv(url)

   config = {
       "df": data,
       "outcome": data.columns[2],
       "treat": data.columns[-1],
       "unitid": data.columns[0],
       "time": data.columns[1],
       "display_graphs": True,
       "save": False,
       "method": "PCR",
       "Frequentist": False,
       "cluster": True,
       "counterfactual_color": ["red", "blue"]
   }

   results = CLUSTERSC(config).fit()

In this single setting, ``mlsynth`` effectively supports multiple distinct estimators: vanilla RSC with and without clustering, convex RSC with and without clustering, and Bayesian variants of the same model. While no library can implement every estimator proposed in the literature, ``mlsynth`` is designed to make entire families of methods accessible through a common, extensible interface, allowing users to easily trace each implementation back to its original methodological source. In terms of data structure, ``mlsynth`` requires only a single long dataframe (where every unit is indexed to one row per time period), which consists of a unit column (a string), a numeric column for time, a numeric outcome variable, and a dummy variable denoting a unit as treated or not (1 if and when treated, else 0). In addition to its simplicity of use, it also provides all of the relevant causal effects, fit statistics, and (where applicable) inferential statistics for hypothesis testing, without needing to call any additional arguments.


Use Cases
---------

- **Comparative Case Studies**  
  At present, ``mlsynth`` is best suited for settings with a single treated unit and many potential control units, which remains the canonical use case for synthetic control methods. While some methods implemented in ``mlsynth``—such as the Factor Model Approach by Li and Sonnier [li2023statistical]_ or the :math:`\ell_2` relaxation by Shi and Wang [l2relax]_—are, in principle, compatible with staggered adoption or multiple treated units, these extensions are not yet exposed in the current API. Support for these settings is planned for future releases. Users who require staggered adoption today would need to extend the existing codebase themselves.

- **Market-Level and Clustered Experiments**  
  More recently, ``mlsynth`` can also be used as a *design tool* for market-level experiments, rather than solely as a post-hoc estimator. This use case is described in detail in my blog post `Synthetic Controls for Marketing Experiments https://jgreathouse9.github.io/docs/scexp.html>`_. In many applied marketing and policy settings, randomized controlled trials (RCTs) are infeasible due to ethical, political, or logistical constraints. Instead of randomizing individual units, researchers may wish to design experiments at the *cluster* or *market* level—for example, neighborhoods, regions, or customer segments—while preserving balance and representativeness. Building on the experimental synthetic control framework of Abadie and Zhou Synthetic Controls for Experimental Design <https://arxiv.org/abs/2108.02196>`_, ``mlsynth`` supports the construction of *synthetic treated* and *synthetic control* groups before any intervention occurs. In this setting, synthetic control methods are not used primarily for estimation, but for *experimental planning*: deciding which units should be treated, which should serve as controls, and why.

  Concretely, ``mlsynth`` allows analysts to:
  
  - Design market experiments at the cluster level when individual randomization is infeasible  
  - Encode operational constraints such as budgets, minimum and maximum treated units per cluster, and fairness considerations  
  - Explore alternative experimental designs (e.g., base, weakly targeted, penalized, or unit-level designs) *ex ante*  
  - Assess pre-treatment fit and stability before committing resources to a real-world intervention  

  This workflow is illustrated using a simulated example inspired by a proposed “Green Stay” sustainability initiative in Curaçao, where neighborhoods are clustered and synthetic treated and control groups are constructed using mixed-integer optimization. Full mathematical details, intuition, and simulation results are provided in the blog post linked above.

  In this sense, ``mlsynth`` extends beyond traditional SCM libraries: it serves not only as a causal estimation toolkit, but also as a decision-support framework for experimental design, allowing practitioners to prototype and stress-test experiments cheaply and safely before deployment.


Methods
-------


At present, ``mlsynth`` supports:

.. list-table:: Estimators in ``mlsynth``
   :widths: 30 55 15
   :header-rows: 1

   * - Estimator
     - Reference
     - Class in ``mlsynth``

   * - `CLUSTERSC <https://doi.org/10.48550/arXiv.2503.21629>`_
     - Saeyoung Rho, Andrew Tang, Noah Bergam, Rachel Cummings, Vishal Misra. "CLUSTERSC: Advancing Synthetic Control with Donor Clustering for Disaggregate-Level Data." 2024.
     - CLUSTERSC

   * - `Factor Model Approach <https://doi.org/10.1177/00222437221137533>`_
     - Kathleen T. Li and Garrett P. Sonnier. "Statistical Inference for the Factor Model Approach to Estimate Causal Effects in Quasi-Experimental Settings." *Journal of Marketing Research* 60(3), 2023.
     - FMA

   * - `Forward Difference-in-Differences <https://doi.org/10.1287/mksc.2022.0212>`_
     - Kathleen T. Li. "Frontiers: A Simple Forward Difference-in-Differences Method." *Marketing Science* 43(2):267–279, 2023.
     - FDID

   * - `Forward Selected Panel Data Approach <https://doi.org/10.1016/j.jeconom.2021.04.009>`_
     - Zhentao Shi and Jingyi Huang. "Forward-selected panel data approach for program evaluation." *Journal of Econometrics* 234(2):512–535, 2023.
     - PDA

   * - `L1PDA <https://doi.org/10.1002/jae.1230>`_
     - Kathleen T. Li and David R. Bell. "Estimation of Average Treatment Effects with Panel Data." *Journal of Econometrics* 197(1):65–75, 2017.
     - PDA

   * - `L2-relaxation for Economic Prediction <https://doi.org/10.13140/RG.2.2.11670.97609>`_
     - Zhentao Shi and Yishu Wang. "L2-relaxation for Economic Prediction." November 2024.
     - PDA

   * - `Principal Component Regression <https://doi.org/10.1080/01621459.2021.1928513>`_
     - Anish Agarwal et al. "On Robustness of Principal Component Regression." *Journal of the American Statistical Association* 116(536):1731–1745, 2021.
     - CLUSTERSC

   * - `Robust PCA Synthetic Control <https://academicworks.cuny.edu/gc_etds/4984>`_
     - Mani Bayani. "Essays on Machine Learning Methods in Economics." CUNY Academic Works, 2022.
     - CLUSTERSC

   * - `Synthetic Control Method (Vanilla SCM) <https://doi.org/10.1198/jasa.2009.ap08746>`_
     - Alberto Abadie, Alexis Diamond, and Jens Hainmueller. "Synthetic Control Methods for Comparative Case Studies." *JASA* 105(490):493–505, 2010.
     - TSSC

   * - `Two Step Synthetic Control <https://doi.org/10.1287/mnsc.2023.4878>`_
     - Kathleen T. Li and Venkatesh Shankar. "A Two-Step Synthetic Control Approach." *Management Science* 70(6):3734–3747, 2023.
     - TSSC

   * - `Synthetic Controls for Experimental Design <https://arxiv.org/abs/2108.02196>`_
     - Alberto Abadie and Jinglong Zhao. "Synthetic Controls for Experimental Design." arXiv:2108.02196, 2025.
     - MAREX

   * - `Synthetic Control with Multiple Outcomes <https://arxiv.org/abs/2304.02272>`_
     - Wei Tian, Seojeong Lee, and Valentyn Panchenko. "Synthetic Controls with Multiple Outcomes." arXiv:2304.02272.
     - SCMO

   * - `Synthetic Interventions <https://arxiv.org/abs/2006.07691>`_
     - Anish Agarwal, Devavrat Shah, and Dennis Shen. "Synthetic Interventions." arXiv:2006.07691.
     - SI

   * - `Forward Selected Synthetic Control Method <https://doi.org/10.1016/j.econlet.2024.111976>`_
     - Giovanni Cerulli. "Optimal Initial Donor Selection for the Synthetic Control Method." *Economics Letters*, 2024.
     - FSCM

   * - `Synthetic Regressing Control Method <https://arxiv.org/abs/2306.02584>`_
     - Rong J. B. Zhu. "Synthetic Regressing Control Method." arXiv:2306.02584.
     - SRC

   * - `PI-SCM <https://arxiv.org/abs/2108.13935>`_
     - Xu Shi et al. "Theory for Identification and Inference with Synthetic Controls." arXiv:2108.13935, 2023.
     - PROXIMAL

   * - `PIS-SCM <https://arxiv.org/abs/2308.09527>`_
     - Jizhou Liu et al. "Proximal Causal Inference for Synthetic Control with Surrogates." arXiv:2308.09527, 2023.
     - PROXIMAL

   * - `Relaxation Approach to Synthetic Control <https://arxiv.org/abs/2508.01793>`_
     - Chengwang Liao, Zhentao Shi, and Yapeng Zheng. "A Relaxation Approach to Synthetic Control." arXiv:2508.01793, 2025.
     - RESCM

   * - `L1-INF Synthetic Control <https://arxiv.org/abs/2510.26053>`_
     - Le Wang, Xin Xing, and Youhui Ye. "A L-infinity Norm Synthetic Control Approach." arXiv:2510.26053, 2025.
     - RESCM



