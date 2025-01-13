About mlsynth
===========================

``mlsynth`` is a Python package that provides a suite of causal inference methods using machine learning techniques. Most of these methodologies come from the synthetic control family of methods. ``mlsynth`` calculates Average Treatment Effects on the Treated (ATTS) and, when possible, computes confidence intervals and inference statistics. ``mlsynth`` also conveniently returns the observed versus predicted values, and metrics of fit.

GitHub Repo
-----------

https://github.com/jgreathouse9/mlsynth

Installation
~~~~~~~~~~~~~

The dependencies for mlsynth are::

    pandas
    numpy
    matplotlib
    scipy
    scikit-learn
    cvxpy
    screenot
    statsmodels


``mlsynth`` may be installed from the command line like

.. code-block:: console

   $ pip install -U git+https://github.com/jgreathouse9/mlsynth.git

Methods
-------

At present, ``mlsynth`` supports:

.. list-table:: Estimators in `mlsynth`
   :widths: 30 50 20
   :header-rows: 1

   * - Estimator
     - Reference
     - Class in mlsynth
   * - `Augmented Difference-in-Differences <https://doi.org/10.1287/mksc.2022.1406>`_
     - Kathleen T. Li and Christophe Van den Bulte. "Augmented Difference-in-Differences." *Marketing Science* 2023 42:4, 746-767.
     - FDID
   * - `CLUSTERSC <#>`_
     - Saeyoung Rho, Andrew Tang, Noah Bergam, Rachel Cummings, Vishal Misra. "CLUSTERSC: Advancing Synthetic Control with Donor Clustering for Disaggregate-Level Data." (2024).
     - CLUSTERSC
   * - `Debiased Convex Regression <https://doi.org/10.1287/inte.2023.0028>`_
     - Luis Costa, Vivek F. Farias, Patricio Foncea, Jingyuan (Donna) Gan, Ayush Garg, Ivo Rosa Montenegro, Kumarjit Pathak, Tianyi Peng, Dusan Popovic. "Generalized Synthetic Control for TestOps at ABI: Models, Algorithms, and Infrastructure." *INFORMS Journal on Applied Analytics* 53(5):336-349, 2023.
     - GSC
   * - `Factor Model Approach <https://doi.org/10.1177/00222437221137533>`_
     - Kathleen T. Li, Garrett P. Sonnier. "Statistical Inference for the Factor Model Approach to Estimate Causal Effects in Quasi-Experimental Settings." *Journal of Marketing Research*, Volume 60, Issue 3.
     - FMA
   * - `Forward Difference-in-Differences <https://doi.org/10.1287/mksc.2022.1406>`_
     - Kathleen T. Li. "Frontiers: A Simple Forward Difference-in-Differences Method." *Marketing Science* 43(2):267-279, 2023.
     - FDID
   * - `Forward Selected Panel Data Approach <https://doi.org/10.1016/j.jeconom.2021.04.009>`_
     - Zhentao Shi, Jingyi Huang. "Forward-selected panel data approach for program evaluation." *Journal of Econometrics*, Volume 234, Issue 2, 2023, Pages 512-535.
     - PDA
   * - `L1PDA <https://doi.org/10.1002/jae.1230>`_
     - Kathleen T. Li, David R. Bell. "Estimation of average treatment effects with panel data: Asymptotic theory and implementation." *Journal of Econometrics*, Volume 197, Issue 1, March 2017, Pages 65-75.
     - PDA
   * - `L2-relaxation for Economic Prediction <https://doi.org/10.13140/RG.2.2.11670.97609>`_
     - Zhentao Shi, Yishu Wang. "L2-relaxation for Economic Prediction." November 2024. DOI: `10.13140/RG.2.2.11670.97609 <https://doi.org/10.13140/RG.2.2.11670.97609>`_.
     - PDA
   * - `Principal Component Regression <https://doi.org/10.1080/01621459.2021.1928513>`_
     - Agarwal, Anish, Devavrat Shah, Dennis Shen, and Dogyoon Song. "On Robustness of Principal Component Regression." *Journal of the American Statistical Association* 116 (536): 1731–45, 2021.
     - CLUSTERSC
   * - `Robust PCA Synthetic Control <https://academicworks.cuny.edu/gc_etds/4984>`_
     - Mani Bayani. "Essays on Machine Learning Methods in Economics." CUNY Academic Works, 2022.
     - CLUSTERSC
   * - `Synthetic Control Method (Vanilla SCM) <https://doi.org/10.1198/jasa.2009.ap08746>`_
     - Abadie, Alberto; Diamond, Alexis; Hainmueller, Jens. "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the American Statistical Association* 105 (490): 493–505, 2010.
     - TSSC
   * - `Two Step Synthetic Control <https://doi.org/10.1287/mnsc.2023.4878>`_
     - Kathleen T. Li, Venkatesh Shankar. "A Two-Step Synthetic Control Approach for Estimating Causal Effects of Marketing Events." *Management Science* 70(6):3734-3747, 2023.
     - TSSC

Mission
-------

The first reason I created this library can be summarized from the CausalML `Charter <https://github.com/uber/causalml/blob/master/CHARTER.md>`_, as I am:

    committed to democratizing causal machine learning through accessible, innovative, and well-documented open-source tools that empower data scientists, researchers, and organizations.

However, there is a slightly more practical reason as well. Frequently in public policy, we are concerned with estimation of casual impacts of some intervention on an outcome we care about. The longstanding traditional workhorses in this field are Difference-in-Differences methodologies and synthetic control methods, and for good reason. Difference-in-Differences is very simple to compute. Numerous advances have been made for the methodology in recent years, both in terms of econometric theory and practical implementation ([ROTH20232218]_ , [chaisesurvey]_). Equally, synthetic control methods have also become very popular amongst economists and policy analysts ([ABADIE2010]_ , [Abadie2021]_), most likely for their interpretability and ease of use in modern statistical software such as Stata or R.

However, as influential as the base toolkit had become, some important problems persist with them. For Difference-in-Differences, frequently the parallel trends assumption is impractical in a variety of real-world applications [Costa2023]_. For synthetic control methods, it is now known that computational issues with standard solvers are a bigger problem than first realized ([BECKER20181]_ , [albalate2021decoupling]_, [malo2023computing]_). Furthermore, synthetic control methods are known to be intractable in high dimensional settings, noisy outcomes, missing data, or where we are unsure on the donors pool to use ([KINN2018]_, [microsynth]_, [scmdisagg]_, [Amjad2018]_, [Agarwal2021]_, [Bayani2021]_). As a result, many developments in the causal inference literature have commonly employed machine learning methods to tackle these issues. Of course, many exciting developments already exist on this front ([aersdid]_, [FECT2024]_).


Why ``mlsynth``?
================================

Particularly in the fields of public policy and economics, synthetic control methods have existed and been used for a long while for empirical researchers. However, the more recent developments in this literature have not been as widely used (certainly by public policy scholars) as they perhaps ought to be. Of course, there are notable exceptions for Difference-in-Differences, and SCMs such as augmented synthetic controls and the synthetic Difference-in-Differences method.

Generalizing as to why this might be is hard. However, I believe this lack of use is primarily due to a host of sociological and historical reasons. Speaking for the public policy field, I believe there are a few reasons for why this is.

- Accessibility

For one, lots of these new developments simply appear in journals/conferences that many applied economists and public policy scholars do not frequent, such as *Journal of Machine Learning Research*, *Journal of Econometrics*, *Marketing Science*, *Journal of Marketing Research*, and other outlets. Thus, scholars may not take advantage of them because they do not know of them.

- Software

Another barrier to entry is the software many of these advances are written for/in (again, speaking only for the public policy field). Not a majority, but many, of the classes which appear in ``mlsynth`` had implementations only/mostly in MATLAB. As Zhentao Shi `writes <https://zhentaoshi.github.io/econ5170/intro.html>`_:

   "MATLAB [JG: and to a lesser degree Stata] may still linger in some areas in engineering, but it will be a dinosaur fossil buried under the wonderland of big data."

Beyond this, even if the software were written for a software more typical in public policy (Stata and R) or economics (Stata, R, and increasingly Python), the vast majority of the methods ``mlsynth`` implements were not wrapped into packages that provided straightforward and off the shelf use without much modification. For example, the Two Step Synthetic Control Method [TSSC]_ Forward Difference-in-Differences [Li2024]_ , Robust Synthetic Control [Amjad2018]_ , or the Factor Model Approach [li2023statistical]_ have publicly available code, but are not very user friendly. All of the public software for these approaches just listed either assumes a very specific data structure (e.g., a wide shaped data frame) or does not automate away the management of critical design elements. For example, users oftentimes must manually change things like the specification of the control group, the number of pre and post-intervention periods, or even critical things such as the number of singular values. Robust PCA Synthetic Control by [Bayani2021]_ had no public implementation, and the code for it (provided to me by my friend and coworker, Mani Bayani) was written for both R and Python, meaning that even if the code were public, analysis would need to use two softwares to use it at all.  These are barriers to entry for applied researchers to actually *use* these tools. In order for applied economists, policy analysts, and business scientists to effectively employ these methods to answer the questions they are concerend with, a simple yet robust, free, and well-documented framework should exist, one which unifies these approaches under a single banner.


Why use ``mlsynth``?
--------------------------------

Plenty of writing exists in the academic literature [causeimben]_ and popular press on the various advances in machine learning more broadly and how it may be applied for causal inference, so I will not iterate over it here. Why is ``mlsynth``  useful, then? I believe ``mlsynth`` is useful because it is an answer to the problems posed above. ``mlsynth`` has a universal and consistent syntax. It requires only a single long dataframe (where every unit is indexed to one row per time period), which consists of a unit column (a string), a numeric column for time, a numeric outcome variable, and a dummy variable denoting a unit as treated or not (1 if and when treated, else 0). In addition to its simplicity of use, it also provides all of the relevant causal effects, fit statistics, and (where applicable) inferential statistics for hypothesis testing. 

Use Cases
-----------------

- **Comparative Case Studies**: At present,  ``mlsynth`` is best suited for cases where we have a single treated unit versus many potential control units. This does not mean that it in principle many not be used due settings of staggered adoption, as the Factor Model Approach by Li and Sonnier [li2023statistical]_ or the :math:`\ell_2` relaxation by Shi and Wang [l2relax]_ come outfitted for this purpose; however, I have not yet written these extensions, so they will be present in future versions of ``mlsynth`` to broaden the use cases as much as possible. Users who wish to use them for the multiple treated unit setting/staggered adoption must extend the current code themselves.
