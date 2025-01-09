.. note::

   This project is under active development. Email the author with questions, comments, or bug reports.

About mlsynth
===========================

``mlsynth`` is a Python package that provides a suite of causal inference methods using machine learning techniques. Most of these methodologies come from the synthetic control family of methods. ``mlsynth`` calculates Average Treatment Effects on the Treated (ATTS) and, when possible, computes confidence intervals and inference statistics. ``mlsynth`` also conveniently returns the observed versus predicted values, and metrics of fit.

GitHub Repo
-----------

https://github.com/jgreathouse9/mlsynth

Mission
-------

The first reason I created this library can be summarized from the CausalML `Charter <https://github.com/uber/causalml/blob/master/CHARTER.md>`_, as I am:

    committed to democratizing causal machine learning through accessible, innovative, and well-documented open-source tools that empower data scientists, researchers, and organizations.

However, there is a slightly more practical reason as well. Frequently in public policy, we are concerned with estimation of casual impacts of some intervention on an outcome we care about. The longstanding traditional workhorses in this field are Difference-in-Differences methodologies and synthetic control methods, and for good reason. Difference-in-Differences is very simple to compute. Numerous advances have been made for the methodology in recent years, both in terms of econometric theory and practical implementation ([ROTH20232218]_ , [chaisesurvey]_). Equally, synthetic control methods have also become very popular amongst economists and policy analysts ([ABADIE2010]_ , [Abadie2021]_), most likely for their interpretability and ease of use in modern statistical software such as Stata or R.

However, as influential as the base toolkit had become, some important problems persist with them. For Difference-in-Differences, frequently the parallel trends assumption is impractical in a variety of real-world applications [Costa2023]_. For synthetic control methods, it is now known that computational issues with standard solvers are a bigger problem than first realized ([BECKER20181]_ , [albalate2021decoupling]_, [malo2023computing]_). Furthermore, synthetic control methods are known to be intractable in high dimensional settings, noisy outcomes, missing data, or where we are unsure on the donors pool to use ([KINN2018]_, [microsynth]_, [scmdisagg]_, [Amjad2018]_, [Agarwal2021]_, [Bayani2021]_). As a result, many developments in the causal inference literature have commonly employed machine learning methods to tackle these issues. Of course, many exciting developments already exist on this front ([aersdid], [FECT2024]).


Why ``mlsynth``?
================================

Particularly in the fields of public policy and economics, synthetic control methods have existed and been used for a long while for empirical researchers. However, the more recent developments in this literature have not been as widely used (certainly by public policy scholars) as they perhaps ought to be. Of course, there are notable exceptions for Difference-in-Differences, and SCMs such as augmented synthetic controls and the synthetic Difference-in-Differences method.

Generalizing as to why this might be is hard. However, I believe this lack of use is primarily due to a host of sociological and historical reasons. Speaking for the public policy field, I believe there are a few reasons for why this is.

- Accessibility

For one, lots of these new developments simply appear in journals/conferences that many applied economists and public policy scholars do not frequent, such as *Journal of Machine Learning Research*, *Journal of Econometrics*, *Marketing Science*, *Journal of Marketing Research*, and other outlets. Thus, scholars may not take advantage of them because they do not know of them.

- Software

Another barrier to entry is the software many of these advances are written for/in (again, speaking only for the public policy field). Not a majority, but many, of the classes which appear in ``mlsynth`` had implementations only/mostly in MATLAB. As Zhentao Shi `writes <https://zhentaoshi.github.io/econ5170/intro.html>`_:

   "MATLAB [JG: and to a lesser degree Stata] may still linger in some areas in engineering, but it will be a dinosaur fossil buried under the wonderland of big data."

Beyond this, even if the software were written for a software more typical in public policy (Stata) or economics (Stata, R, and increasingly Python), the vast majority of the methods ``mlsynth`` implements were not wrapped into packages that provided straightforward and off the shelf use without much modification. For example, the Two Step Synthetic Control Method [TSSC]_ Forward Difference-in-Differences [Li2024]_ , Robust Synthetic Control [Amjad2018]_ , or the Factor Model Approach [li2023statistical]_ have publicly available code, but are not very user friendly. All of the public software for these approaches just listed either assumes a very specific data structure (e.g., a wide shaped data frame) or does not automate away data characteristics simply. For example, in the public versions of the code for the estimators just discussed, users oftentimes must manually change things like the names of the control group, the number of pre and post-intervention periods, or even critical things such as the number of factors is singular values to specify in the regression analysis. Robust PCA Synthetic Control by [Bayani2021]_ had no public implementation, and the code for it (provided to me by my friend and coworker, Mani Bayani) was written for both R and Python. These qualities, amongst others, make it hard for applied researchers to actually *use* these econometric methods. In order for applied economists, policy analysts, and business scientists to effectively employ these methods to answer the questions they are concerend with, a simple yet robust, free, and well-documented framework should exist, one which unifies these approaches under a single banner.


Why use ``mlsynth``?
--------------------------------

Plenty of writing exists in the academic literature [causeimben]_ and popular press on the various advances in machine learning more broadly and how it may be applied for causal inference, so I will not iterate it here. Why is ``mlsynth``  useful, then? I believe ``mlsynth`` is useful because it is an answer to the problems posed above. ``mlsynth`` has a universal and consistent syntax. It requires only a single long dataframe (where every unit is indexed to one row per time period), which consists of a unit column (a string), a numeric column for time, a numeric outcome variable, and a dummy variable denoting a unit as treated or not (1 if and when treated, else 0). In addition to its simplicity of use, it also provides all of the relevant causal effects, fit statistics, and (where applicable) inferential statistics for hypothesis testing. 


Use Cases
-----------------

- **Comparative Case Studies**: At present,  ``mlsynth`` is best suited for cases where we have a single treated unit versus many potential control units. This does not mean that it in principle many not be used due settings of staggered adoption, as the Factor Model Approach by Li and Sonnier [li2023statistical]_ or the :math:`\ell_2` relaxation by Shi and Wang [l2relax]_ come outfitted for this purpose; however, I have not yet written these extensions, so they will be present in future versions of ``mlsynth`` to broaden the use cases as much as possible. Users who wish to use them for the multiple treated unit setting/staggered adoption must extend the current code themselves.
