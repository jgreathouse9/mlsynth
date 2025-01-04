About mlsynth
===========================

``mlsynth`` is a Python package that provides a suite of causal inference methods using machine learning techniques. Most of these emthodologies come from the synthetic control family of methods. ``mlsynth`` calculates Average Treatment Effects on the Treated (ATTS) and oftentimes, when possible, computes confidence intervals and inference statistics. ``mlsynth`` also conveniently returns these, as well as the observed versus fitted values, as well as a host of metrics of fit.

GitHub Repo
-----------

https://github.com/jgreathouse9/mlsynth

Mission
-------

The first reason I created this library can be summarized from the CausalML `Charter <https://github.com/uber/causalml/blob/master/CHARTER.md>`_, as I am:

    committed to democratizing causal machine learning through accessible, innovative, and well-documented open-source tools that empower data scientists, researchers, and organizations.

However, there is a slightly more practical reason as well. Frequently in public policy, we are concerned with estimation of casual impacts of some intervention on an outcome we care about. The longstanding traditional workhorses in this field are Difference-in-Differences methodologies and synthetic control methods, and for good reason. Difference-in-Differences is very simple to compute. Numerous advances have been made for the methodology in recent years, both in terms of econometric theory and practical implementation ([ROTH20232218]_ , [chaisesurvey]_). Equally, synthetic control methods have also become very popular amongst economists and policy analysts ([ABADIE2010]_ , [Abadie2021]_), most likely for their interpretability and ease of use in modern statistical software such as Stata or R.

However, as influential as the base toolkit had become, some important problems persist with them. For Difference-in-Differences, frequently the parallel trends assumption is impractical in a variety of real-world applications [Costa2023]_. For synthetic control methods, it is now known that computational issues with standard solvers are a bigger problem than first realized ([BECKER20181]_ , [albalate2021decoupling]_, [malo2023computing]_). Furthermore, synthetic control methods are known to be intractable in high dimensional settings, noisy outcomes, missing data, or where we are unsure on the donors pool to use ([KINN2018]_, [microsynth]_, [scmdisagg]_, [Amjad2018]_, [Agarwal2021]_, [Bayani2021]_). Of course, many exciting developments already exist on this front ([aersdid], [FECT2024]).

As a result, many developments in the causal inference literature have commonly employed machine learning methods to tackle these issues



Why ``mlsynth``?
================================

Particularly in the field of public policy and to a much lesser degree economics, synthetic control methods have existed and been used for a long while. However, the more recent developments in the literature have not been as widely used (certainly by public policy scholars) as they perhaps ought to be (with notable exceptions for Difference-in-Differences, and SCMs such as augmented synthetic controls and the synthetic Difference-in-Differences method).

While generalizing for why this might be is hard, I believe this is primarily due to a host of sociological and historical reasons. Speaking for the public policy field, I believe there are a few reasons for why this is (of course, there may be others)

- Accessibility

For one, lots of these new developments simply appear in journals/conferences that many applied economists and public policy scholars do not frequent, such as *Journal of Machine Learning Research*, *Journal of Econometrics*, *Marketing Science*, *Journal of Marketing Research*, and other outlets. On one hand, scholars may not take advantage of them because they do not know of them.

- Software

Another barrier to entry is software many of these advances are written for/in (again, speaking only for the public policy field). Not a majority, but many, of the classes which appear in ``mlsynth`` had implementations only/mostly in MATLAB. As Zhentao Shi `writes <https://zhentaoshi.github.io/econ5170/intro.html>`_:

   "MATLAB [JG: and to a lesser degree Stata] may still linger in some areas in engineering, but it will be a dinosaur fossil buried under the wonderland of big data."

Beyond this, even if the software were written for a software more typical in public policy (Stata) or economics (Stata, R, and increasingly Python), the vast majority of the methods ``mlsynth`` implements were not wrapped into packages that provided straightforward and off the shelf use without much modification. For example, the Two Step Synthetic Control Method [TSSC]_ Forward Difference-in-Differences [Li2024]_ , Robust Synthetic Control [Amjad2018]_ , or the Factor Model Approach [li2023statistical]_ have publicly available code, but are not very user friendly. All of the public software for these approaches just listed either assumes a very specific data structure (e.g., a wide shaped data frame) or does not automate away data characteristics simply. For example, in the public versions of the code for the estimators just discussed, users oftentimes must manually change things like the names of the control group, the number of pre and post-intervention periods, or even critical things such as the number of factors is singular values to specify in the regression analysis. Robust PCA Synthetic Control by [Bayani2021]_ had no public implementation, and the code for it (provided to me by my friend and coworker, Mani Bayani) was written for both R and Python. These qualities, amongst others, make actually *using* these econometric methods to answer questions applied economists, policy analysts, and business scientists difficult at best.


Why use ``mlsynth``?
--------------------------------

Plenty of writing exists in the academic literature [causeimben]_ and popular press on the various advances in machine learning more broadly and how it may be applied for a host of scenarios, so I will not iterate it here. Why is ``mlsynth``  useful, then? I believe ``mlsynth`` is useful because it is an answer to the problems posed above. ``mlsynth`` has a universal and consistent syntax. It requires only a single long dataframe (where every unit is indexed to one row per time period), which consists of a unit column (a string), a numeric column for time, a numeric outcome variable, and a dummy variable denoting a unit as treated or not (1 if and when treated, else 0). In addition to its simplicity of use, it also provides all of the relevant causal effects, fit statistics, and (where applicable) inferential statistics for hypothesis testing. 

Causal machine learning is a branch of machine learning that focuses on understanding the cause and effect relationships in data. It goes beyond just predicting outcomes based on patterns in the data, and tries to understand how changing one variable can affect an outcome.
Suppose we are trying to predict a student’s test score based on how many hours they study and how much sleep they get. Traditional machine learning models would find patterns in the data, like students who study more or sleep more tend to get higher scores.
But what if you want to know what would happen if a student studied an extra hour each day? Or slept an extra hour each night? Modeling these potential outcomes or counterfactuals is where causal machine learning comes in. It tries to understand cause-and-effect relationships - how much changing one variable (like study hours or sleep hours) will affect the outcome (the test score).
This is useful in many fields, including economics, healthcare, and policy making, where understanding the impact of interventions is crucial.
While traditional machine learning is great for prediction, causal machine learning helps us understand the difference in outcomes due to interventions.



Difference from Traditional Machine Learning
--------------------------------------------

Traditional machine learning and causal machine learning are both powerful tools, but they serve different purposes and answer different types of questions.
Traditional Machine Learning is primarily concerned with prediction. Given a set of input features, it learns a function from the data that can predict an outcome. It’s great at finding patterns and correlations in large datasets, but it doesn’t tell us about the cause-and-effect relationships between variables. It answers questions like “Given a patient’s symptoms, what disease are they likely to have?”
On the other hand, Causal Machine Learning is concerned with understanding the cause-and-effect relationships between variables. It goes beyond prediction and tries to answer questions about intervention: “What will happen if we change this variable?” For example, in a medical context, it could help answer questions like “What will happen if a patient takes this medication?”
In essence, while traditional machine learning can tell us “what is”, causal machine learning can help us understand “what if”. This makes causal machine learning particularly useful in fields where we need to make decisions based on data, such as policy making, economics, and healthcare.


Measuring Causal Effects
------------------------

**Randomized Control Trials (RCT)** are the gold standard for causal effect measurements.  Subjects are randomly exposed to a treatment and the Average Treatment Effect (ATE) is measured as the difference between the mean effects in the treatment and control groups.  Random assignment removes the effect of any confounders on the treatment.

If an RCT is available and the treatment effects are heterogeneous across covariates, measuring the conditional average treatment effect(CATE) can be of interest.  The CATE is an estimate of the treatment effect conditioned on all available experiment covariates and confounders.  We call these Heterogeneous Treatment Effects (HTEs).


Example Use Cases
-----------------

- **Campaign Targeting Optimization**: An important lever to increase ROI in an advertising campaign is to target the ad to the set of customers who will have a favorable response in a given KPI such as engagement or sales. CATE identifies these customers by estimating the effect of the KPI from ad exposure at the individual level from A/B experiment or historical observational data.

- **Personalized Engagement**: A company might have multiple options to interact with its customers such as different product choices in up-sell or different messaging channels for communications. One can use CATE to estimate the heterogeneous treatment effect for each customer and treatment option combination for an optimal personalized engagement experience.
