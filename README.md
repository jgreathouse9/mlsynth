#  ``mlsynth``


## License

`mlsynth` is open-source and distributed under the [MIT License](LICENSE).

![coverage](coverage.svg)

``mlsynth`` is a python package for doing policy evaluation using panel data estimators. See the [full documentation](https://mlsynth.readthedocs.io/) and the associated tutorials.


## What is  ``mlsynth``
 ``mlsynth`` employs synthetic control methods. It includes difference-in-differences, panel data approaches, and factor modeling.  ``mlsynth`` is a suite of tools for addressing questions like "How does Policy X affect some outcome Y". It operates on the assumption that the user has panel data, or a setup where we have observations for the same units across multiple time points.

## Installing  ``mlsynth``
 ``mlsynth`` is compatible with Python 3.8 or later. To install it, please do

    $ pip install -U git+https://github.com/jgreathouse9/mlsynth.git


Note that  ``mlsynth`` is an active project. New estimators, such as [this one](https://doi.org/10.48550/arXiv.2006.07691), will soon join the toolkit.

## Contributing to ``mlsynth``

``mlsynth`` welcomes expertise and novelty, Bayesian or Frequentist.

Small improvements or fixes are always appreciated. If you are wish to add in new estimators,
inference tests, or other wider ranging changes, please email Jared first. Some of the newer 
estimators on the list for development are [continuous treatments](https://doi.org/10.1080/07350015.2021.1927743), [some](https://economics.mit.edu/sites/default/files/inline-files/_Factor_Bayesian_SC_0.pdf) [Bayesian](https://arxiv.org/pdf/2503.06454) methods, [Random Forests](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.3123), [synthetic historical controls](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4995085), [infernce for simplex weights](https://arxiv.org/pdf/2501.15692), [prediction intervals](https://doi.org/10.1002/jae.3134), [synthetic business cycles](https://arxiv.org/pdf/2505.22388), and other things that we may add.

Whatever changes are proposed, they must be performed on the Basque, Proposition 99, or West Germany dataset, or they must reproduce the findings of the original paper the empirical innovation was proposed for.

In addition to writing code, you may also

- develop tutorials, presentations, and other educational materials using ``mlsynth``
- promote ``mlsynth`` on LinkedIn or in the classroom
- help with outreach and onboard new contributors


    
## Using  ``mlsynth``
We have implemented the following estimators for  ``mlsynth``  

 | Estimator                                | Reference | Class in  ``mlsynth`` |
| ---------------------------------------- | --------- | ---------------- |
| [Augmented Difference-in-Differences](https://doi.org/10.1287/mksc.2022.1406) | Kathleen T. Li and Christophe Van den Bulte. "Augmented Difference-in-Differences." *Marketing Science* 2023 42:4, 746-767. | FDID |
| [CLUSTERSC](#) | Saeyoung Rho, Andrew Tang, Noah Bergam, Rachel Cummings, Vishal Misra. "CLUSTERSC: Advancing Synthetic Control with Donor Clustering for Disaggregate-Level Data." (2024). | CLUSTERSC |
| [Debiased Convex Regression](https://doi.org/10.1287/inte.2023.0028) | Luis Costa, Vivek F. Farias, Patricio Foncea, Jingyuan (Donna) Gan, Ayush Garg, Ivo Rosa Montenegro, Kumarjit Pathak, Tianyi Peng, Dusan Popovic. "Generalized Synthetic Control for TestOps at ABI: Models, Algorithms, and Infrastructure." *INFORMS Journal on Applied Analytics* 53(5):336-349, 2023. | GSC |
| [Factor Model Approach](https://doi.org/10.1177/00222437221137533) | Kathleen T. Li, Garrett P. Sonnier. "Statistical Inference for the Factor Model Approach to Estimate Causal Effects in Quasi-Experimental Settings." *Journal of Marketing Research*, Volume 60, Issue 3. | FMA |
| [Forward Difference-in-Differences](https://doi.org/10.1287/mksc.2022.1406) | Kathleen T. Li. "Frontiers: A Simple Forward Difference-in-Differences Method." *Marketing Science* 43(2):267-279, 2023. | FDID |
| [Forward Selected Panel Data Approach](https://doi.org/10.1016/j.jeconom.2021.04.009) | Zhentao Shi, Jingyi Huang. "Forward-selected panel data approach for program evaluation." *Journal of Econometrics*, Volume 234, Issue 2, 2023, Pages 512-535. | PDA |
| [L1PDA](https://doi.org/10.1016/j.jeconom.2016.01.011) | Li, Kathleen T., and David R. Bell. “Estimation of Average Treatment Effects with Panel Data: Asymptotic Theory and Implementation.” J. Econom., vol. 197, no. 1, March 2017, pp. 65–75. https://doi.org/10.1016/j.jeconom.2016.01.011. | PDA |
| [L2-relaxation for Economic Prediction](https://doi.org/10.13140/RG.2.2.11670.97609) | Zhentao Shi, Yishu Wang. "L2-relaxation for Economic Prediction." November 2024. DOI: [10.13140/RG.2.2.11670.97609](https://doi.org/10.13140/RG.2.2.11670.97609). | PDA |
| [Principal Component Regression](https://doi.org/10.1080/01621459.2021.1928513) | Agarwal, Anish, Devavrat Shah, Dennis Shen, and Dogyoon Song. "On Robustness of Principal Component Regression." *Journal of the American Statistical Association* 116 (536): 1731–45, 2021. | CLUSTERSC |
| [Robust PCA Synthetic Control](https://academicworks.cuny.edu/gc_etds/4984) | Mani Bayani. "Essays on Machine Learning Methods in Economics." CUNY Academic Works, 2022. | CLUSTERSC |
| [Synthetic Control Method (Vanilla SCM)](https://doi.org/10.1198/jasa.2009.ap08746) | Abadie, Alberto; Diamond, Alexis; Hainmueller, Jens. "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the American Statistical Association* 105 (490): 493–505, 2010. | TSSC |
| [Two Step Synthetic Control](https://doi.org/10.1287/mnsc.2023.4878) | Kathleen T. Li, Venkatesh Shankar. "A Two-Step Synthetic Control Approach for Estimating Causal Effects of Marketing Events." *Management Science* 70(6):3734-3747, 2023. | TSSC |
| [Synthetic Control with Multiple Outcomes (TLP and SBMF)](https://arxiv.org/abs/2304.02272) | Wei Tian, Seojeong Lee, and Valentyn Panchenko. "Synthetic Controls with Multiple Outcomes." arXiv 2304.02272. | SCMO |
| [Synthetic Interventions](https://arxiv.org/abs/2006.07691) | Anish Agarwal, Devavrat Shah, and Dennis Shen. "Synthetic Interventions." arXiv 2006.07691. | SI |
| [Forward Selected Synthetic Control Method](https://doi.org/10.1016/j.econlet.2024.111976) | Giovanni Cerulli. "Optimal initial donor selection for the synthetic control method." Economics Letters, 2024. | FSCM |
| [Synthetic Regressing Control Method](https://arxiv.org/abs/2306.02584) | Rong J. B. Zhu. "Synthetic Regressing Control Method." arXiv 2306.02584. | SRC |
