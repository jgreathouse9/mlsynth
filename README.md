#  ``mlsynth``


## License

`mlsynth` is open-source and distributed under the [MIT License](LICENSE).

![coverage](coverage.svg)

``mlsynth`` is a python package for doing policy evaluation using panel data estimators. See the [full documentation](https://mlsynth.readthedocs.io/) and the [landing page](https://jgreathouse9.github.io/mlsynth/webdir/mlsynthlanding.html) for more.


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

| Estimator                                                                                   | Reference                                                                                                                                                                                                                                                 | Class in `mlsynth` |
| ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| [L1PDA](https://doi.org/10.1016/j.jeconom.2016.01.011)                                      | Li, Kathleen T., and David R. Bell. “Estimation of Average Treatment Effects with Panel Data: Asymptotic Theory and Implementation.” *Journal of Econometrics* 197(1):65–75, 2017.                                                                        | PDA                |
| [Synthetic Control Method (Vanilla SCM)](https://doi.org/10.1198/jasa.2009.ap08746)         | Abadie, Alberto; Diamond, Alexis; Hainmueller, Jens. "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the American Statistical Association* 105 (490): 493–505, 2010. | TSSC               |
| [Forward Selected Panel Data Approach](https://doi.org/10.1016/j.jeconom.2021.04.009)       | Zhentao Shi, Jingyi Huang. "Forward-selected panel data approach for program evaluation." *Journal of Econometrics*, 234(2):512-535, 2023.                                                                                                                | PDA                |
| [Principal Component Regression](https://doi.org/10.1080/01621459.2021.1928513)             | Agarwal, Anish; Shah, Devavrat; Shen, Dennis; Song, Dogyoon. "On Robustness of Principal Component Regression." *Journal of the American Statistical Association* 116(536):1731–1745, 2021.                                                               | CLUSTERSC          |
| [Factor Model Approach](https://doi.org/10.1177/00222437221137533)                          | Li, Kathleen T.; Sonnier, Garrett P. "Statistical Inference for the Factor Model Approach to Estimate Causal Effects in Quasi-Experimental Settings." *Journal of Marketing Research* 60(3), 2023.                                                        | FMA                |
| [Augmented Difference-in-Differences](https://doi.org/10.1287/mksc.2022.1406)               | Li, Kathleen T.; Van den Bulte, Christophe. "Augmented Difference-in-Differences." *Marketing Science* 42(4):746-767, 2023.                                                                                                                               | FDID               |
| [Forward Difference-in-Differences](https://doi.org/10.1287/mksc.2022.0212)                 | Li, Kathleen T. "Frontiers: A Simple Forward Difference-in-Differences Method." *Marketing Science* 43(2):267-279, 2023.                                                                                                                                  | FDID               |
| [Two Step Synthetic Control](https://doi.org/10.1287/mnsc.2023.4878)                        | Li, Kathleen T.; Shankar, Venkatesh. "A Two-Step Synthetic Control Approach for Estimating Causal Effects of Marketing Events." *Management Science* 70(6):3734-3755, 2023.                                                                               | TSSC               |
| [Synthetic Control Method with Nonlinear Outcomes](https://arxiv.org/abs/2306.01967)        | Tian, Wei. "The Synthetic Control Method with Nonlinear Outcomes: Estimating the Impact of the 2019 Anti-Extradition Law Amendments Bill Protests on Hong Kong's Economy." arXiv:2306.01967, 2023.                                                        | NSC                |
| [Proximal Causal Inference for SCM (Surrogates)](https://arxiv.org/abs/2308.09527)          | Liu, Jizhou; Tchetgen Tchetgen, Eric J.; Varjão, Carlos. "Proximal Causal Inference for Synthetic Control with Surrogates." arXiv:2308.09527, 2023.                                                                                                       | PROXIMAL           |
| [Proximal SCM Framework](https://arxiv.org/abs/2108.13935)                                  | Shi, Xu; Li, Kendrick; Miao, Wang; Hu, Mengtong; Tchetgen Tchetgen, Eric J. "Theory for Identification and Inference with Synthetic Controls: A Proximal Causal Inference Framework." arXiv:2108.13935, 2023.                                             | PROXIMAL           |
| [Robust PCA Synthetic Control](https://academicworks.cuny.edu/gc_etds/4984)                 | Bayani, Mani. "Essays on Machine Learning Methods in Economics." CUNY Academic Works, 2022.                                                                                                                                                               | CLUSTERSC          |
| [CLUSTERSC](https://doi.org/10.48550/arXiv.2503.21629)                                      | Rho, Saeyoung; Tang, Andrew; Bergam, Noah; Cummings, Rachel; Misra, Vishal. "CLUSTERSC: Advancing Synthetic Control with Donor Clustering for Disaggregate-Level Data." (2024).                                                                           | CLUSTERSC          |
| [Optimal Initial Donor Selection for SCM](https://doi.org/10.1016/j.econlet.2024.111976)    | Cerulli, Giovanni. "Optimal initial donor selection for the synthetic control method." *Economics Letters* 244:111976, 2024.                                                                                                                              | FSCM               |
| [L2-relaxation for Economic Prediction](https://doi.org/10.13140/RG.2.2.11670.97609)        | Shi, Zhentao; Wang, Yishu. "L2-relaxation for Economic Prediction." November 2024. DOI:10.13140/RG.2.2.11670.97609.                                                                                                                                       | PDA                |
| [Synthetic Historical Control](https://ssrn.com/abstract=4995085)                           | Chen, Yi-Ting; Yang, Jui-Chung; Yang, Tzu-Ting. "Synthetic Historical Control for Policy Evaluation." SSRN, September 2024. DOI:10.2139/ssrn.4995085.                                                                                                     | SHC                |
| [Relaxed Balanced Synthetic Control](https://arxiv.org/abs/2508.01793)                      | Liao, Chengwang; Shi, Zhentao; Zheng, Yapeng. "A Relaxation Approach to Synthetic Control." arXiv:2508.01793, 2025.                                                                                                                                       | RESCM              |
| [Synthetic Control with Multiple Outcomes (TLP and SBMF)](https://arxiv.org/abs/2304.02272) | Tian, Wei; Lee, Seojeong; Panchenko, Valentyn. "Synthetic Controls with Multiple Outcomes." arXiv:2304.02272.                                                                                                                                             | SCMO               |
| [Synthetic Controls for Experimental Design](https://arxiv.org/abs/2108.02196)              | Abadie, Alberto; Zhao, Jinglong. "Synthetic Controls for Experimental Design." arXiv:2108.02196, 2025.                                                                                                                                                    | MAREX              |


