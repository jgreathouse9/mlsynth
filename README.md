# mlsynth
 mlsynth is a python package for doing policy evaluation using panel data estimators. 


## What is mlsynth
As the name suggests, it employs synthetic control methods, also includes difference-in-differences, panel data approaches, and factor modeling. mlsynth is a suite of tools for addressing questions like "How does Policy X affect some outcome Y" given a panel data structure across more than one time point (ideally many).

## Installing CausalTensor
mlsynth is compatible with Python 3.6 or later.

    $ pip install causaltensor

Note that CausalTensor is an active project and routinely publishes new releases. In order to upgrade CausalTensor to the latest version, use pip as follows.

    $ pip install -U causaltensor
    
## Using CausalTensor
We have implemented the following estimators for mlsynth  

 | Estimator                                | Reference | Class in mlsynth |
| ---------------------------------------- | --------- | ---------------- |
| [Augmented Difference-in-Differences](https://doi.org/10.1287/mksc.2022.1406) | Kathleen T. Li and Christophe Van den Bulte. "Augmented Difference-in-Differences." *Marketing Science* 2023 42:4, 746-767. | FDID |
| [CLUSTERSC](#) | Saeyoung Rho, Andrew Tang, Noah Bergam, Rachel Cummings, Vishal Misra. "CLUSTERSC: Advancing Synthetic Control with Donor Clustering for Disaggregate-Level Data." (2024). | CLUSTERSC |
| [Debiased Convex Regression](https://doi.org/10.1287/inte.2023.0028) | Luis Costa, Vivek F. Farias, Patricio Foncea, Jingyuan (Donna) Gan, Ayush Garg, Ivo Rosa Montenegro, Kumarjit Pathak, Tianyi Peng, Dusan Popovic. "Generalized Synthetic Control for TestOps at ABI: Models, Algorithms, and Infrastructure." *INFORMS Journal on Applied Analytics* 53(5):336-349, 2023. | GSC |
| [Factor Model Approach](https://doi.org/10.1177/00222437221137533) | Kathleen T. Li, Garrett P. Sonnier. "Statistical Inference for the Factor Model Approach to Estimate Causal Effects in Quasi-Experimental Settings." *Journal of Marketing Research*, Volume 60, Issue 3. | FMA |
| [Forward Difference-in-Differences](https://doi.org/10.1287/mksc.2022.1406) | Kathleen T. Li. "Frontiers: A Simple Forward Difference-in-Differences Method." *Marketing Science* 43(2):267-279, 2023. | FDID |
| [Forward Selected Panel Data Approach](https://doi.org/10.1016/j.jeconom.2021.04.009) | Zhentao Shi, Jingyi Huang. "Forward-selected panel data approach for program evaluation." *Journal of Econometrics*, Volume 234, Issue 2, 2023, Pages 512-535. | PDA |
| [HCW](https://doi.org/10.1002/jae.1230) | Cheng Hsiao, H. Steve Ching, Shui Ki Wan. "A Panel Data Approach for Program Evaluation: Measuring the Benefits of Political and Economic Integration of Hong Kong with Mainland China." *J. Appl. Econ.*, 27:705-740, 2012. | PDA |
| [L2-relaxation for Economic Prediction](https://doi.org/10.13140/RG.2.2.11670.97609) | Zhentao Shi, Yishu Wang. "L2-relaxation for Economic Prediction." November 2024. DOI: [10.13140/RG.2.2.11670.97609](https://doi.org/10.13140/RG.2.2.11670.97609). | PDA |
| [Principal Component Regression](https://doi.org/10.1080/01621459.2021.1928513) | Agarwal, Anish, Devavrat Shah, Dennis Shen, and Dogyoon Song. "On Robustness of Principal Component Regression." *Journal of the American Statistical Association* 116 (536): 1731–45, 2021. | CLUSTERSC |
| [Robust PCA Synthetic Control](https://academicworks.cuny.edu/gc_etds/4984) | Mani Bayani. "Essays on Machine Learning Methods in Economics." CUNY Academic Works, 2022. | CLUSTERSC |
| [Synthetic Control Method (Vanilla SCM)](https://doi.org/10.1198/jasa.2009.ap08746) | Abadie, Alberto; Diamond, Alexis; Hainmueller, Jens. "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the American Statistical Association* 105 (490): 493–505, 2010. | TSSC |
| [Two Step Synthetic Control](https://doi.org/10.1287/mnsc.2023.4878) | Kathleen T. Li, Venkatesh Shankar. "A Two-Step Synthetic Control Approach for Estimating Causal Effects of Marketing Events." *Management Science* 70(6):3734-3747, 2023. | TSSC |


Please visit our [documentation](https://causaltensor.readthedocs.io/) for the usage instructions. Or check the following simple demo as a tutorial:

- [Panel Data Example](https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tutorials/Panel%20Data%20Example.ipynb)
- [Panel Data with Multiple Treatments](https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tutorials/Panel_Regression_with_Multiple_Interventions.ipynb)
- [MC-NNM with covariates and missing data](https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tests/MCNNM_test.ipynb)
