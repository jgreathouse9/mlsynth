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
We have implemented the following estimators including the traditional method Difference-in-Difference and recent proposed methods such as Synthetic Difference-in-Difference, Matrix Completion with Nuclear Norm Minimization, and De-biased Convex Panel Regression.  

| Estimator      | Reference |
| ----------- | ----------- |
| [Difference-in-Difference (DID)](https://en.wikipedia.org/wiki/Difference_in_differences) | [Implemented through two-way fixed effects regression.](http://web.mit.edu/insong/www/pdf/FEmatch-twoway.pdf)       |
| [De-biased Convex Panel Regression (DC-PR)](https://arxiv.org/abs/2106.02780) | Vivek Farias, Andrew Li, and Tianyi Peng. "Learning treatment effects in panels with general intervention patterns." Advances in Neural Information Processing Systems 34 (2021): 14001-14013. |
| [Synthetic Difference-in-Difference (SDID)](https://arxiv.org/pdf/1812.09970.pdf)   | Dmitry Arkhangelsky, Susan Athey, David A. Hirshberg, Guido W. Imbens, and Stefan Wager. "Synthetic difference-in-differences." American Economic Review 111, no. 12 (2021): 4088-4118. |
| [Matrix Completion with Nuclear Norm Minimization (MC-NNM)](https://arxiv.org/abs/1710.10251)| Susan Athey, Mohsen Bayati, Nikolay Doudchenko, Guido Imbens, and Khashayar Khosravi. "Matrix completion methods for causal panel data models." Journal of the American Statistical Association 116, no. 536 (2021): 1716-1730. |

Please visit our [documentation](https://causaltensor.readthedocs.io/) for the usage instructions. Or check the following simple demo as a tutorial:

- [Panel Data Example](https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tutorials/Panel%20Data%20Example.ipynb)
- [Panel Data with Multiple Treatments](https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tutorials/Panel_Regression_with_Multiple_Interventions.ipynb)
- [MC-NNM with covariates and missing data](https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tests/MCNNM_test.ipynb)
