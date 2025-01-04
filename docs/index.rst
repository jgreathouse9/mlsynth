.. mlsynth documentation master file, created by
   sphinx-quickstart on Thu Jan 3 2025.

.. note::

   This project is under active development. As you can see, most of the material is not yet documented. This project would not be possible without the kind assistance/efforts of and/or discussions with Jason Coupet, Kathy Li, Mani Bayani, Zhentao Shi, and Jaume Vives-i-Bastida.

Hi, I'm `Jared <https://jgreathouse9.github.io/>`_. Welcome to mlsynth's documentation! mlsynth is short for "Machine-Learning Synthetic Control" methods, due to it implementing various Synthetic Control based methodologies for program evaluation. The library also includes difference-in-differences, panel data approaches, and factor modeling.

The way you install mlsynth is by doing, from the command line,

.. code-block:: console

   $ pip install -U git+https://github.com/jgreathouse9/mlsynth.git

which simply installs the latest release from my GitHub.

At present (though undocumented at present), ``mlsynth`` supports the following estimators:

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
   * - `HCW <https://doi.org/10.1002/jae.1230>`_
     - Cheng Hsiao, H. Steve Ching, Shui Ki Wan. "A Panel Data Approach for Program Evaluation: Measuring the Benefits of Political and Economic Integration of Hong Kong with Mainland China." *J. Appl. Econ.*, 27:705-740, 2012.
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


Contents:

.. toctree::
   :maxdepth: 2
   :caption: Overview:

   about
   references

MLSYNTH
--------

.. toctree::
   :maxdepth: 1
   :caption: Classes

   fdid
   clustersc
   gsc
   fma
   pda
   tssc
