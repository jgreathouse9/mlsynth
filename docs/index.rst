.. mlsynth documentation master file, created by
   sphinx-quickstart on Thu Jan 3 2025.


To Do List:

- Implement Placebos (time and space), possibly as a helper


.. note::

   This project is under active development. Some material is not yet documented. This project would not be possible without the kind assistance/efforts of and/or discussions with `Jason Coupet <https://aysps.gsu.edu/profile/jason-coupet/>`_, `Kathy Li <https://sites.utexas.edu/kathleenli/>`_, `Mani Bayani <https://www.linkedin.com/in/mani-bayani?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app>`_, `Zhentao Shi <https://zhentaoshi.github.io/>`_, and `Jaume Vives-i-Bastida <https://jvivesb.github.io/>`_.

Hi, I'm `Jared <https://jgreathouse9.github.io/>`_. I wrote the ``mlsynth`` package for Python. Welcome to ``mlsynth``'s documentation! ``mlsynth`` is short for "Machine-Learning Synthetic Control". It implements various Synthetic Control based methodologies for program evaluation, but it also includes difference-in-differences, panel data approaches, and factor modeling that fit within the broader artificial counterfactual setup.


.. note::

   At present, all models are implemented with only a single treated unit, though this will change in the future.


.. toctree::
   :maxdepth: 2
   :caption: Overview:

   about
   references

MLSYNTH
--------

.. toctree::
   :maxdepth: 2
   :caption: Estimators
   :titlesonly:

   fdid
   clustersc
   gsc
   fma
   pda
   tssc
   proximal
   fscm
   src
   scmo
   shc
   si


.. toctree::
   :maxdepth: 1
   :caption: Gallery
   :titlesonly:

   #auto_examples/exampleplot
   #auto_examples/Water


