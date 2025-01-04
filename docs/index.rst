.. mlsynth documentation master file, created by
   sphinx-quickstart on Thu Jan 3 2025.

Hi, I'm `Jared <https://jgreathouse9.github.io/>`_. Welcome to mlsynth's documentation! mlsynth is short for "Machine-Learning Synthetic Control" methods, due to it implementing various Synthetic Control based methodologies for program evaluation. The library also includes difference-in-differences, panel data approaches, and factor modeling.

The way you install mlsynth is by doing, from the command line,

.. code-block:: console

   $ pip install -U git+https://github.com/jgreathouse9/mlsynth.git

which simply installs the latest release from my GitHub.

Contents:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   about
   classes

Classes
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
