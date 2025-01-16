Proximal Synthetic Control
==========================

Next, I discuss the proximal SCM method. This method is a GMM based estimator of the treatment effect, where some units are used as surrogates (or, predictive of the latent unit specific factors which drive the outcome) and proxies, unaffected elements that vary through time and match on the time latent common factors.


.. autoclass:: mlsynth.mlsynth.PROXIMAL
   :show-inheritance:
   :special-members: __init__




Estimating Proximal SCM via ``mlsynth``
--------------------------------------------


This is the plot we get when we estimate the causal impact.


.. image:: https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/examples/PROXIMAL/PanicProx.png
   :alt: Proximal Synthetic Control Estimation
   :align: center
   :width: 600px


As we can see, even when we use only post-intervention data to estimate the causal impact, the result largely agrees with the original estimates.
