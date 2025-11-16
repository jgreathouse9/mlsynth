Forward Difference-in-Differences
==================================

Forward DID uses a greedy algorithm to buildthe donor set iteratively, starting from an empty set. At each iteration, it selects the control unit that, together with the previously selected units, minimizes the pre-treatment loss. Let a candidate donor subset be denoted by :math:`\widehat{U} \subseteq \mathcal{N}_0`. In Forward DID, all selected units are weighted equally, so the uniform weight vector is

.. math::
    \mathbf{w} \in \mathcal{W}_{\text{unif}}(\widehat{U}) =
    \Big\{ w_j = \frac{1}{|\widehat{U}|} \text{ for } j \in \widehat{U}, \; w_j = 0 \text{ for } j \notin \widehat{U} \Big\}.

The inner loss function for a donor subset :math:`\widehat{U}` is

.. math::
    \ell_{\text{FDID}}(\widehat{U}) =
    \min_{\beta \in \mathbb{R}} 
    \Bigg\| 
        \mathbf{y}_1 - \beta \mathbf{1} - \frac{1}{|\widehat{U}|} \sum_{j \in \widehat{U}} \mathbf{y}_j
    \Bigg\|_2^2.

The outer optimization selects the subset that minimizes this loss:

.. math::
    \widehat{U}^\ast_{\text{FDID}} = 
    \operatorname*{argmin}_{\widehat{U} \subseteq \mathcal{N}_0} 
    \ell_{\text{FDID}}(\widehat{U}).

The iterative forward selection algorithm proceeds as follows:

1. Initialize the selected set :math:`\widehat{U}_0 = \emptyset`.
2. At iteration :math:`k`, compute the inner loss for each candidate donor :math:`j \in \mathcal{N}_0 \setminus \widehat{U}_{k-1}`:

   .. math::
       \ell_{\text{FDID}}(\widehat{U}_{k-1} \cup \{j\})

3. Select the donor that minimizes the inner loss:

   .. math::
       (\widehat{U}_k, \ell^\ast_k) = 
       \operatorname*{argmin}_{j \in \mathcal{N}_0 \setminus \widehat{U}_{k-1}} 
       \ell_{\text{FDID}}(\widehat{U}_{k-1} \cup \{j\})

4. Add this donor to the selected set and repeat until all units are considered or a stopping criterion is met.

Because FDID only estimates one parameter (the pre-treatment mean difference), overfitting is impossible. Each inner loss is trivial to compute, and the resulting selected donor subset receives uniform weights automatically.

Implementing FDID via mlsynth
-----------------------------

.. autoclass:: mlsynth.FDID
   :show-inheritance:
   :members:
   :special-members: __init__

