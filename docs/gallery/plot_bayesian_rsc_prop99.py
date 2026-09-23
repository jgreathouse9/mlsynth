"""
Bayesian Robust Synthetic Control: uncertainty on Proposition 99
================================================================

The Basque example draws a single counterfactual line. A synthetic control is an
estimate, though, and an estimate without a sense of its uncertainty is only half
an answer. This example reproduces the Bayesian treatment of Robust Synthetic
Control from Amjad, Shah and Shen (2018, *Robust Synthetic Control*, JMLR
19(22):1-51, Sections 4.4 and 5.2), which returns not a line but a posterior: a
predictive mean and a band around it whose width is the model's own statement of
how much it knows.

Robust Synthetic Control first de-noises the donor pool -- it keeps only the top
few singular values of the donor matrix, on the premise that a handful of latent
factors drive the panel and the rest is noise -- then regresses the treated unit
on the de-noised donors. The Bayesian version puts a Gaussian prior on the donor
weights and reads off the closed-form posterior, whose predictive variance at
each period is

.. math::

   \\sigma^2_{D,t} = \\sigma^2 + \\hat{\\mathbf{M}}_{\\cdot t}^\\top\\,
   \\Sigma_D\\,\\hat{\\mathbf{M}}_{\\cdot t}

(Amjad-Shah-Shen eq. 43): the observation noise :math:`\\sigma^2` plus the
uncertainty the posterior weight covariance :math:`\\Sigma_D` propagates through
the donors. We plot the posterior mean with a band one standard deviation wide on
each side, as the paper does, and reproduce its central observation for
California: the band's width is governed by how many singular values you keep.

We call the Bayesian RSC kernel that ``CLUSTERSC(estimator="bayesian")`` is built
on -- :func:`mlsynth.utils.pcr.core.hsvt` for the de-noising and
:func:`mlsynth.utils.clustersc_helpers.pcr.bayesian.BayesSCM` for the posterior --
so the per-period predictive band can be drawn directly, on California
Proposition 99 (California treated in 1989, 38 donor states, per-capita cigarette
pack sales, 1970-2000).
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# The Bayesian RSC posterior
# --------------------------
# Load the ADH Proposition 99 panel and cast it wide (states x years). The donor
# pool is every state other than California; the pre-period runs 1970-1988.
# ``sigma2`` is the observation-noise plug-in for the predictive variance -- the
# pre-period variance of the treated series -- and the Gaussian prior precision
# (``alpha``) is left at 1. The retained rank is the knob the paper studies.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlsynth.utils.pcr.core import hsvt
from mlsynth.utils.clustersc_helpers.pcr.bayesian import BayesSCM

url = (
    "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
    "refs/heads/main/basedata/smoking_data.csv"
)
panel = pd.read_csv(url)
wide = panel.pivot(index="state", columns="year", values="cigsale").sort_index()

years = np.asarray(wide.columns, dtype=int)
treat_year = 1989
T0 = int((years < treat_year).sum())
donors = [s for s in wide.index if s != "California"]

observed = wide.loc["California"].to_numpy(dtype=float)
donor_full = wide.loc[donors].to_numpy(dtype=float).T      # (T, J)


def bayesian_rsc(rank):
    """Rank-r HSVT de-noising + Gaussian posterior; return mean and per-period SD."""
    sigma2 = float(np.var(observed[:T0], ddof=1))
    denoised = hsvt(donor_full.T, rank=rank)[0]            # (J, T)
    design_pre = denoised[:, :T0].T                        # (T0, J)
    weights, cov, _, _ = BayesSCM(design_pre, observed[:T0], sigma2, 1.0)
    mean = denoised.T @ weights
    predictive_sd = np.sqrt(
        sigma2 + np.einsum("ti,ij,tj->t", denoised.T, cov, denoised.T)
    )
    return mean, predictive_sd


# %%
# Reproducing the paper's Figure 9: one, two, and three singular values
# ---------------------------------------------------------------------
# Amjad, Shah and Shen plot California's synthetic control at successive
# thresholds. Keeping a single singular value is too rigid -- the mean is biased,
# overstating the counterfactual. Two singular values balance bias against
# variance: the mean tracks the pre-period and the band is moderate. At three, the
# mean still matches the classical synthetic control, but the band widens sharply
# -- the paper's central California observation, and the reason it cautions that
# the classical method "may have overestimated the effect of Prop. 99."

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, rank in zip(axes, (1, 2, 3)):
    mean, sd = bayesian_rsc(rank)
    ax.plot(years, observed, color="black", lw=1.8, label="California")
    ax.plot(years, mean, color="#1f5fbf", lw=1.8, ls="--", label="posterior mean")
    ax.fill_between(years, mean - sd, mean + sd, color="#1f5fbf", alpha=0.15,
                    label=r"$\pm 1$ SD")
    ax.axvline(treat_year, color="gray", lw=1, ls=":")
    ax.set_title(f"{rank} singular value" + ("s" if rank > 1 else ""))
    ax.set_xlabel("Year")
axes[0].set_ylabel("Per-capita cigarette sales (packs)")
axes[0].legend(frameon=False, loc="lower left", fontsize=8)
fig.suptitle("Bayesian Robust Synthetic Control: California Proposition 99")
fig.tight_layout()

# %%
# The effect estimate, and how the band grows with rank
# -----------------------------------------------------
# At every rank the posterior mean shows the post-1989 decline -- the estimated
# effect of the program is real. What changes is the confidence around it. The
# post-period predictive standard deviation is flat through rank two, then jumps:
# beyond two singular values the model lets noise directions into the posterior
# and the band widens fast, exactly as the paper reports.

ranks = [1, 2, 3, 4, 5]
att = [float(np.mean(observed[T0:] - bayesian_rsc(r)[0][T0:])) for r in ranks]
post_sd = [float(np.mean(bayesian_rsc(r)[1][T0:])) for r in ranks]
for r, a, s in zip(ranks, att, post_sd):
    print(f"rank {r}: ATT {a:+6.2f} packs/capita   mean post-period SD {s:5.1f}")

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(ranks, post_sd, "o-", color="#b3202c", lw=2)
ax2.axvline(2, color="gray", lw=1, ls=":")
ax2.set_xlabel("Retained singular values (rank)")
ax2.set_ylabel("Mean post-period predictive SD (packs)")
ax2.set_title("Uncertainty grows once the rank exceeds two")
ax2.set_xticks(ranks)
fig2.tight_layout()

# %%
# What to take away
# -----------------
# The Bayesian Robust Synthetic Control returns a distribution, not a point. Both
# it and the classical method agree that Proposition 99 reduced cigarette
# consumption; the Bayesian band adds a second layer of honesty on top of that
# agreement. Read at the rank the model best supports it warns how much of the
# apparent effect is firmly pinned down and how much is the estimator's own
# uncertainty -- for California, enough that the classical point estimate may
# overstate the decline. The same posterior machinery drives
# ``CLUSTERSC(estimator="bayesian")``; see :doc:`/clustersc` for the estimator and
# its donor-clustering options.
