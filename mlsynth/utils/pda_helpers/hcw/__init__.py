"""HCW best-subset PDA (Hsiao, Ching & Wan 2012): estimation + ATE inference."""

from .estimation import best_subset_select, fit_hcw, info_criterion
from .inference import hcw_ate_inference

__all__ = ["best_subset_select", "fit_hcw", "info_criterion", "hcw_ate_inference"]
