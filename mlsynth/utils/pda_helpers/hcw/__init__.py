"""HCW best-subset PDA (Hsiao, Ching & Wan 2012): estimation + ATE inference."""

from .estimation import best_subset_select, fit_hcw, info_criterion
from .holdout import (MIN_HOLDOUT_TRAIN, holdout_schedule,
                      rolling_origin_holdout)
from .inference import hcw_ate_inference

__all__ = ["best_subset_select", "fit_hcw", "info_criterion",
           "hcw_ate_inference", "MIN_HOLDOUT_TRAIN", "holdout_schedule",
           "rolling_origin_holdout"]
