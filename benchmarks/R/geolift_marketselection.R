#!/usr/bin/env Rscript
# Live GeoLiftMarketSelection on the walkthrough PreTest panel, for
# benchmarks/cases/geolift_marketselection_ref.py. Install the reference once
# with benchmarks/R/install_geolift.sh.
#
# Prints the full BestMarkets ranking as tab-separated ROW lines (markets joined
# by '|', since the location field itself contains commas), for the Python case
# to parse and compare against mlsynth's pooled N=2-5 selection.
suppressMessages(library(GeoLift))

args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]                       # basedata/geolift_market_data.csv (PreTest)
df <- read.csv(data_path)

geodata <- GeoDataRead(data = df, date_id = "date", location_id = "location",
                       Y_id = "Y", format = "yyyy-mm-dd", summary = FALSE)

ms <- GeoLiftMarketSelection(
  data = geodata, treatment_periods = c(10, 15), N = c(2, 3, 4, 5),
  Y_id = "Y", location_id = "location", time_id = "time",
  effect_size = seq(0, 0.2, 0.05), lookback_window = 1,
  include_markets = c("chicago"), exclude_markets = c("honolulu"),
  cpic = 7.50, budget = 100000, alpha = 0.1, Correlations = TRUE,
  fixed_effects = TRUE, side_of_test = "two_sided"
)

bm <- ms$BestMarkets
rank_col <- if ("rank" %in% colnames(bm)) "rank" else "Rank"
for (i in seq_len(nrow(bm))) {
  loc <- gsub(", ", "|", bm$location[i])
  cat(sprintf("ROW\t%s\t%d\t%.4f\t%.4f\t%.4f\t%.2f\t%.3f\t%d\n",
              loc, bm$duration[i], bm$EffectSize[i], bm$Power[i],
              bm$AvgScaledL2Imbalance[i], bm$Investment[i],
              bm$abs_lift_in_zero[i], bm[[rank_col]][i]))
}
