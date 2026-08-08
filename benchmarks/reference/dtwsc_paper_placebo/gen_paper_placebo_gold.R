# Reference inputs for the DTWSC paper-placebo replication.
#
# Usage:
#   Rscript gen_paper_placebo_gold.R [/path/to/dataverse_files/data]
#
# Without the argument, writes the Basque panel only. With it, also decodes the
# authors' per-run grid choices. See README.md for provenance.
suppressMessages({library(dplyr)})
out.dir <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)))
if (length(out.dir) == 0 || out.dir == "") out.dir <- "."

## Basque panel, exactly as the authors load it ------------------------------
data(basque, package = "Synth")
d <- basque
colnames(d)[1:4] <- c("id", "unit", "time", "value")
# The 14th special predictor is derived rather than a column of the source data.
d <- d %>% mutate(invest_ratio = invest / value)
write.csv(d, file.path(out.dir, "gold_basque_panel.csv"), row.names = FALSE)
cat(sprintf("gold_basque_panel.csv: %d units, %d rows\n",
            length(unique(d$id)), nrow(d)))

## Per-run grid choices ------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  cat("no archive path given; skipping gold_gridopt.csv\n")
  quit(status = 0)
}
archive <- args[1]

# The sweep, in the authors' own order. expand.grid varies the FIRST factor
# fastest, and grid.id indexes into it, so the ordering here is load-bearing:
# permuting these vectors silently remaps every run's hyperparameters.
step.patterns <- c("symmetricP1", "symmetricP2", "asymmetricP1",
                   "asymmetricP2", "typeIc", "typeId", "mori2006")
grid <- expand.grid(filter.width = (1:9) * 2 + 3,
                    k = 4:9,
                    step.pattern = step.patterns)

panels <- c("Figure_5_1", "Figure_5_2", "Figure_5_3")
rows <- NULL
for (nm in panels) {
  path <- file.path(archive, paste0(nm, "_gridOpt.Rds"))
  if (!file.exists(path)) {
    stop("missing ", path, " -- expected the archive's data/ directory")
  }
  g <- readRDS(path)
  gi <- as.numeric(g$grid.id)
  if (any(gi < 1 | gi > nrow(grid))) {
    stop(nm, ": grid.id out of range for a ", nrow(grid), "-row sweep")
  }
  rows <- rbind(rows, data.frame(
    panel = nm, unit = g$unit, id = g$id,
    data.id = as.numeric(g$data.id), grid.id = gi,
    filter.width = grid$filter.width[gi], k = grid$k[gi],
    step.pattern = as.character(grid$step.pattern[gi]),
    stringsAsFactors = FALSE))
}
write.csv(rows, file.path(out.dir, "gold_gridopt.csv"), row.names = FALSE)
cat(sprintf("gold_gridopt.csv: %d runs\n", nrow(rows)))
print(table(rows$panel))
