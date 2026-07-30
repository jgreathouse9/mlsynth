#!/usr/bin/env Rscript
# Gold for the Lamba et al. (2023) tiger-reserve replication.
#
#   Rscript benchmarks/reference/lamba_tigers/reference.R
#
# Lamba, A., Teo, H. C., Sreekar, R., Zeng, Y., Carrasco, L. R., & Koh, L. P.
# (2023). "Climate co-benefits of tiger conservation." Nature Ecology &
# Evolution 7:1104-1113. doi:10.1038/s41559-023-02069-x. Replication package:
# data + SyntheticControls.R, archived with the paper.
#
# The paper fits one synthetic control per tiger reserve: outcome is cumulative
# forest loss (ha) 2001-2020, the donor pool is the untreated protected areas in
# the same tiger-conservation zone, and thirteen predictors are averaged over
# 2001..treatment year (AgeOfPA is taken at the treatment year alone).
#
# WHICH tidysynth THIS RUNS, AND WHY IT MATTERS
# ---------------------------------------------
# This script captures tidysynth 0.2.0 (current), NOT the version the authors
# used. Their script calls `grab_signficance` -- a misspelling the package fixed
# on 2021-11-30 -- so the published numbers come from v0.1.0 or earlier, which
# also predates a 2022-08-31 commit whose message states that time-ordering
# problems "resulted in downstream distortions of the package output".
#
# The two versions do not agree. On Nawegaon-Nagzira, the same script gives
# 2643.4 ha averted under v0.1.0 (the paper reports 2645) and 1998.9 ha under
# 0.2.0. All eleven reserves in the paper's Table 1 reproduce under v0.1.0 to a
# total of 6556.7 ha against the published 6560. So the paper is exactly
# reproducible -- on a superseded package.
#
# mlsynth is checked against the current package because pinning a library to a
# version its own maintainer has since corrected would be pinning a defect. The
# v0.1.0 numbers are recorded in docs/replications/lamba_tigers.rst as the
# historical record, not as a target. To regenerate them:
#
#   git clone https://github.com/edunford/tidysynth && cd tidysynth
#   git checkout 0389f29~1 && R CMD INSTALL .
#   # then s/grab_significance/grab_signficance/ below
#
# WHAT IS CAPTURED
# ----------------
#   predictors_<reserve>.csv  the 13 x (1 + n_donors) predictor matrix, so
#                             mlsynth's panel construction can be checked at the
#                             SEAM before any solver is involved. This is the
#                             part that must agree; the V search below is the
#                             part that does not.
#   summary.csv               per reserve: averted loss, pre/post MSPE, the MSPE
#                             ratio, rank and placebo p-value.
#   weights_<reserve>.csv     donor weights and predictor (V) weights.
suppressMessages({library(tidysynth); library(dplyr)})

OUT  <- file.path("benchmarks", "reference", "lamba_tigers")
DATA <- file.path("basedata", "tiger_reserves.csv")

# The fifteen reserves the paper reports individually: eleven with significant
# avoided deforestation (Table 1) and four with significantly higher loss than
# their counterfactual (Fig. 3).
RESERVES <- c(
  "Nawegaon-Nagzira Tiger Reserve", "Similipal-Hadagarh WLS/Tiger Reserve",
  "Udanti WLS", "Valmiki WLS/Tiger Reserve", "Nagarjuna Sagar-Srisailam WLS PLIT",
  "Palamau WLS/Tiger Reserve", "Satkosia Gorge WLS",
  "Bandipur National Park/Tiger Reserve",
  "Biligiri Rangaswamy Temple (B.R.T.) WLS/Tiger Reserve",
  "Sariska National Park", "Satpura National Park/Tiger Reserve",
  "Pilibhit Tiger Reserve", "Anamalai Tiger Reserve", "Dampa WLS/Tiger Reserve",
  "Kaziranga National Park/Tiger Reserve")

slug <- function(s) gsub("[^A-Za-z0-9]+", "_", s)

reserveDat <- read.csv(DATA, header = TRUE)
# The authors fold the single-member SB zone into CEEG, and convert travel time
# to the nearest city from minutes to hours. Both are theirs, not ours.
reserveDat$Zone[reserveDat$Zone == "SB"] <- "CEEG"
reserveDat$closest.city <- reserveDat$closest.city / 60

tp <- reserveDat[reserveDat$TR.year != 0, ]
tp <- tp[tp$TR.year < 2016 & tp$TR.year > 2006, ]

build_panel <- function(NAME) {
  row <- tp[tp$name == NAME, ]
  ty  <- row$TR.year
  donorPool <- reserveDat[reserveDat$TR.year == 0, ]
  donorPool <- donorPool[donorPool$Zone == row$Zone, ]
  Sdat <- rbind(row, donorPool)
  df <- NULL
  for (id in 1:nrow(Sdat)) {
    D <- 0
    for (year in 2001:2020) {
      if (Sdat[id, "D"] == 1 && year >= Sdat[id, "TR.year"]) D <- 1
      df <- rbind(df, data.frame(
        ID = id, year = year, name = Sdat[id, "name"],
        loss = 100 * Sdat[id, paste0("Loss.", year)],
        precipitation = Sdat[id, paste0("Rainfall.", year - 1)],
        population = Sdat[id, paste0("Population.", year - 1)],
        baselineForest = 100 * Sdat[id, "Forest.2000"],
        elevation = Sdat[id, "elevation"], slope = Sdat[id, "slope"],
        aspect = Sdat[id, "aspect"], area = Sdat[id, "PAArea"] * 0.0001,
        D = D, AgeOfPA = Sdat[id, "agePA"],
        distanceToCity = Sdat[id, "closest.city"],
        ppp2000 = Sdat[id, "ppp2000"], roadLength = Sdat[id, "Road.length"],
        meanAGB = Sdat[id, "AGB.mean"], stringsAsFactors = FALSE))
    }
  }
  df$roadLength    <- df$roadLength / 1000
  df$area          <- log(df$area)
  df$baselineForest <- log(df$baselineForest)
  list(df = df, ty = ty, n_donors = nrow(donorPool))
}

fit_one <- function(NAME) {
  b <- build_panel(NAME); ty <- b$ty
  as_tibble(b$df) %>%
    synthetic_control(outcome = loss, unit = name, time = year,
                      i_unit = NAME, i_time = ty, generate_placebos = TRUE) %>%
    generate_predictor(time_window = 2001:ty, loss = mean(loss)) %>%
    generate_predictor(time_window = 2001:ty, population = mean(population)) %>%
    generate_predictor(time_window = 2001:ty,
                       precipitation = mean(precipitation, na.rm = TRUE)) %>%
    generate_predictor(time_window = 2001:ty,
                       baselineForest = mean(baselineForest, na.rm = TRUE)) %>%
    generate_predictor(time_window = ty, AgeOfPA = AgeOfPA) %>%
    generate_predictor(time_window = 2001:ty,
                       elevation = mean(elevation, na.rm = TRUE)) %>%
    generate_predictor(time_window = 2001:ty, slope = mean(slope, na.rm = TRUE)) %>%
    generate_predictor(time_window = 2001:ty, aspect = mean(aspect, na.rm = TRUE)) %>%
    generate_predictor(time_window = 2001:ty, area = mean(area, na.rm = TRUE)) %>%
    generate_predictor(time_window = 2001:ty,
                       distanceToCity = mean(distanceToCity, na.rm = TRUE)) %>%
    generate_predictor(time_window = 2001:ty, ppp2000 = mean(ppp2000, na.rm = TRUE)) %>%
    generate_predictor(time_window = 2001:ty, meanAGB = mean(meanAGB, na.rm = TRUE)) %>%
    generate_predictor(time_window = 2001:ty,
                       roadLength = mean(roadLength, na.rm = TRUE)) %>%
    # The authors pass quadopt = "LowRankQP". tidysynth 0.2.0 accepts that value
    # and does not honour it: R/synth.R line 174 calls kernlab::ipop
    # unconditionally inside the V search, and the final solve at line 303 is
    # wrapped in `if (quadopt == "ipop")` with no else branch, so the LowRankQP
    # path assigns no weights at all. (v0.1.0 had eight LowRankQP call sites;
    # 0.2.0 has only the argument validation, after the 2023-05-20 commit that
    # dropped the dependency.) Passing "ipop" explicitly therefore runs what the
    # current package actually does, rather than relying on a silent fallback.
    generate_weights(optimization_window = 2001:ty, quadopt = "ipop") %>%
    generate_control()
}

summ <- NULL
failures <- NULL
for (NAME in RESERVES) {
  b <- build_panel(NAME)
  # ipop is an interior-point routine and fails outright on some of these
  # panels ("system is computationally singular"). Recorded rather than
  # allowed to abort the run: which reserves the current package cannot fit is
  # itself part of what this reference documents.
  out <- tryCatch(fit_one(NAME), error = function(e) {
    failures <<- rbind(failures, data.frame(reserve = NAME, treatment_year = b$ty,
                       n_donors = b$n_donors, error = conditionMessage(e),
                       stringsAsFactors = FALSE))
    cat(sprintf("%-56s ty=%d donors=%2d FAILED: %s\n", NAME, b$ty, b$n_donors,
                conditionMessage(e)))
    NULL })
  if (is.null(out)) next
  sc  <- as.data.frame(out %>% grab_synthetic_control())
  sg  <- as.data.frame(out %>% grab_significance())
  sg  <- sg[sg$type == "Treated", ]
  averted <- as.numeric(sc[nrow(sc), 3]) - as.numeric(sc[nrow(sc), 2])

  s <- slug(NAME)
  write.csv(as.data.frame(out %>% grab_predictors(type = "treated")),
            file.path(OUT, paste0("predictors_treated_", s, ".csv")), row.names = FALSE)
  write.csv(as.data.frame(out %>% grab_predictors(type = "controls")),
            file.path(OUT, paste0("predictors_controls_", s, ".csv")), row.names = FALSE)
  write.csv(as.data.frame(out %>% grab_unit_weights()),
            file.path(OUT, paste0("weights_unit_", s, ".csv")), row.names = FALSE)
  write.csv(as.data.frame(out %>% grab_predictor_weights()),
            file.path(OUT, paste0("weights_pred_", s, ".csv")), row.names = FALSE)
  write.csv(sc, file.path(OUT, paste0("series_", s, ".csv")), row.names = FALSE)

  summ <- rbind(summ, data.frame(
    reserve = NAME, slug = s, treatment_year = b$ty, n_donors = b$n_donors,
    observed_2020 = as.numeric(sc[nrow(sc), 2]),
    synthetic_2020 = as.numeric(sc[nrow(sc), 3]),
    averted_loss_ha = averted,
    pre_mspe = sg$pre_mspe, post_mspe = sg$post_mspe,
    mspe_ratio = sg$mspe_ratio, rank = as.integer(sg$rank),
    p_value = sg$fishers_exact_pvalue, stringsAsFactors = FALSE))
  cat(sprintf("%-56s ty=%d donors=%2d averted=%9.2f\n", NAME, b$ty, b$n_donors, averted))
}
write.csv(summ, file.path(OUT, "summary.csv"), row.names = FALSE)
if (!is.null(failures))
  write.csv(failures, file.path(OUT, "failures.csv"), row.names = FALSE)

writeLines(c(
  sprintf("tidysynth: %s", as.character(packageVersion("tidysynth"))),
  sprintf("kernlab:   %s", as.character(packageVersion("kernlab"))),
  sprintf("R:         %s.%s", R.version$major, R.version$minor),
  sprintf("quadopt:   ipop (see note in fit_one)"),
  sprintf("n_requested: %d", length(RESERVES)),
  sprintf("n_fitted:    %d", nrow(summ)),
  sprintf("n_failed:    %d", if (is.null(failures)) 0L else nrow(failures))),
  file.path(OUT, "versions.txt"))
cat("\n", readLines(file.path(OUT, "versions.txt")), sep = "\n")
