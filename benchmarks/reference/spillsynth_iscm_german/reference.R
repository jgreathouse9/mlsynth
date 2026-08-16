# Melnychuk (2024) spillover-SCM reference, German reunification.
#
# Runs the authors' own functions -- unedited, from the pinned clone -- on the
# panel their Main scripts assemble, and reports every quantity mlsynth's
# SPILLSYNTH cross-checks against. Two backends:
#
#   no covariates   "Functions no covariates.R": scm_weights (demeaned simplex
#                   SCM with an intercept, solved by kernlab::ipop) driving
#                   runSCM / runInclusiveSCM / runIterativeSCM.
#   with covariates "Functions w. covariates.R": Synth::synth with Abadie's
#                   predictor block, driving the *withCov variants.
#
# Both branches of replacePreTreatData are run. The function signature defaults
# it to FALSE; runAllMethods and runAllMethodsWithCov -- the entry points behind
# the paper's tables -- pass TRUE.
#
# The weight sums are reported because they are not 1. ipop returns
# how = "converged" on a primal that misses the simplex constraint by 3.3%
# fitting West Germany and 19.3% fitting Austria, so the reference's synthetic
# controls are not convex combinations. Anything downstream of the weights --
# the cross-weight system, the inclusive correction -- inherits that.
#
# Usage: Rscript reference.R <clone-dir> <repgermany.dta>
# Needs: kernlab, MASS, Synth, foreign.

args <- commandArgs(trailingOnly = TRUE)
CLONE <- args[1]
DTA <- args[2]

# The reference files load future/furrr/yaml at the top for runAllMethods, which
# uses %<-%. Nothing called here needs them, so let only the real dependencies
# through and stub the rest instead of installing a parallel backend.
.real_library <- base::library
library <- function(...) {
  nm <- tryCatch(as.character(substitute(list(...))[[2]]), error = function(e) "")
  if (nm %in% c("MASS", "kernlab", "Synth")) .real_library(nm, character.only = TRUE)
  invisible(NULL)
}
source(file.path(CLONE, "Functions no covariates.R"))
source(file.path(CLONE, "Functions w. covariates.R"))
library <- .real_library
suppressMessages({.real_library("kernlab"); .real_library("Synth")})

d <- foreign::read.dta(DTA)

# ---------------------------------------------------------------- no covariates
# "Main no covariates real data.R": columns are [treated, affected, controls],
# rows 1960:2003, with 1960:1989 the pre-period.
wide <- reshape(d[, c("country", "year", "gdp")], idvar = "year",
                timevar = "country", direction = "wide")
wide <- wide[order(wide$year), ]
years <- wide$year
colnames(wide) <- sub("^gdp\\.", "", colnames(wide))
units <- setdiff(colnames(wide), "year")
ordered_units <- c("West Germany", "Austria",
                   setdiff(units, c("West Germany", "Austria")))
Yall <- as.matrix(wide[, ordered_units])

t <- sum(years <= 1989)
T <- nrow(Yall)
n <- ncol(Yall)
post <- (t + 1):T

dataprep <- list(
  foo = Yall, pre.treat.period = 1:t, full.period = 1:T,
  sp.count = 1, total.ctrl.count = n - 1,
  Y0 = Yall[1:t, ], Y1 = Yall[(t + 1):T, ], A = diag(n)[, c(1, 2)], y0 = 0
)

att <- function(g) mean(as.vector(g)[post])

main <- runSCM(dataprep)
swapped <- replaceSpAndTreated(dataprep, 2)
austria <- runSCM(swapped)

w_A <- main$weights[2]                      # Austria in synthetic West Germany
l_WG <- austria$weights[2]                  # West Germany in synthetic Austria

vals <- list(
  nocov_regular_att   = att(main$gap),
  nocov_inclusive_att = att(runInclusiveSCM(dataprep)$gap),
  nocov_iterative_att_replace_pre_false =
    att(runIterativeSCM(dataprep, FALSE, TRUE)$gap),
  nocov_iterative_att_replace_pre_true =
    att(runIterativeSCM(dataprep, TRUE, TRUE)$gap),
  nocov_w_austria_in_west_germany = w_A,
  nocov_w_west_germany_in_austria = l_WG,
  nocov_omega_det = 1 - w_A * l_WG,
  # simplex feasibility of the two ipop solves
  nocov_weight_sum_west_germany = sum(main$weights),
  nocov_weight_sum_austria = sum(austria$weights)
)

# -------------------------------------------------------------- with covariates
# "Main w. covariates real data.R" dataprep2. The predictor block is
# trade + infrate: the lagged-GDP predictor of Abadie's German specification is
# absent from the reference script, though the paper reports Abadie's 0.42
# weight for Austria.
dataprep2 <- list(
  foo = d,
  predictors = c("trade", "infrate"),
  dependent = "gdp",
  unit.variable = "index",
  time.variable = "year",
  special.predictors = list(
    list("industry", 1981:1990, c("mean")),
    list("schooling", c(1980, 1985), c("mean")),
    list("invest80", 1980, c("mean"))
  ),
  treatment.identifier = 7,
  controls.identifier = setdiff(unique(d$index), c(3, 7)),
  spillover.controls.identifier = 3,
  time.predictors.prior = 1981:1990,
  time.optimize.ssr = 1960:1989,
  unit.names.variable = "country",
  time.plot = 1960:2003
)

vals$cov_regular_att <- att(runRegularSCMwithCov(dataprep2)$gap)
vals$cov_inclusive_att <- att(runInclusiveSCMwithCov(dataprep2)$gap)
vals$cov_iterative_att_replace_pre_false <-
  att(runIterativeSCMwithCov(dataprep2, FALSE, TRUE)$gap)
vals$cov_iterative_att_replace_pre_true <-
  att(runIterativeSCMwithCov(dataprep2, TRUE, TRUE)$gap)

cat("== REFERENCE VALUES ==\n")
for (k in names(vals)) cat(sprintf("%s\t%.6f\n", k, as.numeric(vals[[k]])))
cat("== SESSION ==\n")
print(sessionInfo())
