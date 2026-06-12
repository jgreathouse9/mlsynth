# Live augsynth reference for the GeoLift_Test panel (the GeoLift_Walkthrough fit).
#
# GeoLift's GeoLift(locations=c("chicago","portland"), ...) is augsynth's ridge
# Augmented SCM with fixed effects under the hood. This fits that model directly
# and emits a machine-parseable dump (LAMBDA / ATT / PVAL / W <name> <weight>)
# for benchmarks/cases/geolift_augsynth_ref.py to cross-check mlsynth against.
#
# Install the reference once with benchmarks/R/install_augsynth.sh.
suppressMessages({library(augsynth); library(dplyr)})

args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1) args[1] else "basedata/geolift_test_data.csv"
df <- read.csv(data_path)
df$t <- as.integer(factor(df$date, levels = sort(unique(df$date))))   # 1..105

# Aggregate the two test geos into one treated series (mean), as GeoLift does.
test <- df %>% filter(location %in% c("chicago", "portland")) %>%
        group_by(t) %>% summarise(Y = mean(Y), .groups = "drop") %>%
        mutate(location = "TEST")
panel <- bind_rows(
  df %>% filter(!location %in% c("chicago", "portland")) %>% select(location, t, Y),
  test)
panel$trt <- as.integer(panel$location == "TEST" & panel$t >= 91)

asyn <- augsynth(Y ~ trt, unit = location, time = t, data = panel,
                 progfunc = "ridge", scm = TRUE, fixedeff = TRUE)
s <- summary(asyn, inf_type = "conformal")

cat(sprintf("LAMBDA %.10e\n", asyn$lambda))
cat(sprintf("ATT %.6f\n", s$average_att$Estimate))
cat(sprintf("PVAL %.6f\n", s$average_att$p_val))
w <- asyn$weights[, 1]; w <- w[abs(w) > 1e-3]
for (nm in names(w)) cat(sprintf("W %s %.8f\n", nm, w[nm]))
