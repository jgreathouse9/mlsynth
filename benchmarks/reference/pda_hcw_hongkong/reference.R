# Reference run for the `pda_hcw_hongkong` benchmark case.
#
# Runs the original HCW (Hsiao, Ching & Wan 2012) best-subset panel-data
# approach via its canonical R package -- pampe::pampe() (Vega-Bayo; CRAN
# package "pampe", which implements exactly the method of HCW 2012). pampe's
# engine is leaps::regsubsets (the Furnival-Wilson "leaps and bounds"
# best-subset search) + an AICc choice of model size (regsubsets2aic, K = p + 1
# OLS parameters + 1 for sigma^2) + OLS with an intercept. These are the genuine
# pampe outputs the Python case pins against -- not numbers transcribed from the
# paper's Tables XVI/XVII (which the run nonetheless reproduces to the digit).
#
# Spec matches the Python case exactly: basedata/HongKong.csv, donor pool
# restricted to the ten candidate economies HCW consider, estimation window
# 1993Q1-1997Q2 (rows 1..18 = Time 0..17, T0 = 18), the 1997Q3 sovereignty cut,
# and the post-period truncated at Time 43 (1993Q1-2003Q4, the Table XVI/XVII
# window). pampe selects {Japan, Korea, Taiwan, United States} by AICc and
# returns the OLS counterfactual; the post-1997Q3 average gap is the ATT.
#
# pampe depends only on leaps. See benchmarks/R/install_pampe.sh for the fetch.
#
# Run from the repository root:  Rscript benchmarks/reference/pda_hcw_hongkong/reference.R
suppressWarnings(suppressMessages(library(pampe)))

cands <- c("China", "Indonesia", "Japan", "Korea", "Malaysia",
           "Philippines", "Singapore", "Taiwan", "Thailand", "United States")
T0 <- 18          # pre-period length (1993Q1-1997Q2), HCW Table XVI window

d <- read.csv("basedata/HongKong.csv", stringsAsFactors = FALSE)
d <- d[d$Country %in% c("Hong Kong", cands) & d$Time <= 43, ]

# Wide matrix: rows = Time (sorted), cols = country; pampe wants a data frame
# keyed by time with one column per unit.
wide <- reshape(d[, c("Time", "Country", "GDP")],
                idvar = "Time", timevar = "Country", direction = "wide")
wide <- wide[order(wide$Time), ]
colnames(wide) <- sub("^GDP\\.", "", colnames(wide))
rownames(wide) <- as.character(wide$Time)

M <- as.data.frame(as.matrix(wide[, c("Hong Kong", cands)]))
colnames(M) <- make.names(c("Hong Kong", cands))   # "Hong Kong" -> "Hong.Kong"
treated <- make.names("Hong Kong")
controls <- make.names(cands)

res <- pampe(time.pretr = 1:T0, time.tr = (T0 + 1):nrow(M),
             treated = treated, controls = controls, data = M)

sel <- res$controls                                # AICc-selected control set
coefs <- coef(res$model)
cf <- res$counterfactual                           # Actual / Counterfactual
te <- cf[, "Actual"] - cf[, "Counterfactual"]
att <- mean(te[(T0 + 1):length(te)])               # post-1997Q3 average effect

y_pre <- cf[1:T0, "Actual"]
yhat_pre <- cf[1:T0, "Counterfactual"]
r2_pre <- 1 - sum((y_pre - yhat_pre)^2) / sum((y_pre - mean(y_pre))^2)

# Map the make.names() control labels back to the original economy names.
orig <- cands; names(orig) <- controls

cat("== REFERENCE VALUES ==\n")
cat(sprintf("n_selected\t%d\n", length(sel)))
cat(sprintf("intercept\t%.6f\n", coefs[["(Intercept)"]]))
cat(sprintf("r2_pre\t%.6f\n", r2_pre))
cat(sprintf("att_pct\t%.6f\n", att * 100))
cat(sprintf("weight_japan\t%.6f\n", coefs[[make.names("Japan")]]))
cat(sprintf("weight_taiwan\t%.6f\n", coefs[[make.names("Taiwan")]]))
# Full selected-control weight vector (parsed into `weights`).
for (c in sel) {
  cat(sprintf("weight\t%s\t%.6f\n", orig[[c]], coefs[[c]]))
}
cat("== SESSION INFO ==\n")
print(sessionInfo())
