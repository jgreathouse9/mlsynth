# Reference run for the `pda_hcw_cepa` benchmark case.
#
# The second of the two events Hsiao, Ching & Wan (2012) study: the Closer
# Economic Partnership Arrangement, implemented 2004:Q1. Where the sovereignty
# case restricts the donor pool to ten regional economies because "there are
# only 18 observations between 1993:Q1 and 1997:Q2", here HCW write that "since
# we now have more degrees of freedom, we can use the model selection strategy
# discussed in Section 5" and let AICc choose from every country in the panel.
# It is the paper's headline result and the harder test of the best-subset
# search: 24 candidates rather than 10.
#
# Engine: leaps::regsubsets (the Furnival-Wilson "leaps and bounds" best-subset
# Fortran) + AICc with K = p + 2 + lm with an intercept. That is the engine the
# CRAN package pampe wraps. pampe itself is archived on CRAN and could not be
# installed in this container, so unlike the sibling `pda_hcw_hongkong` bundle
# this capture calls the engine directly rather than pampe's wrapper around it.
# The AICc convention is pampe's, and the sovereignty spec run through this same
# script reproduces that bundle's numbers.
#
# HCW's published values for this event, for comparison: the AICc-selected
# group is Austria, Italy, Korea, Mexico, Norway and Singapore; the average
# treatment effect is 4.03% with "a standard error of 0.016" and a t-statistic
# of 2.5134; the pre-period R^2 is "above 0.93".
#
# That standard error is the standard deviation of the per-period treatment
# effects, not the standard error of their mean -- 0.0403 / 0.016045 = 2.5134
# reproduces their t exactly, where dividing by sd/sqrt(T2) would give 10.36.
# Both are emitted so the distinction is on the record.

args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1) args[1] else "basedata/HongKong.csv"

T0 <- 44                       # 1993:Q1-2003:Q4; CEPA starts 2004:Q1
d <- read.csv(data_path, stringsAsFactors = FALSE)
cands <- sort(setdiff(unique(d$Country), "Hong Kong"))

wide <- reshape(d[, c("Time", "Country", "GDP")],
                idvar = "Time", timevar = "Country", direction = "wide")
wide <- wide[order(wide$Time), ]
colnames(wide) <- sub("^GDP\\.", "", colnames(wide))

y_all <- wide[["Hong Kong"]]
X_all <- as.matrix(wide[, cands])
y <- y_all[1:T0]; X <- X_all[1:T0, , drop = FALSE]
n <- T0; N <- ncol(X)

aicc <- function(rss, p) {
  K <- p + 2
  if (rss <= 0 || n - K - 1 <= 0) return(Inf)
  n * log(rss / n) + 2 * K + 2 * K * (K + 1) / (n - K - 1)
}
rss0 <- sum((y - mean(y))^2)

reg <- leaps::regsubsets(X, y, nvmax = N, method = "exhaustive", intercept = TRUE)
s <- summary(reg)
rss_by_size <- c(rss0, s$rss)
ics <- vapply(0:N, function(r) aicc(rss_by_size[r + 1], r), numeric(1))
best <- which.min(ics) - 1
best_cols <- if (best == 0) integer(0) else unname(which(s$which[best, -1]))

sel <- cands[best_cols]
safe <- make.names(sel)
df_pre <- as.data.frame(X[, best_cols, drop = FALSE]); names(df_pre) <- safe
fit <- lm(y ~ ., data = df_pre)
coefs <- coef(fit)
r2_pre <- 1 - sum(residuals(fit)^2) / sum((y - mean(y))^2)

df_all <- as.data.frame(X_all[, best_cols, drop = FALSE]); names(df_all) <- safe
yhat_all <- predict(fit, newdata = df_all)
eff <- (y_all - yhat_all)[(T0 + 1):length(y_all)]
att <- mean(eff)

cat(sprintf("engine=%s\n", "leaps::regsubsets"))
cat(sprintf("selected=%s\n", paste(sort(sel), collapse = ",")))
cat(sprintf("n_selected=%d\n", length(sel)))
cat(sprintf("aicc=%.4f\n", aicc(sum(residuals(fit)^2), length(best_cols))))
cat(sprintf("r2_pre=%.6f\n", r2_pre))
cat(sprintf("intercept=%.6f\n", coefs[["(Intercept)"]]))
for (i in seq_along(sel))
  cat(sprintf("weight_%s=%.6f\n", gsub(" ", "_", sel[i]), coefs[[safe[i]]]))
cat(sprintf("att_pct=%.6f\n", att * 100))
cat(sprintf("sd_effect=%.6f\n", sd(eff)))
cat(sprintf("t_hcw=%.6f\n", att / sd(eff)))

# The standard error mlsynth reports for the ATT is the root of a Bartlett
# HAC long-run variance of the effect series, divided by T2 -- not sd/sqrt(T2),
# which assumes the per-period effects are independent. Computed here the same
# way so the two sides are the same estimator: every autocovariance divides by
# n (R's acf(type = "covariance"), the Newey-West convention), and the fsPDA
# truncation lag is floor(T2^(1/4)).
T2 <- length(eff)
L <- floor(T2^0.25)
g <- drop(acf(eff, lag.max = L, type = "covariance", demean = TRUE,
              plot = FALSE)$acf)
lrv <- g[1]
if (L >= 1) for (l in 1:L) lrv <- lrv + 2 * (1 - l / (L + 1)) * g[l + 1]
cat(sprintf("hac_lag=%d\n", L))
cat(sprintf("se_hac=%.6f\n", sqrt(lrv / T2)))
cat(sprintf("se_iid=%.6f\n", sd(eff) / sqrt(T2)))
