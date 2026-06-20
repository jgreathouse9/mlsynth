#!/usr/bin/env Rscript
# HCW (Hsiao-Ching-Wan 2012) best-subset on the Hong Kong data -- a faithful
# base-R reference for mlsynth's PDA(method="hcw").
#
# This mirrors the pampe R package's pipeline value-for-value: best subset over
# the control economies (leaps::regsubsets enumerates the lowest-RSS subset of
# each size), the size chosen by AICc, and the counterfactual refit by lm with
# an intercept. We reimplement that pipeline in base R (no pampe/leaps needed,
# so it runs without CRAN access) -- and, when the pampe package *is* installed,
# the same numbers fall out of leaps::regsubsets, since this is precisely what
# it computes. The AICc uses the pampe / HCW Table XVI convention K = p + 2
# (donors + intercept + error variance), which reproduces AICc = -171.771.
#
# Output: one "key=value" line per quantity, for the Python harness to parse.

suppressWarnings(suppressMessages({
  args <- commandArgs(trailingOnly = TRUE)
  data_path <- if (length(args) >= 1) args[1] else "basedata/HongKong.csv"
}))

cands <- c("China", "Indonesia", "Japan", "Korea", "Malaysia",
           "Philippines", "Singapore", "Taiwan", "Thailand", "United States")
T0 <- 18          # pre-period length (1993:Q1-1997:Q2), HCW Table XVI window

d <- read.csv(data_path, stringsAsFactors = FALSE)
d <- d[d$Country %in% c("Hong Kong", cands) & d$Time <= 43, ]

# Wide matrix: rows = Time (sorted), cols = country.
wide <- reshape(d[, c("Time", "Country", "GDP")],
                idvar = "Time", timevar = "Country", direction = "wide")
wide <- wide[order(wide$Time), ]
colnames(wide) <- sub("^GDP\\.", "", colnames(wide))

y_all <- wide[["Hong Kong"]]
X_all <- as.matrix(wide[, cands])
y  <- y_all[1:T0]
X  <- X_all[1:T0, , drop = FALSE]
n  <- T0

# pampe / HCW AICc convention: K = (#donors) + intercept + error variance.
aicc <- function(rss, n, p) {
  K <- p + 2
  if (rss <= 0 || n - K - 1 <= 0) return(Inf)
  n * log(rss / n) + 2 * K + 2 * K * (K + 1) / (n - K - 1)
}

rss_of <- function(cols) {
  if (length(cols) == 0) {
    fit <- lm(y ~ 1)
  } else {
    fit <- lm(y ~ ., data = as.data.frame(X[, cols, drop = FALSE]))
  }
  sum(residuals(fit)^2)
}

# Best subset of each size by RSS (leaps::regsubsets), then min AICc over sizes.
N <- ncol(X)
best_ic <- aicc(rss_of(integer(0)), n, 0)
best_cols <- integer(0)
for (r in 1:N) {
  combos <- combn(N, r)
  # lowest-RSS subset of this size
  rss_r <- apply(combos, 2, function(cc) rss_of(cc))
  j <- which.min(rss_r)
  ic <- aicc(rss_r[j], n, r)
  if (ic < best_ic) {
    best_ic <- ic
    best_cols <- combos[, j]
  }
}

sel <- cands[best_cols]
safe <- make.names(sel)                  # lm mangles "United States" -> "United.States"
# Refit OLS (with intercept) on the selected support; extrapolate full sample.
df_pre <- as.data.frame(X[, best_cols, drop = FALSE]); names(df_pre) <- safe
fit <- lm(y ~ ., data = df_pre)
coefs <- coef(fit)
rss_pre <- sum(residuals(fit)^2)
r2_pre <- 1 - rss_pre / sum((y - mean(y))^2)

df_all <- as.data.frame(X_all[, best_cols, drop = FALSE]); names(df_all) <- safe
yhat_all <- predict(fit, newdata = df_all)
att <- mean((y_all - yhat_all)[(T0 + 1):length(y_all)])

cat(sprintf("selected=%s\n", paste(sort(sel), collapse = ",")))
cat(sprintf("n_selected=%d\n", length(sel)))
cat(sprintf("aicc=%.4f\n", best_ic))
cat(sprintf("r2_pre=%.4f\n", r2_pre))
cat(sprintf("intercept=%.4f\n", coefs[["(Intercept)"]]))
for (i in seq_along(sel))
  cat(sprintf("weight_%s=%.4f\n", gsub(" ", "_", sel[i]), coefs[[safe[i]]]))
cat(sprintf("att_pct=%.4f\n", att * 100))
