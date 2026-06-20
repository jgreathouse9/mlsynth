#!/usr/bin/env Rscript
# HCW (Hsiao-Ching-Wan 2012) best-subset on the Hong Kong data -- the R-side
# reference for mlsynth's PDA(method="hcw").
#
# pampe's engine is leaps::regsubsets (the original Furnival-Wilson "leaps and
# bounds" best-subset Fortran) + an AICc choice of model size + lm with an
# intercept. We call leaps::regsubsets directly -- the genuine engine pampe
# wraps -- so this is a true cross-language check of the Python Furnival-Wilson
# search, not a re-implementation. (The pampe package itself is archived on CRAN
# and not packaged for Debian; leaps is: `apt-get install r-cran-leaps`.) When
# leaps is unavailable we fall back to an exhaustive base-R enumeration, which
# computes exactly the same best-subset-of-each-size.
#
# The AICc uses the pampe / HCW Table XVI convention K = p + 2 (donors +
# intercept + error variance), which reproduces AICc = -171.771.
#
# Output: one "key=value" line per quantity, for a Python harness to parse.

suppressWarnings(suppressMessages({
  args <- commandArgs(trailingOnly = TRUE)
  data_path <- if (length(args) >= 1) args[1] else "basedata/HongKong.csv"
  have_leaps <- requireNamespace("leaps", quietly = TRUE)
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
N  <- ncol(X)

# pampe / HCW AICc convention: K = (#donors) + intercept + error variance.
aicc <- function(rss, p) {
  K <- p + 2
  if (rss <= 0 || n - K - 1 <= 0) return(Inf)
  n * log(rss / n) + 2 * K + 2 * K * (K + 1) / (n - K - 1)
}

rss0 <- sum((y - mean(y))^2)          # intercept-only model (size 0)

if (have_leaps) {
  engine <- "leaps::regsubsets"
  reg <- leaps::regsubsets(X, y, nvmax = N, method = "exhaustive", intercept = TRUE)
  s <- summary(reg)
  # leaps returns the lowest-RSS subset of each size r = 1..N.
  rss_by_size <- c(rss0, s$rss)
  ics <- vapply(0:N, function(r) aicc(rss_by_size[r + 1], r), numeric(1))
  best <- which.min(ics) - 1
  best_cols <- if (best == 0) integer(0) else unname(which(s$which[best, -1]))
} else {
  engine <- "base-R exhaustive"
  rss_of <- function(cols) {
    fit <- if (length(cols) == 0) lm(y ~ 1)
           else lm(y ~ ., data = as.data.frame(X[, cols, drop = FALSE]))
    sum(residuals(fit)^2)
  }
  best_ic <- aicc(rss0, 0); best_cols <- integer(0)
  for (r in 1:N) {
    combos <- combn(N, r)
    rss_r <- apply(combos, 2, rss_of)
    j <- which.min(rss_r)
    ic <- aicc(rss_r[j], r)
    if (ic < best_ic) { best_ic <- ic; best_cols <- combos[, j] }
  }
}

sel <- cands[best_cols]
safe <- make.names(sel)                  # lm mangles "United States" -> "United.States"
df_pre <- as.data.frame(X[, best_cols, drop = FALSE]); names(df_pre) <- safe
fit <- lm(y ~ ., data = df_pre)
coefs <- coef(fit)
rss_pre <- sum(residuals(fit)^2)
r2_pre <- 1 - rss_pre / sum((y - mean(y))^2)
best_ic <- aicc(rss_pre, length(best_cols))

df_all <- as.data.frame(X_all[, best_cols, drop = FALSE]); names(df_all) <- safe
yhat_all <- predict(fit, newdata = df_all)
att <- mean((y_all - yhat_all)[(T0 + 1):length(y_all)])

cat(sprintf("engine=%s\n", engine))
cat(sprintf("selected=%s\n", paste(sort(sel), collapse = ",")))
cat(sprintf("n_selected=%d\n", length(sel)))
cat(sprintf("aicc=%.4f\n", best_ic))
cat(sprintf("r2_pre=%.4f\n", r2_pre))
cat(sprintf("intercept=%.4f\n", coefs[["(Intercept)"]]))
for (i in seq_along(sel))
  cat(sprintf("weight_%s=%.4f\n", gsub(" ", "_", sel[i]), coefs[[safe[i]]]))
cat(sprintf("att_pct=%.4f\n", att * 100))
