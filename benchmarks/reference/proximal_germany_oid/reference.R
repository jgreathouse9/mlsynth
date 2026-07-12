# Reference run for the `proximal_germany_oid` benchmark case.
#
# Reproduces the proximal-inference (PI) headline of Shi, Li, Yu, Miao,
# Kuchibhotla, Hu & Tchetgen Tchetgen (2026), "Theory for Identification and
# Inference with Synthetic Controls: A Proximal Causal Inference Framework"
# (JASA), on the 1990 German reunification, following the authors' own manuscript
# replication code (KenLi93/proximal_sc_manuscript: NC_nocov for the point
# estimate, NC_nocov_gmm for the GMM/Newey-West interval).
#
# The over-identified proximal outcome bridge instruments the 6 donor countries
# W (Austria, Italy, Japan, Netherlands, Switzerland, USA -- the countries
# Cattaneo et al. give nonzero SC weight) with the remaining 10 OECD countries Z,
# a single GDP-per-capita outcome throughout:
#
#   omega = (W'Z Z'W)^{-1} W'Z Z'Y   (pre-period 1960-1990), no covariates,
#   ATT   = mean over 1991-2003 of (Y - W omega).
#
# The ATT standard error is the authors' one-step GMM sandwich over all periods
# with theta = (tau, omega), moment (Y - S1 theta) S2 for S1 = [X, W] and
# S2 = [X, Z(1-X)], and a Bartlett/Newey-West HAC at lag q = 10 (their default).
# These six lines are the authors' method reproduced on the in-repo public data
# (scpi_germany, GDP per capita in thousands of USD); the upstream repo has no
# licence, so the method is reproduced rather than vendored (as with sbc_germany).
#
# Run from the repository root:
#   Rscript benchmarks/reference/proximal_germany_oid/reference.R
d <- read.csv("basedata/scpi_germany.csv")
d <- d[!is.na(d$gdp), c("country", "year", "gdp")]
wide <- reshape(d, idvar = "year", timevar = "country", direction = "wide")
rownames(wide) <- wide$year; wide$year <- NULL
colnames(wide) <- sub("^gdp\\.", "", colnames(wide))
years <- as.integer(rownames(wide))

treated <- "West Germany"
W.names <- c("Austria", "Italy", "Japan", "Netherlands", "Switzerland", "USA")
Z.names <- setdiff(colnames(wide), c(treated, W.names))
T0 <- sum(years <= 1990); T <- length(years)

Y <- as.numeric(wide[[treated]])
W <- as.matrix(wide[, W.names]); Z <- as.matrix(wide[, Z.names])

# Point estimate: over-identified outcome bridge on the pre-period (NC_nocov).
Wp <- W[1:T0, ]; Zp <- Z[1:T0, ]; Yp <- Y[1:T0]
WZ <- t(Wp) %*% Zp
omega <- solve(WZ %*% t(WZ), WZ %*% (t(Zp) %*% Yp))
cf <- as.numeric(W %*% omega)
att <- mean((Y - cf)[(T0 + 1):T])

# ATT SE: joint one-step GMM over all periods, over-identified sandwich, HAC q=10.
X <- c(rep(0, T0), rep(1, T - T0))
theta <- c(att, omega)
S1 <- cbind(X, W); S2 <- cbind(X, Z * (1 - X))
bg <- as.numeric(Y - S1 %*% theta) * S2
G <- t(S2) %*% S1 / T
q <- 10
Omega <- t(bg) %*% bg / T
for (i in 1:q) {
  Oi <- t(bg[-(1:i), ]) %*% bg[1:(T - i), ] / T
  Omega <- Omega + (1 - i / (q + 1)) * (Oi + t(Oi))
}
proj <- solve(t(G) %*% G) %*% t(G)
hacSig <- proj %*% Omega %*% t(proj)
se <- sqrt(hacSig[1, 1] / T)
z90 <- qnorm(0.95)

cat("== REFERENCE VALUES ==\n")
cat(sprintf("att\t%.6f\n", att))
cat(sprintf("att_se\t%.6f\n", se))
cat(sprintf("ci90_lb\t%.6f\n", att - z90 * se))
cat(sprintf("ci90_ub\t%.6f\n", att + z90 * se))
ord <- order(-omega)
for (i in ord) cat(sprintf("weight\t%s\t%.6f\n", W.names[i], omega[i]))
cat("== SESSION INFO ==\n")
print(sessionInfo())
