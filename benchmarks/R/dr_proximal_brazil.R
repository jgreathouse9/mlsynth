#!/usr/bin/env Rscript
# Live-R reference for the DR-proximal Brazil vaccine benchmark.
#
# Source: Qiu, Shi & Tchetgen Tchetgen, "Doubly robust proximal synthetic
# control" -- the authors' empirical analysis
#   DR_Proximal_SC/data_analysis/Brazil vaccine & pneumonia/analysis.Rmd
# pinned at commit 3bcb5ec (2022-10-31). The DR / PI.h / PI.q GMM functions
# below are reproduced verbatim from that commit.
#
# Two modifications, both necessary and documented:
#   1. Data. The Rmd reads data("pnas_brazil", package="InterventionEvaluatR")
#      (which pulls in INLA). We instead read the vendored age-group-9 subset
#      basedata/pnas_brazil_age9.csv -- the *same* data, extracted once -- so
#      the reference needs only gmm + Synth + tidyverse, no INLA.
#   2. Convergence. The Rmd uses optim(BFGS) with reltol ~ 1.5e-8, which stops
#      short of the GMM optimum (mild here; severe in the Kansas analysis). We
#      tighten reltol to 1e-13 so the printed numbers are the genuine optimum
#      that mlsynth's DR-OID solver targets.
#
# Prints parseable lines: "<cell>: <phi.hat * Y.scale>" in hospitalization units.

#   3. Packages. The Rmd loads the full `tidyverse` metapackage but uses only
#      dplyr / tidyr verbs (and the magrittr pipe); we load just those, so the
#      reference provisions without tidyverse's ~50-package dependency tree. No
#      numeric effect -- the GMM functions below are untouched.
suppressMessages({
  library(magrittr); library(dplyr); library(tidyr); library(gmm); library(splines)
})

args <- commandArgs(trailingOnly = TRUE)
csv <- if (length(args) >= 1) args[1] else "basedata/pnas_brazil_age9.csv"
raw.data <- read.csv(csv, check.names = FALSE)
RELTOL <- 1e-13

scales <- raw.data %>% summarize(across(!c(date, t), ~max(.x)))
d1 <- raw.data %>% mutate(across(!c(date, t), ~ .x / max(.x))) %>%
  pivot_longer(!c(date, t), names_to = "cause", values_to = "hospitalization")
Y <- d1 %>% filter(cause == "J12_18") %>% arrange(t) %>% pull(hospitalization)
Y.scale <- scales %>% pull("J12_18")
T <- max(raw.data$t); t <- 1:T; T0 <- 84

donors <- c("cJ20_J22", "E00_99", "E40_46")
W <- d1 %>% filter(cause %in% donors) %>% select(cause, t, hospitalization) %>%
  pivot_wider(names_from = cause, values_from = hospitalization) %>%
  arrange(t) %>% select(-t) %>% as.matrix
Z <- d1 %>% filter(!(cause %in% c("J12_18", donors))) %>%
  select(cause, t, hospitalization) %>%
  pivot_wider(names_from = cause, values_from = hospitalization) %>%
  arrange(t) %>% select(-t) %>% as.matrix
data <- list(t = t, W = W, Z = Z, Y = Y, T = T, T0 = T0)

# ---- DR / PI.h GMM (verbatim from commit 3bcb5ec; reltol tightened) --------
DR <- function(data) {
  T <- data$T; T0 <- data$T0; t <- data$t
  nW <- ncol(data$W); nZ <- ncol(data$Z[, Z.causes, drop = FALSE])
  evalh <- function(th, d) as.numeric(cbind(1, d$W) %*% th[1:(nW + 1)])
  evalq <- function(th, d) as.numeric(exp(cbind(1, d$Z[, Z.causes, drop = FALSE]) %*% th[(nW + 2):(nW + nZ + 2)]))
  gh <- cbind(1, data$Z); ngh <- ncol(gh); gq <- cbind(1, data$W); ngq <- ncol(gq)
  g <- function(th, d) {
    h <- evalh(th, d); q <- evalq(th, d)
    g1 <- (t <= T0) * (d$Y - h) * gh
    g2 <- (t > T0) * (matrix(th[(nW + nZ + 3):(nW + nZ + ngq + 2)], T, ngq, byrow = TRUE) - gq)
    g3 <- (t <= T0) * (q * gq - matrix(th[(nW + nZ + 3):(nW + nZ + ngq + 2)], T, ngq, byrow = TRUE))
    g4 <- (t > T0) * (th[nW + nZ + ngq + 3] - (d$Y - h) + th[nW + nZ + ngq + 4])
    g5 <- (t <= T0) * (th[nW + nZ + ngq + 4] - q * (d$Y - h))
    cbind(g1, g2, g3, g4, g5)
  }
  im.h <- lm(data$Y ~ ., data = as.data.frame(data$W), subset = t <= T0)
  im.q <- glm(I(t > T0) ~ ., data = as.data.frame(data$Z[, Z.causes, drop = FALSE]), family = binomial())
  ib <- coef(im.q); ib[1] <- ib[1] + log((T - T0) / T0)
  ipsi <- colMeans(gq[t > T0, , drop = FALSE])
  ipm <- mean(exp(cbind(1, data$Z[t <= T0, Z.causes, drop = FALSE]) %*% ib) *
              (data$Y[t <= T0] - predict(im.h, newdata = as.data.frame(data$W[t <= T0, , drop = FALSE]))))
  iphi <- mean(data$Y[t > T0] - predict(im.h, newdata = as.data.frame(data$W[t > T0, , drop = FALSE]))) - ipm
  th0 <- c(coef(im.h), ib, ipsi, iphi, ipm)
  m <- gmm(g = g, x = data, t0 = th0, wmatrix = "ident", method = "BFGS",
           control = list(maxit = 2e6, reltol = RELTOL), vcov = "iid")
  as.numeric(coef(m)[length(coef(m)) - 1])   # phi: second-to-last (pm is last)
}

PI.h <- function(data) {
  T <- data$T; T0 <- data$T0; t <- data$t; nW <- ncol(data$W)
  evalh <- function(th, d) as.numeric(cbind(1, d$W) %*% th[1:(nW + 1)])
  gh <- cbind(1, data$Z); ngh <- ncol(gh)
  g <- function(th, d) {
    h <- evalh(th, d)
    cbind(T / T0 * (t <= T0) * (d$Y - h) * gh, (t > T0) * (th[nW + 2] - (d$Y - h)))
  }
  im <- lm(data$Y ~ ., data = as.data.frame(data$W), subset = t <= T0)
  iphi <- mean(data$Y[t > T0] - predict(im, newdata = as.data.frame(data$W[t > T0, , drop = FALSE])))
  th0 <- c(coef(im), iphi)
  m <- gmm(g = g, x = data, t0 = th0, wmatrix = "ident", method = "BFGS",
           control = list(maxit = 2e6, reltol = RELTOL), vcov = "iid")
  as.numeric(coef(m)[nW + 2])
}

cat(sprintf("outcome_bridge_h: %.4f\n", PI.h(data) * Y.scale))
subsets <- list(A = "A10_B99_nopneumo",
                AD = c("A10_B99_nopneumo", "D50_89"),
                APD = c("A10_B99_nopneumo", "P05_07", "D50_89"))
for (nm in names(subsets)) {
  Z.causes <<- subsets[[nm]]
  cat(sprintf("DR_%s: %.4f\n", nm, DR(data) * Y.scale))
}
