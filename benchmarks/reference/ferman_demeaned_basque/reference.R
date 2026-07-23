# Ferman & Pinto (2021), "Synthetic controls with imperfect pretreatment fit",
# Quantitative Economics 12:1197-1221 -- the *demeaned SC* estimator, verbatim
# from the authors' replication code (_aux.R :: synth_control_est_demean): donor
# weights >= 0 summing to 1, with a FREE intercept (the "demeaning"), solved as a
# quadratic program. Basque Country / ETA terrorism (Abadie & Gardeazabal 2003),
# treatment 1975 -- the identified regime (pre-periods 1955-1974 > donors).
suppressMessages(library(quadprog)); suppressMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1) args[1] else "basedata/basque_data.csv"

d <- read.csv(data_path, stringsAsFactors = FALSE)
d <- d[!d$regionname %in% c("Spain (Espana)", "Syntetic Basque Country"), ]
w <- reshape(d[, c("regionname", "year", "gdpcap")], idvar = "year",
             timevar = "regionname", direction = "wide")
w <- w[order(w$year), ]; colnames(w) <- sub("gdpcap.", "", colnames(w), fixed = TRUE)
treated <- "Basque Country (Pais Vasco)"
donors  <- setdiff(colnames(w), c("year", treated))
om <- as.matrix(cbind(w[[treated]], w[, donors])); yr <- w$year
pret <- om[yr <= 1974, ]; postt <- om[yr > 1974, ]

# synth_control_est_demean: min || y - a - X w ||^2  s.t.  w >= 0, sum(w) = 1, a free
y <- pret[, 1]; X <- cbind(1, pret[, -1])
Dmat <- t(X) %*% X; dvec <- t(X) %*% y
Amat <- t(rbind(c(0, rep(1, ncol(X) - 1)), cbind(0, diag(ncol(X) - 1))))
bvec <- c(1, rep(0, ncol(X) - 1))
m <- solve.QP(Dmat, dvec, Amat, bvec, meq = 1)
wt <- m$solution[-1]; intercept <- m$solution[1]
names(wt) <- donors
effects <- as.numeric(-intercept + postt %*% c(1, -wt))

weights <- as.list(round(wt, 8))
out <- list(
  values = list(att = round(mean(effects), 8), intercept = round(intercept, 8),
                top_donor_cataluna = round(unname(wt["Cataluna"]), 8)),
  weights = weights
)
cat(toJSON(out, auto_unbox = TRUE, digits = 10))
