# Reference run for the `scmo_germany` benchmark case.
#
# Runs Tian, Lee & Panchenko's own West-German reunification analysis (Germany.R
# from their replication package) and emits the 1989 balance table -- the
# "Synthetic West Germany (multiple outcomes)" and "(single outcome)" columns of
# the paper's Table 2 / the package's Output/Ger_tab.txt. The synthetic-control
# weights come from the authors' own fn_W quadratic program (quadprog::solve.QP),
# exactly as Germany.R computes them; only the plotting is dropped. These are the
# genuine package outputs the Python case pins against, not numbers transcribed
# from the printed table.
#
# Data are vendored verbatim alongside this script (see NOTICE): repgermany.csv
# (the Abadie et al. panel) and all.csv (a lossless CSV export of the package's
# Data_Germany/all.xlsx OECD indicators -- avoids an readxl dependency; the
# filtering mirrors Germany.R's read of all.xlsx).
#
# Run from the repository root:  Rscript benchmarks/reference/scmo_germany/reference.R
suppressMessages(library(quadprog))

fn_W <- function(Zi, ZJ, V) {
  J <- nrow(ZJ)
  Dmat <- ZJ %*% V %*% t(ZJ) + (10^-7) * diag(J)
  dvec <- ZJ %*% V %*% Zi
  Amat <- cbind(cbind(rep(1, J)), diag(1, J, J))
  bvec <- c(1, rep(0, J))
  solve.QP(Dmat, dvec, Amat, bvec, meq = 1)$solution
}

base <- "benchmarks/reference/scmo_germany"
d <- read.csv(file.path(base, "repgermany.csv"), check.names = FALSE)

# Per-capita GDP matrix: one row per country (in index order), columns 1960-2003.
Y <- c()
for (i in unique(d$index)) Y <- rbind(Y, unlist(d[d$index == i, "gdp"]))
wg <- unique(d$index) == 7          # West Germany is index 7

# --- Single outcome: 30 years of pre-treatment GDP (1960-1989), Abadie et al. ---
Z <- Y[, unique(d$year) < 1990]
Z <- scale(Z, center = FALSE, scale = apply(Z, 2, sd))
V <- rep(1 / ncol(Z), ncol(Z))
W <- fn_W(cbind(Z[wg, ]), Z[!wg, ], diag(V, ncol(Z)))

# --- Multiple outcomes: nine indicators measured in 1989 ---
year <- 1989
data_raw <- read.csv(file.path(base, "all.csv"), check.names = FALSE)
data_raw <- data_raw[data_raw$Year == year,
                     c("Subject", "Country", "Year", "Value", "PowerCode", "Unit")]
var_names <- unique(data_raw$Subject)
country_names <- unique(d$country)
data_raw$Country[data_raw$Country == "United States"] <- "USA"
data_raw$Country[data_raw$Country == "United Kingdom"] <- "UK"
data_raw$Country[data_raw$Country == "Germany"] <- "West Germany"

data <- data.frame(matrix(NA, length(country_names), length(var_names)))
colnames(data) <- var_names
rownames(data) <- country_names
for (i in seq_along(country_names)) {
  for (j in seq_along(var_names)) {
    sel <- data_raw$Country == country_names[i] & data_raw$Subject == var_names[j]
    if (sum(sel) == 1) data[i, j] <- data_raw$Value[sel]
  }
}

data$`Electricity generation` <- data$`Electricity generation` / data$`Population levels`
data$`Triadic patent families` <- data$`Triadic patent families` / data$`Population levels`
data$`GDP per capita` <- d[d$year == year, "gdp"]
data$`Trade openness` <- d[d$year == year, "trade"]

data1 <- data[, c("Private social expenditure",
                  "Total primary energy supply per unit of GDP",
                  "Electricity generation", "Triadic patent families",
                  "Real GDP growth", "CPI: all items", "Trade openness",
                  "Total tax revenue", "GDP per capita")]
Z <- data1
Z <- Z[, complete.cases(t(Z))]
Z <- scale(Z, center = FALSE, scale = apply(Z, 2, sd))
V <- rep(1 / ncol(Z), ncol(Z))
W1 <- fn_W(cbind(Z[wg, ]), Z[!wg, ], diag(V, ncol(Z)))

donors <- as.matrix(data1[!wg, , drop = FALSE])
multi  <- as.numeric(t(donors) %*% W1)   # synthetic (multiple outcomes), 1989
single <- as.numeric(t(donors) %*% W)    # synthetic (single outcome), 1989
names(multi) <- names(single) <- colnames(data1)
donor_countries <- country_names[!wg]

cat("== REFERENCE VALUES ==\n")
cat(sprintf("multi_gdp_pc_1989\t%.6f\n", multi[["GDP per capita"]]))
cat(sprintf("multi_cpi_1989\t%.6f\n", multi[["CPI: all items"]]))
cat(sprintf("multi_trade_1989\t%.6f\n", multi[["Trade openness"]]))
cat(sprintf("multi_tax_1989\t%.6f\n", multi[["Total tax revenue"]]))
cat(sprintf("multi_gdp_growth_1989\t%.6f\n", multi[["Real GDP growth"]]))
cat(sprintf("single_gdp_pc_1989\t%.6f\n", single[["GDP per capita"]]))
for (i in seq_along(donor_countries)) {
  if (W1[i] > 1e-4) cat(sprintf("weight\t%s\t%.6f\n", donor_countries[i], W1[i]))
}
cat("== SESSION INFO ==\n")
print(sessionInfo())
