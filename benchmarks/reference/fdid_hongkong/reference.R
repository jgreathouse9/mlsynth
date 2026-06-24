# Reference run for the `fdid_hongkong` benchmark case.
#
# Runs Kathleen T. Li's own released Forward DiD code (Fun_FDID.R, from the
# Marketing Science replication package for "Forward Difference-in-Differences,"
# DOI 10.1287/mksc.2022.0212) on her released Hong Kong GDP panel (GDP.csv, the
# Hsiao-Ching-Wan 2012 data: Hong Kong treated, 24 control economies, t1 = 44
# pre-treatment quarters). Forward selection is deterministic, so these are the
# genuine FDID.R outputs the Python case pins against -- not numbers transcribed
# from the replication readme.
#
# The conventional DiD (all 24 controls) is computed inline with the same
# intercept-only DiD formula the author's MATLAB driver uses, to capture the
# DID_ATT / %ATT / R^2 the readme reports from MATLAB (Fun_FDID.R itself returns
# only the FDID outputs).
#
# Code and data are vendored verbatim alongside this script (see NOTICE).
#
# Run from the repository root:  Rscript benchmarks/reference/fdid_hongkong/reference.R
source("benchmarks/reference/fdid_hongkong/Fun_FDID.R")

data <- read.csv("benchmarks/reference/fdid_hongkong/GDP.csv", sep = ",", header = TRUE)

t  <- dim(data)[1]
no_control <- dim(data)[2] - 1
control_ID <- 1:no_control
t1 <- 44                      # pre-treatment sample size (author's driver)
y  <- data[, 1]
y1 <- data[1:t1, 1]
y2 <- data[(t1 + 1):t, 1]
x  <- data[, 2:dim(data)[2]]

# --- Forward DiD: the author's own function ---
FDID <- FDID_fun(no_control, control_ID, x, y1, y2, t1, t)

# --- Conventional DiD: all controls, intercept-only DiD (matches the MATLAB
#     driver's reported numbers) ---
x1_DID <- rowMeans(x[1:t1, ])
x2_DID <- rowMeans(x[(t1 + 1):t, ])
beta_DID <- mean(y1 - x1_DID)
y1_hat_DID <- beta_DID + x1_DID
y2_hat_DID <- beta_DID + x2_DID
ATT_DID <- mean(y2 - y2_hat_DID)
ATT_DID_per <- 100 * ATT_DID / mean(y2_hat_DID)
R2_DID <- 1 - mean((y1 - y1_hat_DID)^2) / mean((y1 - mean(y1))^2)

cat("== REFERENCE VALUES ==\n")
cat(sprintf("fdid_att\t%.6f\n", FDID$ATT_FDID))
cat(sprintf("fdid_att_pct\t%.6f\n", FDID$ATT_FDID_per))
cat(sprintf("fdid_r2_pre\t%.6f\n", FDID$R2_FDID))
cat(sprintf("fdid_n_controls\t%d\n", FDID$num_c))
cat(sprintf("did_att\t%.6f\n", ATT_DID))
cat(sprintf("did_att_pct\t%.6f\n", ATT_DID_per))
cat(sprintf("did_r2_pre\t%.6f\n", R2_DID))
cat("== SESSION INFO ==\n")
print(sessionInfo())
