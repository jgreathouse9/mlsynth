# Times the forward-selection loop in Kathleen T. Li's released Fun_FDID.R.
#
# Only the selection is timed. Her function does selection and then the final
# fit in one call, so the loop is reproduced here from her lines 12-56 and the
# final fit left out -- the selection is what scales in the donor count, and
# timing her whole function would fold in a constant.
#
# Usage: Rscript time_selection.R <panel.csv> <t1> <reps>
#   panel.csv: column 1 the treated unit, the rest controls, one row per period.

args <- commandArgs(trailingOnly = TRUE)
dat  <- as.matrix(read.csv(args[1]))
t1   <- as.integer(args[2])
reps <- as.integer(args[3])

y  <- dat[, 1]
x  <- dat[, 2:ncol(dat), drop = FALSE]
t  <- nrow(dat)
y1 <- y[1:t1]
no_control <- ncol(x)
control_ID <- 1:no_control

select_once <- function() {
  # her lines 12-25: pick the first control by R^2
  R2 <- rep(0, no_control)
  for (j in 1:no_control) {
    x1_DID <- x[1:t1, control_ID[j]]
    beta_DID <- mean(y1 - x1_DID)
    y1_hat_DID <- beta_DID + x1_DID
    R2[j] <- 1 - (mean((y1 - y1_hat_DID)^2)) / (mean((y1 - mean(y1))^2))
  }
  select <- which.max(R2)
  R2final <- max(R2)

  # her lines 31-56: add one at a time, recomputing every candidate average
  for (k in 2:no_control) {
    left <- setdiff(1:no_control, select)
    R2 <- rep(0, length(left))
    for (jj in seq_along(left)) {
      control_1 <- x[1:t1, c(select, left[jj]), drop = FALSE]
      x1_f_DID <- rowMeans(control_1)
      beta_f_DID <- mean(y1 - x1_f_DID)
      y1_hat_f_DID <- beta_f_DID + x1_f_DID
      R2[jj] <- 1 - (mean((y1 - y1_hat_f_DID)^2)) / (mean((y1 - mean(y1))^2))
    }
    index <- left[which.max(R2)]
    R2final <- c(R2final, max(R2))
    select <- append(select, index)
  }
  list(select = select, num_c = which.max(R2final))
}

out <- select_once()                       # once for the answer
elapsed <- numeric(reps)
for (r in 1:reps) {
  t0 <- proc.time()[["elapsed"]]
  select_once()
  elapsed[r] <- proc.time()[["elapsed"]] - t0
}

cat(sprintf("impl\tR (Li Fun_FDID.R selection loop)\n"))
cat(sprintf("num_c\t%d\n", out$num_c))
cat(sprintf("selected\t%s\n", paste(out$select[1:out$num_c], collapse = ",")))
cat(sprintf("median_seconds\t%.6f\n", median(elapsed)))
cat(sprintf("min_seconds\t%.6f\n", min(elapsed)))
cat(sprintf("reps\t%d\n", reps))
