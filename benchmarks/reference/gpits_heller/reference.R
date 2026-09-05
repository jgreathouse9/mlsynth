# Ground truth: gpss::gp_its on the D.C. Heller series, exactly as
# gpits/code/02_heller_main.R configures it. Writes reference.json.
library(gpss)

dc <- read.csv("dc_series.csv", stringsAsFactors = FALSE)
dc$date <- as.Date(dc$date)
dc$month <- factor(sprintf("%02d", as.integer(format(dc$date, "%m"))))

res <- gpss::gp_its(
  y            = dc$handgun_rate,
  dates        = dc$date,
  date_treat   = as.Date("2008-07-01"),
  covariates   = data.frame(month = dc$month),
  kernel_type  = "gaussian_periodic_linear",
  period       = 12,
  scale        = TRUE,
  optimize     = TRUE,
  interval_type = "prediction",
  alpha        = 0.05,
  mixed_data   = TRUE,
  cat_columns  = "month",
  placebo_check = FALSE,
  verbose      = FALSE
)

out <- list(
  b       = res$gp_model$b,
  s2      = res$gp_model$s2,
  Y_sd    = res$gp_model$Y.init.sd,
  Y_mean  = res$gp_model$Y.init.mean,
  period_scaled = res$gp_model$period_scaled,
  y0_hat  = as.numeric(res$y0_hat),
  y0_se   = as.numeric((res$y0_hat_upr - res$y0_hat) / qnorm(0.975)),
  tau_t   = as.numeric(res$estimates$tau_t),
  tau_cum = as.numeric(res$estimates$tau_cum),
  tau_cum_se = as.numeric(res$se$tau_cum_se),
  tau_avg = as.numeric(res$estimates$tau_avg)
)
cat(sprintf("b = %.10f\ns2 = %.10f\nY.init.sd = %.10f\n", out$b, out$s2, out$Y_sd))
cat("tau_cum   :", sprintf("%.10f", out$tau_cum), "\n")
cat("tau_cum_se:", sprintf("%.10f", out$tau_cum_se), "\n")
cat(sprintf("cumulative 4-month: %.4f  95%% CI [%.4f, %.4f]\n",
            tail(out$tau_cum, 1),
            tail(out$tau_cum, 1) - qnorm(0.975) * tail(out$tau_cum_se, 1),
            tail(out$tau_cum, 1) + qnorm(0.975) * tail(out$tau_cum_se, 1)))

json <- paste0("{\n", paste(sapply(names(out), function(k) {
  v <- out[[k]]
  sprintf('  "%s": %s', k, if (length(v) == 1) sprintf("%.12g", v)
          else paste0("[", paste(sprintf("%.12g", v), collapse = ", "), "]"))
}), collapse = ",\n"), "\n}\n")
writeLines(json, "reference.json")
cat("wrote reference.json\n")
