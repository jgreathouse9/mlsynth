library(gpss)
dc <- read.csv("dc_series.csv", stringsAsFactors=FALSE); dc$date <- as.Date(dc$date)
dc$month <- factor(sprintf("%02d", as.integer(format(dc$date, "%m"))))
r <- gpss::gp_its(y=dc$handgun_rate, dates=dc$date, date_treat=as.Date("2008-07-01"),
  covariates=data.frame(month=dc$month), kernel_type="gaussian_periodic_linear",
  period=12, mixed_data=TRUE, cat_columns="month", interval_type="prediction",
  placebo_check=TRUE, placebo_periods=4)
pe <- r$placebo_estimates
print(pe[, c("time_id","relative_time","tau","se","z_score","cover")], digits=12)
write.csv(pe, "placebo_reference.csv", row.names=FALSE)
