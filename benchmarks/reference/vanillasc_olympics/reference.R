#!/usr/bin/env Rscript
# tidysynth gold for the Yoneoka et al. (2022) Tokyo Olympics panel.
#
#   Rscript benchmarks/reference/vanillasc_olympics/reference.R
#   (install the reference first: bash benchmarks/R/install_tidysynth_olympics.sh)
#
# Reads the SAME vendored panel the Python case reads --
# basedata/yoneoka_olympics_covid.parquet -- through nanoparquet. No CSV detour,
# so the two sides cannot run on different inputs.
#
# The call follows the authors' res.Rmd: the 34-predictor specification, the
# CG outer search, and LowRankQP for the inner quadratic program. All three
# intervention timings the paper reports are run -- the main one at date2 = 53
# (23 July 2021, the opening ceremony) and the +/- 7-day sensitivity checks at
# 60 and 46.
#
# tidysynth's `i_time` is the LAST PRE-TREATMENT period, not the first treated
# one: its own grab_significance() splits on `time_unit <= trt_time`, and the
# authors' optimization_window runs 25:53 inclusive. The Python side sets its
# treatment indicator from date2 > i_time to match.
#
# The lagged-case predictors are spot values at 25/32/39/46/53(/60) -- four,
# five or six of them depending on how much pre-period the timing leaves.
#
# Emitted:
#   gold_trend_<T>.csv    observed, synthetic and gap by period
#   gold_weights_<T>.csv  donor weights (W)
#   gold_predw_<T>.csv    predictor weights (V)
#   gold_signif_<T>.csv   the placebo table: pre/post MSPE, ratio, rank, p
#   provenance.txt

suppressMessages({
  library(dplyr)
  library(readr)
  library(tidysynth)
  library(nanoparquet)
})

PANEL <- "basedata/yoneoka_olympics_covid.parquet"
OUT <- "benchmarks/reference/vanillasc_olympics"
TIMINGS <- c(46, 53, 60)
LAGS <- list(`46` = c(25, 32, 39, 46),
             `53` = c(25, 32, 39, 46, 53),
             `60` = c(25, 32, 39, 46, 53, 60))
# The authors name only the 53-spec's lag predictors ("3-week lagged cases" and
# so on); the other two scripts leave them as cases2..cases6. The names do not
# enter the fit, so they are generated uniformly here.
lag_label <- function(t, trt) paste0((trt - t) / 7, "-week lagged cases")

stopifnot(file.exists(PANEL))
full <- as.data.frame(read_parquet(PANEL))

for (TRT in TIMINGS) {
  d <- full %>% filter(date2 < 75)

  obj <- d %>%
    synthetic_control(outcome = new_cases_smoothed_per_million,
                      unit = location, time = date2,
                      i_unit = "Japan", i_time = TRT,
                      generate_placebos = TRUE) %>%
    generate_predictor(time_window = 25:TRT,
      `Number of tests for COVID-19` = mean(new_tests_smoothed_per_thousand, na.rm = TRUE),
      `COVID-19 vaccination` = mean(people_fully_vaccinated_per_hundred, na.rm = TRUE),
      `Stringency index` = mean(stringency_index, na.rm = TRUE),
      Temparature = mean(temparature, na.rm = TRUE),
      Precipitation = mean(rainfall, na.rm = TRUE),
      `Human mobility (retail and recreation)` = mean(retail, na.rm = TRUE),
      `Human mobility (groceries and pharmacies)` = mean(grocery, na.rm = TRUE),
      `Human mobility (transit stations)` = mean(transit, na.rm = TRUE),
      `Human mobility (workplaces)` = mean(work, na.rm = TRUE),
      `Human mobility (parks)` = mean(park, na.rm = TRUE),
      `Human mobility (residential)` = mean(resident, na.rm = TRUE)) %>%
    generate_predictor(time_window = 25,
      Age = median_age, `Population (0-14)` = Population_14,
      `Population (65-)` = Population_65, `Population density` = Population_density,
      `Electoral democracy index` = democracy,
      `Infant mortality rate` = Mortality_rate_infant,
      `Gross domestic product (GDP)` = gdp_per_capita,
      `Human Development Index` = human_development_index,
      `Diabetes prevalence` = diabetes_prevalence,
      `Health expenditure` = `Current health expenditure`,
      life_expectancy = life_expectancy, `Surface area` = `Surface area`,
      `Unemployment rate` = Unemployment, PM2.5 = PM2.5,
      `International migrant stock` = `International migrant stock`,
      `Houseshold size` = houseshold_size,
      `Number of hospital beds` = hospital_beds_per_thousand,
      `Asia flag` = Asia)

  # The lagged-outcome spot predictors, one generate_predictor() per lag. The
  # name is built per iteration, so it goes in through tidy evaluation.
  for (t in LAGS[[as.character(TRT)]]) {
    nm <- lag_label(t, TRT)
    obj <- generate_predictor(obj, time_window = t,
                              !!nm := new_cases_smoothed_per_million)
  }

  obj <- obj %>%
    generate_weights(optimization_window = 25:TRT,
                     optimization_method = "CG", quadopt = "LowRankQP") %>%
    generate_control()

  sc <- obj %>% grab_synthetic_control(placebo = FALSE) %>%
    mutate(diff = real_y - synth_y)
  write_csv(sc, file.path(OUT, sprintf("gold_trend_%d.csv", TRT)))
  write_csv(grab_unit_weights(obj, placebo = FALSE),
            file.path(OUT, sprintf("gold_weights_%d.csv", TRT)))
  write_csv(grab_predictor_weights(obj, placebo = FALSE),
            file.path(OUT, sprintf("gold_predw_%d.csv", TRT)))
  sig <- obj %>% grab_significance()
  write_csv(sig, file.path(OUT, sprintf("gold_signif_%d.csv", TRT)))

  cat(sprintf("T=%d  pre-RMSPE %.5f  Japan p %.6f\n", TRT,
              sqrt(mean(sc$diff[sc$time_unit <= TRT]^2)),
              sig$fishers_exact_pvalue[sig$unit_name == "Japan"]))
}

writeLines(c(
  paste("tidysynth:", as.character(packageVersion("tidysynth"))),
  "commit: 58eda583ea28645a3531cd769b7852ff504656df (edunford/tidysynth)",
  paste("Synth:", as.character(packageVersion("Synth"))),
  paste("LowRankQP:", as.character(packageVersion("LowRankQP"))),
  paste("nanoparquet:", as.character(packageVersion("nanoparquet"))),
  paste("R:", paste0(R.version$major, ".", R.version$minor)),
  paste("panel:", PANEL),
  paste("timings:", paste(TIMINGS, collapse = ",")),
  "call: tidysynth, 34 predictors, optimization_method=CG, quadopt=LowRankQP"
), file.path(OUT, "provenance.txt"))
