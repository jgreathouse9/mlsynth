suppressMessages(library(augsynth))
d <- read.csv("basedata/Teachingaugsynth.scv")
d <- d[!d$State %in% c("DC","WI","AK","HI"), ]
d <- d[d$year >= 1959 & d$year <= 1997, ]
d$cbr <- as.integer(d$year >= ifelse(is.na(d$YearCBrequired), Inf, d$YearCBrequired))

# vignette covariates: 1959 snapshots of per-capita income and student-teacher ratio
snap <- d[d$year == 1959, c("State", "perinc", "studteachratio")]
names(snap) <- c("State", "perinc_1959", "studteachratio_1959")
d <- merge(d, snap, by = "State")

m <- multisynth(lnppexpend ~ cbr | perinc_1959 + studteachratio_1959,
                State, year, data = d)
s <- summary(m)
a <- s$att[s$att$Level == "Average" & s$att$Time %in% 0:10, ]
a <- a[order(a$Time), ]

cat("== REFERENCE VALUES (covariates) ==\n")
cat(sprintf("nu\t%.7f\n", m$nu))
cat(sprintf("global_l2\t%.6f\n", m$global_l2))
cat(sprintf("scaled_global_l2\t%.6f\n", m$scaled_global_l2))
cat(sprintf("ind_l2\t%.6f\n", m$ind_l2))
cat(sprintf("scaled_ind_l2\t%.6f\n", m$scaled_ind_l2))
cat(sprintf("att\t%.6f\n", mean(a$Estimate)))
for (k in 0:10) cat(sprintf("tau_%02d\t%.6f\n", k, a$Estimate[k + 1]))
for (k in 0:10) cat(sprintf("se_%02d\t%.6f\n", k, a$Std.Error[k + 1]))
cat("== SESSION ==\n"); cat(sprintf("augsynth %s | %s\n", as.character(packageVersion("augsynth")), R.version.string))
