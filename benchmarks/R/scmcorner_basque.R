suppressMessages({library(kernlab); library(LowRankQP); library(lpSolve)})
have_lrqp <- requireNamespace("LowRankQP", quietly=TRUE)
source("/tmp/SCM-Debug/scm.corner.R")
d <- read.csv("/home/user/mlsynth/basedata/basque_mscmt.csv", check.names=FALSE)
d <- d[d$regionname != "Spain (Espana)", ]
treated <- "Basque Country (Pais Vasco)"
donors <- setdiff(unique(d$regionname), treated)
# Y over fit window 1960-1969
yrs <- 1960:1969
wide <- reshape(d[d$year %in% yrs, c("regionname","year","gdpcap")],
                idvar="year", timevar="regionname", direction="wide")
wide <- wide[order(wide$year), ]
colnames(wide) <- sub("^gdpcap\\.", "", colnames(wide))
Y1pre <- as.matrix(wide[[treated]])
Y0pre <- as.matrix(wide[, donors])
# Predictors: AG means over their windows (just need X for step2 V; W is from step1)
predspec <- list(
  c("school.illit",1964,1969), c("school.prim",1964,1969), c("school.med",1964,1969),
  c("school.higher",1964,1969), c("invest",1964,1969), c("gdpcap",1960,1969),
  c("sec.agriculture",1961,1969), c("sec.energy",1961,1969), c("sec.industry",1961,1969),
  c("sec.construction",1961,1969), c("sec.services.venta",1961,1969),
  c("sec.services.nonventa",1961,1969), c("popdens",1969,1969))
mk <- function(v,a,b){ sub <- d[d$year>=as.numeric(a)&d$year<=as.numeric(b),]
  sapply(c(treated,donors), function(u) mean(sub[sub$regionname==u, v], na.rm=TRUE)) }
P <- t(sapply(predspec, function(s) mk(s[1],s[2],s[3])))
X1 <- as.matrix(P[,treated]); X0 <- P[,donors]
res <- scm.corner(Y1pre, Y0pre, X1, X0)
W <- res$W[,2]
names(W) <- donors; W <- W[order(-W)]
cat("=== scm.corner Basque W (>0.001) ===\n"); print(round(W[W>0.001],4))
