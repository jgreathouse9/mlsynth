suppressMessages({library(foreign); library(scinference)})
ct <- read.dta("basedata/carbontax_data.dta")
years <- sort(unique(ct$year)); u <- unique(as.character(ct$country))
W <- sapply(u, function(x){s <- ct[as.character(ct$country)==x,]
                           s$CO2_transport_capita[match(years,s$year)]})
Y1 <- W[,"Sweden"]; Y0 <- W[,u!="Sweden",drop=FALSE]
T0 <- sum(years<1990); T1 <- sum(years>=1990); T <- T0+T1
cat(sprintf("%16s %18s\n","injected effect","scinference mb p"))
for (eff in c(0,1,5,20,100)) {
  y <- Y1; y[(T0+1):T] <- y[(T0+1):T] + eff
  r <- scinference(Y1=y,Y0=Y0,T1=T1,T0=T0,inference_method="conformal",
                   estimation_method="sc",permutation_method="mb",alpha=0.1,lsei_type=1)
  cat(sprintf("%16.1f %18.6f\n", eff, r$p_val))
}
cat(sprintf("\nT1/T = %.6f   1/T = %.6f\n", T1/T, 1/T))
