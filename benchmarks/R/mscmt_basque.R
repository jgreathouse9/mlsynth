suppressMessages({library(Synth); library(MSCMT)})
data(basque)
Basque <- listFromLong(basque, unit.variable="regionno", time.variable="year",
                       unit.names.variable="regionname")
school.sum <- with(Basque, colSums(school.illit+school.prim+school.med+school.high+school.post.high))
Basque$school.higher <- Basque$school.high + Basque$school.post.high
for (item in c("school.illit","school.prim","school.med","school.higher"))
  Basque[[item]] <- 5*100*t(t(Basque[[item]])/school.sum)
treatment.identifier <- "Basque Country (Pais Vasco)"
controls.identifier  <- setdiff(colnames(Basque[[1]]), c(treatment.identifier,"Spain (Espana)"))
times.dep  <- cbind("gdpcap"=c(1960,1969))
times.pred <- cbind("school.illit"=c(1964,1969),"school.prim"=c(1964,1969),
  "school.med"=c(1964,1969),"school.higher"=c(1964,1969),"invest"=c(1964,1969),
  "gdpcap"=c(1960,1969),"sec.agriculture"=c(1961,1969),"sec.energy"=c(1961,1969),
  "sec.industry"=c(1961,1969),"sec.construction"=c(1961,1969),
  "sec.services.venta"=c(1961,1969),"sec.services.nonventa"=c(1961,1969),"popdens"=c(1969,1969))
agg.fns <- rep("mean", ncol(times.pred))
res <- mscmt(Basque, treatment.identifier, controls.identifier, times.dep, times.pred,
             agg.fns, seed=42, outer.optim="DEoptim", verbose=FALSE)
w <- res$w; w <- w[order(-w)]
cat("=== MSCMT Basque weights (>0.001) ===\n"); print(round(w[w>0.001],5))
cat(sprintf("RMSPE=%.6f\n", res$rmspe))
dd <- did(res, range.post=c(1970,1990))
cat(sprintf("did effect.size=%.6f  average.post=%.6f\n", dd$effect.size, dd$average.post))
