#!/usr/bin/env Rscript
# Which subsetting rule is Wan, Xie & Hsiao (2018) Table 2's "MAE-rule" column?
#
# Nineteen of the twenty cells of the paper's Design 6a column reproduce (see
# benchmarks/cases/wan_pda_vs_scm.py). The (5,5) MAE-rule PDA cell does not: the
# paper prints 0.77 and every reconstruction lands near 0.99. Since the
# 1000-rule cell beside it agrees, the estimator is not the problem and the
# subsetting rule is.
#
# This script settles it with the authors' own code. It is sim6a_c9.R at the
# (J,T0) = (5,5) cell with the estimation untouched -- their DGP, their seed,
# their pampe(select="AICc", nvmax=t0-4), their Synth(method="BFGS") -- with the
# xlsx export replaced by a print of the PDA and SCM means under every rule the
# script defines, plus the equal-K adjustment the paper's Section 3 describes:
#
#   s1  MAE_{0,SCM}  < 0.2 |ybar_1|        (Gardeazabal & Vega-Bayo's rule)
#   s2  0.5 |ybar_1| < |ybar_all|
#   s3  MAE_{0,PDA}  < 0.2 |ybar_1|
#   s4  MAE_{0,PDA}  < 0.2 |ybar_all|
#   s5  adjusted R^2 > 0.9
#   s6  |ybar_1| / mean|ybar_j| < 2
#   eqK the K best PDA matches, K = #{s1}
#
# At 400 replications the PDA column spans 0.847 to 0.874 across all seven,
# against an unfiltered 0.861 and the paper's printed 0.77, while the SCM
# column reproduces under every one of them. The result is negative: no rule in
# the authors' own script produces the printed cell.
#
# Usage:
#   ITR=400 Rscript benchmarks/R/wan_subsetting_rules.R
set.seed(1234)
library(Synth); library(leaps); library(pampe)
skipError = function(e) return(NA)

itr = as.integer(Sys.getenv("ITR", "400")); j = 5; t0 = 5; t1 = 10; t = t0 + t1
n = itr
mse.scm=mer.scm=map.scm=pre.scm=lossv=mse.pda=mer.pda=map.pda=pre.pda=adjr2 =
  miuy=miuyall=miuyalla=ratioy=ratioa=nk.pda = matrix(0,nrow=itr,ncol=1)

for (it in 1:itr){
  if (it %% 25 == 0) { cat(sprintf("progress %d/%d\n", it, itr)); flush(stdout()) }
  eta = rchisq(1,1)
  lam = matrix(0,nrow=t,ncol=1)
  for (s in 2:t){ lam[s] = eta + lam[s-1] + rnorm(1,mean=0,sd=0.5) }
  miu = matrix(1,nrow=1,ncol=(j+1))
  eps = matrix(rnorm(t*(j+1),mean=0,sd=0.25),nrow=t,ncol=(j+1))
  y   = lam%*%miu + eps

  miuy[it,1]=mean(y[1:t0,1]); miuyall[it,1]=mean(y[1:t0,])
  miuyalla[it,1]=mean(abs(apply(y[1:t0,],2,mean)))
  ratioy[it,1]=abs(miuy[it,1])/miuyall[it,1]
  ratioa[it,1]=abs(miuy[it,1])/miuyalla[it,1]

  ydata = as.matrix(y)
  nam = as.matrix(seq(1,(j+1),1))%x%rep(1,t)
  yr0 = as.matrix(seq(1,t,1))%*%rep(1,(j+1)); yr = matrix(yr0,nrow=t*(j+1),ncol=1)
  xx  = matrix(0,nrow=t*(j+1),ncol=t0)
  for (ix in 1:t0){ xx[,ix] = as.matrix(ydata[ix,] %x% rep(1,t)) }
  ycol = matrix(ydata,nrow=t*(j+1),ncol=1)
  data = as.data.frame(cbind(nam,yr,ycol,xx))
  colnames(data) = c("country","time","ycol", paste0("p",1:t0))
  s.data = dataprep(foo=data, predictors=paste0("p",1:t0), predictors.op="mean",
                    time.predictors.prior=1:t0, dependent="ycol",
                    unit.variable="country", time.variable="time",
                    treatment.identifier=1, controls.identifier=c(2:(j+1)),
                    time.optimize.ssr=1:t0, time.plot=1:t)
  scm = tryCatch(synth(data.prep.obj=s.data, method="BFGS"), error=skipError)
  w.scm   = tryCatch(as.matrix(scm$solution.w), error=function(e) NA)
  x0.post = tryCatch(as.matrix(ydata[(t0+1):t,2:(j+1)]), error=function(e) NA)
  x1.post = tryCatch(as.matrix(ydata[(t0+1):t,1]), error=function(e) NA)
  a.scm   = tryCatch(as.matrix(x1.post - x0.post%*%w.scm), error=function(e) NA)
  mse.scm[it] = tryCatch(mean(a.scm^2), error=function(e) NA)
  mer.scm[it] = tryCatch(mean(a.scm), error=function(e) NA)
  map.scm[it] = tryCatch(100*mean(abs(a.scm)/abs(x1.post)), error=function(e) NA)
  x0.pre = tryCatch(as.matrix(ydata[1:t0,2:(j+1)]), error=function(e) NA)
  x1.pre = tryCatch(as.matrix(ydata[1:t0,1]), error=function(e) NA)
  pre.scm[it] = tryCatch(mean(abs(x1.pre - x0.pre%*%w.scm)), error=function(e) NA)
  lossv[it]   = tryCatch(sqrt(scm$loss.v), error=function(e) NA)

  d.pda = as.data.frame(y)
  pda = tryCatch(pampe(time.pretr=1:t0, time.tr=(t0+1):t, treated=1,
                       nvmax=(t0-4), data=d.pda, select="AICc"), error=skipError)
  yhat.pda = tryCatch(as.matrix(pda$counterfactual[(t0+1):t,2]), error=function(e) NA)
  nk.pda[it] = tryCatch(length(pda$controls), error=function(e) NA)
  a.pda = tryCatch(as.matrix(x1.post - yhat.pda), error=function(e) NA)
  mse.pda[it] = tryCatch(mean(a.pda^2), error=function(e) NA)
  mer.pda[it] = tryCatch(mean(a.pda), error=function(e) NA)
  map.pda[it] = tryCatch(100*mean(abs(a.pda)/abs(x1.post)), error=function(e) NA)
  x.pda = tryCatch(as.matrix(d.pda[1:t0,pda$controls]), error=function(e) NA)
  y.pda = as.matrix(d.pda[1:t0,1])
  lm.pda = tryCatch(lm(y.pda~x.pda), error=skipError)
  adjr2[it] = tryCatch(summary(lm.pda)$adj.r.squared, error=function(e) NA)
  pre.pda[it] = tryCatch(mean(abs(y.pda - cbind(1,x.pda)%*%as.matrix(lm.pda$coefficients))),
                         error=function(e) NA)
}

itr.table = cbind(mse.scm,mse.pda,mer.scm,mer.pda,map.scm,map.pda,nk.pda,lossv,
                  adjr2,pre.scm,pre.pda,miuy,miuyall,miuyalla,ratioy,ratioa)
colnames(itr.table) = c("mse.scm","mse.pda","mer.scm","mer.pda","map.scm","map.pda",
                        "nk.pda","lossv","adjr2","pre.scm","pre.pda",
                        "miuy","miuyall","miuyalla","ratioy","ratioa")
rules = list(
  all = rep(TRUE,itr),
  s1  = itr.table[,"pre.scm"]  < 0.2*abs(itr.table[,"miuy"]),
  s2  = 0.5*abs(itr.table[,"miuy"]) < abs(itr.table[,"miuyall"]),
  s3  = itr.table[,"pre.pda"]  < 0.2*abs(itr.table[,"miuy"]),
  s4  = itr.table[,"pre.pda"]  < 0.2*abs(itr.table[,"miuyall"]),
  s5  = itr.table[,"adjr2"]    > 0.9,
  s6  = itr.table[,"ratioa"]   < 2)
cat(sprintf("%-5s %8s %8s %6s\n","rule","mse.pda","mse.scm","n"))
for (nm in names(rules)){
  k = rules[[nm]]; k[is.na(k)] = FALSE
  cat(sprintf("%-5s %8.3f %8.3f %6d\n", nm,
      mean(itr.table[k,"mse.pda"],na.rm=TRUE),
      mean(itr.table[k,"mse.scm"],na.rm=TRUE), sum(k)))
}
# the paper's Section 3 fix: the K best PDA matches, K = #{SCM passes}
k1 = rules$s1; k1[is.na(k1)] = FALSE; K = sum(k1)
ord = order(itr.table[,"pre.pda"]/abs(itr.table[,"miuy"]))[1:K]
cat(sprintf("%-5s %8.3f %8.3f %6d\n","eqK", mean(itr.table[ord,"mse.pda"],na.rm=TRUE),
            mean(itr.table[k1,"mse.scm"],na.rm=TRUE), K))
cat(sprintf("\npaper Table 2 (5,5) Design 6a: 1000-rule PDA 0.91 SCM 0.10 | MAE-rule PDA 0.77 SCM 0.10\n"))
