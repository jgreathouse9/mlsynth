## Deterministic SMC oracle: build the Basque matching matrix via Synth::dataprep,
## run SMCV with fixed V=1 (no optimx => deterministic), dump inputs & outputs.
suppressMessages({library(MASS); library(quadprog); library(Synth)})
data("basque")

dataprep.out <- dataprep(
  foo = basque,
  predictors = c("school.illit","school.prim","school.med","school.high","school.post.high","invest"),
  predictors.op = "mean", time.predictors.prior = 1964:1969,
  special.predictors = list(
    list("gdpcap",1961:1969,"mean"),
    list("sec.agriculture",seq(1961,1969,1),"mean"),
    list("sec.energy",seq(1961,1969,1),"mean"),
    list("sec.industry",seq(1961,1969,1),"mean"),
    list("sec.construction",seq(1961,1969,1),"mean"),
    list("sec.services.venta",seq(1961,1969,1),"mean"),
    list("sec.services.nonventa",seq(1961,1969,1),"mean"),
    list("popdens",seq(1961,1969,1),"mean")),
  dependent="gdpcap", unit.variable="regionno", unit.names.variable="regionname",
  time.variable="year", treatment.identifier=17, controls.identifier=c(2:16,18),
  time.optimize.ssr=1960:1969, time.plot=1955:1997)

YT<-cbind(dataprep.out$Y1plot,dataprep.out$Y0plot);
ZT0<-cbind(dataprep.out$Z1,dataprep.out$Z0);
XT0<-cbind(dataprep.out$X1,dataprep.out$X0);
Y0<-as.matrix(rbind(XT0,ZT0)); Y0<-matrix(Y0,dim(Y0)); YT<-matrix(YT,dim(YT));
b<-1;
XT00<-XT0; kk<-dim(ZT0)[1]; zsd<-c();
for(i in 1:kk){zsd[i]<-sd(ZT0[i,]);}
for(i in 1:dim(XT0)[1]){XT00[i,]<-XT0[i,]/sd(XT0[i,])*mean(zsd);}
Y00<-as.matrix(rbind(XT00,ZT0));
X<-Y00[,-b]; y<-Y00[,b];

## --- SMCV core (verbatim from reference), fixed V ---
SMCV<-function(X,y,V){
  p<-dim(X)[2]; n<-length(y);
  y0<-V^0.5*y; X0<-diag(V^0.5)%*%X;
  y<-y0-mean(y0); X<-X0-rep(1,n)%*%t(rep(1/n,n))%*%X0;
  e_Mdr<-matrix(NA,n,p); d_Mdr<-c();
  sigmahat<-sum((y-X%*%solve(t(X)%*%X)%*%t(X)%*%y)^2)/(n-p);
  BETA<-c();
  for(i in 1:p){
    Xd0<-X[,i];
    BETA[i]<-ginv(t(Xd0)%*%Xd0)%*%t(Xd0)%*%y;
    H0<-(Xd0)%*%solve(t(Xd0)%*%(Xd0))%*%t(Xd0);
    e_Mdr[,i]<-H0%*%y;
    d_Mdr[i]<-t(H0%*%y)%*%y - sigmahat*sum(diag(H0));
  }
  S_Mdr<-t(e_Mdr)%*%e_Mdr+0.001*diag(rep(1,p));
  Amat<-t(rbind(diag(-1,p),diag(1,p))); bvec<-c(rep(-1,p),rep(0,p));
  w<-solve.QP(S_Mdr,d_Mdr,Amat,bvec,meq=0)$solution;
  r<-as.numeric(mean(y0)-mean(X0%*%(BETA*w)));
  list(weights=w,bias=r,beta=BETA,sigmahat=sigmahat)
}
V<-rep(1,length(y));
res<-SMCV(X,y,V);
XT<-YT[,-b]; muT<-YT[,b];
mu_SM<-res$bias+XT%*%(res$beta*res$weights);

## --- reference outputs (compare against mlsynth.utils.smc_helpers.smc_weights) ---
## Optionally dump inputs/outputs to CSV for a cell-by-cell Python diff:
##   Rscript benchmarks/R/smc_oracle.R <out_dir>
args<-commandArgs(trailingOnly=TRUE)
donor_ids<-c(2:16,18)
if(length(args)>=1){
  D<-args[1]
  write.csv(X, file.path(D,"X.csv"), row.names=FALSE)
  write.csv(data.frame(y=y), file.path(D,"y.csv"), row.names=FALSE)
  write.csv(YT, file.path(D,"YT.csv"), row.names=FALSE)
  write.csv(data.frame(donor=donor_ids, beta=res$beta, weight=res$weights,
                       combined=res$beta*res$weights),
            file.path(D,"oracle_weights.csv"), row.names=FALSE)
  write.csv(data.frame(bias=res$bias, sigmahat=res$sigmahat),
            file.path(D,"oracle_scalars.csv"), row.names=FALSE)
  write.csv(data.frame(year=1955:1997, muT=muT, mu_SM=mu_SM),
            file.path(D,"oracle_cf.csv"), row.names=FALSE)
}
cat("n=",length(y)," p=",dim(X)[2]," sigmahat=",res$sigmahat," bias=",res$bias,"\n")
cat("sum(combined)=",sum(res$beta*res$weights)," pre-1970 RMSE=",
    sqrt(mean((muT[1:15]-mu_SM[1:15])^2))," meanATTpost=",mean(muT[16:43]-mu_SM[16:43]),"\n")
