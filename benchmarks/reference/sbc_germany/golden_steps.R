# Golden per-step values for the SBC unit tests, captured from the authors' own
# functions (Shi-Xi-Xie Germany.R: lsq detrending, trend_predict forecast, and
# Synth::synth) run on the authoritative repgermany.dta. Emits a parseable block
# the Python tests pin mlsynth's per-step output against. The weight step also
# records the authors' Synth result + its cyclical SSE so the tests can show
# mlsynth attains a lower (better) objective.
suppressMessages({library(foreign); library(Synth); library(kernlab)})
d <- read.dta("basedata/repgermany.dta")
countries <- unique(d$country); years <- sort(unique(d$year))
M <- matrix(NA_real_, length(years), length(countries), dimnames=list(years, countries))
M[cbind(match(d$year,years), match(d$country,countries))] <- d$gdp
treated<-"West Germany"; donors<-setdiff(countries,treated)
Germany <- data.frame(Date=years, Germany=M[,treated], M[,donors], check.names=FALSE)
h<-4;p<-2;Fh<-h; T0<-which(Germany$Date==1990); predata<-Germany[1:T0,]
lsq<-function(z){tt=length(z);y=z[(h+p):tt];X=embed(z[1:(tt-h)],p);y<-ifelse(is.na(y),0,y);X<-ifelse(is.na(X),0,X);lm(y~.,data=data.frame(cbind(y,X)))}
exr<-function(L) do.call(cbind,lapply(L,residuals))
tpred<-function(z,b){tt=length(z);pr=numeric(h);for(i in 1:h)pr[i]=sum(c(1,z[(tt-h+i):(tt-h-p+i+1)])*b);pr}
ld<-apply(Germany[,3:ncol(Germany)],2,function(c) lsq(c)); dd<-exr(ld)[1:(T0+Fh-h-p+1),]
fit_g<-lsq(predata[,2]); dgp<-fit_g$residuals; bh<-fit_g$coefficients
tgpost<-tpred(predata[,2],bh)
X1<-as.matrix(dgp); X0<-as.matrix(dd[1:length(dgp),])
s<-synth(X1=X1,X0=X0,Z0=X0,Z1=X1,custom.v=rep(1,(T0-h-p+1)),optimxmethod=c("Nelder-Mead","BFGS"),genoud=FALSE,quadopt="ipop",Margin.ipop=5e-4,Sigf.ipop=5,Bound.ipop=10,verbose=FALSE)
w<-s$solution.w[,1]; sse<-sum((dgp - X0%*%w)^2)
arr<-function(v) paste(sprintf("%.8f",v),collapse=",")
cat("== REFERENCE VALUES ==\n")
cat("treated_trend_coef\t",arr(bh),"\n",sep="")
cat("treated_cycle_pre\t",arr(dgp),"\n",sep="")
cat("trend_forecast\t",arr(tgpost),"\n",sep="")
for(nm in c("Netherlands","Greece","Italy")) cat("donor_cycle_full:",nm,"\t",arr(dd[,nm]),"\n",sep="")
cat("synth_loose_sse\t",sprintf("%.4f",sse),"\n",sep="")
for(nm in names(w)[w>1e-4]) cat("synth_loose_weight:",nm,"\t",sprintf("%.6f",w[nm]),"\n",sep="")
cat("== END ==\n")
