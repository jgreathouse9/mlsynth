# run from this directory
source("function_SCM-CS_v07.R")  # see README: not vendored
Ymat <- as.matrix(read.csv("Ymat.csv", header=FALSE))
weightsmat <- as.matrix(read.csv("weightsmat.csv", header=FALSE))
N <- ncol(Ymat); T0 <- 19; treated <- 3; precision <- 30; significance <- 4/39
res <- data.frame()
# uniform assignment, both effect classes
for (type in c("constant","linear")) {
  b <- SCM.CS(Ymat, weightsmat, treated, T0, 0, matrix(0,1,N), precision, type, significance, FALSE)
  u<-as.vector(b$u); l<-as.vector(b$l); n<-length(u)
  d <- if (type=="linear") (n-T0) else 1
  res <- rbind(res, data.frame(type=type, phi=0, vlab="zero",
                               lower=l[n]/d, upper=u[n]/d))
}
# sensitivity: tilt assignment toward the treated unit, and away from it
v_tr <- matrix(0,1,N); v_tr[1,treated] <- 1
for (ph in c(0.5, 1, 2)) {
  for (lab in c("treated","donors")) {
    vv <- if (lab=="treated") v_tr else (1-v_tr)
    b <- try(SCM.CS(Ymat, weightsmat, treated, T0, ph, vv, precision, "linear", significance, FALSE), silent=TRUE)
    if (inherits(b,"try-error")) { res <- rbind(res, data.frame(type="linear", phi=ph, vlab=lab, lower=NA, upper=NA)); next }
    u<-as.vector(b$u); l<-as.vector(b$l); n<-length(u); d<-n-T0
    res <- rbind(res, data.frame(type="linear", phi=ph, vlab=lab, lower=l[n]/d, upper=u[n]/d))
  }
}
write.csv(res, "ref_full.csv", row.names=FALSE)
print(format(res, digits=16))
