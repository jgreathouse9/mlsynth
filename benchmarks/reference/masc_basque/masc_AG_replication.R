NearestNeighbors_AG <- function(treated,
                             donors,
                             treated.covariates,
                             donors.covariates,
                             treatment,
                             tune.pars,
                             ...){
  if(is.null(donors.covariates)){
    dist <- apply(((treated[1:(treatment-1),]-donors[1:(treatment-1),])^2),MARGIN=2,
                  FUN=function(x) sum(x))
  }
  else{
    covariates <- rbind(treated.covariates[time < treatment,],
                        donors.covariates[time < treatment,])
    covariates<-covariates[,lapply(.SD,mean,na.rm=TRUE),by=unit,.SDcols=names(covariates)[!names(covariates)%in%c("unit","time")]]
    setorder(covariates,unit)
    covariates[,unit:=NULL]
    covariates<-t(covariates)
    #Key difference from the main file is here:
    #AG construct their estimator using EDUCATION SHARES, so they divide education groups
    #by population.
    lowest  <- which(rownames(covariates)=="school.illit")
    highest <- which(rownames(covariates)=="school.high")
    covariates[lowest:highest,] <-
      100 * scale(covariates[lowest:highest,],
                  center=FALSE,
                  scale=colSums(covariates[lowest:highest,])
      )


    dist <- apply(((covariates[,1]-covariates[,-1])^2),MARGIN=2,
                  FUN=function(x) sum(x*(tune.pars$matchVfun(treated.covariates=covariates[,1],
                                                             donors.covariates=covariates[,-1]))))
  }
  W <-rep(0,length(dist))
  W[which(dist%in%sort(dist)[1:tune.pars$m])]<-1/length(which(dist%in%sort(dist)[1:tune.pars$m]))
  return(pars=W)
}

sc_estimator_AG<-function(treated,
                          donors,
                       treated.covariates,
                       donors.covariates,
                       treatment,nogurobi = FALSE,
                       psdtol = 1e6, BarConvTol = 1e-8, BarIterLimit = 1e5, ...) {
  if(is.null(donors.covariates)!=is.null(treated.covariates)) stop("Must specify covariates for both treated and control units, or neither.")
  if(!is.null(donors.covariates)){
    covariates <- rbind(treated.covariates[time < treatment,],
                        donors.covariates[time < treatment,])
    covariates<-covariates[,lapply(.SD,mean,na.rm=TRUE),by=unit,.SDcols=names(covariates)[!names(covariates)%in%c("unit","time")]]
    setorder(covariates,unit)
    covariates[,unit:=NULL]
    covariates<-t(covariates)
    #Key difference from the main file is here:
    #AG construct their estimator using EDUCATION SHARES, so they divide education groups
    #by population.
    lowest  <- which(rownames(covariates)=="school.illit")
    highest <- which(rownames(covariates)=="school.high")
    covariates[lowest:highest,] <-
      100 * scale(covariates[lowest:highest,],
                  center=FALSE,
                  scale=colSums(covariates[lowest:highest,])
      )



    params<-synth(Z1=as.matrix(treated[1:(treatment-1),]),
                  Z0=donors[1:(treatment-1),],
                  X1=as.matrix(covariates[,1]),
                  X0=covariates[,-1],
                  ...)
  }
  else{

    obj.synthetic <-
      function(donors, treated, treatment) {
        return(-2 * t(as.matrix(treated[1:(treatment-1),])) %*% donors[1:(treatment-1),])
      }
    objcon.synthetic <-
      function(donors, treated, treatment) {
        return(t(as.matrix(treated[1:(treatment-1),])) %*% treated[1:(treatment-1),])
      }
    Q.synthetic <-
      function(donors, treated, treatment) {
        return(t(donors[1:(treatment-1),]) %*% donors[1:(treatment-1),])
      }


    params <- list()

    modelreg <- list(
      A = matrix(
        rep(1, length(donors[1, ]))
        ,
        nrow = 1,
        ncol = ncol(donors),
        byrow = TRUE
      ),
      sense = c('='),
      rhs = c(1),
      lb = rep(0, length(donors[1, ])),
      ub = rep(1, length(donors[1, ])),
      obj = obj.synthetic(
        donors = donors,
        treated = treated,
        treatment = treatment),
      objcon = objcon.synthetic(
        donors = donors,
        treated = treated,
        treatment = treatment),
      Q = Q.synthetic(
        donors = donors,
        treated = treated,
        treatment = treatment)
    )



    continue <- 0

    if(nogurobi==FALSE){
      tparams <-
        gurobi::gurobi(
          modelreg,
          params = list(
            OutputFlag = 0,
            PSDTol = psdtol,
            BarConvTol = BarConvTol,
            BarIterLimit = BarIterLimit
          )
        )

      tparams <- plyr::rename(tparams, c('x' = 'pars'))
      if (tparams$status != 'OPTIMAL') {
        print('WARNING: FAILED TO SOLVE PROBLEM')
        continue
      }
      params$solution.w <- tparams$pars
      params$loss.w <- tparams$objval
    }
    else {
      tparams <- LowRankQP::LowRankQP(Vmat=2*modelreg$Q,
                                      dvec=modelreg$obj,
                                      Amat=modelreg$A,
                                      bvec=modelreg$rhs,
                                      uvec=modelreg$ub,
                                      method = 'LU')
      params$solution.w <- as.vector(tparams$alpha)
      params$loss.w<-mean((treated-donors%*%params$weights.sc)^2)
    }
  }
  return(params)
}


