##################################
#OPTIMIZATION PROBLEM
#################################

#' Nearest Neighbor selection on covariates or outcome paths
#' Defaults to unweighted distance on outcome paths if covariates are not specified,
#' or standard deviations with respect to covariates if covariates are specified.
NearestNeighbors <- function(treated,
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
    treated.covariates<-as.data.table(treated.covariates)
    donors.covariates<-as.data.table(donors.covariates)
    donors.covariates[,unit:=as.numeric(as.factor(unit))]
    treated.covariates[,unit:=0]

    covariates <- rbind(treated.covariates[time < treatment,],
                        donors.covariates[time < treatment,])
    covariates <- covariates[,lapply(.SD,mean,na.rm=TRUE),by=unit,.SDcols=names(covariates)[!names(covariates)%in%c("unit","time")]]
    setorder(covariates,unit)
    covariates[,unit:=NULL]
    covariates <- t(covariates)

  dist <- apply(((covariates[,1]-covariates[,-1])^2),MARGIN=2,
                FUN=function(x) sum(x*(tune.pars$matchVfun(treated.covariates=covariates[,1],
                                                      donors.covariates=covariates[,-1]))))
  }
  W <-rep(0,length(dist))
  W[which(dist%in%sort(dist)[1:tune.pars$m])]<-1/length(which(dist%in%sort(dist)[1:tune.pars$m]))
  return(pars=W)
}

#' Function determining weights on covariates by their standard deviation
Cov.Vars<-function(treated.covariates,
                   donors.covariates){
  output<- 1/apply(cbind(treated.covariates,donors.covariates), 1, var)
  return(output)
}

#' solving for standard synthetic control weights
#'
#' Solves for synthetic control weights up to the designated treatment period,
#' using only information in the outcome paths (no other covariates).
#'
#' @param donors See \link{masc}.
#' @param treated: See \link{masc}.
#' @param covariates.donors: See \link{masc}.
#' @param treated.donors: See \link{masc}.
#' @param treatment: An integer. The period T' in which forecasting begins (either the true treatment
#' period or the first period after a cross-validation fold).
#'
#' @param nogurobi A logical value. If true, uses \link[LowRankQP]{LowRankQP} to solve the synthetic control estimator,
#' rather than \code{gurobi}.

#' @return A list. The named component \code{weights.sc} contains the vector of synthetic control weights.
#'  Weights are ordered in the same manner as the columns in \code{Z0}.
#'  The \code{objval.sc} component contains the objective value (pre-period fit) of the synthetic control.
#' @family masc functions
#' @references Kellogg, M., M. Mogstad, G. Pouliot, and A. Torgovitsky. Combining Matching and Synthetic Control to Trade
#'  off Biases from Extrapolation and Interpolation. Working Paper, 2019.
#' @export
sc_estimator<-function(treated,
                       donors,
                       treated.covariates,
                       donors.covariates,
                       treatment,nogurobi = FALSE,
                       psdtol = 1e6, BarConvTol = 1e-8, BarIterLimit = 1e5,
                       sigf.ipop=5, margin.ipop=5e-04,
                       ...) {
  if(is.null(donors.covariates)!=is.null(treated.covariates)) stop("Must specify covariates for both treated and control units, or neither.")
  if(!is.null(donors.covariates)){
    treated.covariates<-as.data.table(treated.covariates)
    donors.covariates<-as.data.table(donors.covariates)
      covariates <- rbind(treated.covariates[time < treatment,],
                          donors.covariates[time < treatment,])
      covariates <- covariates[,lapply(.SD,mean,na.rm=TRUE),by=unit,.SDcols=names(covariates)[!names(covariates)%in%c("unit","time")]]
      setorder(covariates,unit)
      covariates[,unit:=NULL]
      covariates <- t(covariates)



  params<-synth(Z1=as.matrix(treated[1:(treatment-1),]),
                Z0=donors[1:(treatment-1),],
                X1=as.matrix(covariates[,1]),
                X0=covariates[,-1],
                sigf.ipop=sigf.ipop, margin.ipop=margin.ipop,
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



#' solving for matching and synthetic control weights
#'
#' Solves separately for synthetic control weights and matching weights for a given
#' matching estimator, using data up to the treatment period.
#'
#' @param donors A \eqn{TxN} matrix of outcome paths for untreated units, each column being a control unit.
#' @param treated: A \eqn{Tx1} matrix of outcomes for the treated unit.
#' @param treatment: An integer. The period T' in which forecasting begins (either the true treatment
#' period or the first period after a cross-validation fold).
#'
#'@param sc_est A \code{function} which constructs weights associated with a synthetic control-type estimator. See
#'\link{sc_estimator} for input and output if you'd prefer to substitute your own estimator.
#'
#' @param tune_pars A \code{list} containing one element:
#' \describe{\item{m:}{an integer identifying the candidate matching (nearest neighbor) estimator.}}
#'
#' @return A list containing the weights associated with the two estimators. Weights are ordered
#' in the same manner as the columns in \code{donors}. The \code{weights.sc} and \code{weights.match}
#' named components contain the vector of synthetic control and nearest neighbor weights respectively.
#'  The \code{objval.sc} component contains the objective value (pre-period fit) of the synthetic control.
#' @family masc functions
#' @references Kellogg, M., M. Mogstad, G. Pouliot, and A. Torgovitsky. Combining Matching and Synthetic Control to Trade
#'  off Biases from Extrapolation and Interpolation. Working Paper, 2021
#' @export
solve_masc <-
  function(treated,
           donors,
           treated.covariates,
           donors.covariates,
           treatment,
           sc_est,
           match_est,
           tune.pars,
           ...) {

    if(is.null(donors.covariates)!=is.null(treated.covariates)) stop("Must specify covariates for both treated and control units, or neither.")

    SC<-sc_est(treated = treated,
                   donors=donors,
                   treated.covariates=treated.covariates,
                   donors.covariates=donors.covariates,
                   treatment=treatment,
                ...)

    Match<-match_est(treated = treated,
                     donors = donors,
                     treated.covariates = treated.covariates,
                     donors.covariates = donors.covariates,
                     treatment = treatment,
                     tune.pars=tune.pars,
                     ...)

    output<-list(weights.sc=as.vector(SC$solution.w), weights.match = as.vector(Match))
    output$solution.v = SC$solution.v
    output$m = tune.pars$m
    output$loss.v<-SC$loss.v
    output$loss.w<-SC$loss.w
    return(output)
  }


