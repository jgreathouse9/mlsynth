#' Report cross-validation error and Prediction Error, Moving between Matching and Synthetic Controls
#'
#'Reports treatment effect and cross-validation errors for estimators
#' of the form of the matching and synthetic control (masc) estimator of Kellogg, Mogstad,
#'  Pouliot, and Torgovitsky (2019). For a set of masc-type estimators defined by a synthetic control
#'  estimator, a matching estimator (m) and a weight (phi), this function returns output associated with
#'  the masc estimator constructed by placing a weight of phi on the matching estimator and (1-phi) on
#'  the synthetic control estimator.
#'
#'
#' @param treated A \eqn{Tx1} matrix of outcomes for the treated unit.
#'
#' @param donors A \eqn{TxN} matrix of outcome paths for untreated units, each column being a control unit.
#'
#' @param treatment An integer. The period T' in which forecasting begins.
#'  If \code{NULL} or T'>T, then we assume all data is pre-treatment.
#'
#'@param sc_est A \code{function} which constructs weights associated with a synthetic control-type estimator. See
#'\link{sc_estimator} for input and output if you'd prefer to substitute your own estimator.
#'
#' @param tune_pars A \code{list} containing 3 elements. You must specify the first, and you may specify
#' only one of the last two elements. The last two elements describe the folds we include in the cross-validation procedure.
#' Each fold \code{f} is denoted by the last period it uses for estimation. That is, fold \code{f} will fit estimators using
#' data from period 1 through period \code{f}, and forecast into period \code{f+1}. \describe{
#' \item{m:}{ a vector of integers. Denotes the set of nearest neighbor estimators from which we are allowed to pick.
#' E.g., \code{tune_pars_list$m=c(1,3,5)} would allow us to pick from 1-NN, 3-NN, or 5-NN.
#' Alternatively, \code{tune_pars_list$m} permits a logical vector. In this case, e.g., \code{tune_pars_list$m=c(FALSE,TRUE,TRUE)}
#' would allow us to pick from 2-NN or 3-NN.
#' If \code{NULL}, we default to allowing all possible nearest neighbor estimators.}
#' \item{min_preperiods:}{an integer. The smallest number of estimation periods allowed in a fold used for cross-validation.
#' We use all folds from fold \code{min_preperiods} up to the latest possible fold \code{treatment-2}.}
#' \item{set_f:}{a \code{list} containing a single element, a vector of integers. Identifies the set of folds used
#'  for cross-validation. As above, each integer identifies a fold by the last time period it uses in estimation.
#'  E.g., set_f=c(7,8,9) would implement cross-validation using fold 7, fold 8, and fold 9.}
#' }
#' If neither \code{min_preperiods} nor \code{set_f} are specified, then we set \code{min_preperiods} to \code{ceiling(treatment/2)}.
#' In other words, we pick the first cross-validation fold so that it is estimated on the first half of the pre-period data.
#'
#'
#' @param nogurobi A logical value. If true, uses \link[LowRankQP]{LowRankQP} to solve the synthetic control estimator,
#' rather than \code{gurobi}.
#'
#' @param phivals A vector of real values between 0 and 1. Indexes a weighted
#' average of the synthetic control estimator with a matching estimator, where \code{phival}
#'indicates the weight on the matching estimator (\code{1-phival} being the
#' weight on synthetic controls).
#'
#' @param treatinterval A vector. Indicates the post-treatment periods used when reporting average
#' treatment effects in the column \code{pred} of the output. E.g., \code{treatinterval=1:5} causes the \code{pred}
#' column to report the average treatment effect over the first 5 treatment periods. If \code{NULL}, we default
#' to averaging over all post-treatment periods.
#'
#' @return Returns a \code{data.frame} with each row defined by a value of \code{m} and \code{phi} taken respectively
#' from \code{tune_pars$m} and \code{tune_pars$phivals}. The columns
#' \code{cv.error} and \code{pred} return respectively the cross-validation error and
#' a measure of prediction error (AKA treatment effect) associated with the masc estimator defined by \code{m} and \code{phi}.
#'
#' @family masc functions
#' @references Kellogg, M., M. Mogstad, G. Pouliot, and A. Torgovitsky. Combining Matching and Synthetic Control to Trade
#'  off Biases from Extrapolation and Interpolation. Working Paper, 2019.
#'
#
#'@examples
#'  ##Example: Terrorism in the Basque Region, from
#' ##Abadie and Gardeazabal (2003).
#'
#' #First, load the Synth package, which includes the dataset:
#'
#' if (requireNamespace("Synth",quietly=TRUE) & requireNamespace("data.table",quietly=TRUE)){
#' library(Synth)
#' library(data.table)
#' data(basque)
#' basque<-as.data.table(basque)
#' basque <- basque[regionno!=1,]
#' basque[,regionname:= gsub(" (.*)","",regionname)]
#' #Grabbing region names:
#' names<- c(unique(basque[regionno==17,regionname]),unique(basque[regionno!=17,regionname]))
#' basque <- cbind(basque[regionno==17,gdpcap],
#'                                             t(reshape(basque[regionno!=17,.(regionno,year,gdpcap)],
#'                                              idvar='regionno', timevar='year',direction='wide')[,-"regionno",with=FALSE]))
#'
#'
#' result <- masc(treated=basque[,1], donors=basque[,-1],treatment=16, tune_pars_list=list(m=1:10,
#'                                              min_preperiods=8))
#' names(result$weights)<-names[-1]
#'
#' #weights on control units:
#' print(round(result$weights,3))
#'
#' #Treatment effects of terrorism on GDP per capita
#' #in thousands of 1986 US dollars, over 1970-1975:
#' #(first 6 years of treatment)
#' print(result$pred.error[1:6,])
#'
#' #Selected tuning parameters?
#' print(paste0("Selected matching estimator: ",result$m_hat))
#' print(paste0("Selected weight on matching: ",result$phi_hat))
#'
#' #Now, examine the shape of A) the CV error (mean square prediciton error in pre-period) and
#' # B) average prediction error (AKA treatment effect) over the first 5 treatment years,
#' #both over values of phi, fixing the matching estimator (moving from matching to synthetic controls)
#' phis<-seq(0,1,length.out=100)
#'phi_table<-masc_by_phi(treated=basque[,1], donors=basque[,-1],treatment=16, tune_pars=list(m=result$m_hat,
#'                                              min_preperiods=8,phis=phis))
#'#Printing CV error and prediction error over values of phi. CV error is clearly lowest at intermediary values of phi,
#'#suggesting an estimator between matching and synthetic controls does best at forecasting. The average medium-run treatment
#'#effect is monotonically increasing as we move away from synthetic control and toward matching.
#'print(phi_table)
#'}
#'
#'
#' @export
#' @import data.table
#' @import plyr
#' @import Synth
masc_by_phi<-function(treated,
                      donors,
                      treated.covariates=NULL,
                      donors.covariates=NULL,
                      treatment=NULL,
                      sc_est=sc_estimator,
                      match_est = NearestNeighbors,
                      tune_pars=list(min_preperiods=NULL,set_f=NULL,
                                     m=NULL,phis=seq(from=0,to=1,length.out=100)),
                      cv_pars = list(forecast.minlength = 1, forecast.maxlength = 1),
                      treatinterval=NULL,
                      ...){

  phi_table<-NULL
  if(is.logical(tune_pars$m)) tune_pars$m<-which(tune_pars$m)
  if(is.null(tune_pars$m))tune_pars$m<-1:ncol(donors)
for(mpos in 1:length(tune_pars$m)){
  mval <- tune_pars$m[mpos]
  temp_table<-data.frame(phi=tune_pars$phis)
  temp_table$m<-mval
  temp_table$cv<-NA
  temp_table$pred<-NA
  if(all(is.null(treatinterval),!is.null(treatment),treatment<=length(treated))) treatinterval<-1:(length(treated)-treatment+1)
  if(any(treatment>length(treated),is.null(treatment))) treatinterval<-NA
  for(ph in 1:length(tune_pars$phis)){
  tunevals<-tune_pars
  tunevals$m<-mval
phi_result<-cv_masc(treated=treated, donors=donors,
                    treated.covariates = treated.covariates,
                    donors.covariates = donors.covariates,
                    treatment=treatment,sc_est=sc_est,tune_pars=tunevals,  cv_pars=cv_pars,
        phival=tune_pars$phis[ph])
        temp_table[ph,]$cv<-phi_result$cv.error
        if(!any(is.na(treatinterval))) temp_table[ph,"pred"]<-mean(phi_result$pred.error[treatinterval])
        else temp_table[ph,"pred"]<-NA
}
  phi_table<-rbind(phi_table,temp_table)
}
return(phi_table)
}

#' Solve for optimal model averaging parameter and associated weights and errors conditional on matching estimator.
#'
#' Implements the matching and synthetic control (masc) estimator of Kellogg, Mogstad,
#'  Pouliot, and Torgovitsky (2019), \emph{conditional on a given matching estimator} characterized by m.
#'  \link{masc} loops over evaluations of this function for each candidate matching estimator,
#'  and selects the one which minimizes cross-validation error.
#'
#'
#' @param tune_pars A \code{list} containing 5 elements. You must specify the first, and you may specify
#' only one of \code{min_preperiods} and \code{set_f}. These elements describe the folds we include in the cross-validation procedure.
#' Each fold \code{f} is denoted by the last period it uses for estimation. That is, fold \code{f} will fit estimators using
#' data from period 1 through period \code{f}, and forecast into period \code{f+1}. \describe{
#' \item{m:}{ an integer representing the  nearest neighbor estimator used.}
#' \item{min_preperiods:}{an integer. The smallest number of estimation periods allowed in a fold used for cross-validation.
#' We use all folds from fold \code{min_preperiods} up to the latest possible fold \code{treatment-2}.}
#' \item{set_f:}{a \code{list} containing a single element, a vector of integers. Identifies the set of folds used
#'  for cross-validation. As above, each integer identifies a fold by the last time period it uses in estimation.
#'  E.g., set_f=c(7,8,9) would implement cross-validation using fold 7, fold 8, and fold 9.}
#'  \item{weights_f:}{a \code{vector} of length \code{length(set_f)} or \code{length(min_preperiods:(treatment-2))},
#' containing weakly positive relative weight values for each of the cross-validation folds. The elements of \code{weights_f}
#' are normalized to sum to 1}
#' \item{matchVfun:}{a \code{function} that governs how unit characteristics are weighted together for matching.
#' If the estimator is purely outcome-based, then the default behavior is raw matching on the outcome paths.
#' If the estimator uses covariates, then the default behavior is to weigh outcomes by their standard deviations
#' across units.}
#' }
#' #' If neither \code{min_preperiods} nor \code{set_f} are specified, then we set \code{min_preperiods} to \code{ceiling(treatment/2)}.
#' In other words, we pick the first cross-validation fold so that it is estimated on the first half of the pre-period data.
#' By default, all folds are equally weighted.
#'
#' @param cv_pars A \code{list} containing 2 integer elements, \code{forecast.minlength} and \code{forecast.maxlength}. Cross-validation fold \code{f} will forecast into periods
#' \code{f+forecast.minlength}  and up to period \code{f+forecast.maxlength} or the treatment period (whichever comes first).
#' If \code{f+forecast.minlength}  lies in the treatment  interval for one of the folds \code{f} given by the user,
#' then \code{masc} returns an error.
#'

#' @param phival A real value between 0 and 1. If specified, hard-codes the masc estimator to take the specified weighted
#' average of matching and synthetic controls, where \code{phival} indicates the weight on matching (\code{1-phival} being the
#' weight on synthetic controls).
#'
#' @return returns a list containing five objects:
#' \describe{
#' \item{phi_hat:}{selected value for the model averaging parameter (1 is pure matching, 0 pure synthetic control).}
#' \item{m_hat:}{selected matching estimator (number of nearest neighbor).}
#' \item{weights:}{The vector length N containing weights placed on each control unit.}
#' \item{pred.error:}{The vector of treatment effects implied by the masc counterfactual, for periods T' to T.}
#' \item{cv.error:}{The average (weighted by \code{weights_f}) of the cross-validation errors generated by each fold.}
#' \item{cv.error.byfold:}{The cross-validation error generated by each fold.}
#' }
#'
#' @family masc functions
#' @references Kellogg, M., M. Mogstad, G. Pouliot, and A. Torgovitsky. Combining Matching and Synthetic Control to Trade
#'  off Biases from Extrapolation and Interpolation. Working Paper, 2019.
#' @export
#' @import data.table
#' @import plyr
#' @import Synth
cv_masc <-
  function(treated,
           donors,
           treated.covariates = NULL,
           donors.covariates = NULL,
           treatment=NULL,
           sc_est=sc_estimator,
           match_est = NearestNeighbors,
           tune_pars=list(min_preperiods=NULL,set_f=NULL,m=NULL, weights_f = NULL, matchVfun = NULL),
           cv_pars = list(forecast.minlength = 1, forecast.maxlength = 1),
            phival=NULL,
           ...) {
    treated<-as.matrix(treated)

    if(is.null(tune_pars$min_preperiods)) tune_pars$min_preperiods<-NA
    if(is.null(tune_pars$set_f)) tune_pars$set_f<-NA
    if(is.null(tune_pars$m)) stop("Must specify a matching estimator in tune_pars")
    if(is.null(tune_pars$matchVfun)) tune_pars$matchVfun <- Cov.Vars

    if(is.null(treated.covariates) != is.null(donors.covariates)) stop("If specifying covariates for either the treated unit or donor units, you must specify covariates for both of them.")

    min_preperiods = tune_pars$min_preperiods
    set_f = tune_pars$set_f
    if (!is.na(min_preperiods) & !any(is.na(set_f)))
      stop("One of min_preperiods and set_f must be non-missing.")

    if (is.na(min_preperiods) & any(is.na(set_f))) min_preperiods<-ceiling(treatment/2)
    if (!is.na(min_preperiods)) set_f <- min_preperiods:(treatment - 2)
    if(is.null(tune_pars$weights_f)) weights_f<-rep(1,length(set_f))
    else weights_f <- tune_pars$weights_f

    forecast.minlength<-cv_pars$forecast.minlength
    forecast.maxlength<-cv_pars$forecast.maxlength

  if(is.null(treatment)) treatment<-nrow(treated)+1
    treat <- treatment
    weights_f <- weights_f[which(treatment - set_f - 1 >= 1)]
    set_f <- set_f[which(treatment - set_f - 1 >= 1)]
    weights_f <- weights_f/sum(weights_f)
    ####################################
    finalweights<-solve_masc(treatment=treat,
                                   donors=donors,
                                   treated=as.matrix(treated),
                             donors.covariates = donors.covariates,
                             treated.covariates = treated.covariates,
                            sc_est=sc_est,
                            match_est = match_est,
                          tune.pars=list(
                           m=tune_pars$m,
                           matchVfun=tune_pars$matchVfun
                         ),
                         ...)

    Y_sc<-NULL
    Y_match<-NULL
    foldweights<-list()
    Y_treated<-NULL
    objweight<-NULL
    #SOLVE MATCHING AND SC ESTIMATORS FOR EACH FOLD:
    position<-1
    for(k in set_f){
      treatinterval<-(k+forecast.minlength):min(k+forecast.maxlength,treatment-1)
      treat<-k+1
      foldweights[[position]]<-solve_masc(treatment=treat,
                                         donors=donors,
                                         treated=as.matrix(treated),
                                         donors.covariates = donors.covariates,
                                         treated.covariates = treated.covariates,
                                         sc_est=sc_est,
                                         match_est = match_est,
                                         tune.pars=list(
                                           m=tune_pars$m,
                                           matchVfun = tune_pars$matchVfun
                                         ),
                                         ...)
      Y_sc<-   c(Y_sc,donors[treatinterval,]%*%foldweights[[position]]$weights.sc)

      Y_match<- c(Y_match,donors[treatinterval,]%*%foldweights[[position]]$weights.match)
      Y_treated <- c(Y_treated, treated[treatinterval,])
      objweight <- c(objweight, rep(weights_f[position]/length(treatinterval), length(treatinterval)))


      position=position+1
    }
    #ANALYTIC EXPRESSION FOR PHI:
    if(is.null(phival)){
    phi<-(objweight*(Y_treated-Y_sc))%*%(Y_match-Y_sc)/((objweight*(Y_match-Y_sc))%*%(Y_match-Y_sc))
    phi<-max(0,phi)
    phi<-min(1,phi)
    }
    else phi = phival
    #IMPLIED COUNTERFACTUAL AND ERRORS:
    cv.error<- sum(objweight*(Y_treated-phi*Y_match-(1-phi)*Y_sc)^2)
    #CV ERROR by FOLD:
    position<-1
    cv.error.byfold<-rep(0,length(set_f))
    for(k in set_f){
      treatinterval<-(k+forecast.minlength):min(k+forecast.maxlength,treatment-1)
      treat<-k+1
      Y_scfold<-as.vector(donors[treatinterval,]%*%foldweights[[position]]$weights.sc)
      Y_matchfold<-as.vector(donors[treatinterval,]%*%foldweights[[position]]$weights.match)
      objweight_fold<-rep(weights_f[position]/length(treatinterval), length(treatinterval))
      cv.error.byfold[position] <- mean((treated[treatinterval,]-phi*Y_matchfold-(1-phi)*Y_scfold)^2)
      position=position+1
    }

    weight<-phi*finalweights$weights.match+(1-phi)*finalweights$weights.sc

    output<-list(phi_hat=phi,m_hat=tune_pars$m,weights=weight,
                 solution.v = finalweights$solution.v,
                 loss.v = finalweights$loss.v,
                 loss.w = finalweights$loss.w)
    if(treatment <= nrow(donors)) output$pred.error<-treated[treatment:nrow(donors),] - donors[treatment:nrow(donors),]%*%weight
    else output$pred.error<-NA
    output$cv.error<-cv.error
    output$cv.error.byfold <- cv.error.byfold

    return(output)
  }


#' Cross-validated Estimation of the Matching and Synthetic Control Estimator.
#'
#' Implements the matching and synthetic control (masc) estimator of Kellogg, Mogstad,
#'  Pouliot, and Torgovitsky (2021).
#'
#'  The \code{masc} estimator takes a convex combination of a
#'  nearest neighbor estimator and a synthetic control estimator. That combination
#'  is parametrized by a model averaging parameter which takes a value of 1 when
#'  the estimator is equivalent to a pure matching estimator, and 0 when it is equivalent
#'  to a pure synthetic control estimator.
#'   This function selects the nearest neighbor estimator and model
#'  averaging parameter by a rolling-origin cross-validation procedure. Computationally,
#'  we minimize the cross-validation criterion in two steps. First, for each candidate
#'  nearest neighbor estimator, we solve for the model averaging parameters using an analytic
#'  expression (see Equation 15 of the working paper). Then, we select the candidate nearest
#'  neighbor estimator which produces the smallest cross-validation criterion value.
#'
#'  This implementation by default may use the \code{gurobi} interfaced with R to solve for the synthetic control estimator.
#'  Gurobi and its associated R package are available on the gurobi website:
#'  \url{https://cran.r-project.org/web/packages/prioritizr/vignettes/gurobi_installation.html}
#'
#'  It is recommended to use this implementation of the \code{masc} estimator. However, the function may
#'  alternatively use an implementation based on \link[LowRankQP]{LowRankQP} for convenience.
#'
#' @param treated A \eqn{Tx1} matrix of outcomes for the treated unit.
#'
#' @param donors A \eqn{TxN} matrix of outcome paths for untreated units, each column being a control unit.
#'
#' #' Note that  \code{donors} and \code{treated} should only contain
#' the time series values for a single outcome for each unit.
#'
#' @param treated.covariates An optional argument; A matrix consisting of time series data on covariates,
#' for the treated unit.
#'
#' @param donors.covariates An optional argument; A matrix consisting of time series data on covariates,
#' for the control units.
#'
#' If specified, both \code{treated.covariates} and \code{donors.covariates} should be formatted in the same way.
#' If using \code{K} covariates, each matrix should have \code{K+2} columns.
#' \code{K} of these columns represent the covariates, and unit-time observations compose the rows.
#' The matrices should have two self-explanatory named columns:
#' \describe{
#' \item{unit:}{ can be of any type, and may be omitted from \code{treated.covariates}.}
#' \item{time:}{ a numeric column, indicating when the covariates were realized
#' relative to the rows of \code{treated} and \code{donors}.
#' If \code{time = 2}, then the covariates in that row were realized at the same time
#' as outcomes of row 2 in \code{treated} and \code{donors} (or were
#' realized after row 1 but sometime before row 2).
#' For time-invariant pre-determined characteristics, you can set \code{time = 0}.
#' }
#' }
#'
#'
#'
#' The covariates are averaged over the pre-period, and the resulting averages are used to construct
#' the estimators
#' (during cross-validation, they will average over averages will be taken within each fold).
#'
#' @param treatment An integer. The period T' in which forecasting begins. If \code{NULL} or T'>T, then we
#'  assume all data is pre-treatment.
#'
#'
#'@param sc_est A \code{function} which constructs weights associated with a synthetic control-type estimator.
#'It must accept the arguments \code{treated}, \code{donors}, \code{treated.covariates}, \code{donors.covariates},
#'and \code{treatment} arguments as defined above.
#'Defaults to \link{sc_estimator}, which is a wrapper function that implements  \link[Synth]{synth}.
#'
#' @param match_est A \code{function} which constructs weights associated with a matching estimator.
#'It must accept the arguments \code{treated}, \code{donors}, \code{treated.covariates}, \code{donors.covariates},
#' and \code{treatment} arguments as defined above.
#' Additionally, it must accept a \code{tune_pars} argument, governing how tuning parameters
#' control the matching estimator.
#'Defaults to \link{NearestNeighbor}.
#'
#' @param tune_pars_list A \code{list} containing 5 elements.
#' You may specify
#' only one of \code{min_preperiods} and \code{set_f}. Those elements describe the folds we include in the cross-validation procedure.
#' Each fold \code{f} is indexed by the last period it uses for estimation.
#' That is, fold \code{f} will fit estimators using
#' data from period 1 through period \code{f}, and forecast into period \code{f+1}. \describe{
#' \item{m:}{ a vector of integers. Denotes the set of nearest neighbor estimators from which we are allowed to pick.
#' E.g., \code{tune_pars_list$m=c(1,3,5)} would allow us to pick from 1-NN, 3-NN, or 5-NN.
#' Alternatively, \code{tune_pars_list$m} permits a logical vector. In this case, e.g., \code{tune_pars_list$m=c(FALSE,TRUE,TRUE)}
#' would allow us to pick from 2-NN or 3-NN.
#' If \code{NULL}, we default to allowing all possible nearest neighbor estimators.}
#' \item{min_preperiods:}{an integer. The smallest number of estimation periods allowed in a fold used for cross-validation.
#' We use all folds from fold \code{min_preperiods} up to the latest possible fold \code{treatment-2}.}
#' \item{set_f:}{a vector of integers. Identifies the set of folds used
#'  for cross-validation. As above, each integer identifies a fold by the last time period it uses in estimation.
#'  E.g., set_f=c(7,8,9) would implement cross-validation using fold 7, fold 8, and fold 9.}
#'  \item{weights_f:}{a \code{vector} of length \code{length(set_f)} or \code{length(min_preperiods:(treatment-2))},
#' containing weakly positive relative weight values for each of the cross-validation folds. The elements of \code{weights_f}
#' are normalized to sum to 1}
#' \item{matchVfun:}{a \code{function} that governs how unit characteristics are weighted together for matching.
#' If the estimator is purely outcome-based, then the default behavior is raw matching on the outcome paths.
#' If the estimator uses covariates, then the default behavior is to weigh outcomes by their standard deviations
#' across units.}
#' }
#' If neither \code{min_preperiods} nor \code{set_f} are specified, then we set \code{min_preperiods} to \code{ceiling(treatment/2)}.
#' In other words, we pick the first cross-validation fold so that it is estimated on the first half of the pre-period data.
#' By default, all folds are equally weighted.

#'
#' @param cv_pars A \code{list} containing 2 integer elements, \code{forecast.minlength} and \code{forecast.maxlength}. Cross-validation fold \code{f} will forecast into periods
#' \code{f+forecast.minlength}  and up to period \code{f+forecast.maxlength} or the treatment period (whichever comes first).
#' If \code{f+forecast.minlength}  lies in the treatment  interval for one of the folds \code{f} given by the user,
#' then \code{masc} returns an error.
#'
#'
#' @param alloutput A logical value. If true, output includes a list \code{all.results} containing
#' full set of output associated with each candidate nearest neighbor estimator.
#'
#'
#' @param phival If specified, a numeric value describing how the matching and
#' SC estimator should be combined (without cross-validation). A value of 1 represents
#' using only the matching estimator, a value of 0 represents using only the SC estimator.
#'
#'@param ... Other arguments to pass to \code{sc_est} and \code{match_est}
#'
#' @return By default, returns a list containing five objects:
#' \describe{
#' \item{phi_hat:}{selected value for the model averaging parameter (1 is pure matching, 0 pure synthetic control).}
#' \item{m_hat:}{selected matching (nearest neighbor) estimator.}
#' \item{weights:}{The vector length N containing weights placed on each control unit.}
#' \item{pred.error:}{The vector of treatment effects implied by the masc counterfactual, for periods T' to T.}
#' \item{cv.error:}{The average (weighted by \code{weights_f}) of the cross-validation errors generated by each fold.}
#' \item{cv.error.byfold:}{The cross-validation error generated by each fold.}
#' }
#' Additionally, returns the following object if \code{alloutput=TRUE}:
#' \describe{
#' \item{all.results:}{A list containing the five above output components for each candidate matching estimator. Values for
#' each candidate are appended as columns.}
#' }
#'
#' @family masc functions
#' @references Kellogg, M., M. Mogstad, G. Pouliot, and A. Torgovitsky. Combining Matching and Synthetic Control to Trade
#'  off Biases from Extrapolation and Interpolation. Working Paper, 2019.
#'
#' @examples
#' ##First example: a small, arbitrary dataset.
#'
#' data3 <- data.frame(t = 1:10, treated = 1:10,
#' control1 = c(15,12,14,22,21,28,29,30,29,31),
#' control2 = c(10,5,10,12,16,20,22,21,23,21),
#' control3 = c(-7,-11,-7,-3,-4,-9,-7,-3,-7,-11),
#' control4 = c(-15,-13,-14,-16,-15,-12,-13,-13,-14,-14))
#'
#' #controls are above stored as differences from treated unit. Translating into levels:
#' data3$control1 <- data3$control1 + data3$treated
#' data3$control2 <- data3$control2 + data3$treated
#' data3$control3 <- data3$control3 + data3$treated
#' data3$control4 <- data3$control4 + data3$treated
#'
#'#defining treatment period:
#'treatperiod <- 6
#'
#'result<-masc(treated=data3$treated,donors=data3[,-c(1,2)],treatment=treatperiod,
#'             tune_pars_list=list(m=1:3,min_preperiods=3))
#'
#'#an equivalent specification:
#'result<-masc(treated=data3$treated,donors=data3[,-c(1,2)],treatment=treatperiod,
#'             tune_pars_list=list(m=1:3,set_f=3:4))
#'
#' #Weights selected, for controls 1 through 4 respectively:
#' print(round(result$weights,2))
#'
#' ##Second example: Terrorism in the Basque Region, from
#' ##Abadie and Gardeazabal (2003).
#' ##Fitting estimators purely on outcome paths.
#'
#' #First, load the Synth package, which includes the dataset:
#' if (requireNamespace("Synth",quietly=TRUE) & requireNamespace("data.table",quietly=TRUE)){
#' library(Synth)
#' library(data.table)
#' data(basque)
#' basque<-as.data.table(basque)
#' basque <- basque[regionno!=1,]
#' basque[,regionname:= gsub(" (.*)","",regionname)]
#' #Grabbing region names:
#' names<- c(unique(basque[regionno==17,regionname]),unique(basque[regionno!=17,regionname]))
#' basque <- cbind(basque[regionno==17,gdpcap],
#'                                             t(reshape(basque[regionno!=17,.(regionno,year,gdpcap)],
#'                                              idvar='regionno', timevar='year',direction='wide')[,-"regionno",with=FALSE]))
#'
#'
#' result <- masc(treated=basque[,1], donors=basque[,-1],treatment=16, tune_pars_list=list(m=1:10,
#'                                              min_preperiods=8))
#' names(result$weights)<-names[-1]
#'
#' #weights on control units:
#' print(round(result$weights,3))
#'
#' #Treatment effects of terrorism on GDP per capita
#' #in thousands of 1986 US dollars, over 1970-1975:
#' #(first 6 years of treatment)
#' print(result$pred.error[1:6,])
#'
#' #Selected tuning parameters?
#' print(paste0("Selected matching estimator: ",result$m_hat))
#' print(paste0("Selected weight on matching: ",result$phi_hat))
#'
#' #Now, examine the shape of A) the CV error (mean square prediciton error in pre-period) and
#' # B) average prediction error (AKA treatment effect) over the first 5 treatment years,
#' #both over values of phi, fixing the matching estimator (moving from matching to synthetic controls)
#' phis<-seq(0,1,length.out=100)
#'phi_table<-masc_by_phi(treated=basque[,1], donors=basque[,-1],treatment=16, tune_pars=list(m=result$m_hat,
#'                                              min_preperiods=8,phis=phis))
#'#Printing CV error and prediction error over values of phi. CV error is clearly lowest at intermediary values of phi,
#'#suggesting an estimator between matching and synthetic controls does best at forecasting. The average medium-run treatment
#'#effect is monotonically increasing as we move away from synthetic control and toward matching.
#'print(phi_table)
#'
#' ##Third example: Terrorism in the Basque Region, from
#' ##Abadie and Gardeazabal (2003).
#' ##Fitting estimators on outcome paths and covariates.
#' ##data setup:
#' data(basque)
#' basque<-as.data.table(basque)
#' basque<-basque[regionno!=1,]
#' basque[,regionname:= gsub(" (.*)","",regionname)]
#' basque[,school.high:=school.high+school.post.high]
#' basque[,school.post.high:=NULL]
#'
#' #Grabbing region names:
#' names<- c(unique(basque[regionno==17,regionname]),unique(basque[regionno!=17,regionname]))
#'
#' #Setting up outcomes and covariates:
#' outcomes <- cbind(basque[regionno==17,gdpcap],
#'                                             t(reshape(basque[regionno!=17,.(regionno,year,gdpcap)],
#'                                              idvar='regionno', timevar='year',direction='wide')[,-"regionno",with=FALSE]))
#' #Abadie and Gardeazabal do not use the first 5 years of data (1955-1959):
#' outcomes <- outcomes[-(1:5),]
#' covariates <- copy(basque)
#' covariates <- covariates[year>=1960,]
#' covariates[,time:=year-min(year)+1]
#' covariates[,year:=NULL]
#' covariates[,regionno:=NULL]
#' #treating population density as fixed, assigning a common value to all years
#' #for each region
#' covariates[,popdens:=mean(popdens,na.rm=TRUE),by=regionname]
#' for(u in 1:length(names)){
#' covariates[regionname==names[u],unit:=u]
#' }
#' covariates[,regionname:=NULL]
#' #Reordering covariates, to match Abadie and Gardeazabal
#' covariates <- covariates[,c(8:11,13,1:7,12,14,15),with=FALSE]
#'
#'#Solving for the MASC estimator. Note that the matching and SC functions used
#'#are NOT the default functions.
#'#This is because some transformations of the education-related covariates must
#'#be done to convert average levels over a given time period to shares.
#'
#'#NOTE: the paper checks m=1:10, but that can take a while to run.
#' result <- masc(treated=outcomes[,1], donors=outcomes[,-1],
#'                treated.covariates = covariates[unit==1,],
#'                donors.covariates = covariates[unit!=1,],
#'                treatment=11,
#'                tune_pars_list=list(m=1,
#'                                    min_preperiods=5))
#'}
#' @export
#' @import data.table
#' @import plyr
#' @import Synth
masc <- function(treated,
           donors,
           treated.covariates = NULL,
           donors.covariates = NULL,
           treatment=NULL,
           sc_est=sc_estimator,
           match_est = NearestNeighbors,
           tune_pars_list = list(),
           cv_pars = list(forecast.minlength = 1, forecast.maxlength = 1),
           alloutput = FALSE,
           phival = NULL,
           ...
           ) {

  if(is.list(tune_pars_list$set_f)) tune_pars_list$set_f <- unlist(tune_pars_list$set_f)

    treated<-as.matrix(treated)
    donors<-as.matrix(donors)
    if(is.null(treatment)) treatment<-nrow(treated)+1
    if ((!is.null(tune_pars_list$min_preperiods) &
         !any(is.null(tune_pars_list$set_f))))
      stop("You may specify precisely one of min_preperiods and set_f in tune_pars_list. The other must be empty (NULL)")

    if(is.null(tune_pars_list$min_preperiods) &
        any(is.null(tune_pars_list$set_f))) tune_pars_list$min_preperiods<-ceiling(treatment/2)

    if(is.null(tune_pars_list$min_preperiods)) tune_pars_list$min_preperiods<-NA

    if(is.null(tune_pars_list$m)) tune_pars_list$m<-1:ncol(donors)
    if(is.logical(tune_pars_list$m)) tune_pars_list$m<-which(tune_pars_list$m)

    if(is.null(tune_pars_list$weights_f) &
       ! is.null(tune_pars_list$set_f)) tune_pars_list$weights_f<-rep(1,length(tune_pars_list$set_f))

    if(is.null(tune_pars_list$weights_f) &
       ! is.na(tune_pars_list$min_preperiods)) tune_pars_list$weights_f<-rep(1,length(tune_pars_list$min_preperiods:(treatment-2)))

    if(any(tune_pars_list$weights_f <0)) stop("fold-specific weights must be weakly positive.")

    if(is.null(tune_pars_list$matchVfun)) tune_pars_list$matchVfun <- Cov.Vars



    tune_pars_joint <- list()
      position = 1

      for (d in 1:length(tune_pars_list$m)) {
              tune_pars_joint[[position]] <- list(
                m = tune_pars_list$m[d],
                min_preperiods = tune_pars_list$min_preperiods,
                set_f = tune_pars_list$set_f,
                weights_f = tune_pars_list$weights_f,
                matchVfun = tune_pars_list$matchVfun
              )
              position = position + 1

      }
      allresults <-
        lapply(tune_pars_joint, function(x)
        cv_masc(treated = treated,
                donors=donors,
                treated.covariates=treated.covariates,
                donors.covariates=donors.covariates,
                treatment=treatment,
                sc_est=sc_est,
                match_est = match_est,
                tune_pars = x,
                cv_pars=cv_pars,
                phival=phival,
          ...))
    #allresults is a list of unnamed things, each with named list (weights, pred error, fold stuff, tune pars, cv errors)
    output <-
      allresults[[which.min(lapply(allresults, function(x)
        x$cv.error))]]
    if(alloutput==TRUE){
    output$all.results <- list()

    output$all.results$phi_hat<-sapply(allresults, function(x)
      x$phi_hat)
    names(output$all.results$phi_hat)<-paste0(tune_pars_list$m,"-NN")

    output$all.results$weights<-sapply(allresults, function(x)
      x$weights)
    colnames(output$all.results$weights)<-paste0(tune_pars_list$m,"-NN")

    output$all.results$cv.error <-
      sapply(allresults, function(x)
        x$cv.error)
    names(output$all.results$cv.error)<-paste0(tune_pars_list$m,"-NN")

      output$all.results$pred.error <-
        sapply(allresults, function(x)
          x$pred.error)
      colnames(output$all.results$pred.error)<-paste0(tune_pars_list$m,"-NN")

      output$all.results$SC.loss.v <-
        sapply(allresults,
               function(x) x$SC.loss.v)
      names(output$all.results$SC.loss.v) <- paste0(tune_pars_list$m,"-NN")

      output$all.results$SC.loss.w <-
        sapply(allresults,
               function(x) x$SC.loss.w)
      names(output$all.results$SC.loss.w) <- paste0(tune_pars_list$m,"-NN")

      output$all.results$SCfolds.loss.v <-
        sapply(allresults, function(x)
          x$SCfolds.loss.v)
      names(output$all.results$SCfolds.loss.v) <- paste0(tune_pars_list$m,"-NN")

      output$all.results$SCfolds.loss.w <-
        sapply(allresults, function(x)
          x$SCfolds.loss.w)
      names(output$all.results$SCfolds.loss.w) <- paste0(tune_pars_list$m,"-NN")
    }
    return(output)
  }
