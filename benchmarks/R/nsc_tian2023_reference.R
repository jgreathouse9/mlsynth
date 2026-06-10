

# used for the case of single treated unit

NSC = function(Data, ID, Time, Outcome, Treatment, Covariates = NULL, 
               Negativity = TRUE, a = NULL, b = NULL, Weighted = TRUE, grid = .2,
               CI = TRUE, CI_pre = TRUE, CI_level = .95) {
  # ID, Time, Y, D, X can be column name or index

  require("quadprog")
  
  
  ## preparing data
  
  # Data=d; ID=1; Time=2; Outcome=3; Treatment=4; Covariates=NULL; 
  # Negativity = TRUE; a = NULL; Weighted = TRUE; b = NULL; grid = .2; CI = TRUE; CI_pre = TRUE; CI_level = .95
  Data = Data[order(Data[,ID],Data[,Time]), ]
  
  N = length(unique(Data[,ID]))
  J = N-1
  T1 = length(unique(Data[Data[,Treatment]==1,Time]))
  T0 = length(unique(Data[,Time]))-T1
  TT = T0+T1
  k = length(Covariates)
  a0 = a
  b0 = b
  
  # Y is the N by TT matrix of outcomes
  # D is the N by 1 vector of treatment status
  # Z is the N by (k*T0+T0) matrix of observed covariates and pretreatment outcomes (the last T0 columns)
  Y = matrix(Data[,Outcome],nrow=N,byrow=T)
  Data[,Time] = rep(1:TT,N)
  D = (Data[Data[,Time]==T0+1,Treatment]==1)*1
  treated = which(D==1)
  Z = c()
  for (i in 1:N) {
    Z = rbind(Z,c(as.vector(t(Data[,Covariates][1:T0+(i-1)*TT,])),Y[i,1:T0]))
  } 
  Z = Z[,!duplicated(as.list(data.frame(Z)))] # remove duplicate columns
  Z = Z[,complete.cases(t(Z))] # remove columns with missing values
  Z = scale(Z, center = T, scale = apply(Z, 2, sd))
  
  # pairwise distances
  dist = matrix(NA,N,N)
  for (i in 1:N) {
    for (j in i:N) {
      dist[i,j] = sum((Z[i,]-Z[j,])^2)^.5
      dist[j,i] = dist[i,j]
    }
  }
  
  
  
  ## functions
  
  if (Negativity) {
    fn_weights = function(Z1,ZJ,a,b,dist_J){
      # elastic net with weighted L1 penalty 
      # (modified from quad.int for lars.c, PACLasso, James et al. 2019 Penalized and Constrained Optimization)
      # a and b transformed so that they are selected from [0,1]
      J = nrow(ZJ)
      ZJstar = rbind(ZJ,-ZJ)
      Dmat0 = ZJstar%*%t(ZJstar)
      Dmat0[lower.tri(Dmat0)] = t(Dmat0)[lower.tri(Dmat0)] # ensure symmetry not affected by machine precision
      eg0 = eigen(Dmat0)$values
      eg0 = eg0[eg0>10^-7]
      if (b>0) {b = sort(eg0)[ceiling(b*length(eg0))]*b}
      Dmat = Dmat0 + (b+10^-8)*diag(2*J) # add very small number to be positive denifite
      Dmat[lower.tri(Dmat)] = t(Dmat)[lower.tri(Dmat)]
      eg1 = eigen(Dmat)$values
      eg1 = eg1[eg1>10^-7]
      if (a>0) {a = sort(eg1)[ceiling(a*length(eg1))]*a}
      dvec = ZJstar%*%Z1 - .5*a*(rep(dist_J,2)/mean(dist_J)*Weighted+1*(!Weighted))
      Amat = t(rbind(c(rep(1,J),rep(-1,J)),diag(2*J)))
      bvec = c(1,rep(0,2*J))
      temp = solve.QP(Dmat, dvec, Amat, bvec, meq = 1)
      W = temp$sol[1:J] - temp$sol[(J+1):(2*J)];
      return(W)
    }
  } else {
    fn_weights = function(Z1,ZJ,a,b,dist_J){
      J = nrow(ZJ)
      ZJstar = ZJ
      Dmat0 = ZJstar%*%t(ZJstar)
      Dmat0[lower.tri(Dmat0)] = t(Dmat0)[lower.tri(Dmat0)] # ensure symmetry not affected by machine precision
      eg0 = eigen(Dmat0)$values
      eg0 = eg0[eg0>10^-7]
      if (b>0) {b = sort(eg0)[ceiling(b*length(eg0))]*b}
      Dmat = Dmat0 + (b+10^-8)*diag(J) # add very small number to be positive denifite
      Dmat[lower.tri(Dmat)] = t(Dmat)[lower.tri(Dmat)]
      eg1 = eigen(Dmat)$values
      eg1 = eg1[eg1>10^-7]
      if (a>0) {a = sort(eg1)[ceiling(a*length(eg1))]*a}
      dvec = ZJstar%*%Z1 - .5*a*(dist_J/mean(dist_J)*Weighted+1*(!Weighted))
      Amat = t(rbind(c(rep(1,J)),diag(J)))
      bvec = c(1,rep(0,J))
      temp = solve.QP(Dmat, dvec, Amat, bvec, meq = 1)
      W = temp$sol
      return(W)
    }
  }
  
  
  fn_cv = function(a,b,dist1=dist) {
    # CV using own pretreatment outcomes only
    # because unlike regression parameters, the vector of weights are different for different units
    perr = c()
    for (j in 1:J) {
      Z0 = Z[-treated,]
      Z1 = cbind(Z0[j,])
      index = sample(setdiff(1:J,j),1)
      ZJ = rbind(Z0[-j,],Z0[index,])
      dist_J = c(dist1[-treated,-treated][j,-j],dist1[-treated,-treated][j,index])
      W = fn_weights(Z1,ZJ,a,b,dist_J)
      perr = c(perr, mean((Y[-treated,][j,(T0+1):TT]-t(rbind(Y[-treated,][-j,(T0+1):TT],Y[-treated,][index,(T0+1):TT]))%*%W)^2))
    }
    sqrt(mean(perr))
  }
  
  fn_tuning_a = function(b) {
    perr = c()
    for (s in seq(0,1,grid)) {
      perr = c(perr,fn_cv(s,b))
    }
    a = seq(0,1,grid)[order(perr)[1]]
    perr = min(perr)
    return(list(a=a,perr=perr))
  }
  
  fn_tuning_b = function(a) {
    perr = c()
    for (s in seq(0,1,grid)) {
      perr = c(perr,fn_cv(a,s))
    }
    b = seq(0,1,grid)[order(perr)[1]]
    perr = min(perr)
    return(list(b=b,perr=perr))
  }
  
  fn_tuning = function(a = NULL, b = NULL) {
    # print('Selecting tuning parameters...')
    if (is.null(a) & !is.null(b)) {
      a = fn_tuning_a(b)$a
    } else if (!is.null(a) & is.null(b)) {
      b = fn_tuning_b(a)$b
    } else if (is.null(a) & is.null(b)) {
      tuning_a = fn_tuning_a(0)
      a = tuning_a$a
      # print(paste('a=',a,'RMSPE=',tuning_a$perr))
      tuning_b = fn_tuning_b(a)
      b = tuning_b$b
      # print(paste('b=',b,'RMSPE=',tuning_b$perr))
      if (b!=0 & tuning_a$perr-tuning_b$perr>.01*tuning_a$perr) {
        repeat {
          tuning_a = fn_tuning_a(b)
          a = tuning_a$a
          # print(paste('a=',a,'RMSPE=',tuning_a$perr))
          if (tuning_b$perr-tuning_a$perr<.01*tuning_b$perr) {break}
          tuning_b = fn_tuning_b(a)
          b = tuning_b$b
          # print(paste('b=',b,'RMSPE=',tuning_b$perr))
          if (tuning_a$perr-tuning_b$perr<.01*tuning_a$perr) {break}
        }
      }
    }
    return(list(a = a, b = b))
  }
  

  
  ## choosing tuning parameters
  if (is.null(a0) & is.null(b0)) {
    result_tuning = fn_tuning()
    a = result_tuning$a
    b = result_tuning$b
  } else if (is.null(a0) & !is.null(b0)) {
    result_tuning = fn_tuning(b = b0)
    a = result_tuning$a
    b = b0
  } else if (!is.null(a0) & is.null(b0)) {
    result_tuning = fn_tuning(a = a0)
    b = result_tuning$b
    a = a0
  } else if (!is.null(a0) & !is.null(b0)) {
    a = a0
    b = b0
  }
  
  
  ## estimating ITE
  
  Z0 = Z
  Z1 = cbind(Z0[treated, ])
  ZJ = Z0[-treated,]
  weights = fn_weights(Z1,ZJ,a,b,dist[treated,-treated])
  ITE = Y[treated,1:TT] - weights%*%Y[-treated,1:TT]

  
  ## inference
  
  CI_lower = matrix(NA,1,TT)
  CI_upper = matrix(NA,1,TT)
  if (CI) {
    # print('Constructing confidence interval...')
    perr = c()
    for (j in 1:J) {
      Z0 = Z[-treated,]
      Z1 = cbind(Z0[j,])
      index = sample(setdiff(1:J,j),1)
      ZJ = rbind(Z0[-j,],Z0[index,])
      dist_J = c(dist[-treated,-treated][j,-j],dist[-treated,-treated][j,index])
      W = fn_weights(Z1,ZJ,a,b,dist_J)
      perr = cbind(perr, Y[-treated,][j,]-t(rbind(Y[-treated,][-j,],Y[-treated,][index,]))%*%W)
    }
    for (t in 1:TT) {
      se = sqrt(sum((perr[t,])^2)/(J-1))
      CI_lower[t] = ITE[t] + qnorm((1-CI_level)/2)*se
      CI_upper[t] = ITE[t] + qnorm((1+CI_level)/2)*se
    }
  }
    

  return(list(tuning = rbind(a,b), weights = round(weights,3), n_donor = sum(abs(weights)>1e-3), ITE = ITE, CI_lower=CI_lower, CI_upper=CI_upper))
}


