scm.corner <- function(Y1pre,Y0pre,X1,X0){
  ##step1
  Tpre <- dim(Y0pre)[1]
  nDonors <- dim(Y0pre)[2]
  #QP setup
  c1 <- -t(Y0pre) %*% Y1pre
  H1 <- t(Y0pre) %*% Y0pre
  A1 <- matrix(rep(1,nDonors), ncol = nDonors)
  b1 <- 1
  r1 <- 0
  l1 <- matrix(rep(0,nDonors), nrow = nDonors)
  u1 <- matrix(rep(1,nDonors), nrow = nDonors)
  #run QP
  step1_ipop <- ipop(c = c1, H = H1, A = A1, b = b1, l = l1, u = u1, r = r1, 
                     margin = 0.0005, maxiter = 1000, sigf = 7, bound = 10) #QP_Solver1
  step1_lowr <- LowRankQP(Vmat = H1, dvec = c1, Amat = A1, bvec = b1, uvec = u1, 
                          method = "LU") #QP_Solver2
  W_ipop <- matrix(step1_ipop@primal, nrow = nDonors)
  W_lowr <- step1_lowr$alpha
  L1_ipop <- (t(Y1pre) %*% Y1pre)/Tpre + 2/Tpre * (t(c1) %*% W_ipop 
                                                   + 0.5 * t(W_ipop) %*% H1 %*% W_ipop)
  L1_lowr <- (t(Y1pre) %*% Y1pre)/Tpre + 2/Tpre * (t(c1) %*% W_lowr 
                                                   + 0.5 * t(W_lowr) %*% H1 %*% W_lowr)
  
  ##step2
  #normalize X - Synth
  nvarsV <- dim(X0)[1]
  big.dataframe <- cbind(X0, X1)
  divisor <- sqrt(apply(big.dataframe, 1, var))
  scaled.matrix <- t(t(big.dataframe) %*% ( 1/(divisor) 
                                            * diag(rep(dim(big.dataframe)[1], 1)) ))
  X0.scaled <- scaled.matrix[,c(1:(dim(X0)[2]))]
  if(is.vector(X0.scaled)==TRUE)
  {X0.scaled <- t(as.matrix(X0.scaled))}
  X1.scaled <- scaled.matrix[,dim(scaled.matrix)[2]]
  #LP setup
  f.obj_ipop <- (X1.scaled - X0.scaled %*% W_ipop)^2
  f.obj_lowr <- (X1.scaled - X0.scaled %*% W_lowr)^2
  f.con <- rbind(rep(1,nvarsV), diag(x = 1, nrow = nvarsV))
  f.dir <- c("=", rep(">=",nvarsV))
  f.rhs <- c(1, rep(0,nvarsV))
  #run LP
  step2_ipop <- lp ("min", f.obj_ipop, f.con, f.dir, f.rhs)
  step2_lowr <- lp ("min", f.obj_lowr, f.con, f.dir, f.rhs)
  V_ipop <- step2_ipop$solution
  V_lowr <- step2_lowr$solution
  L2_ipop <- step2_ipop$objval
  L2_lowr <- step2_lowr$objval
  
  scm.corner.out <- list(W = cbind(W_ipop,W_lowr), V = cbind(V_ipop,V_lowr), 
                         Lv = c(L1_ipop,L1_lowr), Lw = c(L2_ipop,L2_lowr))
  return(scm.corner.out)
}