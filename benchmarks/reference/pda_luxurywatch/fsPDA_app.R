#' PLS estimation by the iterative algorithm via Rmosek
#'
#' @param treated T-vector. The time series of the treated unit.
#' @param control T-by-N matrix. Each column is the time series of a control unit.
#' @param intervention_time An indicator of the time when the intervention takes place
#' @param stopping_criterion The default is "BIC". Use the BIC to stop the iteration
#' 
#' 
#' @return A list 
#' \item{control_group}{The identities of the selected units}
#' \item{treatment_effect}{ A T2-vector of the treatment effect in the post-treatment period}
#' \item{t_test}{Information out the average treatment effect and the t-statistic}
#'
#' 
#' @export
#'
#'



fsPDA = function (treated, control, intervention_time, stopping_criterion = "BIC"){

  
  # suppose the names/colnames are time
  obs_treated = names(treated)
  obs_control = rownames(control)
  
  if( any(obs_treated != obs_control) )
    stop ('observation periods for treated and control are different')
  
  t = length(treated)
  t1 = which(names(treated) == intervention_time) - 1
  t2 = t - t1
  N = ncol(control)
  
  pre_treated = treated[1:t1]
  pre_control = control[1:t1,]
  post_treated = treated[(t1+1):t]
  post_control = control[(t1+1):t,]
  
  R2 = rep(0, N)
  
  # select the first control unit
  for (j in 1:N){
    X = cbind(1, pre_control[,j])
    b = MASS::ginv(t(X) %*% X) %*% t(X) %*% pre_treated
    SSE = (t(X %*% b)) %*% (X %*% b)
    
    R2[j] = SSE
  }
  select = which.max(R2)
  
  X = cbind(1, pre_control[,select])
  b = MASS::ginv(t(X) %*% X) %*% t(X) %*% pre_treated
  y.hat = as.vector (X %*% b)
  e = pre_treated - y.hat
  sigma.sq = t(e) %*% e / t1
  B = log(log(N))*log(t1)/t1
  Q.new = log(sigma.sq) + B
  Q.old = Q.new + 1
  
  # iteration after the first selction
  
  while (Q.new < Q.old ) {
    left = setdiff(1:N, select)
    pre_control.LEFT = pre_control[,left]
    
    Q.old = Q.new
    
    R2 = rep(0, length(left))
    
    for (j in 1:length(left)){
      
      X = cbind(1, pre_control [,select], pre_control.LEFT[,j])
      b = MASS::ginv(t(X) %*% X) %*% t(X) %*% pre_treated
      SSE = (t(X %*% b)) %*% (X %*% b)
      
      R2[j] = SSE
    }
    
    index = left[which.max(R2)]
    select = append(select, index)
    k = length(select)
    
    X = cbind(1, pre_control[,select])
    b = MASS::ginv(t(X) %*% X) %*% t(X) %*% pre_treated
    y.hat = as.vector (X %*% b)
    e = pre_treated - y.hat
    sigma.sq = t(e) %*% e / t1
    # print(log(sigma.sq))
    B = log(log(N))*k*log(t1)/t1 
    Q.new = log(sigma.sq) + B
  }  
  
  select = select[1:(length(select)-1)]
  control_group = colnames(control)[select]
  
  X = cbind(1, pre_control[,select])
  b = MASS::ginv(t(X) %*% X) %*% t(X) %*% pre_treated
  treated_hat = drop(cbind(1, control[,select]) %*% b)
  
  treatment_effect = (treated - treated_hat)[(t1+1):t]
  
  ate = mean(treatment_effect)
  lvr = sandwich::lrvar(treatment_effect, type = "Newey-West", prewhite = T, adjust = T)
  stu = ate/sqrt(lvr)
  p_value = (1 - pnorm(abs(stu)))*2
  vec = c(ate, lvr, stu, p_value)
  names(vec) = c("average treatment effect","long-run variance","t-statistic", "p-value")
  
  return(list(control_group = control_group, treatment_effect = treatment_effect, t_test = vec))
  
}