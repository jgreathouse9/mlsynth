################################################################

FDID_fun <- function(no_control,control_ID,x,y1,y2,t1,t){
  # no_control: number of control
  # control_ID: control group ID
  # x:          Control group coutcomes
  # y1:         Treated group before treated
  # y2:         Teated group after treated
  # t1:         Pretreatment time length
  # t:          total time length
  N=no_control
  R2=rep(0,no_control)
  
  # select the first control unit, t(x) = transpose of X
  for (j in 1:N){
    
    x1_DID=x[1:t1,control_ID[j]]      # j-th control pre-data, t1 by 1
    x2_DID=x[(t1+1):t,control_ID[j]]  # j-th control post-data, t2 by 1
    
    beta_DID = mean(y1 - x1_DID)    # \hat \alpha = bar y1 - (bar x1_DID)
    y1_hat_DID = beta_DID + x1_DID  # DID in-sample-fit, t1 by 1
    y2_hat_DID=beta_DID+x2_DID      # DID out-of-sample prediction, t2 by 1
    y_hat_DID=c(y1_hat_DID,y2_hat_DID)  # DID fit and prediction, t by 1
    
    ATT_DID = mean( y2 - y2_hat_DID)           # ATT_{DID}
    ATT_DID_r = 100*ATT_DID/mean(y2_hat_DID)   # ATT_{DID} in percentage
    r_2_DID =1-(mean((y1-y1_hat_DID)^2))/(mean((y1-mean(y1))^2)) # R-square
    R2[j]=r_2_DID
    
  }
  select=which.max(R2)
  R2final= max(R2)
  
  
  for (k in 2:no_control){
    left = setdiff(1:N, select)
    # control_left = x[,left]
    R2 = rep(0, length(left))
    for (jj in 1:length(left)){
      control_1=x[1:t1,c(select,left[jj])] # add jj-th to the previous selected controls 
      control_2=x[(t1+1):t,c(select,left[jj])] # add one-at-a-time
      
      
      x1_f_DID = rowMeans(control_1)    # ave. of above selected controls, it is t1 by 1
      beta_f_DID = mean( y1 - x1_f_DID )  # estimate of the intercept 
      x2_f_DID = rowMeans(control_2)      # ave. of selected controls, it is t2 by 1           
      y1_hat_f_DID = beta_f_DID + x1_f_DID    # t1 by 1, F-DID in-sample-fit
      y2_hat_f_DID=beta_f_DID+x2_f_DID        # t2 by 1, F-DID out-of-sample prediction
      y_hat_f_DID = c(y1_hat_f_DID,y2_hat_f_DID) # t by 1, F-DID fit and prediction        
      R2_f_DID=1-(mean((y1-y1_hat_f_DID)^2 ))/(mean((y1-mean(y1))^2 )) # R-square
      R2[jj]=R2_f_DID
      
    }
    
    index = left[which.max(R2)]
    if (length(index>1)){
      index=index[1]
    }
    R2final = c(R2final,max(R2))
    select = append(select, index)   
    
  }
  
  num_c = which.max(R2final)
  
  control=x[,select[1:num_c]]   # f-selected controls:select(1:num_c)
  control_10=control[1:t1,]      # control group pretreatment data
  control_20=control[(t1+1):t,]  # control group posttreatment data
  
  x1_forward_DID = rowMeans(control_10)  # average over control units, t1 by 1 
  x2_forward_DID = rowMeans(control_20)  # average over control units, t2 by 1
  beta_forward_DID = mean(y1- x1_forward_DID) # F-DID intercept estimate
  
  y1_hat_forward_DID = beta_forward_DID + x1_forward_DID # in-sample fit, t1 by 1
  y2_hat_forward_DID = beta_forward_DID + x2_forward_DID # prediction, t2 by 1
  
  ATT_forward=mean(y2-y2_hat_forward_DID)  # ATT estimate by the f-DID method
  ATT_forward_per=100*ATT_forward/mean(y2_hat_forward_DID) # ATT in percentage
  R2_forward_DID=1-(mean((y1-y1_hat_forward_DID)^2))/mean((y1-mean(y1))^2) # R-square 
  
  return(list(y1_hat_FDID=y1_hat_forward_DID,y2_hat_FDID=y2_hat_forward_DID,
              R2=R2final,select=select,num_c=num_c,
              ATT_FDID=ATT_forward,ATT_FDID_per=ATT_forward_per,
              R2_FDID=R2_forward_DID))
  
}

