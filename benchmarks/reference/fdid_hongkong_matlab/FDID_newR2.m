
function [y1_hat_forward_DID,y2_hat_forward_DID,R2final,control_10,control_20]=FDID_newR2(no_control,control_ID,x,y1,y2,t1,t)

R2=zeros(no_control,1); 
R2final=zeros(no_control,1);

select_c=zeros(1,no_control); % record order of selected control units (ID) 
    
for j=1:no_control    % this loop selects the FIRST control unit, N_{co}=1        
  x1_DID=x(1:t1,control_ID(j));      % j-th control pre-data, t1 by 1
  x2_DID=x((t1+1):t,control_ID(j));  % j-th control post-data, t2 by 1
        
  beta_DID = mean(y1 - x1_DID);    % \hat \alpha = bar y1 - (bar x1_DID)
  y1_hat_DID = beta_DID + x1_DID;  % DID in-sample-fit, t1 by 1
  y2_hat_DID=beta_DID+x2_DID;      % DID out-of-sample prediction, t2 by 1
  y_hat_DID=[y1_hat_DID;y2_hat_DID];  % DID fit and prediction, t by 1
        
 ATT_DID = mean( y2 - y2_hat_DID);           % ATT_{DID}
 ATT_DID_r = 100*ATT_DID/mean(y2_hat_DID);   % ATT_{DID} in percentage
 r_2_DID =1-(mean((y1-y1_hat_DID).^2))/(mean((y1-mean(y1)).^2)); % R-square
 R2(j,:)=r_2_DID;
    
end
 R2final(1,:)=max(R2);  % 1st row of R2final records the largest R2 for N_co=1
 first_c=find(R2==max(nonzeros(R2))); % it finds the ID of the control that has the largest R2
  if length(first_c) > 1 
     first_c = first_c(1,1); 
  end
 select_c(1,1)=first_c;  % It is the first selected control unit's ID
   
 for k=2:no_control   % consider N_{co} size from 2 to no_control
  left=setdiff(control_ID,select_c); % it removes control ID selected from previous step
  control_left=x(:,left); % t by (N_co - removed numbers)
  R2=zeros(length(left),1); % redefine R2 as a vector of zeros for remaining controls 
 
 for jj=1:length(left)   
 % when k=2, jj from 1 to no_control -1 to select the 2nd control,when k=3, selects 3rd,
 % ... when k=no_control, jj=1, pick up the last control and uses all the controls
 
control_1=x(1:t1,[nonzeros(select_c)',left(jj)]); % add jj-th to the previous selected controls 
control_2=x((t1+1):t,[nonzeros(select_c)',left(jj)]); % add one-at-a-time
        
 x1_f_DID = [mean(control_1,2)];    % ave. of above selected controls, it is t1 by 1
 beta_f_DID = mean( y1 - x1_f_DID );  % estimate of the intercept 
 x2_f_DID = mean(control_2,2);      % ave. of selected controls, it is t2 by 1           
    
 y1_hat_f_DID = beta_f_DID + x1_f_DID ;   % t1 by 1, F-DID in-sample-fit
 y2_hat_f_DID=beta_f_DID+x2_f_DID;        % t2 by 1, F-DID out-of-sample prediction
 y_hat_f_DID = [y1_hat_f_DID; y2_hat_f_DID]; % t by 1, F-DID fit and prediction        
 R2_f_DID=1-(mean((y1-y1_hat_f_DID).^2 ))/(mean((y1-mean(y1)).^2 )); % R-square
 R2(jj,:)=R2_f_DID;
 end
  R2final(k,:)=max(R2);
  select=left(find(R2==max(nonzeros(R2))));  % find the newly added ID that gives largest R2
  select_c(1,k)=select; % assign the above ID to the k-th position of select_c
 
end
select_c % It lists orders of controls ID by the order of selection
newR2=[select_c',R2final];   
num_c=find(R2final==max(R2final)); % # controls that has max R^2

Number_controls_selected_by_fDID = num_c

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Forward DID ATT, ATT% and R-squares
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

control=x(1:end,select_c(1:num_c)); % f-selected controls:select_c(1:num_c)
control_10=control(1:t1,:);         % control group pretreatment data
control_20=control((t1+1):t,:);     % control group posttreatment data

x1_forward_DID = mean(control_10,2);  % average over control units, t1 by 1 
x2_forward_DID = mean(control_20,2);  % average over control units, t2 by 1
beta_forward_DID = mean(y1- x1_forward_DID); % F-DID intercept estimate

y1_hat_forward_DID = beta_forward_DID + x1_forward_DID; % in-sample fit, t1 by 1
y2_hat_forward_DID = beta_forward_DID + x2_forward_DID; % prediction, t2 by 1
% R2final
end