%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate ATT by the Forward DID method and conventional DID
% method that uses all controls. The GDP data is from Hsiao, Ching and Wan 
% (2012, J. of Applied Econometrics) and is available at JAE data archive 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
dataxy=csvread('GDP.csv',1,0);% download data, remove 1st row country names
datay=dataxy(:,1);   % 1st column is Hong Kong GDP growth = y, treated unit
datax=dataxy(:,2:end);               % control countries GDP growth data 
Control_data_dimension = size(datax) % t by N_co, t=61, N_co=24 for the DGP data 

t=size(datay,1); % the total time period sample size, t=61 for the GDP data
t1 = 44;      % t1 = 44 is the pretreatment sample size, provided by user
t2=t-t1;      % t2 = 17 is the post-treatment sample size
y1=datay(1:t1,1);   % t1 by 1, treatment unit's pre-treatment data
y2=datay(t1+1:t,1); % t2 by 1, treatment unit's post-treatment data
y=[y1;y2];             % t by 1, treatment unit's data
x = datax;             % control units' data matrix
x1 = x(1:t1,:);        % pretreatment control units' data matrix
x2 = x(t1+1:t,:);      % posttreatment control units' data matrix
no_control=size(x,2);  % number of control units, it is 24 for the GDP data
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            Compute ATT using the forward DID method 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

control_ID=[1:no_control];  % 1 row vector from 1 to no_control 
[y1_hat_FDID,y2_hat_FDID,R2final]=FDID_newR2(no_control,control_ID,x,y1,y2,t1,t);
% The above line calls the subroutine FDID_newR2 that delivers in-sample-fit, out-of-sample
% predicted counterfactual outcomes, and the final vector of R-square
% R2final is a no_control by 1 vector that gives R2 for all submodels from N_co=1 to N_co=no_control

y_hat_FDID=[y1_hat_FDID;y2_hat_FDID]; % fit & prediction, t by 1 
ATT_FDID=mean(y2-y2_hat_FDID)  % ATT estimate by the f-DID method
ATT_FDID_per=100*ATT_FDID/mean(y2_hat_FDID) % ATT in percentage
R2_forward_DID=1-(mean((y1-y1_hat_FDID).^2))/mean((y1-mean(y1)).^2) % R-square 

u1_FDID = y1 - y1_hat_FDID;     % estimated residual, t1 by 1
Omega_1_hat_FDID=(t2/t1)*mean(u1_FDID.^2);% \hat\Sigma_{1,FDID}
Omega_2_hat_FDID=mean(u1_FDID.^2);        % \hat\Sigma_{2,FDID}   
std_Omega_hat_FDID=sqrt(Omega_1_hat_FDID+Omega_2_hat_FDID); 
% square-root of \hat Sigma^2_{FDID}
 
ATT_std_FDID=sqrt(t2)*ATT_FDID/std_Omega_hat_FDID 
% standardized ATT, it is N(0,1) under H0: ATT = 0
p_value_forward_DID=2*(1-normcdf(abs(ATT_std_FDID))) % p-value for ATT=0
p_value_f_one_sided=(1-normcdf(ATT_std_FDID)) % p-value for 1-sided test
CI_95_FDID_left= ATT_FDID-1.96*std_Omega_hat_FDID/sqrt(t2);
CI_95_FDID_right=ATT_FDID+1.96*std_Omega_hat_FDID/sqrt(t2);
CI_95_FDID_width=[CI_95_FDID_left,CI_95_FDID_right,CI_95_FDID_right-CI_95_FDID_left]
% It reports 95% confidence interval of the FDID ATT estimate and the width of the interval

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Compute ATT using the conventional DID method with all control units
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 x_DID = mean(x,2);        % t by 1 vector, ave. outcome of control units
 x1_DID = x_DID(1:t1,:);   % the first t1 components of x_DID
 x2_DID = x_DID(t1+1:t,:); % the last t2 components of x_DID
 b_DID = mean(y1 - x1_DID);  % DID intercept estimator
 y1_DID = b_DID + x1_DID ;     % DID in-sample-fit
 y2_DID = b_DID + x2_DID ;     % DID out-of-sample prediction
 y_DID = [y1_DID; y2_DID];
     
 ATT_DID = mean( y2 - y2_DID)                   % DID ATT estimate
 ATT_DID_percentage = 100*ATT_DID/mean(y2_DID)  % DID ATT in percentage
 R2_DID=1-(mean((y1-y1_DID).^2 ))/(mean((y1-mean(y1)).^2))  % DID R-square
   u1_DID = y1 - y1_DID;  % estimated DID residual, t1 by 1
   Omega_1_hat_DID=(t2/t1)*mean(u1_DID.^2);  % \hat \Sigma_{1,DID}
   Omega_2_hat_DID=mean(u1_DID.^2);          % \hat \Sigma_{2,DID}
    
std_Omega_hat_DID=sqrt(Omega_1_hat_DID+Omega_2_hat_DID); % \hat Sigma_{DID}
ATT_std_DID = sqrt(t2)*ATT_DID/std_Omega_hat_DID % it is N(0,1) under H0, ATT=0
p_value_DID = 2*(1 - normcdf(abs(ATT_std_DID)))  % p-value for H0: ATT=0
p_value_one_sided=(1-normcdf(ATT_std_DID))       % p-value for 1-sided test
CI_95_DID_left= ATT_DID-1.96*std_Omega_hat_DID/sqrt(t2);
CI_95_DID_right=ATT_DID+1.96*std_Omega_hat_DID/sqrt(t2);
CI_95_DID_width=[CI_95_DID_left,CI_95_DID_right,CI_95_DID_right-CI_95_DID_left]
% It reports 95% confidence interval of the DID ATT estimate and the width of the interval
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Plot the DID figure that uses all controls (figure 1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    y_hat = y_DID;   
    figure1 = figure;
    n=[1:1:t]';
    plot(n,y_hat,'--b','linewidth',1.5);
    hold on
    plot(n,y,'-k');
    upb = 1.1*max(max(y),max(y_hat));
    lpb = min(0.5*min(min(y),min(y_hat)),1.5*min(min(y),min(y_hat))); 
    axis([0 t lpb upb])
    line([n(t1),n(t1)],[lpb,upb]);
    hold off
     xlabel('Time (in quarters)') 
     ylabel('Hong Kong GDP growth rate')
  LEG=legend('DID','Actual','Location','northwest');
  set(LEG,'FontSize',10);
  
   
    saveas(gcf,'HK_GDP_DID.pdf');
    % saveas(gcf,'HK_GDP_DID.fig');
    % saveas(gcf,'HK_GDP_DID.eps');
    % saveas(gcf,'HK_GDP_DID.png');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Plot the Forward selection DID figure (figure 2)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    n=[1:1:t]';
    figure2 = figure;
    y_hat = y_hat_FDID;     
    plot(n,y_hat,'--b','linewidth',1.5);
    hold on
    plot(n,y,'-k');
    upb = 1.1*max(max(y),max(y_hat));
    lpb = min(0.5*min(min(y),min(y_hat)),1.5*min(min(y),min(y_hat)));   
    axis([0 t lpb upb])
    line([n(t1),n(t1)],[lpb,upb]);
    hold off   
     xlabel('Time (in quarters)') 
     ylabel('Hong Kong GDP growth rate')
    LEG=legend('Forward DID','Actual','Location','northwest');
    set(LEG,'FontSize',10);
 
   
    saveas(gcf,'HK_GDP_forward_DID.pdf');
    % saveas(gcf,'HK_GDP_forward_DID.fig');
    % saveas(gcf,'HK_GDP_forward_DID.eps');
    % saveas(gcf,'HK_GDP_forward_DID.png');
format long 

