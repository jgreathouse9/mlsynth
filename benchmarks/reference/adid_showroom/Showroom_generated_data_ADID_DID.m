%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADID method with N_co=10; We generate data using a factor model to fit
% each control unit's data first, then we use the fitted data to estimate
% ATT
% ATT by the ADID method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
 city = 1; t1 = 83; % for Boston
% city = 2; t1 = 90;   % for Columbus
t=110;  
t2=t-t1;
if city == 1; new=csvread('Showroom_Generate_Data_Boston.csv');
   elseif city == 2; new=csvread('Showroom_Generate_Data_Columbus.csv');   
 end     
 datay=new(:,1);     % t by 1 of treatment unit's outcome (t=110)
 datax=new(:,2:end); % t by N_co control units' outcomes (N_co=10)
 const=ones(t,1);  % t by 1 vector of ones (for intercept)
 x=[const,datax];  % add an intercept to control unit data matrix, t by N (N=11)               
 x1=x(1:t1,:);     %  control units' pretreatment data matrix, t1 by N
 x2=x(t1+1:t,:);   %  control units' pretreatment data matrix, t2 by N
 y = datay;        % t by 1 of treatment unit's outcome (t=110)
 y1=datay(1:t1,1);  % t1 by 1 vector of treatment unit's pre-treatment data
 y2=datay(t1+1:t,1); % t2 by 1 vector of treatment unit's post-treatment data
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Below is ATT estimation by DID method  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x10=x1(:,2:end);  % x10 removes an intercept from x1
x20=x2(:,2:end);  % x20 removes an intercept from x2

x1_DID = [mean(x10,2)];  % t1 by 1 vector of average control outcome
beta_DID = mean(y1 - x1_DID);  % This is DID intercept estimate
r_2_DID=1-mean((y1-beta_DID-x1_DID).^2 )/mean((y1-mean(y1)).^2); % R-square
 x2_DID = mean(x20,2);  % t2 by 1 vector of average control outcome
 y1_DID = beta_DID + x1_DID ;   % DID in-sample-fit
 y2_DID = beta_DID + x2_DID ;   % DID out-of-sample prediction
 ATT_DID = mean( y2 - y2_DID )  % DID ATT estimate
 ATT_DID_per = 100*ATT_DID/mean( y2_DID) % DID ATT in percentage
 y_DID = [y1_DID; y2_DID]; % t by 1 vector of DID fit/prediction
      
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % Below is ATT estimation by ADID method 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
 x10 = datax(1:t1,:);
 x20 = datax(1+t1:t,:);
 x1_ADID =[ones(size(x10,1),1),mean(x10,2)];
 x2_ADID =[ones(size(x20,1),1),mean(x20,2)];
 b_ADID = (x1_ADID'*x1_ADID)\(x1_ADID'*y1); % ADID estimate of delta
 y1_ADID = x1_ADID*b_ADID;   % t1 by 1 vector of ADID in-sample fit
 y2_ADID = x2_ADID*b_ADID;   % t2 by 1 vector of ADID prediction
 y_ADID =[y1_ADID; y2_ADID]; % t by 1 vector of ADID fit/prediction
 ATT_ADID = mean(y2 - y2_ADID)            % ATT by ADID
 ATT_ADID_per=100*ATT_ADID/mean(y2_ADID)  % ATT in percentage by ADID
 
 e1_ADID=y1-y1_ADID; % t1 by 1 vector of treatment unit's (pre-treatment) residuals  
 sigma2_ADID = mean(e1_ADID.^2);  %   \hat sigma^2_e
 eta_ADID = mean(x2_ADID)';         
 psi_ADID = x1_ADID'*x1_ADID/t1;
  
   Omega_1_ADID = sigma2_ADID*eta_ADID'*inv(psi_ADID)*eta_ADID;
   Omega_2_ADID = sigma2_ADID;
   Omega_ADID  = (t2/t1)*Omega_1_ADID + Omega_2_ADID; % Variance of sqrt{T_2}(\hat \Delta_1 - \Delat_1)
   
   ATT_ADID_std = sqrt(t2)*ATT_ADID/sqrt(Omega_ADID) % standardize statistic,
   % ATT_ADID_Std is distributed as N(0,1) under H0: ATT=0, one can get p-value
   % for a 1-sided test H1: ATT>0, p-value = 1 - normcdf(ATT_ADID_std).
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Below is ADID using generated data  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
y_hat = y_ADID;
t_trend=[1:1:t]';  % a t by 1 vector of [1,2,...,t]', it is the x-axis
 c_min=1.1;
 c_max=1.1;
y_min = c_min*min(min(y),min(y_hat));  %  figure's minimum value in y-axis
y_max = c_max*max(max(y),max(y_hat));  %  figure's maximum value in y-axis
figure1=figure;
 plot(t_trend,y_hat,'--b','linewidth',1.5);
 hold on
 plot(t_trend,y,'-k'); 
 hold off
 axis([0 t y_min y_max]) 
 line([t1,t1],[y_min,y_max]);     % draw a vertical line at t=t1, indicating treatment time
 legend('ADID\_fitted','Actual','Location','northwest')
%  saveas(figure1,'ATT_ADID_fitted_Boston.eps');  % save it as an eps file
%  saveas(figure1,'ATT_ADID_fitted_Boston.pdf');  % save it as a pdf file 
 saveas(figure1,'ATT_ADID_fitted_Columbus.eps');  % save it as an eps file
 saveas(figure1,'ATT_ADID_fitted_Columbus.pdf');  % save it as a pdf file 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Plot DID using generated data 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 y_hat = y_DID; 
 t_trend=[1:1:t]';  % a t by 1 vector of [1,2,...,t]', it is the x-axis
 y_min = c_min*min(min(y),min(y_hat));  %  figure's minimum value in y-axis
 y_max = c_max*max(max(y),max(y_hat));  %  figure's maximum value in y-axis
 figure2=figure;
 plot(t_trend,y_hat,'--b','linewidth',1.5);
 hold on
 plot(t_trend,y,'-k');
 hold off
 axis([0 t y_min y_max]) 
 line([t1,t1],[y_min,y_max]);     % draw a vertical line at t=t1, indicating treatment time
 legend('DID\_fitted','Actual','Location','northwest')
%  saveas(figure2,'ATT_DID_fitted_Boston.eps');  % save it as an eps file
%  saveas(figure2,'ATT_DID_fitted_Boston.pdf');  % save it as a pdf file
 saveas(figure2,'ATT_DID_fitted_Columbus.eps');  % save it as an eps file
 saveas(figure2,'ATT_DID_fitted_Columbus.pdf');  % save it as a pdf file 


