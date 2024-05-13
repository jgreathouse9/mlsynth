%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Raw data plot using mock data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
tic
format long 
rng(1000)  
diary Mockdata_log.log
opt_lsq = optimoptions('lsqlin','Display','off');
opt_fmin = optimoptions('fmincon','Display','off');
nb =10000;
V_hatI = zeros(2,2);
data_yx=csvread("Data.csv",1,0);
y = data_yx(:,1);
Xbar = data_yx(:,2:end);

t=size(data_yx,1);
t=110;
t1=76;
t2=t-t1;

y1 = y(1:t1);
y2 = y(t1+1:t);
control = Xbar;
% dim_co=size(control,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 const=ones(t,1);        % t by 1 vector of ones
 x=[const,Xbar];         % intercept and control units data                    
 x1=x(1:t1,:);          
 x2=x(t1+1:t,:);

 n = size(x,2);     % Total # of units, n-1 = # of control units
 bm_MSC_c = zeros(n,nb);
 
 Rt=[0,  ones(1,n-1); 1,zeros(1,n-1)];  % test joint restriction of sum to one and zero intercept (H0)
 qt = [1;0];
 R1t = [0,  ones(1,n-1)];  % test single restriction of sum to 1 (H0a)
 R2t = [1,  zeros(1,n-1)]; % test single restriction of zero intercept (H0b)
 q1t = 1;
 q2t=0;

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %  Modified Synthetic Control with \beta_j \geq 0 for j\geq 2. %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
   % Below is SC weight estimates 
     lbO = zeros(n-1,1);
     ubO = ones(n-1,1);
     AeqO = ones(1,n-1); 
     beq0 = 1;      % The constraint is Aeq*b_SC = beq, sum slopes to 1.  
     b_SC = lsqlin(x1(:,2:end),y1,[],[],AeqO,beq0,lbO,ubO,[],opt_lsq);   % This is SC
%  b_SC0=b_SC; % will be used a initial value in the following 
%  fun = @(b_SC)sum((y1-x1(:,2:end)*b_SC).^2); % define a function 
%  b_SC = fmincon(fun,b_SC0,[],[],AeqO,beq0,lbO,ubO,[],opt_fmin); % need to set initial value; 

  % Below is MSC(a) weight estimates 
    lb=zeros(n,1);    
    ub = ones(n,1);
    ub(1) = Inf;
    lb(1)=-Inf;
    Aeq=ones(1,n); Aeq(1,1)=0; beq=1;      % The constraint is Aeq*b_Syn = beq, sum slopes to 1.
 b_MSC_a =lsqlin(x1,y1,[],[],Aeq,beq,lb,ub,[],opt_lsq);  % MSC(a), sum to one with intercept

% b_MSC_a0=b_MSC_a; % will be used a initial value in the following 
% fun2 = @(b_MSC_a)sum((y1-x1*b_MSC_a).^2); % define a function 
% b_MSC_a = fmincon(fun2,b_MSC_a0,[],[],Aeq,beq,lb,ub,[],opt_fmin); % need to set initial value; 

 % Below is MSC(b) weight estimates 
  lb=zeros(n-1,1);
     b_MSC_b =lsqlin(x1(:,2:end),y1,[],[],[],[],lb,[],[],opt_lsq); % This is MSC(b) 
    
 % Below is MSC(c) weight estimates 
     lb=zeros(n,1);
     lb(1)=-Inf;
     b_MSC_c=lsqlin(x1,y1,[],[],[],[],lb,[],[],opt_lsq); % This is MSC(c)

     
    d1t = R1t*b_MSC_c - q1t ;
    d2t = R2t*b_MSC_c - q2t ;
    dt = Rt*b_MSC_c - qt ;
    test1 = t1*(d1t'*d1t);  % test statistic for single restriction hypothesis test  sum to one H0a
    test2 = t1*(d2t'*d2t);  % test statistic for single restriction hypothesis test zero intercept H0b
    
     y1_SC = x1(:,2:end)*b_SC;   % pre-tr (hat )y by SC
     y1_MSC_a = x1*b_MSC_a;      % pre-tr (hat )y by MSC(a)
     y1_MSC_b = x1(:,2:end)*b_MSC_b;  % pre-tr (hat )y by MSC(b)
     y1_MSC_c = x1*b_MSC_c;       % pre-tr (hat )y by MSC(c)
     
     y2_SC = x2(:,2:end)*b_SC;       % post-tr (hat y) by SC
     y2_MSC_a = x2*b_MSC_a;          % by MSC(a)
     y2_MSC_b = x2(:,2:end)*b_MSC_b; % by MSC(b)
     y2_MSC_c = x2*b_MSC_c;          % by MSC(c)   
    
     z1=[x1,y1];
     bm_MSC_c_sum0 = zeros(nb,1);
  for g=1:nb                 % The number of bootstrap loop, the # of boostrap = nb  
 m=40;   % m is the subsampling sample size
    zm = datasample(z1,m,1);  % randomly picks up m rows from z1_{T_1 by (N+1)}
    ym = zm(:,n+1);  % the last column is y
    xm = zm(:,1:n);  
    
    lb=zeros(n,1);
    lb(1)=-Inf;
    bm_MSC_c(:,g)=lsqlin(xm,ym,[],[],[],[],lb,[],[],opt_lsq); % Constrained estimation with subsample m 
    
   bm_MSC_c_g=bm_MSC_c(:,g);
   bm_MSC_c_sum0(g) = sum(bm_MSC_c_g(2:end));

  dt_s = Rt*bm_MSC_c_g - qt;
  dt_ss =  Rt*(bm_MSC_c_g - b_MSC_c); 
  V_hatI = V_hatI + (m/nb)*dt_ss*dt_ss';
   
  d1t_s(g) = R1t*(bm_MSC_c_g - b_MSC_c);
  d2t_s(g) = R2t*(bm_MSC_c_g - b_MSC_c);

  test1_s(g) = m*(d1t_s(g)'*d1t_s(g));
  test2_s(g) = m*(d2t_s(g)'*d2t_s(g));
    
 end         % end of subsampling loop
   
  V_hat = inv(V_hatI); 

  for ggg=1:nb                                % for joint test
      ds =  Rt*(bm_MSC_c(:,ggg) - b_MSC_c);
      Js_test(ggg) = m*ds'*V_hat*ds ;
  end   % end of the above 2-line loop, ggg=1 to nb
   
    J_test = t1*dt'*V_hat*dt; 
 
     y_SC = [y1_SC; y2_SC]; 
     y_MSC_a=[y1_MSC_a;y2_MSC_a];
     y_MSC_b =[y1_MSC_b; y2_MSC_b];
     y_MSC_c =[y1_MSC_c; y2_MSC_c];

     t_trend=1:1:t;  
     t_trend=t_trend';
   y_tr=y;   % y_tr = y_treat 
   tt=1:t;
   lpb  = min(y_tr)-2000;  % for Figure 1's lower bound
   upb = max(y_tr)+1000;   % for Figure 1's upper bound
   
   % Data plots
    figure1=figure;
    p1=plot(tt,control,'color',[0 0 0]+.8,'linewidth',2); 
    hold on
    p2=plot(tt,y_tr,'k','linewidth',2);
    axis([0 110 lpb upb])
    line([tt(t1),tt(t1)],[lpb,upb]);
    xlabel('Week') 
    ylabel('Sales (in dollars)') 
    hold off
   legend([p2 p1(1)],'Treatment','Control','Location','northwest')
   saveas(figure1,'Treatment_control_data_plot.eps');
   saveas(figure1,'Treatment_control_data_plot.pdf');
     
% Figure for the SC fitted curve
  figure2=figure;
  y_hat = y_SC;
  plot(t_trend,y,'--k');
  hold on;
  plot(t_trend,y_hat,'b');
  hold off;

  upb = max(0.95*max(max(y),max(y_hat)),1.05*max(max(y),max(y_hat)));
  lpb = min(0.95*min(min(y),min(y_hat)),1.05*min(min(y),min(y_hat))); 
  axis([0 t lpb upb]);
  line([t1,t1],[lpb, upb]);
  legend('Actual','SC', 'Location', 'northwest');
  saveas(figure2,'SC_curve.eps');
  saveas(figure2,'SC_curve.pdf');

 % Figure for the MSC(a) fitted curve
  figure3=figure;
  y_hat = y_MSC_a;
  plot(t_trend,y,'--k')
  hold on
  plot(t_trend,y_hat,'b')
  hold off

  upb = max(0.95*max(max(y),max(y_hat)),1.05*max(max(y),max(y_hat)));
  lpb = min(0.95*min(min(y),min(y_hat)),1.05*min(min(y),min(y_hat)));
  axis([0 t lpb upb])
  line([t1,t1],[lpb upb]);
  legend('Actual','MSC(a)', 'Location', 'northwest')
  saveas(figure3,'MSC(a)_curve.eps');   % MSCb = no intercept
  saveas(figure3,'MSC(a)_curve.pdf');
 

% Figure for the MSC(b) fitted curve
 figure4=figure;
 y_hat = y_MSC_b;
 plot(t_trend,y,'--k')
 hold on
 plot(t_trend,y_hat,'b')
 hold off
 
  upb = max(0.95*max(max(y),max(y_hat)),1.05*max(max(y),max(y_hat)));
  lpb = min(0.95*min(min(y),min(y_hat)),1.05*min(min(y),min(y_hat)));
  axis([0 t lpb upb])
  line([t1,t1],[lpb upb]);
  legend('Actual','MSC(b) (no intercept)', 'Location', 'northwest')
  saveas(figure4,'MSC(b)_curve.eps');    % MSC(b)
  saveas(figure4,'MSC(b)_curve.pdf');
 
% Figure for the MSC(c) fitted curve
  figure5=figure;
  y_hat = y_MSC_c;
  plot(t_trend,y,'--k')
  hold on
  plot(t_trend,y_hat,'b')
  hold off
 
 upb = max(0.95*max(max(y),max(y_hat)),1.05*max(max(y),max(y_hat)));
 lpb = min(0.95*min(min(y),min(y_hat)),1.05*min(min(y),min(y_hat)));
 axis([0 t lpb upb])
 line([t1,t1],[lpb upb]);
 legend('Actual','MSC(c)', 'Location', 'northwest')
 saveas(figure5,'MSC(c)_curve.eps');
 saveas(figure5,'MSC(c)_curve.pdf');
  

  ATT_SC = mean(y2-y2_SC )    % ATT by SC
  ATT_SC_percentage = 100*ATT_SC/mean(y2_SC)  % ATT in percentage by SC

  ATT_MSC_a = mean(y2-y2_MSC_a)    % bMSC(a)
  ATT_MSC_a_percentage = 100*ATT_MSC_a/mean(y2_MSC_a)

  ATT_MSC_b = mean(y2-y2_MSC_b)  % MSC(b)
  ATT_MSC_b_percentage = 100*ATT_MSC_b/mean(y2_MSC_b)      

  ATT_MSC_c = mean(y2-y2_MSC_c )   % MSC(c) 
  ATT_MSC_c_percentage = 100*ATT_MSC_c/mean(y2_MSC_c)
    
 % Below are p-values
 pJ = mean( J_test < Js_test)  % p-value for joint hypothesis H0. If fail to reject, use Original SC in Step 2. If reject, then look at p1 
 p1=mean(test1<test1_s)  % p-value for single restriciton hypothesis test of sum to one H0a. If fail to reject, use MSCa in Step 2. If reject, then look at p2
 p2=mean(test2<test2_s)  % p-value for single restriciton hypothesis test of zero intercept H0b. It fail to reject, use MSCb in step 2. Otherwise, use MSC in step 2. 


 diary off
 toc
 elapsedTime = toc; 
      