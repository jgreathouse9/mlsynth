
# rm(list=ls())

fn_W = function(Zi,ZJ,V){
  require("quadprog")
  J = nrow(ZJ)
  Dmat = ZJ %*% V %*% t(ZJ)+ (10^-7) * diag(J)
  dvec = ZJ %*% V %*% Zi
  Amat = cbind(cbind(rep(1,J)),diag(1,J,J))
  bvec = c(1,rep(0,J))
  W = solve.QP(Dmat, dvec, Amat, bvec, meq = 1)$solution
  return(W)
}

rmse = function(x){sqrt(mean(x^2))}

load("data_COVID/data.RData")


########################## data analysis ##########################

start_date = as.Date('2019-01-01')
day_back = as.Date("2019-10-01")
day_treat_vec = c()
day_treat_vec[1] = as.Date("2020-03-28")
day_treat_vec[2] = as.Date("2020-02-15")
day_treat_vec[3] = as.Date("2020-02-15")

codes0 = c('AUT','BEL','BGR','HRV','CZE','DNK','EST','FIN','FRA',
          'DEU','GRC','HUN','IRL','ITA','LVA','LTU','NLD','NOR',
          'POL','PRT','ROU','SVK','SVN','ESP','SWE','CHE','GBR')

locations = c("Austria","Belgium","Bulgaria","Croatia","Czech Republic","Denmark","Estonia","Finland","France",
              "Germany","Greece","Hungary","Ireland","Italy","Latvia","Lithuania","Netherlands","Norway",
              "Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden","Switzerland","United Kingdom")

outcomes_d = c("covid_cases","covid_deaths")
outcomes_w = c("deaths_M_TOTAL","deaths_M_Y_LT60","deaths_M_Y60-69","deaths_M_Y70-79","deaths_M_Y80-89","deaths_M_Y_GE90",
               "deaths_F_TOTAL","deaths_F_Y_LT60","deaths_F_Y60-69","deaths_F_Y70-79","deaths_F_Y80-89","deaths_F_Y_GE90",
               "deaths_TOTAL")
outcomes_m = c('industrial',"retail_both","retail_food","retail_nfood",'import','export','CPI',
               'CPI1','CPI2','CPI3','CPI4','CPI5','CPI6','CPI7','CPI8','CPI9','CPI10','CPI11','CPI12')
outcomes_q = c('gdp_a','gdp_c','gdp_g','gdp_i','gdp_x','gdp_m',
               'labour_employ','labour_employ_male','labour_employ_female',
               'labour_employ_young','labour_employ_middle','labour_employ_old',
               'labour_employ_edu1','labour_employ_edu2','labour_employ_edu3',
               'labour_employ_manage', 'labour_employ_prof', 'labour_employ_tech', 
               'labour_employ_clerk', 'labour_employ_sales', 'labour_employ_worker', 
               'labour_employ_trader', 'labour_employ_machine', 'labour_employ_element', 
               'labour_hours','labour_hours_male','labour_hours_female',
               'labour_absence','labour_absence_male','labour_absence_female',
               'labour_absence_layoff','labour_absence_holiday','labour_absence_ill','labour_absence_other')

outcomes = c()
ylabs =  c()
titles = c()

# public health outcomes
outcomes[[1]] = c("covid_cases","covid_deaths","deaths_TOTAL")
ylabs[[1]] = rep("per million",3)
titles[[1]] = c("Cumulative COVID-19 Cases","Cumulative COVID-19 Deaths","All Deaths")

# labour market outcomes
outcomes[[2]] = c("labour_employ",'labour_absence','labour_hours')
ylabs[[2]] = c('% of population','% of employment','index, 2006=100')
titles[[2]] = c('Employment','Absence From Work','Total Hours Worked')

# economic outcomes
outcomes[[3]] = c("gdp_a",'import','export',"industrial","retail_both","CPI")
ylabs[[3]] = c('per capita, 2015 EURO','per capita, 2015 EURO','per capita, 2015 EURO',
               'index, 2015=100','index, 2015=100','index, 2015=100')
titles[[3]] = c('GDP','Import','Export','Industrial Production','Retail','CPI')

titles_agg = c('Public Health Outcomes','Labour Market Outcomes','Economic Outcomes')


W_SWE = c()
radar0 = radar0_ND = radar0_BD = radar0_SG = radar0_LOO = radar0_LOOO = c()
radar1 = radar1_ND = radar1_BD = radar2_BD =radar1_SG = radar1_LOO = radar1_LOOO = c()
gap_list = gap_percent_list = vector('list',length(unlist(outcomes)))
names(gap_list) = names(gap_percent_list) = unlist(outcomes)

cex.lab = 1.3
cex.axis = 1.4
cex.main = 1.6

for (s in 1:3) {
  
  day_treat = as.Date(day_treat_vec[s],origin = "1970-01-01")
  
  if (s==1) {
    codes = setdiff(codes0,'IRL')
  } else if (s==2) {
    codes = setdiff(codes0,'DEU')
  } else {
    codes = codes0
  }
  N = length(codes)
  
  
  # backdate, no demean, demean
  W = c()
  for (t in 1:3) {
    
    Z = c()
    for (i in codes) {
      values = data[data$code==i & data$date>=start_date & data$date<=c(day_back,day_treat,day_treat)[t], outcomes[[s]]]
      if (t!=2) {
        for (col in 1:ncol(values)) {
          values[,col] = values[,col]-mean(values[,col],na.rm=T)
        }
      }
      Z = rbind(Z,unlist(values))
    }
    Z = Z[,!apply(Z, 2, anyNA)]
    Z = Z[,apply(Z, 2, sd)!=0]
    Z = scale(Z, center = F, scale = apply(Z, 2, sd))
    
    V = c()
    for (l in outcomes[[s]]) {
      count = sum(sapply(colnames(Z),substr,1,nchar(l))==l)
      V = c(V, rep(1/count,count))
    }
    
    W = cbind(W,fn_W(cbind(Z[codes=='SWE',]),Z[codes!='SWE',],diag(V,ncol(Z))))
    
  }
  
  W_back = W[,1]
  W_ND = W[,2]
  W_SWE[[s]] = W[,3]
  
  # permutation weights
  W_mat = c()
  for(i in setdiff(codes,'SWE')){
    W_mat = cbind(W_mat,fn_W(cbind(Z[codes==i,]),Z[codes!=i,],diag(V,ncol(Z))))
  }
  
  
  # leave-one-out weights
  LOO = setdiff(codes,'SWE')[round(W_SWE[[s]],2)>=.01]
  W_LOO = matrix(0,N-1,length(LOO))
  for(k in LOO){
    W_LOO[setdiff(codes,'SWE')!=k,which(LOO==k)] = fn_W(cbind(Z[codes=='SWE',]),Z[!codes%in%c('SWE',k),],diag(V,ncol(Z)))
  }
  
  
  preloss_agg = c()
  postloss_agg = c()
  gaps_agg = gap_std_agg = vector("list", length(outcomes[[s]]))
  
  for (y in outcomes[[s]]) {
    
    Y = c()
    for (i in codes) {
      Y = rbind(Y,c(unlist(data[data$code==i & data$date>=start_date, y])))
    }
    
    dates = unique(data$date[data$date>=start_date])[!apply(Y, 2, anyNA)]
    
    Y = cbind(Y[,!apply(Y, 2, anyNA)])
    Yi = Y[codes=='SWE',]
    YJ = Y[codes!='SWE',]
    
    ymin = min(Y)
    ymax = max(Y)
    ylim = c(ymin-(ymax-ymin)/5,ymax)
    

    if (y%in%c(outcomes_d,outcomes_w)) {
      day_treat_fig = as.Date("2020-03-28")
    } else if (y%in%c(outcomes_m)) {
      day_treat_fig = as.Date("2020-03-01")
    } else if (y%in%c(outcomes_q)) {
      day_treat_fig = as.Date("2020-01-01")
    }
    
    if (y%in%c(outcomes_d)) {
      tics = as.Date(paste0("2020-",1:10,"-1"))
      tics1 = as.Date(paste0("2020-",1:9,"-15"))
      labs1 = c(1:9)
      tics2 = as.Date(c("2019-1-1"))
    } else if (y%in%c(outcomes_w,outcomes_m)) {
      tics = as.Date(c(paste0("2019-",1:12,"-1"),paste0("2020-",1:10,"-1")))
      tics1 = as.Date(c(paste0("2019-",c(3,6,9,12),"-15"),paste0("2020-",c(3,6,9),"-15")))
      labs1 = c(c(3,6,9,12),c(3,6,9))
      tics2 = as.Date(c("2020-1-1"))
    } else if (y%in%c(outcomes_q)) {
      tics = as.Date(c(paste0("2019-",c(1,4,7,10),"-1"),paste0("2020-",c(1,4,7,10),"-1")))
      tics1 = as.Date(c(paste0("2019-",c(2,5,8,11),"-16"),paste0("2020-",c(2,5,8),"-16")))
      labs1 = c('Q1','Q2','Q3','Q4','Q1','Q2','Q3')
      tics2 = as.Date(c("2020-1-1"))
    }
    
    # descriptive figures
    pdf(file = paste0("./Output/des_",y,".pdf"), width = 5.5, height = 5, family = "Times",pointsize = 12)
    plot(dates,Yi,
         type='l',lwd=2,col="black",
         ylim=ylim, xaxt = "n",
         xlab ="", ylab = ylabs[[s]][which(outcomes[[s]]==y)],cex.lab=cex.lab,cex.axis=cex.axis)
    axis(side=1, at=tics, labels=F, cex.axis=cex.axis)
    axis(side=1, at=tics1, labels=labs1, tick=F, cex.axis=cex.axis)
    axis(side=1, at=tics2, labels=F, tck=-0.04, cex.axis=cex.axis)
    for (i in 1:(N-1)) {
      lines(dates,YJ[i,],col="gray")
    }
    lines(dates,Yi,lwd=2,col="black")
    abline(v=day_treat_fig,lty="dotted")
    title(titles[[s]][which(outcomes[[s]]==y)], line=-20, cex.main=cex.main)
    dev.off()
    
    
    # synthetic control
    synthY1 = t(YJ)%*%W_SWE[[s]]
    synthY1 = synthY1-mean(synthY1[dates<=day_treat])+mean(Yi[dates<=day_treat])
    
    pdf(file = paste0("./Output/SC_",y,".pdf"), width = 5.5, height = 5, family = "Times",pointsize = 12)
    plot(dates, Yi, type='l',col="black",lwd=2, xaxt = "n",
         ylim=ylim, xlab="", ylab=ylabs[[s]][which(outcomes[[s]]==y)], cex.lab=cex.lab, cex.axis=cex.axis)
    axis(side=1, at=tics, labels=F, cex.axis=cex.axis)
    axis(side=1, at=tics1, labels=labs1, tick=F, cex.axis=cex.axis)
    axis(side=1, at=tics2, labels=F, tck=-0.04, cex.axis=cex.axis)
    title(titles[[s]][which(outcomes[[s]]==y)], line=-20, cex.main=cex.main)
    lines(dates, synthY1, lty="dashed",col="black",lwd=2)
    abline(v=day_treat_fig,lty="dotted")
    dev.off()
    
    radar0 = cbind(radar0,rbind(ymax,ymin,mean(Yi[dates<=day_treat]),mean(synthY1[dates<=day_treat])))
    radar1 = cbind(radar1,rbind(ymax,ymin,mean(Yi[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')]),mean(synthY1[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')])))
    
    
    # SC using single outcomes
    Z = Y[,dates<=day_treat]
    Z = Z[,!apply(Z, 2, anyNA)]
    Z = Z[,apply(Z, 2, sd)!=0]
    for (i in 1:N) {
      Z[i,] = Z[i,]-mean(Z[i,])
    }
    Z = scale(Z, center = F, scale = apply(Z, 2, sd))
    V = rep(1/ncol(Z),ncol(Z))
    W = fn_W(cbind(Z[codes=='SWE',]),Z[codes!='SWE',],diag(V,ncol(Z)))
    cbind(setdiff(codes,'SWE'),round(W,2))
    synthSG = t(YJ)%*%W
    synthSG = synthSG-mean(synthSG[dates<=day_treat])+mean(Yi[dates<=day_treat])
    
    pdf(file = paste0("./Output/SC_SG_",y,".pdf"), width = 5.5, height = 5, family = "Times",pointsize = 12)
    plot(dates, Yi, type='l',col="black",lwd=2, xaxt = "n",
         ylim=ylim, xlab="", ylab=ylabs[[s]][which(outcomes[[s]]==y)], cex.lab=cex.lab, cex.axis=cex.axis)
    axis(side=1, at=tics, labels=F, cex.axis=cex.axis)
    axis(side=1, at=tics1, labels=labs1, tick=F, cex.axis=cex.axis)
    axis(side=1, at=tics2, labels=F, tck=-0.04, cex.axis=cex.axis)
    title(titles[[s]][which(outcomes[[s]]==y)], line=-20, cex.main=cex.main)
    lines(dates, synthSG, lty="dashed",col="black",lwd=2)
    abline(v=day_treat_fig,lty="dotted")
    dev.off()
    
    radar0_SG = cbind(radar0_SG,rbind(ymax,ymin,mean(Yi[dates<=day_treat]),mean(synthY1[dates<=day_treat]),mean(synthSG[dates<=day_treat])))
    radar1_SG = cbind(radar1_SG,rbind(ymax,ymin,mean(Yi[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')]),
                                                mean(synthY1[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')]),
                                                mean(synthSG[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')])))
    
    
    # no demean
    synthND = t(YJ)%*%W_ND
    
    pdf(file = paste0("./Output/SC_ND_",y,".pdf"), width = 5.5, height = 5, family = "Times",pointsize = 12)
    plot(dates, Yi, type='l',col="black",lwd=2, xaxt = "n",
         ylim=ylim, xlab="", ylab=ylabs[[s]][which(outcomes[[s]]==y)], cex.lab=cex.lab, cex.axis=cex.axis)
    axis(side=1, at=tics, labels=F, cex.axis=cex.axis)
    axis(side=1, at=tics1, labels=labs1, tick=F, cex.axis=cex.axis)
    axis(side=1, at=tics2, labels=F, tck=-0.04, cex.axis=cex.axis)
    title(titles[[s]][which(outcomes[[s]]==y)], line=-20, cex.main=cex.main)
    lines(dates, synthND, lty="dashed",col="black",lwd=2)
    abline(v=day_treat_fig,lty="dotted")
    dev.off()
    
    radar0_ND = cbind(radar0_ND,rbind(ymax,ymin,mean(Yi[dates<=day_treat]),mean(synthY1[dates<=day_treat]),mean(synthND[dates<=day_treat])))
    radar1_ND = cbind(radar1_ND,rbind(ymax,ymin,mean(Yi[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')]),
                                                mean(synthY1[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')]),
                                                mean(synthND[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')])))
    
    
    # backdating
    synthBD = t(YJ)%*%W_back
    if (sum(dates<=day_back)>1) {
      synthBD = synthBD-mean(synthBD[dates<=day_back])+mean(Yi[dates<=day_back])
    }
    
    pdf(file = paste0("./Output/SC_BD_",y,".pdf"), width = 5.5, height = 5, family = "Times",pointsize = 12)
    plot(dates, Yi, type='l',col="black",lwd=2, xaxt = "n",
         ylim=ylim, xlab="", ylab=ylabs[[s]][which(outcomes[[s]]==y)], cex.lab=cex.lab, cex.axis=cex.axis)
    axis(side=1, at=tics, labels=F, cex.axis=cex.axis)
    axis(side=1, at=tics1, labels=labs1, tick=F, cex.axis=cex.axis)
    axis(side=1, at=tics2, labels=F, tck=-0.04, cex.axis=cex.axis)
    title(titles[[s]][which(outcomes[[s]]==y)], line=-20, cex.main=cex.main)
    lines(dates, synthBD, lty="dashed",col="black",lwd=2)
    abline(v=day_back,lty="dotted")
    dev.off()
    
    radar0_BD = cbind(radar0_BD,rbind(ymax,ymin,mean(Yi[dates<=day_back]),mean(synthY1[dates<=day_back]),mean(synthBD[dates<=day_back])))
    radar1_BD = cbind(radar1_BD,rbind(ymax,ymin,mean(Yi[dates>=as.Date('2019-10-01')&dates<as.Date('2020-01-01')]),
                                                mean(synthY1[dates>=as.Date('2019-10-01')&dates<as.Date('2020-01-01')]),
                                                mean(synthBD[dates>=as.Date('2019-10-01')&dates<as.Date('2020-01-01')])))
    radar2_BD = cbind(radar2_BD,rbind(ymax,ymin,mean(Yi[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')]),
                                                mean(synthY1[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')]),
                                                mean(synthBD[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')])))

    
    # leave-one-unit-out distribution
    pdf(file = paste0("./Output/SC_LOO_",y,".pdf"), width = 5.5, height = 5, family = "Times",pointsize = 12)
    plot(dates, Yi, type='l',col="black",lwd=2, xaxt = "n",
         ylim=ylim, xlab="", ylab=ylabs[[s]][which(outcomes[[s]]==y)], cex.lab=cex.lab, cex.axis=cex.axis)
    axis(side=1, at=tics, labels=F, cex.axis=cex.axis)
    axis(side=1, at=tics1, labels=labs1, tick=F, cex.axis=cex.axis)
    axis(side=1, at=tics2, labels=F, tck=-0.04, cex.axis=cex.axis)
    title(titles[[s]][which(outcomes[[s]]==y)], line=-20, cex.main=cex.main)
    lines(dates, synthY1, lty="dashed",col="black",lwd=2)
    radar0_LOO_c = c()
    radar1_LOO_c = c()
    for (k in LOO) {
      synthLOO = t(YJ)%*%W_LOO[,which(LOO==k)]
      synthLOO = synthLOO-mean(synthLOO[dates<=day_treat])+mean(Yi[dates<=day_treat])
      radar0_LOO_c = rbind(radar0_LOO_c,mean(synthLOO[dates<=day_treat]))
      radar1_LOO_c = rbind(radar1_LOO_c,mean(synthLOO[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')]))
      lines(dates, synthLOO, lty="solid",col="gray",lwd=2)
    }
    lines(dates, Yi, lty="solid",col="black",lwd=2)
    lines(dates, synthY1, lty="dashed",col="black",lwd=2)
    abline(v=day_treat_fig,lty="dotted")
    dev.off()
    
    radar0_LOO = cbind(radar0_LOO,rbind(ymax,ymin,mean(Yi[dates<=day_treat]),mean(synthY1[dates<=day_treat]),
                                        min(radar0_LOO_c),max(radar0_LOO_c)))
    radar1_LOO = cbind(radar1_LOO,rbind(ymax,ymin,
                                        mean(Yi[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')]),
                                        mean(synthY1[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')]),
                                        min(radar1_LOO_c),max(radar1_LOO_c)))
    
    
    # leave-one-outcome-out distribution
    pdf(file = paste0("./Output/SC_LOOO_",y,".pdf"), width = 5.5, height = 5, family = "Times",pointsize = 12)
    plot(dates, Yi, type='l',col="black",lwd=2, xaxt = "n",
         ylim=ylim, xlab="", ylab=ylabs[[s]][which(outcomes[[s]]==y)], cex.lab=cex.lab, cex.axis=cex.axis)
    axis(side=1, at=tics, labels=F, cex.axis=cex.axis)
    axis(side=1, at=tics1, labels=labs1, tick=F, cex.axis=cex.axis)
    axis(side=1, at=tics2, labels=F, tck=-0.04, cex.axis=cex.axis)
    title(titles[[s]][which(outcomes[[s]]==y)], line=-20, cex.main=cex.main)
    lines(dates, synthY1, lty="dashed",col="black",lwd=2)
    radar0_LOOO_c = c()
    radar1_LOOO_c = c()
    for (k in 1:length(outcomes[[s]])) {
      Z = c()
      for (i in codes) {
        values = data[data$code==i & data$date>=start_date & data$date<=day_treat, outcomes[[s]][-k]]
        for (col in 1:ncol(values)) {
          values[,col] = values[,col]-mean(values[,col],na.rm=T)
        }
        Z = rbind(Z,unlist(values))
      }
      Z = Z[,!apply(Z, 2, anyNA)]
      Z = Z[,apply(Z, 2, sd)!=0]
      Z = scale(Z, center = F, scale = apply(Z, 2, sd))
      
      V = c()
      for (l in outcomes[[s]][-k]) {
        count = sum(sapply(colnames(Z),substr,1,nchar(l))==l)
        V = c(V, rep(1/count,count))
      }
      
      W = fn_W(cbind(Z[codes=='SWE',]),Z[codes!='SWE',],diag(V,ncol(Z)))
      synthLOOO = t(YJ)%*%W
      synthLOOO = synthLOOO-mean(synthLOOO[dates<=day_treat])+mean(Yi[dates<=day_treat])
      radar0_LOOO_c = rbind(radar0_LOOO_c,mean(synthLOOO[dates<=day_treat]))
      radar1_LOOO_c = rbind(radar1_LOOO_c,mean(synthLOOO[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')]))
      lines(dates, synthLOOO, lty="solid",col="gray",lwd=2)
    }
    lines(dates, Yi, lty="solid",col="black",lwd=2)
    lines(dates, synthY1, lty="dashed",col="black",lwd=2)
    abline(v=day_treat_fig,lty="dotted")
    dev.off()
    
    radar0_LOOO = cbind(radar0_LOOO,rbind(ymax,ymin,mean(Yi[dates<=day_treat]),mean(synthY1[dates<=day_treat]),
                                        min(radar0_LOOO_c),max(radar0_LOOO_c)))
    radar1_LOOO = cbind(radar1_LOOO,rbind(ymax,ymin,
                                        mean(Yi[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')]),
                                        mean(synthY1[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')]),
                                        min(radar1_LOOO_c),max(radar1_LOOO_c)))
    
    
    # gaps in outcomes
    gaps = c(synthY1-Yi)
    for(k in setdiff(codes,'SWE')){
      sY1 = t(Y[codes!=k,])%*%W_mat[,setdiff(codes,'SWE')==k]
      sY1 = sY1-mean(sY1[dates<=day_treat])+mean(Y[codes==k,][dates<=day_treat])
      gaps = cbind(gaps,sY1-Y[codes==k,])
    }
    colnames(gaps) = c("SWE*",setdiff(codes,'SWE'))
    rownames(gaps) = dates
    
    if (y%in%c('labour_absence')) {gaps = -gaps}
    
    pdf(file = paste0("./Output/SC_gap_",y,".pdf"), width = 5.5, height = 5, family = "Times",pointsize = 12)
    plot(dates, gaps[,1], type='l',col="black",lwd=2, xaxt = "n",
         ylim=c(min(gaps),max(gaps)), xlab="", ylab="gap", cex.lab=cex.lab, cex.axis=cex.axis)
    axis(side=1, at=tics, labels=F, cex.axis=cex.axis)
    axis(side=1, at=tics1, labels=labs1, tick=F, cex.axis=cex.axis)
    axis(side=1, at=tics2, labels=F, tck=-0.04, cex.axis=cex.axis)
    title(titles[[s]][which(outcomes[[s]]==y)], line=-20, cex.main=cex.main)
    for (i in 2:ncol(gaps)) {
      lines(dates, gaps[,i], col="gray")
    }
    lines(dates, gaps[,1], col="black",lwd=2)
    abline(v=day_treat_fig,lty="dotted")
    abline(h=0,lty="dashed")
    dev.off()

    
    T0 = length(dates[dates<=day_treat])
    TT = length(dates)
    
    gap1 = -gaps[,1][dates>day_treat]
    gap_std = gap1/mean(apply(Y[,dates>day_treat], 2, sd))
    gap_percent = gap1/Yi[dates>day_treat]*100
    names(gap1) = names(gap_percent) = dates[dates>day_treat]
    gap_list[[which(unlist(outcomes)==y)]] = gap1
    gap_std_agg[[which(outcomes[[s]]==y)]] = gap_std
    gap_percent_list[[which(unlist(outcomes)==y)]] = gap_percent

    
    gaps[(T0+1):TT,][gaps[(T0+1):TT,]>0] = 0 # one-sided inference
    gaps[(T0+1):TT,] = abs(gaps[(T0+1):TT,])
    gaps = gaps/mean(apply(Y[,dates>day_treat], 2, sd))
    gaps_agg[[which(outcomes[[s]]==y)]] = gaps[dates>day_treat,]

    postloss = apply(cbind(gaps[(T0+1):TT,]),2,rmse)
    preloss = apply(cbind(gaps[1:T0,]),2,rmse)
    eta = 0.01
    postloss = postloss + eta
    preloss = preloss + eta
    ratio = sort(postloss/preloss)
    
    postloss_agg = cbind(postloss_agg,postloss)
    preloss_agg = cbind(preloss_agg,preloss)
    pvalue = 1-(which(names(ratio)=='SWE*')-1)/N
    
    pdf(file = paste0("./Output/SC_ratio_",y,".pdf"), width = 5.5, height = 5, family = "Times", pointsize = 12)
    dotchart(ratio, cex.lab=cex.lab, cex.axis=cex.axis, pch=19)
    title(paste0(titles[[s]][which(outcomes[[s]]==y)],' (',round(pvalue,2),')'), line=-20, cex.main=cex.main)
    dev.off()
    
    
    if (y%in%c(outcomes_d,outcomes_w,outcomes_m)) {
      tics = as.Date(paste0("2020-",1:10,"-1"))
      tics1 = as.Date(paste0("2020-",1:9,"-15"))
      labs1 = c(1:9)
    } else if (y%in%c(outcomes_q)) {
      tics = as.Date(paste0("2020-",c(1,4,7,10),"-1"))
      tics1 = as.Date(paste0("2020-",c(2,5,8),"-16"))
      labs1 = c('Q1','Q2','Q3')
    }
    
    
    # p values for each posttreatment period
    pvalues = c()
    for (t in (T0+1):TT) {
      ratio = sort((gaps[t,] + eta)/preloss)
      pvalues = c(pvalues,1-(which(names(ratio)=='SWE*')-1)/N)
    }
    
    pdf(file = paste0("./Output/SC_pvalue_",y,".pdf"), width = 5.5, height = 5, family = "Times",pointsize = 12)
    plot(dates[(T0+1):(TT)],pvalues,type='o', col="black", xaxt = "n",
         ylim=c(0,1), ylab='P-Value', xlab="", cex.lab=cex.lab, cex.axis=cex.axis)
    axis(side=1, at=tics, labels=F, cex.axis=cex.axis)
    axis(side=1, at=tics1, labels=labs1, tick=F, cex.axis=cex.axis)
    abline(h=3/N,lty="dashed")
    title(paste0(titles[[s]][which(outcomes[[s]]==y)],' (',round(pvalue,2),')'), line=-20, cex.main=cex.main)
    dev.off()
    
  }
  
  
  # Aggregate P values for each posttreatment period
  if (s==1) {
    dates = rownames(gaps_agg[[which(outcomes[[s]]=='deaths_TOTAL')]])
  } else {
    dates = as.numeric(as.Date(paste0("2020-",c(3,6,9),"-16")))
  }
  dates1 = c(day_treat,as.Date(as.numeric(dates),origin = "1970-01-01"))
  
  taus = c()
  pvalues = c()
  for (t in 2:length(dates1)) {
    gap1s = c()
    for (i in 1:length(outcomes[[s]])) {
      gap1s = c(gap1s,mean(gap_std_agg[[i]][names(gap_std_agg[[i]])%in%c(dates1[t-1]:dates1[t])]))
    }
    taus = c(taus,mean(gap1s))
    
    gaps = c()
    for (i in 1:length(outcomes[[s]])) {
      gaps = cbind(gaps,colMeans(rbind(gaps_agg[[i]][rownames(gaps_agg[[i]])%in%c(dates1[t-1]:dates1[t]),])))
    }
    ratio = sort(rowMeans(gaps)/rowMeans(preloss_agg))
    pvalues = c(pvalues,1-(which(names(ratio)=='SWE*')-1)/N)
  }
  ratio = sort(rowMeans(postloss_agg)/rowMeans(preloss_agg))
  pvalue = 1-(which(names(ratio)=='SWE*')-1)/N

  if (s==1) {
    tics = as.Date(paste0("2020-",1:10,"-1"))
    tics1 = as.Date(paste0("2020-",1:9,"-15"))
    labs1 = c(1:9)
  } else {
    dates = as.Date(paste0("2020-",c(2,5,8),"-16"))
    tics = as.Date(paste0("2020-",c(1,4,7,10),"-1"))
    tics1 = as.Date(paste0("2020-",c(2,5,8),"-16"))
    labs1 = c('Q1','Q2','Q3')
  }
  
  pdf(file = paste0("./Output/SC_tau_agg",s,".pdf"), width = 5.5, height = 5, family = "Times",pointsize = 12)
  plot(dates,taus,type='o', col="black", xaxt = "n",
       ylab='Aggregate Treatment Effect', xlab="",cex.lab=cex.lab, cex.axis=cex.axis)
  axis(side=1, at=tics, labels=F, cex.axis=cex.axis)
  axis(side=1, at=tics1, labels=labs1, tick=F, cex.axis=cex.axis)
  title(paste0(titles_agg[[s]],' (',round(mean(taus),2),')'), line=-20, cex.main=cex.main)
  dev.off()
  
  pdf(file = paste0("./Output/SC_pvalue_agg",s,".pdf"), width = 5.5, height = 5, family = "Times",pointsize = 12)
  plot(dates,pvalues,type='o', col="black", xaxt = "n",
       ylim=c(0,1), ylab='P-Value', xlab="",cex.lab=cex.lab, cex.axis=cex.axis)
  axis(side=1, at=tics, labels=F, cex.axis=cex.axis)
  axis(side=1, at=tics1, labels=labs1, tick=F, cex.axis=cex.axis)
  title(paste0(titles_agg[[s]],' (',round(pvalue,2),')'), line=-20, cex.main=cex.main)
  abline(h=3/N,lty="dashed")
  dev.off()
  
  pdf(file = paste0("./Output/SC_ratio_agg",s,".pdf"), width = 5.5, height = 5, family = "Times",pointsize = 12)
  dotchart(ratio,cex.lab=cex.lab,cex.axis=cex.axis,pch=19)
  title(paste0(c('Public Health Outcomes','Labour Market Outcomes','Economic Outcomes')[s],' (',round(pvalue,2),')'), line=-20, cex.main=cex.main)
  dev.off()
  
} # close loop of outcome lists

for (i in 1:2) {
  print(c(names(gap_list)[i], gap_list[[i]]['2020-07-26'], gap_percent_list[[i]]['2020-07-26']))
}
mean(gap_percent_list[[3]][names(gap_percent_list[[3]])>"2020-04-12" & names(gap_percent_list[[3]])<="2020-05-17"])
sum(gap_list[[3]][names(gap_list[[3]])<="2020-07-26"])

for (i in 4:6) {
  print(c(names(gap_list)[i], gap_list[[i]]['2020-05-16'], gap_percent_list[[i]]['2020-05-16']))
}
print(c(names(gap_list)[11], gap_list[[11]][1:3], gap_percent_list[[11]][1:3]))




library(fmsb)
radar0_BD[is.na(radar0_BD)] = 0
radar1_BD[is.na(radar1_BD)] = 0
for (df in c('radar0','radar0_ND','radar0_BD','radar0_SG','radar0_LOO','radar0_LOOO','radar1','radar1_ND','radar1_BD','radar2_BD','radar1_SG','radar1_LOO','radar1_LOOO')) {
  assign(df, as.data.frame(get(df)))
  assign(df, setNames(get(df), c("COVID-19 Cases","COVID-19 Deaths","All Deaths","Employment","Absence From Work","Total Hours Worked",
                                  "GDP","Import","Export","Industrial Production","Retail","CPI")))
  assign(df, get(df)[,c("COVID-19 Cases","COVID-19 Deaths","All Deaths",
                        "GDP","Import","Export","Industrial Production","Retail","CPI",
                        "Employment","Absence From Work","Total Hours Worked")])
}

for (df in c('','_ND','_BD','_SG','_LOO','_LOOO')) {
  pdf(file = paste0("./Output/SC_radar",df,".pdf"), width = 10+5*(df=='_BD'), height = 5, family = "Times",pointsize = 12)
  if (df=='_BD') {
    layout(mat = matrix(c(1,2,3,4,4,4),nrow=2,byrow=T),heights = c(6,1))
    for (i in 0:2){
      par(mar = c(0,.5,1,.5))
      radarchart(get(paste0('radar',i,df)),cglcol='black',vlcex=1.7,
                 pcol=c('black','black','red'),
                 plty=c("solid","dashed","solid"),
                 plwd=c(2,2,1))
    }
  } else {
    layout(mat = matrix(c(1,2,3,3),nrow=2,byrow=T),heights = c(6,1))
    for (i in 0:1){
      par(mar = c(0,.5,1,.5))
      radarchart(get(paste0('radar',i,df)),cglcol='black',vlcex=1.3,
                 pcol=c('black','black','red','red'),
                 plty=c("solid","dashed","solid","solid"),
                 plwd=c(2,2,2,2))
    }
  }
  par(mar = c(0,0,0,0))
  plot(1, type = "n", axes=FALSE, xlab="", ylab="")
  if (df=='_BD') {
    title('Pretreatment',adj=.13,line=-2, cex.main = 2.2)
    title('Q4, 2019',adj=.5,line=-2, cex.main = 2.2)
    title('Q2, 2020',adj=.86,line=-2, cex.main = 2.2)
  } else {
    title('Pretreatment',adj=.21,line=-2, cex.main = 1.6)
    title('Q2, 2020',adj=.78,line=-2, cex.main = 1.6)
  }
  if (df%in%c('_LOO','_LOOO')) {
    legend(x=.9,y=1.5,cex=1.3,bg="white",bty='n',
           legend=c("Sweden","Synthetic Sweden (benchmark)","Synthetic Sweden (LOO min and max)"),
           lty=c("solid","dashed","solid"),
           lwd=c(2,2,2,2),
           col=c("black","black","red"))
  } else if (df%in%c('_BD')) {
    legend(x=1.1,y=1.5,cex=1.4,bg="white",bty='n',
           legend=c("Sweden","Synthetic Sweden (benchmark)","Synthetic Sweden"),
           lty=c("solid","dashed","solid"),
           lwd=c(2,2,2),
           col=c("black","black","red"))
  } else if (df%in%c('_ND','_SG')) {
    legend(x=.9,y=1.5,cex=1.3,bg="white",bty='n',
           legend=c("Sweden","Synthetic Sweden (benchmark)","Synthetic Sweden"),
           lty=c("solid","dashed","solid"),
           lwd=c(2,2,2),
           col=c("black","black","red"))
  } else if (df=='') {
    legend(x=.92,y=1.5,cex=1.3,bg="white",bty='n',
           legend=c("Sweden","Synthetic Sweden"),
           lty=c("solid","dashed"),
           lwd=c(2,2),
           col=c("black","black"))
  }
  dev.off()
}

codes = codes0
N = length(codes)
tab_weight = matrix(NA,N-1,3)
tab_weight[setdiff(codes,'SWE')!='IRL',1] = W_SWE[[1]]
tab_weight[setdiff(codes,'SWE')!='DEU',2] = W_SWE[[2]]
tab_weight[,3] = W_SWE[[3]]
tab_weight = cbind(locations[locations!='Sweden'],round(tab_weight,2))
tab_weight[is.na(tab_weight)] = '--'
colnames(tab_weight) = c("Country","Health","Labor","Economic")
tab_weight = cbind(tab_weight[1:((N-1)/2),],tab_weight[((N-1)/2+1):(N-1),])
library(xtable)
tab_weight = xtable(tab_weight)
print(tab_weight, include.rownames=F)
print(tab_weight, include.rownames=FALSE, file="./tab_weights.txt")




# further analysis

outcomes = c()
ylabs =  c()
titles = c()

# deaths by age groups
outcomes[[1]] = c("deaths_M_TOTAL","deaths_M_Y_LT60","deaths_M_Y60-69","deaths_M_Y70-79","deaths_M_Y80-89","deaths_M_Y_GE90",
                  "deaths_F_TOTAL","deaths_F_Y_LT60","deaths_F_Y60-69","deaths_F_Y70-79","deaths_F_Y80-89","deaths_F_Y_GE90")
ylabs[[1]] = rep("per million",12)
titles[[1]] = rep(c("Total","Less than 60","60-69","70-79","80-89","90 or over"),2)

# labour market outcomes
outcomes[[2]] = c('labour_employ_manage', 'labour_employ_prof', 'labour_employ_tech',
                  'labour_employ_clerk', 'labour_employ_sales', 'labour_employ_worker',
                  'labour_employ_trader', 'labour_employ_machine', 'labour_employ_element',
                  'labour_employ_edu1','labour_employ_edu2','labour_employ_edu3',
                  'labour_employ_young','labour_employ_middle','labour_employ_old',
                  'labour_employ','labour_employ_male','labour_employ_female',
                  'labour_hours','labour_hours_male','labour_hours_female',
                  'labour_absence','labour_absence_male','labour_absence_female',
                  'labour_absence_layoff','labour_absence_holiday','labour_absence_ill','labour_absence_other')
ylabs[[2]] = c(rep('% of population',18),rep('index, 2006=100',3),rep('% of employment',7))
titles[[2]] = c('Manager', 'Professional', 'Technician', 'Clerk', 'Service', 'Skilled Worker', 'Trades Worker', 'Factory Worker', 'Elementary',
                'Primary and Lower Secondary', 'Upper Secondary', 'Tertiary',
                'Young','Middle','Old',
                'Total','Male','Female','Total','Male','Female','Total','Male','Female',
                'Layoff', 'Holiday', 'Illness', 'Other')

# economic outcomes
outcomes[[3]] = c("retail_both",'retail_food', 'retail_nfood',
                  'CPI1','CPI2','CPI3','CPI4','CPI5','CPI6','CPI7','CPI8','CPI9','CPI10','CPI11','CPI12',
                  "gdp_a","gdp_c","gdp_g","gdp_i","gdp_x","gdp_m")
ylabs[[3]] = c(rep('index, 2015=100',15),rep('per capita, 2015 EURO',6))
titles[[3]] = c('All','Food, Beverages and Tobacco','Non-food Products',
                'Food','Alcohol and Tobacco','Clothing','Utility','Household','Health','Transport',
                'Communications','Recreation','Education','Restaurants','Miscellaneous',
                'GDP','Household Consumption','Government Consumption',
                'Gross Capital Formation','Exports','Imports')


radar0 = radar1 = c()
gap_list_further = gap_percent_list_further = vector('list',length(unlist(outcomes)))
names(gap_list_further) = names(gap_percent_list_further) = unlist(outcomes)

for (s in 1:3) {

  day_treat = day_treat_vec[s]

  if (s==1) {
    codes = setdiff(codes0,'IRL')
  } else if (s==2) {
    codes = setdiff(codes0,'DEU')
  } else {
    codes = codes0
  }
  N = length(codes)

  for (y in outcomes[[s]]) {

    if (y%in%c(outcomes_d,outcomes_w)) {
      day_treat_fig = as.Date("2020-03-28")
    } else if (y%in%c(outcomes_m)) {
      day_treat_fig = as.Date("2020-03-01")
    } else if (y%in%c(outcomes_q)) {
      day_treat_fig = as.Date("2020-01-01")
    }

    if (y%in%c(outcomes_d)) {
      tics = as.Date(paste0("2020-",1:10,"-1"))
      tics1 = as.Date(paste0("2020-",1:9,"-15"))
      labs1 = c(1:9)
      tics2 = as.Date(c("2019-1-1"))
    } else if (y%in%c(outcomes_w,outcomes_m)) {
      tics = as.Date(c(paste0("2019-",1:12,"-1"),paste0("2020-",1:10,"-1")))
      tics1 = as.Date(c(paste0("2019-",c(3,6,9,12),"-15"),paste0("2020-",c(3,6,9),"-15")))
      labs1 = c(c(3,6,9,12),c(3,6,9))
      tics2 = as.Date(c("2020-1-1"))
    } else if (y%in%c(outcomes_q)) {
      tics = as.Date(c(paste0("2019-",c(1,4,7,10),"-1"),paste0("2020-",c(1,4,7,10),"-1")))
      tics1 = as.Date(c(paste0("2019-",c(2,5,8,11),"-16"),paste0("2020-",c(2,5,8),"-16")))
      labs1 = c('Q1','Q2','Q3','Q4','Q1','Q2','Q3')
      tics2 = as.Date(c("2020-1-1"))
    }


    Y = c()
    for (i in codes) {
      Y = rbind(Y,c(unlist(data[data$code==i & data$date>=start_date, y])))
    }

    if (substr(y,1,6)=="deaths") {
      Y[codes=='DEU',] = colMeans(Y,na.rm = T)
    }

    dates = unique(data$date[data$date>=start_date & data$code%in%codes])[!apply(Y, 2, anyNA)]

    Y = cbind(Y[,!apply(Y, 2, anyNA)])
    Yi = Y[codes=='SWE',]
    YJ = Y[codes!='SWE',]

    ymin = min(Y)
    ymax = max(Y)
    ylim = c(ymin-(ymax-ymin)/5,ymax)


    # synthetic control
    synthY1 = t(YJ)%*%W_SWE[[s]]
    synthY1 = synthY1-mean(synthY1[dates<=day_treat])+mean(Yi[dates<=day_treat])

    radar0 = cbind(radar0,rbind(ymax,ymin,mean(Yi[dates<=day_treat]),mean(synthY1[dates<=day_treat])))
    radar1 = cbind(radar1,rbind(ymax,ymin,mean(Yi[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')]),mean(synthY1[dates>=as.Date('2020-04-01')&dates<as.Date('2020-07-01')])))

    pdf(file = paste0("./Output/SC_further_",y,".pdf"), width = 5.5, height = 5, family = "Times",pointsize = 12)
    plot(dates, Yi, type='l',col="black",lwd=2, xaxt = "n",
         ylim=ylim, xlab="", ylab=ylabs[[s]][which(outcomes[[s]]==y)], cex.lab=cex.lab, cex.axis=cex.axis)
    axis(side=1, at=tics, labels=F, cex.axis=cex.axis)
    axis(side=1, at=tics1, labels=labs1, tick=F, cex.axis=cex.axis)
    axis(side=1, at=tics2, labels=F, tck=-0.04, cex.axis=cex.axis)
    title(titles[[s]][which(outcomes[[s]]==y)], line=-20, cex.main=cex.main)
    lines(dates, synthY1, lty="dashed",col="black",lwd=2)
    abline(v=day_treat_fig,lty="dotted")
    dev.off()


    T0 = length(dates[dates<=day_treat])
    TT = length(dates)

    gaps = Yi-synthY1
    if (substr(y,1,14)=='labour_absence') {gaps = -gaps}
    
    gap1 = gaps[,1][dates>day_treat]
    gap_percent = gap1/Yi[dates>day_treat]*100
    if (substr(y,1,6)=="deaths") {
      for (t in (T0+1):TT) {
        gap1[t-T0] = sum(gaps[(T0+1):t,1])
        gap_percent[t-T0] = gap1[t-T0]/sum(Yi[(T0+1):t])*100
      }
    }
    names(gap1) = names(gap_percent) = dates[dates>day_treat]
    gap_list_further[[which(unlist(outcomes)==y)]] = gap1
    gap_percent_list_further[[which(unlist(outcomes)==y)]] = gap_percent

  }
}

for (df in c('radar0','radar1')) {
  assign(df, as.data.frame(get(df)))
  assign(df, setNames(get(df), unlist(titles)))
  assign(paste0(df,'deaths_M'), get(df)[,substr(unlist(outcomes),1,8)=='deaths_M'])
  assign(paste0(df,'deaths_F'), get(df)[,substr(unlist(outcomes),1,8)=='deaths_F'])
  assign(paste0(df,'labour_M'), get(df)[,unlist(outcomes)%in%c("labour_employ_male","labour_absence_male","labour_hours_male")])
  assign(paste0(df,'labour_F'), get(df)[,unlist(outcomes)%in%c("labour_employ_female","labour_absence_female","labour_hours_female")])
  assign(paste0(df,'employ_occ'), get(df)[,unlist(outcomes)%in%c("labour_employ_manage","labour_employ_prof","labour_employ_tech",
                                                                 "labour_employ_clerk","labour_employ_sales","labour_employ_worker",
                                                                 "labour_employ_trader","labour_employ_machine","labour_employ_element")])
  assign(paste0(df,'employ_edu'), get(df)[,unlist(outcomes)%in%c("labour_employ_edu1","labour_employ_edu2","labour_employ_edu3")])
  assign(paste0(df,'employ_age'), get(df)[,unlist(outcomes)%in%c("labour_employ_young","labour_employ_middle","labour_employ_old")])
  assign(paste0(df,'absence'), get(df)[,unlist(outcomes)%in%c("labour_absence_layoff","labour_absence_holiday","labour_absence_ill","labour_absence_other")])
  assign(paste0(df,'retail'), get(df)[,substr(unlist(outcomes),1,6)=='retail'])
  assign(paste0(df,'CPI'), get(df)[,substr(unlist(outcomes),1,3)=='CPI'])
  assign(paste0(df,'gdp'), get(df)[,substr(unlist(outcomes),1,3)=='gdp'])
}

radar0deaths_M[1:2,] = radar0deaths_F[1:2,] = radar1deaths_M[1:2,] = radar1deaths_F[1:2,] = round(rbind(pmax(radar0deaths_M[1,],radar0deaths_F[1,]),pmin(radar0deaths_M[2,],radar0deaths_F[2,]))/2,1)
radar0deaths = rbind(radar0deaths_F,radar0deaths_M[3:4,])
radar1deaths = rbind(radar1deaths_F,radar1deaths_M[3:4,])
radar0absence[1:2,] = radar1absence[1:2,] = round(c(max(radar0absence[1,]),min(radar0absence[2,])),1)
colnames(radar0retail)[2] = colnames(radar1retail)[2] = 'Food'
radar0retail[1:2,] = radar1retail[1:2,] = round(c(max(radar0retail[3:4,],radar1retail[3:4,]),min(radar0retail[3:4,],radar1retail[3:4,])),1)
radar0labour_M[1:2,] = radar0labour_F[1:2,] = radar1labour_M[1:2,] = radar1labour_F[1:2,] = round(rbind(pmax(radar0labour_M[1,],radar0labour_F[1,]),pmin(radar0labour_M[2,],radar0labour_F[2,])),1)
colnames(radar0labour_M) = colnames(radar0labour_F) = colnames(radar1labour_M) = colnames(radar1labour_F) = c('Employment','Absence From Work','Total Hours Worked')
radar0employ_occ[1:2,] = radar1employ_occ[1:2,] = c(max(radar0employ_occ[1,]),min(radar1employ_occ[2,]))
radar0employ_edu[1:2,] = radar1employ_edu[1:2,] = c(max(radar0employ_edu[1,]),min(radar1employ_edu[2,]))
radar0employ_age[1:2,] = radar1employ_age[1:2,] = c(max(radar0employ_age[1,]),min(radar1employ_age[2,]))
radar0CPI[1:2,] = radar1CPI[1:2,] = c(max(radar0CPI[1,]),min(radar1CPI[2,]))
radar0gdp[1:2,] = radar1gdp[1:2,] = c(max(radar0gdp[3:4,],radar1gdp[3:4,]),min(radar0gdp[3:4,],radar1gdp[3:4,]))


for (df in c('deaths','absence','retail')) {
  pdf(file = paste0("./Output/SC_radar_",df,".pdf"), width = 10, height = 5, family = "Times", pointsize = 12)
  layout(mat = matrix(c(1,2,3,3),nrow=2,byrow=T),heights = c(6,1))
  par(mar = c(0,.5,0,.5))
  if (df=='deaths') {
    for (i in 0:1){
      radarchart(get(paste0('radar',i,df)),cglcol='black',vlcex=1.3,
                 axistype=2,axislabcol='black',palcex=1.1,
                 pcol=c('red','red','blue','blue'),
                 plty=c("solid","dashed","solid","dashed"),
                 plwd=c(2,2,2,2))
    }
  } else {
    for (i in 0:1){
      radarchart(get(paste0('radar',i,df)),cglcol='black',vlcex=1.3,
                 axistype=2*(df%in%c('deaths_M','deaths_F')),axislabcol='black',palcex=1.1,
                 pcol=c('black','black'),
                 plty=c("solid","dashed"),
                 plwd=c(2,2))
    }
  }
  par(mar = c(0,0,0,0))
  plot(1, type = "n", axes=FALSE, xlab="", ylab="")
  title('Pretreatment',adj=.21,line=-2, cex.main = 1.6)
  title('Q2, 2020',adj=.78,line=-2, cex.main = 1.6)
  if (df=='deaths') {
    legend(x=.88,y=1.5,cex=1.2,bg="white",bty='n',
           legend=c('Female','Male',"Sweden","Synthetic Sweden"),
           lty=c("solid","solid","solid","dashed"),
           lwd=c(2,2,2,2),
           col=c('Red','Blue',"black","black"),
           ncol=2,text.width=.06)
  } else {
    legend(x=.92,y=1.5,cex=1.2,bg="white",bty='n',
           legend=c("Sweden","Synthetic Sweden"),
           lty=c("solid","dashed"),
           lwd=c(2,2),
           col=c("black","black"))
  }
  dev.off()
}

for (i in 1:12) {
  print(c(names(gap_list_further)[i], gap_list_further[[i]]['2020-07-26']))
}
