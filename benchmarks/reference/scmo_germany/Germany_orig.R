
rm(list=ls())

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

library(foreign)
d = read.dta("Data_Germany/repgermany.dta")
# write.csv(d, file = "Data_Germany/repgermany.csv", row.names = FALSE)

Y = c()
for (i in unique(d$index)) {
  Y = rbind(Y,unlist(d[d$index==i, 'gdp']))
}


pdf(file = "./Output/Ger_fig1.pdf", width = 5.5, height = 5.5, family = "Times",pointsize = 12)
plot(1960:2003,Y[unique(d$index)==7,],
     type="l",ylim=c(0,33000),col="black",lty="solid",
     ylab ="Per-Capita GDP (PPP, 2002 USD)",
     xlab ="Year",
     xaxs = "i", yaxs = "i",
     lwd=2
)
for (i in 1:(nrow(Y)-1)) {
  lines(1960:2003,Y[-7,][i,],col="gray",lty="solid",lwd=1)
}
lines(1960:2003,colMeans(Y[-7,]),col="black",lty="dashed",lwd=2)
abline(v=1990,lty="dotted")
legend(x="bottomright",
       legend=c("West Germany","Other OECD Countries","Mean of Other OECD Countries")
       ,lty=c("solid","solid","dashed"),col=c("black","gray","black")
       ,cex=.8,bg="white",lwd=c(2,1,2))
Text.height = 23000
arrows(1987,Text.height,1989,Text.height,col="black",length=.1)
text(1982.5,Text.height,"reunification",cex=.8)
dev.off()


##### Replicating Abadie et al 2015 using 30 years of pretreatment outcomes
Z = Y[,unique(d$year)<1990]
Z = scale(Z, center = F, scale = apply(Z, 2, sd))
V = rep(1/ncol(Z),ncol(Z))
W = fn_W(cbind(Z[unique(d$index)==7,]),Z[unique(d$index)!=7,],diag(V,ncol(Z)))


##### Replicating Abadie et al 2015 using only 1989 outcomes

year = 1989

# General Stats of 16 OECD countries 1985-1990 obtained from https://stats.oecd.org/ 
library(readxl)
data_raw = read_excel("Data_Germany/all.xlsx")
data_raw = data_raw[data_raw$Year==year,c("Subject","Country","Year","Value","PowerCode","Unit")]
var_names = unique(data_raw$Subject)
country_names = unique(d$country)
data_raw[data_raw$Country=='United States',"Country"] = 'USA'
data_raw[data_raw$Country=='United Kingdom',"Country"] = 'UK'
data_raw[data_raw$Country=='Germany',"Country"] = 'West Germany'

data = data.frame(matrix(NA,length(country_names),length(var_names)))
colnames(data) = var_names
rownames(data) = country_names
for (i in 1:length(country_names)) {
  for (j in 1:length(var_names)) {
    if (sum(data_raw$Country==country_names[i] & data_raw$Subject==var_names[j])==1) {
      data[i,j] = data_raw[data_raw$Country==country_names[i] & data_raw$Subject==var_names[j],"Value"]
    }
  }
}

W_p = rep(1/(length(country_names)-1),length(country_names)-1) # weights for simple average

data$`Electricity generation` = data$`Electricity generation`/data$`Population levels`
data$`Triadic patent families` = data$`Triadic patent families`/data$`Population levels`
data$`GDP per capita` = d[d$year==year,"gdp"]
data$`Trade openness` = d[d$year==year,"trade"]

data1 = data[,c("Private social expenditure"
                ,"Total primary energy supply per unit of GDP"
                ,"Electricity generation"
                ,"Triadic patent families"
                ,"Real GDP growth"
                ,"CPI: all items"
                ,"Trade openness"
                ,"Total tax revenue"
                ,"GDP per capita"
                )]

Z = data1
Z = Z[,complete.cases(t(Z))]
Z = scale(Z, center = F, scale = apply(Z, 2, sd))
V = rep(1/ncol(Z),ncol(Z))
W1 = fn_W(cbind(Z[unique(d$index)==7,]),Z[unique(d$index)!=7,],diag(V,ncol(Z)))

synthY0 = t(Y[unique(d$index)!=7,])%*%W
synthY01 = t(Y[unique(d$index)!=7,])%*%W1

pdf(file = "./Output/Ger_fig2.pdf", width = 5.5, height = 5.5, family = "Times",pointsize = 12)
plot(1960:2003,Y[unique(d$index)==7,],
     type="l",ylim=c(0,33000),col="black",lty="solid",
     ylab ="Per-Capita GDP (PPP, 2002 USD)",
     xlab ="Year",
     xaxs = "i", yaxs = "i",
     lwd=2
)
lines(1960:2003,synthY0,col="black",lty="dashed",lwd=2)
lines(1960:2003,synthY01,col="black",lty="dotted",lwd=2)
abline(v=1990,lty="dotted")
legend(x="bottomright",
       legend=c("West Germany","SC (single outcome in 1960-1990)","SC (multiple outcomes in 1989)")
       ,lty=c("solid","dashed","dotted"),col=c("black","black","black")
       ,cex=.8,bg="white",lwd=c(2,2,2))
Text.height = 23000
arrows(1987,Text.height,1989,Text.height,col="black",length=.1)
text(1982.5,Text.height,"Reunification",cex=.8)
dev.off()


library(xtable)
tab_com = cbind(round(t(data1[unique(d$index)==7,]),1),
                round(t(data1[unique(d$index)!=7,])%*%W1,1),
                round(t(data1[unique(d$index)!=7,])%*%W,1),
                round(t(data1[unique(d$index)!=7,])%*%W_p,1))
tab_com
rownames(tab_com) = c("Private social expenditure"
                      ,"Energy supply per GDP"
                      ,"Electricity generation"
                      ,"Triadic patent families"
                      ,"Real GDP growth"
                      ,"CPI"
                      ,"Trade openness"
                      ,"Total tax revenue"
                      ,"GDP per capita"
                      )
colnames(tab_com) = c('West Germany','Synthetic West Germany (multiple outcomes)','Synthetic West Germany (single outcome)','OECD Sample')
tab_com = xtable(tab_com, digits=c(0,1,1,1,1))
print(tab_com, include.rownames=T)
print(tab_com, include.rownames=T, file="./Output/Ger_tab.txt")



