import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
from sklearn import preprocessing
import cvxpy as cp
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
import pandas as pd
import seaborn as sns

## Function Definitions


def shrink(X,tau):
    Y = np.abs(X)-tau
    return np.sign(X) * np.maximum(Y,np.zeros_like(Y))
def SVT(X,tau):
    U,S,VT = np.linalg.svd(X,full_matrices=0)
    out = U @ np.diag(shrink(S,tau)) @ VT
    return out
def RPCA(X):#,mu,lambd):
    n1,n2 = X.shape
    mu = n1*n2/(4*np.sum(np.abs(X.reshape(-1))))
    lambd =1/np.sqrt(np.maximum(n1,n2))
    thresh = 10**(-7) * np.linalg.norm(X)
    
    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    L = np.zeros_like(X)
    count = 0
    while (np.linalg.norm(X-L-S) > thresh) and (count < 1000):
        L = SVT(X-S+(1/mu)*Y,1/mu)
        S = shrink(X-L+(1/mu)*Y,lambd/mu)
        Y = Y + mu*(X-L-S)
        count += 1
    return L,S

def  RMSE(Y_pred,Y_acutal):
    return np.sqrt(np.average(np.square(Y_pred-Y_acutal)))


abadie = np.array([2117.006,2218.653,2341.527,2452.516,2627.328,2782.917,3008.872,3184.185
                   ,3484.512,3864.664,4290.689,4664.163,5084.445,5603.405,6182.171,6672.881,7303.861,8058.932
                   ,8785.722,9869.054,10906.853,12001.092,12714.605,13540.317,14392.317,15273.670,15950.330
                   ,16684.858,17851.175,19157.064,20456.970,21333.792,22065.724,22520.611,23435.119,24252.454
                   ,25216.310,26118.999,26953.839,27936.264,29587.023,30266.606,31300.751,32227.807])


abadie_placebo= np.array([2243.501,2329.424,2445.830,2554.472,2722.214,2887.547,3129.130,3280.654
                          ,3561.095,3888.492,4273.404,4643.754,5081.923,5605.596,6198.890,6731.356
                           ,7433.985,8239.899,8984.699,10120.270,11104.061,12205.059,12915.074
                           ,13873.131,14773.280,15653.366,16360.138,17118.086,18265.897,19511.173])

shah = np.array([ 2164.52277295,  2286.50604073,  2424.6906971 ,  2549.60643591,
        2726.06631043,  2896.89121883,  3113.70608484,  3299.60042025,
        3603.31917047,  4004.29616859,  4389.80134638,  4747.47000547,
        5151.32660518,  5706.39113315,  6288.78990456,  6803.26608744,
        7450.91138887,  8087.35073611,  8863.31160093,  9873.81410354,
       10948.99422282, 12035.32581841, 12764.53692538, 13491.4211896 ,
       14396.5142543 , 15270.76581648, 15978.01646597, 16746.60591701,
       17853.81290409, 19043.04769091, 20147.0817962 , 20976.72774786,
       21683.90761276, 22205.12747333, 23255.38001448, 24233.99677048,
       25235.01516217, 26352.06183555, 27173.93011122, 28318.60029927,
       30367.96968217, 31457.05336976, 32555.82160358, 33370.68498922])

shah_placebo = np.array([ 2273.86154917,  2377.83587673,  2509.52468295,  2627.60762298,
        2797.9247609 ,  2966.72034367,  3173.86409556,  3325.75305925,
        3602.91680676,  3985.16604435,  4325.3333641 ,  4659.1224082 ,
        5029.27606879,  5550.35778395,  6117.69504728,  6569.3667234 ,
        7151.19991497,  7723.88794618,  8439.45568548,  9366.97998277,
       10360.29704388, 11405.86344127, 12065.72758903, 12770.58595069,
       13676.51961042, 14491.1642467 , 15157.50563607, 15850.94073731,
       16801.04387126, 17845.23906027])

Data =np.genfromtxt('clustered_data.csv',delimiter=',')[1:,2:]
#Data_process=preprocessing.scale(Data)
X_pre =np.delete(Data[:,0:30],(5),axis=0)
X =np.delete(Data,(5),axis=0)
'''
mu = np.linspace(0.01,3,num=50)
lambd= np.linspace(0.01,1,num=50)
mse = np.zeros((len(mu),len(lambd)))

for i in range(len(mu)):
    for j in range(len(lambd)):
        L,S = RPCA(X_pre,mu[i],lambd[j])
        beta = cp.Variable(11)
        objective = cp.Minimize(cp.sum_squares(Data_process[5,0:34]-beta@L))
        constraints = [cp.sum(beta)==1]
        prob = cp.Problem(objective, constraints)
        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        mse[i,j] =np.sum((Data_process[5,0:34] - np.dot(beta.value,L))**2)

mu_op =mu[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[0]]
lambd_op =lambd[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[1]]
'''

##############################################

L,S = RPCA(X)
L_neg = L[:,0:30]
L_pos = L[:,31:]
beta = cp.Variable(11)
objective = cp.Minimize(cp.sum_squares(Data[5,0:30]-beta@L_neg))
constraints = [beta >= 0]
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
print(beta.value)

#### Figures for L and S
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(8,4))
fig.tight_layout() 
ax1.plot(range(1960,2004),L.transpose())
ax1.set_xlabel('Year')
ax1.set_ylabel('Per Capita GDP (PPP, 2002 USD)')
ax2.plot(range(1960,2004), S.transpose())
ax2.set_xlabel('Year')
plt.show()

########

counter_factual = np.dot(L.T,beta.value)
counter_factual_main = counter_factual

plt.plot(range(1960,2004),Data[5,:],label='West Germany')
plt.plot(range(1960,2004),counter_factual,label='Robust PCA Synthetic Control',linestyle='dashed')
plt.plot(range(1960,2004),abadie,label='Synthetic Control',linestyle='dashdot')
plt.plot(range(1960,2004),shah,label='Robust Synthetic Control',linestyle='dotted')
plt.legend()
plt.vlines(x=1990,ymin=0,ymax=np.max(counter_factual),linestyles='dashed',colors='silver')
plt.annotate('Reunification', xy =(1990, 5000), 
                xytext =(1975,4500),  
                arrowprops = dict(facecolor ='black', 
                                  shrink = 0.1)) 
plt.xlabel('Year')
plt.ylabel('Per Capita GDP (PPP, 2002 USD)')
#plt.savefig("main_plot.png")
plt.show()
##############

plt.plot(range(1960,2004),Data[5,:] - counter_factual,color='black',label='Robust PCA Synthetic Control')
plt.plot(range(1960,2004),Data[5,:] - abadie,linestyle='dashed',label='Synthetic Control')
plt.plot(range(1960,2004),Data[5,:] - shah,linestyle='dashdot',label='Robust Synthetic Control')
plt.vlines(x=1990,ymin=-7000,ymax=4000,linestyles='dashed',colors='silver')
plt.hlines(y=0,xmin=1960,xmax=2003,linestyle='dashed',color='silver')
plt.annotate('Reunification', xy =(1990, -3000), 
                xytext =(1975,-3100),  
                arrowprops = dict(facecolor ='black', 
                                  shrink = 0.1)) 
plt.legend(loc='lower left')
plt.xlabel('Year')
plt.ylabel('Per Capita GDP (PPP, 2002 USD)')
#plt.savefig("Gap.png",bbox_inches='tight')
plt.show()

############################ Placebo in time
abadie_placebo= np.array([2243.501,2329.424,2445.830,2554.472,2722.214,2887.547,3129.130,3280.654
                          ,3561.095,3888.492,4273.404,4643.754,5081.923,5605.596,6198.890,6731.356
                           ,7433.985,8239.899,8984.699,10120.270,11104.061,12205.059,12915.074
                           ,13873.131,14773.280,15653.366,16360.138,17118.086,18265.897,19511.173])

L,S = RPCA(X[:,0:30])#,mu_op,lambd_op)
L_neg = L[:,0:15]
L_pos = L[:,16:]
beta = cp.Variable(11)
objective = cp.Minimize(cp.sum_squares(Data[5,0:15]-beta@L_neg))
constraints = [#cp.sum(beta)==1]
                         beta >= 0]
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
print(beta.value)





counter_factual = np.dot(L.T,beta.value)

plt.plot(range(1960,1990),abadie_placebo,label='Synthetic Control',linestyle='dashdot')
plt.plot(range(1960,1990),Data[5,0:30],label='West Germany')
plt.plot(range(1960,1990),counter_factual,label='Robust PCA Synthetic Control',linestyle='dashed')
plt.plot(range(1960,1990),shah_placebo,label='Robust Synthetic Control',linestyle='dotted')
plt.legend(loc='best')
plt.vlines(x=1975,ymin=0,ymax=14000,linestyles='dashed',colors='silver')

plt.annotate('Reunification', xy =(1975, 1000), 
                xytext =(1965,900),  
                arrowprops = dict(facecolor ='black', 
                                  shrink = 0.1)) 
plt.xlabel('Year')
plt.ylabel('Per Capita GDP (PPP, 2002 USD)')
#plt.savefig("placebo_time.png")
plt.show()

################################################# test test test test

L,S = RPCA(X[:,0:30])#,mu_op,lambd_op)
L_neg = L[:,0:15]
L_pos = L[:,16:]
beta = cp.Variable(11)
objective = cp.Minimize(cp.sum_squares(Data[5,0:15]-beta@L_neg))
constraints = [#cp.sum(beta)==1]
                         beta >= 0]
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
print(beta.value)





counter_factual = np.dot(L.T,beta.value)

plt.plot(range(1960,1990),abadie_placebo-Data[5,0:30],label='Synthetic Control',linestyle='dashdot')
#plt.plot(range(1960,1990),Data[5,0:30],label='West Germany')
plt.plot(range(1960,1990),counter_factual-Data[5,0:30],label='Robust PCA Synthetic Control',linestyle='dashed')
plt.plot(range(1960,1990),shah_placebo-Data[5,0:30],label='Robust Synthetic Control',linestyle='dotted')
plt.legend(loc='best')
#plt.vlines(x=1975,ymin=0,ymax=14000,linestyles='dashed',colors='silver')

#plt.annotate('Reunification', xy =(1975, 1000), 
#                xytext =(1965,900),  
#                arrowprops = dict(facecolor ='black', 
#                                  shrink = 0.1)) 
plt.xlabel('Year')
plt.ylabel('Per Capita GDP (PPP, 2002 USD)')
#plt.savefig("placebo_time.png")
plt.show()


############################################## Placebo in space
Pre_RMSE = np.zeros(Data.shape[0])
Post_RMSE = np.zeros(Data.shape[0])


for i in range(Data.shape[0]):
    X =np.delete(Data,(1),axis=0)
    
    L,S = RPCA(X)
    L_neg = L[:,0:30]
    L_pos = L[:,31:]
    beta = cp.Variable(11)
    objective = cp.Minimize(cp.sum_squares(Data[i,0:30]-beta@L_neg))
    constraints = [#cp.sum(beta)==1]
                             beta >= 0]
    prob = cp.Problem(objective, constraints)
    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    counter_factual = np.dot(L.T,beta.value)
    Pre_RMSE[i] = RMSE(Data[i,0:30],counter_factual[:30])
    Post_RMSE[i] = RMSE(Data[i,31:],counter_factual[31:])

Placebo_space = Post_RMSE/Pre_RMSE

countries = ["UK","Austria","Belgium","Denmark","France",
             "West Germany","Italy","Netherlands","Norway","Japan","Australia","New Zealand"]

sorted_countries = []

for i in range(len(countries)):
    sorted_countries.append(countries[np.argsort(Placebo_space)[i]])


plt.figure(figsize=(16, 30))
plt.subplot(3, 1, 1)
plt.plot(Placebo_space[np.argsort(Placebo_space)],sorted_countries,"o",markersize=10)
plt.hlines(sorted_countries,xmin = 0, xmax = 23, linestyles='dashed',color="0.9")
plt.xlabel("Postperiod RMSPE/Preperiod RMSPE for Robust PCA Synthetic Control", fontsize=16)
plt.yticks(size = 16)
#plt.savefig("placebo_inspace_Robust_PCA.png")

############################################# Place in space others


RMSE_abadie = pd.read_csv('RMSE_abadie.csv')
RMSE_shah = pd.read_csv('RMSE_shah.csv')

countries = RMSE_abadie["Country"].values

sorted_countries = []

for i in range(len(countries)):
    sorted_countries.append(countries[np.argsort(RMSE_abadie['RMSE'].values)[i]])



plt.subplot(3, 1, 2)
plt.plot(RMSE_abadie['RMSE'].values[np.argsort(RMSE_abadie['RMSE'].values)],sorted_countries,"o",markersize=10)
plt.hlines(sorted_countries,xmin = 0, xmax = 23, linestyles='dashed',color="0.9")
plt.xlabel("Postperiod RMSPE/Preperiod RMSPE for Synthetic Control.", fontsize=16)
plt.yticks(size = 16)
#plt.savefig("placebo_inspace_abadie.png")



countries = RMSE_shah["Country"].values

sorted_countries = []

for i in range(len(countries)):
    sorted_countries.append(countries[np.argsort(RMSE_shah['RMSE'].values)[i]])

plt.subplot(3, 1, 3)
#plt.figure(figsize=(12, 10))
plt.plot(RMSE_shah['RMSE'].values[np.argsort(RMSE_shah['RMSE'].values)],sorted_countries,"o",markersize=10)
plt.hlines(sorted_countries,xmin = 0, xmax = 23, linestyles='dashed',color="0.9")
plt.xlabel("Postperiod RMSPE/Preperiod RMSPE for Robust Sythetic Control", fontsize=16)
plt.yticks(size = 16)
plt.savefig("placebo_inspace.png")









#################################################### Placebo Austria
X =np.delete(Data,(1),axis=0)
'''
mse = np.zeros((len(mu),len(lambd)))

for i in range(len(mu)):
    for j in range(len(lambd)):
        L,S = RPCA(X_pre,mu[i],lambd[j])
        beta = cp.Variable(11)
        objective = cp.Minimize(cp.sum_squares(Data_process[3,0:34]-beta@L))
        constraints = [cp.sum(beta)==1]
        prob = cp.Problem(objective, constraints)
        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        mse[i,j] =np.sum((Data_process[3,0:34] - np.dot(beta.value,L))**2)

mu_op =mu[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[0]]
lambd_op =lambd[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[1]]
'''
L,S = RPCA(X)
L_neg = L[:,0:30]
L_pos = L[:,31:]
beta = cp.Variable(11)
objective = cp.Minimize(cp.sum_squares(Data[1,0:30]-beta@L_neg))
constraints = [#cp.sum(beta)==1]
                         beta >= 0]
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
#print(beta.value)


counter_factual = np.dot(L.T,beta.value)

plt.plot(range(1960,2004),Data[1,:],label='Austria')
plt.plot(range(1960,2004),counter_factual,label='Synthtetic Austria',linestyle='dashed')
plt.legend()
plt.vlines(x=1990,ymin=0,ymax=np.max(counter_factual),linestyles='dashed')
plt.annotate('Reunification', xy =(1990, 25000), 
                xytext =(1975,25000),  
                arrowprops = dict(facecolor ='black', 
                                  shrink = 0.05)) 
plt.xlabel('Year')
plt.ylabel('Per Capita GDP (PPP, 2002 USD)')
plt.savefig("Austria.png")
plt.show()

#################################################### Placebo Denmark
X =np.delete(Data,(3),axis=0)
'''
mse = np.zeros((len(mu),len(lambd)))

for i in range(len(mu)):
    for j in range(len(lambd)):
        L,S = RPCA(X_pre,mu[i],lambd[j])
        beta = cp.Variable(11)
        objective = cp.Minimize(cp.sum_squares(Data_process[3,0:34]-beta@L))
        constraints = [cp.sum(beta)==1]
        prob = cp.Problem(objective, constraints)
        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        mse[i,j] =np.sum((Data_process[3,0:34] - np.dot(beta.value,L))**2)

mu_op =mu[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[0]]
lambd_op =lambd[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[1]]
'''
L,S = RPCA(X)
L_neg = L[:,0:30]
L_pos = L[:,31:]
beta = cp.Variable(11)
objective = cp.Minimize(cp.sum_squares(Data[3,0:30]-beta@L_neg))
constraints = [#cp.sum(beta)==1]
                         beta >= 0]
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
#print(beta.value)




counter_factual = np.dot(L.T,beta.value)

plt.plot(range(1960,2004),Data[3,:],label='Denmark')
plt.plot(range(1960,2004),counter_factual,label='Synthtetic Denmark',linestyle='dashed')
plt.legend()
plt.vlines(x=1990,ymin=0,ymax=np.max(counter_factual),linestyles='dashed')
plt.annotate('Reunification', xy =(1990, 25000), 
                xytext =(1975,25000),  
                arrowprops = dict(facecolor ='black', 
                                  shrink = 0.05)) 
plt.xlabel('Year')
plt.ylabel('Per Capita GDP (PPP, 2002 USD)')
plt.show()


################################################ Placebo Belgium

X =np.delete(Data,(2),axis=0)
'''
mse = np.zeros((len(mu),len(lambd)))

for i in range(len(mu)):
    for j in range(len(lambd)):
        L,S = RPCA(X_pre,mu[i],lambd[j])
        beta = cp.Variable(11)
        objective = cp.Minimize(cp.sum_squares(Data_process[2,0:34]-beta@L))
        constraints = [cp.sum(beta)==1]
        prob = cp.Problem(objective, constraints)
        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        mse[i,j] =np.sum((Data_process[2,0:34] - np.dot(beta.value,L))**2)

mu_op =mu[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[0]]
lambd_op =lambd[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[1]]
'''
L,S = RPCA(X)
L_neg = L[:,0:30]
L_pos = L[:,31:]
beta = cp.Variable(11)
objective = cp.Minimize(cp.sum_squares(Data[2,0:30]-beta@L_neg))
constraints = [#cp.sum(beta)==1]
                         beta >= 0]
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
#print(beta.value)


counter_factual = np.dot(L.T,beta.value)

plt.plot(range(1960,2004),Data[2,:],label='Belgium')
plt.plot(range(1960,2004),counter_factual,label='Synthtetic Belgium',linestyle='dashed')
plt.legend()
plt.vlines(x=1990,ymin=0,ymax=np.max(counter_factual),linestyles='dashed')
plt.annotate('Reunification', xy =(1990, 25000), 
                xytext =(1975,25000),  
                arrowprops = dict(facecolor ='black', 
                                  shrink = 0.05)) 
plt.xlabel('Year')
plt.ylabel('Per Capita GDP (PPP, 2002 USD)')
plt.show()

###############################   Placebo Italy

X_pre =np.delete(Data,(6),axis=0)
'''
mse = np.zeros((len(mu),len(lambd)))

for i in range(len(mu)):
    for j in range(len(lambd)):
        L,S = RPCA(X_pre,mu[i],lambd[j])
        beta = cp.Variable(11)
        objective = cp.Minimize(cp.sum_squares(Data_process[2,0:34]-beta@L))
        constraints = [cp.sum(beta)==1]
        prob = cp.Problem(objective, constraints)
        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        mse[i,j] =np.sum((Data_process[2,0:34] - np.dot(beta.value,L))**2)

mu_op =mu[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[0]]
lambd_op =lambd[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[1]]
'''
L,S = RPCA(X)
L_neg = L[:,0:30]
L_pos = L[:,31:]
beta = cp.Variable(11)
objective = cp.Minimize(cp.sum_squares(Data[6,0:30]-beta@L_neg))
constraints = [#cp.sum(beta)==1]
                         beta >= 0]
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
#print(beta.value)



counter_factual = np.dot(L.T,beta.value)

plt.plot(range(1960,2004),Data[6,:],label='Italy')
plt.plot(range(1960,2004),counter_factual,label='Synthtetic Italy',linestyle='dashed')
plt.legend()
plt.vlines(x=1990,ymin=0,ymax=np.max(counter_factual),linestyles='dashed')
plt.annotate('Reunification', xy =(1990, 20000), 
                xytext =(1975,20000),  
                arrowprops = dict(facecolor ='black', 
                                  shrink = 0.05)) 
plt.xlabel('Year')
plt.ylabel('Per Capita GDP (PPP, 2002 USD)')
plt.show() 

######################### Placebo Netherlands

X =np.delete(Data,(7),axis=0)
'''
mse = np.zeros((len(mu),len(lambd)))

for i in range(len(mu)):
    for j in range(len(lambd)):
        L,S = RPCA(X_pre,mu[i],lambd[j])
        beta = cp.Variable(11)
        objective = cp.Minimize(cp.sum_squares(Data_process[2,0:34]-beta@L))
        constraints = [cp.sum(beta)==1]
        prob = cp.Problem(objective, constraints)
        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        mse[i,j] =np.sum((Data_process[2,0:34] - np.dot(beta.value,L))**2)

mu_op =mu[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[0]]
lambd_op =lambd[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[1]]
'''
L,S = RPCA(X)
L_neg = L[:,0:30]
L_pos = L[:,31:]
beta = cp.Variable(11)
objective = cp.Minimize(cp.sum_squares(Data[7,0:30]-beta@L_neg))
constraints = [#cp.sum(beta)==1]
                         beta >= 0]
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
#print(beta.value)


counter_factual = np.dot(L.T,beta.value)

plt.plot(range(1960,2004),Data[7,:],label='Netherlands')
plt.plot(range(1960,2004),counter_factual,label='Synthtetic Netherlands',linestyle='dashed')
plt.legend()
plt.vlines(x=1990,ymin=0,ymax=np.max(counter_factual),linestyles='dashed')
plt.annotate('Reunification', xy =(1990, 20000), 
                xytext =(1975,20000),  
                arrowprops = dict(facecolor ='black', 
                                  shrink = 0.05)) 
plt.xlabel('Year')
plt.ylabel('Per Capita GDP (PPP, 2002 USD)')
plt.show() 

##################################Placebo France

X =np.delete(Data,(4),axis=0)
'''
mse = np.zeros((len(mu),len(lambd)))

for i in range(len(mu)):
    for j in range(len(lambd)):
        L,S = RPCA(X_pre,mu[i],lambd[j])
        beta = cp.Variable(11)
        objective = cp.Minimize(cp.sum_squares(Data_process[2,0:34]-beta@L))
        constraints = [cp.sum(beta)==1]
        prob = cp.Problem(objective, constraints)
        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        mse[i,j] =np.sum((Data_process[2,0:34] - np.dot(beta.value,L))**2)

mu_op =mu[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[0]]
lambd_op =lambd[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[1]]
'''
L,S = RPCA(X)
L_neg = L[:,0:30]
L_pos = L[:,31:]
beta = cp.Variable(11)
objective = cp.Minimize(cp.sum_squares(Data[4,0:30]-beta@L_neg))
constraints = [#cp.sum(beta)==1]
                         beta >= 0]
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
#print(beta.value)


counter_factual = np.dot(L.T,beta.value)

plt.plot(range(1960,2004),Data[4,:],label='France')
plt.plot(range(1960,2004),counter_factual,label='Synthtetic France',linestyle='dashed')
plt.legend()
plt.vlines(x=1990,ymin=0,ymax=np.max(counter_factual),linestyles='dashed')
plt.annotate('Reunification', xy =(1990, 20000), 
                xytext =(1975,20000),  
                arrowprops = dict(facecolor ='black', 
                                  shrink = 0.05)) 
plt.xlabel('Year')
plt.ylabel('Per Capita GDP (PPP, 2002 USD)')
plt.savefig("France.png")
plt.show() 

################## Placebo Norway


X =np.delete(Data,(8),axis=0)
'''
mse = np.zeros((len(mu),len(lambd)))

for i in range(len(mu)):
    for j in range(len(lambd)):
        L,S = RPCA(X_pre,mu[i],lambd[j])
        beta = cp.Variable(11)
        objective = cp.Minimize(cp.sum_squares(Data_process[2,0:34]-beta@L))
        constraints = [cp.sum(beta)==1]
        prob = cp.Problem(objective, constraints)
        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        mse[i,j] =np.sum((Data_process[2,0:34] - np.dot(beta.value,L))**2)

mu_op =mu[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[0]]
lambd_op =lambd[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[1]]
'''
L,S = RPCA(X)
L_neg = L[:,0:30]
L_pos = L[:,31:]
beta = cp.Variable(11)
objective = cp.Minimize(cp.sum_squares(Data[8,0:30]-beta@L_neg))
constraints = [#cp.sum(beta)==1]
                         beta >= 0]
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
#print(beta.value)


counter_factual = np.dot(L.T,beta.value)

plt.plot(range(1960,2004),Data[8,:],label='Norway')
plt.plot(range(1960,2004),counter_factual,label='Synthtetic Norway',linestyle='dashed')
plt.legend()
plt.vlines(x=1990,ymin=0,ymax=np.max(counter_factual),linestyles='dashed')
plt.annotate('Reunification', xy =(1990, 20000), 
                xytext =(1975,20000),  
                arrowprops = dict(facecolor ='black', 
                                  shrink = 0.05)) 
plt.xlabel('Year')
plt.ylabel('Per Capita GDP (PPP, 2002 USD)')
plt.savefig("Norway.png")
plt.show() 

################ Placebo New Zealand

X =np.delete(Data,(11),axis=0)
'''
mse = np.zeros((len(mu),len(lambd)))

for i in range(len(mu)):
    for j in range(len(lambd)):
        L,S = RPCA(X_pre,mu[i],lambd[j])
        beta = cp.Variable(11)
        objective = cp.Minimize(cp.sum_squares(Data_process[2,0:34]-beta@L))
        constraints = [cp.sum(beta)==1]
        prob = cp.Problem(objective, constraints)
        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        mse[i,j] =np.sum((Data_process[2,0:34] - np.dot(beta.value,L))**2)

mu_op =mu[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[0]]
lambd_op =lambd[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[1]]
'''
L,S = RPCA(X)
L_neg = L[:,0:30]
L_pos = L[:,31:]
beta = cp.Variable(11)
objective = cp.Minimize(cp.sum_squares(Data[11,0:30]-beta@L_neg))
constraints = [#cp.sum(beta)==1]
                         beta >= 0]
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
#print(beta.value)


counter_factual = np.dot(L.T,beta.value)

plt.plot(range(1960,2004),Data[11,:],label='New Zealand')
plt.plot(range(1960,2004),counter_factual,label='Synthtetic New Zealand',linestyle='dashed')
plt.legend()
plt.vlines(x=1990,ymin=0,ymax=np.max(counter_factual),linestyles='dashed')
plt.annotate('Reunification', xy =(1990, 20000), 
                xytext =(1975,20000),  
                arrowprops = dict(facecolor ='black', 
                                  shrink = 0.05)) 
plt.xlabel('Year')
plt.ylabel('Per Capita GDP (PPP, 2002 USD)')
plt.show() 

#### Placebo UK

X =np.delete(Data,(0),axis=0)
'''
mse = np.zeros((len(mu),len(lambd)))

for i in range(len(mu)):
    for j in range(len(lambd)):
        L,S = RPCA(X_pre,mu[i],lambd[j])
        beta = cp.Variable(11)
        objective = cp.Minimize(cp.sum_squares(Data_process[2,0:34]-beta@L))
        constraints = [cp.sum(beta)==1]
        prob = cp.Problem(objective, constraints)
        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        mse[i,j] =np.sum((Data_process[2,0:34] - np.dot(beta.value,L))**2)

mu_op =mu[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[0]]
lambd_op =lambd[np.unravel_index(np.argmin(mse, axis=None), mse.shape)[1]]
'''
L,S = RPCA(X)
L_neg = L[:,0:30]
L_pos = L[:,31:]
beta = cp.Variable(11)
objective = cp.Minimize(cp.sum_squares(Data[0,0:30]-beta@L_neg))
constraints = [#cp.sum(beta)==1]
                         beta >= 0]
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
#print(beta.value)


counter_factual = np.dot(L.T,beta.value)

plt.plot(range(1960,2004),Data[0,:],label='UK')
plt.plot(range(1960,2004),counter_factual,label='Synthtetic UK',linestyle='dashed')
plt.legend()
plt.vlines(x=1990,ymin=0,ymax=np.max(counter_factual),linestyles='dashed')
plt.annotate('Reunification', xy =(1990, 20000), 
                xytext =(1975,20000),  
                arrowprops = dict(facecolor ='black', 
                                  shrink = 0.05)) 
plt.xlabel('Year')
plt.ylabel('Per Capita GDP (PPP, 2002 USD)')
plt.show() 




####################### Robustness
fig = plt.figure(figsize=(8,4))

ax = fig.add_subplot(1,1,1)

for k in [0,1,4,8,11]:
    X =np.delete(Data,[k,5],axis=0)
    L,S = RPCA(X)
    L_neg = L[:,0:30]
    L_pos = L[:,31:]
    beta = cp.Variable(10)
    objective = cp.Minimize(cp.sum_squares(Data[5,0:30]-beta@L_neg))
    constraints = [#cp.sum(beta)==1]
                         beta >= 0]
    prob = cp.Problem(objective, constraints)
    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    #print(beta.value)

    counter_factual = np.dot(L.T,beta.value)
    ax.plot(range(1960,2004),counter_factual,color ='gray',label='Synthetic West Germany (leave-one-out)',linewidth=0.5)
    
ax.plot(range(1960,2004),Data[5,:],color='black',label='West Germany')
ax.plot(range(1960,2004),counter_factual_main,color='blue',label='Synthetic West Germany (R-PCA SC)',linestyle="dashed")
plt.vlines(x=1990,ymin=0,ymax=np.max(Data[5,:]),linestyles='dashed',colors='silver')
handles, labels = ax.get_legend_handles_labels()
handle_list, label_list = [], []
for handle, label in zip(handles, labels):
    if label not in label_list:
        handle_list.append(handle)
        label_list.append(label)
plt.legend(handle_list, label_list)
plt.annotate('Reunification', xy =(1990, 5000), 
                xytext =(1975,4500),  
                arrowprops = dict(facecolor ='black', 
                                  shrink = 0.1)) 
#plt.legend(loc='upper left')
ax.set_xlabel('Year')
ax.set_ylabel('Per Capita GDP (PPP, 2002 USD)')
plt.savefig("robustness_robust_PCA.png",bbox_inches='tight')


########################### Robustness abadie


robust_abadie =np.genfromtxt('fig5_abadie.csv',delimiter=',')


fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1,1,1)

for k in range(5):

    ax.plot(range(1960,2004),robust_abadie[:,k],color ='gray',label='Synthetic West Germany (leave-one-out)',linewidth=0.5)
    
ax.plot(range(1960,2004),Data[5,:],color='black',label='West Germany')
ax.plot(range(1960,2004),abadie,color='blue',label='Synthetic West Germany (SC)',linestyle="dashed")
plt.vlines(x=1990,ymin=0,ymax=np.max(Data[5,:]),linestyles='dashed',colors='silver')
handles, labels = ax.get_legend_handles_labels()
handle_list, label_list = [], []
for handle, label in zip(handles, labels):
    if label not in label_list:
        handle_list.append(handle)
        label_list.append(label)
plt.legend(handle_list, label_list)
plt.annotate('Reunification', xy =(1990, 5000), 
                xytext =(1975,4500),  
                arrowprops = dict(facecolor ='black', 
                                  shrink = 0.1)) 
#plt.legend(loc='upper left')
ax.set_xlabel('Year')
ax.set_ylabel('Per Capita GDP (PPP, 2002 USD)')
plt.savefig("robustness_abadie.png",bbox_inches='tight')



############################## Plots in R

FPCA = np.array([0.963,0.98, 0.99, 0.998,0.999,0.9995541,0.9998752, 0.9999721,0.9999928,1.0000000])

plt.plot(np.arange(1,11),FPCA,color = "b",marker='o')

plt.ylabel("Proportion of Explained Variation")
#plt.vlines(x=2,ymin=np.min(spectrum),ymax=np.max(spectrum),linestyles='dashed',colors='silver')
plt.xlabel("Number of FPC-scores") 
plt.grid()
plt.figure()


wws = np.array([16.0000000,6.1137485,1.5733251,0.8578428,0.6676960,0.6051365,0.4396883,0.5444146])
plt.plot(np.arange(1,9),wws,color = "b",marker='o')
plt.ylabel("Within Groups Sum of Squares")
#plt.vlines(x=2,ymin=np.min(spectrum),ymax=np.max(spectrum),linestyles='dashed',colors='silver')
plt.xlabel("Number of Clusters") 
plt.grid()
plt.figure()


sil = np.array([0.0000000,0.6662833,0.7199898,0.5808211,0.5691587,0.5248693,0.5209997,0.4424342])
plt.plot(np.arange(1,9),sil,color = "b",marker='o')
plt.ylabel("Silhouette Coefficient")
#plt.vlines(x=2,ymin=np.min(spectrum),ymax=np.max(spectrum),linestyles='dashed',colors='silver')
plt.xlabel("Number of Clusters") 
plt.grid()
plt.figure()


