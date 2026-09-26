import math
import matplotlib.pyplot as plt
import numpy as np
from PE_Data import get_dat

buyout_file = "Buyout_Funds_Stats_2014.xlsx"

#read in and organize historical data
dat = get_dat(buyout_file, "Called Up", in_between=90)
dat_train = dat[2]
dat_test = dat[2]

#70-30 for training and test data
dat_train.dropna(inplace=True)
dat_test.dropna(inplace = True)

dat_train = dat_train.iloc[0:math.floor(0.7 * len(dat_train)),:]
dat_test = dat_test.iloc[math.floor(0.7 * len(dat_test)):len(dat_test),:]
dat_test = np.array(dat_test.iloc[:,0])

x = np.array(dat_train)
y = np.array(dat_train.shift(-1))
x = np.delete(x, len(x)-1, axis = 0)
y = np.delete(y, len(y)-1, axis = 0)


#Least squares estimates of mean reverting (O-U) model, using train set data
#assumes dynamics follow dX = (m-l*X)*dt + sigma*dW
a_num = sum(x*y) - sum(x)*sum(y)/len(y)
a_denom = sum(x**2) - sum(x)**2/len(x)
a_hat = a_num/a_denom

b_hat = (sum(y)-sum(a_hat*x))/len(x)

m_hat = b_hat/(1-a_hat)
l_hat = 1-a_hat

e = y - (x*(1-l_hat) + l_hat*m_hat)

sigma_hat = np.std(e)

'''
#Conditional least squares estimates of mean reverting square root (CIR) model
#assumes dynamics follow dX = (a+bX)dt + sigma*sqrt(X)dW

b_num = sum((y-np.mean(y))*(x - np.mean(x)))/len(y)
b_denom = sum((x - np.mean(x))**2)/len(x)
b_hat = np.log(b_num/b_denom)

a_num = np.mean(y) - np.exp(b_hat)*np.mean(x)
a_denom = np.exp(b_hat) - 1
a_hat = a_num/a_denom*b_hat

eta_0 = a_hat/(2*b_hat**2)*(np.exp(b_hat)-1)**2
eta_1 = 1/b_hat*np.exp(b_hat)*(np.exp(b_hat)-1)

sigma_hat = np.sqrt(sum((y - (a_hat + b_hat*x))**2/(eta_0 + eta_1*x))/len(x))
'''

#MC Simulation for test set
m = 5000
pred = np.zeros([m,len(dat_test)])

for k in range(m):
    pred[k][0] = dat_train.iloc[len(dat_train)-1]
    for i in range(1,len(dat_test)):
        #Euler-Maruyama discretization for recurrence relation
        pred[k][i] = (pred[k][i-1] + l_hat*(m_hat - pred[k][i-1]) + sigma_hat*np.random.normal())
        '''
        #recurrence relation used for mean reverting sqrt process 
        pred.append(max([pred[i] + (a_hat+b_hat*pred[i]) + sigma_hat*np.sqrt(pred[i])*np.random.normal(),0.01]))
        '''

p = np.mean(pred,axis = 0)
pred = np.vstack((pred,p))

#Plotting
plt.plot(pred[m])
plt.plot(dat_test)

#Calculate error in prediction
mse = sum((pred[m] - dat_test)**2)/len(dat_test)
print(mse)
    