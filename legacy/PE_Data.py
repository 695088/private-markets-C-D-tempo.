import pandas as pd
import numpy as np
import math

def interpolate_gaussian(data, num_between, var=8):#Data should be univariate time series in pd dataframe
    #num_between is number of interpolation time steps between existing datapoints
    #var is a parameter by which the variance of the noise term is scaled
    
    new_dat = np.zeros(1)
    
    for i in range(len(data)):
        
        new_dat = np.append(new_dat, data.iloc[i,:].values[0])
        
        if (i!= (len(data) - 1)):
            step = (data.iloc[i+1,:].values[0] - data.iloc[i,:].values[0])/(num_between + 1)
            
            for j in range(num_between):
                
                #each interpolated point consists of "drift" term from linear interpolation plus white noise term
                k = data.iloc[i,:].values[0] + j*step + np.random.normal(loc=0.0,scale=1/(var*(num_between+1)))
                new_dat = np.append(new_dat,k)
                
                
    interpolated_data = pd.DataFrame(new_dat)
    interpolated_data = interpolated_data.iloc[1:len(interpolated_data),:]
        
    return(interpolated_data)
            
            
def get_dat(filename, metric = "Called Up", in_between = 90):
    
    #Read cashflow data for each fund size into a dataframe
    dat_all = pd.read_excel(filename, sheet_name = "All", header = 2, usecols = "A:G", index_col = 0)[[metric]]
    dat_large = pd.read_excel(filename, sheet_name = "Large", header = 2, usecols = "A:G",index_col = 0)[[metric]]
    dat_mid = pd.read_excel(filename, sheet_name = "Mid", header = 2, usecols = "A:G",index_col = 0)[[metric]]
    dat_small = pd.read_excel(filename, sheet_name = "Small", header = 2, usecols = "A:G",index_col = 0)[[metric]]
    
    #Take log differences (we are modelling changes in CFs), interpolate data
    dat_all = dat_all.astype("float")
    delta_all = dat_all.shift(1)
    delta_all = delta_all.drop(delta_all.index[0])
    dat_all = dat_all.drop(dat_all.index[0])
    
    #drop 0 rows, could be first 5 or so quarters of life of fund
    dat_all = dat_all.loc[~(dat_all==0).all(axis=1)]
    delta_all = delta_all.loc[~(delta_all==0).all(axis=1)]
    delta_all = np.log(dat_all/delta_all)
    
    #different variances in noise for training and test set for more robust prediction
    delta_all_train = interpolate_gaussian(delta_all.iloc[0:math.floor(0.7*len(delta_all))],in_between,6)
    delta_all_test = interpolate_gaussian(delta_all.iloc[math.floor(0.7*len(delta_all)):len(delta_all)],in_between,8)
    delta_all = pd.concat([delta_all_train, delta_all_test])
    
    #repeat for other size benchmarks within the same fund type
    dat_large = dat_large.astype("float")
    delta_large = dat_large.shift(1)
    delta_large = delta_large.drop(delta_large.index[0])
    dat_large = dat_large.drop(dat_large.index[0])
    dat_large = dat_large.loc[~(dat_large==0).all(axis=1)]
    delta_large = delta_large.loc[~(delta_large==0).all(axis=1)]
    delta_large = np.log(dat_large/delta_large)
    delta_large_train = interpolate_gaussian(delta_large.iloc[0:math.floor(0.7*len(delta_large))],in_between,6)
    delta_large_test = interpolate_gaussian(delta_large.iloc[math.floor(0.7*len(delta_large)):len(delta_large)],in_between,8)
    delta_large = pd.concat([delta_large_train, delta_large_test])
    
    dat_mid = dat_mid.astype("float")
    delta_mid = dat_mid.shift(1)
    delta_mid = delta_mid.drop(delta_mid.index[0])
    dat_mid = dat_mid.drop(dat_mid.index[0])
    dat_mid = dat_mid.loc[~(dat_mid==0).all(axis=1)]#
    delta_mid = delta_mid.loc[~(delta_mid==0).all(axis=1)]
    delta_mid = np.log(dat_mid/delta_mid)
    delta_mid_train = interpolate_gaussian(delta_mid.iloc[0:math.floor(0.7*len(delta_mid))],in_between,6)
    delta_mid_test = interpolate_gaussian(delta_mid.iloc[math.floor(0.7*len(delta_mid)):len(delta_mid)],in_between,8)
    delta_mid = pd.concat([delta_mid_train, delta_mid_test])
    
    dat_small = dat_small.astype("float")
    delta_small = dat_small.shift(1)
    delta_small = delta_small.drop(delta_small.index[0])
    dat_small = dat_small.drop(dat_small.index[0])
    dat_small = dat_small.loc[~(dat_small==0).all(axis=1)]
    delta_small = delta_small.loc[~(delta_small==0).all(axis=1)]
    delta_small = np.log(dat_small/delta_small)
    delta_small_train = interpolate_gaussian(delta_small.iloc[0:math.floor(0.7*len(delta_small))],in_between,6)
    delta_small_test = interpolate_gaussian(delta_small.iloc[math.floor(0.7*len(delta_small)):len(delta_small)],in_between,8)
    delta_small = pd.concat([delta_small_train, delta_small_test])
    
    #returns vectors of interpolated data, with training and test sets (70-30) concatenated for each fund size
    return ([delta_all, delta_large,delta_mid,delta_small])

def prep_dat(d, n_step = 180, n_ahead = 90): #Train a network for given prediction horizon (default 1 quarter)
    #Prepare data for input/output of NNs, n_step is number of historical time steps (days) to include in regression task
   
    a = d.copy()
    
    for i in range(1,n_step):
        ind = "t+" + str(i)
        a[ind] = a.iloc[:,0].shift(-i)#Use n_step historical observations to predict cash flow in n_ahead time steps
        
    a["Q_Next"] = a.iloc[:,0].shift(-(n_step + n_ahead))#Next prediction
        
    a.dropna(inplace=True) #last (n_step + n_ahead) rows will be NANs since shifting forward for "Next Quarter" Variable
    
    #return dataframe of time series, transforming the data from the get_dat function into a form usable
    #for a supervised learning task
    return(a)
