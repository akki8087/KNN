# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:01:14 2018

@author: NP
"""

import numpy as np
import pandas as pd
import math
from sklearn.metrics.pairwise import euclidean_distances
from statistics import mode

data = pd.read_csv('loantrain.csv')

X = data.iloc[:,6:10] 
y = data.iloc[:,-1]
X["LoanAmount"] = X["LoanAmount"].fillna(X.median()["LoanAmount"])
X = X.fillna({"Loan_Amount_Term":360})

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

Train_index = list(X_train.index)
test_index = list(X_test.index)
#X_test.loc['LS'] = 0
for i in test_index:
    #X_train.loc['d'] = 0
    for j in Train_index:
        d = euclidean_distances([X_test.iloc[i,:]],[X_train.iloc[j,:-1]])
        X_train['d'][j] = d
    s = X_train['d'].sort_values()
    c = s[:k+1]
    k_index = list(c.index)
    X_test['LS'][i] = mode(data['Loan_Status'][k_index])
    
        
        