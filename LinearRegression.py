# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:00:09 2018

@author: Palash Tichkule
"""

import quandl
import math
import numpy as np
from sklearn import preprocessing , cross_validation , svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = 'Enter_your_key_here' # Get you key at  https://www.quandl.com/account/api

#Choose the ticker symbol to use, 
#df = quandl.get('WIKI/GOOGL')
df = quandl.get('WIKI/NVDA')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-999999,inplace=True)  # will consider value -999999 as outlier, so shall not affect accuracy

forecast_out = int(math.ceil(0.01*len(df)))
print("Forcasting {} trading days ahead".format(forecast_out))
#>>    Forcasting 49 trading days ahead

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(X)

    
X_train , X_test , y_train , y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

#clf = LinearRegression()
clf = LinearRegression(n_jobs=-1) # To multiple threads in parallel
#clf = svm.SVR()  # To use support vector machine
#clf = svm.SVR(kernel='poly') 

clf.fit(X_train,y_train)  # fit is for training
accuracy = clf.score(X_test,y_test)  # score is for testing

print("Accuracy is",accuracy)
#>>    Accuracy is 0.970299355595
