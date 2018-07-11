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
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

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

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])

X_train , X_test , y_train , y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

#clf = LinearRegression()
clf = LinearRegression(n_jobs=-1) # To multiple threads in parallel
#clf = svm.SVR()  # To use support vector machine
#clf = svm.SVR(kernel='poly') 

clf.fit(X_train,y_train)  # fit is for training

#Save Classifier
#Required if and only if we want to use the classifier later
with open('LinearRegression.pickle','wb') as f:
    pickle.dump(clf,f)

#Load Saved Classifier
#Not necessarily required, just reloading the classifier saved at previous step for demo purpose 
pickle_in = open('LinearRegression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test)  # score is for testing

#>>    Accuracy is 0.970299355595

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for z in range(len(df.columns) -1 )]  + [i]
    #df.loc[] is used to address index of data frame

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

