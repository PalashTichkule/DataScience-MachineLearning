import pandas as pd
from sklearn import cross_validation, preprocessing, model_selection, metrics
from sklearn.linear_model import LinearRegression
import numpy as np

housing_df = pd.read_csv('Housing.csv')
print(housing_df[0:6])
#print(housing_df.columns)
#print(housing_df.isnull().sum())

housing_df['prefarea'] = housing_df['prefarea'].map({ 'yes': 1 , 'no': 0})
housing_df['mainroad'] = housing_df['mainroad'].map({ 'yes': 1 , 'no': 0})
housing_df['airconditioning'] = housing_df['airconditioning'].map({ 'yes': 1 , 'no': 0})
housing_df['guestroom'] = housing_df['guestroom'].map({ 'yes': 1 , 'no': 0})
housing_df['basement'] = housing_df['basement'].map({ 'yes': 1 , 'no': 0})
housing_df['hotwaterheating'] = housing_df['hotwaterheating'].map({ 'yes': 1, 'no' : 0})

dummies = pd.get_dummies(housing_df['furnishingstatus'])

housing_df = pd.concat([housing_df,dummies],axis=1)
print(housing_df[0:6])
housing_df.drop(['furnishingstatus','unfurnished'],axis=1,inplace=True)

print(housing_df[0:6])

def normalize (x): 
    return ( (x-min(x))/ (max(x) - min(x)))
                                                  
# applying normalize ( ) to all columns 
housing_df = housing_df.apply(normalize)

X = housing_df.drop(['price'],axis=1)
y = housing_df['price']


X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,train_size=0.7 ,test_size = 0.3, random_state=100)


clf = LinearRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('Score', clf.score(X_test,y_test))
print('R2', metrics.r2_score(y_test,y_pred))
