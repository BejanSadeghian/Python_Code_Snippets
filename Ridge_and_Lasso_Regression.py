# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 19:00:40 2015

@author: Bejan Sadeghian

Purpose: Homework 5
"""

import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn import linear_model
from scipy import stats

trainraw = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 2\Data\bostonderived_train.csv')
testraw = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 2\Data\bostonderived_test.csv')
folds = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 2\Data\bostonderived_folds.csv', header=None)
test = testraw[['lstat', 'rm','chas','indus','tax','rad','black','medv']]
train = trainraw[['lstat', 'rm','chas','indus','tax','rad','black','medv']]
train['Fold'] = folds

##Part A
ridge_para = [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1,1,10,100,1000,10000]
lasso_para = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

dict_lasso = {}
lasso_coef = {}
for alp in lasso_para:
    error = 0
    lasso_model = sklearn.linear_model.Lasso(alpha = alp)
    for i in range(5):
        holdout_cv = train[train['Fold']==i]
        train_cv = train[train['Fold']!=i]
        X = train_cv.as_matrix(['lstat','rm','chas','indus','tax','rad','black'])
        preprocessing.scale(X, axis=0, copy=False)
        y = train_cv.as_matrix(['medv'])
        lasso_fit = lasso_model.fit(X,y)
        
        X_hold = holdout_cv.as_matrix(['lstat','rm','chas','indus','tax','rad','black'])
        preprocessing.scale(X_hold, axis=0, copy=False)
        y_hold = np.array(holdout_cv['medv'])
        y_predict = lasso_fit.predict(X_hold)
        #print('Fold $d with alpha %.5f' % i, alpha)
        #print(np.average((y_predict - y_hold)**2))
        error = error + (len(holdout_cv)*math.sqrt(np.average((y_predict - y_hold)**2)))
    if dict_lasso.get(alp, "empty") == 'empty':
        dict_lasso[alp] = error/len(train)
    lasso_coef[alp] = lasso_fit.coef_
            
print('Average K-Fold CV for LASSO')
print dict_lasso

ridge_coef = {}
dict_ridge = {}
error = 0
for alp in ridge_para:
    error = 0
    ridge_model = sklearn.linear_model.Ridge(alpha = alp)
    for i in range(5):
        holdout_cv = train[train['Fold']==i]
        train_cv = train[train['Fold']!=i]
        X = train_cv.as_matrix(['lstat','rm','chas','indus','tax','rad','black'])
        preprocessing.scale(X, axis=0, copy=False)
        y = train_cv.as_matrix(['medv'])
        ridge_fit = ridge_model.fit(X,y)
        
        X_hold = holdout_cv.as_matrix(['lstat','rm','chas','indus','tax','rad','black'])
        preprocessing.scale(X_hold, axis=0, copy=False)
        y_hold = holdout_cv.as_matrix(['medv'])
        y_predict = ridge_fit.predict(X_hold)
        #print('Fold $d with alpha %.5f' % i, alpha)
        #print(np.average((y_predict - y_hold)**2))
        error = error + (len(holdout_cv)*math.sqrt(np.average((y_predict - y_hold)**2)))
    if dict_ridge.get(alp, "empty") == 'empty':
        dict_ridge[alp] = error/len(train)
    ridge_coef[alp] = ridge_fit.coef_
    #print(ridge_fit.coef_)
    
print('Average K-Fold CV for Ridge')
print dict_ridge
            
##Part B
lasso_coef_df = pd.DataFrame()
for i in lasso_para:
    lasso_coef_df[i] = lasso_coef[i]
lasso_coef_df = lasso_coef_df.T
lasso_coef_df['Alpha'] = lasso_para

ridge_coef_df = pd.DataFrame()
for i in ridge_para:
    ridge_coef_df[i] = ridge_coef[i][0]
ridge_coef_df = ridge_coef_df.T
ridge_coef_df['Alpha'] = ridge_para

plt.figure(figsize=(10,7))
plt.plot(ridge_coef_df['Alpha'],ridge_coef_df[[0]])
plt.plot(ridge_coef_df['Alpha'],ridge_coef_df[[1]])
plt.plot(ridge_coef_df['Alpha'],ridge_coef_df[[2]])
plt.plot(ridge_coef_df['Alpha'],ridge_coef_df[[3]])
plt.plot(ridge_coef_df['Alpha'],ridge_coef_df[[4]])
plt.plot(ridge_coef_df['Alpha'],ridge_coef_df[[5]])
plt.plot(ridge_coef_df['Alpha'],ridge_coef_df[[6]])
plt.legend(['lstat','rm','chas','indus','tax','rad','black'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Ridge Regression (Larger Sweep)')
plt.xlabel('Beta Values')
plt.ylabel('Normalization Parameter')
#plt.ylim(0.1)
plt.show()

plt.figure(figsize=(10,7))
plt.plot(lasso_coef_df['Alpha'],lasso_coef_df[[0]])
plt.plot(lasso_coef_df['Alpha'],lasso_coef_df[[1]])
plt.plot(lasso_coef_df['Alpha'],lasso_coef_df[[2]])
plt.plot(lasso_coef_df['Alpha'],lasso_coef_df[[3]])
plt.plot(lasso_coef_df['Alpha'],lasso_coef_df[[4]])
plt.plot(lasso_coef_df['Alpha'],lasso_coef_df[[5]])
plt.plot(lasso_coef_df['Alpha'],lasso_coef_df[[6]])
plt.legend(['lstat','rm','chas','indus','tax','rad','black'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('LASSO Regression')
plt.xlabel('Beta Values')
plt.ylabel('Normalization Parameter')
plt.show()



##Part C

#Best Alphas for Ridge and Lasso
alpha_lasso = 0.2 #CV RMSE of 5.6774 (based on centered and scaled data)
alpha_ridge = 0.1 #CV RMSE of 5.6920 (Barely better than any other alpha) (based on centered and scaled data)

#This time not centering data since it was not requested (to be used for both LASSO and Ridge)
X = train.as_matrix(['lstat','rm','chas','indus','tax','rad','black'])
preprocessing.scale(X, axis=0, copy=False)
y = train.as_matrix(['medv'])

#------------Lasso best------------
lasso_model = sklearn.linear_model.Lasso(alpha = alpha_lasso)
lasso_fit = lasso_model.fit(X,y)

#test set
X_test = test.as_matrix(['lstat','rm','chas','indus','tax','rad','black'])
preprocessing.scale(X_test, axis=0, copy=False)
y_test = np.array(test['medv'])

#predict and calculate RMSE
y_predict = lasso_fit.predict(X_test)
print('LASSO best')
print(math.sqrt(np.average((y_predict - y_test)**2))) #RMSE of 5.0743


#------------Ridge best------------
ridge_model = sklearn.linear_model.Ridge(alpha = alpha_ridge)
ridge_fit = ridge_model.fit(X,y)

#test set
X_test = test.as_matrix(['lstat','rm','chas','indus','tax','rad','black'])
preprocessing.scale(X_test, axis=0, copy=False)
y_test = test.as_matrix(['medv'])

#predict and calculate RMSE
y_predict = ridge_fit.predict(X_test)
print('Ridge best')
print(math.sqrt(np.average((y_predict - y_test)**2))) #RMSE of 5.0090

#------------OLS------------
#Must recreate X and y since the Linear Regression method accepts np arrays instead of matrices
X = np.array(train[['lstat','rm','chas','indus','tax','rad','black']])
preprocessing.scale(X, axis=0, copy=False)
y = np.array(train['medv'])

#Model fit
ols_model = sklearn.linear_model.LinearRegression()
ols_fit = ols_model.fit(X,y)

#test set
X_test = test.as_matrix(['lstat','rm','chas','indus','tax','rad','black']) 
preprocessing.scale(X_test, axis=0, copy=False)
y_test = np.array(test['medv'])

#predict and calculate RMSE
y_predict = ols_fit.predict(X_test)
print('OLS best')
print(math.sqrt(np.average((y_predict - y_test)**2))) #RMSE of 5.0090

##Part D
lasso_fit.coef_ #array([-3.6131463 , 3.65175444 , 0.62009438 , -0. , -0.76302161 , -0. , 0.68751515])
#Lasso removed 'indus' and 'rad'

#------------OLS------------
#Must recreate X and y since the Linear Regression method accepts np arrays instead of matrices
X = np.array(train[['lstat','rm','chas','tax','black']])
preprocessing.scale(X, axis=0, copy=False)
y = np.array(train['medv'])

#Model fit
ols_model = sklearn.linear_model.LinearRegression()
ols_fit = ols_model.fit(X,y)

#test set
X_test = test.as_matrix(['lstat','rm','chas','tax','black']) 
preprocessing.scale(X_test, axis=0, copy=False)
y_test = np.array(test['medv'])

#predict and calculate RMSE
y_predict = ols_fit.predict(X_test)
print('OLS (5 params)')
print(math.sqrt(np.average((y_predict - y_test)**2))) #RMSE of 5.0369

##Part E
#See Document