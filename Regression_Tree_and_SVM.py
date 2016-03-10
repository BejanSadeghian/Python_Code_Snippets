# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 19:30:22 2015

@author: Bejan Sadeghian

Purpose: Homework 4 Question 4
"""
import pandas as pd
import numpy as np
from ggplot import *
from sklearn import tree
from sklearn import svm

raw_data = pd.read_table(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 4\nist_data.txt', sep=' ', header=None, names=['Target','Covariate'])
raw_train = pd.read_table(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 4\nist_train.txt', sep=' ', header=None)
raw_test = pd.read_table(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 4\nist_test.txt', sep=' ', header=None)


#Part A
depth = [4,6,8,10,12,14]
MSE_list = list()
train_set = raw_data.ix[raw_train[0],:]
test_set = raw_data.ix[raw_test[0],:]

x_train = train_set[['Covariate']]
y_train = (train_set['Target'])
x_test = test_set[['Covariate']]
y_test = (test_set['Target'])

for i, d in enumerate(depth):
    tree_model = tree.DecisionTreeRegressor(max_depth=d)
    fitted_model = tree_model.fit(x_train, y_train)
    predicted = fitted_model.predict(x_test)
    MSE_list.append(np.mean((predicted-y_test)**2))

Tree_MSE = pd.concat([pd.Series(depth), pd.Series(MSE_list)], axis=1)
Tree_MSE.columns = ['Depth','MSE']

ggplot(Tree_MSE, aes(x='Depth', y='MSE', label = 'Depth')) + geom_text(hjust=0, vjust=0.5) + geom_line(colour='blue') + geom_point(colour='red') + ggtitle('Depth vs MSE') + xlab('Depth') + ylab('MSE')


#Part B
gamma = [0.0001,0.001,0.01,0.1]
y_train = np.array(train_set[['Target']])
MSE_list = list()

for i, g in enumerate(gamma):
    svm_model = svm.SVR(kernel = 'rbf', gamma=g, C=1000)
    svm_fitted = svm_model.fit(x_train, y_train.squeeze())
    predicted_svm = svm_fitted.predict(x_test)
    MSE_list.append(np.mean((predicted_svm-y_test)**2))

SVM_MSE = pd.concat([pd.Series(np.log(gamma)), pd.Series((gamma)), pd.Series(MSE_list)], axis=1)
SVM_MSE.columns = ['Log Gamma', 'Gamma','MSE']

ggplot(SVM_MSE, aes(x='Log Gamma',y='MSE', label = 'Gamma')) + geom_text(hjust=0, vjust=0.5) + geom_line(colour='blue') + geom_point(colour='red') + ggtitle('Log Gamma vs MSE (Labels are Normal Gamma)') + xlab('Log Gamma') + ylab('MSE')

#Part C
#Best Depth and Gamma
tree_model = tree.DecisionTreeRegressor(max_depth=10) #Best Depth is 10
fitted_model = tree_model.fit(x_train, y_train)
predicted = fitted_model.predict(x_test)
svm_model = svm.SVR(kernel = 'rbf', gamma=0.001, C=1000) #Best Gamma is 0.001
svm_fitted = svm_model.fit(x_train, y_train.squeeze())
predicted_svm = svm_fitted.predict(x_test)

#First plot
y_test = pd.Series(y_test).reset_index()
x_test = x_test.reset_index()
plot1 = pd.concat([y_test.ix[:,1], pd.Series(predicted), pd.Series(predicted_svm), x_test.ix[:,1]], axis=1)
plot1.columns = ['Y_Actual', 'Tree_Predict','SVR_Predict', 'Covariate']
plot1 = pd.melt(plot1, id_vars=['Covariate'], value_vars=['Y_Actual', 'Tree_Predict', 'SVR_Predict'])
ggplot(plot1, aes(x='Covariate  ', y='value', colour='variable')) + geom_point() 

#Second plot
plot2 = pd.concat([y_test.ix[:,1], pd.Series(predicted), pd.Series(predicted_svm)], axis=1)
plot2.columns = ['Y_Actual', 'Tree_Predict','SVR_Predict']
plot2 = pd.melt(plot2, id_vars=['Y_Actual'], value_vars=['Tree_Predict', 'SVR_Predict'])
ggplot(plot2, aes(x='Y_Actual  ', y='value', colour='variable')) + geom_point() 









