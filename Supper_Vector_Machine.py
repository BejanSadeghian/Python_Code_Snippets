# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 19:42:45 2015

@author: Bejan Sadeghian

Purpose: Homework 4 Question 3
"""

from sklearn import svm, grid_search
import pandas as pd
import numpy as np

raw_train = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 4\diabetes_train-std.csv')
raw_test = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 4\diabetes_test-std.csv')

train_x = raw_train.ix[:,0:8]
train_y = raw_train.ix[:,8]
test_x = raw_test.ix[:,0:8]
test_y = raw_test.ix[:,8]

##Part A (Linear)
#SVM Linear
parameters = {'kernel':['linear'], 'C':[0.001,0.01, 0.1, 1, 2, 3, 4, 5, 6]}

svd_model = svm.SVC()
clf = grid_search.GridSearchCV(svd_model, parameters, cv=10)

#Fit
cv_fitted = clf.fit(train_x, train_y)
best_c = cv_fitted.best_params_['C']

#Model With best C
svd_model = svm.SVC(C=best_c, kernel='linear')
svd_fit = svd_model.fit(train_x, train_y)
predict_y = svd_fit.predict(test_x)
error_rate_linear = np.mean(predict_y != test_y)

#Part B (Gaussian)
parameters = {'kernel':['rbf'], 'C':[0.001,0.01, 0.1, 1, 2, 3, 4, 5, 6]}

svd_model = svm.SVC()
clf = grid_search.GridSearchCV(svd_model, parameters, cv=10)

#Fit
cv_fitted = clf.fit(train_x, train_y)
best_c = cv_fitted.best_params_['C']

#Model With best C
svd_model = svm.SVC(C=best_c, kernel='rbf')
svd_fit = svd_model.fit(train_x, train_y)
predict_y = svd_fit.predict(test_x)
error_rate_rbf = np.mean(predict_y != test_y)














