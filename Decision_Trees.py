# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 14:02:18 2015

@author: Bejan Sadeghian
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from sklearn import tree
from sklearn import metrics
from collections import Counter

rawdata = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 3\BreastCancer.csv')
raw_test = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 3\test_idx.csv', header=None)
raw_train1 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 3\train_idx1.csv', header=None)
raw_train2 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 3\train_idx2.csv', header=None)


##Part B i)

#Train Set 1 Parsing
train_set1 = rawdata.ix[raw_train1[0],:]
train_set1.drop('id', axis=1, inplace=True)
train1_X = train_set1.ix[:,1:] #Extract Predictors Only
train1_Y = train_set1.ix[:,0] #Extract Responses Only

names = train1_X.columns

#Generate Models
gini_model = DecisionTreeClassifier(criterion = 'gini', max_depth=2)
entropy_model = DecisionTreeClassifier(criterion ='entropy', max_depth=2)

gini_fit = gini_model.fit(train1_X, train1_Y)
entropy_fit = entropy_model.fit(train1_X, train1_Y)

with open(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 3\Submission\ibreastcance_gini.dot', 'w') as f:
    f = tree.export_graphviz(gini_fit, out_file=f, feature_names=names)

with open(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 3\Submission\ibreastcance_entropy.dot', 'w') as f:
    f = tree.export_graphviz(entropy_fit, out_file=f, feature_names=names)


##Part B ii)

#Train Set 2 Parsing
train_set2 = rawdata.ix[raw_train2[0],:]
train_set2.drop('id', axis=1, inplace=True)
train2_X = train_set2.ix[:,1:] #Extract Predictors Only
train2_Y = train_set2.ix[:,0] #Extract Responses Only

#Generate Model
gini_model2 = DecisionTreeClassifier(criterion = 'gini', max_depth=2)
gini_fit2 = gini_model2.fit(train2_X, train2_Y)

#Test Set Parsing
test_set = rawdata.ix[raw_test[0],:]
test_set.drop('id', axis=1, inplace=True)
test_X = test_set.ix[:,1:] #Extract Predictors Only
test_Y = test_set.ix[:,0] #Extract Responses Only

#Predict for both Gini Models
gini_predict_1 = gini_fit.predict(test_X)
gini_predict_2 = gini_fit2.predict(test_X)

with open(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 3\Submission\ibreastcance_gini_model2_set2.dot', 'w') as f:
    f = tree.export_graphviz(gini_fit2, out_file=f, feature_names=names)


#Calculate the F1 Scores
gini_f1_1_B = metrics.f1_score(test_Y, gini_predict_1, pos_label = 'B')
gini_f1_1_M = metrics.f1_score(test_Y, gini_predict_1, pos_label = 'M')
gini_f1_2_B = metrics.f1_score(test_Y, gini_predict_2, pos_label = 'B')
gini_f1_2_M = metrics.f1_score(test_Y, gini_predict_2, pos_label = 'M')


#Predicting With the third model and finding the majority
entropy_predict = entropy_fit.predict(test_X)

entropy_predict = pd.Series(entropy_predict, name='Entropy')
gini_predict_1 = pd.Series(gini_predict_1, name='Gini_1')
gini_predict_2 = pd.Series(gini_predict_2, name='Gini_2')

all_predictions = pd.concat([entropy_predict,gini_predict_1,gini_predict_2], axis=1)

final_prediction = pd.Series([None]*len(all_predictions), name='Final Prediction')

for i in range(len(all_predictions)):
    temp_series = pd.Series([all_predictions.ix[i,'Entropy'], all_predictions.ix[i,'Gini_1'], all_predictions.ix[i,'Gini_2']])
    top_predict = Counter(temp_series)
    final_prediction[i] = [ x for (x, y) in top_predict.most_common(1)][0]


Final_f1_B = metrics.f1_score(test_Y, final_prediction, pos_label = 'B')
Final_f1_M = metrics.f1_score(test_Y, final_prediction, pos_label = 'M')



