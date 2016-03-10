# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:14:48 2015

@author: Bejan Sadeghian

Purpose: Homework 5 Question 1
"""

from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn import cross_validation
import pandas as pd
import numpy as np
from ggplot import *
import random

random.seed(1)
raw_train = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\diabetes_train-std.csv')
raw_test = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\diabetes_test-std.csv')

train_x = raw_train.ix[:,0:8]
train_y = np.array(raw_train[['classvariable']])[:,0]

test_x = raw_test.ix[:,0:8]
test_y = np.array(raw_test[['classvariable']])[:,0]

##Part A - Classification Tree
tree_model = tree.DecisionTreeClassifier()

fitted_model = tree_model.fit(train_x, train_y)

#Outputting the dot tree plot file
with open(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\Submission\part_a_tree.dot','w') as f:
    f = tree.export_graphviz(fitted_model, out_file=f)

#Predicting and making the confusion matrix
predicted = fitted_model.predict(test_x)

CrossTable = pd.crosstab(pd.Series(test_y, name='Actual'), pd.Series(predicted, name='Predicted'))

ErrorRate = np.mean(predicted != test_y)


##Part B - Random Forest
max_trees = range(1,100, 5)
max_para = [2, 3, 4, 5, 6, 7]
RF_Error_Rate = list()
Max_Trees_List = list()
Max_Para_List = list()
fold_index = cross_validation.KFold(len(test_y), n_folds=5)

#search for best tree number to use and number of parameters to use
for i in max_trees:
    for j in max_para:
        temp_error = list()

        for train_index, test_index in fold_index:
            temp_train = raw_train.ix[train_index,:]
            temp_test = raw_train.ix[test_index,:]
            
            temp_train_x = temp_train.drop('classvariable', axis=1)
            temp_train_y = np.array(temp_train[['classvariable']])[:,0]
            temp_test_x = temp_test.drop('classvariable', axis=1)
            temp_test_y = np.array(temp_test[['classvariable']])[:,0]
            
            RF_model = ensemble.RandomForestClassifier(n_estimators=i, max_features=j)
            RF_fitted_model = RF_model.fit(temp_train_x, temp_train_y)
            RF_predicted = RF_fitted_model.predict(temp_test_x)
            CrossTable_RF = pd.crosstab(pd.Series(temp_test_y, name='Actual'), pd.Series(RF_predicted, name='Predicted'))
            temp_error.append(np.mean(RF_predicted != temp_test_y))
        
        RF_Error_Rate.append(np.mean(temp_error))
        Max_Trees_List.append(i)
        Max_Para_List.append(j)
RF_dataframe = pd.concat([pd.Series(Max_Para_List, name='Parameters'), pd.Series(Max_Trees_List, name='Trees'), pd.Series(RF_Error_Rate, name='Error')], axis=1)

ggplot(RF_dataframe, aes(x='Trees', y='Error', colour='Parameters')) + geom_line() 

#Now fitting the best tree
RF_model1 = ensemble.RandomForestClassifier(n_estimators=30, max_features=2)
RF_fitted_model = RF_model1.fit(train_x, train_y)
RF_predicted = RF_fitted_model.predict(test_x)
CrossTable_RF = pd.crosstab(pd.Series(test_y, name='Actual'), pd.Series(RF_predicted, name='Predicted'))
np.mean(test_y != RF_predicted)

##Part C - Gradient Boosted Decision Tree
learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
estimators = range(50,500,50)
depth = [2,4,6,8,10,12,14,16]
errors = list()
learning_e = list()
estimators_e = list()
depth_e = list()

for i in learning_rate:
    for j in estimators:
        for k in depth:
            for train_index, test_index in fold_index:
                temp_train = raw_train.ix[train_index,:]
                temp_test = raw_train.ix[test_index,:]
                temp_train_x = temp_train.drop('classvariable', axis=1)
                temp_train_y = np.array(temp_train[['classvariable']])[:,0]
                temp_test_x = temp_test.drop('classvariable', axis=1)
                temp_test_y = np.array(temp_test[['classvariable']])[:,0]
                
                
                GB_model = ensemble.GradientBoostingClassifier(loss='exponential',n_estimators=j, learning_rate=i, max_depth=k)
                GB_fit = GB_model.fit(temp_train_x, temp_train_y)
                GB_predict = GB_fit.predict(temp_test_x)
                errors.append(np.mean(GB_predict != temp_test_y))
                learning_e.append(i)
                estimators_e.append(j)
                depth_e.append(k)
GB_dataframe = pd.concat([pd.Series(estimators_e, name='Estimators'), pd.Series(learning_e, name='Learning_Rate'), pd.Series(depth_e, name='Max_Depth'), pd.Series(errors, name='Error')], axis=1)
group_df = GB_dataframe.groupby(['Estimators','Learning_Rate','Max_Depth']).mean().reset_index()
#GB_dataframe = pd.concat([pd.Series(estimators_e, name='Estimators'), pd.Series(learning_e, name='Learning_Rate'), pd.Series(depth_e, name='Max_Depth'), pd.Series(np.array(group_df)[:,0], name='Error')], axis=1)


#melt = pd.melt(GB_dataframe, id_vars=')
plot= ggplot(group_df, aes(x='Estimators', y='Error', colour='Learning_Rate')) + geom_point() + facet_wrap("Max_Depth")
ggsave(filename=r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\Submission\facet_PartC_exp.jpg', plot=plot)

GB_model = ensemble.GradientBoostingClassifier(n_estimators=250, learning_rate=0.01, max_depth=2)
GB_fit = GB_model.fit(train_x, train_y)
GB_predict = GB_fit.predict(test_x)
np.mean(test_y != GB_predict)
CrossTable_GB = pd.crosstab(pd.Series(test_y, name='Actual'), pd.Series(GB_predict, name='Predicted'))


