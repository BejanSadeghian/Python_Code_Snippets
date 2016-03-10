# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:01:10 2015

@author: Bejan Sadeghian

Purpose: Homework 5 Question 2
"""
import pandas as pd
import numpy as np
import random
import math
import glob
from sklearn import linear_model
from sklearn import metrics

random.seed(1)
raw_train_x = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_train1.csv', header = None)
raw_train_y = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_tr_label1.csv', header=None)

##Part A
def dnsmpl(xtrain, ytrain, pos_ratio=0.1): 
    temp_y = ytrain.ix[:,0]
    temp_x = xtrain
    count_0 = temp_y.value_counts()[0]
    count_1 = temp_y.value_counts()[1]
    index_0 = temp_y.ix[temp_y==0].index.values
    index_1 = temp_y.ix[temp_y==1].index.values
    sample_1 = index_1
    if float(count_1)/count_0 >= pos_ratio:
        sample_0 = index_0
    else:
        sample_0 = random.sample(index_0, int(math.floor(len(index_1)/pos_ratio)-len(index_1)-1)) #-1 for greater than
    xtrain_sample = pd.concat([temp_x.ix[sample_0,:],temp_x.ix[sample_1,:]])
    ytrain_sample = pd.concat([temp_y.ix[sample_0],temp_y.ix[sample_1]])
    return (xtrain_sample, ytrain_sample)

#Read in files
raw_train_x1 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_train1.csv', header = None).ix[:,1:]
raw_train_y1 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_tr_label1.csv', header=None)
raw_test_x1 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_test1.csv', header = None).ix[:,1:]
raw_test_y1 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_tst_label1.csv', header = None)
train = raw_train_x1
for i in train.columns:
    if i < 3:
        continue
    raw_train_x1.ix[:,i] = (raw_train_x1.ix[:,i] - np.mean(train.ix[:,i]))/np.std(train.ix[:,i])
    raw_test_x1.ix[:,i] = (raw_test_x1.ix[:,i] - np.mean(train.ix[:,i]))/np.std(train.ix[:,i])

raw_train_x2 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_train2.csv', header = None).ix[:,1:]
raw_train_y2 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_tr_label2.csv', header=None)
raw_test_x2 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_test2.csv', header = None).ix[:,1:]
raw_test_y2 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_tst_label2.csv', header = None)
train = raw_train_x2
for i in train.columns:
    if i < 3:
        continue
    raw_train_x2.ix[:,i] = (raw_train_x2.ix[:,i] - np.mean(train.ix[:,i]))/np.std(train.ix[:,i])
    raw_test_x2.ix[:,i] = (raw_test_x2.ix[:,i] - np.mean(train.ix[:,i]))/np.std(train.ix[:,i])

raw_train_x3 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_train3.csv', header = None).ix[:,1:]
raw_train_y3 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_tr_label3.csv', header=None)
raw_test_x3 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_test3.csv', header = None).ix[:,1:]
raw_test_y3 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_tst_label3.csv', header = None)
train = raw_train_x3
for i in train.columns:
    if i < 3:
        continue
    raw_train_x3.ix[:,i] = (raw_train_x3.ix[:,i] - np.mean(train.ix[:,i]))/np.std(train.ix[:,i])
    raw_test_x3.ix[:,i] = (raw_test_x3.ix[:,i] - np.mean(train.ix[:,i]))/np.std(train.ix[:,i])

raw_train_x4 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_train4.csv', header = None).ix[:,1:]
raw_train_y4 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_tr_label4.csv', header=None)
raw_test_x4 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_test4.csv', header = None).ix[:,1:]
raw_test_y4 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_tst_label4.csv', header = None)
train = raw_train_x4
for i in train.columns:
    if i < 3:
        continue
    raw_train_x4.ix[:,i] = (raw_train_x4.ix[:,i] - np.mean(train.ix[:,i]))/np.std(train.ix[:,i])
    raw_test_x4.ix[:,i] = (raw_test_x4.ix[:,i] - np.mean(train.ix[:,i]))/np.std(train.ix[:,i])

raw_train_x5 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_train5.csv', header = None).ix[:,1:]
raw_train_y5 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_tr_label5.csv', header=None)
raw_test_x5 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_test5.csv', header = None).ix[:,1:]
raw_test_y5 = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 5\vowel_data\vowel_data\vowel_tst_label5.csv', header = None)
train = raw_train_x5
for i in train.columns:
    if i < 3:
        continue
    raw_train_x5.ix[:,i] = (raw_train_x5.ix[:,i] - np.mean(train.ix[:,i]))/np.std(train.ix[:,i])
    raw_test_x5.ix[:,i] = (raw_test_x5.ix[:,i] - np.mean(train.ix[:,i]))/np.std(train.ix[:,i])

#Run logistic regression
F1_score = list()
p_val = list()
for j in [0.1, 0.15, 0.2, 0.25, 0.3]:
    for i in range(1,6):
        exec 'temp_train_x = raw_train_x%s' % (i)
        exec 'temp_train_y = raw_train_y%s' % (i)
        exec 'temp_test_x = raw_test_x%s' % (i)
        exec 'temp_test_y = raw_test_y%s' % (i)
        train_x, train_y = dnsmpl(temp_train_x, temp_train_y, j)
        model = linear_model.LogisticRegression()
        fitted_model = model.fit(train_x, train_y)
        predict = fitted_model.predict(temp_test_x)
        F1_score.append(metrics.f1_score(temp_test_y, predict))
        p_val.append(j)
df = pd.concat([pd.Series(p_val, name='pos_ratio'),pd.Series(F1_score, name='F1_score')], axis=1)
scores_dn = df.groupby('pos_ratio').mean().reset_index()
scores_dn[['STD']] = df.groupby('pos_ratio').std().reset_index()[['F1_score']]

#Part B
def upsmpl(xtrain, ytrain, pos_ratio=0.1):
    temp_y = ytrain.ix[:,0]
    temp_x = xtrain
    count_0 = temp_y.value_counts()[0]
    count_1 = temp_y.value_counts()[1]
    index_0 = temp_y.ix[temp_y==0].index.values
    index_1 = temp_y.ix[temp_y==1].index.values
    sample_0 = index_0
    if float(count_1)/count_0 == pos_ratio/(1-float(pos_ratio)):
        sample_1 = index_1
    else:
        sample_1 = np.random.choice(index_1, size=math.floor((pos_ratio/(1-pos_ratio))*count_0), replace=True)
    
    xtrain_sample = pd.concat([temp_x.ix[sample_0,:],temp_x.ix[sample_1,:]])
    ytrain_sample = pd.concat([temp_y.ix[sample_0],temp_y.ix[sample_1]])
    return (xtrain_sample, ytrain_sample)

#Run logistic regression
F1_score = list()
p_val = list()
for j in [0.1, 0.15, 0.2, 0.25, 0.3]:
    for i in range(1,6):
        exec 'temp_train_x = raw_train_x%s' % (i)
        exec 'temp_train_y = raw_train_y%s' % (i)
        exec 'temp_test_x = raw_test_x%s' % (i)
        exec 'temp_test_y = raw_test_y%s' % (i)
        train_x, train_y = upsmpl(temp_train_x, temp_train_y, j)
        model = linear_model.LogisticRegression()
        fitted_model = model.fit(train_x, train_y)
        predict = fitted_model.predict(temp_test_x)
        F1_score.append(metrics.f1_score(temp_test_y, predict))
        p_val.append(j)
df = pd.concat([pd.Series(p_val, name='pos_ratio'),pd.Series(F1_score, name='F1_score')], axis=1)
scores_up = df.groupby('pos_ratio').mean().reset_index()
scores_up[['STD']] = df.groupby('pos_ratio').std().reset_index()[['F1_score']]

##Part C
F1_score = list()
for i in range(1,6):
    exec 'temp_train_x = raw_train_x%s' % (i)
    exec 'temp_train_y = raw_train_y%s' % (i)
    exec 'temp_test_x = raw_test_x%s' % (i)
    exec 'temp_test_y = raw_test_y%s' % (i)
    model = linear_model.LogisticRegression(class_weight={0:1,1:2})
    fitted_model = model.fit(temp_train_x, temp_train_y)
    predict = fitted_model.predict(temp_test_x)
    F1_score.append(metrics.f1_score(temp_test_y, predict))
df = pd.Series(F1_score, name='F1_score')
scores_weight = df.mean()
scores_weight_std = df.std()

##Part D
F1_score = list()
for i in range(1,6):
    exec 'temp_train_x = raw_train_x%s' % (i)
    exec 'temp_train_y = raw_train_y%s' % (i)
    exec 'temp_test_x = raw_test_x%s' % (i)
    exec 'temp_test_y = raw_test_y%s' % (i)
    model = linear_model.LogisticRegression()
    fitted_model = model.fit(temp_train_x, temp_train_y)
    predict = fitted_model.predict(temp_test_x)
    F1_score.append(metrics.f1_score(temp_test_y, predict))
df = pd.Series(F1_score, name='F1_score')
scores_base = df.mean()
scores_base_std = df.std()