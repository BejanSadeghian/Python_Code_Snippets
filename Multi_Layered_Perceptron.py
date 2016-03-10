# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:31:42 2015

@author: Bejan Sadeghian

Purpose: Homework 3 Question 5
"""

from sknn.mlp import Classifier, Layer
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

raw_train = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 3\pendigits_train.csv', header=None)
raw_test = pd.read_csv(r'C:\Users\beins_000\Dropbox\Grad School\2015 Fall\Advanced Predictive Modeling\Homework 3\pendigits_test.csv', header=None)

train_x = np.array(raw_train.ix[:,:15])
train_y = np.array(raw_train.ix[:,16])
test_x = np.array(raw_test.ix[:,:15])
test_y = np.array(raw_test.ix[:,16])


#Create multidimensional train_y  and test_y
train_y_array = np.array([None] * len(train_y))
test_y_array = np.array([None] * len(test_y))


for i in set(train_y):
    temp_col = train_y == i
    train_y_array = np.column_stack((train_y_array, temp_col))
    temp_col = test_y == i
    test_y_array = np.column_stack((test_y_array, temp_col))

train_y_df = pd.DataFrame(train_y_array).replace(to_replace=False, value = 0)
train_y_df.replace(to_replace = True, value = 1, inplace = True)
train_y_df.drop(0, axis=1, inplace = True)
train_y_df.columns = [0,1,2,3,4,5,6,7,8,9] #Renaming the columns appropriately

test_y_df = pd.DataFrame(test_y_array).replace(to_replace=False, value = 0)
test_y_df.replace(to_replace = True, value = 1, inplace = True)
test_y_df.drop(0, axis=1, inplace = True)
test_y_df.columns = [0,1,2,3,4,5,6,7,8,9] #Renaming the columns appropriately


#Begin Neural Net
hidden_units = [10, 20]
learning_rate = 0.01
epochs = [50, 100]

for hu in hidden_units:
    for ep in epochs:
        mlp_layer1 = Layer('Sigmoid', name='HL1', units = hu)
        mlp_layer_out = Layer('Sigmoid', name='Out')
        
        nn = Classifier([mlp_layer1, mlp_layer_out], learning_rate = learning_rate, n_iter = ep)
        
        nn.fit(train_x, np.array(train_y_df))
        
        predict_y = nn.predict(test_x)
        predict_y = pd.DataFrame(predict_y)
        
        print'F1 Score for Hidden Units:', hu, 'and Max Epochs:', ep
        print f1_score(test_y_df, predict_y, average=None)
        print'Average F1 Score', f1_score(test_y_df, predict_y)
        
        
        #Recombine the arrays
        test_y_comb = [0] * len(test_y_df)
        for i in xrange(len(test_y_df)):
            counter = 0
            for j in test_y_df.ix[i,:]:
                if j == 1:
                    test_y_comb[i] = counter
                    break
                elif j >9:
                    test_y_comb[i] = 10
                else:
                    counter = counter + 1
        
        predict_y_comb = [0] * len(predict_y)
        for i in xrange(len(predict_y)):
            counter = 0
            for j in predict_y.ix[i,:]:
                if j == 1:
                    predict_y_comb[i] = counter
                    break
                else:
                    counter = counter + 1
        
        print pd.crosstab(pd.Series(test_y_comb, name='Ground Truth'), pd.Series(predict_y_comb, name='Predicted'))






