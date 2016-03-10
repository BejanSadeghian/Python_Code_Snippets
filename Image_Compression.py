# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:17:56 2015

@author: Bejan Sadeghian

Purpose: Homework 2 Question 3
"""

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy.linalg import *
from numpy import *
import pylab


img_mat = np.empty((0,10304))
img_path = 'C:\\Users\\beins_000\\Dropbox\\Grad School\\2015 Fall\\Advanced Predictive Modeling\\Homework 2\\Data\\images\\'

for i in range(400):
    img = misc.imread(img_path + str(i+1) + '.pgm')
    plt.imshow(img, cmap=plt.cm.gray)
    img_mat = np.vstack((img_mat, img.flatten()))

##Part A (Calculate the mean vector for each column and subtract the col mean from the observations)
mean_vector = np.mean(img_mat,axis=0)
center_mat = img_mat - mean_vector[np.newaxis,:]

##Part B perform a SVD on the matrix
U, s, V = linalg.svd(center_mat, full_matrices=False)

##Part C, pick k top values and vectors and project onto respective dimensions
k = [1,10,50,100,150,200,250,400]
arr_images = {}

sarr = np.diag(s)
for n,i in enumerate(k):
    Ui = U[:,:i]
    Vi = V[:i,:]
    si = sarr[:i,:i]
    svd_mat = Ui.dot(si).dot(Vi)
    rec_img = {}
    for i in range(400):
        rec_img[i] = np.reshape(mean_vector + svd_mat[i,:], (112,92))
    arr_images[n] = rec_img

##Part D
#test = svd_mat[i,:]

num_image = 10
f = pylab.figure()
f.subplots_adjust(hspace=1,wspace=1)
for i,n in enumerate(k):
    pylab.imshow(arr_images[i][num_image], cmap=plt.cm.gray)
    pylab.show()

