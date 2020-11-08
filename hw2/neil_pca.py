# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:39:31 2019

@author: NEIL
"""

import pandas as pd
import numpy as np
import os
import cv2
import glob

dir="C:\CS 559 Machine learning\hw2\face_data";
X_data = []
files = glob.glob ("C:/CS 559 Machine learning/hw2/face_data/*.bmp")
for i in range(0,157):
    print(files[i])
    image1= cv2.imread(files[i],0)
    image=image1.reshape(-1)
    X_data.append (image)

print('X_data shape:', np.array(X_data).shape)
N_train=np.array(X_data)

X_data1 = []
files = glob.glob ("C:/CS 559 Machine learning/hw2/face_data/*.bmp")
for i in range(158,177):
    print(files[i])
    image1= cv2.imread(files[i],0)
    image=image1.reshape(-1)
    X_data1.append (image)
N_test=np.array(X_data1)

sum_img=0;
for i in range(0,157):
    sum_img += X_data[i];
sum_img=np.array(sum_img)

mean_img=N_train/157;

for i in range(0,157):
    mean_train=X_data[i]-mean_img;

mean_train=np.array(mean_train)

cov_mat=np.cov(mean_train)
    
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
idx = eig_vals.argsort()[::-1]   
eigenValues = eig_vals[idx]
eigenVectors = eig_vecs[:,idx]
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()
"""key = np.argsort(E)[::-1][:30]
E, V = E[key], V[:, key]
U= np.dot(mean_train,V)
matrix_w = np.hstack((eig_pairs[0][1], eig_pairs[1][1]))
"""from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(mean_train)
   """ 
    





