# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:21:49 2019

@author: NEIL
"""


import pandas as pd
import numpy as np
import os
import cv2
import glob
import pdb

dir="C:\CS 559 Machine learning\hw2\face_data";
X_data = []
files = glob.glob ("C:/CS 559 Machine learning/hw2/face_data/*.bmp")
for i in range(0,157):
    print(files[i])
    image1= cv2.imread(files[i],0).flatten()
    #image=image1.reshape(-1)
    X_data.append (image1)

print('X_data shape:', np.array(X_data).shape)
N_train=np.array(X_data)

mu = np.mean(N_train, 0)
ma_data = N_train - mu
a=ma_data[1]

#e_faces, sigma, v = np.linalg.svd(ma_data.transpose(), full_matrices=False)
cov_mat = np.cov(ma_data)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
idx = np . argsort ( - eig_vals )
eigenvalues = eig_vals [ idx ]
eigenvectors = eig_vecs [: , idx ]
eigenvalues = eigenvalues [0: 30 ]. copy ()
eigenvectors = eigenvectors [: ,0: 30 ]. copy ()

np . dot ( ,W)


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

weights = np.dot(ma_data, e_faces)

recon = mu + np.dot(weights[0, 0:30], e_faces[:, 0:30].T)

