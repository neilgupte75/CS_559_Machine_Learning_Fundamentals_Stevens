# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 00:25:21 2019

@author: NEIL
"""

import numpy as np
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

iris=load_iris()
X=iris['data']
Y=(iris['target'])

n_x = 4 # size of input layer`
n_h = 3
n_y = 3# size of output layer


W1 = np.random.randn(n_h,n_x) * 0.01
b1 = np.zeros(shape=(n_h, 1))
W2 = np.random.randn(n_y,n_h) * 0.01
b2 = np.zeros(shape=(n_y, 1))


def sigmoid(s):
    # activation function 
    return 1/(1+np.exp(-s))
# Implement Forward Propagation to calculate A2 (probabilities)
Z1 = np.dot(W1,X.T) + b1
A1 = np.tanh(Z1)
Z2 = np.dot(W2,A1) + b2
A2 = np.tanh(Z2) # Final output prediction

m=150
logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
cost = - np.sum(logprobs) / m    
    
cost = np.squeeze(cost)

dZ2 = A2 - Y
dW2 = (1 / m) * np.dot(dZ2, A1.T)
db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
dW1 = (1 / m) * np.dot(dZ1, X)
db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

learning_rate = 1.2
W1 = W1 - learning_rate * dW1
b1 = b1 - learning_rate * db1
W2 = W2 - learning_rate * dW2
b2 = b2 - learning_rate * db2

class NeuralNetwork(object):
    def __init__(self):
        #param
        self.inputSize=4
        self.outputSize=3
        self.hiddensize=3
        
        #weights
        self.W1=np.random.randn(self.inputSize,self.hiddenSize)
        self.W2=np.random.randn(self.hiddenSize,self.outputSize)
        
    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3) # final activation function
        return o 

    def sigmoid(self, s):
        # activation function 
        return 1/(1+np.exp(-s))
        
    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)
        
    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error
        
        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error
        
        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights
        
    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

        
        
        
        
        
        