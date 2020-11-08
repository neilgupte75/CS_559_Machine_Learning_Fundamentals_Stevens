# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 19:56:49 2019

@author: NEIL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import copy

X=np.array([[5.9,3.2],[4.6,2.9],[6.2 ,2.8],[4.7 ,3.2],[5.5 ,4.2],[5.0 ,3.0],[4.9 ,3.1],[6.7 ,3.1],[5.1 ,3.8],[6.0 ,3.0]])

plt.scatter(X[:,0],X[:,1],s=50)
plt.show()

colors=["r","g","b"]
class K_Means:
    def __init__(self,k=3,tol=0.001,max_iter=300):
        self.k=k
        self.max_iter=max_iter
        self.tol=tol
        
    def fit(self,data):
        self.centroids={
                0: [6.2,3.2],
                1: [6.6,3.7],
                2: [6.5,3.0]
                }
        for i in range(self.max_iter):
            self.classifications={}
            for i in range(self.k):
                self.classifications[i]=[]
            for featureset in data:
                distances=[np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification=distances.index(min(distances))
                self.classifications[classification].append(featureset)
            
            prev_centroids=dict(self.centroids)
            
            for classification in self.classifications:
                pass
                #self.centroids[classification]=np.average(self.classifications[classification],axis=0)
            optimized=True
            
            for c in self.centroids:
                original_centroid=prev_centroids[c]
                current_centroid=self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0)>self.tol:
                    optimized=False
            if optimized:
                break
            
            
            
            
            
    def predict(self,data):
        distances=[np.linalg.norm(data-self.centroids[centroid])for centroid in self.centroid]
        classification=distances.index(min(distances))
        return classification
    
clf=K_Means()
clf.fit(X)


for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker="o",color="k",s=50,linewidths=5)

for classification in clf.classifications:
    color=colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],marker="x",color=color,s=50,linewidths=5)
        