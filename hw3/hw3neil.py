# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:57:24 2019

@author: NEIL
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

df = pd.DataFrame({
    'x': [5.9,4.6,6.2,4.7,5.5,5.0,4.9,6.7,5.1,6.0],
    'y': [3.2,2.9,2.8,3.2,4.2,3.0,3.1,3.1,3.8,3.0]
})

k = 3

centroids = {
    1: [6.2,3.2],
    2: [6.6,3.7],
    3: [6.5,3.0]
}

fig = plt.figure(figsize=(15, 15))
plt.scatter(df['x'], df['y'], color='k')
colmap = {1: 'r', 2: 'g', 3: 'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(2.5, 8)
plt.ylim(2.5, 5)
plt.show()


def assignment(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

df = assignment(df, centroids)
print(df.head())

fig = plt.figure(figsize=(15, 15))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(2.5,8)
plt.ylim(2.5,5)
plt.show()









old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

centroids = update(centroids)
    
fig = plt.figure(figsize=(15,15))
ax = plt.axes()
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(2.5,8 )
plt.ylim(2.5, 5)
for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x, old_y, dx, dy, head_width=0.05, head_length=0.1, fc=colmap[i], ec=colmap[i])
plt.show()


df = assignment(df, centroids)
print(df)


fig = plt.figure(figsize=(15, 15))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(2.5,8)
plt.ylim(2.5,5)
plt.show()
number_iterations=1

while True:
    number_iterations=number_iterations+1
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    print(df)

    if closest_centroids.equals(df['closest']):
        break
    
fig = plt.figure(figsize=(15, 15))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(2.5,8)
plt.ylim(2.5,5)
plt.show()

print("Number of iterations :",number_iterations)




