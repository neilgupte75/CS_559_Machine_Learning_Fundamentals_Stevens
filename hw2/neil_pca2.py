# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:28:30 2019
@author: NEIL
"""
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image

files = [img for img in glob.glob('face_data/*.bmp')]

train=files[:157]
test=files[157:]

alldata = []
disp = []

for face_images in train:
    face_image=Image.open(face_images)
    disp.append(face_image)
    face_image = np.asarray(face_image,dtype=float)/255.0
    alldata.append(face_image)
    

size=256,256

fig, axes = plt.subplots(14, 11)
count=0
for x in range(14):
    for y in range (11):
        draw_image = disp[count]
        draw_image.thumbnail(size)
        draw_image= np.asarray(draw_image,dtype=float)/255.0
        axes[x][y].imshow(draw_image,cmap=plt.cm.gray)
        axes[x][y].axis('off')
        count=count+1
        
fig.canvas.set_window_title('Showing faces')
plt.show()
alldata=np.asarray(alldata)

mean = np.mean(alldata, 0)
fig, axes = plt.subplots(1, 1)
axes.imshow(mean, cmap=plt.cm.gray)
fig.canvas.set_window_title('mean faces')
plt.show()

flatten_Array = []
for x in range(len(alldata)):
    flat_Array = alldata[x].flatten()
    flatten_Array.append(flat_Array)
flatten_Array = np.asarray(flatten_Array)
mean = mean.flatten()

substract_mean_from_original = np.subtract(flatten_Array, mean)
U, s, V = np.linalg.svd(substract_mean_from_original, full_matrices=False)
Eigen_faces=[]
for x in range(V.shape[0]):
    fig=np.reshape(V[x],(256,256))
    Eigen_faces.append(fig)

fig, axes = plt.subplots(14, 11)
count = 0
for x in range(14):
    for y in range(11):
        draw_image = Eigen_faces[count]
        axes[x][y].imshow(draw_image, cmap=plt.cm.gray)
        axes[x][y].axis('off')
        count = count + 1
fig.canvas.set_window_title('Eigen Faces')
plt.show()

k=10
print("K=10-------------------------->")

weights=np.dot(substract_mean_from_original, V.T)
reconstruction = mean + np.dot(weights[:,0:k], V[0:k,:])

fig, axes = plt.subplots(14, 11)
count = 0
for x in range(14):
    for y in range(11):
        draw_image = np.reshape(reconstruction[count,:],(256,256))
        axes[x][y].imshow(draw_image, cmap=plt.cm.gray)
        axes[x][y].axis('off')
        count = count + 1
fig.canvas.set_window_title('Reconstructed faces for k='+str(k))
plt.show()

k=30
print("K=30------------------------------>")

weights=np.dot(substract_mean_from_original, V.T)
reconstruction = mean + np.dot(weights[:,0:k], V[0:k,:])

fig, axes = plt.subplots(14, 11)
count = 0
for x in range(14):
    for y in range(11):
        draw_image = np.reshape(reconstruction[count,:],(256,256))
        axes[x][y].imshow(draw_image, cmap=plt.cm.gray)
        axes[x][y].axis('off')
        count = count + 1
fig.canvas.set_window_title('Reconstructed faces for k='+str(k))
plt.show()

k=50
print("K=50---------------------------------------->")

weights=np.dot(substract_mean_from_original, V.T)
reconstruction = mean + np.dot(weights[:,0:k], V[0:k,:])

fig, axes = plt.subplots(14, 11)
count = 0
for x in range(14):
    for y in range(11):
        draw_image = np.reshape(reconstruction[count,:],(256,256))
        axes[x][y].imshow(draw_image, cmap=plt.cm.gray)
        axes[x][y].axis('off')
        count = count + 1
fig.canvas.set_window_title('Reconstructed faces for k='+str(k))
plt.show()

k=100
print("K=100--------------------------------->")

weights=np.dot(substract_mean_from_original, V.T)
reconstruction = mean + np.dot(weights[:,0:k], V[0:k,:])

fig, axes = plt.subplots(14, 11)
count = 0
for x in range(14):
    for y in range(11):
        draw_image = np.reshape(reconstruction[count,:],(256,256))
        axes[x][y].imshow(draw_image, cmap=plt.cm.gray)
        axes[x][y].axis('off')
        count = count + 1
fig.canvas.set_window_title('Reconstructed faces for k='+str(k))
plt.show()

test_images=[]
for images in test:
    test_faces = Image.open(images)
    test_faces = np.asarray(test_faces, dtype=float) / 255.0
    test=(256,256,3)
    if test_faces.shape == test:
        test_faces=test_faces[:,:,0]
        test_images.append(test_faces)
    else:
        test_images.append(test_faces)

flat_test_Array = []
for x in range(len(test_images)):
    flat_Array = test_images[x].flatten()
    flat_test_Array.append(flat_Array)
flat_test_Array = np.asarray(flat_test_Array)
    
test_images=np.asarray(test_images)
test_from_mean=np.subtract(flat_test_Array, mean)

k=10
print("K=10----------------------------------->")

eig_wt = np.dot(V[:k, :],substract_mean_from_original.T)
threshold = 6000
for i in range(test_from_mean.shape[0]):
    test_weight = np.dot(V[:k, :],test_from_mean[i:i + 1,:].T)
    dist_ary = np.sum((eig_wt - test_weight) ** 2, axis=0)
    image_closest = np.argmin(np.sqrt(dist_ary))
    fig, axes = plt.subplots(1, 2)
    to_plot=np.reshape(flat_test_Array[i,:], (256,256))
    axes[0].imshow(to_plot, cmap=plt.cm.gray)
    axes[0].axis('off')
    if (dist_ary[image_closest] <= threshold):
        axes[1].imshow(alldata[image_closest,:,:], cmap=plt.cm.gray)
    axes[1].axis('off')
plt.show()

k=30
print("K=30------------------------------>")

eig_wt = np.dot(V[:k, :],substract_mean_from_original.T)
threshold = 6000
for i in range(test_from_mean.shape[0]):
    test_weight = np.dot(V[:k, :],test_from_mean[i:i + 1,:].T)
    dist_ary = np.sum((eig_wt - test_weight) ** 2, axis=0)
    image_closest = np.argmin(np.sqrt(dist_ary))
    fig, axes = plt.subplots(1, 2)
    to_plot=np.reshape(flat_test_Array[i,:], (256,256))
    axes[0].imshow(to_plot, cmap=plt.cm.gray)
    axes[0].axis('off')
    if (dist_ary[image_closest] <= threshold):
        axes[1].imshow(alldata[image_closest,:,:], cmap=plt.cm.gray)
    axes[1].axis('off')
plt.show()

k=50
print("K=50------------------------------------>")

eig_wt = np.dot(V[:k, :],substract_mean_from_original.T)
threshold = 6000
for i in range(test_from_mean.shape[0]):
    test_weight = np.dot(V[:k, :],test_from_mean[i:i + 1,:].T)
    dist_ary = np.sum((eig_wt - test_weight) ** 2, axis=0)
    image_closest = np.argmin(np.sqrt(dist_ary))
    fig, axes = plt.subplots(1, 2)
    to_plot=np.reshape(flat_test_Array[i,:], (256,256))
    axes[0].imshow(to_plot, cmap=plt.cm.gray)
    axes[0].axis('off')
    if (dist_ary[image_closest] <= threshold):
        axes[1].imshow(alldata[image_closest,:,:], cmap=plt.cm.gray)
    axes[1].axis('off')
plt.show()

k=100
print("K=100---------------------------------->")

eig_wt = np.dot(V[:k, :],substract_mean_from_original.T)
threshold = 6000
for i in range(test_from_mean.shape[0]):
    test_weight = np.dot(V[:k, :],test_from_mean[i:i + 1,:].T)
    dist_ary = np.sum((eig_wt - test_weight) ** 2, axis=0)
    image_closest = np.argmin(np.sqrt(dist_ary))
    fig, axes = plt.subplots(1, 2)
    to_plot=np.reshape(flat_test_Array[i,:], (256,256))
    axes[0].imshow(to_plot, cmap=plt.cm.gray)
    axes[0].axis('off')
    if (dist_ary[image_closest] <= threshold):
        axes[1].imshow(alldata[image_closest,:,:], cmap=plt.cm.gray)
    axes[1].axis('off')
plt.show()