import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1
plt.figure(figsize=(20,15))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='g', label='Iris Non Verginica')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='b', label='Iris Verginica')
plt.title("SCATTER PLOT 1")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')


class LogReg:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            print(f"ITERATION NUMBER: {i} IS STARTING   :------------------------------------>")
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
            print(f'loss: {loss} \t')
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()
                
model = LogReg(lr=0.1, num_iter=310000)
model.fit(X, y)
preds = model.predict(X)
(preds == y).mean()
model.theta


plt.figure(figsize=(20, 15))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='g', label='Iris Non Verginica')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='b', label='Iris Verginica')
plt.legend()
plt.title("SCATTER PLOT 2")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='red');
