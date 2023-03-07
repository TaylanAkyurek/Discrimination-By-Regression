#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:10:37 2022

@author: alitaylanakyurek
"""

import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = np.genfromtxt("/Users/alitaylanakyurek/Downloads/hw02_data_points.csv", delimiter = ",")
y = np.genfromtxt("/Users/alitaylanakyurek/Downloads/hw02_class_labels.csv", delimiter = ",")

W = np.genfromtxt("/Users/alitaylanakyurek/Downloads/hw02_W_initial.csv", delimiter = ",")
w0 = np.genfromtxt("/Users/alitaylanakyurek/Downloads/hw02_w0_initial.csv", delimiter = ",")

x_train = x[:10000]
x_test = x[10000:]

y_train = y[:10000]
y_test = y[10000:]


x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


y_truth = y_train.astype(int)
y_test_truth =  y_test.astype(int)
K = 10
N = x_train.shape[0]
N_test = x_test.shape[0]

Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), y_truth - 1] = 1

Y_test_truth = np.zeros((N_test, K)).astype(int)
Y_test_truth[range(N_test), y_test_truth - 1] = 1


def sigmoid(X, W, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, W) + w0))))

#derivatives of sum squared errors with respect to W and w0

def gradient_W(X, Y_truth, Y_predicted):
    return (np.array([-np.matmul((Y_truth[:, c] - Y_predicted[:, c]) * Y_predicted[:,c] * (1- Y_predicted[:,c]), X) for c in range(K)]).T)

def gradient_w0(Y_truth, Y_predicted):
    return(-np.sum((Y_truth - Y_predicted)*Y_predicted*(1 - Y_predicted), axis = 0))



eta = 0.00001
iteration = 1
objective_values = []
for i in range(1000):
    Y_predicted = sigmoid(x_train, W, w0)

    objective_values = np.append(objective_values, 0.5*np.sum((Y_truth - Y_predicted)**2))

    W_old = W
    w0_old = w0

    W = W - eta * gradient_W(x_train, Y_truth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)

 
    iteration = iteration + 1
    
plt.figure(figsize = (8, 4))
plt.plot(range(1, iteration), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()



print("W:")
print(W)
print("\n")
print("w0:")
print(w0)
print("\n")

print("train confussion matrix:")
print("\n")

Y_predicted = np.argmax(Y_predicted, axis = 1) + 1

confusion_matrix = pd.crosstab(Y_predicted, y_truth, rownames = ['y_pred'], colnames = ['y_truth'])


print(confusion_matrix)
print("\n")


Y_test_predicted = sigmoid(x_test, W, w0)

y_test_predicted = np.argmax(Y_test_predicted, axis = 1) + 1

confusion_matrix = pd.crosstab(y_test_predicted, y_test_truth, rownames = ['y_test_pred'], colnames = ['y_test_truth'])
print("test confussion matrix:")
print("\n")

print(confusion_matrix)
