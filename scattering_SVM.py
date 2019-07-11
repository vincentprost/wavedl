# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:49:31 2019

@author: admin-local
"""

import numpy as np
from kymatio import Scattering2D
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
import mnist
import spams


mnist.init()


"""
data = np.load("C:/Hung/Data/variable_stock/MNIST for clsuter/data_training.npy")
labels = np.load("C:/Hung/Data/variable_stock/MNIST for clsuter/labels_training.npy")
test = np.load("C:/Hung/Data/variable_stock/MNIST for clsuter/data_testing.npy")
labels_test = np.load("C:/Hung/Data/variable_stock/MNIST for clsuter/labels_testing.npy")
"""


data, labels, test, labels_test = mnist.load()


from sklearn.svm import SVC
clf = SVC(kernel = "linear")


n_train = 10000
n_test = 1000
X_train = np.array(data[:n_train],dtype = np.float32)/255.
y_train = labels[:n_train]
X_test = np.array(test[:n_test],dtype = np.float32)/255.
y_test = labels_test[:n_test]

clf.fit(X_train,y_train)


n_correct_train = np.array([y_train[i] == clf.predict([X_train[i,:]])
                                          for i in range(X_train.shape[0])]).nonzero()[0].size/float(y_train.shape[0])
        
n_correct_test = np.array([y_test[i] == clf.predict([X_test[i,:]])
                                          for i in range(X_test.shape[0])]).nonzero()[0].size/float(y_test.shape[0])



X_train2D = X_train.reshape(n_train,1,28,28)
X_test2D = X_test.reshape(n_test,1,28,28)

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(X_train2D[0,1,:,:],cmap = plt.cm.gray)


X_train2D_tensor = torch.tensor(X_train2D)
X_test2D_tensor = torch.tensor(X_test2D)

scattering = Scattering2D(J=2, shape=(28, 28), max_order=1)

X_train_scat = scattering(X_train2D_tensor)
X_test_scat = scattering(X_test2D_tensor)


patch_nb = X_test_scat.shape[2]
patch_size = X_test_scat.shape[3]






X_train_2D_scat = np.array(X_train_scat.reshape(n_train, patch_nb * patch_size**2))
X_test_2D_scat = np.array(X_test_scat.reshape(n_test, patch_nb * patch_size**2))



clf.fit(X_train_2D_scat ,y_train)



n_correct_train_scat_svm = np.array([y_train[i] == clf.predict([X_train_2D_scat[i,:]])
                                          for i in range(X_train_2D_scat.shape[0])]).nonzero()[0].size/float(y_train.shape[0])
        
n_correct_test_scat_svm = np.array([y_test[i] == clf.predict([X_test_2D_scat[i,:]])
                                          for i in range(X_test_2D_scat.shape[0])]).nonzero()[0].size/float(y_test.shape[0])


########## Dictionary learning


X_train_2D_scat = (X_train_2D_scat.T / np.sqrt(np.sum(X_train_2D_scat * X_train_2D_scat,1))).T
X_test_2D_scat = (X_test_2D_scat.T / np.sqrt(np.sum(X_test_2D_scat * X_test_2D_scat,1))).T


lamda = 0.01
param = { 'K' : 100, # learns a dictionary with 100 elements
          'lambda1' : lamda, 'numThreads' : 4, 'batchsize' : 400,
          'iter' : 1000}


D = spams.trainDL(X_train_2D_scat.T,**param)
alpha_train = spams.lasso(X_train_2D_scat.T, D = D, lambda1 = lamda).toarray().T
alpha_test = spams.lasso(X_test_2D_scat.T, D = D, lambda1 = lamda).toarray().T



clf = SVC(kernel = "linear")
clf.fit(alpha_train ,y_train)



n_correct_train_scat_dl = np.array([y_train[i] == clf.predict([alpha_train[i,:]])
                                          for i in range(X_train_2D_scat.shape[0])]).nonzero()[0].size/float(y_train.shape[0])
        
n_correct_test_scat_dl = np.array([y_test[i] == clf.predict([alpha_test[i,:]])
                                          for i in range(X_test_2D_scat.shape[0])]).nonzero()[0].size/float(y_test.shape[0])

print(n_correct_test_scat)
