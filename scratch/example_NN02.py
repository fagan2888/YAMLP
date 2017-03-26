"""
This script shows an example of how to use a neural network and optimise the regularisation parameter or the amount of
training data used. It also plots learning curves/"high-bias-high-variance" curves.
"""

import importData
import CoulombMatrix
import NN_02
import numpy as np
import matplotlib.pyplot as plt

### --------------- ** Importing the data ** -----------------
importData.XMLtoCSV("/Users/walfits/Repositories/tensorflow/AMP/input1.xml")

X_total = importData.loadX("X.csv")
Y_total = importData.loadY("Y.csv")

angles = np.arange(175, 97.5, -2.5)
angles = angles.reshape(Y_total.shape)
Y_total = np.concatenate((Y_total, angles), axis=1)


### --------------- ** Interpolating ** -----------------------

# X_total, Y_total = importData.interpolData(X_total, Y_total, 2)

### --------------- ** Turning cartesian coordinates into coulomb matrix ** -----------------

descriptor = CoulombMatrix.CoulombMatrix(matrixX=X_total)
descriptor.generate()
descriptor.trim()
descriptor.normalise_2()

### --------------- ** Splitting the data into training, cross-validation and validation set ** -----------------

dataProportions = np.array([1, 0, 0])
splitX, splitY = importData.splitData(descriptor.coulMatrix, Y_total, dataProportions)

X_trainSet = splitX[0]
Y_trainSet = splitY[0]
reshapeY = np.reshape(Y_trainSet[:,0], (Y_trainSet[:,0].shape[0], 1))

X_crossVal = splitX[1]
Y_crossVal = splitY[1]
reshapeYcross = np.reshape(Y_crossVal[:,0], (Y_crossVal[:,0].shape[0], 1))

X_val = splitX[2]
Y_val = splitY[2]

### --------------- ** Training the neural network ** -----------------

#########################################################
#   Good parameters for the small training set:         #
#   n_hidden_layer=50                                   #
#   learning_rate=0.0005                                #
#   iterations=50000                                    #
#   eps=0.001                                           #
#   beta=0                                              #
#   batch_size=10                                       #
#########################################################

method = NN_02.NeuralNetwork(n_hidden_layer=10, learning_rate=0.0005, iterations=30000, eps=0.001)

method.fit(X_trainSet, reshapeY, beta=0, batch_size=2, plot=True)

predictions = method.predict(X_trainSet)

fig1 = plt.figure(figsize=(7,7))
ax1 = fig1.add_subplot(111)

x1 = Y_trainSet[:,1]
y_data = Y_trainSet[:,0]
y_pred = predictions[:,0]

ax1.scatter(x1, y_data, label = "actual data", marker=".", c="r")
ax1.scatter(x1, y_pred, label = "predictions", marker=".", c='b')

ax1.set_xlabel('CHC angle')
ax1.set_ylabel('Energy')


plt.show()




