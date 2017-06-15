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

# X_test, Y_test = importData.interpolData(X_total, Y_total, 1)

### --------------- ** Turning cartesian coordinates into coulomb matrix ** -----------------

descriptor = CoulombMatrix.CoulombMatrix(matrixX=X_total)
descriptor.generate()
# descriptor.trim()
descriptor.normalise_3()

# Descriptor for the test set of interpolated data

# test_X = CoulombMatrix.CoulombMatrix(matrixX=X_test)
# test_X.generate()
# test_X.trim()
# test_X.normalise_3()
# reshapeYtest = np.reshape(Y_test[:,0], (Y_test[:,0].shape[0], 1))

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

points = 7
iterations=5000

neuralNet = NN_02.NeuralNetwork(n_hidden_layer=50, learning_rate=5, iterations=iterations, eps=0.001)


for _ in range(points):
    neuralNet.fit(X_trainSet, reshapeY, beta=0, batch_size=10, plot=False)


### --------------- ** Plotting learning curves ** -----------------

# learnCurv = plt.figure(figsize=(7,7))
# ax = learnCurv.add_subplot(111)
#
# x_training = range(0,points*iterations)
# y_training = neuralNet.trainCost
# x_cross = range(0,points*iterations,iterations)
# y_cross = neuralNet.trainCost
#
# ax.scatter(x_training, y_training, label = "training set cost", marker=".", c="r")
# ax.scatter(x_cross, y_cross, label = "cross set cost", marker=".", c='b')
# ax.set_xlabel('iteration')
# ax.set_ylabel('cost')
# ax.legend()
#
# plt.show()


### --------------- ** Plotting the data set and predictions ** -----------------

# f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
#
# predictions1 = neuralNet.predict(test_X.coulMatrix)
# testCost = neuralNet.costTest(test_X.coulMatrix, reshapeYtest)
# print "The cost for the test set is " + str(testCost) + "\n"

predictions1 = neuralNet.predict(X_trainSet)
testCost = neuralNet.costTest(X_trainSet, reshapeY)
print "The cost for the test set is " + str(testCost) + "\n"

x1 = Y_trainSet[:,1]
y_data1 = Y_trainSet[:,0]
y_pred1 =predictions1[:,0]
#
#
# predictions2 = neuralNet.predict(X_trainSet)
# x2 = Y_trainSet[:,1]
# y_data2 = Y_trainSet[:,0]
# y_pred2 = predictions2[:,0]
#
# ax1.scatter(x1, y_data1, label = "actual data", marker="o", c="r")
# ax1.scatter(x1, y_pred1, label = "predictions", marker="o", c='b')
#
# ax2.scatter(x2, y_data2, label = "actual data", marker="o", c="r")
# ax2.scatter(x2, y_pred2, label = "predictions", marker="o", c='b')
#
# ax1.set_xlabel('CHC angle')
# ax1.set_ylabel('Energy')
#
#
# plt.show()


#  Ad hoc

# fig1, ax = plt.subplots()
# ax.scatter(x2, y_data2, label = "CAS-SCF", marker="o", c="r")
# ax.scatter(x2, y_pred2, label = "NN", marker="o", c='b')
# ax.set_xlabel('CHC angle')
# ax.set_ylabel('Energy (kJ/mol)')
# ax.legend()

fig2, ax2 = plt.subplots(figsize=(8,7))
ax2.scatter(x1, y_data1, label = "CAS-SCF", marker="o", c="r")
ax2.scatter(x1, y_pred1, label = "NN", marker="o", c='b')
ax2.set_xlabel('CHC angle')
ax2.set_ylabel('Energy (kJ/mol)')
ax2.legend()

fig3, ax3 = plt.subplots(figsize=(8,7))
ax3.scatter(y_data1, y_pred1, label = "CAS-SCF", marker="o", c="r")
ax3.set_xlabel('CAS-SCF Energy (kJ/mol)')
ax3.set_ylabel('NN Energy (kJ/mol)')




plt.show()


