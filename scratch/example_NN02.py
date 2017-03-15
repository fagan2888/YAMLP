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
importData.XYZtoCSV("/Users/walfits/Repositories/trainingdata/per-user-trajectories/CH4+CN/combinedTraj.xyz")

X_total = importData.loadX("X.csv")
Y_total = importData.loadY("Y.csv")


### --------------- ** Turning cartesian coordinates into coulomb matrix ** -----------------

descriptor = CoulombMatrix.CoulombMatrix(matrixX=X_total)
descriptor.generate()

### --------------- ** Splitting the data into training, cross-validation and validation set ** -----------------

dataProportions = np.array([0.9, 0.1, 0])
splitX, splitY = importData.splitData(descriptor.coulMatrix, Y_total, dataProportions)

X_trainSet = splitX[0]
Y_trainSet = splitY[0]

X_crossVal = splitX[1]
Y_crossVal = splitY[1]

X_val = splitX[2]
Y_val = splitY[2]

### --------------- ** Training the neural network ** -----------------

method = NN_02.NeuralNetwork(n_hidden_layer=20, learning_rate=0.05, iterations=80, eps=0.01)

betaGrid = np.arange(0.005,0.027, 0.002)
trainingCost = []
testCost = []

currentTrainCost = method.fit(X_trainSet, Y_trainSet, beta=0.02, batch_size=10, plot=True)
trainingCost.append(currentTrainCost)
currentTestCost = method.costTest(X_crossVal, Y_crossVal)
testCost.append(currentTestCost)

print trainingCost
print testCost

# plt.plot(betaGrid, trainingCost, label = "training cost", c="red")
# plt.plot(betaGrid, testCost, label = "test cost", c="blue")
# plt.legend(loc="upper right")
# plt.show()




