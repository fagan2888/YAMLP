"""
This script shows how to use the current framework for importing data, training it with a very simple neural network and
plot the results.
"""


import importData
import CoulombMatrix
import NN_02
import numpy as np
import matplotlib.pyplot as plt

### --------------- ** Importing the data ** -----------------

importData.XYZtoCSV("/Users/walfits/Repositories/trainingdata/per-user-trajectories/CH4+CN/combinedTraj.xyz")
importData.CCdistance() # This calculates the CC distance in each configuration

X_total = importData.loadX("X.csv")
Y_total = importData.loadY("Y.csv")
Z_total = importData.loadY("Z.csv")     # This contains the CC distances

# This appends the CC distance on the side of the energy, so that when the data is shuffled, this info is not lost
Y_total = np.concatenate((Y_total, Z_total), axis=1)


### --------------- ** Turning cartesian coordinates into coulomb matrix ** -----------------

descriptor = CoulombMatrix.CoulombMatrix(matrixX=X_total)
descriptor.generate()

### --------------- ** Splitting the data into training, cross-validation and validation set ** -----------------

dataProportions = np.array([0.9, 0.1, 0])
splitX, splitY = importData.splitData(descriptor.coulMatrix, Y_total, dataProportions)

X_trainSet = splitX[0]
Y_trainSet = splitY[0]
# This way the shape of the training set fits the requirement of the fit function
reshapeY = np.reshape(Y_trainSet[:,0], (Y_trainSet[:,0].shape[0], 1))

X_crossVal = splitX[1]
Y_crossVal = splitY[1]

X_val = splitX[2]
Y_val = splitY[2]

### --------------- ** Training the neural network ** -----------------

method = NN_02.NeuralNetwork(n_hidden_layer=20, learning_rate=0.01, iterations=50, eps=0.01)
method.fit(X_trainSet, reshapeY, beta=0.02, batch_size=100, plot=True)
predictions = method.predict(X_crossVal)

x = Y_crossVal[:, 1]
y_data = Y_crossVal[:, 0]
y_pred = predictions

plt.scatter(x, y_data, label = "actual data", marker=(5, 2), c="red")
plt.scatter(x, y_pred, label = "predictions", marker=(5, 1), c="blue")
plt.legend(loc="upper right")
plt.show()
