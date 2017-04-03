"""
This script shows how to use the current framework for importing data, training it with a very simple neural network and
plot the results.
"""


import importData
import CoulombMatrix
import NN_02
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


### --------------- ** Importing the data ** -----------------

importData.XYZtoCSV("/Users/walfits/Repositories/trainingdata/per-user-trajectories/CH4+CN/combinedTraj.xyz")
importData.CCdistance() # This calculates the CC distance in each configuration

X_total = importData.loadX("X.csv")
Y_total = importData.loadY("Y.csv")
Z_total1, Z_total2 = importData.loadZ("Z.csv")     # This contains the CC distances and C-H distance

# This appends the CC distance on the side of the energy, so that when the data is shuffled, this info is not lost
Y_total = np.concatenate((Y_total, Z_total1, Z_total2), axis=1)


### --------------- ** Turning cartesian coordinates into coulomb matrix ** -----------------

descriptor = CoulombMatrix.CoulombMatrix(matrixX=X_total)
descriptor.generate()
descriptor.trim()
descriptor.normalise_2()

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

method = NN_02.NeuralNetwork(n_hidden_layer=40, learning_rate=0.01, iterations=100, eps=0.01)
method.fit(X_trainSet, reshapeY, beta=0.02, batch_size=100, plot=False)
predictions = method.predict(X_crossVal)

### --------------- ** Plotting the data ** -----------------
x1 = Y_crossVal[:, 1]
x2 = Y_crossVal[:, 2]
y_data = Y_crossVal[:, 0]
y_pred = predictions

fig1 = plt.figure(figsize=(7,7))
fig2 = plt.figure(figsize=(7,7))
ax1 = fig1.add_subplot(111, projection='3d')
ax2 = fig2.add_subplot(111, projection='3d')

ax1.scatter(x1, x2, y_data, label = "actual data", marker="o", c="r")
ax1.set_xlabel('CC distance')
ax1.set_ylabel('NH distance')
ax1.set_zlabel('Energy')

ax2.scatter(x1, x2, y_pred, label = "actual data", marker="o", c='b')
ax2.set_xlabel('CC distance')
ax2.set_ylabel('NH distance')
ax2.set_zlabel('Energy')

ax1.azim = 100
ax1.elev = 20
ax2.azim = 100
ax2.elev = 20

plt.show()
