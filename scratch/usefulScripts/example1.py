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

# This data is the small CAS-SCF dataset. This function converts the XML data into CSV format.
importData.XMLtoCSV("CASSCFdata.xml")

# The coordinates and the energies are imported from the CSV files
X_total = importData.loadX("X.csv")
Y_total = importData.loadY("Y.csv")

# The CHC angle distance is calculated and added to the energy matrix. So Y_total has a shape (n_samples, 2)
angles = np.arange(175, 97.5, -2.5)
angles = angles.reshape(Y_total.shape)
Y_total = np.concatenate((Y_total, angles), axis=1)


### --------------- ** Turning cartesian coordinates into coulomb matrix ** -----------------


descriptor = CoulombMatrix.CoulombMatrix(matrixX=X_total)
descriptor.generate()
descriptor.normalise_3()


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

points = 7
iterations=5000

neuralNet = NN_02.NeuralNetwork(n_hidden_layer=50, learning_rate=5, iterations=iterations, eps=0.001)


for _ in range(points):
    neuralNet.fit(X_trainSet, reshapeY, beta=0, batch_size=10, plot=False)

### --------------- ** Plotting the data set and predictions ** -----------------


# Calculating the energies predicted from the neural network
predictions1 = neuralNet.predict(X_trainSet)

# Calculating the cost in these predictions
testCost = neuralNet.costTest(X_trainSet, reshapeY)
print "The cost for the test set is " + str(testCost) + "\n"

# Angles
x1 = Y_trainSet[:,1]

# Original and predicted values
y_data1 = Y_trainSet[:,0]
y_pred1 =predictions1[:,0]

#  Energy as a function of CHC angle
fig2, ax2 = plt.subplots(figsize=(8,7))
ax2.scatter(x1, y_data1, label = "CAS-SCF", marker="o", c="r")
ax2.scatter(x1, y_pred1, label = "NN", marker="o", c='b')
ax2.set_xlabel('CHC angle')
ax2.set_ylabel('Energy (kJ/mol)')
ax2.legend()

# Correlation plot
fig3, ax3 = plt.subplots(figsize=(8,7))
ax3.scatter(y_data1, y_pred1, label = "CAS-SCF", marker="o", c="r")
ax3.set_xlabel('CAS-SCF Energy (kJ/mol)')
ax3.set_ylabel('NN Energy (kJ/mol)')

plt.show()


