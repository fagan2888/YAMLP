import importData
import CoulombMatrix
import NN_02
import numpy as np

### --------------- ** Importing the data ** -----------------

importData.XMLtoCSV("/Users/walfits/Repositories/tensorflow/AMP/input1.xml")
X_total = importData.loadX("X.csv")
Y_total = importData.loadY("Y.csv")

### --------------- ** Turning cartesian coordinates into coulomb matrix ** -----------------

descriptor = CoulombMatrix.CoulombMatrix(matrixX=X_total)
descriptor.generate()

### --------------- ** Splitting the data into training, cross-validation and validation set ** -----------------

dataProportions = np.array([1, 0, 0])
splitX, splitY = importData.splitData(descriptor.coulMatrix, Y_total, dataProportions)

X_trainSet = splitX[0]
Y_trainSet = splitY[0]

X_crossVal = splitX[1]
Y_crossVal = splitY[1]

X_val = splitX[2]
Y_val = splitY[2]

### --------------- ** Training the neural network ** -----------------

method = NN_02.NeuralNetwork(n_hidden_layer=20, learning_rate=0.01, iterations=200, eps=0.01)
method.fit(X_trainSet, Y_trainSet)

