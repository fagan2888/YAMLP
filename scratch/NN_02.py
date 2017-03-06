import numpy as np

class NeuralNetwork():
    """
    Class implementing an artificial neural network.

    The constructor takes the following arguments:
        n_hidden_layer: number of units in the single hidden layer
        learning_rate: the learning rate for the Adams optimiser
        iterations: number of iterations of the minimisation algorithm
        eps: small value that decides the range of random initialisation of the variables

    """

    def __init__(self, n_hidden_layer, learning_rate, iterations, eps):
        self.n_hidden_layer = n_hidden_layer
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.eps = eps
        self.X_trainSet = np.array(1)
        self.Y_trainSet = np.array(1)
        self.X_crossVal = np.array(1)
        self.Y_crossVal = np.array(1)
        self.X_val = np.array(1)
        self.Y_val = np.array(1)

    def loadData(self, filePath):
        """
        :param filePath: This is the path to the XML file that contains the data set
        :return: X matrix of shape (n_samples, n_features) and a Y matrix of shape (n_samples, 1)
        """





    def fit(self, X, Y):


