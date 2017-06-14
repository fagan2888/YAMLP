"""
This code implements a Tensorflow single hidden layer neural network in a way that is compatible with the grid search method of
Scikit learn.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score



class MLPRegFlow(BaseEstimator, ClassifierMixin):

    def __init__(self, hidden_layer_sizes=(100,), alpha=0.0001, batch_size="auto",
                 learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True,
                 random_state=None, tol=1e-4, verbose=False, momentum=0.9, nesterovs_momentum=True,
                 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8):

        # Initialising the parameters
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # Initialising parameters needed for the Tensorflow part
        self.w1 = 0
        self.b1 = 0
        self.w2 = 0
        self.b2 = 0
        self.alreadyInitialised = False
        self.trainCost = []

    def fit(self, X, y):
        """
        Fit the model to data matrix X and target y.
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features) - The input data.
        :param y: array-like, shape (n_samples,) - The target values
        :return: None
        """

        print "Starting the fitting process ... \n"

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Modification of the y data, because tensorflow wants a column vector, while scikit learn uses a row vector
        y = np.reshape(y, (len(y),1))

        # Check that the architecture has only 1 hidden layer
        if len(self.hidden_layer_sizes) != 1:
            print "Error: This model currently only works with one hidden layer. "
            return None

        self.n_feat = X.shape[1]
        self.n_samples = X.shape[0]

        # Check the value of the batch size
        self.batch_size = self.checkBatchSize()

        # Initial set up of the NN
        X_train = tf.placeholder(tf.float32, [None, self.n_feat])
        Y_train = tf.placeholder(tf.float32, [None, 1])

        # This part either randomly initialises the weights and biases or restarts training from wherever it was stopped
        if self.alreadyInitialised == False:
            eps = 0.01
            weights1 = tf.Variable(tf.random_normal([self.hidden_layer_sizes[0], self.n_feat]) * 2 * eps - eps)
            bias1 = tf.Variable(tf.zeros([self.hidden_layer_sizes[0]]))
            weights2 = tf.Variable(tf.random_normal([1, self.hidden_layer_sizes[0]]) * 2 * eps - eps)
            bias2 = tf.Variable(tf.zeros([1]))
            parameters = [weights1, bias1, weights2, bias2]
            self.alreadyInitialised = True
        else:
            parameters = [tf.Variable(self.w1), tf.Variable(self.b1), tf.Variable(self.w2), tf.Variable(self.b2)]

        model = self.modelNN(X_train, parameters)
        cost = self.costReg(model, Y_train, [parameters[0], parameters[2]], self.alpha)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_init).minimize(cost)

        # Initialisation of the model
        init = tf.global_variables_initializer()

        # Running the graph
        with tf.Session() as sess:
            sess.run(init)

            for iter in range(self.max_iter):
                # This is the total number of batches in which the training set is divided
                n_batches = int(self.n_samples / self.batch_size)
                # This will be used to calculate the average cost per iteration
                avg_cost = 0
                # Learning over the batches of data
                for i in range(n_batches):
                    batch_x = X[i * self.batch_size:(i + 1) * self.batch_size, :]
                    batch_y = y[i * self.batch_size:(i + 1) * self.batch_size, :]
                    opt, c = sess.run([optimizer, cost], feed_dict={X_train: batch_x, Y_train: batch_y})
                    avg_cost += c / n_batches
                self.trainCost.append(avg_cost)
                if iter % 100 == 0:
                    print "Completed " + str(iter) + " iterations. \n"

            self.w1 = sess.run(parameters[0])
            self.b1 = sess.run(parameters[1])
            self.w2 = sess.run(parameters[2])
            self.b2 = sess.run(parameters[3])

    def modelNN(self, X, parameters):
        """
        This function calculates the model neural network
        :param X: This is the data to be used to generate the model
        :param parameters: these are the weights and the biases, arranged as a list of tf.Variables
        :return: it returns a tensor with the model
        """

        # Definition of the model
        a1 = tf.matmul(X, tf.transpose(parameters[0])) + parameters[1]  # output of layer1, size = n_sample x n_hidden_layer
        a1 = tf.nn.tanh(a1)
        model = tf.matmul(a1, tf.transpose(parameters[2])) + parameters[3]  # output of last layer, size = n_samples x 1

        return model

    def costReg(self, model, Y_data, parameters, regu):
        """
        This function calculates the cost function.
        :param model: This is the tensor with the structure of the neural network
        :param Y_data: The Y part of the training data (it is a tensorflow place holder)
        :param parameters: a list of TF global_variables with the weights.
        :param regu: the regularisation parameter
        :return: it returns the cost function (TF global_variable).
        """
        cost = tf.reduce_mean(tf.nn.l2_loss((model - Y_data)))  # using the quadratic cost function
        regulariser = tf.nn.l2_loss(parameters[0]) + tf.nn.l2_loss(parameters[1])
        cost = tf.reduce_mean(cost + regu * regulariser)

        return cost

    def plotTrainCost(self):
        fig2, ax2 = plt.subplots(figsize=(6,6))
        ax2.plot(self.trainCost)
        ax2.set_xlabel('Number of iterations')
        ax2.set_ylabel('Cost Value in train set')
        ax2.legend()
        plt.show()

    def checkBatchSize(self):
        """
        This function is called to check if the batch size has to take the default value or a user-set value.
        If it is a user set value, it checks whether it is a reasonable value.
        :return: batch_size
        """
        if self.batch_size == 'auto':
            batch_size = min(100, self.n_samples)
        else:
            if self.batch_size < 1 or self.batch_size > self.n_samples:
                print "Warning: Got `batch_size` less than 1 or larger than sample size. It is going to be clipped"
                batch_size = np.clip(self.batch_size, 1, self.n_samples)
            else:
                batch_size = self.batch_size

        return batch_size

    def checkIsFitted(self):
        """
        This function checks whether the weights and biases have been changed from their initial values.
        :return: A boolean. False if the weights and biases are zero. True otherwise.
        """
        if self.alreadyInitialised == False:
            print "Error: The fit function has not been called yet"
            return False
        else:
            return True

    def predict(self, X):
        """
        This function uses the X data and plugs it into the model and then returns the predicted y
        :param X: numpy array of size (n_samples, n_features)
        :return: numpy array of size (n_samples,)
        """
        print "Starting the predictions. \n"

        if self.checkIsFitted():
            check_array(X)

            X_test = tf.placeholder(tf.float32, [None, self.n_feat])

            parameters = [tf.Variable(self.w1), tf.Variable(self.b1), tf.Variable(self.w2), tf.Variable(self.b2)]
            model = self.modelNN(X_test, parameters)

            init = tf.global_variables_initializer()

            with tf.Session() as sess:
                sess.run(init)
                predictions = sess.run(model, feed_dict={X_test: X})
                predictions = np.reshape(predictions,(predictions.shape[0],))

            return predictions
        else:
            return

    def score(self, X, y, sample_weight=None):
        """
        Returns the mean accuracy on the given test data and labels. It calculates the R^2 value.
        :param X: The x values - numpy array of shape (N_samples, n_features)
        :param y: The true values for X - numpy array of shape (N_samples,)
        :param sample_weight: sample_weight : array-like, shape = [n_samples], optional
            Sample weights (not sure what this is, but i need it for inheritance from the BaseEstimator)
        :return: r2 - between 0 and 1. Tells how good the correlation plot is.
        """

        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        return r2

    def scoreFull(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels. It calculates the R^2 value.
        :param X: The x values - numpy array of shape (N_samples, n_features)
        :param y: The true values for X - numpy array of shape (N_samples,)
        :return: r2 - between 0 and 1. Tells how good the correlation plot is. rmsekJmol - the root mean square error in
        kJ/mol. maekJmol - the mean absolute error in kJ/mol.
        """

        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        rmseHa = np.sqrt(mean_squared_error(y, y_pred))
        maeHa = mean_absolute_error(y, y_pred)
        rmsekJmol = rmseHa * 2625.50
        maekJmol = maeHa * 2625.50
        return r2, rmsekJmol, maekJmol


# This example tests the module on fitting a simple quadratic function and then plots the results

if __name__ == "__main__":

    estimator = MLPRegFlow(hidden_layer_sizes=(5,), learning_rate_init=0.01, max_iter=5000, alpha=0)

    x = np.arange(-2.0, 2.0, 0.1)
    X = np.reshape(x, (len(x), 1))
    y = np.reshape(X ** 3, (len(x),))

    estimator.fit(X, y)
    estimator.plotTrainCost()
    y_pred = estimator.predict(X)

    #  Visualisation of predictions
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.scatter(x, y, label="original", marker="o", c="r")
    ax2.scatter(x, y_pred, label="predictions", marker="o", c='b')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()

    # Correlation plot
    fig3, ax3 = plt.subplots(figsize=(6,6))
    ax3.scatter(y, y_pred, label="CAS-SCF", marker="o", c="r")
    ax3.set_xlabel('original y')
    ax3.set_ylabel('prediction y')

    plt.show()