import numpy as np
import tensorflow as tf

class NeuralNetwork():
    """
    Class implementing an artificial neural network with one hidden layer (with a variable number of units).

    """

    def __init__(self, n_hidden_layer, learning_rate, iterations, eps):
        """
        This is the constructor for the neural network class.
        :param n_hidden_layer: the number of units in the single hidden layer
        :param learning_rate: the learning rate for the Adams optimiser
        :param iterations: number of iterations of the minimisation algorithm
        :param eps: small value that decides the range of random initialisation of the variables
        """
        self.n_hidden_layer = n_hidden_layer
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.eps = eps
        self.initialisation = True

        self.X_trainSet = np.array(1)
        self.Y_trainSet = np.array(1)
        self.X_crossVal = np.array(1)
        self.Y_crossVal = np.array(1)
        self.X_val = np.array(1)
        self.Y_val = np.array(1)

        # These will contain the values of the weights and biases once the function "fit" has run
        self.w1 = 0
        self.b1 = 0
        self.w2 = 0
        self.b2 = 0

        # This will contain the cost as the training set is fitted
        self.trainCost = []
        self.CrossValCost = []

    def modelNN(self, X_train, parameters):
        """
        This function calculates the model neural network
        :param X_train: This is the data to be used to generate the model
        :param parameters: these are the weights and the biases, arranged as a list of tf.Variables
        :return: it returns a tensor with the model
        """

        # Definition of the model
        a1 = tf.matmul(X_train, tf.transpose(parameters[0])) + parameters[1]  # output of layer1, size = n_sample x n_hidden_layer
        a1 = tf.nn.sigmoid(a1)
        model = tf.matmul(a1, tf.transpose(parameters[2])) + parameters[3]  # output of last layer, size = n_samples x 1

        return model

    def costReg(self, model, Y_data, parameters, regBeta):
        """
        This function calculates the cost function.
        :param model: This is the tensor with the structure of the neural network
        :param Y_data: The Y part of the training data (it is a tensorflow place holder)
        :param parameters: a list of TF variables with the weights.
        :param regBeta: the regularisation parameter
        :return: it returns the cost function (TF variable).
        """
        cost = tf.reduce_mean(tf.nn.l2_loss((model - Y_data)))  # using the quadratic cost function
        regulariser = tf.nn.l2_loss(parameters[0]) + tf.nn.l2_loss(parameters[1])
        cost = tf.reduce_mean(cost + regBeta * regulariser)

        return cost

    def fit(self, X, Y, beta, batch_size, plot=False):
        """
        This function calculates the weights and biases that better fit the data
        :param X: This is a numpy 2D array of size (n_samples, n_features)
        :param Y: This is a numpy 2D array of size (n_samples, 1)
        :beta: regularisation parameter
        :batch_size: size of the batch for mini-batch gradient descent
        :plot: a boolean that tells whether to plot the cost or not
        :return: None
        """
        print "Starting the fitting. \n"

        self.n_feat = X.shape[1]
        self.n_samples = X.shape[0]

        # Initial set up of the NN
        X_train = tf.placeholder(tf.float32, [None, self.n_feat])
        Y_train = tf.placeholder(tf.float32, [None, 1])

        randIn = tf.constant(self.eps, dtype=tf.float32)  # This sets the range of weights/biases initial values

        if self.initialisation:
            # weights1 = tf.Variable(tf.random_normal([self.n_hidden_layer, self.n_feat]) * 2 * randIn - randIn)
            weights1 = tf.Variable(tf.random_normal([self.n_hidden_layer, self.n_feat], mean=0.0, stddev=(1/np.sqrt(self.n_feat)) ))
            bias1 = tf.Variable(tf.zeros([self.n_hidden_layer]))
            weights2 = tf.Variable(tf.random_normal([1, self.n_hidden_layer], mean=0.0, stddev=(1/np.sqrt(self.n_hidden_layer)) ))
            bias2 = tf.Variable(tf.zeros([1]))
            parameters = [weights1, bias1, weights2, bias2]
            self.initialisation = False
        else:
            parameters = [tf.Variable(self.w1), tf.Variable(self.b1), tf.Variable(self.w2), tf.Variable(self.b2)]

        model = self.modelNN(X_train, parameters)
        cost = self.costReg(model, Y_train, [parameters[0], parameters[2]], beta)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Initialisation of the model
        init = tf.initialize_all_variables()


        # Running the graph
        with tf.Session() as sess:
            sess.run(init)

            for iter in range(self.iterations):
                # This is the total number of batches in which the training set is divided
                n_batches = int(self.n_samples/batch_size)
                # This will be used to calculate the average cost per iteration
                avg_cost = 0
                for i in range(n_batches):
                    batch_x = X[i*batch_size:(i+1)*batch_size, :]
                    batch_y = Y[i * batch_size:(i + 1) * batch_size, :]
                    opt, c = sess.run([optimizer, cost], feed_dict={X_train: batch_x, Y_train: batch_y})
                    avg_cost += c / n_batches
                self.trainCost.append(avg_cost)

            self.w1 = sess.run(parameters[0])
            self.b1 = sess.run(parameters[1])
            self.w2 = sess.run(parameters[2])
            self.b2 = sess.run(parameters[3])

            print "The value of the cost with beta=" + str(beta) + " is "
            print str(self.trainCost[-1])

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(self.trainCost)
            plt.show()

        return None

    def predict(self, X):
        """
        This function uses the X data and plugs it into the model and then returns the predicted Y
        :param X: numpy array of size (n_samples, n_features)
        :return: numpy 2D array of size (n_samples, 1)
        """
        X_test = tf.placeholder(tf.float32, [None, self.n_feat])

        parameters = [tf.Variable(self.w1), tf.Variable(self.b1), tf.Variable(self.w2), tf.Variable(self.b2)]
        model = self.modelNN(X_test, parameters)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            results = sess.run(model, feed_dict={X_test: X})

        return results

    def costTest(self, X, Y):
        """
        This function calculates the cost on a dataset X and returns it. It also appends it to an array so that if done
        during the fitting it can be used to plot learning curves.
        :param X: numpy array of size (n_samples, n_features)
        :param Y: numpy array of size (n_samples, 1)
        :return: the cost
        """
        n_feat = X.shape[1]

        X_test = tf.placeholder(tf.float32, [None, n_feat])
        Y_test = tf.placeholder(tf.float32, [None, 1])

        parameters = [tf.Variable(self.w1), tf.Variable(self.b1), tf.Variable(self.w2), tf.Variable(self.b2)]
        model = self.modelNN(X_test, parameters)
        cost = self.costReg(model, Y_test, [parameters[0], parameters[2]], 0)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            testCost = sess.run(cost, feed_dict={X_test: X, Y_test: Y})

        self.CrossValCost.append(testCost)

        return testCost



