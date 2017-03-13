import numpy as np
import tensorflow as tf

class NeuralNetwork():
    """
    Class implementing an artificial neural network.

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

    def modelNN(self, X_train, parameters):
        """
        This function calculates the model neural network
        :param X_train: This is the data to be used to generate the model
        :param parameters: these are the weights and the biases, arranged as a list of tf.Variables
        :return: it returns a tensor with the model
        """

        # Definition of the model
        a1 = tf.matmul(X_train, tf.transpose(parameters[0])) + parameters[1]  # output of layer1, size = n_sample x n_hidden_layer
        model = tf.matmul(a1, tf.transpose(parameters[2])) + parameters[3]  # output of last layer, size = n_samples x 1

        return model


    def fit(self, X, Y, plot=False):
        """
        This function calculates the weights and biases that better fit the data
        :param X: This is a numpy 2D array of size (n_samples, n_features)
        :param Y: This is a numby 2D array of size (n_samples, 1)
        :plot: a boolean that tells whether to plot or not
        :return: None
        """

        self.n_feat = X.shape[1]
        self.n_samples = X.shape[0]
        batch_size = 50

        # Initial set up of the NN
        X_train = tf.placeholder(tf.float32, [None, self.n_feat])
        Y_train = tf.placeholder(tf.float32, [None, 1])

        randIn = tf.constant(self.eps, dtype=tf.float32)  # This sets the range of weights/biases initial values

        weights1 = tf.Variable(tf.random_normal([self.n_hidden_layer, self.n_feat]) * 2 * randIn - randIn)
        bias1 = tf.Variable(tf.random_normal([self.n_hidden_layer]) * 2 * randIn - randIn)
        weights2 = tf.Variable(tf.random_normal([1, self.n_hidden_layer]) * 2 * randIn - randIn)
        bias2 = tf.Variable(tf.random_normal([1]) * 2 * randIn - randIn)

        model = self.modelNN(X_train, [weights1, bias1, weights2, bias2])

        cost = tf.reduce_mean(tf.nn.l2_loss((model - Y_train)))  # using the quadratic cost function
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Initialisation of the model
        init = tf.initialize_all_variables()
        cost_array = []

        # Running the graph
        with tf.Session() as sess:
            sess.run(init)
            # for iter in range(self.iterations):
            #     opt, c = sess.run([optimizer, cost], feed_dict={X_train: X, Y_train: Y})
            #     cost_array.append(c)

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
                cost_array.append(avg_cost)

                if iter % 50 == 0:
                    print "iteration: " + str(iter)

            self.w1 = sess.run(weights1)
            self.b1 = sess.run(bias1)
            self.w2 = sess.run(weights2)
            self.b2 = sess.run(bias2)

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(cost_array)
            plt.show()
            print "The value of the cost on the last iteration is: "
            print str(cost_array[-1])




    def predict(self, X):

        X_train = tf.placeholder(tf.float32, [None, self.n_feat])

        parameters = [tf.Variable(self.w1), tf.Variable(self.b1), tf.Variable(self.w2), tf.Variable(self.b2)]
        model = self.modelNN(X_train, parameters)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            results = sess.run(model, feed_dict={X_train: X})

        return results
