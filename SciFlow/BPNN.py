"""
This module implements a Behler-Parinello Neural network that is compatible with Scikit learn and can therefore be
used with Osprey hyperparameter optimisation.

This code was written following closely the code written by Zachary Ulissi (Department of Chemical Engineering,
Stanford University) in the tflow.py module of the AMP package.
"""
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class BPNN(BaseEstimator, ClassifierMixin):
    """
    The parameter labels says to which atom each feature in X corresponds to. It is a list of atom label plus the number
    of how many features correspond to that atom.

    :labels: list of length (2*n_atoms,) containing strings and int
    """
    def __init__(self, hidden_layer_sizes=(5,), alpha=0.0001, batch_size='auto', learning_rate_init=0.001,
                 max_iter=80, labels=(0,)):

        # Initialising the parameters
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        # Terrible coding, need to find a better solution to this
        if labels == (0,):
            self.labels = ['N', 4, 'C', 4, 'C', 4, 'H', 4, 'H', 4, 'H', 4, 'H', 4]
        else:
            self.labels = labels

    def fit(self, X, y):
        """
        X is the training set. It is a numpy array containing all the descriptors for each atom concatenated.

        :X: numpy array of shape (n_samples, n_features)
        :y: array of shape (n_samples,)
        """
        self.n_samples = X.shape[0]

        # Modifying shape of y to be compatible with tensorflow
        y = np.reshape(y, (len(y), 1))

        # Creating a place holder for y
        y_tf = tf.placeholder(tf.float32, [None, 1])

        # Make a list of unique elements labels
        self.unique_ele = self.__unique_elements()

        # Split the data into the descriptors for the different atoms
        X_input = self.__split_input(X)

        # Create a list of tensorflow placeholders, one item per atom in the system
        inputs = []

        for ii in range(0,len(self.labels),2):
            inputs.append(tf.placeholder(tf.float32, [None, self.labels[ii+1]]))

        # Generate the weights and the biases that match the network architecture
        self.weights = {}
        self.biases = {}

        for key, value in self.unique_ele.iteritems():
            w, b = self.__generate_weights(value)
            self.weights[key] = w
            self.biases[key] = b

        # Evaluate the model for each atom
        model_list = []
        for ii in range(len(inputs)):
            lab = self.labels[2*ii]
            model_atom = self.__generate_model(inputs[ii], lab)
            model_list.append(model_atom)

        # Summing the results to get the total energy
        model_tot = tf.add_n(model_list)

        # Calculating the cost function with L2 regularisation term
        cost = self.__reg_cost(model_tot, y_tf)

        # Training step
        optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate_init).minimize(cost)

        # Initialisation of the model
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            self.cost_list = []

            for iter in range(self.max_iter):

                # This is the total number of batches in which the training set is divided
                n_batches = int(self.n_samples / self.batch_size)
                # This will be used to calculate the average cost per iteration
                avg_cost = 0

                # Learning over the batches of data
                for i in range(n_batches):

                    # Create a dictionary that maps the python data to the tf placeholders
                    idx1 = i * self.batch_size
                    idx2 = (i + 1) * self.batch_size
                    feeddict = {i: d[idx1:idx2, :] for i, d in zip(inputs, X_input)}
                    feeddict[y_tf] = y[idx1:idx2]
                    opt, c = sess.run([optimiser, cost], feed_dict=feeddict)
                    avg_cost += c / n_batches

                self.cost_list.append(avg_cost)

    def plot_cost(self):
        """
        This function plots the cost as a function of training iterations. It can only be called after the model has
        been trained.

        :return: None
        """
        try:
            self.cost_list
        except AttributeError:
            raise AttributeError("No values for the cost. Make sure that the model has been trained with the function "
                            "fit().")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(self.cost_list, label="Train set", color="b")
        ax.set_xlabel('Number of iterations')
        ax.set_ylabel('Cost Value')
        ax.legend()
        plt.yscale("log")
        plt.show()

    def __unique_elements(self):
        """
        This function takes the 'labels' parameter and extracts the unique elements. These are placed into a dictionary
        where the value is the number of features that each unique element has in the descriptor.

        :return: dictionary of size (n_unique_elements)
        """
        feat_dict = {}

        for ii in range(0,len(self.labels), 2):
            feat_dict[self.labels[ii]] = self.labels[ii+1]

        return feat_dict

    def __generate_weights(self, n_input_layer):
        """
        This function generates the weights and the biases for each element-specific neural network. It does so by
        looking at the size of the hidden layers. The weights are initialised randomly.

        :n_input_layer: number of features in the descriptor for one atom - int
        :return: lists (of length n_hidden_layers + 1) of tensorflow variables
        """

        weights = []
        biases = []

        eps = 0.01
        # Weights from input layer to first hidden layer
        weights.append(tf.Variable(tf.random_normal([self.hidden_layer_sizes[0], n_input_layer]) * 2 * eps - eps))
        biases.append(tf.Variable(tf.zeros([self.hidden_layer_sizes[0]])))

        # Weights from one hidden layer to the next
        for ii in range(len(self.hidden_layer_sizes)-1):
            weights.append(tf.Variable(tf.random_normal([self.hidden_layer_sizes[ii+1], self.hidden_layer_sizes[ii]]) * 2 * eps - eps))
            biases.append(tf.Variable(tf.zeros([self.hidden_layer_sizes[ii+1]])))

        # Weights from lat hidden layer to output layer
        weights.append(tf.Variable(tf.random_normal([1, self.hidden_layer_sizes[len(self.hidden_layer_sizes)-1]]) * 2 * eps - eps))
        biases.append(tf.Variable(tf.zeros([1])))

        return weights, biases

    def __generate_model(self, X, label):
        """
        This function calculates the single atom energy. It requires the descriptor X for a particular atom and the
        label of that atom.

        :X: placeholder
        :label: string

        :return: tf model
        """
        z = tf.add(tf.matmul(X, tf.transpose(self.weights[label][0])), self.biases[label][0])
        h = tf.nn.sigmoid(z)

        for ii in range(len(self.hidden_layer_sizes)):
            z = tf.add(tf.matmul(h, tf.transpose(self.weights[label][ii+1])), self.biases[label][ii+1])
            h = tf.nn.sigmoid(z)

        return h

    def __split_input(self, X):
        """
        This function takes the data where the descriptor of all the atoms are concatenated into one line. It then splits
        it into n_atoms different data sets that will all be fed into a different mini-network.

        :X: numpy array of shape (n_samples, n_features_tot)
        :return: list of numpy array of shape (n_samples, n_features)
        """
        split_X = []

        counter = 0
        for ii in range(0, len(self.labels), 2):
            idx1 = counter
            idx2 = counter + self.labels[ii+1]
            split_X.append(X[:,idx1:idx2])
            counter = counter + self.labels[ii+1]

        return split_X

    def __reg_cost(self, nn_energy, qm_energy):
        """
        This function calculates the cost function with L2 regularisation. It requires the energies predicted by the
        neural network and the energies calculated through quantum mechanics.

        :nn_energy: tf.Variable of shape [n_samples, 1]
        :qm_energy: tf.placeholder of shape [n_samples, 1]
        :return: tf.Variable of shape [1]
        """

        err = tf.subtract(qm_energy, nn_energy)
        cost = tf.nn.l2_loss(err)   # scalar
        reg_l2 = tf.Variable(tf.zeros([1])) # scalar

        for key, value in self.unique_ele.iteritems():
            for ii in range(len(self.hidden_layer_sizes)+1):
                reg_l2 = tf.add(reg_l2, tf.nn.l2_loss(self.weights[key][ii]))

        reg_l2 = reg_l2 * self.alpha
        cost_reg = tf.add(cost, reg_l2)

        return cost_reg

if __name__ == "__main__":

    def testMatrix1():
        X = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.8 ],
             [0.1, 0.0, 0.0, 0.9, 0.0, 0.0, -0.5, -0.5, -0.5, 0.1, 0.5, 0.5, 0.6, 0.5, 0.5,],
                [-0.1, 0.0, 0.0, 1.1, 0.0, 0.0, 1.0, 1.0, 1.0, 0.3, 0.5, 0.3, 1.5, 2.5, 0.5,]])
        y = np.array([4.0, 3.0, 1.0])
        return X, y

    def testMatrix2():
        X = np.random.rand(200, 15)
        y = np.random.rand(200)

        return X, y

    X, y = testMatrix2()
    nn = BPNN(hidden_layer_sizes=(50, 10, 5), labels=['N', 3, 'C', 3, 'C', 3, 'H', 3, 'H', 3], max_iter=300, batch_size=20)
    nn.fit(X, y)
    nn.plot_cost()